"""问题 4：定向干扰源的搜索、定位与清除（接口自适应版）。

与问题 3 的代码组织保持一致：本模块**只依赖一个"类 RobotClient 接口"的对象**
（鸭子类型），因此同一份策略既能接真实 ``client.RobotClient``（由
``problem4_main.py`` 启动），也能接本地规则仿真器 ``DirectionalLocalSimulator``
离线批量验证。用到的接口成员：

    enter() / exit()
    measure(x, y, channel) -> .result("direction"|"near"|"no_signal") / .svd_deg
    clear(x, y, channel)   -> .cleared
    current_position / current_channel / virtual_time_s / remaining_real_s
    is_time_up(margin_s) / in_session

问题 4 与问题 3 的唯一物理差别：定向源只在**发射方向两侧各 90° 的半圆**内辐射，
所以 ``no_signal`` 既可能是"超出接收半径"，也可能是"落在定向覆盖范围之外"，
不能再当作几何距离约束使用。策略据此做三点改动：

1. 可行域只由 ``direction`` 观测裁剪（楔形 ∩ 接收盘），``no_signal`` 一律忽略；
2. 巡检点使用严格证明过的 **21 个确定性检测点**：圆心 1 个 + 半径 996 m 的
   内正八边形 8 个点 + 外接目标圆的正十二边形 12 个点（外接圆半径
   ``R_o = 1800 / cos15° = 1863.497 m``）。证明以 998 m 为认证半径，把区域分解为
   40 个三点安全单元，因此任意位置、任意发射方向的源都必有一点同时落在其
   1000 m 接收圆与 90° 定向半圆内。
3. 巡检途中把已经可靠定位的清除中心与尚未访问的安全检测点放入同一条
   滚动开放路径，避免走完整条巡检骨架后再折返清除。

本模块只提供策略、本地仿真器与**批量验证**入口；开发过程中被评估/否决方案的
实测依据汇总在 ``problem4/尝试记录.md``。

用法::

    python problem4/problem4_main.py                    # 接真实模拟器的唯一入口（演练/正式）
    python problem4/problem4_strategy.py --cases 200    # 本模块的本地批量验证
    python problem4/problem4_strategy.py --cases 10 --source-count 10
    python problem4/problem4_strategy.py --random-state 42   # 复现某次运行

不指定 ``--random-state`` 时每次运行都重新随机源的位置、接收半径、
全向/定向搭配与发射方向；``--source-count 10`` 可把每个案例固定为 10 个源。

自行记录：策略内部按动作落一份结构化日志（``ActionLogger``，表头与问题3 相同），
``problem4_main.py`` 把它写到 ``actions.tsv``，通信层原始请求/响应另见
``q4_logs/<时间戳>/http/robot_*.txt``；本地批量加 ``--log-dir`` 也会写首案例的动作日志。
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# 同时兼容 `python problem4/xxx.py`、`python -m` 与从项目根目录导入
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_P3 = os.path.join(_ROOT, "problem3")
for _path in (_ROOT, _HERE, _P3):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from problem3_geometry import (
    circumscribed_polygon,
    clip_wedge,
    distance_point_to_polygon,
    farthest_pair,
    intersect_disk,
    min_enclosing_circle,
    nearest_neighbor_route,
    perpendicular_left,
    two_opt_open,
    unit_from_deg,
)
from problem3_strategy import ActionLogger
from simulator import (
    BEARING_ERROR_DEG,
    CHANNEL_MAX,
    CHANNEL_MIN,
    CLEAR_FAIL_TIME_S,
    CLEAR_RADIUS,
    CLEAR_SUCCESS_TIME_S,
    LocalSimulator,
    MAX_RECEIVE_RADIUS,
    MEASURE_TIME_S,
    MIN_RECEIVE_RADIUS,
    MOVE_SPEED_MPS,
    MeasureResult,
    NEAR_RADIUS,
    SOURCE_COUNT_MAX,
    SOURCE_COUNT_MIN,
    SWITCH_TIME_S,
    Source,
    TARGET_RADIUS,
)

Point = tuple[float, float]

# --------------------------------------------------------------------------- #
# 21 点严格覆盖巡检方案与策略参数
# --------------------------------------------------------------------------- #
INNER_OCTAGON_RADIUS_M = 996.0
OUTER_DODECAGON_RADIUS_M = TARGET_RADIUS / math.cos(math.pi / 12.0)
COVERAGE_CERTIFICATION_RADIUS_M = 998.0
COVERAGE_SAFE_CELL_COUNT = 40

# 可行域半径不超过该值时，两次清除法（先圆心、再按测向偏移）保证 20 m 内命中
SAFE_LOCALIZATION_RADIUS_M = 58.0
# 清除网格兜底用的三角格点边长：覆盖半径恰为 CLEAR_RADIUS 的 0.999 倍
COVER_LATTICE_SIDE_M = CLEAR_RADIUS * math.sqrt(3.0) * 0.999

# 第一次测向后用于拉开交会角的固定候选补测点，见式 Q± = S + 300u ± 200v
PROBE_FORWARD_M = 300.0
PROBE_SIDE_M = 200.0


class TimeBudgetExceeded(RuntimeError):
    pass


def distance(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


# --------------------------------------------------------------------------- #
# 几何对象
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DirectionObservation:
    """一次 ``direction`` 观测：测点位置 + 示向度（度）。"""
    position: Point
    bearing_deg: float


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float

    @property
    def center(self) -> Point:
        return (self.x, self.y)


def feasible_polygon(observations: list[DirectionObservation]) -> np.ndarray:
    """由 ``direction`` 观测得到的保守可行域（凸多边形，必含真实位置）。

    起始区域是半径 1800 m 目标圆的外切正多边形；每次观测取
    ``楔形(测点, 示向度 ± 1°) ∩ 接收盘(测点, 1500 m)`` 的交。
    定向遮挡造成的 ``no_signal`` 不能转成距离约束，因此**不参与**裁剪，
    这样只会让可行域偏大，不会漏掉干扰源。
    """
    polygon = circumscribed_polygon((0.0, 0.0), TARGET_RADIUS)
    half_angle = math.radians(BEARING_ERROR_DEG + 1e-12)
    for observation in observations:
        apex = (float(observation.position[0]), float(observation.position[1]))
        polygon = clip_wedge(
            polygon, apex, math.radians(observation.bearing_deg), half_angle
        )
        polygon = intersect_disk(polygon, apex, MAX_RECEIVE_RADIUS)
        if len(polygon) == 0:
            # 理论上不会发生（真实源一定在楔形 ∩ 接收盘内），退化时保持保守
            return circumscribed_polygon((0.0, 0.0), TARGET_RADIUS)
    return polygon


def minimum_enclosing_circle(polygon: np.ndarray) -> Circle:
    """可行域的最小包围圆；退化时退化为整个目标圆（保守）。"""
    result = min_enclosing_circle(polygon)
    if result is None:
        return Circle(0.0, 0.0, TARGET_RADIUS)
    center, radius = result
    return Circle(float(center[0]), float(center[1]), float(radius))


def first_probe_points(observation: DirectionObservation) -> list[Point]:
    """由第一次测向生成两个固定候选补测点（侧向拉开，形成大交会角）。"""
    anchor = np.asarray(observation.position, dtype=float)
    forward = unit_from_deg(observation.bearing_deg)
    side = perpendicular_left(forward)
    candidates = [
        anchor + PROBE_FORWARD_M * forward + PROBE_SIDE_M * side,
        anchor + PROBE_FORWARD_M * forward - PROBE_SIDE_M * side,
    ]
    return [(float(point[0]), float(point[1])) for point in candidates]


# --------------------------------------------------------------------------- #
# 开放路径工具（节点用整型 id 标识，便于把"清除点"与"补测点"混排）
# --------------------------------------------------------------------------- #
def open_route_length(
    start: Point, route: list[int], nodes: dict[int, Point]
) -> float:
    total = 0.0
    current = (float(start[0]), float(start[1]))
    for node in route:
        point = nodes[node]
        total += distance(current, point)
        current = (float(point[0]), float(point[1]))
    return total


def _index_route_length(
    start: Point, route: list[int], points: list[Point]
) -> float:
    """按 ``points`` 下标计价的开放路径长度（内部复用，避免反复建字典）。"""
    total = 0.0
    current = (float(start[0]), float(start[1]))
    for index in route:
        point = points[index]
        total += distance(current, point)
        current = (float(point[0]), float(point[1]))
    return total


def or_opt_open_route(
    route: list[int], start: Point, points: list[Point], max_segment: int = 3
) -> list[int]:
    """Or-opt 局部搜索：把长度 ≤ ``max_segment`` 的片段（含反向）搬到更优位置。

    2-opt 只能反转片段，动不了"整段搬移"；巡检路线是"两圈 + 圆心 + 清除点"
    的结构，搬移比反转更有效。每轮只落一个改进再重扫，收敛很快。
    """
    best = list(route)
    if len(best) < 4:
        return best
    while True:
        base = _index_route_length(start, best, points)
        improved = False
        count = len(best)
        for segment_length in range(1, max_segment + 1):
            for begin in range(count - segment_length + 1):
                piece = best[begin : begin + segment_length]
                rest = best[:begin] + best[begin + segment_length :]
                for insert_at in range(len(rest) + 1):
                    if insert_at == begin:
                        continue
                    for candidate in (piece, piece[::-1]):
                        trial = rest[:insert_at] + candidate + rest[insert_at:]
                        value = _index_route_length(start, trial, points)
                        if value < base - 1e-9:
                            best = trial
                            base = value
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            return best


def optimized_open_route(start: Point, nodes: dict[int, Point]) -> list[int]:
    """最近邻 + 2-opt + Or-opt 的开放路径；节点较多时再加若干极角起点取最优。

    动态重排一次案例里会调用几十次，所以起点数按规模给：小规模子问题只做
    Or-opt，20 个以上节点固定 6 个极角起点。实测（10 例 × 10 源）总里程
    23108 m → 22639 m（仅 Or-opt）→ 22323 m（加 6 起点），总时间 -3.1%。
    """
    keys = sorted(nodes)
    if not keys:
        return []
    points = [nodes[key] for key in keys]
    best = or_opt_open_route(
        two_opt_open(nearest_neighbor_route(start, points), start, points),
        start,
        points,
    )
    extra_starts = min(6, len(keys) // 4) if len(keys) >= 8 else 0
    if extra_starts == 0:
        return [keys[index] for index in best]

    order = sorted(
        range(len(points)),
        key=lambda index: math.atan2(
            points[index][1] - float(start[1]),
            points[index][0] - float(start[0]),
        ),
    )
    candidates = [best]
    stride = max(1, len(order) // extra_starts)
    for position in range(0, len(order), stride):
        first = order[position]
        route = [first] + [
            index for index in range(len(points)) if index != first
        ]
        candidates.append(
            or_opt_open_route(two_opt_open(route, start, points), start, points)
        )
    best = min(
        candidates, key=lambda route: _index_route_length(start, route, points)
    )
    return [keys[index] for index in best]


def exact_open_route(
    start: Point, nodes: dict[int, Point], end: Point | None = None
) -> list[int]:
    """Held-Karp 精确路径；可选 ``end`` 时把末节点到终点也计入代价。"""
    keys = sorted(nodes)
    count = len(keys)
    if count == 0:
        return []
    points = [nodes[key] for key in keys]

    best: dict[tuple[int, int], tuple[float, int | None]] = {}
    for index in range(count):
        best[(1 << index, index)] = (distance(start, points[index]), None)

    for size in range(2, count + 1):
        for subset in itertools.combinations(range(count), size):
            mask = 0
            for item in subset:
                mask |= 1 << item
            for last in subset:
                previous_mask = mask ^ (1 << last)
                best_cost = math.inf
                best_previous = None
                for previous in subset:
                    if previous == last:
                        continue
                    cost = best[(previous_mask, previous)][0] + distance(
                        points[previous], points[last]
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_previous = previous
                best[(mask, last)] = (best_cost, best_previous)

    full_mask = (1 << count) - 1
    current = min(
        range(count),
        key=lambda index: (
            best[(full_mask, index)][0]
            + (0.0 if end is None else distance(points[index], end))
        ),
    )
    route: list[int] = []
    mask = full_mask
    while current is not None:
        route.append(keys[current])
        _, previous = best[(mask, current)]
        mask ^= 1 << current
        current = previous
    route.reverse()
    return route


def twenty_one_point_stations(
    inner_radius_m: float | None = None,
    outer_radius_m: float | None = None,
) -> list[Point]:
    """返回证明中的 21 点：圆心 + 内正八边形 + 外正十二边形。

    内圈为 ``I_k = 996(cos(k*pi/4), sin(k*pi/4))``；外圈为
    ``E_j = R(cos(j*pi/6), sin(j*pi/6))``，其中
    ``R = 1800/cos(15°)``。正式参数下，证明用半径 998 m 的 40 个三点
    安全单元覆盖外十二边形，从而严格覆盖整个半径 1800 m 的目标圆。

    两个可选半径参数只为几何回归实验保留；正式策略始终使用上述默认值。
    """
    inner_radius = (
        INNER_OCTAGON_RADIUS_M if inner_radius_m is None else inner_radius_m
    )
    outer_radius = (
        OUTER_DODECAGON_RADIUS_M if outer_radius_m is None else outer_radius_m
    )
    inner = [
        (
            inner_radius * math.cos(index * math.pi / 4.0),
            inner_radius * math.sin(index * math.pi / 4.0),
        )
        for index in range(8)
    ]
    outer = [
        (
            outer_radius * math.cos(index * math.pi / 6.0),
            outer_radius * math.sin(index * math.pi / 6.0),
        )
        for index in range(12)
    ]

    # 证明中可直接应用“短边三角形整体安全”的两类代表边。
    directly_safe_lengths = (
        inner_radius,
        distance(inner[0], inner[1]),
        distance(outer[0], outer[1]),
        distance(inner[1], outer[1]),
        distance(inner[1], outer[2]),
    )
    if max(directly_safe_lengths) > COVERAGE_CERTIFICATION_RADIUS_M + 1e-8:
        raise ValueError("21 点结构的短边安全单元超过 998 m 认证半径")
    if not (
        inner_radius < COVERAGE_CERTIFICATION_RADIUS_M < MIN_RECEIVE_RADIUS
    ):
        raise ValueError("必须满足内八边形半径 < 998 m < 最小接收半径")
    if outer_radius * math.cos(math.pi / 12.0) < TARGET_RADIUS - 1e-8:
        raise ValueError("外圈十二边形未覆盖目标圆")
    stations = [(0.0, 0.0)] + inner + outer
    if len(stations) != 21:
        raise AssertionError("21 点巡检结构生成失败")
    return stations


# 保留旧函数名，避免已有评测/复现实验脚本的导入接口失效。
concentric_dodecagon_stations = twenty_one_point_stations


def coverage_certificate_triplets() -> list[tuple[int, int, int]]:
    """返回证明附录中的 40 个三点安全单元（按 21 点列表下标编号）。

    下标 ``0`` 是圆心，``1..8`` 是 ``I_0..I_7``，``9..20`` 是
    ``E_0..E_11``。十个基本三元组分别旋转 0°、90°、180°、270°。
    这些三元组是离线核验证书；在线巡检只需要访问 21 个检测点。
    """

    def inner(index: int) -> int:
        return 1 + index % 8

    def outer(index: int) -> int:
        return 9 + index % 12

    base = (
        (("I", 1), ("E", 1), ("E", 2)),
        (("I", 0), ("E", 0), ("E", 11)),
        (("I", 0), ("E", 0), ("E", 1)),
        (("O", 0), ("I", 0), ("I", 1)),
        (("O", 0), ("I", 0), ("I", 7)),
        (("I", 0), ("I", 1), ("E", 1)),
        (("I", 0), ("I", 7), ("E", 11)),
        (("I", 1), ("I", 3), ("E", 3)),
        (("I", 1), ("E", 2), ("E", 3)),
        (("I", 1), ("E", 0), ("E", 1)),
    )

    result: list[tuple[int, int, int]] = []
    for quarter_turn in range(4):
        inner_shift = 2 * quarter_turn
        outer_shift = 3 * quarter_turn
        for triple in base:
            indices = []
            for ring, index in triple:
                if ring == "O":
                    indices.append(0)
                elif ring == "I":
                    indices.append(inner(index + inner_shift))
                else:
                    indices.append(outer(index + outer_shift))
            result.append(tuple(indices))
    if len(result) != COVERAGE_SAFE_CELL_COUNT or len(set(result)) != len(result):
        raise AssertionError("40 个安全单元生成失败")
    return result


def source_orientation_coverage_gap(
    stations: list[Point],
    radii: np.ndarray | None = None,
    angles: np.ndarray | None = None,
) -> float:
    """巡检点集的"(G, e) 认证空隙"最大值；≤ 180° ⇔ 任意位置 + 任意发射方向必被检出。

    对目标圆域内的每个位置 G，只有距 G 不超过最小接收半径（1000 m）的测点才可能收到
    信号，而每个这样的测点在 G 处张成 ±90° 的可视方向区间；整圈方向都被覆盖 ⇔ 这些
    方向的最大空隙不超过 180°。

    这正是"空频道证书"的判据：若某频道在巡检点集上都收不到信号，就必须保证不存在任何
    被漏掉的 (G, e)。迁移自 q3 的认证思想（那里用实际覆盖地图，这里换成正向遮挡判据）。
    """
    if radii is None:
        radii = np.linspace(0.0, TARGET_RADIUS, 37)
    if angles is None:
        angles = np.linspace(0.0, 2.0 * math.pi, 73, endpoint=False)
    worst = 0.0
    for radius in radii:
        for angle in angles:
            gx = float(radius * math.cos(angle))
            gy = float(radius * math.sin(angle))
            directions = sorted(
                math.degrees(math.atan2(point[1] - gy, point[0] - gx)) % 360.0
                for point in stations
                if math.hypot(point[0] - gx, point[1] - gy)
                <= MIN_RECEIVE_RADIUS + 1e-9
            )
            if not directions:
                return math.inf
            gap = max(
                (directions[(index + 1) % len(directions)] - directions[index])
                % 360.0
                for index in range(len(directions))
            )
            worst = max(worst, gap)
    return worst


def concentric_dodecagon_route(
    inner_radius_m: float | None = None,
    outer_radius_m: float | None = None,
) -> list[Point]:
    """21 点巡检路线的固定顺序版本（路线长度对照/回归验证用，生产路径走动态重排）。
    """
    stations = concentric_dodecagon_stations(inner_radius_m, outer_radius_m)
    origin = stations[0]
    nodes = {index: point for index, point in enumerate(stations[1:], start=1)}
    order = optimized_open_route(origin, nodes)
    return [origin] + [nodes[index] for index in order]


def clear_by_oriented_triangular_cover(
    robot,
    channel: int,
    polygon: np.ndarray,
    time_margin_s: float = 0.0,
) -> None:
    """用"可行域最长轴对齐"的三角格点覆盖清除，作为确定性兜底。

    边长 ``sqrt(3) * r`` 的三角格点覆盖半径恰为 ``r``；这里取 ``r = 20 m`` 的
    0.999 倍，保证可行域内任意点到最近格点严格小于 20 m，命中是必然的。
    只保留距可行域不超过 20 m 的格点，并在四种蛇形走向中挑最短的一条。
    """
    if len(polygon) == 0:
        raise ValueError("不能对空可行域做网格清除")

    if len(polygon) == 1:
        only = (float(polygon[0][0]), float(polygon[0][1]))
        if not robot.clear(only[0], only[1], channel).cleared:
            raise RuntimeError("单点可行域清除失败")
        return

    farthest_pair = max(
        itertools.combinations(polygon, 2),
        key=lambda pair: (
            (pair[1][0] - pair[0][0]) ** 2 + (pair[1][1] - pair[0][1]) ** 2
        ),
    )
    axis_dx = float(farthest_pair[1][0] - farthest_pair[0][0])
    axis_dy = float(farthest_pair[1][1] - farthest_pair[0][1])
    axis_length = math.hypot(axis_dx, axis_dy)
    if axis_length <= 1e-12:
        axis = (1.0, 0.0)
    else:
        axis = (axis_dx / axis_length, axis_dy / axis_length)
    normal = (-axis[1], axis[0])

    def to_local(point) -> Point:
        return (
            float(point[0]) * axis[0] + float(point[1]) * axis[1],
            float(point[0]) * normal[0] + float(point[1]) * normal[1],
        )

    def to_global(point) -> Point:
        return (
            point[0] * axis[0] + point[1] * normal[0],
            point[0] * axis[1] + point[1] * normal[1],
        )

    local_polygon = [to_local(point) for point in polygon]
    minimum_x = min(point[0] for point in local_polygon)
    maximum_x = max(point[0] for point in local_polygon)
    minimum_y = min(point[1] for point in local_polygon)
    maximum_y = max(point[1] for point in local_polygon)

    row_height = COVER_LATTICE_SIDE_M * math.sqrt(3.0) / 2.0
    minimum_row = math.floor((minimum_y - CLEAR_RADIUS) / row_height) - 1
    maximum_row = math.ceil((maximum_y + CLEAR_RADIUS) / row_height) + 1
    rows: list[list[Point]] = []
    for row_index in range(minimum_row, maximum_row + 1):
        y = row_index * row_height
        offset_x = (row_index & 1) * COVER_LATTICE_SIDE_M / 2.0
        minimum_column = (
            math.floor((minimum_x - CLEAR_RADIUS - offset_x) / COVER_LATTICE_SIDE_M)
            - 1
        )
        maximum_column = (
            math.ceil((maximum_x + CLEAR_RADIUS - offset_x) / COVER_LATTICE_SIDE_M)
            + 1
        )
        row: list[Point] = []
        for column in range(minimum_column, maximum_column + 1):
            local_point = (column * COVER_LATTICE_SIDE_M + offset_x, y)
            global_point = to_global(local_point)
            if (
                distance_point_to_polygon(global_point, polygon)
                <= CLEAR_RADIUS + 1e-8
            ):
                row.append(global_point)
        if row:
            row.sort(key=to_local)
            rows.append(row)

    route_options: list[list[Point]] = []
    for reverse_rows in (False, True):
        ordered_rows = list(reversed(rows)) if reverse_rows else rows
        for reverse_first_row in (False, True):
            candidate_route: list[Point] = []
            for row_offset, row in enumerate(ordered_rows):
                reverse_row = (row_offset % 2 == 1) ^ reverse_first_row
                candidate_route.extend(reversed(row) if reverse_row else row)
            route_options.append(candidate_route)

    def path_length(points: list[Point]) -> float:
        current = robot.current_position
        total = 0.0
        for point in points:
            total += distance(current, point)
            current = point
        return total

    candidates = min(route_options, key=path_length)
    for point in candidates:
        if robot.is_time_up(time_margin_s):
            raise TimeBudgetExceeded("网格兜底清除尚未完成，时间已不足")
        if robot.clear(point[0], point[1], channel).cleared:
            return
    raise RuntimeError("网格兜底遍历完毕仍未清除目标")


# --------------------------------------------------------------------------- #
# 干扰源与本地仿真器（接口与 client.RobotClient 完全一致）
# --------------------------------------------------------------------------- #
@dataclass
class MixedSource(Source):
    """在问题 3 的 ``Source`` 上增加"是否定向 + 发射方向"。"""
    directional: bool = False
    emission_angle_rad: float | None = None


def generate_mixed_sources(
    seed: int,
    directional_probability: float = 0.5,
    force_mixed: bool = True,
    source_count: int | None = None,
) -> list[MixedSource]:
    """按题面假设生成一批源，其中一部分为定向源。

    ``source_count`` 为 ``None`` 时在题面的 10~16 之间随机；给定具体数值时
    强制用该数量（例如固定 10 个源的随机测试场景）。位置、接收半径、
    全向/定向标志与发射方向始终随机生成。
    """
    if not 0.0 <= directional_probability <= 1.0:
        raise ValueError("directional_probability 必须落在 [0, 1]")
    if source_count is not None and not (
        SOURCE_COUNT_MIN <= source_count <= CHANNEL_MAX
    ):
        raise ValueError(
            f"source_count 必须在 {SOURCE_COUNT_MIN}..{CHANNEL_MAX} 之间"
        )
    rng = np.random.default_rng(seed)
    if source_count is None:
        source_count = int(rng.integers(SOURCE_COUNT_MIN, SOURCE_COUNT_MAX + 1))
    channels = rng.choice(
        np.arange(CHANNEL_MIN, CHANNEL_MAX + 1), size=source_count, replace=False
    )
    radial = TARGET_RADIUS * np.sqrt(rng.random(source_count))
    angles = 2.0 * math.pi * rng.random(source_count)
    receive_radii = rng.uniform(
        MIN_RECEIVE_RADIUS, MAX_RECEIVE_RADIUS, size=source_count
    )
    directional = rng.random(source_count) < directional_probability
    if force_mixed and source_count >= 2:
        if not np.any(directional):
            directional[int(rng.integers(0, source_count))] = True
        if np.all(directional):
            directional[int(rng.integers(0, source_count))] = False
    emission_angles = 2.0 * math.pi * rng.random(source_count)

    return [
        MixedSource(
            channel=int(channel),
            position=np.array(
                [float(radius * math.cos(angle)), float(radius * math.sin(angle))]
            ),
            receive_radius=float(receive_radius),
            directional=bool(is_directional),
            emission_angle_rad=float(emission_angle) if is_directional else None,
        )
        for channel, radius, angle, receive_radius, is_directional, emission_angle in zip(
            channels,
            radial,
            angles,
            receive_radii,
            directional,
            emission_angles,
            strict=True,
        )
    ]


def directional_source_blocks(source, x: float, y: float) -> bool:
    """判断测点是否落在定向源的 90° 半圆之外（即被定向遮挡）。"""
    if not isinstance(source, MixedSource) or not source.directional:
        return False
    if source.emission_angle_rad is None:
        raise RuntimeError("定向干扰源缺少发射方向")
    dx = float(x) - float(source.position[0])
    dy = float(y) - float(source.position[1])
    return (
        math.cos(source.emission_angle_rad) * dx
        + math.sin(source.emission_angle_rad) * dy
        < -1e-9
    )


class DirectionalLocalSimulator(LocalSimulator):
    """本地规则仿真器：与 ``client.RobotClient`` 接口一致，额外支持定向遮挡。"""
    def __init__(self, sources: list[MixedSource], error_seed: int = 0, **kwargs):
        super().__init__(error_seed, num_sources=len(sources), **kwargs)
        self.sources: dict[int, MixedSource] = {
            source.channel: source for source in sources
        }
        self.directional_block_count = 0
        self.range_no_signal_count = 0
        self.absent_no_signal_count = 0

    def measure(self, x: float, y: float, channel: int) -> MeasureResult:
        if not self._in_session:
            raise RuntimeError("尚未 enter()")
        self.request_counter += 1
        move = self.estimate_move_time(x, y)
        switch = 0.0
        if self._current_channel is not None and int(channel) != self._current_channel:
            switch = SWITCH_TIME_S
        self._virtual_time += move + switch + MEASURE_TIME_S
        self._position = np.array([float(x), float(y)])
        self._current_channel = int(channel)

        source = self.sources.get(int(channel))
        result: str | None = None
        svd: float | None = None
        if source is None or source.cleared:
            self.absent_no_signal_count += 1
        else:
            source_distance = distance(
                (float(x), float(y)),
                (float(source.position[0]), float(source.position[1])),
            )
            if source_distance > source.receive_radius + 1e-9:
                self.range_no_signal_count += 1
            elif directional_source_blocks(source, x, y):
                self.directional_block_count += 1
            elif source_distance <= NEAR_RADIUS + 1e-9:
                result = "near"
            else:
                true_bearing = math.degrees(
                    math.atan2(source.position[1] - y, source.position[0] - x)
                ) % 360.0
                svd = (
                    true_bearing + self._bearing_error(int(channel), x, y)
                ) % 360.0
                result = "direction"

        return MeasureResult(
            result=result if result is not None else "no_signal",
            svd_deg=svd,
            position=(float(x), float(y)),
            channel=int(channel),
            virtual_time_s=self._virtual_time,
        )


# --------------------------------------------------------------------------- #
# 策略
# --------------------------------------------------------------------------- #
class Problem4Strategy:
    """定向源场景下的巡检 → 定位 → 清除策略，只依赖类 RobotClient 接口。"""
    PROBE_NODE_OFFSET = 100
    SURVEY_NODE_OFFSET = 1000
    MAX_PROBE_COMBINATIONS = 64
    MAX_EXACT_ROUTE_NODES = 12
    MIN_SURVEY_CROSSING_ANGLE_DEG = 0.0
    # 迁移自 q3_empty_channel_strategy 的三个判据（都只作用于"可选动作"，不影响完备性）：
    # 1) 边际收益：顺路复测只在"预期少跑的距离/5 - 本次花费 > 余量"时才做；
    # 2) completes 例外：若这一次测向就能把可行域压到可直接清除，无条件下做；
    # 3) 自适应放宽：已确认源数达到阈值后 latch 一次，把余量放低（越到后面专程跑越贵）。
    MARGINAL_REMEASURE_MARGIN_S = 2.0
    MARGINAL_REMEASURE_MARGIN_RELAXED_S = 0.0
    RESUME_REMEASURE_AT_KNOWN_SOURCES = 8
    ROUTE_END_AT_ORIGIN = False

    def __init__(
        self,
        robot,
        survey_detected_channels: bool = True,
        route_end_at_origin: bool | None = None,
        time_margin_s: float = 30.0,
        log_path: str | None = None,
        verbose: bool = False,
    ):
        self.robot = robot
        self.survey_detected_channels = survey_detected_channels
        self.route_end_at_origin = (
            self.ROUTE_END_AT_ORIGIN
            if route_end_at_origin is None
            else bool(route_end_at_origin)
        )
        self.time_margin_s = time_margin_s

        # ---- 自行记录：结构化动作日志（与问题3 同表头 / 同字段语义）----
        # 不给 log_path、也不开 verbose 时 _logging 为假，_log() 直接返回，
        # 不产生任何几何计算开销（本地批量验证仍按原速跑）。
        self.logger = ActionLogger(log_path, verbose)
        self._logging = bool(log_path) or bool(verbose)
        self.seq = 0
        self._phase = "INIT"

        self.channels = list(range(CHANNEL_MIN, CHANNEL_MAX + 1))
        self.observations: dict[int, list[DirectionObservation]] = {
            channel: [] for channel in self.channels
        }
        self.detected: set[int] = set()
        self.cleared: set[int] = set()
        self.directional_block_points: dict[int, list[Point]] = {
            channel: [] for channel in self.channels
        }
        # 已被"可证伪"证据确认的定向频道（见 _certify_directional）
        self.certified_directional: set[int] = set()
        self.attempted_probes: dict[int, set[int]] = {
            channel: set() for channel in self.channels
        }

        self.stats: dict[str, float] = {
            "measures": 0.0,
            "clears": 0.0,
            "clear_success": 0.0,
            "move_distance": 0.0,
            "switches": 0.0,
            "probes": 0.0,
            "survey_remeasures": 0.0,
            "survey_inserted_clears": 0.0,
            "survey_localized_skips": 0.0,
            "finish_shared_remeasures": 0.0,
            "directional_constraints": 0.0,
            "directional_probe_reorders": 0.0,
            "survey_remeasure_value_skips": 0.0,
            "survey_remeasure_completes": 0.0,
        }
        # 自适应放宽的 latch（q3 adaptive_resume 的同一机制）
        self._remeasure_margin_relaxed = False
        self.survey_measurement_count = 0
        self.survey_visited_station_count = 0
        self.grid_fallback_count = 0
        self.probe_no_signal_count = 0
        self.skipped_impossible_remeasure_count = 0
        self.max_final_radius_m = 0.0
        self.survey_end_time_s = 0.0
        self.incomplete_reason = ""
        self._session_owned = False

    # -------------------------------------------------------------- 运行入口
    def run(self) -> dict[str, object]:
        self._enter()
        try:
            self._phase = "SURVEY"
            self.run_survey()
            self._phase = "FINISH"
            self.finish()
        except TimeBudgetExceeded as error:
            self.incomplete_reason = str(error)
            self._phase = "ABORT"
            self._log(
                "abort", *self._current_point(), CHANNEL_MIN,
                result="time_budget_exceeded", note=str(error),
            )
        finally:
            self._phase = "EXIT"
            self._exit()
        return self.summary()

    def _enter(self) -> None:
        if not getattr(self.robot, "in_session", False):
            self.robot.enter()
            self._session_owned = True
        self._phase = "INIT"
        self._log(
            "enter", *self._current_point(), CHANNEL_MIN,
            result="entered",
            note=(
                "remaining_real_s="
                f"{float(getattr(self.robot, 'remaining_real_s', float('nan'))):.1f}"
            ),
        )

    def _exit(self) -> None:
        try:
            if getattr(self.robot, "in_session", False):
                self.robot.exit()
                self._log(
                    "exit", *self._current_point(), CHANNEL_MIN,
                    result="exit",
                    note=(
                        f"detected={len(self.detected)} cleared={len(self.cleared)} "
                        f"virtual_time_s={float(self.robot.virtual_time_s):.1f}"
                    ),
                )
        except Exception as error:  # noqa: BLE001 - 收尾失败不应掩盖主流程结果
            self._log(
                "exit_failed", *self._current_point(), CHANNEL_MIN,
                result="error", note=repr(error),
            )

    def summary(self) -> dict[str, object]:
        pending = sorted(self.detected - self.cleared)
        directional_channels = sorted(self.certified_directional)
        return {
            "survey_station_count": self.survey_visited_station_count,
            "survey_planned_station_count": len(concentric_dodecagon_stations()),
            "coverage_certificate_radius_m": COVERAGE_CERTIFICATION_RADIUS_M,
            "coverage_safe_cell_count": len(coverage_certificate_triplets()),
            "survey_measurement_count": self.survey_measurement_count,
            "detected": len(self.detected),
            "cleared": len(self.cleared),
            # 定向/全向的统计口径见 _certify_directional：
            # 定向只能被"证伪"（接收半径内却收不到信号），全向无法被证实。
            "confirmed_directional_count": len(directional_channels),
            "confirmed_directional_channels": directional_channels,
            "not_confirmed_directional_count": len(
                self.detected - self.certified_directional
            ),
            "not_confirmed_directional_channels": sorted(
                self.detected - self.certified_directional
            ),
            "complete": not pending and not self.incomplete_reason,
            "incomplete_reason": self.incomplete_reason,
            "pending_channels": pending,
            "route_end_at_origin": self.route_end_at_origin,
            "survey_time_s": self.survey_end_time_s,
            "virtual_time_s": float(self.robot.virtual_time_s),
            "grid_fallback_count": self.grid_fallback_count,
            "probe_no_signal_count": self.probe_no_signal_count,
            "skipped_impossible_remeasure_count": (
                self.skipped_impossible_remeasure_count
            ),
            "max_final_radius_m": self.max_final_radius_m,
            "stats": dict(self.stats),
        }

    def _time_left(self) -> bool:
        return not self.robot.is_time_up(self.time_margin_s)

    # ------------------------------------------------------------ 底层动作封装
    def _measure(self, x: float, y: float, channel: int, is_probe: bool = False):
        before = self.robot.current_position
        moved = distance(before, (float(x), float(y)))
        self.stats["move_distance"] += moved
        current = self.robot.current_channel
        switched = current is not None and current != channel
        if switched:
            self.stats["switches"] += 1
        result = self.robot.measure(x, y, channel)
        self.stats["measures"] += 1
        if is_probe:
            self.stats["probes"] += 1
        self._log(
            "measure_probe" if is_probe else "measure", float(x), float(y), channel,
            result=result.result, svd_deg=result.svd_deg,
            note=self._action_note(moved, switched),
        )
        return result

    def _clear(self, x: float, y: float, channel: int):
        before = self.robot.current_position
        moved = distance(before, (float(x), float(y)))
        self.stats["move_distance"] += moved
        result = self.robot.clear(x, y, channel)
        self.stats["clears"] += 1
        if result.cleared:
            self.stats["clear_success"] += 1
        self._log(
            "clear", float(x), float(y), channel,
            result="success" if result.cleared else "no_target_in_range",
            note=self._action_note(moved, False),
        )
        return result

    # ---------------------------------------------------------------- 日志
    def _current_point(self) -> Point:
        try:
            x, y = self.robot.current_position
            return (float(x), float(y))
        except Exception:  # noqa: BLE001 - 记录失败不应影响主流程
            return (0.0, 0.0)

    @staticmethod
    def _action_note(moved_m: float, switched: bool = False) -> str:
        """把一次动作的移动距离 / 是否切频写进日志备注，便于赛后核对耗时。"""
        return f"move={moved_m:.1f}m" + (" switch=1" if switched else "")

    def _channel_status(self, channel: int) -> str:
        if channel in self.cleared:
            return "CLEARED"
        if channel in self.certified_directional:
            return "DIRECTIONAL"
        if channel in self.detected:
            return "DETECTED"
        return "UNKNOWN"

    def _feasible_radius(self, channel: int) -> float | None:
        """当前可行域（测向楔形交集）的最小包围圆半径；无观测时为 None。"""
        observations = self.observations.get(channel)
        if not observations:
            return None
        polygon = feasible_polygon(observations)
        if len(polygon) == 0:
            return None
        return float(minimum_enclosing_circle(polygon).radius)

    def _log(
        self,
        action: str,
        x: float,
        y: float,
        channel: int,
        result: str = "-",
        svd_deg: float | None = None,
        note: str = "",
    ) -> None:
        """写一行动作日志；未开启日志时零开销直接返回。"""
        if not self._logging:
            return
        self.seq += 1
        self.logger.log(
            seq=self.seq,
            virtual_time=float(self.robot.virtual_time_s),
            phase=self._phase,
            action=action,
            x=float(x),
            y=float(y),
            channel=int(channel),
            result=result,
            svd_deg=svd_deg,
            channel_status=self._channel_status(channel),
            feasible_radius=self._feasible_radius(channel),
            note=note,
        )

    # ------------------------------------------------------------ 可行域维护
    def _polygon(self, channel: int) -> np.ndarray:
        """测向楔形交集；``no_signal`` 刻意排除在几何裁剪之外。

        ``no_signal @ P`` 只意味着「``|P-G| > r_c`` **或** ``(P-G)·e < 0``」（超距
        或被遮挡），对单个测点、类型未知的源不构成任何约束——当成距离约束用会漏检。
        唯一可靠的转化是「发射方向弧」约束：对候选位置 G，量程内（``|P-G| ≤ 1000``，
        因 ``r_c ≥ 1000`` 必在量程内）的 no_signal 点必须都在背后、测到方向的点必须
        都在前方，二者要能用一条过 G 的直线分开；无解则 G 不可能（等价于「全部测向
        点落在某个 180° 弧内、量程内 no_signal 点全在弧外」）。

        对未检出频道，它退化为「量程内测点必须塞进一个开半平面」，即判空证书。
        21 点方案的严格完备性来自 998 m 认证半径下的 40 个三点安全单元覆盖；因此某频道
        只有在 21 个证明测点全部无信号后才能安全判空，不能提前按无信号次数停止。

        实测（种子 3000-3011）：
        * 对已定位频道几乎没有压缩力：80 次清除/补测决策里只有 3 次能缩小可行域
          （中位包围半径 24 → 24 m），因为那时可行域只剩 ~29 m 半径，域内各候选点
          的观测几何已经一致，要么全可行要么全不可行。
        * ``r_c`` 的上界 1500 不能当约束：``r_c`` 可能只有 1000，no_signal 永远
          证明不了「超距」。
        """
        return feasible_polygon(self.observations[channel])

    def _circle(self, channel: int) -> Circle:
        return minimum_enclosing_circle(self._polygon(channel))

    def _record_measurement(
        self,
        channel: int,
        point: Point,
        result: str,
        bearing_deg: float | None,
    ) -> None:
        if result == "direction" and bearing_deg is not None:
            self.detected.add(channel)
            self.observations[channel].append(
                DirectionObservation(
                    (float(point[0]), float(point[1])), float(bearing_deg)
                )
            )
        elif result == "near":
            self.detected.add(channel)
            if not self._clear(point[0], point[1], channel).cleared:
                raise RuntimeError("near 结果未能清除目标")
            self.cleared.add(channel)
        elif result == "no_signal":
            self._certify_directional(channel, point)
        else:
            raise RuntimeError(f"未知的检测结果 {result!r}")

    def _certify_directional(self, channel: int, point: Point) -> None:
        """把"接收半径内却收不到信号"的 ``no_signal`` 记为定向证据。

        所有干扰源的有效接收半径都 ≥ 1000 m，且可行域是真实位置的保守包含集。
        因此若可行域整体落在测点 1000 m 以内，真实源必然也在接收半径内，
        此时 ``no_signal`` 只可能来自定向遮挡，可以**确定**该源是定向源。
        反过来收不到这种证据并不能证明是全向源，所以只做"定向已确认"统计，
        不改变任何策略决策。
        """
        if not self.observations[channel]:
            return
        polygon = self._polygon(channel)
        if len(polygon) == 0:
            return
        farthest = max(
            math.hypot(
                float(vertex[0]) - point[0], float(vertex[1]) - point[1]
            )
            for vertex in polygon
        )
        if farthest <= MIN_RECEIVE_RADIUS + 1e-9:
            self.certified_directional.add(channel)
            blocked_points = self.directional_block_points[channel]
            candidate = (float(point[0]), float(point[1]))
            if all(distance(candidate, old) > 1e-6 for old in blocked_points):
                blocked_points.append(candidate)
                self.stats["directional_constraints"] += 1
                self._log(
                    "certify_directional", point[0], point[1], channel,
                    result="directional",
                    note=(
                        f"polygon_farthest={farthest:.1f}m "
                        f"blocked_points={len(blocked_points)}"
                    ),
                )

    def _unresolved(self) -> set[int]:
        return {
            channel
            for channel in self.channels
            if channel not in self.detected
        }

    # ---------------------------------------------------------------- 阶段A
    def run_survey(self) -> float:
        """访问全部 21 个证明测点；清除插入后动态重排剩余路线。

        尚未检出的频道必须在**每个**测点都测一次，不能按"空检测计数"降频抽查。
        原因是判空与必检出使用同一个 ``(位置, 发射朝向)`` 覆盖条件；模型只证明完整
        21 点集合充分覆盖，并未证明任意前缀或任意删点后的集合仍然覆盖。唯一可安全提前
        结束的情形是已检出题面给出的源数上界 ``SOURCE_COUNT_MAX`` 个。
        """
        stations = concentric_dodecagon_stations()
        remaining = {
            index: point for index, point in enumerate(stations[1:], start=1)
        }
        station = stations[0]
        visit_index = 0

        while True:
            self.survey_visited_station_count += 1
            # 已清除频道**永不**被复测：``_unresolved()`` 只含"从未检出"的频道，
            # 而下面这条只从 ``detected - cleared`` 里补，再经
            # ``_needs_free_survey_measurement``（对 cleared 直接返回 False）。
            channels = self._unresolved()
            if self.survey_detected_channels:
                for channel in self.detected - self.cleared:
                    if self._needs_free_survey_measurement(channel, station):
                        channels.add(channel)
            ordered = sorted(channels, reverse=visit_index % 2 == 1)
            current = self.robot.current_channel
            if current in ordered:
                ordered.remove(current)
                ordered.insert(0, current)
            for channel in ordered:
                if not self._time_left():
                    self.incomplete_reason = "现实时间不足，未完成全部安全巡检点"
                    return self._end_survey()
                result = self._measure(station[0], station[1], channel)
                self.survey_measurement_count += 1
                if channel in self.detected:
                    self.stats["survey_remeasures"] += 1
                self._record_measurement(
                    channel, station, result.result, result.svd_deg
                )
                # 题面给出 16 个源是硬上限：全部找到后后续测点不可能再有新源
                if len(self.detected) >= SOURCE_COUNT_MAX:
                    return self._end_survey()

            if not remaining:
                return self._end_survey()

            while True:
                clear_circles: dict[int, Circle] = {}
                for channel in self.detected - self.cleared:
                    circle = self._circle(channel)
                    if circle.radius <= SAFE_LOCALIZATION_RADIUS_M:
                        clear_circles[channel] = circle
                nodes = {
                    self.SURVEY_NODE_OFFSET + node: point
                    for node, point in remaining.items()
                }
                nodes.update(
                    {
                        channel: circle.center
                        for channel, circle in clear_circles.items()
                    }
                )
                if len(nodes) <= self.MAX_EXACT_ROUTE_NODES:
                    route = exact_open_route(self.robot.current_position, nodes)
                else:
                    route = optimized_open_route(
                        self.robot.current_position, nodes
                    )
                next_node = route[0]
                if next_node < self.SURVEY_NODE_OFFSET:
                    self.stats["survey_inserted_clears"] += 1
                    self._clear_circle(next_node, clear_circles[next_node])
                    continue

                station_node = next_node - self.SURVEY_NODE_OFFSET
                station = remaining.pop(station_node)
                visit_index += 1
                break

    def _end_survey(self) -> float:
        self.survey_end_time_s = float(self.robot.virtual_time_s)
        self._log(
            "survey_end", *self._current_point(), CHANNEL_MIN,
            result="survey_finished",
            note=(
                f"visited={self.survey_visited_station_count}/"
                f"{len(concentric_dodecagon_stations())} "
                f"detected={len(self.detected)} cleared={len(self.cleared)}"
            ),
        )
        return self.survey_end_time_s

    def _needs_free_survey_measurement(self, channel: int, station: Point) -> bool:
        """判断在巡检测点上能否"免费"补一次已有频道的测向。

        注意**不要**把"外圈不再复测已检出频道"当成省时间的优化。纯方位定位的
        距离（沿方位）精度由基线长决定：``σ_r ≈ d²σ_β/L``，σ_β 是固定角误差
        （1°，不随距离退化），所以外圈给出的长基线（L≈1800 m，σ_r≈10 m）比内圈
        基线（L≈500 m，σ_r≈35 m）更能把可行域锁死；外圈点的横向误差大，但它补的
        是内圈完全缺失的"距离"维度。实测（16 例 × 10 源）把外圈复测全禁掉：
        补测 0.4 → 3.6 次/例、里程 21951 → 25719 m（+17%）、时间 6480 → 7287 s
        （+12.5%）。所以这里只按边际价值判断，不按圈层一刀切。
        """
        if channel in self.cleared or not self.observations[channel]:
            return False
        polygon = self._polygon(channel)
        circle = minimum_enclosing_circle(polygon)
        if circle.radius <= SAFE_LOCALIZATION_RADIUS_M:
            self.stats["survey_localized_skips"] += 1
            return False
        if distance_point_to_polygon(station, polygon) > MAX_RECEIVE_RADIUS + 1e-9:
            self.skipped_impossible_remeasure_count += 1
            return False
        if (
            self.MIN_SURVEY_CROSSING_ANGLE_DEG > 0.0
            and self._maximum_crossing_angle(channel, station, circle)
            < self.MIN_SURVEY_CROSSING_ANGLE_DEG
        ):
            return False
        return self._survey_remeasure_is_worthwhile(channel, station, circle)

    # ---------------------------------------------------------- 顺路复测的价值
    def _hypothetical_circle_after_measure(
        self, channel: int, station: Point, circle: Circle
    ) -> Circle:
        """假设该频道在 ``station`` 测到指向当前可行域中心的示向度后的最小包围圆。

        这是 q3 里 ``hypothetical[channel] = remaining.difference(disk)`` 的第四问版本：
        定向遮挡下 ``no_signal`` 不提供几何约束，所以只能对"测到方向"这一情形做假设，
        用可行域大小的变化量当作顺路复测的收益来源。
        """
        center = circle.center
        bearing_deg = math.degrees(
            math.atan2(center[1] - station[1], center[0] - station[0])
        )
        hypothetical = list(self.observations[channel])
        hypothetical.append(
            DirectionObservation(
                (float(station[0]), float(station[1])), bearing_deg
            )
        )
        return minimum_enclosing_circle(feasible_polygon(hypothetical))

    def _survey_remeasure_margin_s(self) -> float:
        """顺路复测要求的最小净收益（秒）；已知源数达标后 latch 放宽一次。"""
        if self._remeasure_margin_relaxed:
            return self.MARGINAL_REMEASURE_MARGIN_RELAXED_S
        if len(self.detected) >= self.RESUME_REMEASURE_AT_KNOWN_SOURCES:
            self._remeasure_margin_relaxed = True
            return self.MARGINAL_REMEASURE_MARGIN_RELAXED_S
        return self.MARGINAL_REMEASURE_MARGIN_S

    def _survey_remeasure_is_worthwhile(
        self, channel: int, station: Point, circle: Circle
    ) -> bool:
        """q3 ``_marginal_scan`` 的迁移：只在重算后的计划确实更便宜时才顺路补测。

        * ``completes`` 例外：这一次测向就能把可行域压到可直接清除（半径 ≤ 58 m），
          等价于 q3 的 ``always_finish_channel``，无条件做；
        * 否则按"预期少跑的专程距离 ÷ 5 m/s − 本次检测与切换耗时 > 余量"判定，
          余量在已确认源数达标后由 latch 放宽（越到后面专程跑一趟越贵）。
        """
        immediate_s = MEASURE_TIME_S
        if self.robot.current_channel not in (None, channel):
            immediate_s += SWITCH_TIME_S
        hypothesized = self._hypothetical_circle_after_measure(
            channel, station, circle
        )
        if hypothesized.radius <= SAFE_LOCALIZATION_RADIUS_M:
            self.stats["survey_remeasure_completes"] += 1
            return True
        saved_s = (circle.radius - hypothesized.radius) / MOVE_SPEED_MPS
        if saved_s - immediate_s > self._survey_remeasure_margin_s():
            return True
        self.stats["survey_remeasure_value_skips"] += 1
        return False

    def _maximum_crossing_angle(
        self, channel: int, point: Point, circle: Circle
    ) -> float:
        center = circle.center
        candidate_vector = (
            point[0] - center[0],
            point[1] - center[1],
        )
        candidate_norm = math.hypot(candidate_vector[0], candidate_vector[1])
        if candidate_norm <= 1e-9:
            return 90.0
        maximum_crossing_angle = 0.0
        for observation in self.observations[channel]:
            existing_vector = (
                observation.position[0] - center[0],
                observation.position[1] - center[1],
            )
            existing_norm = math.hypot(existing_vector[0], existing_vector[1])
            if existing_norm <= 1e-9:
                return 90.0
            cosine = abs(
                (
                    candidate_vector[0] * existing_vector[0]
                    + candidate_vector[1] * existing_vector[1]
                )
                / (candidate_norm * existing_norm)
            )
            crossing_angle = math.degrees(
                math.acos(min(1.0, max(0.0, cosine)))
            )
            maximum_crossing_angle = max(maximum_crossing_angle, crossing_angle)
        return maximum_crossing_angle

    def _finish_shared_measure_at(
        self, point: Point, exclude_channel: int | None = None
    ) -> None:
        """收尾阶段动态融合：在已到达点顺带补测其它未收敛频道。"""
        candidates: list[tuple[float, int, Circle]] = []
        for channel in self.detected - self.cleared:
            if channel == exclude_channel or not self.observations[channel]:
                continue
            polygon = self._polygon(channel)
            circle = minimum_enclosing_circle(polygon)
            if circle.radius <= SAFE_LOCALIZATION_RADIUS_M:
                continue
            if distance_point_to_polygon(point, polygon) > MAX_RECEIVE_RADIUS + 1e-9:
                continue
            angle = self._maximum_crossing_angle(channel, point, circle)
            if angle < self.MIN_SURVEY_CROSSING_ANGLE_DEG:
                continue
            # 与巡检途中同一套价值判据（q3 marginal_scan）：收益不够就不顺路测
            if not self._survey_remeasure_is_worthwhile(channel, point, circle):
                continue
            candidates.append((-angle, channel, circle))

        candidates.sort()
        for _, channel, _ in candidates[:2]:
            if not self._time_left() or channel in self.cleared:
                break
            result = self._measure(point[0], point[1], channel)
            self.stats["finish_shared_remeasures"] += 1
            if result.result == "no_signal":
                self.probe_no_signal_count += 1
            self._record_measurement(channel, point, result.result, result.svd_deg)

    def _route_score(self, route: list[int], nodes: dict[int, Point]) -> float:
        score = open_route_length(self.robot.current_position, route, nodes)
        if self.route_end_at_origin and route:
            score += distance(nodes[route[-1]], (0.0, 0.0))
        return score

    # ---------------------------------------------------------------- 阶段B
    def finish(self) -> None:
        """未清除频道统一调度：补测缩小可行域、两次清除、网格兜底。"""
        while True:
            if not self.detected - self.cleared:
                return
            if not self._time_left():
                raise TimeBudgetExceeded("仍有未清除的干扰源，但时间已不足")
            pending = sorted(self.detected - self.cleared)
            clear_circles: dict[int, Circle] = {}
            probe_groups: list[tuple[int, list[tuple[int, Point]]]] = []
            fallback_channels: list[int] = []

            for channel in pending:
                circle = self._circle(channel)
                if circle.radius <= SAFE_LOCALIZATION_RADIUS_M:
                    clear_circles[channel] = circle
                    continue
                probes = self._probe_options(channel)
                if probes:
                    probe_groups.append((channel, probes))
                else:
                    fallback_channels.append(channel)

            if fallback_channels:
                chosen = min(
                    fallback_channels,
                    key=lambda item: distance(
                        self.robot.current_position, self._circle(item).center
                    ),
                )
                self._grid_clear(chosen)
                continue

            option_lists = [options for _, options in probe_groups]
            combination_count = math.prod(len(options) for options in option_lists)
            if combination_count <= self.MAX_PROBE_COMBINATIONS:
                combinations = (
                    itertools.product(*option_lists) if option_lists else [()]
                )
            else:
                combinations = [
                    tuple(
                        min(
                            options,
                            key=lambda item: distance(
                                self.robot.current_position, item[1]
                            ),
                        )
                        for options in option_lists
                    )
                ]

            best_route: list[int] | None = None
            best_probe_actions: dict[int, tuple[int, int, Point]] = {}
            best_length = math.inf
            for selected in combinations:
                nodes: dict[int, Point] = {
                    channel: circle.center
                    for channel, circle in clear_circles.items()
                }
                probe_actions: dict[int, tuple[int, int, Point]] = {}
                for (channel, _), (probe_index, point) in zip(
                    probe_groups, selected, strict=True
                ):
                    node = self.PROBE_NODE_OFFSET + 2 * channel + probe_index
                    nodes[node] = point
                    probe_actions[node] = (channel, probe_index, point)
                if len(nodes) <= self.MAX_EXACT_ROUTE_NODES:
                    route = exact_open_route(
                        self.robot.current_position,
                        nodes,
                        end=(0.0, 0.0) if self.route_end_at_origin else None,
                    )
                else:
                    route = optimized_open_route(self.robot.current_position, nodes)
                length = self._route_score(route, nodes)
                if length + 1e-9 < best_length:
                    best_length = length
                    best_route = route
                    best_probe_actions = probe_actions

            if not best_route:
                raise RuntimeError("问题 4 滚动路线没有可执行动作")
            next_node = best_route[0]
            if next_node in best_probe_actions:
                channel, probe_index, point = best_probe_actions[next_node]
                self._measure_probe(channel, probe_index, point)
                self._finish_shared_measure_at(self.robot.current_position, channel)
            else:
                self._clear_circle(next_node, clear_circles[next_node])
                self._finish_shared_measure_at(self.robot.current_position, next_node)

    def _block_separation(
        self, channel: int, center: Point, point: Point
    ) -> float:
        """候选测点方向离"已被遮挡方向"的总角距（越大越可能看得见源）。"""
        candidate_angle = math.atan2(
            point[1] - center[1], point[0] - center[0]
        )
        score = 0.0
        for blocked in self.directional_block_points[channel]:
            blocked_angle = math.atan2(
                blocked[1] - center[1], blocked[0] - center[0]
            )
            separation = abs(
                (candidate_angle - blocked_angle + math.pi) % (2.0 * math.pi)
                - math.pi
            )
            score += separation
        return score

    def _probe_options(self, channel: int) -> list[tuple[int, Point]]:
        probes = first_probe_points(self.observations[channel][0])
        options = [
            (index, point)
            for index, point in enumerate(probes)
            if index not in self.attempted_probes[channel]
        ]
        if len(options) <= 1 or not self.directional_block_points[channel]:
            return options

        center = self._circle(channel).center
        reordered = sorted(
            options,
            key=lambda item: (
                -self._block_separation(channel, center, item[1]),
                distance(self.robot.current_position, item[1]),
                item[0],
            ),
        )
        if [item[0] for item in reordered] != [item[0] for item in options]:
            self.stats["directional_probe_reorders"] += 1
        return reordered

    def _measure_probe(self, channel: int, probe_index: int, point: Point) -> None:
        self.attempted_probes[channel].add(probe_index)
        result = self._measure(point[0], point[1], channel, is_probe=True)
        if result.result == "no_signal":
            self.probe_no_signal_count += 1
        self._record_measurement(channel, point, result.result, result.svd_deg)

    def _grid_clear(self, channel: int) -> None:
        """网格兜底清除：可行域三角格点遍历，最后的安全网。"""
        self.grid_fallback_count += 1
        self._log(
            "grid_clear_begin", *self._current_point(), channel,
            result="start",
            note=f"fallback_no={self.grid_fallback_count}",
        )
        clear_by_oriented_triangular_cover(
            self.robot,
            channel,
            self._polygon(channel),
            time_margin_s=self.time_margin_s,
        )
        self.cleared.add(channel)
        self._log(
            "grid_clear_done", *self._current_point(), channel,
            result="cleared",
            note="格点内部逐点 clear 见通信日志",
        )

    def _clear_circle(self, channel: int, circle: Circle) -> None:
        """可行域半径 ≤ 58 m 时的两次清除法，失败则转网格兜底。"""
        self.max_final_radius_m = max(self.max_final_radius_m, circle.radius)
        if self._clear(circle.x, circle.y, channel).cleared:
            self.cleared.add(channel)
            return

        result = self._measure(circle.x, circle.y, channel)
        if result.result == "direction" and result.svd_deg is not None:
            bearing_rad = math.radians(float(result.svd_deg))
            self.observations[channel].append(
                DirectionObservation(
                    (circle.x, circle.y), float(result.svd_deg)
                )
            )
            travel = (CLEAR_RADIUS + circle.radius) / (
                2.0 * math.cos(math.radians(BEARING_ERROR_DEG))
            )
            repair_point = (
                circle.x + travel * math.cos(bearing_rad),
                circle.y + travel * math.sin(bearing_rad),
            )
            if self._clear(repair_point[0], repair_point[1], channel).cleared:
                self.cleared.add(channel)
                return
        elif result.result == "near":
            if self._clear(circle.x, circle.y, channel).cleared:
                self.cleared.add(channel)
                return
        self._grid_clear(channel)


# --------------------------------------------------------------------------- #
# 单案例 / 批量运行
# --------------------------------------------------------------------------- #
@dataclass
class Problem4CaseResult:
    case_id: int
    case_name: str
    seed: int
    directional_probability: float
    source_count: int
    directional_count: int
    omnidirectional_count: int
    confirmed_directional_count: int
    cleared_count: int
    success: bool
    total_time_s: float
    seconds_per_source: float
    movement_time_s: float
    measurement_time_s: float
    switching_time_s: float
    clearing_time_s: float
    survey_time_s: float
    localization_time_s: float
    measurement_count: float
    clear_attempt_count: float
    failed_clear_count: float
    survey_station_count: int
    grid_fallback_count: int
    probe_no_signal_count: int
    directional_block_count: int
    range_no_signal_count: int
    skipped_impossible_remeasure_count: int
    survey_remeasure_count: float
    survey_inserted_clear_count: float
    survey_localized_skip_count: float
    finish_shared_remeasure_count: float
    directional_constraint_count: float
    directional_probe_reorder_count: float


def run_problem4_case(
    case_id: int,
    seed: int,
    directional_probability: float = 0.5,
    force_mixed: bool = True,
    survey_detected_channels: bool = True,
    route_end_at_origin: bool | None = None,
    source_count: int | None = None,
    log_path: str | None = None,
    verbose: bool = False,
) -> Problem4CaseResult:
    """随机场景：源数默认在 10~16 随机，位置/接收半径/定向标志全部随机生成。

    ``source_count=10`` 可固定为"恰好 10 个源"的随机场景（源数固定，
    但全向/定向搭配、位置、接收半径、发射方向每次运行都重新随机）。
    ``log_path`` 给定时同时写一份结构化动作日志（自行记录）。
    """
    sources = generate_mixed_sources(
        seed,
        directional_probability=directional_probability,
        force_mixed=force_mixed,
        source_count=source_count,
    )
    return _evaluate_case(
        case_id=case_id,
        case_name=(
            f"random {len(sources)} sources seed={seed}"
            if source_count is not None
            else f"random seed={seed}"
        ),
        seed=seed,
        sources=sources,
        directional_probability=directional_probability,
        survey_detected_channels=survey_detected_channels,
        route_end_at_origin=route_end_at_origin,
        log_path=log_path,
        verbose=verbose,
    )


def _evaluate_case(
    case_id: int,
    case_name: str,
    seed: int,
    sources: list[MixedSource],
    directional_probability: float,
    survey_detected_channels: bool = True,
    route_end_at_origin: bool | None = None,
    log_path: str | None = None,
    verbose: bool = False,
) -> Problem4CaseResult:
    """在本地仿真器上跑一个给定场景，并做真值校验（漏检、定向判定假阳性）。"""
    simulator = DirectionalLocalSimulator(sources, error_seed=seed + 10_000)
    strategy = Problem4Strategy(
        simulator,
        survey_detected_channels=survey_detected_channels,
        route_end_at_origin=route_end_at_origin,
        log_path=log_path,
        verbose=verbose,
    )
    summary = strategy.run()

    success = all(source.cleared for source in sources)
    if not success:
        missing = [source.channel for source in sources if not source.cleared]
        raise RuntimeError(f"问题 4 策略漏检频道 {missing}")

    # 本地仿真的真值：定向/全向各几个
    directional_channels = {
        source.channel for source in sources if source.directional
    }
    # 策略侧"已确认定向"只能来自可靠证据，必须落在真值之内
    confirmed = set(summary["confirmed_directional_channels"])
    if not confirmed <= directional_channels:
        raise RuntimeError(
            f"定向判定出现假阳性：{sorted(confirmed - directional_channels)}"
        )

    stats = summary["stats"]
    clear_success = float(stats["clear_success"])
    clear_attempt = float(stats["clears"])
    total_time = float(simulator.virtual_time_s)
    survey_time = float(summary["survey_time_s"])
    return Problem4CaseResult(
        case_id=case_id,
        case_name=case_name,
        seed=seed,
        directional_probability=directional_probability,
        source_count=len(sources),
        directional_count=len(directional_channels),
        omnidirectional_count=len(sources) - len(directional_channels),
        confirmed_directional_count=len(confirmed),
        cleared_count=sum(source.cleared for source in sources),
        success=success,
        total_time_s=total_time,
        seconds_per_source=total_time / len(sources),
        movement_time_s=float(stats["move_distance"]) / MOVE_SPEED_MPS,
        measurement_time_s=float(stats["measures"]) * MEASURE_TIME_S,
        switching_time_s=float(stats["switches"]) * SWITCH_TIME_S,
        clearing_time_s=(
            clear_success * CLEAR_SUCCESS_TIME_S
            + (clear_attempt - clear_success) * CLEAR_FAIL_TIME_S
        ),
        survey_time_s=survey_time,
        localization_time_s=total_time - survey_time,
        measurement_count=float(stats["measures"]),
        clear_attempt_count=clear_attempt,
        failed_clear_count=clear_attempt - clear_success,
        survey_station_count=int(summary["survey_station_count"]),
        grid_fallback_count=int(summary["grid_fallback_count"]),
        probe_no_signal_count=int(summary["probe_no_signal_count"]),
        directional_block_count=int(simulator.directional_block_count),
        range_no_signal_count=int(simulator.range_no_signal_count),
        skipped_impossible_remeasure_count=int(
            summary["skipped_impossible_remeasure_count"]
        ),
        survey_remeasure_count=float(stats["survey_remeasures"]),
        survey_inserted_clear_count=float(stats["survey_inserted_clears"]),
        survey_localized_skip_count=float(stats["survey_localized_skips"]),
        finish_shared_remeasure_count=float(stats["finish_shared_remeasures"]),
        directional_constraint_count=float(stats["directional_constraints"]),
        directional_probe_reorder_count=float(stats["directional_probe_reorders"]),
    )


def summarize_problem4(
    results: list[Problem4CaseResult],
    random_state: int,
    directional_probability: float,
    survey_detected_channels: bool,
    route_end_at_origin: bool = Problem4Strategy.ROUTE_END_AT_ORIGIN,
) -> dict[str, object]:
    totals = np.array([result.total_time_s for result in results], dtype=float)
    source_counts = np.array(
        [result.source_count for result in results], dtype=float
    )
    directional_counts = np.array(
        [result.directional_count for result in results], dtype=float
    )
    total_directional = int(np.sum(directional_counts))
    total_confirmed = int(
        sum(result.confirmed_directional_count for result in results)
    )
    # 单源定位清除时间 = 该案例总虚拟时间 / 该案例干扰源数
    per_source = totals / source_counts
    return {
        "strategy": "problem4_21_point_octagon_dodecagon",
        "random_state": random_state,
        "case_count": len(results),
        "case_names": [result.case_name for result in results],
        "directional_probability": directional_probability,
        "survey_detected_channels": survey_detected_channels,
        "route_end_at_origin": route_end_at_origin,
        "survey_station_count": results[0].survey_station_count,
        "inner_octagon_radius_m": INNER_OCTAGON_RADIUS_M,
        "outer_dodecagon_radius_m": OUTER_DODECAGON_RADIUS_M,
        "coverage_certificate_radius_m": COVERAGE_CERTIFICATION_RADIUS_M,
        "coverage_safe_cell_count": COVERAGE_SAFE_CELL_COUNT,
        "success_count": sum(result.success for result in results),
        "success_rate": float(
            np.mean([result.success for result in results])
        ),
        "mean_source_count": float(np.mean(source_counts)),
        "total_source_count": int(np.sum(source_counts)),
        "mean_directional_count": float(np.mean(directional_counts)),
        "mean_omnidirectional_count": float(
            np.mean([result.omnidirectional_count for result in results])
        ),
        "total_directional_count": total_directional,
        "total_omnidirectional_count": int(
            sum(result.omnidirectional_count for result in results)
        ),
        "mean_confirmed_directional_count": float(
            np.mean([result.confirmed_directional_count for result in results])
        ),
        "directional_confirmation_rate": (
            float(total_confirmed / total_directional)
            if total_directional > 0
            else 0.0
        ),
        "average_seconds_per_source": float(
            np.sum(totals) / np.sum(source_counts)
        ),
        "average_clear_time_s": float(np.mean(per_source)),
        "median_clear_time_s": float(np.median(per_source)),
        "max_clear_time_s": float(np.max(per_source)),
        "mean_total_time_s": float(np.mean(totals)),
        "median_total_time_s": float(np.median(totals)),
        "max_total_time_s": float(np.max(totals)),
        "mean_movement_time_s": float(
            np.mean([result.movement_time_s for result in results])
        ),
        "mean_measurement_time_s": float(
            np.mean([result.measurement_time_s for result in results])
        ),
        "mean_switching_time_s": float(
            np.mean([result.switching_time_s for result in results])
        ),
        "mean_clearing_time_s": float(
            np.mean([result.clearing_time_s for result in results])
        ),
        "mean_survey_time_s": float(
            np.mean([result.survey_time_s for result in results])
        ),
        "mean_localization_time_s": float(
            np.mean([result.localization_time_s for result in results])
        ),
        "mean_measurement_count": float(
            np.mean([result.measurement_count for result in results])
        ),
        "mean_clear_attempt_count": float(
            np.mean([result.clear_attempt_count for result in results])
        ),
        "mean_failed_clear_count": float(
            np.mean([result.failed_clear_count for result in results])
        ),
        "mean_grid_fallback_count": float(
            np.mean([result.grid_fallback_count for result in results])
        ),
        "mean_probe_no_signal_count": float(
            np.mean([result.probe_no_signal_count for result in results])
        ),
        "mean_directional_block_count": float(
            np.mean([result.directional_block_count for result in results])
        ),
        "mean_range_no_signal_count": float(
            np.mean([result.range_no_signal_count for result in results])
        ),
        "mean_skipped_impossible_remeasure_count": float(
            np.mean(
                [
                    result.skipped_impossible_remeasure_count
                    for result in results
                ]
            )
        ),
        "mean_survey_remeasure_count": float(
            np.mean([result.survey_remeasure_count for result in results])
        ),
        "mean_survey_inserted_clear_count": float(
            np.mean([result.survey_inserted_clear_count for result in results])
        ),
        "mean_survey_localized_skip_count": float(
            np.mean([result.survey_localized_skip_count for result in results])
        ),
        "mean_finish_shared_remeasure_count": float(
            np.mean([result.finish_shared_remeasure_count for result in results])
        ),
        "mean_directional_constraint_count": float(
            np.mean([result.directional_constraint_count for result in results])
        ),
        "mean_directional_probe_reorder_count": float(
            np.mean([result.directional_probe_reorder_count for result in results])
        ),
        "assumptions": {
            "source_position": "半径为 1800 m 的圆域内按面积均匀、相互独立",
            "receive_radius_m": "在 [1000, 1500] 上独立均匀",
            "directional_flag": "Bernoulli 分布，并可强制混合场景",
            "emission_direction": "在 [0, 2*pi) 上独立均匀",
        },
    }


def write_results(
    results: list[Problem4CaseResult],
    summary: dict[str, object],
    output_prefix: Path | str,
) -> None:
    """写 ``<prefix>.csv``（逐案，表头/取值均为英文）与 ``<prefix>.json``（汇总）。"""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with output_prefix.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def resolve_random_state(random_state: int | None) -> int:
    """未显式指定种子时，每次运行重新取一个随机种子（并写进结果便于复现）。"""
    if random_state is not None:
        return int(random_state)
    return int(np.random.default_rng().integers(0, 2**31 - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题4 定向源本地仿真批量验证")
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="随机种子；不指定则每次运行都重新随机（结果里会记录本次种子）",
    )
    parser.add_argument("--directional-probability", type=float, default=0.5)
    parser.add_argument(
        "--source-count",
        type=int,
        default=None,
        help="固定每个案例的干扰源数（例如 10）；默认在题面的 10~16 之间随机",
    )
    parser.add_argument("--allow-pure", action="store_true")
    parser.add_argument(
        "--survey-unresolved-only",
        action="store_true",
        help="关闭巡检途中对已发现频道的顺路补测",
    )
    parser.add_argument(
        "--no-route-end-at-origin",
        action="store_true",
        help="收尾 TSP 不把最后节点到原点的距离计入代价",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="给定时把首个案例的结构化动作日志写到该目录（自行记录用）",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(_HERE) / "outputs/tables/problem4_strategy_local",
        help="本地评测逐案结果落盘前缀（写 <prefix>.csv 与 <prefix>.json）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    survey_detected_channels = not args.survey_unresolved_only
    random_state = resolve_random_state(args.random_state)

    source_hint = (
        f"每个案例固定 {args.source_count} 个源"
        if args.source_count is not None
        else "源数在 10~16 随机"
    )
    print(
        f"随机状态 {random_state}（{args.cases} 个案例，{source_hint}，"
        f"全向/定向按 p={args.directional_probability} 随机；"
        f"复现本次运行请加 --random-state {random_state}）"
    )

    results = [
        run_problem4_case(
            case_id=index + 1,
            seed=random_state + index,
            directional_probability=args.directional_probability,
            force_mixed=not args.allow_pure,
            survey_detected_channels=survey_detected_channels,
            route_end_at_origin=not args.no_route_end_at_origin,
            source_count=args.source_count,
            log_path=(
                os.path.join(
                    args.log_dir, f"problem4_local_seed{random_state + index}.tsv"
                )
                if args.log_dir and index == 0
                else None
            ),
        )
        for index in range(args.cases)
    ]
    summary = summarize_problem4(
        results,
        random_state=random_state,
        directional_probability=args.directional_probability,
        survey_detected_channels=survey_detected_channels,
        route_end_at_origin=not args.no_route_end_at_origin,
    )

    # 逐案例明细（每个案例的单源平均定位清除时间等）
    for result in results:
        print(
            f"[{result.case_id:>2}] {result.case_name}"
            f"｜源 {result.source_count}"
            f"（定向 {result.directional_count} / "
            f"全向 {result.omnidirectional_count}）"
            f"｜清除 {result.cleared_count}/{result.source_count}"
            f"｜单源平均 {result.seconds_per_source:8.2f} s"
            f"｜总 {result.total_time_s / 60.0:7.2f} min"
            f"｜巡检 {result.survey_time_s / 60.0:7.2f} min"
            f"｜已确认定向 {result.confirmed_directional_count}"
            f"｜兜底 {result.grid_fallback_count}"
        )
    print(
        f"合计：{len(results)} 例，成功 {sum(r.success for r in results)}/"
        f"{len(results)}；单源平均 "
        f"{float(summary['average_clear_time_s']):.2f} s"
        f"（中位 {float(summary['median_clear_time_s']):.2f} s，"
        f"最差 {float(summary['max_clear_time_s']):.2f} s）"
    )
    write_results(results, summary, args.output_prefix)
    print(f"逐案结果(CSV)：{args.output_prefix.with_suffix('.csv')}")
    print(f"汇总(JSON)：{args.output_prefix.with_suffix('.json')}")
    # 如确需完整汇总，可取消下一行注释（会输出大段 JSON）：
    # print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
