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
2. 巡检点使用 **25 个确定性检测点**：圆心 1 个 + 内正十二边形 12 个（半径 930 m）
   + 外正十二边形 12 个（半径 ``R_o = 1800 / cos15° = 1863.497 m``），两圈错开 15°。
   外圈半径使半径 1800 m 的目标圆恰好内切于外圈，内圈再铺满中心区域，
   因此任意位置、任意发射方向的源都必有一点同时落在其 1000 m 接收圆与 90° 定向半圆内。
3. 巡检途中把已经可靠定位的清除中心与尚未访问的安全检测点放入同一条
   滚动开放路径，避免走完整条巡检骨架后再折返清除。

本模块只提供策略、本地仿真器与**批量验证**入口；单元测试与随机场景验证
（每个案例恰好 10 个源）在 ``problem4/test_problem4.py``。

用法::

    python problem4/problem4_directional_simulation.py                 # 本地 30 个案例，种子随机
    python problem4/problem4_directional_simulation.py --cases 200
    python problem4/problem4_directional_simulation.py --cases 10 --source-count 10
    python problem4/problem4_directional_simulation.py --random-state 42   # 复现某次运行

不指定 ``--random-state`` 时每次运行都重新随机源的位置、接收半径、
全向/定向搭配与发射方向；``--source-count 10`` 可把每个案例固定为 10 个源。
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
from statistics import NormalDist

import numpy as np

# 同时兼容 `python problem4/xxx.py`、`python -m` 与从项目根目录导入
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_P3 = os.path.join(_ROOT, "problem3")
for _path in (_ROOT, _HERE, _P3):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from problem3_geometry import (  # noqa: E402
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
from problem3_local_sim import LocalSimulator, MeasureResult, Source  # noqa: E402
from problem3_strategy import (  # noqa: E402
    BEARING_ERROR_DEG,
    CHANNEL_MAX,
    CHANNEL_MIN,
    CLEAR_FAIL_TIME_S,
    CLEAR_RADIUS,
    CLEAR_SUCCESS_TIME_S,
    MAX_RECEIVE_RADIUS,
    MEASURE_TIME_S,
    MIN_RECEIVE_RADIUS,
    MOVE_SPEED_MPS,
    NEAR_RADIUS,
    SOURCE_COUNT_MAX,
    SOURCE_COUNT_MIN,
    SWITCH_TIME_S,
    TARGET_RADIUS,
)

Point = tuple[float, float]

# --------------------------------------------------------------------------- #
# 25 点巡检方案与策略参数
# --------------------------------------------------------------------------- #
INNER_DODECAGON_RADIUS_M = 930.0
OUTER_DODECAGON_RADIUS_M = TARGET_RADIUS / math.cos(math.pi / 12.0)  # 1863.497 m
DODECAGON_INTERLEAVE_ANGLE_DEG = 15.0  # 内外两圈错开角

# 可行域半径不超过该值时，两次清除法（先圆心、再按测向偏移）保证 20 m 内命中
SAFE_LOCALIZATION_RADIUS_M = 58.0
# 清除网格兜底用的三角格点边长：覆盖半径恰为 CLEAR_RADIUS 的 0.999 倍
COVER_LATTICE_SIDE_M = CLEAR_RADIUS * math.sqrt(3.0) * 0.999
# 备选的等边三角格点巡检方案（边长 999 m < 最小接收半径 1000 m）
TRIANGULAR_LATTICE_SIDE_M = 999.0
TRIANGULAR_LATTICE_OFFSET = (1.0 / 12.0, 1.0 / 12.0)

# 第一次测向后用于拉开交会角的固定候选补测点，见式 Q± = S + 750u ± 600v
PROBE_FORWARD_M = 750.0
PROBE_SIDE_M = 600.0

# β-Cautious（Vander Hook / Tokekar / Isler）迁移：
# Lemma 1 给出"补测必须站多远"，使补测落在源背后的（模糊）概率不超过 β：
#     r(i) = σx / sqrt(σβ² − σs²),  σβ = (π/2)·Φ⁻¹(1 − β/2)
# β 是**感知模糊风险**的容忍度（不是定位精度），β→0 时站距发散；
# 方向取"垂直于最大不确定方向"（可行域最长轴的法线），站距随不确定度自适应。
CAUTIOUS_PROBE_BETA = 0.05
CAUTIOUS_SENSOR_SIGMA_RAD = math.radians(BEARING_ERROR_DEG)
CAUTIOUS_MIN_PROBE_M = 25.0
CAUTIOUS_MAX_PROBE_M = 900.0


class TimeBudgetExceeded(RuntimeError):
    """现实/虚拟时间不足以完成剩余动作。"""


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


def optimized_open_route(start: Point, nodes: dict[int, Point]) -> list[int]:
    """最近邻 + 2-opt 的开放路径（节点数较多时的默认选择）。"""
    keys = sorted(nodes)
    if not keys:
        return []
    points = [nodes[key] for key in keys]
    route = nearest_neighbor_route(start, points)
    route = two_opt_open(route, start, points)
    return [keys[index] for index in route]


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


def concentric_dodecagon_stations(
    inner_radius_m: float | None = None,
    outer_radius_m: float | None = None,
) -> list[Point]:
    """圆心 1 个 + 内外两圈各 12 点的正十二边形，共 25 个检测点。

    内圈顶点取 ``15° + k·30°``，外圈取 ``k·30°``，两圈错开 15°；
    外圈半径取 ``1800 / cos15°``，使半径 1800 m 的目标圆恰好内切于外圈，
    从而外圈顶点对目标圆外的环带也保证 1000 m 覆盖。
    """
    inner_radius = (
        INNER_DODECAGON_RADIUS_M if inner_radius_m is None else inner_radius_m
    )
    outer_radius = (
        OUTER_DODECAGON_RADIUS_M if outer_radius_m is None else outer_radius_m
    )
    angle_step = 2.0 * math.pi / 12.0
    half_step = angle_step / 2.0
    inner = [
        (
            inner_radius * math.cos(half_step + index * angle_step),
            inner_radius * math.sin(half_step + index * angle_step),
        )
        for index in range(12)
    ]
    outer = [
        (
            outer_radius * math.cos(index * angle_step),
            outer_radius * math.sin(index * angle_step),
        )
        for index in range(12)
    ]

    required_lengths = (
        inner_radius,
        distance(inner[0], inner[1]),
        distance(outer[0], outer[1]),
        distance(inner[0], outer[0]),
    )
    if max(required_lengths) > MIN_RECEIVE_RADIUS + 1e-8:
        raise ValueError("十二边形三角剖分存在超过 1000 m 的边")
    if outer_radius * math.cos(half_step) < TARGET_RADIUS - 1e-8:
        raise ValueError("外圈十二边形未覆盖目标圆")
    return [(0.0, 0.0)] + inner + outer


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
    """25 点巡检路线：固定 25 点不变，用开放路径排序减少途中折返。"""
    stations = concentric_dodecagon_stations(inner_radius_m, outer_radius_m)
    origin = stations[0]
    nodes = {index: point for index, point in enumerate(stations[1:], start=1)}
    order = optimized_open_route(origin, nodes)
    return [origin] + [nodes[index] for index in order]


def triangular_lattice_stations(
    side_m: float = TRIANGULAR_LATTICE_SIDE_M,
    offset: tuple[float, float] = TRIANGULAR_LATTICE_OFFSET,
) -> list[Point]:
    """备选巡检方案：平移的等边三角格点（边长 < 1000 m）。

    只保留与半径 1800 m 目标圆相交的三角形所用到的顶点，比"大包围圆内所有格点"少很多。
    """
    if not 0.0 < side_m < MIN_RECEIVE_RADIUS:
        raise ValueError("三角格点边长必须落在 (0, 1000)")
    if len(offset) != 2 or not all(math.isfinite(value) for value in offset):
        raise ValueError("格点偏移必须包含两个有限系数")

    height = side_m * math.sqrt(3.0) / 2.0
    offset_x = offset[0] * side_m + offset[1] * side_m / 2.0
    offset_y = offset[1] * height

    def lattice_point(index: tuple[int, int]) -> Point:
        column, row = index
        return (
            offset_x + column * side_m + row * side_m / 2.0,
            offset_y + row * height,
        )

    index_bound = math.ceil((TARGET_RADIUS + 2.0 * side_m) / height) + 2
    used_indices: set[tuple[int, int]] = set()
    for column in range(-index_bound, index_bound + 1):
        for row in range(-index_bound, index_bound + 1):
            faces = (
                ((column, row), (column + 1, row), (column, row + 1)),
                (
                    (column + 1, row + 1),
                    (column + 1, row),
                    (column, row + 1),
                ),
            )
            for face in faces:
                triangle = np.asarray([lattice_point(index) for index in face])
                if (
                    distance_point_to_polygon((0.0, 0.0), triangle)
                    <= TARGET_RADIUS + 1e-9
                ):
                    used_indices.update(face)

    return sorted(
        (lattice_point(index) for index in used_indices),
        key=lambda point: (point[1], point[0]),
    )


def triangular_lattice_route(
    side_m: float = TRIANGULAR_LATTICE_SIDE_M,
    scan_origin: bool = False,
) -> list[Point]:
    points = triangular_lattice_stations(side_m)
    origin = (0.0, 0.0)
    nodes = {index: point for index, point in enumerate(points)}
    route = optimized_open_route(origin, nodes)
    ordered = [nodes[index] for index in route]
    return ([origin] if scan_origin else []) + ordered


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
    MAX_SURVEY_EDGE_DETOUR_M = 375.0
    LOW_DENSITY_SURVEY_EDGE_DETOUR_M = 500.0
    LOW_DENSITY_DETECTED_LIMIT = 10
    MIN_SURVEY_CROSSING_ANGLE_DEG = 25.0
    # 迁移自 q3_empty_channel_strategy 的三个判据（都只作用于"可选动作"，不影响完备性）：
    # 1) 边际收益：顺路复测只在"预期少跑的距离/5 - 本次花费 > 余量"时才做；
    # 2) completes 例外：若这一次测向就能把可行域压到可直接清除，无条件下做；
    # 3) 自适应放宽：已确认源数达到阈值后 latch 一次，把余量放低（越到后面专程跑越贵）。
    MARGINAL_REMEASURE_MARGIN_S = 2.0
    MARGINAL_REMEASURE_MARGIN_RELAXED_S = 0.0
    RESUME_REMEASURE_AT_KNOWN_SOURCES = 8
    # β-Cautious 自适应补测（论文 Algorithm 1 的一步贪心）：默认**关闭**的消融项。
    # 实测（10 例×10 源，种子 1000-1009 / 2000-2009）：10 例里只被评估 7 次、
    # 3 次真正用上，单源时间 643.39→643.54 / 661.05→662.01 s，收益 ≈ 0；
    # 原因是收尾阶段的补测几乎用不到——约 9.4 次/例的清除已在巡检途中完成。
    CAUTIOUS_PROBE_ENABLED = False
    CAUTIOUS_MAX_PROBE_STEPS = 3
    ADAPTIVE_PROBE_INDEX = 90
    # 「弱空频道剪枝」只作消融，默认关闭且**当前阈值不可达**：25 点方案的 24 个
    # 非圆心站方位恰好是 0,15,...,345°，角覆盖上限 = 345°（需访完全部 24 站），
    # 因此 350° 永不触发（实测 20 例标记 0 次、跳过 0 次，时间与基线一致）。
    # 一旦把阈值降到可达区间就会开始漏检（20 例×10 源实测）：
    #   330° → 单源 631.8 s（-3.5%），漏检 1/20；300° → 509.3 s（-22%），漏检 7/20；
    #   270° → 517.6 s（-21%），漏检 6/20。
    # 原因：判空的正确条件是"对每个候选源位置 G，1000 m 内的测点都能围住 G"，
    # 而本规则只检查测点相对**原点**的角覆盖，且把超距测点也当成证据。
    WEAK_UNKNOWN_MIN_NO_SIGNAL_POINTS = 15
    WEAK_UNKNOWN_MIN_ANGLE_COVERAGE_DEG = 350.0
    WEAK_UNKNOWN_FINAL_PROBES = 3
    ROUTE_END_AT_ORIGIN = False

    def __init__(
        self,
        robot,
        survey_detected_channels: bool = True,
        dynamic_survey_route: bool = True,
        weak_unknown_pruning_enabled: bool = False,
        weak_unknown_min_no_signal_points: int | None = None,
        weak_unknown_min_angle_coverage_deg: float | None = None,
        route_end_at_origin: bool | None = None,
        time_margin_s: float = 30.0,
    ):
        self.robot = robot
        self.survey_detected_channels = survey_detected_channels
        self.dynamic_survey_route = dynamic_survey_route
        self.weak_unknown_pruning_enabled = weak_unknown_pruning_enabled
        self.weak_unknown_min_no_signal_points = (
            self.WEAK_UNKNOWN_MIN_NO_SIGNAL_POINTS
            if weak_unknown_min_no_signal_points is None
            else int(weak_unknown_min_no_signal_points)
        )
        self.weak_unknown_min_angle_coverage_deg = (
            self.WEAK_UNKNOWN_MIN_ANGLE_COVERAGE_DEG
            if weak_unknown_min_angle_coverage_deg is None
            else float(weak_unknown_min_angle_coverage_deg)
        )
        self.route_end_at_origin = (
            self.ROUTE_END_AT_ORIGIN
            if route_end_at_origin is None
            else bool(route_end_at_origin)
        )
        self.time_margin_s = time_margin_s

        self.channels = list(range(CHANNEL_MIN, CHANNEL_MAX + 1))
        self.observations: dict[int, list[DirectionObservation]] = {
            channel: [] for channel in self.channels
        }
        self.detected: set[int] = set()
        self.cleared: set[int] = set()
        self.weak_unknown_channels: set[int] = set()
        self.unknown_no_signal_points: dict[int, list[Point]] = {
            channel: [] for channel in self.channels
        }
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
            "weak_unknown_marked": 0.0,
            "weak_unknown_survey_skips": 0.0,
            "weak_unknown_final_probes": 0.0,
            "weak_unknown_recovered": 0.0,
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
            self.run_survey()
            self.finish()
        except TimeBudgetExceeded as error:
            self.incomplete_reason = str(error)
        finally:
            self._exit()
        return self.summary()

    def _enter(self) -> None:
        if not getattr(self.robot, "in_session", False):
            self.robot.enter()
            self._session_owned = True

    def _exit(self) -> None:
        try:
            if getattr(self.robot, "in_session", False):
                self.robot.exit()
        except Exception:  # noqa: BLE001 - 收尾失败不应掩盖主流程结果
            pass

    def summary(self) -> dict[str, object]:
        pending = sorted(self.detected - self.cleared)
        directional_channels = sorted(self.certified_directional)
        return {
            "survey_station_count": self.survey_visited_station_count,
            "survey_planned_station_count": len(concentric_dodecagon_stations()),
            "dynamic_survey_route": self.dynamic_survey_route,
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
            "weak_unknown_channels": sorted(self.weak_unknown_channels),
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
        self.stats["move_distance"] += distance(before, (float(x), float(y)))
        current = self.robot.current_channel
        if current is not None and current != channel:
            self.stats["switches"] += 1
        result = self.robot.measure(x, y, channel)
        self.stats["measures"] += 1
        if is_probe:
            self.stats["probes"] += 1
        return result

    def _clear(self, x: float, y: float, channel: int):
        before = self.robot.current_position
        self.stats["move_distance"] += distance(before, (float(x), float(y)))
        result = self.robot.clear(x, y, channel)
        self.stats["clears"] += 1
        if result.cleared:
            self.stats["clear_success"] += 1
        return result

    # ------------------------------------------------------------ 可行域维护
    def _polygon(self, channel: int) -> np.ndarray:
        # no_signal 可能来自定向遮挡，故刻意排除在几何裁剪之外
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
            self.weak_unknown_channels.discard(channel)
            self.observations[channel].append(
                DirectionObservation(
                    (float(point[0]), float(point[1])), float(bearing_deg)
                )
            )
        elif result == "near":
            self.detected.add(channel)
            self.weak_unknown_channels.discard(channel)
            if not self._clear(point[0], point[1], channel).cleared:
                raise RuntimeError("near 结果未能清除目标")
            self.cleared.add(channel)
        elif result == "no_signal":
            if channel not in self.detected:
                self._record_unknown_no_signal(channel, point)
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

    def _unresolved(self) -> set[int]:
        return {
            channel
            for channel in self.channels
            if channel not in self.detected
            and (
                not self.weak_unknown_pruning_enabled
                or channel not in self.weak_unknown_channels
            )
        }

    @staticmethod
    def _angle_coverage_deg(points: list[Point]) -> float:
        angles = sorted(
            math.degrees(math.atan2(point[1], point[0])) % 360.0
            for point in points
            if math.hypot(point[0], point[1]) > 1e-9
        )
        if len(angles) < 2:
            return 0.0
        gaps = [
            angles[index + 1] - angles[index]
            for index in range(len(angles) - 1)
        ]
        gaps.append(angles[0] + 360.0 - angles[-1])
        return 360.0 - max(gaps)

    def _record_unknown_no_signal(self, channel: int, point: Point) -> None:
        if not self.weak_unknown_pruning_enabled:
            return
        points = self.unknown_no_signal_points[channel]
        candidate = (float(point[0]), float(point[1]))
        if all(distance(candidate, old) > 1e-6 for old in points):
            points.append(candidate)
        if channel in self.weak_unknown_channels:
            return
        if len(points) < self.weak_unknown_min_no_signal_points:
            return
        if self._angle_coverage_deg(points) < self.weak_unknown_min_angle_coverage_deg:
            return
        self.weak_unknown_channels.add(channel)
        self.stats["weak_unknown_marked"] += 1

    # ---------------------------------------------------------------- 阶段A
    def run_survey(self) -> float:
        """访问全部 25 个安全点；清除插入后动态重排剩余路线。"""
        stations = concentric_dodecagon_stations()
        remaining = {
            index: point for index, point in enumerate(stations[1:], start=1)
        }
        plan = (
            []
            if self.dynamic_survey_route
            else optimized_open_route(stations[0], remaining)
        )
        station = stations[0]
        visit_index = 0

        while True:
            self.survey_visited_station_count += 1
            channels = self._unresolved()
            if self.weak_unknown_pruning_enabled:
                self.stats["weak_unknown_survey_skips"] += len(
                    self.weak_unknown_channels - self.detected
                )
            if self.survey_detected_channels:
                channels.update(
                    channel
                    for channel in self.detected - self.cleared
                    if self._needs_free_survey_measurement(channel, station)
                )
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

            if self.dynamic_survey_route:
                # 把所有已经可靠定位、迟早必须访问的清除中心与剩余巡检点
                # 放入同一开放路径。每次只执行第一个节点，随后用新观测重算。
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
            else:
                next_station = remaining[plan[0]]
                self._clear_targets_near_next_survey_edge(next_station)
                next_node = plan.pop(0)
                station = remaining.pop(next_node)
                visit_index += 1

    def _end_survey(self) -> float:
        self.survey_end_time_s = float(self.robot.virtual_time_s)
        return self.survey_end_time_s

    def _clear_targets_near_next_survey_edge(self, next_station: Point) -> None:
        """巡检途中，若顺路清除某个已定位源只需很小绕路，就顺手清掉。"""
        max_detour = self._survey_edge_detour_limit()
        while True:
            best: tuple[float, int, Circle] | None = None
            for channel in self.detected - self.cleared:
                circle = self._circle(channel)
                if circle.radius > SAFE_LOCALIZATION_RADIUS_M:
                    continue
                detour = (
                    distance(self.robot.current_position, circle.center)
                    + distance(circle.center, next_station)
                    - distance(self.robot.current_position, next_station)
                )
                candidate = (detour, channel, circle)
                if best is None or candidate[0:2] < best[0:2]:
                    best = candidate
            if best is None or best[0] > max_detour:
                return
            _, channel, circle = best
            self.stats["survey_inserted_clears"] += 1
            self._clear_circle(channel, circle)

    def _survey_edge_detour_limit(self) -> float:
        if len(self.detected) <= self.LOW_DENSITY_DETECTED_LIMIT:
            return self.LOW_DENSITY_SURVEY_EDGE_DETOUR_M
        return self.MAX_SURVEY_EDGE_DETOUR_M

    def _needs_free_survey_measurement(self, channel: int, station: Point) -> bool:
        """判断在巡检测点上能否"免费"补一次已有频道的测向。"""
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

    def _weak_unknown_probe_candidates(self, channel: int) -> list[tuple[float, Point]]:
        tried = self.unknown_no_signal_points[channel]
        tried_angles = [
            math.degrees(math.atan2(point[1], point[0])) % 360.0
            for point in tried
            if math.hypot(point[0], point[1]) > 1e-9
        ]
        stations = [
            station
            for station in concentric_dodecagon_route()
            if math.hypot(station[0], station[1]) > 1e-9
            and all(distance(station, old) > 1e-6 for old in tried)
        ]
        if not tried_angles:
            return [
                (0.0, station)
                for station in sorted(
                    stations,
                    key=lambda point: distance(self.robot.current_position, point),
                )
            ]

        def nearest_angular_gap(station: Point) -> float:
            angle = math.degrees(math.atan2(station[1], station[0])) % 360.0
            return min(
                abs((angle - tried_angle + 180.0) % 360.0 - 180.0)
                for tried_angle in tried_angles
            )

        return [
            (nearest_angular_gap(station), station)
            for station in sorted(
                stations,
                key=lambda point: (
                    -nearest_angular_gap(point),
                    distance(self.robot.current_position, point),
                ),
            )
        ]

    def _probe_weak_unknown_channels(self) -> bool:
        if not self.weak_unknown_pruning_enabled:
            return False
        candidates: list[tuple[float, float, int, Point]] = []
        for channel in sorted(self.weak_unknown_channels):
            if channel in self.detected:
                continue
            for angular_gap, point in self._weak_unknown_probe_candidates(channel):
                candidates.append(
                    (
                        -angular_gap,
                        distance(self.robot.current_position, point),
                        channel,
                        point,
                    )
                )
        candidates.sort()
        used_channels: set[int] = set()
        for _, _, channel, point in candidates:
            if len(used_channels) >= self.WEAK_UNKNOWN_FINAL_PROBES:
                break
            if channel in used_channels or channel in self.detected:
                continue
            if not self._time_left():
                return False
            used_channels.add(channel)
            result = self._measure(point[0], point[1], channel, is_probe=True)
            self.stats["weak_unknown_final_probes"] += 1
            self._record_measurement(channel, point, result.result, result.svd_deg)
            if channel in self.detected:
                self.stats["weak_unknown_recovered"] += 1
                return True
        return False

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
                if self._probe_weak_unknown_channels():
                    continue
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

    def _cautious_probe_point(self, channel: int) -> Point | None:
        """β-Cautious 自适应补测点（论文 Lemma 1 / Algorithm 1）。

        ``r(i) = σx / sqrt(σβ² − σs²)``，其中 ``σβ = (π/2)·Φ⁻¹(1 − β/2)``：
        站得越远，补测落进定向源"背后"的模糊概率越低，但行程越长；β 就是该风险的
        容忍度。方向取垂直于最大不确定方向（可行域最长轴的法线），站距随当前不确定度
        自适应缩放（区域是硬约束而非高斯，取 σx ≈ R/2）。

        只在"整块可行域都在接收半径内"时才给出候选——否则这次补测可能白跑。
        """
        if not self.CAUTIOUS_PROBE_ENABLED or not self.observations[channel]:
            return None
        observed_steps = sum(
            1
            for index in self.attempted_probes[channel]
            if index >= self.ADAPTIVE_PROBE_INDEX
        )
        if observed_steps >= self.CAUTIOUS_MAX_PROBE_STEPS:
            return None
        polygon = self._polygon(channel)
        if len(polygon) < 3:
            return None
        circle = minimum_enclosing_circle(polygon)
        sigma_beta = 0.5 * math.pi * NormalDist().inv_cdf(
            1.0 - 0.5 * CAUTIOUS_PROBE_BETA
        )
        denominator = max(
            sigma_beta * sigma_beta - CAUTIOUS_SENSOR_SIGMA_RAD**2, 1e-6
        )
        distance_est = (circle.radius / 2.0) / math.sqrt(denominator)
        distance_est = min(distance_est, CAUTIOUS_MAX_PROBE_M)
        if distance_est < CAUTIOUS_MIN_PROBE_M:
            return None

        axis, _ = farthest_pair(polygon)
        normal = (-axis[1], axis[0])
        center = circle.center
        candidates = [
            (
                center[0] + distance_est * normal[0],
                center[1] + distance_est * normal[1],
            ),
            (
                center[0] - distance_est * normal[0],
                center[1] - distance_est * normal[1],
            ),
        ]
        usable = [
            point
            for point in candidates
            if max(distance(point, vertex) for vertex in polygon)
            <= MIN_RECEIVE_RADIUS + 1e-9
        ]
        if not usable:
            return None
        return min(
            usable,
            key=lambda point: (
                -self._block_separation(channel, center, point),
                distance(self.robot.current_position, point),
            ),
        )

    def _probe_options(self, channel: int) -> list[tuple[int, Point]]:
        probes = first_probe_points(self.observations[channel][0])
        options = [
            (index, point)
            for index, point in enumerate(probes)
            if index not in self.attempted_probes[channel]
        ]
        adaptive_index = self.ADAPTIVE_PROBE_INDEX + sum(
            1
            for index in self.attempted_probes[channel]
            if index >= self.ADAPTIVE_PROBE_INDEX
        )
        adaptive = self._cautious_probe_point(channel)
        if adaptive is not None:
            options.insert(0, (adaptive_index, adaptive))
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
        self.grid_fallback_count += 1
        clear_by_oriented_triangular_cover(
            self.robot,
            channel,
            self._polygon(channel),
            time_margin_s=self.time_margin_s,
        )
        self.cleared.add(channel)

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
    weak_unknown_marked_count: float
    weak_unknown_survey_skip_count: float
    weak_unknown_final_probe_count: float
    weak_unknown_recovered_count: float


def run_problem4_case(
    case_id: int,
    seed: int,
    directional_probability: float = 0.5,
    force_mixed: bool = True,
    survey_detected_channels: bool = True,
    dynamic_survey_route: bool = True,
    weak_unknown_pruning_enabled: bool = False,
    weak_unknown_min_no_signal_points: int | None = None,
    weak_unknown_min_angle_coverage_deg: float | None = None,
    route_end_at_origin: bool | None = None,
    source_count: int | None = None,
) -> Problem4CaseResult:
    """随机场景：源数默认在 10~16 随机，位置/接收半径/定向标志全部随机生成。

    ``source_count=10`` 可固定为"恰好 10 个源"的随机场景（源数固定，
    但全向/定向搭配、位置、接收半径、发射方向每次运行都重新随机）。
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
            f"随机 {len(sources)} 源 seed={seed}"
            if source_count is not None
            else f"随机 seed={seed}"
        ),
        seed=seed,
        sources=sources,
        directional_probability=directional_probability,
        survey_detected_channels=survey_detected_channels,
        dynamic_survey_route=dynamic_survey_route,
        weak_unknown_pruning_enabled=weak_unknown_pruning_enabled,
        weak_unknown_min_no_signal_points=weak_unknown_min_no_signal_points,
        weak_unknown_min_angle_coverage_deg=weak_unknown_min_angle_coverage_deg,
        route_end_at_origin=route_end_at_origin,
    )


def _evaluate_case(
    case_id: int,
    case_name: str,
    seed: int,
    sources: list[MixedSource],
    directional_probability: float,
    survey_detected_channels: bool = True,
    dynamic_survey_route: bool = True,
    weak_unknown_pruning_enabled: bool = False,
    weak_unknown_min_no_signal_points: int | None = None,
    weak_unknown_min_angle_coverage_deg: float | None = None,
    route_end_at_origin: bool | None = None,
) -> Problem4CaseResult:
    """在本地仿真器上跑一个给定场景，并做真值校验（漏检、定向判定假阳性）。"""
    simulator = DirectionalLocalSimulator(sources, error_seed=seed + 10_000)
    strategy = Problem4Strategy(
        simulator,
        survey_detected_channels=survey_detected_channels,
        dynamic_survey_route=dynamic_survey_route,
        weak_unknown_pruning_enabled=weak_unknown_pruning_enabled,
        weak_unknown_min_no_signal_points=weak_unknown_min_no_signal_points,
        weak_unknown_min_angle_coverage_deg=weak_unknown_min_angle_coverage_deg,
        route_end_at_origin=route_end_at_origin,
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
        weak_unknown_marked_count=float(stats["weak_unknown_marked"]),
        weak_unknown_survey_skip_count=float(stats["weak_unknown_survey_skips"]),
        weak_unknown_final_probe_count=float(stats["weak_unknown_final_probes"]),
        weak_unknown_recovered_count=float(stats["weak_unknown_recovered"]),
    )


def summarize_problem4(
    results: list[Problem4CaseResult],
    random_state: int,
    directional_probability: float,
    survey_detected_channels: bool,
    dynamic_survey_route: bool = True,
    weak_unknown_pruning_enabled: bool = False,
    weak_unknown_min_no_signal_points: int | None = None,
    weak_unknown_min_angle_coverage_deg: float | None = None,
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
        "strategy": "problem4_concentric_dodecagons",
        "random_state": random_state,
        "case_count": len(results),
        "case_names": [result.case_name for result in results],
        "directional_probability": directional_probability,
        "survey_detected_channels": survey_detected_channels,
        "dynamic_survey_route": dynamic_survey_route,
        "weak_unknown_pruning_enabled": weak_unknown_pruning_enabled,
        "weak_unknown_min_no_signal_points": (
            Problem4Strategy.WEAK_UNKNOWN_MIN_NO_SIGNAL_POINTS
            if weak_unknown_min_no_signal_points is None
            else int(weak_unknown_min_no_signal_points)
        ),
        "weak_unknown_min_angle_coverage_deg": (
            Problem4Strategy.WEAK_UNKNOWN_MIN_ANGLE_COVERAGE_DEG
            if weak_unknown_min_angle_coverage_deg is None
            else float(weak_unknown_min_angle_coverage_deg)
        ),
        "route_end_at_origin": route_end_at_origin,
        "survey_station_count": results[0].survey_station_count,
        "inner_dodecagon_radius_m": INNER_DODECAGON_RADIUS_M,
        "outer_dodecagon_radius_m": OUTER_DODECAGON_RADIUS_M,
        "dodecagon_interleave_angle_deg": DODECAGON_INTERLEAVE_ANGLE_DEG,
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
        "mean_weak_unknown_marked_count": float(
            np.mean([result.weak_unknown_marked_count for result in results])
        ),
        "mean_weak_unknown_survey_skip_count": float(
            np.mean([result.weak_unknown_survey_skip_count for result in results])
        ),
        "mean_weak_unknown_final_probe_count": float(
            np.mean([result.weak_unknown_final_probe_count for result in results])
        ),
        "mean_weak_unknown_recovered_count": float(
            np.mean([result.weak_unknown_recovered_count for result in results])
        ),
        "assumptions": {
            "source_position": "半径为 1800 m 的圆域内按面积均匀、相互独立",
            "receive_radius_m": "在 [1000, 1500] 上独立均匀",
            "directional_flag": "Bernoulli 分布，并可强制混合场景",
            "emission_direction": "在 [0, 2*pi) 上独立均匀",
        },
    }


def write_problem4_results(
    results: list[Problem4CaseResult],
    summary: dict[str, object],
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
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
        "--static-survey-route",
        action="store_true",
        help="关闭清除插入后的剩余巡检点动态重排（用于配对基线）",
    )
    parser.add_argument(
        "--enable-weak-unknown-pruning",
        action="store_true",
        help="启用实验性弱空频道剪枝；该选项不提供确定性无遗漏保证",
    )
    parser.add_argument(
        "--weak-unknown-min-no-signal",
        type=int,
        default=Problem4Strategy.WEAK_UNKNOWN_MIN_NO_SIGNAL_POINTS,
        help="弱空频道判定所需的最少 no_signal 不同测点数",
    )
    parser.add_argument(
        "--weak-unknown-min-coverage",
        type=float,
        default=Problem4Strategy.WEAK_UNKNOWN_MIN_ANGLE_COVERAGE_DEG,
        help="弱空频道判定所需的最小方位覆盖角度（度）",
    )
    parser.add_argument(
        "--no-route-end-at-origin",
        action="store_true",
        help="收尾 TSP 不把最后节点到原点的距离计入代价",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(_HERE) / "outputs/tables/problem4_directional_local",
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
            dynamic_survey_route=not args.static_survey_route,
            weak_unknown_pruning_enabled=args.enable_weak_unknown_pruning,
            weak_unknown_min_no_signal_points=args.weak_unknown_min_no_signal,
            weak_unknown_min_angle_coverage_deg=args.weak_unknown_min_coverage,
            route_end_at_origin=not args.no_route_end_at_origin,
            source_count=args.source_count,
        )
        for index in range(args.cases)
    ]
    summary = summarize_problem4(
        results,
        random_state=random_state,
        directional_probability=args.directional_probability,
        survey_detected_channels=survey_detected_channels,
        dynamic_survey_route=not args.static_survey_route,
        weak_unknown_pruning_enabled=args.enable_weak_unknown_pruning,
        weak_unknown_min_no_signal_points=args.weak_unknown_min_no_signal,
        weak_unknown_min_angle_coverage_deg=args.weak_unknown_min_coverage,
        route_end_at_origin=not args.no_route_end_at_origin,
    )
    write_problem4_results(results, summary, args.output_prefix)

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
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
