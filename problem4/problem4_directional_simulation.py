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
不能再当作几何距离约束使用。策略据此做两点改动：

1. 可行域只由 ``direction`` 观测裁剪（楔形 ∩ 接收盘），``no_signal`` 一律忽略；
2. 巡检点使用 **25 个确定性检测点**：圆心 1 个 + 内正十二边形 12 个（半径 930 m）
   + 外正十二边形 12 个（半径 ``R_o = 1800 / cos15° = 1863.497 m``），两圈错开 15°。
   外圈半径使半径 1800 m 的目标圆恰好内切于外圈，内圈再铺满中心区域，
   因此任意位置、任意发射方向的源都必有一点同时落在其 1000 m 接收圆与 90° 定向半圆内。

用法::

    python problem4/problem4_directional_simulation.py                 # 本地 30 个案例
    python problem4/problem4_directional_simulation.py --cases 200
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

from problem3_geometry import (  # noqa: E402
    circumscribed_polygon,
    clip_wedge,
    distance_point_to_polygon,
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


def exact_open_route(start: Point, nodes: dict[int, Point]) -> list[int]:
    """Held-Karp 精确开放路径（节点数很少时使用，保证路线最短）。"""
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
    current = min(range(count), key=lambda index: best[(full_mask, index)][0])
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
) -> list[MixedSource]:
    """按题面假设生成 10~16 个源，其中一部分为定向源。"""
    if not 0.0 <= directional_probability <= 1.0:
        raise ValueError("directional_probability 必须落在 [0, 1]")
    rng = np.random.default_rng(seed)
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
    MAX_PROBE_COMBINATIONS = 64
    MAX_EXACT_ROUTE_NODES = 12
    MAX_SURVEY_EDGE_DETOUR_M = 375.0
    MIN_SURVEY_CROSSING_ANGLE_DEG = 25.0

    def __init__(
        self,
        robot,
        survey_detected_channels: bool = True,
        time_margin_s: float = 30.0,
    ):
        self.robot = robot
        self.survey_detected_channels = survey_detected_channels
        self.time_margin_s = time_margin_s

        self.channels = list(range(CHANNEL_MIN, CHANNEL_MAX + 1))
        self.observations: dict[int, list[DirectionObservation]] = {
            channel: [] for channel in self.channels
        }
        self.detected: set[int] = set()
        self.cleared: set[int] = set()
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
        }
        self.survey_measurement_count = 0
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
            "survey_station_count": len(concentric_dodecagon_stations()),
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

    def _unresolved(self) -> set[int]:
        return {channel for channel in self.channels if channel not in self.detected}

    # ---------------------------------------------------------------- 阶段A
    def run_survey(self) -> float:
        """25 点巡检：先圆心、再按最短路扫内外两圈。"""
        for visit_index, station in enumerate(concentric_dodecagon_route()):
            if visit_index > 0:
                self._clear_targets_near_next_survey_edge(station)
            channels = self._unresolved()
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
        return self._end_survey()

    def _end_survey(self) -> float:
        self.survey_end_time_s = float(self.robot.virtual_time_s)
        return self.survey_end_time_s

    def _clear_targets_near_next_survey_edge(self, next_station: Point) -> None:
        """巡检途中，若顺路清除某个已定位源只需很小绕路，就顺手清掉。"""
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
            if best is None or best[0] > self.MAX_SURVEY_EDGE_DETOUR_M:
                return
            _, channel, circle = best
            self.stats["survey_inserted_clears"] += 1
            self._clear_circle(channel, circle)

    def _needs_free_survey_measurement(self, channel: int, station: Point) -> bool:
        """判断在巡检测点上能否"免费"补一次已有频道的测向。"""
        if channel in self.cleared or not self.observations[channel]:
            return False
        polygon = self._polygon(channel)
        circle = minimum_enclosing_circle(polygon)
        if circle.radius <= SAFE_LOCALIZATION_RADIUS_M:
            return False
        if distance_point_to_polygon(station, polygon) > MAX_RECEIVE_RADIUS + 1e-9:
            self.skipped_impossible_remeasure_count += 1
            return False
        if self.MIN_SURVEY_CROSSING_ANGLE_DEG > 0.0:
            center = circle.center
            candidate_vector = (
                station[0] - center[0],
                station[1] - center[1],
            )
            candidate_norm = math.hypot(candidate_vector[0], candidate_vector[1])
            if candidate_norm > 1e-9:
                maximum_crossing_angle = 0.0
                for observation in self.observations[channel]:
                    existing_vector = (
                        observation.position[0] - center[0],
                        observation.position[1] - center[1],
                    )
                    existing_norm = math.hypot(
                        existing_vector[0], existing_vector[1]
                    )
                    if existing_norm <= 1e-9:
                        maximum_crossing_angle = 90.0
                        break
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
                    maximum_crossing_angle = max(
                        maximum_crossing_angle, crossing_angle
                    )
                if maximum_crossing_angle < self.MIN_SURVEY_CROSSING_ANGLE_DEG:
                    return False
        return True

    # ---------------------------------------------------------------- 阶段B
    def finish(self) -> None:
        """未清除频道统一调度：补测缩小可行域、两次清除、网格兜底。"""
        while self.detected - self.cleared:
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
                route_function = (
                    exact_open_route
                    if len(nodes) <= self.MAX_EXACT_ROUTE_NODES
                    else optimized_open_route
                )
                route = route_function(self.robot.current_position, nodes)
                length = open_route_length(
                    self.robot.current_position, route, nodes
                )
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
            else:
                self._clear_circle(next_node, clear_circles[next_node])

    def _probe_options(self, channel: int) -> list[tuple[int, Point]]:
        probes = first_probe_points(self.observations[channel][0])
        return [
            (index, point)
            for index, point in enumerate(probes)
            if index not in self.attempted_probes[channel]
        ]

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


def run_problem4_case(
    case_id: int,
    seed: int,
    directional_probability: float = 0.5,
    force_mixed: bool = True,
    survey_detected_channels: bool = True,
) -> Problem4CaseResult:
    sources = generate_mixed_sources(
        seed,
        directional_probability=directional_probability,
        force_mixed=force_mixed,
    )
    simulator = DirectionalLocalSimulator(sources, error_seed=seed + 10_000)
    strategy = Problem4Strategy(
        simulator, survey_detected_channels=survey_detected_channels
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
    )


def summarize_problem4(
    results: list[Problem4CaseResult],
    random_state: int,
    directional_probability: float,
    survey_detected_channels: bool,
) -> dict[str, object]:
    totals = np.array([result.total_time_s for result in results], dtype=float)
    source_counts = np.array(
        [result.source_count for result in results], dtype=float
    )
    # 单源定位清除时间 = 该案例总虚拟时间 / 该案例干扰源数
    per_source = totals / source_counts
    return {
        "strategy": "problem4_concentric_dodecagons",
        "random_state": random_state,
        "case_count": len(results),
        "directional_probability": directional_probability,
        "survey_detected_channels": survey_detected_channels,
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
        "mean_directional_count": float(
            np.mean([result.directional_count for result in results])
        ),
        "mean_omnidirectional_count": float(
            np.mean([result.omnidirectional_count for result in results])
        ),
        "total_directional_count": int(
            sum(result.directional_count for result in results)
        ),
        "total_omnidirectional_count": int(
            sum(result.omnidirectional_count for result in results)
        ),
        "mean_confirmed_directional_count": float(
            np.mean(
                [result.confirmed_directional_count for result in results]
            )
        ),
        "directional_confirmation_rate": (
            float(
                sum(result.confirmed_directional_count for result in results)
                / sum(result.directional_count for result in results)
            )
            if sum(result.directional_count for result in results) > 0
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题4 定向源本地仿真批量验证")
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--directional-probability", type=float, default=0.5)
    parser.add_argument("--allow-pure", action="store_true")
    parser.add_argument(
        "--survey-unresolved-only",
        action="store_true",
        help="关闭巡检途中对已发现频道的顺路补测",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(_HERE) / "outputs/tables/problem4_directional_local",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [
        run_problem4_case(
            case_id=index + 1,
            seed=args.random_state + index,
            directional_probability=args.directional_probability,
            force_mixed=not args.allow_pure,
            survey_detected_channels=not args.survey_unresolved_only,
        )
        for index in range(args.cases)
    ]
    summary = summarize_problem4(
        results,
        random_state=args.random_state,
        directional_probability=args.directional_probability,
        survey_detected_channels=not args.survey_unresolved_only,
    )
    write_problem4_results(results, summary, args.output_prefix)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
