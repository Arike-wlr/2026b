"""问题 3：机器狗自动搜索、定位与清除策略。

字典序目标：

1. 保证所有合法情况下都不漏检，并最终清除全部干扰源；
2. 在满足第 1 项的策略中，尽量减小“均匀面积分布”下的平均虚拟时间。

策略层只依赖一个“类 RobotClient 接口”的对象（鸭子类型），因此既能接真实
``client.RobotClient``，也能接离线 ``problem3_local_sim.LocalSimulator``。用到的成员：

    enter() / measure(x, y, channel) / clear(x, y, channel) / exit()
    current_position / current_channel / virtual_time_s / remaining_real_s
    is_time_up(margin_s) / estimate_move_time(x, y) / in_session

本模块提供两套可互换的策略：

* ``Problem3Strategy``：固定七点巡检 + 全局动态任务池，只依赖 numpy；
* ``CoverageStrategy``：实际覆盖地图 + 联合行程规划 + 同点共享定位 + 补盲位置区域选点，
  需要 ``shapely``（未安装时仅该类不可用，基础策略不受影响）。

本模块不做任何 HTTP，可在没有模拟器时用本地仿真完整跑通。
"""

from __future__ import annotations

import math
import os
from itertools import combinations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from problem3_geometry import (
    clip_halfplane,
    clip_wedge,
    circumscribed_polygon,
    farthest_pair,
    grid_cover_points,
    intersect_convex,
    intersect_disk,
    min_enclosing_circle,
    nearest_neighbor_route,
    perpendicular_left,
    point_in_polygon,
    polygon_area,
    two_opt_open,
    unit_from_deg,
)

# 覆盖地图策略（CoverageStrategy）需要 shapely 做精确的多边形差集/并集运算。
# 这里做可选导入：没装 shapely 时，基础 Problem3Strategy 仍然可以正常使用。
try:  # pragma: no cover - 取决于运行环境
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import nearest_points, unary_union

    SHAPELY_AVAILABLE = True
except ImportError:  # noqa: BLE001 - 缺少 shapely 不是致命错误
    LineString = Point = Polygon = None  # type: ignore[assignment]
    nearest_points = unary_union = None  # type: ignore[assignment]
    SHAPELY_AVAILABLE = False


# --------------------------------------------------------------------------- #
# 题面常量
# --------------------------------------------------------------------------- #
TARGET_RADIUS = 1800.0          # 目标圆域半径 m
MIN_RECEIVE_RADIUS = 1000.0     # 干扰源最小有效接收半径 m
MAX_RECEIVE_RADIUS = 1500.0     # 干扰源最大有效接收半径 m
NEAR_RADIUS = 5.0               # 返回 near 的距离阈值 m
CLEAR_RADIUS = 20.0             # 清除半径 m
BEARING_ERROR_DEG = 1.0         # 示向度最大绝对误差 度
MOVE_SPEED_MPS = 5.0            # 移动速度 m/s
MEASURE_TIME_S = 5.0            # 单次检测时间 s
SWITCH_TIME_S = 1.0             # 切换检测频道时间 s
CLEAR_FAIL_TIME_S = 3.0         # 清除失败操作时间 s
CLEAR_SUCCESS_TIME_S = 5.0      # 清除成功操作时间 s
CHANNEL_MIN = 1
CHANNEL_MAX = 20
SOURCE_COUNT_MIN = 10
SOURCE_COUNT_MAX = 16
SURVEY_RING_RADIUS = 1150.0     # 六边形巡检点半径 m
SAFE_REGION_RADIUS = 58.0       # 两次清除法可行域半径上界 m

# 策略内部可调参数
GRID_SPACING = CLEAR_RADIUS * math.sqrt(2.0) * (1.0 - 1e-3)  # 保证网格覆盖 20 m 圆
PROBE_R_EST_MIN = 100.0         # 动态拉偏：目标距离估计下限
PROBE_R_EST_MAX = 800.0         # 动态拉偏：目标距离估计上限
PROBE_SIDE_RATIO = 0.6          # 动态拉偏：侧向偏移比例，兼顾交会角与移动距离
PROBE_SIDE_MAX = 350.0          # 动态拉偏：侧向偏移上限
PROBE_FORWARD_RATIO = 0.5       # 动态拉偏：前向跟进比例
PROBE_BACKUP_FORWARD = 350.0    # 短距离备用补测点，避免动态拉偏失败后过早大步长外跑
PROBE_BACKUP_SIDE = 200.0
PROBE_FAR_BACKUP_FORWARD = 600.0  # 全距离接收保证备用补测点
PROBE_FAR_BACKUP_SIDE = 300.0
PROBE_STAGNATION_LIMIT = 3      # 连续多少次补测未缩小可行域就转网格兜底
PROBE_MAX_PER_CHANNEL = 12      # 单频道补测次数上限
REGION_CLUSTER_DISTANCE = 150.0  # 可行域中心小于该距离则归为同一空间批次
OPPORTUNISTIC_MAX_DETOUR = 120.0  # 顺路清除允许增加的最大路程 m
TSP_EXACT_LIMIT = 9          # Held-Karp 精确开放路径的节点上限

STATUS_UNKNOWN = "UNKNOWN"
STATUS_DETECTED = "DETECTED"
STATUS_CLEARED = "CLEARED"
STATUS_EMPTY = "EMPTY_CERTIFIED"


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
class ActionLogger:
    """结构化动作日志：每条动作一行，便于赛后统计与复盘。"""

    def __init__(self, path: Optional[str] = None, verbose: bool = False):
        self.path = path
        self.verbose = verbose
        self._header_written = False
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "seq\ttime_s\tphase\taction\tx\ty\tchannel\tresult\tsvd_deg\t"
                    "channel_status\tfeasible_radius\tnote\n"
                )
            self._header_written = True

    def log(self, seq: int, virtual_time: float, phase: str, action: str,
            x: float, y: float, channel: int, result: str = "-",
            svd_deg: Optional[float] = None, channel_status: str = "-",
            feasible_radius: Optional[float] = None, note: str = "") -> None:
        svd_text = "-" if svd_deg is None else f"{svd_deg:.3f}"
        radius_text = "-" if feasible_radius is None else f"{feasible_radius:.3f}"
        line = (
            f"{seq}\t{virtual_time:.3f}\t{phase}\t{action}\t{x:.3f}\t{y:.3f}\t"
            f"{channel}\t{result}\t{svd_text}\t{channel_status}\t{radius_text}\t{note}"
        )
        if self.verbose:
            print(line)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #
@dataclass
class ChannelState:
    channel: int
    status: str = STATUS_UNKNOWN
    # 可行域：凸多边形（保守包含真实位置）
    feasible: Optional[np.ndarray] = None
    # 排除圆列表 (cx, cy, r)，只做记录与诊断
    exclusions: List[Tuple[float, float, float]] = field(default_factory=list)
    # 最小包围圆 (center(np.array), radius)
    enclosing: Optional[Tuple[np.ndarray, float]] = None
    # 第一次测向（位置、单位方向），用于第一组补测点
    first_direction: Optional[Tuple[np.ndarray, np.ndarray]] = None
    # 补测计划（第一组两个点，先近后远）
    probe_plan: List[np.ndarray] = field(default_factory=list)
    probe_count: int = 0
    stagnation: int = 0
    grid_mode: bool = False
    last_probe: Optional[np.ndarray] = None
    completed_points: Set[int] = field(default_factory=set)

    @property
    def radius(self) -> Optional[float]:
        return None if self.enclosing is None else float(self.enclosing[1])

    @property
    def center(self) -> Optional[np.ndarray]:
        return None if self.enclosing is None else self.enclosing[0]


@dataclass
class StrategyConfig:
    target_radius: float = TARGET_RADIUS
    survey_radius: float = SURVEY_RING_RADIUS
    bearing_error_deg: float = BEARING_ERROR_DEG
    safe_region_radius: float = SAFE_REGION_RADIUS
    time_margin_s: float = 30.0          # 现实时间安全余量
    verbose: bool = False
    log_path: Optional[str] = None
    # 达到 16 个干扰源后可直接结束
    stop_at_source_count: int = SOURCE_COUNT_MAX
    # Region Clustering：按可行域中心聚类，批次内连续局部化/清除，减少跨区域往返
    region_cluster_distance: float = REGION_CLUSTER_DISTANCE
    # Opportunistic Strike：移动途中若小半径目标几乎顺路，则立即清除
    opportunistic_max_detour: float = OPPORTUNISTIC_MAX_DETOUR
    opportunistic_enabled: bool = True


@dataclass
class RouteTask:
    kind: str
    point: np.ndarray
    channel_state: ChannelState
    note: str = ""


# --------------------------------------------------------------------------- #
# 七点巡检点
# --------------------------------------------------------------------------- #
def build_survey_points(radius: float = SURVEY_RING_RADIUS) -> List[np.ndarray]:
    """P0 = 原点，P1..P6 = 半径 radius 的正六边形顶点（角度 0,60,...,300°）。"""
    pts = [np.array([0.0, 0.0])]
    for k in range(6):
        a = math.radians(60.0 * k)
        pts.append(np.array([radius * math.cos(a), radius * math.sin(a)]))
    return pts


# --------------------------------------------------------------------------- #
# 策略主体
# --------------------------------------------------------------------------- #
class Problem3Strategy:
    def __init__(self, robot, config: Optional[StrategyConfig] = None):
        self.robot = robot
        self.cfg = config or StrategyConfig()
        self.logger = ActionLogger(self.cfg.log_path, self.cfg.verbose)

        self.channels: Dict[int, ChannelState] = {
            c: ChannelState(channel=c) for c in range(CHANNEL_MIN, CHANNEL_MAX + 1)
        }
        self.seq = 0
        self.cleared_count = 0
        self._session_owned = False

        # 统计
        self.stats: Dict[str, float] = {
            "measures": 0, "clears": 0, "clear_success": 0,
            "move_distance": 0.0, "switch_count": 0,
            "probes": 0, "grid_clears": 0,
            "two_clear_used": 0, "grid_channels": 0,
            "clusters": 0, "opportunistic_strikes": 0,
            "shared_measures": 0,
            "empty_pruned": 0, "survey_skipped_empty": 0,
            "clipped_to_clear": 0,
        }

    # -------------------------------------------------------------- 运行入口
    def run(self) -> Dict[str, object]:
        self._enter()
        try:
            self._phase_survey()
            self._phase_tsp_routing()
        finally:
            self._exit()
        return self._summary()

    # ------------------------------------------------------------ 底层动作封装
    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _log(self, phase: str, action: str, x: float, y: float, channel: int,
             result: str = "-", svd_deg: Optional[float] = None,
             cs: Optional[ChannelState] = None, note: str = "") -> None:
        self.logger.log(
            seq=self._next_seq(),
            virtual_time=self.robot.virtual_time_s,
            phase=phase, action=action, x=x, y=y, channel=channel,
            result=result, svd_deg=svd_deg,
            channel_status=cs.status if cs else "-",
            feasible_radius=cs.radius if cs else None,
            note=note,
        )

    def _enter(self) -> None:
        if not getattr(self.robot, "in_session", False):
            self.robot.enter()
            self._session_owned = True
        self._log("INIT", "enter", 0.0, 0.0, CHANNEL_MIN,
                  result="entered",
                  note=f"remaining_real_s={self.robot.remaining_real_s:.1f}")

    def _exit(self) -> None:
        try:
            if getattr(self.robot, "in_session", False):
                self.robot.exit()
                self._log("EXIT", "exit", self.robot.current_position[0],
                          self.robot.current_position[1], CHANNEL_MIN,
                          result="exit", note=f"cleared={self.cleared_count}")
        except Exception as error:  # noqa: BLE001 - 收尾失败不应掩盖主流程结果
            self._log("EXIT", "exit_failed", 0.0, 0.0, CHANNEL_MIN,
                      result="error", note=repr(error))

    def _measure(self, x: float, y: float, channel: int, phase: str,
                 cs: Optional[ChannelState] = None, note: str = ""):
        before = self.robot.current_position
        self.stats["move_distance"] += math.hypot(x - before[0], y - before[1])
        cur_ch = self.robot.current_channel
        if cur_ch is not None and cur_ch != channel:
            self.stats["switch_count"] += 1
        res = self.robot.measure(x, y, channel)
        self.stats["measures"] += 1
        self._log(phase, "measure", x, y, channel, result=res.result,
                  svd_deg=res.svd_deg, cs=cs, note=note)
        return res

    def _clear(self, x: float, y: float, channel: int, phase: str,
               cs: Optional[ChannelState] = None, note: str = ""):
        before = self.robot.current_position
        self.stats["move_distance"] += math.hypot(x - before[0], y - before[1])
        res = self.robot.clear(x, y, channel)
        self.stats["clears"] += 1
        if res.cleared:
            self.stats["clear_success"] += 1
        self._log(phase, "clear", x, y, channel,
                  result="success" if res.cleared else "no_target_in_range",
                  cs=cs, note=note)
        return res

    def _time_left(self) -> bool:
        return not self.robot.is_time_up(self.cfg.time_margin_s)

    # ------------------------------------------------------------ 可行域更新
    def _initial_region(self) -> np.ndarray:
        return circumscribed_polygon((0.0, 0.0), self.cfg.target_radius)

    def _update_direction(self, cs: ChannelState, S: np.ndarray,
                          svd_deg: float) -> None:
        """direction：楔形 ∩ 盘(S,1500)，并排除盘(S,5)。"""
        if cs.feasible is None:
            cs.feasible = self._initial_region()
        theta = math.radians(svd_deg)
        half = math.radians(self.cfg.bearing_error_deg + 1e-12)
        cs.feasible = clip_wedge(cs.feasible, S, theta, half)
        cs.feasible = intersect_disk(cs.feasible, S, MAX_RECEIVE_RADIUS)
        cs.exclusions.append((float(S[0]), float(S[1]), NEAR_RADIUS))
        if cs.first_direction is None:
            cs.first_direction = (np.asarray(S, dtype=float),
                                  unit_from_deg(svd_deg))
        self._recompute_enclosing(cs)

    def _update_near(self, cs: ChannelState, S: np.ndarray) -> None:
        """near：盘(S,5)。"""
        if cs.feasible is None:
            cs.feasible = self._initial_region()
        cs.feasible = intersect_disk(cs.feasible, S, NEAR_RADIUS)
        self._recompute_enclosing(cs)

    def _update_no_signal(self, cs: ChannelState, S: np.ndarray) -> None:
        """no_signal：记录排除盘；若此前有信号点，则加入保守线性半平面。"""
        cs.exclusions.append((float(S[0]), float(S[1]), MIN_RECEIVE_RADIUS))
        if cs.feasible is not None and cs.first_direction is not None:
            before_radius = cs.radius
            signal_point, _ = cs.first_direction
            q = np.asarray(S, dtype=float)
            p = np.asarray(signal_point, dtype=float)
            direction = q - p
            # 由 |G-p| <= rc < |G-q| 得 2G·(q-p) <= |q|^2 - |p|^2。
            # clip_halfplane 形式是 a*x+b*y+c >= 0，故整体取负。
            rhs = float(np.dot(q, q) - np.dot(p, p))
            cs.feasible = clip_halfplane(
                cs.feasible,
                -2.0 * float(direction[0]),
                -2.0 * float(direction[1]),
                rhs,
            )
            self._recompute_enclosing(cs)
            if (
                (before_radius is None or before_radius > self.cfg.safe_region_radius)
                and cs.radius is not None
                and cs.radius <= self.cfg.safe_region_radius
            ):
                self.stats["clipped_to_clear"] += 1

    def _update_clear_failure(self, cs: ChannelState, C: np.ndarray) -> None:
        cs.exclusions.append((float(C[0]), float(C[1]), CLEAR_RADIUS))

    def _recompute_enclosing(self, cs: ChannelState) -> None:
        if cs.feasible is None or len(cs.feasible) == 0 or polygon_area(cs.feasible) < 1e-6:
            cs.enclosing = None
            return
        cs.enclosing = min_enclosing_circle(cs.feasible)

    def _region_contains_safe(self, cs: ChannelState) -> bool:
        return cs.feasible is not None and len(cs.feasible) >= 3

    # ---------------------------------------------------------------- 阶段A
    def _channel_order(self, station_id: int) -> List[int]:
        if station_id % 2 == 0:
            return list(range(CHANNEL_MIN, CHANNEL_MAX + 1))
        return list(range(CHANNEL_MAX, CHANNEL_MIN - 1, -1))

    def _known_source_count(self) -> int:
        """已确认存在过干扰源的频道数：DETECTED + CLEARED。"""
        return sum(
            1 for cs in self.channels.values()
            if cs.status in (STATUS_DETECTED, STATUS_CLEARED)
        )

    def _mark_empty(self, cs: ChannelState, note: str) -> None:
        if cs.status == STATUS_UNKNOWN:
            cs.status = STATUS_EMPTY
            self.stats["empty_pruned"] += 1
            self._log("PRUNE", "mark_empty", self.robot.current_position[0],
                      self.robot.current_position[1], cs.channel,
                      result="empty", cs=cs, note=note)

    def _certify_empty_channels(self, stations_count: int) -> None:
        """只执行确定性安全的空频道判定。

        1. 频道完成全部七点检测仍未发现，则由七点覆盖定理判空；
        2. 已确认存在的频道数达到题面上限 16，则剩余 UNKNOWN 频道必为空。
        """
        all_station_ids = set(range(stations_count))
        for cs in self.channels.values():
            if cs.status == STATUS_UNKNOWN and cs.completed_points == all_station_ids:
                self._mark_empty(cs, "all survey stations no_signal")

        if self._known_source_count() >= SOURCE_COUNT_MAX:
            for cs in self.channels.values():
                if cs.status == STATUS_UNKNOWN:
                    self._mark_empty(cs, "source upper bound reached")

    def _should_skip_survey_channel(self, cs: ChannelState,
                                    stations_count: int) -> bool:
        self._certify_empty_channels(stations_count)
        if cs.status in (STATUS_CLEARED, STATUS_EMPTY):
            self.stats["survey_skipped_empty"] += 1
            return True
        if self._known_source_count() >= SOURCE_COUNT_MAX and cs.status == STATUS_UNKNOWN:
            self._mark_empty(cs, "source upper bound reached")
            self.stats["survey_skipped_empty"] += 1
            return True
        return False

    def _phase_survey(self) -> None:
        stations = build_survey_points(self.cfg.survey_radius)
        for station_id, pos in enumerate(stations):
            if not self._time_left():
                break
            if self.cleared_count >= self.cfg.stop_at_source_count:
                break
            for channel in self._channel_order(station_id):
                cs = self.channels[channel]
                if self._should_skip_survey_channel(cs, len(stations)):
                    continue
                if not self._time_left():
                    break
                res = self._measure(float(pos[0]), float(pos[1]), channel,
                                    phase="SURVEY", cs=cs,
                                    note=f"station={station_id}")
                cs.completed_points.add(station_id)

                if res.result == "direction":
                    cs.status = STATUS_DETECTED
                    self._update_direction(cs, pos, float(res.svd_deg))
                elif res.result == "near":
                    cs.status = STATUS_DETECTED
                    self._update_near(cs, pos)
                    clear_res = self._clear(float(pos[0]), float(pos[1]),
                                            channel, phase="SURVEY", cs=cs,
                                            note="near during survey")
                    if clear_res.cleared:
                        cs.status = STATUS_CLEARED
                        self.cleared_count += 1
                    else:
                        # 理论上不会发生；记录并保留状态，后续兜底
                        cs.note_anomaly = True  # type: ignore[attr-defined]
                else:
                    self._update_no_signal(cs, pos)

                self._certify_empty_channels(len(stations))

        self._certify_empty_channels(len(stations))

    # ---------------------------------------------------------------- 阶段B
    def _detected_pending(self) -> List[ChannelState]:
        return [
            cs for cs in self.channels.values()
            if cs.status == STATUS_DETECTED and cs.feasible is not None
        ]

    def _state_center(self, cs: ChannelState) -> np.ndarray:
        if cs.center is not None:
            return np.asarray(cs.center, dtype=float)
        if cs.feasible is not None and len(cs.feasible) > 0:
            return np.mean(cs.feasible, axis=0)
        return np.asarray(self.robot.current_position, dtype=float)

    def _cluster_center(self, cluster: List[ChannelState]) -> np.ndarray:
        centers = [self._state_center(cs) for cs in cluster]
        return np.mean(np.vstack(centers), axis=0)

 
    @staticmethod
    def _point_segment_distance(point: np.ndarray, start: np.ndarray,
                                end: np.ndarray) -> float:
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            return float(np.linalg.norm(point - start))
        t = float(np.dot(point - start, segment) / length_sq)
        t = max(0.0, min(1.0, t))
        projection = start + t * segment
        return float(np.linalg.norm(point - projection))

    def _opportunistic_strike(self, target: np.ndarray,
                              exclude_channel: Optional[int] = None) -> None:
        """动态顺路拦截：去目标点前，顺手清除已经足够小的可行域。

        只对 radius <= 20 m 的频道做单点清除，因此不会牺牲“保证清除”的确定性；
        额外路程受 opportunistic_max_detour 限制，避免为了顺手反而绕远。
        """
        if not self.cfg.opportunistic_enabled or not self._time_left():
            return

        start = np.asarray(self.robot.current_position, dtype=float)
        target = np.asarray(target, dtype=float)
        candidates: List[Tuple[float, float, ChannelState]] = []
        for cs in self._detected_pending():
            if cs.channel == exclude_channel or cs.status == STATUS_CLEARED:
                continue
            if cs.radius is None or cs.radius > CLEAR_RADIUS or cs.center is None:
                continue
            center = np.asarray(cs.center, dtype=float)
            detour = (
                math.hypot(center[0] - start[0], center[1] - start[1])
                + math.hypot(target[0] - center[0], target[1] - center[1])
                - math.hypot(target[0] - start[0], target[1] - start[1])
            )
            segment_distance = self._point_segment_distance(center, start, target)
            if detour <= self.cfg.opportunistic_max_detour:
                candidates.append((detour, segment_distance, cs))

        candidates.sort(key=lambda item: (item[0], item[1], item[2].channel))
        for _, _, cs in candidates:
            if not self._time_left() or cs.status == STATUS_CLEARED:
                continue
            if cs.center is None or cs.radius is None or cs.radius > CLEAR_RADIUS:
                continue
            center = np.asarray(cs.center, dtype=float)
            res = self._clear(float(center[0]), float(center[1]), cs.channel,
                              phase="OPPORTUNISTIC", cs=cs,
                              note="opportunistic strike on route")
            self.stats["opportunistic_strikes"] += 1
            if res.cleared:
                self._mark_cleared(cs)
            else:
                self._update_clear_failure(cs, center)

 
    def _phase_tsp_routing(self) -> None:
        """统一目标池 + 动态重规划。

        每轮把所有待处理频道合并成一个任务池：
        - LOCALIZE：可行域仍大的频道，选择下一补测点；
        - CLEAR：半径足够小的频道，直接清除或两次清除；
        - GRID：需要兜底覆盖的频道，每轮只执行一个网格点。

        任务池用带“顺路收益”的贪心 TSP 排序；每执行一个任务后重新建池，
        让新测向、新清除和顺路机会即时进入下一轮决策。
        """
        for cs in self._detected_pending():
            self._recompute_enclosing(cs)

        while self._time_left():
            if self.cleared_count >= self.cfg.stop_at_source_count:
                break
            tasks = self._build_route_tasks()
            if not tasks:
                break
            order = self._tsp_task_order(tasks)
            if not order:
                break
            task = tasks[order[0]]
            self._execute_route_task(task)

    def _build_route_tasks(self) -> List[RouteTask]:
        tasks: List[RouteTask] = []
        robot_pos = np.asarray(self.robot.current_position, dtype=float)

        for cs in self._detected_pending():
            if cs.status == STATUS_CLEARED:
                continue
            if cs.enclosing is None:
                self._recompute_enclosing(cs)

            if cs.radius is not None and cs.radius <= self.cfg.safe_region_radius:
                tasks.append(RouteTask("clear", cs.center.copy(), cs, "safe clear"))
                continue

            if cs.radius is not None and cs.radius > self.cfg.safe_region_radius \
                    and not cs.grid_mode and cs.probe_count < PROBE_MAX_PER_CHANNEL:
                probe = self._peek_probe(cs, robot_pos)
                if probe is not None:
                    point, kind = probe
                    tasks.append(RouteTask("localize", point, cs, kind))
                    continue
                cs.grid_mode = True

            if cs.feasible is not None:
                tasks.append(RouteTask("grid", self._clear_target(cs), cs, "grid step"))

        return tasks

    def _ready_clear_points(self, tasks: List[RouteTask]) -> List[np.ndarray]:
        return [
            np.asarray(task.point, dtype=float)
            for task in tasks
            if task.kind == "clear"
            and task.channel_state.radius is not None
            and task.channel_state.radius <= CLEAR_RADIUS
        ]

    def _edge_cost(self, start: np.ndarray, end: np.ndarray,
                   _ready_clear_points: List[np.ndarray]) -> float:
        return float(np.linalg.norm(end - start))

    def _tsp_task_order(self, tasks: List[RouteTask]) -> List[int]:
        """以欧氏距离为边权的开放 TSP；小任务集用 Held-Karp 精确求解。"""
        if not tasks:
            return []
        if len(tasks) <= TSP_EXACT_LIMIT:
            return self._held_karp_task_order(tasks)

        remaining = set(range(len(tasks)))
        order: List[int] = []
        current = np.asarray(self.robot.current_position, dtype=float)
        ready_clear_points = self._ready_clear_points(tasks)

        while remaining:
            best_idx = min(
                remaining,
                key=lambda idx: (
                    self._edge_cost(current, np.asarray(tasks[idx].point, dtype=float), ready_clear_points),
                    0 if tasks[idx].kind == "clear" else 1 if tasks[idx].kind == "localize" else 2,
                    tasks[idx].channel_state.channel,
                ),
            )
            order.append(best_idx)
            remaining.remove(best_idx)
            current = np.asarray(tasks[best_idx].point, dtype=float)
        points = [task.point for task in tasks]
        return two_opt_open(order, self.robot.current_position, points)

    def _held_karp_task_order(self, tasks: List[RouteTask]) -> List[int]:
        n = len(tasks)
        if n == 0:
            return []
        start = np.asarray(self.robot.current_position, dtype=float)
        ready_clear_points = self._ready_clear_points(tasks)
        points = [np.asarray(task.point, dtype=float) for task in tasks]

        dp: Dict[Tuple[int, int], Tuple[float, Optional[int]]] = {}
        for j in range(n):
            mask = 1 << j
            dp[(mask, j)] = (self._edge_cost(start, points[j], ready_clear_points), None)

        for size in range(2, n + 1):
            for subset in combinations(range(n), size):
                mask = 0
                for item in subset:
                    mask |= 1 << item
                for j in subset:
                    prev_mask = mask ^ (1 << j)
                    best_cost = float("inf")
                    best_prev = None
                    for i in subset:
                        if i == j:
                            continue
                        prev_cost = dp[(prev_mask, i)][0]
                        cost = prev_cost + self._edge_cost(points[i], points[j], ready_clear_points)
                        if cost < best_cost:
                            best_cost = cost
                            best_prev = i
                    dp[(mask, j)] = (best_cost, best_prev)

        full_mask = (1 << n) - 1
        end = min(range(n), key=lambda j: dp[(full_mask, j)][0])
        route: List[int] = []
        mask = full_mask
        current = end
        while current is not None:
            route.append(current)
            _, prev = dp[(mask, current)]
            mask ^= 1 << current
            current = prev
        route.reverse()
        return route

    def _execute_route_task(self, task: RouteTask) -> None:
        cs = task.channel_state
        self._opportunistic_strike(task.point, exclude_channel=cs.channel)
        if cs.status == STATUS_CLEARED:
            self._shared_measure_at(np.asarray(self.robot.current_position, dtype=float),
                                    exclude_channel=cs.channel)
            return

        if task.kind == "localize":
            self._execute_probe(cs, task.point, task.note)
        elif task.kind == "clear":
            self._clear_step(cs)
        else:
            self._grid_clear_step(cs)
        self._shared_measure_at(np.asarray(self.robot.current_position, dtype=float),
                                exclude_channel=cs.channel)

    def _max_vertex_distance(self, point: np.ndarray, poly: np.ndarray) -> float:
        if poly is None or len(poly) == 0:
            return float("inf")
        diff = poly - point
        return float(np.max(np.hypot(diff[:, 0], diff[:, 1])))

    def _estimated_intersection_angle_deg(self, point: np.ndarray,
                                          cs: ChannelState) -> float:
        if cs.first_direction is None or cs.center is None:
            return 0.0
        _, old_u = cs.first_direction
        new_vec = np.asarray(cs.center, dtype=float) - point
        norm = float(np.linalg.norm(new_vec))
        if norm < 1e-9:
            return 90.0
        new_u = new_vec / norm
        dot = abs(float(np.dot(old_u, new_u)))
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def _shared_measure_at(self, point: np.ndarray,
                           exclude_channel: Optional[int] = None) -> None:
        candidates: List[Tuple[float, ChannelState]] = []
        for cs in self._detected_pending():
            if cs.channel == exclude_channel or cs.status == STATUS_CLEARED:
                continue
            if cs.feasible is None or cs.radius is None:
                continue
            if cs.radius <= self.cfg.safe_region_radius or cs.grid_mode:
                continue
            if self._max_vertex_distance(point, cs.feasible) > 999.0:
                continue
            angle = self._estimated_intersection_angle_deg(point, cs)
            if angle < 20.0:
                continue
            candidates.append((-angle, cs))

        candidates.sort(key=lambda item: (item[0], item[1].channel))
        for _, cs in candidates[:2]:
            if not self._time_left() or cs.status == STATUS_CLEARED:
                break
            res = self._measure(float(point[0]), float(point[1]), cs.channel,
                                phase="SHARED", cs=cs,
                                note="shared measure at same stop")
            self.stats["shared_measures"] += 1
            if res.result == "near":
                self._update_near(cs, point)
                clear_res = self._clear(float(point[0]), float(point[1]), cs.channel,
                                        phase="SHARED", cs=cs,
                                        note="shared near")
                if clear_res.cleared:
                    self._mark_cleared(cs)
            elif res.result == "direction":
                self._update_direction(cs, point, float(res.svd_deg))
            else:
                # 理论上 max vertex <=999 时不会无信号；若模拟器返回了，仍做保守更新。
                self._update_no_signal(cs, point)

    def _build_q2_probe_plan(self, cs: ChannelState,
                             robot_pos: np.ndarray) -> List[np.ndarray]:
        """按问题2思想生成动态拉偏补测点：侧向拉开，形成大交会角。"""
        if cs.first_direction is None:
            return []
        S, u = cs.first_direction
        v = perpendicular_left(u)
        r_est = 600.0
        if cs.center is not None:
            r_est = float(np.linalg.norm(np.asarray(cs.center, dtype=float) - S))
        r_est = float(np.clip(r_est, PROBE_R_EST_MIN, PROBE_R_EST_MAX))
        side_offset = min(r_est * PROBE_SIDE_RATIO, PROBE_SIDE_MAX)
        forward_offset = r_est * PROBE_FORWARD_RATIO
        q2_points = [
            S + forward_offset * u + side_offset * v,
            S + forward_offset * u - side_offset * v,
        ]
        backup_points = [
            S + PROBE_BACKUP_FORWARD * u + PROBE_BACKUP_SIDE * v,
            S + PROBE_BACKUP_FORWARD * u - PROBE_BACKUP_SIDE * v,
        ]
        far_backup_points = [
            S + PROBE_FAR_BACKUP_FORWARD * u + PROBE_FAR_BACKUP_SIDE * v,
            S + PROBE_FAR_BACKUP_FORWARD * u - PROBE_FAR_BACKUP_SIDE * v,
        ]
        q2_points.sort(
            key=lambda point: math.hypot(point[0] - robot_pos[0], point[1] - robot_pos[1])
        )
        backup_points.sort(
            key=lambda point: math.hypot(point[0] - robot_pos[0], point[1] - robot_pos[1])
        )
        far_backup_points.sort(
            key=lambda point: math.hypot(point[0] - robot_pos[0], point[1] - robot_pos[1])
        )
        points = q2_points + backup_points + far_backup_points
        return [np.asarray(point, dtype=float) for point in points]

    def _peek_probe(self, cs: ChannelState,
                    robot_pos: np.ndarray) -> Optional[Tuple[np.ndarray, str]]:
        if cs.first_direction is None:
            return None

        if not cs.probe_plan and cs.probe_count < 6:
            cs.probe_plan = self._build_q2_probe_plan(cs, robot_pos)

        if cs.probe_plan:
            kind = "q2_candidate" if cs.probe_count < 2 else "q2_backup"
            return np.asarray(cs.probe_plan[0], dtype=float), kind

        if cs.enclosing is None or cs.radius is None or cs.feasible is None:
            return None
        C = cs.center
        R = cs.radius
        e, _ = farthest_pair(cs.feasible)
        n = perpendicular_left(e)
        h = max(40.0, min(250.0, 950.0 - R))
        sign = 1.0 if cs.probe_count % 2 == 0 else -1.0
        point = C + sign * h * n
        if R + h > MIN_RECEIVE_RADIUS:
            point = C.copy()
        if cs.last_probe is not None and \
                math.hypot(point[0] - cs.last_probe[0],
                           point[1] - cs.last_probe[1]) < 1e-6:
            return None
        return np.asarray(point, dtype=float), "adaptive"

    def _next_probe(self, cs: ChannelState,
                    robot_pos: np.ndarray) -> Optional[Tuple[np.ndarray, str]]:
        """给出该频道下一个补测点；返回 (point, kind) 或 None（应转网格）。"""
        if cs.first_direction is None:
            return None

        # 第一组：问题2的 q2 候选瓣优先；远距离目标再使用接收保证备用点
        if not cs.probe_plan and cs.probe_count < 4:
            cs.probe_plan = self._build_q2_probe_plan(cs, robot_pos)

        if cs.probe_plan:
            point = cs.probe_plan.pop(0)
            if cs.last_probe is not None and \
                    math.hypot(point[0] - cs.last_probe[0],
                               point[1] - cs.last_probe[1]) < 1e-6:
                return None
            kind = "q2_candidate" if cs.probe_count < 2 else "q2_backup"
            return point, kind

        # 自适应横向补测：沿可行域最长轴的垂直方向、在最小包围圆圆心两侧交替
        if cs.enclosing is None or cs.radius is None or cs.feasible is None:
            return None
        C = cs.center
        R = cs.radius
        e, _ = farthest_pair(cs.feasible)
        n = perpendicular_left(e)
        h = max(40.0, min(250.0, 950.0 - R))
        sign = 1.0 if cs.probe_count % 2 == 0 else -1.0
        point = C + sign * h * n
        if R + h > MIN_RECEIVE_RADIUS:
            point = C.copy()
        if cs.last_probe is not None and \
                math.hypot(point[0] - cs.last_probe[0],
                           point[1] - cs.last_probe[1]) < 1e-6:
            return None
        return np.asarray(point, dtype=float), "adaptive"

    def _execute_probe(self, cs: ChannelState, point: np.ndarray, kind: str) -> None:
        R_before = cs.radius if cs.radius is not None else float("inf")
        channel = cs.channel
        res = self._measure(float(point[0]), float(point[1]), channel,
                            phase="LOCALIZE", cs=cs,
                            note=f"probe#{cs.probe_count + 1} {kind}")
        self.stats["probes"] += 1
        cs.probe_count += 1
        cs.last_probe = np.asarray(point, dtype=float)

        if res.result == "near":
            cs.status = STATUS_DETECTED
            self._update_near(cs, point)
            clear_res = self._clear(float(point[0]), float(point[1]), channel,
                                    phase="LOCALIZE", cs=cs, note="near probe")
            if clear_res.cleared:
                cs.status = STATUS_CLEARED
                self.cleared_count += 1
            return

        if res.result == "direction":
            cs.status = STATUS_DETECTED
            self._update_direction(cs, point, float(res.svd_deg))
        else:
            self._update_no_signal(cs, point)

        R_after = cs.radius if cs.radius is not None else float("inf")
        if R_after < R_before - 0.5:
            cs.stagnation = 0
        else:
            cs.stagnation += 1
        if cs.stagnation >= PROBE_STAGNATION_LIMIT:
            cs.grid_mode = True

   
    def _clear_target(self, cs: ChannelState) -> np.ndarray:
        if cs.enclosing is None:
            self._recompute_enclosing(cs)
        if cs.enclosing is None:
            return np.asarray(self.robot.current_position, dtype=float)
        if cs.radius is not None and cs.radius <= self.cfg.safe_region_radius:
            return cs.center.copy()
        if not cs.grid_mode:
            cs.grid_mode = True
            self.stats["grid_channels"] += 1
        grid = getattr(cs, "grid_plan", None)
        if not grid:
            grid = grid_cover_points(cs.feasible, GRID_SPACING)
            cs.grid_plan = self._order_grid(grid)  # type: ignore[attr-defined]
        grid = getattr(cs, "grid_plan", None)
        if grid:
            return np.asarray(grid[0], dtype=float)
        return np.asarray(self.robot.current_position, dtype=float)

    def _clear_step(self, cs: ChannelState) -> bool:
        if cs.status == STATUS_CLEARED:
            return False
        if cs.enclosing is None or cs.radius is None:
            return self._grid_clear_step(cs)

        if cs.radius <= CLEAR_RADIUS:
            if self._single_clear(cs):
                return True
            self._update_clear_failure(cs, cs.center)
            return self._grid_clear_step(cs)

        if cs.radius <= self.cfg.safe_region_radius:
            return self._two_clear(cs)

        return self._grid_clear_step(cs)

    def _order_grid(self, grid: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """蛇形网格按“离机器人较近的一端”起步。"""
        if not grid:
            return []
        start = np.asarray(self.robot.current_position, dtype=float)
        first = np.asarray(grid[0], dtype=float)
        last = np.asarray(grid[-1], dtype=float)
        d_first = math.hypot(first[0] - start[0], first[1] - start[1])
        d_last = math.hypot(last[0] - start[0], last[1] - start[1])
        return grid if d_first <= d_last else grid[::-1]

    def _clear_channel(self, cs: ChannelState) -> bool:
        if cs.status == STATUS_CLEARED:
            return True
        if cs.enclosing is None or cs.radius is None:
            return self._grid_clear(cs)

        if cs.radius <= CLEAR_RADIUS:
            if self._single_clear(cs):
                return True
            self._update_clear_failure(cs, cs.center)
            return self._grid_clear(cs)

        if cs.radius <= self.cfg.safe_region_radius:
            return self._two_clear(cs)

        return self._grid_clear(cs)

    def _mark_cleared(self, cs: ChannelState) -> None:
        if cs.status != STATUS_CLEARED:
            cs.status = STATUS_CLEARED
            self.cleared_count += 1

    def _single_clear(self, cs: ChannelState) -> bool:
        C = cs.center
        res = self._clear(float(C[0]), float(C[1]), cs.channel,
                          phase="CLEAR", cs=cs, note="single clear at center")
        if res.cleared:
            self._mark_cleared(cs)
            return True
        return False

    def _two_clear(self, cs: ChannelState) -> bool:
        """R ≤ 58：先在最小包围圆圆心清除，失败则按测向偏移后再清一次。"""
        C = cs.center.copy()
        R = float(cs.radius)

        first = self._clear(float(C[0]), float(C[1]), cs.channel,
                            phase="CLEAR", cs=cs, note=f"two-clear step1 R={R:.2f}")
        if first.cleared:
            self._mark_cleared(cs)
            return True

        self._update_clear_failure(cs, C)

        m = self._measure(float(C[0]), float(C[1]), cs.channel,
                          phase="CLEAR", cs=cs, note="two-clear bearing at center")
        if m.result == "near":
            res = self._clear(float(C[0]), float(C[1]), cs.channel,
                              phase="CLEAR", cs=cs, note="two-clear near at center")
            if res.cleared:
                self._mark_cleared(cs)
                return True
        elif m.result == "direction":
            u = unit_from_deg(float(m.svd_deg))
            L = (CLEAR_RADIUS + R) / (2.0 * math.cos(math.radians(self.cfg.bearing_error_deg)))
            Q = C + L * u
            res = self._clear(float(Q[0]), float(Q[1]), cs.channel,
                              phase="CLEAR", cs=cs,
                              note=f"two-clear step2 L={L:.2f}")
            if res.cleared:
                self._mark_cleared(cs)
                self.stats["two_clear_used"] += 1
                return True
            self._update_clear_failure(cs, Q)
        # 理论必然成功的第二次清除失败，或测向退化：转网格兜底
        return self._grid_clear(cs)

    def _grid_clear(self, cs: ChannelState) -> bool:
        """20 m 清除圆覆盖剩余可行域，作为确定性兜底。"""
        if cs.feasible is None:
            return False
        grid = getattr(cs, "grid_plan", None)
        if not grid:
            grid = self._order_grid(grid_cover_points(cs.feasible, GRID_SPACING))
            cs.grid_plan = grid  # type: ignore[attr-defined]
        for point in grid:
            if not self._time_left():
                return False
            res = self._clear(float(point[0]), float(point[1]), cs.channel,
                              phase="GRID", cs=cs, note="grid cover")
            self.stats["grid_clears"] += 1
            if res.cleared:
                self._mark_cleared(cs)
                return True
            self._update_clear_failure(cs, np.asarray(point, dtype=float))
        return False

    def _grid_clear_step(self, cs: ChannelState) -> bool:
        """网格兜底的轮转版：每次只清一个网格点，避免单频道长时间霸占。"""
        if cs.feasible is None or not self._time_left():
            return False
        grid = getattr(cs, "grid_plan", None)
        if not grid:
            grid = self._order_grid(grid_cover_points(cs.feasible, GRID_SPACING))
            cs.grid_plan = grid  # type: ignore[attr-defined]
        if not grid:
            return False

        point = grid.pop(0)
        res = self._clear(float(point[0]), float(point[1]), cs.channel,
                          phase="GRID", cs=cs, note="grid cover step")
        self.stats["grid_clears"] += 1
        if res.cleared:
            self._mark_cleared(cs)
            return True
        self._update_clear_failure(cs, np.asarray(point, dtype=float))
        return True

    # ---------------------------------------------------------------- 统计
    def _summary(self) -> Dict[str, object]:
        detected = sum(1 for cs in self.channels.values() if cs.status in (STATUS_DETECTED, STATUS_CLEARED))
        cleared = sum(1 for cs in self.channels.values() if cs.status == STATUS_CLEARED)
        empty = sum(1 for cs in self.channels.values() if cs.status == STATUS_EMPTY)
        unknown = sum(1 for cs in self.channels.values() if cs.status == STATUS_UNKNOWN)

        total_virtual = float(self.robot.virtual_time_s)
        # 理论耗时分解（用于交叉核对）
        action_time = self.stats["measures"] * MEASURE_TIME_S \
            + self.stats["switch_count"] * SWITCH_TIME_S \
            + self.stats["clear_success"] * CLEAR_SUCCESS_TIME_S \
            + (self.stats["clears"] - self.stats["clear_success"]) * CLEAR_FAIL_TIME_S
        move_time = self.stats["move_distance"] / MOVE_SPEED_MPS

        return {
            "channels_total": len(self.channels),
            "detected": detected,
            "cleared": cleared,
            "empty_certified": empty,
            "unknown": unknown,
            "all_resolved": cleared + empty == len(self.channels) and unknown == 0,
            "virtual_time_s": total_virtual,
            "virtual_time_min": total_virtual / 60.0,
            "move_time_s": move_time,
            "action_time_s": action_time,
            "avg_clear_time_s": (total_virtual / cleared) if cleared else None,
            "stats": dict(self.stats),
        }


# --------------------------------------------------------------------------- #
# 覆盖地图策略（CoverageStrategy）
#
# 从 problem3/q3_practice_single.py 内嵌的 coverage_strategy 模块移植而来。
# 与基础 Problem3Strategy 的固定“七点巡检 + 目标调度”相比，这里的结构性改进是：
#
# 1. 实际覆盖地图：每个频道单独保存“目标圆域外包多边形 − 已检测接收圆内接多边形”的
#    剩余区域，只在剩余几何集合真正为空（或已确认 16 个频道有源）时才判空，
#    不再用“七点都被扫过”这种对具体路径敏感的判据；
# 2. 联合行程规划：把已发现目标的补测点、清除点与动态补盲任务放进同一个任务池，
#    做最近邻 + 2-opt 开路排序，未来覆盖只用于规划、执行一次动作后立即重算；
# 3. 同点共享定位：一个停靠点顺带为其他已发现频道补测，避免反复回到各自首个测点；
# 4. 接近后侧移：先朝估计目标靠近再取侧向视差，替代一开始就大幅横移；
# 5. 补盲位置区域：对剩余连通区域求包围圆，构造“任意位置都能覆盖它”的检测区域，
#    再选贴近既有路线的位置，而不是绑定到固定网格点；
# 6. 选择扫描：普通停靠扫描只在高增益时才顺带测，强制补盲不受该门槛限制。
#
# 本类保留了完整的失败显式化：时间不足、可行域退化、有限网格用尽都会直接抛错，
# 而不是悄悄返回一个未完成的“成功”结果。
# --------------------------------------------------------------------------- #
def reception(point):
    """检测点(point)的实际接收区域：半径略小于 1000 m 的内接多边形。

    Shapely buffer 的顶点落在 999.99 m 圆上，因此用它做差集永远不会越界
    排除真实干扰源（真实接收半径 ≥ 1000 m）。
    """
    return Point(float(point[0]), float(point[1])).buffer(999.99, quad_segs=32)


class ChannelCoverage:
    """单个频道仍未被任何检测覆盖的剩余区域（实际覆盖地图）。"""

    def __init__(self):
        self.domain = Polygon(circumscribed_polygon((0, 0), 1800.00001, n=256))
        self.remaining = {ch: self.domain for ch in range(CHANNEL_MIN, CHANNEL_MAX + 1)}
        self.observations = {ch: [] for ch in range(CHANNEL_MIN, CHANNEL_MAX + 1)}

    def record(self, channel, point, result):
        self.observations[channel].append((tuple(map(float, point)), result))
        # 即使检测到了信号，这次检测的实际覆盖范围也已经确定；该频道后续交给
        # 目标定位状态机处理，不再依赖覆盖地图判空。
        self.remaining[channel] = self.remaining[channel].difference(reception(point))

    def gain(self, channel, point):
        return self.remaining[channel].intersection(reception(point)).area

    def complete(self, channel):
        # 绝不用“剩余面积很小”当判空证书。
        return self.remaining[channel].is_empty


@dataclass
class MapConfig:
    """覆盖地图策略的运行档位与阈值。"""

    mode: str = "route_joint_regions_selective_approach"
    scan_gain: float = 350000.0
    max_probes: int = 10
    max_actions: int = 1200
    side: float = 100.0


@dataclass
class Task:
    """覆盖地图策略的单个待执行任务（search / probe / clear / grid）。"""

    kind: str
    point: np.ndarray
    channel: int = 0


class CoverageStrategy(Problem3Strategy):
    """基于实际覆盖地图 + 联合行程规划的滚动策略。

    用法与 ``Problem3Strategy`` 相同，只是多一个 ``map_config``::

        strategy = CoverageStrategy(robot, StrategyConfig(bearing_error_deg=1.01),
                                    MapConfig(mode="route_joint_regions_selective_approach"))
        summary = strategy.run()
    """

    def __init__(self, robot, config=None, map_config=None):
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                'CoverageStrategy 需要 shapely：python -m pip install "shapely>=2.0"'
            )
        super().__init__(robot, config or StrategyConfig(bearing_error_deg=1.01))
        self.mc = map_config or MapConfig()
        self.map = ChannelCoverage()
        self.used = {ch: [] for ch in self.channels}
        self.decisions = []
        self.search_stops = 0
        self.piggyback_measures = 0
        self.piggyback_area = 0.0
        self._grid_cache = {}
        # 候选扫描点：四个半径 × 24 个方向，用于快速挑“覆盖收益最大的补盲位置”。
        self.pool = [np.array([r * math.cos(a), r * math.sin(a)])
                     for r in (650, 1000, 1250, 1500) for a in np.arange(24) * math.pi / 12]
        self.pool_disks = [reception(p) for p in self.pool]

    # -------------------------------------------------------- 动作包装与预算
    def _measure(self, x, y, channel, phase, cs=None, note=''):
        self._check_action_budget(x, y, 6)
        res = super()._measure(x, y, channel, phase, cs, note)
        self.map.record(channel, (x, y), res.result)
        self.used[channel].append(np.array([x, y]))
        return res

    def _clear(self, x, y, channel, phase, cs=None, note=''):
        self._check_action_budget(x, y, 5)
        return super()._clear(x, y, channel, phase, cs, note)

    def _check_action_budget(self, x, y, action_s):
        if not self._time_left():
            raise RuntimeError('Insufficient real or virtual time')
        limit = getattr(self.robot, 'max_virtual_duration_s', 360000)
        if limit is not None and \
                self.robot.virtual_time_s + self.robot.estimate_move_time(x, y) + action_s >= limit:
            raise RuntimeError('Next action exceeds virtual-time budget')

    def _recompute_enclosing(self, cs):
        # 不丢弃“很小但仍合法”的可行域；同时独立地把半径放大到包含每个多边形的顶点，
        # 这样即使最小包围圆有舍入误差，也不会漏掉可行域边界上的点。
        if cs.feasible is None or len(cs.feasible) == 0:
            raise RuntimeError('Empty feasible region; completion cannot be certified')
        c, r = min_enclosing_circle(cs.feasible)
        r = max(r, float(np.max(np.linalg.norm(cs.feasible - c, axis=1)))) + 1e-6
        cs.enclosing = (c, r)

    # ------------------------------------------------------------ 判空与扫描
    def certify(self):
        upper = self._known_source_count() == SOURCE_COUNT_MAX
        for ch, cs in self.channels.items():
            if cs.status == STATUS_UNKNOWN and (upper or self.map.complete(ch)):
                self._mark_empty(
                    cs,
                    '%d known sources' % SOURCE_COUNT_MAX if upper
                    else 'actual per-channel coverage complete',
                )

    def unknown(self):
        return [ch for ch, s in self.channels.items() if s.status == STATUS_UNKNOWN]

    def sense(self, ch, p, phase):
        """对一个频道做一次检测并按结果更新状态机。"""
        cs = self.channels[ch]
        res = self._measure(*map(float, p), ch, phase, cs)
        if res.result == 'direction':
            cs.status = STATUS_DETECTED
            self._update_direction(cs, p, float(res.svd_deg))
        elif res.result == 'near':
            cs.status = STATUS_DETECTED
            self._update_near(cs, p)
            if not self._single_clear(cs):
                raise RuntimeError('near clear failed')
        elif res.result == 'no_signal':
            self._update_no_signal(cs, p)
        else:
            raise RuntimeError('Invalid measurement result')
        self.certify()
        return res

    def scan(self, p, force=False):
        """在位置 p 对仍未知的频道做一轮扫描。

        ``force=True`` 是计划内的补盲扫描（每个增益 > 0 的频道都测）；
        ``force=False`` 是顺路搭车（只在增益达到门槛时才测）。
        """
        current = self.robot.current_channel
        # 优先使用当前所在频道，省一次 1 秒切频。
        order = sorted(self.unknown(), key=lambda ch: (ch != current, ch))
        changed = False
        for ch in order:
            if self.channels[ch].status != STATUS_UNKNOWN:
                continue
            if not self._time_left():
                raise RuntimeError('Time exhausted during scan')
            gain = self.map.gain(ch, p)
            threshold = 1000000.0 if 'selective' in self.mc.mode else self.mc.scan_gain
            if gain <= 0 or (not force and gain < threshold):
                continue
            self.sense(ch, p, 'SEARCH' if force else 'PIGGYBACK')
            if not force:
                self.piggyback_measures += 1
                self.piggyback_area += gain
            changed = True
        if changed and force:
            self.search_stops += 1
        return changed

    # ------------------------------------------------------------ 任务与路由
    def remaining_shape(self):
        return unary_union([self.map.remaining[ch] for ch in self.unknown()])

    def search_point(self, need):
        """在候选池里选覆盖收益最大的补盲位置（并列时取离当前位置最近的）。"""
        if need.is_empty:
            return None
        pos = np.array(self.robot.current_position)
        gains = np.array([need.intersection(d).area for d in self.pool_disks])
        best = float(gains.max())
        if best > 1e-8:
            ids = np.flatnonzero(gains >= (0.99 if 'joint' in self.mc.mode else 0.65) * best)
            idx = min(ids, key=lambda j: float(np.linalg.norm(self.pool[j] - pos)))
            return self.pool[int(idx)].copy()
        # 处理任意细小的盲区条带，同时不会错误地宣布“已覆盖完成”。
        p = need.representative_point()
        return np.array([p.x, p.y])

    def local_point(self, cs):
        """已发现目标的下一个补测点：先靠近估计位置，再取侧向视差。"""
        pos = np.array(self.robot.current_position)
        if self.mc.mode.endswith('approach') or self.mc.mode in ('nn_approach', 'hex_online'):
            # 先朝估计目标靠近再取侧向视差：小的横向分量比一开始就大幅绕行更省时间。
            c = cs.center
            vec = c - pos
            d = np.linalg.norm(vec)
            u = vec / d if d > 1e-6 else cs.first_direction[1]
            n = np.array([-u[1], u[0]])
            h = min(self.mc.side, max(30, cs.radius * 0.2))
            candidates = [c + h * n, c - h * n]
            if cs.probe_count >= 3:
                candidates += self._build_q2_probe_plan(cs, pos)
        else:
            candidates = self._build_q2_probe_plan(cs, pos)
        # 不重复使用该频道已经测过的位置（否则拿不到新的交会角）。
        candidates = [p for p in candidates
                      if all(np.linalg.norm(p - q) > 1 for q in self.used[cs.channel])]
        if not candidates:
            return None
        return min(candidates, key=lambda p: float(np.linalg.norm(p - pos)))

    def target_tasks(self):
        """所有已发现目标的清除/补测/网格任务。"""
        tasks = []
        for ch, cs in self.channels.items():
            if cs.status != STATUS_DETECTED:
                continue
            if cs.radius is None:
                raise RuntimeError('Detected source has no enclosing region')
            if cs.radius <= self.cfg.safe_region_radius:
                tasks.append(Task('clear', cs.center.copy(), ch))
            elif cs.probe_count < self.mc.max_probes:
                p = self.local_point(cs)
                tasks.append(Task('probe', p, ch) if p is not None
                             else Task('grid', cs.center.copy(), ch))
            else:
                tasks.append(Task('grid', cs.center.copy(), ch))
        return tasks

    def planned_searches(self, tasks):
        """把“未来还要补的盲区”折算成若干顺路补盲任务。

        注意：这里只是规划用的启发式估计，不是地图更新、更不能当判空证书；
        真正执行一次动作后预测全部丢弃并重算。
        """
        need = self.remaining_shape()
        if self.mc.mode.startswith('route'):
            for t in tasks:
                need = need.difference(reception(t.point))
                if t.kind == 'probe':
                    need = need.difference(reception(self.channels[t.channel].center))
        searches = []
        for _ in range(12):
            if need.is_empty:
                break
            p = (self.region_search_point(need, tasks + searches)
                 if 'regions' in self.mc.mode else self.search_point(need))
            searches.append(Task('search', p))
            updated = need.difference(reception(p))
            if updated.equals(need):
                raise RuntimeError('No coverage progress')
            need = updated
        return searches

    def region_search_point(self, need, tasks):
        """把一整块残余盲区当成一个整体来选择检测位置区域，再挑贴近既有路线的点。

        这改变的是补盲任务本身，而不只是任务之间的访问顺序。
        """
        pieces = list(need.geoms) if hasattr(need, 'geoms') else [need]
        piece = max(pieces, key=lambda g: g.area)
        if 'batch' in self.mc.mode:
            # 几块不相连的盲区可能被同一个接收圆一并覆盖：先尝试合并。
            for other in sorted(pieces, key=lambda g: g.distance(piece)):
                if other.equals(piece):
                    continue
                combined = piece.union(other)
                h = combined.convex_hull
                if h.geom_type != 'Polygon':
                    continue
                vv = np.array(h.exterior.coords)[:-1]
                cc, rr = min_enclosing_circle(vv)
                rr = max(rr, float(np.max(np.linalg.norm(vv - cc, axis=1))))
                if rr < 995:
                    piece = combined
        hull = piece.convex_hull
        if hull.geom_type != 'Polygon':
            return self.search_point(need)
        vertices = np.array(hull.exterior.coords)[:-1]
        c, r = min_enclosing_circle(vertices)
        r = max(r, float(np.max(np.linalg.norm(vertices - c, axis=1)))) + 0.001
        # 内接接收多边形的内切半径约 999.689 m，这里留出余量。
        # 若盲区被 B(C, R) 包含且 R < 995，则 B(C, 999.5-R) 内任意检测位置都能覆盖它。
        if r < 995:
            allowed = Point(*c).buffer(999.5 - r, quad_segs=24)
            pts = [np.asarray(self.robot.current_position)] + [t.point for t in tasks]
            if len(pts) > 2:
                order = nearest_neighbor_route(pts[0], pts[1:])
                order = two_opt_open(order, pts[0], pts[1:])
                pts = [pts[0]] + [pts[j + 1] for j in order]
            path = LineString(pts) if len(pts) > 1 else Point(*pts[0])
            q = nearest_points(allowed, path)[0]
            return np.array([q.x, q.y])
        return self.search_point(need)

    def choose(self):
        """选出下一个要执行的任务：任务池排序后取第一个。"""
        tasks = self.target_tasks()
        if self.mc.mode.startswith('route'):
            tasks += self.planned_searches(tasks)
            if not tasks:
                return None
            points = [t.point for t in tasks]
            route = nearest_neighbor_route(self.robot.current_position, points)
            route = two_opt_open(route, self.robot.current_position, points)
            return tasks[route[0]]
        p = self.search_point(self.remaining_shape()) if self.unknown() else None
        if p is not None:
            tasks.append(Task('search', p))
        if not tasks:
            return None
        pos = np.array(self.robot.current_position)

        def cost(t):
            action = (5 * len(self.unknown()) + max(0, len(self.unknown()) - 1)
                      if t.kind == 'search' else 6)
            return np.linalg.norm(t.point - pos) / 5 + action

        return min(tasks, key=cost)

    def execute(self, t):
        """执行一个任务，并在结束后做顺路搭车扫描与共享定位。"""
        self.decisions.append(dict(kind=t.kind, point=t.point.tolist(), channel=t.channel,
                                   time_s=self.robot.virtual_time_s,
                                   unknown=len(self.unknown())))
        if t.kind == 'search':
            if not self.scan(t.point, force=True):
                raise RuntimeError('Search action made no progress')
        elif t.kind == 'probe':
            cs = self.channels[t.channel]
            cs.probe_count += 1
            self.stats['probes'] += 1
            self.sense(t.channel, t.point, 'LOCALIZE')
        elif t.kind == 'clear':
            if not self._clear_channel(self.channels[t.channel]):
                raise RuntimeError('Clear failed')
        else:
            # 网格计划只生成一次：父类的“用尽后自动重生成”会让失败变成死循环，
            # 这里让失败显式抛错。
            if not self._grid_clear(self.channels[t.channel]):
                raise RuntimeError('Finite grid exhausted')
        if self.mc.mode not in ('nn_separate',):
            self.scan(np.array(self.robot.current_position), force=False)
        if 'joint' in self.mc.mode:
            self.shared_localize(np.array(self.robot.current_position))
        self.certify()

    def shared_localize(self, p):
        """一个停靠点顺带为其他已发现频道补测，避免反复回到各自首个（过时的）测点区域。"""
        for ch, cs in self.channels.items():
            if cs.status != STATUS_DETECTED or cs.radius is None \
                    or cs.radius <= self.cfg.safe_region_radius:
                continue
            if any(np.linalg.norm(p - q) < 60 for q in self.used[ch]):
                continue
            if np.linalg.norm(p - cs.center) > 1200:
                continue
            if self._estimated_intersection_angle_deg(p, cs) < 12:
                continue
            self.sense(ch, p, 'SHARED_LOCALIZE')
            self.stats['shared_measures'] += 1

    def run_hex(self):
        """对照基线：原点全扫，然后每到一个巡检站之前先把已发现目标处理完。

        这是用于消融对比的重建版本，不代表任何缺失的“边扫边清六边形”实现。
        """
        ring = build_survey_points()[1:]
        self.scan(np.array([0., 0.]), True)
        idx = 0
        while True:
            tasks = self.target_tasks()
            if tasks:
                p = np.array(self.robot.current_position)
                t = min(tasks, key=lambda task: np.linalg.norm(task.point - p))
                self.execute(t)
            elif idx < len(ring) and self.unknown():
                p = np.array(self.robot.current_position)
                j = min(range(idx, len(ring)), key=lambda k: np.linalg.norm(ring[k] - p))
                ring[idx], ring[j] = ring[j], ring[idx]
                self.scan(ring[idx], True)
                idx += 1
            else:
                break

    # -------------------------------------------------------------- 运行入口
    def run(self):
        self._enter()
        try:
            if self.mc.mode == 'hex_online':
                self.run_hex()
            else:
                self.scan(np.array([0., 0.]), True)
                for _ in range(self.mc.max_actions):
                    if not self._time_left():
                        raise RuntimeError('Time limit; no completion certificate')
                    t = self.choose()
                    if t is None:
                        break
                    self.execute(t)
                else:
                    raise RuntimeError('Action budget exhausted')
            self.certify()
            result = self._summary()
            if not result['all_resolved'] or not SOURCE_COUNT_MIN <= result['cleared'] <= SOURCE_COUNT_MAX:
                raise RuntimeError('Incomplete search or uncleared target')
            result.update(search_stops=self.search_stops,
                          piggyback_measures=self.piggyback_measures,
                          piggyback_area_m2=self.piggyback_area,
                          policy=self.mc.mode)
            return result
        finally:
            self._exit()
