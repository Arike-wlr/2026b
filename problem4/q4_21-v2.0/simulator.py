"""Offline simulator extracted from 2026b/problem3/problem3_local_sim.py.
Generation and action semantics unchanged; removed circular strategy import and old CLI.
This is a recovered dependency, not the missing original simulator.py.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TARGET_RADIUS = 1800.0
MIN_RECEIVE_RADIUS = 1000.0
MAX_RECEIVE_RADIUS = 1500.0
NEAR_RADIUS = 5.0
CLEAR_RADIUS = 20.0
BEARING_ERROR_DEG = 1.0
MOVE_SPEED_MPS = 5.0
MEASURE_TIME_S = 5.0
SWITCH_TIME_S = 1.0
CLEAR_FAIL_TIME_S = 3.0
CLEAR_SUCCESS_TIME_S = 5.0
CHANNEL_MIN = 1
CHANNEL_MAX = 20
SOURCE_COUNT_MIN = 10
SOURCE_COUNT_MAX = 16


DEFAULT_VIRTUAL_LIMIT_S = 360000.0
DEFAULT_REAL_BUDGET_S = 1200.0


# --------------------------------------------------------------------------- #
# 结果对象（字段与 client.py 对齐）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MeasureResult:
    result: str                     # direction | near | no_signal
    svd_deg: Optional[float]
    position: Tuple[float, float]
    channel: int
    virtual_time_s: float

    @property
    def has_signal(self) -> bool:
        return self.result == "direction"

    @property
    def is_near(self) -> bool:
        return self.result == "near"


@dataclass(frozen=True)
class ClearResult:
    cleared: bool
    position: Tuple[float, float]
    channel: int
    virtual_time_s: float


@dataclass
class Source:
    channel: int
    position: np.ndarray
    receive_radius: float
    cleared: bool = False


# --------------------------------------------------------------------------- #
# 仿真器
# --------------------------------------------------------------------------- #
class LocalSimulator:
    def __init__(self, seed: int, num_sources: Optional[int] = None,
                 virtual_limit_s: float = DEFAULT_VIRTUAL_LIMIT_S,
                 real_budget_s: float = DEFAULT_REAL_BUDGET_S):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        if num_sources is None:
            num_sources = int(self.rng.integers(SOURCE_COUNT_MIN, SOURCE_COUNT_MAX + 1))
        self.num_sources = num_sources

        channels = self.rng.choice(
            np.arange(1, 21), size=num_sources, replace=False
        )
        self.sources: Dict[int, Source] = {}
        for ch in channels:
            # 圆域内按面积均匀采样
            r = TARGET_RADIUS * math.sqrt(float(self.rng.random()))
            ang = float(self.rng.random()) * 2.0 * math.pi
            pos = np.array([r * math.cos(ang), r * math.sin(ang)])
            receive = float(self.rng.uniform(MIN_RECEIVE_RADIUS, MAX_RECEIVE_RADIUS))
            self.sources[int(ch)] = Source(channel=int(ch), position=pos,
                                           receive_radius=receive)

        # 机器狗状态
        self._in_session = False
        self._position = np.array([0.0, 0.0])
        self._current_channel: Optional[int] = None
        self._virtual_time = 0.0
        self._virtual_limit = virtual_limit_s
        self._real_budget = real_budget_s
        self._error_cache: Dict[str, float] = {}
        self.request_counter = 0

    # ------------------------------------------------------------ 生命周期
    @property
    def in_session(self) -> bool:
        return self._in_session

    def enter(self) -> Dict:
        self._in_session = True
        self._position = np.array([0.0, 0.0])
        self._current_channel = 1
        self._virtual_time = 0.0
        return {
            "accepted": True,
            "virtual_time_s": 0.0,
            "max_virtual_duration_s": self._virtual_limit,
            "max_real_duration_s": self._real_budget,
            "remaining_real_duration_s": self._real_budget,
        }

    def exit(self) -> str:
        self._in_session = False
        return "user_exit"

    # ------------------------------------------------------------ 状态属性
    @property
    def current_position(self) -> Tuple[float, float]:
        return (float(self._position[0]), float(self._position[1]))

    @property
    def current_channel(self) -> Optional[int]:
        return self._current_channel

    @property
    def virtual_time_s(self) -> float:
        return self._virtual_time

    @property
    def max_virtual_duration_s(self) -> float:
        return self._virtual_limit

    @property
    def remaining_real_s(self) -> float:
        return self._real_budget

    def is_time_up(self, margin_s: float = 0.0) -> bool:
        return self._virtual_time + margin_s >= self._virtual_limit

    # ------------------------------------------------------------ 耗时预估
    def estimate_move_time(self, x: float, y: float) -> float:
        return math.hypot(x - self._position[0], y - self._position[1]) / MOVE_SPEED_MPS

    def estimate_measure_cost(self, x: float, y: float, channel: int) -> float:
        switch = 0.0
        if self._current_channel is not None and channel != self._current_channel:
            switch = SWITCH_TIME_S
        return self.estimate_move_time(x, y) + switch + MEASURE_TIME_S

    def estimate_clear_cost(self, x: float, y: float, found: bool = True) -> float:
        action = CLEAR_SUCCESS_TIME_S if found else CLEAR_FAIL_TIME_S
        return self.estimate_move_time(x, y) + action

    # ------------------------------------------------------------ 误差模型
    def _bearing_error(self, channel: int, x: float, y: float) -> float:
        key = f"{channel}|{round(x, 3)}|{round(y, 3)}"
        cached = self._error_cache.get(key)
        if cached is None:
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            u = int.from_bytes(digest[:8], "big") / float(1 << 64)
            cached = (u * 2.0 - 1.0) * BEARING_ERROR_DEG
            self._error_cache[key] = cached
        return cached

    # ------------------------------------------------------------ 动作
    def measure(self, x: float, y: float, channel: int) -> MeasureResult:
        if not self._in_session:
            raise RuntimeError("尚未 enter()")
        self.request_counter += 1
        move = self.estimate_move_time(x, y)
        switch = 0.0
        if self._current_channel is not None and channel != self._current_channel:
            switch = SWITCH_TIME_S
        self._virtual_time += move + switch + MEASURE_TIME_S
        self._position = np.array([float(x), float(y)])
        self._current_channel = int(channel)

        source = self.sources.get(int(channel))
        if source is None or source.cleared:
            result, svd = "no_signal", None
        else:
            d = float(np.hypot(*(source.position - self._position)))
            if d <= NEAR_RADIUS:
                result, svd = "near", None
            elif d <= source.receive_radius:
                true_bearing = math.degrees(
                    math.atan2(source.position[1] - y, source.position[0] - x)
                ) % 360.0
                svd = (true_bearing + self._bearing_error(channel, x, y)) % 360.0
                result = "direction"
            else:
                result, svd = "no_signal", None

        return MeasureResult(result=result, svd_deg=svd,
                             position=(float(x), float(y)), channel=int(channel),
                             virtual_time_s=self._virtual_time)

    def clear(self, x: float, y: float, channel: int) -> ClearResult:
        if not self._in_session:
            raise RuntimeError("尚未 enter()")
        self.request_counter += 1
        move = self.estimate_move_time(x, y)
        self._position = np.array([float(x), float(y)])

        source = self.sources.get(int(channel))
        cleared = False
        if source is not None and not source.cleared:
            d = float(np.hypot(*(source.position - self._position)))
            if d <= CLEAR_RADIUS:
                source.cleared = True
                cleared = True
        self._virtual_time += move + (CLEAR_SUCCESS_TIME_S if cleared else CLEAR_FAIL_TIME_S)
        # /clear 不改变测向机当前频道
        return ClearResult(cleared=cleared, position=(float(x), float(y)),
                           channel=int(channel), virtual_time_s=self._virtual_time)

    # ------------------------------------------------------------ 统计辅助
    @property
    def cleared_count(self) -> int:
        return sum(1 for s in self.sources.values() if s.cleared)

    @property
    def total_sources(self) -> int:
        return len(self.sources)

    def remaining_channels(self) -> List[int]:
        return sorted(ch for ch, s in self.sources.items() if not s.cleared)

