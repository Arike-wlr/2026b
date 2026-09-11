"""问题 3 本地规则仿真器 + 批量验证。

`LocalSimulator` 复刻官方模拟器的关键行为（与 ``client.py`` 的语义一致），
对外暴露与 ``RobotClient`` **完全相同的方法/属性**，因此 ``Problem3Strategy``
可以不加修改地在“真实接口”和“本地仿真”之间切换。

仿真假设（与规格第 14 节一致）：

* 干扰源数量在 10–16 均匀生成；
* 位置在半径 1800 m 圆域内按面积均匀、相互独立；
* 有效接收半径在 1000–1500 m 均匀；
* 示向度误差在 [-1°, 1°]，**同一位置（同一频道）误差固定**；
* 频道从 1–20 无放回抽取。

用法::

    python problem3_local_sim.py                 # 默认 100 个案例
    python problem3_local_sim.py --cases 200 --seed 42
    python problem3_local_sim.py --cases 1 --verbose
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

from problem3_strategy import (  # noqa: E402
    BEARING_ERROR_DEG,
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
    Problem3Strategy,
    StrategyConfig,
)

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


# --------------------------------------------------------------------------- #
# 单案例 / 批量运行
# --------------------------------------------------------------------------- #
def run_case(seed: int, verbose: bool = False,
             log_path: Optional[str] = None,
             num_sources: Optional[int] = None) -> Dict[str, object]:
    sim = LocalSimulator(seed, num_sources=num_sources)
    cfg = StrategyConfig(verbose=verbose, log_path=log_path)
    strategy = Problem3Strategy(sim, cfg)
    summary = strategy.run()

    total = sim.total_sources
    cleared = sim.cleared_count
    summary.update({
        "seed": seed,
        "source_total": total,
        "source_cleared": cleared,
        "success": cleared == total,
        "clear_ratio": cleared / total if total else 0.0,
    })
    return summary


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values), q))


def run_batch(cases: int, seed: int, verbose_every: int = 0,
              log_dir: Optional[str] = None) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    for i in range(cases):
        case_log = None
        if log_dir and (i == 0):
            case_log = os.path.join(log_dir, f"problem3_local_seed{seed}.txt")
        res = run_case(seed + i, verbose=False, log_path=case_log)
        results.append(res)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  已完成 {i + 1}/{cases} 例")

    successes = [r for r in results if r["success"]]
    times = [float(r["virtual_time_min"]) for r in results]
    avg_clear = [float(r["avg_clear_time_s"]) for r in successes]
    source_counts = [int(r["source_total"]) for r in results]

    def mean(key: str) -> float:
        return float(np.mean([float(r["stats"][key]) for r in results]))  # type: ignore[index]

    report = {
        "cases": cases,
        "success_case": len(successes),
        "success_rate": len(successes) / cases if cases else 0.0,
        "avg_source_total": float(np.mean(source_counts)),
        "avg_virtual_min": float(np.mean(times)),
        "median_virtual_min": float(statistics.median(times)),
        "min_virtual_min": float(np.min(times)),
        "max_virtual_min": float(np.max(times)),
        "p90_virtual_min": _percentile(times, 90),
        "avg_clear_time_s": float(np.mean(avg_clear)),
        "avg_move_min": float(np.mean([float(r["move_time_s"]) / 60.0 for r in results])),
        "avg_action_min": float(np.mean([float(r["action_time_s"]) / 60.0 for r in results])),
        "avg_measures": mean("measures"),
        "avg_switches": mean("switch_count"),
        "avg_probes": mean("probes"),
        "avg_grid_clears": mean("grid_clears"),
        "avg_shared_measures": mean("shared_measures"),
        "avg_opportunistic": mean("opportunistic_strikes"),
        "avg_empty_pruned": mean("empty_pruned"),
        "avg_survey_skipped_empty": mean("survey_skipped_empty"),
        "avg_clipped_to_clear": mean("clipped_to_clear"),
        "results": results,
    }
    return report


def write_report(report: Dict[str, object], path: str) -> None:
    lines = [
        "# 问题3 本地仿真批量验证报告",
        "",
        "> 由 `problem3_local_sim.py` 依据公开规则生成，用于验证策略正确性与量级；",
        "> 不等同于官方模拟器成绩，正式提交前须在官方“问题3 演练测试”中复验。",
        "",
        "## 总体指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 案例数 | {report['cases']} |",
        f"| 成功案例 | {report['success_case']} / {report['cases']} "
        f"（{float(report['success_rate']):.1%}） |",
        f"| 平均干扰源数 | {float(report['avg_source_total']):.2f} |",
        f"| 平均总虚拟时间 | {float(report['avg_virtual_min']):.2f} min |",
        f"| 中位总虚拟时间 | {float(report['median_virtual_min']):.2f} min |",
        f"| 最短总虚拟时间 | {float(report['min_virtual_min']):.2f} min |",
        f"| 最长总虚拟时间 | {float(report['max_virtual_min']):.2f} min |",
        f"| P90 总虚拟时间 | {float(report['p90_virtual_min']):.2f} min |",
        f"| 平均单源定位清除时间 | {float(report['avg_clear_time_s']):.2f} s |",
        "",
        "## 耗时分解（平均每案例）",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        f"| 移动时间 | {float(report['avg_move_min']):.2f} min |",
        f"| 检测/切换/清除动作时间 | {float(report['avg_action_min']):.2f} min |",
        f"| 检测次数 | {float(report['avg_measures']):.1f} |",
        f"| 切频次数 | {float(report['avg_switches']):.1f} |",
        f"| 补测次数 | {float(report['avg_probes']):.1f} |",
        f"| 共享测向次数 | {float(report['avg_shared_measures']):.1f} |",
        f"| 顺路清除次数 | {float(report['avg_opportunistic']):.1f} |",
        f"| 空频道剪枝数 | {float(report['avg_empty_pruned']):.1f} |",
        f"| 巡检跳过空/已清频道次数 | {float(report['avg_survey_skipped_empty']):.1f} |",
        f"| 几何裁剪直接入清除队列次数 | {float(report['avg_clipped_to_clear']):.1f} |",
        f"| 网格兜底清除次数 | {float(report['avg_grid_clears']):.1f} |",
        "",
        "## 安全剪枝说明",
        "",
        "几何裁剪直接入清除队列是基于 no_signal 半平面约束的安全剪枝：",
        "当一次无信号检测把该频道可行域裁剪到最小外接圆半径不超过安全半径时，",
        "后续不再继续补测该频道，而是交由全局 TSP 调度进入清除队列。",
        "该判断只收缩已被几何约束证明的可行域，不排除任何仍可能存在干扰源的位置，",
        "因此不损失最优性。",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="问题3 本地仿真批量验证")
    parser.add_argument("--cases", type=int, default=100, help="案例数")
    parser.add_argument("--seed", type=int, default=42, help="起始随机种子")
    parser.add_argument("--verbose", action="store_true", help="打印单案例动作日志")
    parser.add_argument("--no-report", action="store_true", help="不生成报告")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(here, "logs")
    report_path = os.path.join(here, "problem3_batch_report.md")

    if args.cases == 1 and args.verbose:
        summary = run_case(args.seed, verbose=True)
        for key in ("source_total", "source_cleared", "success",
                    "virtual_time_min", "avg_clear_time_s"):
            print(f"{key}: {summary[key]}")
        return

    print(f"运行 {args.cases} 个案例（seed={args.seed} ... {args.seed + args.cases - 1}）")
    report = run_batch(args.cases, args.seed, verbose_every=max(1, args.cases // 10),
                       log_dir=log_dir)
    print(f"成功案例：{report['success_case']}/{report['cases']}"
          f"（{float(report['success_rate']):.1%}）")
    print(f"平均总虚拟时间：{float(report['avg_virtual_min']):.2f} min，"
          f"中位 {float(report['median_virtual_min']):.2f} min，"
          f"最长 {float(report['max_virtual_min']):.2f} min")
    print(f"平均单源定位清除时间：{float(report['avg_clear_time_s']):.2f} s")
    if not args.no_report:
        write_report(report, report_path)
        print(f"报告已写入：{report_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
