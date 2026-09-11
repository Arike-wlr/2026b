"""研究用策略的**接口适配积木**：离线记录、演练安全门、覆盖地图 policy 分发。

这里只放"把策略接到本仓库仿真器/真实客户端"所需的胶水，**不含任何策略算法**：

* ``RecordingRobot``  —— 把策略的动作原样转发给真身并记录时间线（不改变行为），
  被包装后的对象与 ``client.RobotClient`` 方法签名一致；
* ``action_times``    —— 把时间线拆成移动/检测/切频/清除/尾部补盲；
* ``practice_guard``  —— 建 HTTP 之前的安全门（队号 + 显式确认 + 回环地址 + 人工输入）；
* ``run_coverage_case`` —— 覆盖地图模型（policy=…）或原文件基线（policy=supplied）
  的单案离线运行，供 ``coverage_main.py`` 的 offline 分支使用。

依赖关系：本文件只依赖本仓库的 ``problem3_local_sim`` / ``problem3_strategy``；
覆盖地图模型经 ``q3_practice_v2_local_benchmark.load_external_implementation`` 惰性加载。
"""

from __future__ import annotations

import math
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from problem3_local_sim import LocalSimulator  # noqa: E402
from problem3_strategy import (  # noqa: E402
    CLEAR_FAIL_TIME_S,
    CLEAR_SUCCESS_TIME_S,
    MEASURE_TIME_S,
    MOVE_SPEED_MPS,
    SWITCH_TIME_S,
    Problem3Strategy,
    StrategyConfig,
)

#: 原文件基线在对比表里的策略名
SUPPLIED_POLICY = "supplied"

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class RecordingRobot:
    """把动作原样转发给真身，同时记录时间线（只加记录，不加干预）。"""

    def __init__(self, inner, trace: list[dict[str, Any]]):
        self._inner = inner
        self._trace = trace

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def enter(self):
        result = self._inner.enter()
        self._trace.append(
            {"kind": "enter", "virtual_time_s": self._inner.virtual_time_s}
        )
        return result

    def exit(self):
        result = self._inner.exit()
        self._trace.append(
            {"kind": "exit", "virtual_time_s": self._inner.virtual_time_s}
        )
        return result

    def _moved(self, x: float, y: float) -> float:
        before = self._inner.current_position
        return math.hypot(float(x) - before[0], float(y) - before[1])

    def measure(self, x: float, y: float, channel: int):
        before_time = float(self._inner.virtual_time_s)
        before_channel = self._inner.current_channel
        moved = self._moved(x, y)
        result = self._inner.measure(x, y, channel)
        self._trace.append(
            {
                "kind": "measure",
                "x": float(x),
                "y": float(y),
                "channel": int(channel),
                "result": result.result,
                "svd_deg": result.svd_deg,
                "move_m": moved,
                "switched": before_channel is not None
                and int(channel) != before_channel,
                "duration_s": float(self._inner.virtual_time_s) - before_time,
                "virtual_time_s": float(self._inner.virtual_time_s),
            }
        )
        return result

    def clear(self, x: float, y: float, channel: int):
        before_time = float(self._inner.virtual_time_s)
        moved = self._moved(x, y)
        result = self._inner.clear(x, y, channel)
        self._trace.append(
            {
                "kind": "clear",
                "x": float(x),
                "y": float(y),
                "channel": int(channel),
                "cleared": bool(result.cleared),
                "move_m": moved,
                "duration_s": float(self._inner.virtual_time_s) - before_time,
                "virtual_time_s": float(self._inner.virtual_time_s),
            }
        )
        return result


def action_times(trace: list[dict[str, Any]]) -> dict[str, float]:
    """把动作时间线拆成移动/检测/切频/清除/尾部补盲（秒）。"""
    movement = sum(row.get("move_m", 0.0) for row in trace) / MOVE_SPEED_MPS
    measures = sum(1 for row in trace if row["kind"] == "measure")
    switches = sum(1 for row in trace if row.get("switched"))
    successes = sum(1 for row in trace if row["kind"] == "clear" and row["cleared"])
    failures = sum(1 for row in trace if row["kind"] == "clear" and not row["cleared"])
    clear_times = [
        row["virtual_time_s"]
        for row in trace
        if row["kind"] == "clear" and row["cleared"]
    ]
    total = trace[-1]["virtual_time_s"] if trace else 0.0
    return {
        "movement_time_s": movement,
        "measurement_time_s": measures * MEASURE_TIME_S,
        "switching_time_s": switches * SWITCH_TIME_S,
        "clearing_time_s": successes * CLEAR_SUCCESS_TIME_S
        + failures * CLEAR_FAIL_TIME_S,
        "tail_after_last_clear_s": (total - max(clear_times)) if clear_times else 0.0,
        "measure_count": float(measures),
        "clear_count": float(successes + failures),
        "clear_success_count": float(successes),
    }


class PracticeGuardError(SystemExit):
    """安全门拒绝启动（用 SystemExit，调用方无法忽略）。"""


def practice_guard(args, *, interactive: bool = True) -> None:
    """在创建 HTTP 客户端之前校验 practice 模式：队号 + 显式确认 + 回环地址。"""
    robot_id = getattr(args, "robot_id", None)
    if not robot_id or str(robot_id).strip() in ("", "<参赛队号>"):
        raise PracticeGuardError("practice 模式必须给出 --robot-id（当前登录队号）")

    if not getattr(args, "confirm_problem3_practice", False):
        raise PracticeGuardError(
            "practice 会消耗真实测试机会：确认已登录、已在赛方客户端选择"
            "「问题 3 演练测试」、倒计时结束后，再加 --confirm-problem3-practice"
        )

    base_url = getattr(args, "base_url", "") or ""
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    allowed = LOOPBACK_HOSTS | _local_addresses()
    if parsed.scheme != "http" or not host:
        raise PracticeGuardError(f"base-url 必须是 http 回环地址，收到 {base_url!r}")
    if host not in allowed:
        raise PracticeGuardError(
            f"只允许本机回环地址，收到 {base_url!r}；如确需其它地址请自行确认它属于本机"
        )

    if interactive and sys.stdin and sys.stdin.isatty():
        print(f"[practice] 将连接 {base_url}（队号 {robot_id}）。")
        print("HTTP 接口无法区分演练与正式，请自行核对赛方客户端界面。")
        try:
            answer = input("输入「问题3演练测试」后回车开始，其他输入取消：").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "问题3演练测试":
            raise PracticeGuardError("已取消 practice 运行，未发送任何 HTTP 请求")


def _local_addresses() -> set[str]:
    addresses = {"127.0.0.1", "localhost", "::1"}
    try:
        addresses.add(socket.gethostname())
        addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return addresses


def run_coverage_case(
    policy: str,
    seed: int,
    *,
    num_sources: int | None = None,
    verbose: bool = False,
    log_path: str | None = None,
    keep_trace: bool = False,
    source: str | Path | None = None,
    **policy_kwargs,
) -> dict[str, Any]:
    """覆盖地图单个案例的离线运行（``coverage_main.py`` 的 offline 分支用）。"""
    from q3_practice_v2_local_benchmark import (  # 惰性，避免循环导入
        DEFAULT_EXTERNAL_SOURCE,
        load_external_implementation,
    )

    simulator = LocalSimulator(seed, num_sources=num_sources)
    trace: list[dict[str, Any]] = []
    config = StrategyConfig(verbose=verbose, log_path=log_path)
    if policy == SUPPLIED_POLICY:
        strategy = Problem3Strategy(RecordingRobot(simulator, trace), config)
    else:
        types = load_external_implementation(source or DEFAULT_EXTERNAL_SOURCE)
        coverage_class = types[1]
        map_config_class = types[2]
        strategy = coverage_class(
            RecordingRobot(simulator, trace),
            config,
            map_config_class(mode=policy, **policy_kwargs),
        )

    error: str | None = None
    summary: dict[str, Any] = {}
    try:
        summary = dict(strategy.run() or {})
    except Exception as exc:  # noqa: BLE001 - 单案失败不终止整批
        error = repr(exc)

    source_total = int(simulator.total_sources)
    cleared = int(simulator.cleared_count)
    total_time = float(simulator.virtual_time_s)
    split = action_times(trace)
    result: dict[str, Any] = {
        "policy": policy,
        "seed": seed,
        "error": error,
        "source_total": source_total,
        "source_cleared": cleared,
        "success": error is None and cleared == source_total,
        "all_resolved": bool(summary.get("all_resolved", False)),
        "total_virtual_time_s": total_time,
        "virtual_time_min": total_time / 60.0,
        "seconds_per_source": (total_time / source_total) if source_total else None,
        "model_summary": summary,
        **split,
    }
    if result["seconds_per_source"] is not None:
        result["per_source_movement_s"] = split["movement_time_s"] / source_total
        result["per_source_action_s"] = (
            split["measurement_time_s"]
            + split["switching_time_s"]
            + split["clearing_time_s"]
        ) / source_total
        result["per_source_tail_s"] = split["tail_after_last_clear_s"] / source_total
    if keep_trace:
        result["trace"] = trace
    return result
