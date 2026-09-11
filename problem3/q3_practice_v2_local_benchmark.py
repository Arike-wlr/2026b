"""``q3_practice_v2`` 系列的**离线接口适配层**（不含任何策略算法）。

``q3_empty_channel_strategy.py`` 通过本模块的五个名字跑批：

    DEFAULT_EXTERNAL_SOURCE      外部"模型包"文件（用 --source 覆盖）
    load_external_implementation 取出基类，返回元组，[1] 就是可被继承的基类
    run_case(case_id, seed, strategy_types, strategy_class=...)  跑一个案例
    summarize(results, random_state)                              汇总
    write_results(results, summary, output_prefix)                落 CSV + JSON

``load_external_implementation`` 认两种外部文件：

1. **单文件打包**（如 ``q3_practice_v2.py``）：文件里有 ``load_implementation()``，
   它把内嵌的 client / problem3_geometry / problem3_strategy / coverage_strategy
   注册进 ``sys.modules``，并返回 ``(RobotClient, CoverageStrategy, MapConfig,
   StrategyConfig)``——本适配层直接沿用这个元组，``[1]`` 即基类；
2. **普通模块**：模块里直接定义了策略类时按名字取（CoverageStrategy → …）。

离线运行用它接到本仓库现役的 ``problem3_local_sim.LocalSimulator``，并用
``research_adapter.RecordingRobot`` 记录时间线；被包装后的对象与真实
``client.RobotClient`` 签名一致，所以同一份策略类能原样上真实接口
（见 ``q3_empty_channel_practice.py``）。**模型文件一行不改。**

用法::

    python problem3/q3_practice_v2_local_benchmark.py --cases 3 --source problem3/q3_practice_v2.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from problem3_local_sim import LocalSimulator  # noqa: E402
from problem3_strategy import StrategyConfig  # noqa: E402
from research_adapter import RecordingRobot, action_times  # noqa: E402

#: 外部"模型包"；本仓库现成的是单文件打包 q3_practice_v2.py
DEFAULT_EXTERNAL_SOURCE = _HERE / "q3_practice_v2.py"

#: 普通模块里认基类时的优先名
PREFERRED_CLASS_NAMES = ("CoverageStrategy", "Problem3Strategy", "Strategy")


def load_external_implementation(path: Path | str) -> tuple[Any, ...]:
    """加载外部模型文件，返回元组且保证 ``[1]`` 是可继承的策略基类。

    单文件打包返回其 ``load_implementation()`` 的原始元组
    （``RobotClient, CoverageStrategy, MapConfig, StrategyConfig``）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到外部模型文件 {path}；用 --source 指到你那份策略文件"
            f"（单文件打包或普通模块均可）。"
        )
    spec = importlib.util.spec_from_file_location(f"_external_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {path} 建立模块规格")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    loader = getattr(module, "load_implementation", None)
    if callable(loader):
        types = loader()
        if (
            isinstance(types, (tuple, list))
            and len(types) >= 2
            and isinstance(types[1], type)
        ):
            return tuple(types)
        raise LookupError(
            f"{path} 的 load_implementation() 未返回 "
            f"(RobotClient, CoverageStrategy, MapConfig, StrategyConfig) 元组"
        )

    for name in PREFERRED_CLASS_NAMES:
        candidate = getattr(module, name, None)
        if isinstance(candidate, type) and hasattr(candidate, "run"):
            return (
                module,
                candidate,
                getattr(module, "MapConfig", None),
                getattr(module, "StrategyConfig", None),
            )
    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and hasattr(value, "run")
        and getattr(value, "__module__", "") == module.__name__
    ]
    if len(candidates) == 1:
        return (module, candidates[0], None, None)
    raise LookupError(
        f"{path} 里有 {len(candidates)} 个候选策略类 "
        f"{[cls.__name__ for cls in candidates]}；请让文件提供 load_implementation() "
        f"或把基类命名为 {' / '.join(PREFERRED_CLASS_NAMES)} 之一"
    )


def instantiate(
    strategy_class: type,
    robot,
    *,
    config_class: type | None = None,
    verbose: bool = False,
    log_path: str | None = None,
):
    """按基类构造签名决定是否传配置对象（``(robot, config)`` 或 ``(robot)``）。

    优先用模型自带的配置类（``config_class``），避免跨版本字段不一致。
    """
    config_type = config_class or StrategyConfig
    parameters = [
        parameter
        for parameter in inspect.signature(strategy_class.__init__).parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(parameters) - 1 >= 2:  # 减掉 self
        return strategy_class(robot, config_type(verbose=verbose, log_path=log_path))
    return strategy_class(robot)


def run_case(
    case_id: int,
    seed: int,
    strategy_types: tuple[Any, ...] | None = None,
    *,
    strategy_class: type | None = None,
    num_sources: int | None = None,
    verbose: bool = False,
    log_path: str | None = None,
    keep_trace: bool = False,
) -> dict[str, Any]:
    """在本地仿真器上跑一个案例；返回与批量汇总一致的逐案指标。"""
    if strategy_class is None:
        if strategy_types is None:
            raise ValueError("strategy_class 与 strategy_types 至少要给一个")
        strategy_class = strategy_types[1]
    config_class = (
        strategy_types[3]
        if strategy_types is not None
        and len(strategy_types) >= 4
        and isinstance(strategy_types[3], type)
        else None
    )

    simulator = LocalSimulator(seed, num_sources=num_sources)
    trace: list[dict[str, Any]] = []
    strategy = instantiate(
        strategy_class,
        RecordingRobot(simulator, trace),
        config_class=config_class,
        verbose=verbose,
        log_path=log_path,
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
        "case_id": case_id,
        "seed": seed,
        "strategy": strategy_class.__name__,
        "error": error,
        "source_total": source_total,
        "source_cleared": cleared,
        "success": error is None and cleared == source_total,
        "total_virtual_time_s": total_time,
        "virtual_time_min": total_time / 60.0,
        "seconds_per_source": (total_time / source_total) if source_total else None,
        "strategy_summary": summary,
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


def summarize(results: list[dict[str, Any]], random_state: int) -> dict[str, Any]:
    """逐案"总时间 ÷ 真实源数"后再跨案取均值；失败案例保留参与统计。"""
    per_source = [
        float(case["seconds_per_source"])
        for case in results
        if case["seconds_per_source"] is not None
    ]
    return {
        "case_count": len(results),
        "random_state": random_state,
        "success_count": sum(1 for case in results if case["success"]),
        "success_rate": (
            sum(1 for case in results if case["success"]) / len(results)
            if results
            else 0.0
        ),
        "mean_source_count": (
            statistics.fmean(float(case["source_total"]) for case in results)
            if results
            else None
        ),
        "mean_seconds_per_source": (
            statistics.fmean(per_source) if per_source else None
        ),
        "median_seconds_per_source": (
            statistics.median(per_source) if per_source else None
        ),
        "p90_seconds_per_source": _percentile(per_source, 90),
        "share_within_250s": (
            sum(1 for value in per_source if value <= 250.0) / len(per_source)
            if per_source
            else None
        ),
        "mean_per_source_movement_s": _mean(results, "per_source_movement_s"),
        "mean_per_source_action_s": _mean(results, "per_source_action_s"),
        "mean_per_source_tail_s": _mean(results, "per_source_tail_s"),
        "mean_total_virtual_time_s": _mean(results, "total_virtual_time_s"),
        "errors": [case["error"] for case in results if case["error"]],
    }


def _mean(results: list[dict[str, Any]], key: str) -> float | None:
    values = [float(case[key]) for case in results if case.get(key) is not None]
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = int(position // 1)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


@dataclass
class _CaseRow:
    case_id: int
    seed: int
    strategy: str
    source_total: int
    source_cleared: int
    success: bool
    seconds_per_source: float | None
    error: str | None


def write_results(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    output_prefix: Path | str,
) -> None:
    """写 ``<prefix>.csv``（逐案）与 ``<prefix>.json``（汇总）。"""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        asdict(
            _CaseRow(
                case_id=int(case["case_id"]),
                seed=int(case["seed"]),
                strategy=str(case["strategy"]),
                source_total=int(case["source_total"]),
                source_cleared=int(case["source_cleared"]),
                success=bool(case["success"]),
                seconds_per_source=case["seconds_per_source"],
                error=case["error"],
            )
        )
        for case in results
    ]
    with output_prefix.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="q3_practice_v2 系列策略的离线基准（不含模型）"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_EXTERNAL_SOURCE)
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--source-count", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=_HERE / "outputs/tables/q3_practice_v2_local_benchmark",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    strategy_types = load_external_implementation(args.source)
    results = [
        run_case(
            index + 1,
            args.random_state + index,
            strategy_types,
            num_sources=args.source_count,
        )
        for index in range(args.cases)
    ]
    summary = summarize(results, args.random_state)
    summary["source_path"] = str(Path(args.source).resolve())
    summary["base_strategy"] = strategy_types[1].__name__
    write_results(results, summary, args.output_prefix)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["success_count"] == summary["case_count"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
