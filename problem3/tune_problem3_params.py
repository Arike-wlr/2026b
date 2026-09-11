"""问题3策略参数网格搜索。

用于比较 Region Clustering 与 Opportunistic Strike 的两个核心参数：

    REGION_CLUSTER_DISTANCE
    OPPORTUNISTIC_MAX_DETOUR

运行：
    python problem3/tune_problem3_params.py --cases 40
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from problem3_local_sim import LocalSimulator  # noqa: E402
from problem3_strategy import Problem3Strategy, StrategyConfig  # noqa: E402


def run_one(seed: int, cluster_distance: float, max_detour: float) -> Dict[str, object]:
    sim = LocalSimulator(seed)
    cfg = StrategyConfig(
        region_cluster_distance=cluster_distance,
        opportunistic_max_detour=max_detour,
    )
    summary = Problem3Strategy(sim, cfg).run()
    summary["source_total"] = sim.total_sources
    summary["source_cleared"] = sim.cleared_count
    summary["success"] = sim.cleared_count == sim.total_sources
    return summary


def evaluate(cases: int, seed: int, cluster_distance: float, max_detour: float) -> Dict[str, float]:
    results = [
        run_one(seed + index, cluster_distance, max_detour)
        for index in range(cases)
    ]
    times = [float(item["virtual_time_min"]) for item in results]
    successes = [item for item in results if item["success"]]

    def mean_stat(name: str) -> float:
        return float(np.mean([float(item["stats"][name]) for item in results]))  # type: ignore[index]

    return {
        "cluster_distance": cluster_distance,
        "max_detour": max_detour,
        "success_rate": len(successes) / cases,
        "avg_virtual_min": float(np.mean(times)),
        "median_virtual_min": float(statistics.median(times)),
        "p90_virtual_min": float(np.percentile(np.asarray(times), 90)),
        "max_virtual_min": float(np.max(times)),
        "avg_move_min": float(np.mean([float(item["move_time_s"]) / 60.0 for item in results])),
        "avg_measures": mean_stat("measures"),
        "avg_clears": mean_stat("clears"),
        "avg_clusters": mean_stat("clusters"),
        "avg_opportunistic": mean_stat("opportunistic_strikes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="问题3 Region Clustering / 顺路拦截参数搜索")
    parser.add_argument("--cases", type=int, default=40, help="每组参数仿真案例数")
    parser.add_argument("--seed", type=int, default=9000, help="起始随机种子")
    args = parser.parse_args()

    cluster_values = [0.0, 250.0, 400.0, 520.0, 700.0, 900.0, 1200.0]
    detour_values = [0.0, 40.0, 90.0, 160.0, 260.0]

    rows: List[Dict[str, float]] = []
    total = len(cluster_values) * len(detour_values)
    done = 0
    for cluster_distance in cluster_values:
        for max_detour in detour_values:
            done += 1
            print(f"[{done}/{total}] cluster={cluster_distance:.0f}, detour={max_detour:.0f}")
            rows.append(evaluate(args.cases, args.seed, cluster_distance, max_detour))

    rows.sort(key=lambda item: (item["success_rate"] < 1.0, item["avg_virtual_min"], item["p90_virtual_min"]))

    print()
    print("排名  cluster  detour  成功率  平均min  中位min  P90min  最长min  移动min  清除数  顺路数")
    print("-" * 96)
    for rank, row in enumerate(rows[:15], start=1):
        print(
            f"{rank:>2}  "
            f"{row['cluster_distance']:>7.0f}  "
            f"{row['max_detour']:>6.0f}  "
            f"{row['success_rate'] * 100:>5.1f}%  "
            f"{row['avg_virtual_min']:>7.2f}  "
            f"{row['median_virtual_min']:>7.2f}  "
            f"{row['p90_virtual_min']:>6.2f}  "
            f"{row['max_virtual_min']:>7.2f}  "
            f"{row['avg_move_min']:>7.2f}  "
            f"{row['avg_clears']:>6.1f}  "
            f"{row['avg_opportunistic']:>6.1f}"
        )

    best = rows[0]
    print()
    print(
        "建议参数："
        f"REGION_CLUSTER_DISTANCE = {best['cluster_distance']:.0f}, "
        f"OPPORTUNISTIC_MAX_DETOUR = {best['max_detour']:.0f}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
