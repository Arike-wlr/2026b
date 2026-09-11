"""``q3_empty_channel_strategy`` 到真实接口（``client.RobotClient``）的适配入口。

同一个策略类，两种跑法：

* ``--mode offline``（默认）：走 ``q3_practice_v2_local_benchmark`` 的离线基准，
  用 ``problem3_local_sim.LocalSimulator`` 跑，不碰网络；
* ``--mode practice``：先用 ``practice_guard`` 过安全门（必须有 ``--robot-id``、
  必须显式 ``--confirm-problem3-practice``、base-url 必须是回环地址），再建
  ``client.RobotClient`` 跑同一份策略类，最后把结果写到独立目录。

策略构造参数与 ``q3_empty_channel_strategy.py`` **完全共用同一套 argparse 定义**
（这里只是借用它的 ``parse_args``），所以两边不会漂移；模型/base 策略一行不改。

用法::

    # 离线（默认），参数与 q3_empty_channel_strategy.py 一致
    python problem3/q3_empty_channel_practice.py --cases 5 --source <base策略文件>

    # 真实接口（消耗一次演练机会；倒计时结束后、核对过平台界面再加确认）
    python problem3/q3_empty_channel_practice.py --mode practice \
        --robot-id 202610038037 --confirm-problem3-practice --source <base策略文件>

不提供 formal 模式：HTTP 协议无法区分演练与正式。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import q3_empty_channel_strategy as q3

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """只解析 practice 相关开关，其余参数原样透传给策略脚本自己的 argparse。"""
    parser = argparse.ArgumentParser(
        description="q3_empty_channel 的离线/演练双模式入口",
        epilog=(
            "其余参数（--cases/--random-state/--source/--hex-cover/--ring-radius 等）"
            "全部沿用 q3_empty_channel_strategy.py 的定义，"
            "完整列表见：python problem3/q3_empty_channel_strategy.py --help"
        ),
    )
    parser.add_argument("--mode", choices=["offline", "practice"], default="offline")
    parser.add_argument("--robot-id", help="参赛队号，必须与当前登录队号一致")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument(
        "--confirm-problem3-practice",
        action="store_true",
        help="确认已登录、已开始“问题 3 演练测试”、倒计时已结束",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_known_args(argv)


def external_args(rest: list[str]) -> argparse.Namespace:
    """用策略脚本自己的 argparse 解析剩余参数（借用其定义，避免两处漂移）。"""
    saved = sys.argv
    sys.argv = [str(_HERE / "q3_empty_channel_strategy.py"), *rest]
    try:
        return q3.parse_args()
    finally:
        sys.argv = saved


def build_strategy_class(args: argparse.Namespace, strategy_types) -> type:
    """与 q3_empty_channel_strategy.main 的构造分支逐条对应。"""
    gain_threshold = (
        q3.DISABLED_PIGGYBACK_GAIN_M2 if args.disable_piggyback else args.gain_threshold
    )
    always_finish_channel = not (
        args.disable_finish_exception or args.disable_piggyback
    )
    if args.hex_cover:
        return q3.build_hex_cover_strategy(
            strategy_types[1],
            gain_threshold,
            always_finish_channel=always_finish_channel,
            ring_radius_m=args.ring_radius,
            resume_piggyback_at_known_sources=args.resume_piggyback_at_known_sources,
            resumed_gain_threshold_m2=args.resumed_gain_threshold,
            resume_dynamic_search=args.resume_dynamic_search,
            resume_by_search_stop=args.resume_by_search_stop,
            enable_marginal_scan=args.marginal_scan,
            marginal_scan_margin_s=args.marginal_scan_margin,
            enable_neighborhood_cover=args.neighborhood_cover,
            neighborhood_iterations=args.neighborhood_iterations,
        )
    return q3.build_empty_channel_strategy(
        strategy_types[1],
        gain_threshold,
        always_finish_channel=always_finish_channel,
        resume_piggyback_at_known_sources=args.resume_piggyback_at_known_sources,
        resumed_gain_threshold_m2=args.resumed_gain_threshold,
        resume_by_search_stop=args.resume_by_search_stop,
    )


def run_practice(mode_args: argparse.Namespace, rest: list[str]) -> int:
    from research_adapter import practice_guard  # 发出任何 HTTP 之前的安全门
    from q3_practice_v2_local_benchmark import (
        instantiate,
        load_external_implementation,
    )

    practice_guard(mode_args)

    args = external_args(rest)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = mode_args.output_dir or (_HERE / "research" / f"q3_practice_{stamp}")
    out.mkdir(parents=True, exist_ok=False)

    strategy_types = load_external_implementation(args.source)
    strategy_class = build_strategy_class(args, strategy_types)

    # 优先用模型自带的 RobotClient（单文件打包里内嵌了一份），保证与其一致
    robot_class = strategy_types[0] if isinstance(strategy_types[0], type) else None
    if robot_class is None:
        from client import RobotClient as robot_class

    robot = robot_class(
        mode_args.robot_id,
        base_url=mode_args.base_url,
        log_dir=str(out / "http"),
    )
    strategy = instantiate(
        strategy_class, robot, log_path=str(out / "actions.tsv")
    )
    result: dict[str, object] = {
        "mode": "practice",
        "strategy": strategy_class.__name__,
        "robot_client": robot_class.__name__,
        "hex_cover": bool(args.hex_cover),
        "source": str(Path(args.source).resolve()),
    }
    try:
        result.update(dict(strategy.run() or {}))
    except Exception as error:  # noqa: BLE001 - 任何失败都落盘后再退出
        result["error"] = repr(error)
    finally:
        robot.close()
        (out / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"输出目录：{out}")
    return 0 if result.get("all_resolved") else 1


def main(argv: list[str] | None = None) -> int:
    mode_args, rest = parse_args(argv)
    if mode_args.mode == "offline":
        # 离线就是策略脚本自己的批量基准，本适配层不重复实现
        saved = sys.argv
        sys.argv = [str(_HERE / "q3_empty_channel_strategy.py"), *rest]
        try:
            return q3.main()
        finally:
            sys.argv = saved
    return run_practice(mode_args, rest)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
