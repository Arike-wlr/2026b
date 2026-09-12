"""问题 3 最终演练入口：把 ``problem3_strategy`` 的模型 ``RobotClient`` 与策略类串起来。

策略与模型全部在 ``problem3_strategy.py`` 中（同一份策略类既能接真实
``RobotClient``，也能接本地仿真器）；本文件是运行的那个：

    python problem3/main.py --check                    # 只检查依赖/参数，不联网
    python problem3/main.py --robot-id 你们的参赛队号 \
        --confirm-problem3-practice                    # 问题3 演练测试（会消耗一次机会）

前置条件：赛方客户端已登录、已选择「问题3 演练测试」、倒计时结束、接口就绪。
HTTP 无法区分演练与正式，**不要选择正式测试**。
本地随机测试（离线批量、不联网）请运行 ``python problem3/problem3_strategy.py``。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# 让脚本既能 `python problem3/main.py` 直接运行，也能被导入。
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE.parent), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from problem3_strategy import (  # noqa: E402
    DEFAULT_HEX_RING_RADIUS_M,
    DEFAULT_PIGGYBACK_GAIN_M2,
    build_strategy,
)

# --------------------------------------------------------------------------- #
# 运行参数（与附件2示例一致，直接改这里即可）
# --------------------------------------------------------------------------- #
ROBOT_ID = "202610038037"       # 参赛队号，必须与当前登录的队号一致
BASE_URL = "http://127.0.0.1:2026"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class PracticeGuardError(SystemExit):
    """安全门拒绝启动（用 SystemExit，调用方无法忽略）。"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="问题3 最终演练入口（空频道低源覆盖策略）",
    )
    parser.add_argument("--robot-id", default=ROBOT_ID, help="当前登录参赛队号")
    parser.add_argument("--base-url", default=BASE_URL, help="模拟器地址（仅本机回环）")
    parser.add_argument(
        "--confirm-problem3-practice",
        action="store_true",
        help="确认已登录、已开始「问题3 演练测试」、倒计时已结束",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="结果输出目录")
    parser.add_argument("--check", action="store_true", help="只检查依赖与参数，不联网")
    parser.add_argument("--verbose", action="store_true", help="打印通信层请求/响应日志")
    # ---- 策略开关（默认为最终演练采用的组合，一般无需改动）----
    parser.add_argument("--hex-cover", action="store_true", help="启用认证六边形覆盖骨架")
    parser.add_argument("--ring-radius", type=float, default=DEFAULT_HEX_RING_RADIUS_M)
    parser.add_argument("--gain-threshold", type=float, default=DEFAULT_PIGGYBACK_GAIN_M2)
    parser.add_argument(
        "--disable-finish-exception",
        action="store_true",
        help="关闭“这次测点即可测完该频道剩余区域则必测”的规则",
    )
    parser.add_argument(
        "--disable-piggyback",
        action="store_true",
        help="仅在强制搜索点批量覆盖未知频道",
    )
    parser.add_argument(
        "--resume-piggyback-at-known-sources",
        type=int,
        default=None,
        help="确认该数量源后恢复选择性顺带覆盖",
    )
    parser.add_argument("--resumed-gain-threshold", type=float, default=DEFAULT_PIGGYBACK_GAIN_M2)
    parser.add_argument("--resume-dynamic-search", action="store_true")
    parser.add_argument("--resume-by-search-stop", type=int, default=None)
    parser.add_argument("--marginal-scan", action="store_true")
    parser.add_argument("--marginal-scan-margin", type=float, default=2.0)
    parser.add_argument("--neighborhood-cover", action="store_true")
    parser.add_argument("--neighborhood-iterations", type=int, default=4)
    return parser


def _local_addresses() -> set[str]:
    addresses = set(LOOPBACK_HOSTS)
    try:
        addresses.add(socket.gethostname())
        addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return addresses


def practice_guard(args: argparse.Namespace) -> None:
    """在创建 HTTP 客户端之前校验：队号 + 显式确认 + 回环地址。

    非交互：只要给出 --robot-id 与 --confirm-problem3-practice、且 base-url 是
    本机回环地址，即直接开始，不再要求手动输入确认词。
    """
    robot_id = getattr(args, "robot_id", None)
    if not robot_id or str(robot_id).strip() in ("", "<参赛队号>"):
        raise PracticeGuardError("practice 必须给出 --robot-id（当前登录队号）")
    if not getattr(args, "confirm_problem3_practice", False):
        raise PracticeGuardError(
            "practice 会消耗真实测试机会：确认已登录、已在赛方客户端选择"
            "「问题3 演练测试」、倒计时结束后，再加 --confirm-problem3-practice"
        )
    base_url = getattr(args, "base_url", "") or ""
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if parsed.scheme != "http" or not host:
        raise PracticeGuardError(f"base-url 必须是 http 回环地址，收到 {base_url!r}")
    if host not in _local_addresses():
        raise PracticeGuardError(
            f"只允许本机回环地址，收到 {base_url!r}；如确需其它地址请自行确认它属于本机"
        )


def check_dependencies() -> int:
    """只检查依赖与参数，不发送任何 HTTP 请求。"""
    try:
        import numpy
        import requests
        import shapely
    except ImportError as error:
        print("缺少依赖：", error)
        print('请运行：python -m pip install numpy "shapely>=2.0" "requests>=2.32,<3"')
        return 2
    print("检查通过；没有发送任何 HTTP 请求。")
    print(f"numpy {numpy.__version__}  shapely {shapely.__version__}  requests {requests.__version__}")
    strategy_class, strategy_types = build_strategy()
    print(f"基础策略：{strategy_types[1].__name__} -> {strategy_class.__name__}")
    return 0


def print_summary(summary: dict, action_log: object, robot) -> None:
    """按原 problem3 入口的版式打印运行结果。"""
    print("=" * 64)
    print("问题3 运行结果")
    print("=" * 64)
    print(f"干扰源：检测到 {summary['detected']} 个，已清除 {summary['cleared']} 个，"
          f"空频道证书 {summary['empty_certified']} 个，未决 {summary['unknown']} 个")
    print(f"所有频道均已解决：{summary['all_resolved']}")
    print(f"总虚拟时间：{float(summary['virtual_time_min']):.2f} min "
          f"（{float(summary['virtual_time_s']):.1f} s）")
    print(f"移动时间：{float(summary['move_time_s']) / 60.0:.2f} min，"
          f"动作时间：{float(summary['action_time_s']) / 60.0:.2f} min")
    if summary["avg_clear_time_s"] is not None:
        print(f"平均单源定位清除时间：{float(summary['avg_clear_time_s']):.2f} s")
    print(f"动作日志：{action_log}")
    print(f"通信日志：{robot.log_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.check:
        return check_dependencies()

    practice_guard(args)

    strategy_class, strategy_types = build_strategy(
        hex_cover=args.hex_cover,
        gain_threshold_m2=args.gain_threshold,
        disable_piggyback=args.disable_piggyback,
        disable_finish_exception=args.disable_finish_exception,
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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = args.output_dir or (_HERE / "q3_logs" / f"practice_{stamp}")
    out.mkdir(parents=True, exist_ok=False)

    # 优先用内嵌模型自带的 RobotClient，保证与策略同源。
    robot_class = strategy_types[0] if isinstance(strategy_types[0], type) else None
    if robot_class is None:  # pragma: no cover - 兜底
        from client import RobotClient as robot_class
    robot = robot_class(
        args.robot_id,
        base_url=args.base_url,
        log_dir=str(out / "http"),
        verbose=args.verbose,
    )
    # 记录动作日志（bearing_error_deg=1.01 与基类内部默认值一致，行为不变）。
    config = strategy_types[3](
        bearing_error_deg=1.01,
        log_path=str(out / "actions.tsv"),
        verbose=args.verbose,
    )
    strategy = strategy_class(robot, config)

    error: str | None = None
    summary: dict = {}
    try:
        summary = dict(strategy.run() or {})
    except Exception as exc:  # noqa: BLE001 - 任何失败都落盘后再退出
        error = repr(exc)
    finally:
        robot.close()

    result: dict[str, object] = {
        "mode": "practice",
        "strategy": strategy_class.__name__,
        "robot_client": robot_class.__name__,
        "hex_cover": bool(args.hex_cover),
        **summary,
    }
    if error is not None:
        result["error"] = error
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if error is not None:
        print(f"运行失败：{error}")
    action_log = getattr(getattr(strategy, "logger", None), "path", None)
    if summary:
        print_summary(summary, action_log, robot)
    print(f"输出目录：{out}")
    return 0 if summary.get("all_resolved") else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
