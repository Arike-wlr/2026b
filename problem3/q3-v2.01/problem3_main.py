"""问题 3 最终演练入口：把 ``problem3_strategy`` 的模型 ``RobotClient`` 与策略类串起来。

策略入口在 ``problem3_strategy.py``，旧模型及通信层在 ``problem3_legacy.py``。
同一策略类既能接真实 ``RobotClient``，也能接本地仿真器：

    python problem3_main.py --check                    # 只检查依赖/参数，不联网
    python problem3_main.py --robot-id 你们的参赛队号 \
        --confirm-problem3-practice                    # 问题3 演练测试（会消耗一次机会）

前置条件：赛方客户端已登录、已选择「问题3 演练测试」、倒计时结束、接口就绪。
HTTP 无法区分演练与正式，**不要选择正式测试**。
本地随机测试（离线批量、不联网）请运行 ``python problem3_strategy.py``。
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time
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
        description="问题3 最终演练入口（v2.01 六扇区残余覆盖策略）",
    )
    parser.add_argument("--robot-id", default=ROBOT_ID, help="当前登录参赛队号")
    parser.add_argument("--base-url", default=BASE_URL, help="模拟器地址（仅本机回环）")
    parser.add_argument(
        "--confirm-problem3-practice",
        action="store_true",
        help="确认已登录、已开始「问题3 演练测试」、倒计时已结束",
    )
    parser.add_argument(
        "--case-code",
        default=None,
        help="可选预填；通常不需要，联网测试 /exit 后会提示录入界面案例编码",
    )
    parser.add_argument(
        "--source-count",
        type=int,
        default=None,
        help="测试结束后界面显示的干扰源总数；用于未完成案例计算清除比例",
    )
    parser.add_argument(
        "--no-post-run-prompt",
        action="store_true",
        help="结束后不交互补录案例编码/源总数（适合无人值守运行）",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="结果输出目录")
    parser.add_argument("--check", action="store_true", help="只检查依赖与参数，不联网")
    parser.add_argument("--verbose", action="store_true", help="打印通信层请求/响应日志")
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


class RuntimeRecorder:
    """给问题3内嵌客户端补充程序运行时间记录，不改变任何机器人动作。

    优先用 /enter 与 /exit 响应中的 ``real_timestamp_ms`` 计算服务器运行的
    程序运行时间；取不到时回退到本机单调时钟（从 /enter 到 /exit 的耗时）。
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._enter_response: dict[str, object] = {}
        self._exit_response: dict[str, object] = {}
        self._enter_monotonic_s: float | None = None
        self._program_runtime_s: float | None = None
        self._program_runtime_source: str | None = None
        self._exit_reason: str | None = None

        original_post = inner._post

        def recording_post(path: str, payload: dict) -> dict:
            data = original_post(path, payload)
            if path == "/enter":
                self._enter_response = dict(data or {})
                self._enter_monotonic_s = time.monotonic()
            elif path == "/exit":
                self._exit_response = dict(data or {})
                self._capture_server_runtime()
            return data

        # 内嵌策略直接持有 inner；只替换它的请求方法即可完整记录响应。
        inner._post = recording_post

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    @staticmethod
    def _find_value(data: object, names: tuple[str, ...]) -> object | None:
        if isinstance(data, dict):
            for name in names:
                if name in data and data[name] not in (None, ""):
                    return data[name]
            for value in data.values():
                found = RuntimeRecorder._find_value(value, names)
                if found not in (None, ""):
                    return found
        elif isinstance(data, list):
            for value in data:
                found = RuntimeRecorder._find_value(value, names)
                if found not in (None, ""):
                    return found
        return None

    @staticmethod
    def _as_float(value: object) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _capture_server_runtime(self) -> None:
        enter_ms = self._as_float(
            self._find_value(self._enter_response, ("real_timestamp_ms", "realTimestampMs"))
        )
        exit_ms = self._as_float(
            self._find_value(self._exit_response, ("real_timestamp_ms", "realTimestampMs"))
        )
        if enter_ms is not None and exit_ms is not None and exit_ms >= enter_ms:
            self._program_runtime_s = (exit_ms - enter_ms) / 1000.0
            self._program_runtime_source = "server_real_timestamp_ms"

    def enter(self):
        result = self._inner.enter()
        if not self._enter_response and isinstance(result, dict):
            self._enter_response = dict(result)
        if self._enter_monotonic_s is None:
            self._enter_monotonic_s = time.monotonic()
        return result

    def exit(self):
        result = self._inner.exit()
        if not self._exit_response and isinstance(result, dict):
            self._exit_response = dict(result)
            self._capture_server_runtime()
        reason = result or self._find_value(
            self._exit_response,
            ("exit_reason", "exitReason", "reason", "message"),
        )
        self._exit_reason = None if reason in (None, "") else str(reason)
        if self._program_runtime_s is None and self._enter_monotonic_s is not None:
            self._program_runtime_s = max(0.0, time.monotonic() - self._enter_monotonic_s)
            self._program_runtime_source = "local_monotonic_clock"
        return result

    def close(self) -> None:
        self._inner.close()

    @property
    def enter_response(self) -> dict[str, object]:
        return dict(self._enter_response)

    @property
    def exit_response(self) -> dict[str, object]:
        return dict(self._exit_response)

    @property
    def exit_reason(self) -> str | None:
        return self._exit_reason

    @property
    def program_runtime_s(self) -> float | None:
        if self._program_runtime_s is not None:
            return self._program_runtime_s
        if self._enter_monotonic_s is not None:
            return max(0.0, time.monotonic() - self._enter_monotonic_s)
        return None

    @property
    def program_runtime_source(self) -> str | None:
        if self._program_runtime_source is not None:
            return self._program_runtime_source
        if self._enter_monotonic_s is not None:
            return "local_monotonic_clock"
        return None

    @property
    def test_case_code(self) -> str | None:
        value = self._find_value(
            self._exit_response,
            (
                "test_case_code",
                "testCaseCode",
                "case_code",
                "caseCode",
                "test_case_id",
                "testCaseId",
                "case_id",
                "caseId",
            ),
        )
        if value in (None, ""):
            value = self._find_value(
                self._enter_response,
                (
                    "test_case_code",
                    "testCaseCode",
                    "case_code",
                    "caseCode",
                    "test_case_id",
                    "testCaseId",
                    "case_id",
                    "caseId",
                ),
            )
        return None if value in (None, "") else str(value)

    @property
    def reported_source_count(self) -> int | None:
        value = self._find_value(
            self._exit_response,
            (
                "source_count",
                "sourceCount",
                "total_source_count",
                "totalSourceCount",
                "interference_source_count",
                "interferenceSourceCount",
            ),
        )
        if value in (None, ""):
            value = self._find_value(
                self._enter_response,
                (
                    "source_count",
                    "sourceCount",
                    "total_source_count",
                    "totalSourceCount",
                    "interference_source_count",
                    "interferenceSourceCount",
                ),
            )
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None


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
    runtime = getattr(robot, "program_runtime_s", None)
    runtime_source = getattr(robot, "program_runtime_source", None)
    print(f"程序运行时间：{'未取得' if runtime is None else f'{float(runtime):.3f} s'}"
          f"{'' if runtime_source is None else f'（{runtime_source}）'}")
    print(f"动作日志：{action_log}")
    print(f"通信日志：{robot.log_path}")


def collect_post_run_metadata(
    args: argparse.Namespace,
    robot: RuntimeRecorder,
    summary: dict,
) -> tuple[str | None, int | None]:
    """在 /exit 之后补录界面字段，补录耗时不计入程序运行时间。"""
    case_code = args.case_code or robot.test_case_code
    source_count = args.source_count
    if source_count is None:
        source_count = robot.reported_source_count

    may_prompt = (
        not args.no_post_run_prompt
        and sys.stdin.isatty()
        and bool(robot.enter_response)
        and robot.exit_reason is not None
    )
    if may_prompt and not case_code:
        print("测试已经结束，以下录入不计入程序运行时间。")
        try:
            entered = input("请输入模拟器界面显示的测试案例编码（可留空）：").strip()
        except EOFError:
            entered = ""
        case_code = entered or None
    if may_prompt and source_count is None and not summary.get("all_resolved"):
        try:
            entered = input("请输入该案例干扰源总数（可留空）：").strip()
        except EOFError:
            entered = ""
        if entered:
            try:
                source_count = int(entered)
            except ValueError:
                print("干扰源总数不是整数，已留空。")
    return case_code, source_count


def build_required_result(
    summary: dict,
    robot: RuntimeRecorder,
    case_code: str | None,
    source_count: int | None,
) -> dict[str, object]:
    cleared = int(summary.get("cleared", 0) or 0)
    virtual_time_s = (
        float(summary["virtual_time_s"])
        if summary.get("virtual_time_s") is not None
        else None
    )
    average_s = (
        virtual_time_s / cleared
        if virtual_time_s is not None and cleared
        else None
    )
    if source_count is None and summary.get("all_resolved"):
        source_count = cleared
    ratio = cleared / source_count if source_count and source_count > 0 else None
    return {
        "test_case_code": case_code,
        "cleared_source_count": cleared,
        "average_localization_clear_time_s": average_s,
        "program_runtime_s": robot.program_runtime_s,
        "program_runtime_source": robot.program_runtime_source,
        "virtual_total_time_s": virtual_time_s,
        "reported_or_certified_source_count": source_count,
        "cleared_source_ratio": ratio,
        "exit_reason": robot.exit_reason,
    }


def write_required_result(out: Path, required: dict[str, object]) -> None:
    """同时写机器可读 JSON 和可直接合并进论文表 1 的单行 CSV。"""
    (out / "table1_row.json").write_text(
        json.dumps(required, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    headers = [
        "测试案例编码",
        "清除干扰源个数",
        "平均定位清除时间(秒)",
        "程序运行时间(秒)",
    ]
    row = {
        headers[0]: required["test_case_code"],
        headers[1]: required["cleared_source_count"],
        headers[2]: required["average_localization_clear_time_s"],
        headers[3]: required["program_runtime_s"],
    }
    with (out / "table1_row.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def print_required_result(required: dict[str, object]) -> None:
    print("-" * 64)
    print("论文表 1 所需结果")
    print("测试类型：联网测试（演练/正式由模拟器界面决定）")
    print(
        "测试案例编码："
        f"{required['test_case_code'] or '未取得，请从模拟器界面抄录'}"
    )
    print(f"清除干扰源个数：{required['cleared_source_count']}")
    average = required["average_localization_clear_time_s"]
    runtime = required["program_runtime_s"]
    print(f"平均定位清除时间：{'未定义' if average is None else f'{float(average):.2f} s'}")
    print(f"程序运行时间：{'未取得' if runtime is None else f'{float(runtime):.3f} s'}")
    ratio = required["cleared_source_ratio"]
    if ratio is not None:
        print(f"被清除干扰源比例：{float(ratio):.6f}")
    print("若本次为正式测试：还必须从模拟器导出官方加密行为日志，并保持原文件名。")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.check:
        return check_dependencies()

    practice_guard(args)

    strategy_class, strategy_types = build_strategy()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = args.output_dir or (_HERE / "q3_logs" / f"practice_{stamp}")
    out.mkdir(parents=True, exist_ok=False)

    # 优先用内嵌模型自带的 RobotClient，保证与策略同源。
    robot_class = strategy_types[0] if isinstance(strategy_types[0], type) else None
    if robot_class is None:  # pragma: no cover - 兜底
        from client import RobotClient as robot_class
    inner_robot = robot_class(
        args.robot_id,
        base_url=args.base_url,
        log_dir=str(out / "http"),
        verbose=args.verbose,
    )
    robot = RuntimeRecorder(inner_robot)
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

    case_code, source_count = collect_post_run_metadata(args, robot, summary)
    required = build_required_result(summary, robot, case_code, source_count)

    result: dict[str, object] = {
        "mode": "practice",
        "strategy": strategy_class.__name__,
        "robot_client": robot_class.__name__,
        "hex_cover": True,
        "program_runtime_s": robot.program_runtime_s,
        "program_runtime_source": robot.program_runtime_source,
        "enter_response": robot.enter_response,
        "exit_response": robot.exit_response,
        **summary,
        "document_required_result": required,
    }
    if error is not None:
        result["error"] = error
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_required_result(out, required)

    if error is not None:
        print(f"运行失败：{error}")
    action_log = getattr(getattr(strategy, "logger", None), "path", None)
    if summary:
        print_summary(summary, action_log, robot)
    print_required_result(required)
    print(f"表1 CSV：{out / 'table1_row.csv'}")
    print(f"表1 JSON：{out / 'table1_row.json'}")
    print(f"输出目录：{out}")
    return 0 if summary.get("all_resolved") else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
