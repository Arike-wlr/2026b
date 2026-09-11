"""问题 3 正式入口：把 ``client.RobotClient`` 与 ``Problem3Strategy`` 串起来。

队号等参数像 ``test/`` 下的脚本一样直接写在文件顶部，直接运行即可::

    python problem3/problem3_main.py            # 使用下面硬编码的 ROBOT_ID
    python problem3/problem3_main.py --verbose  # 实时打印动作日志

也可以临时用命令行覆盖（可选）::

    python problem3/problem3_main.py 202610038037 --base-url http://127.0.0.1:2026

前置条件：模拟器已登录、已开始“问题 3 演练测试/正式测试”、5 秒倒计时结束、接口就绪。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# 让脚本既能 `python problem3/problem3_main.py` 直接运行，也能被导入
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from client import (  # noqa: E402
    DEFAULT_ARENA_ID,
    DEFAULT_BASE_URL,
    HttpError,
    NotInSessionError,
    RequestRejected,
    RobotClient,
    TransportError,
)
from problem3_strategy import Problem3Strategy, StrategyConfig  # noqa: E402

# --------------------------------------------------------------------------- #
# 运行参数（与 test/ 下的脚本一致，直接改这里即可）
# --------------------------------------------------------------------------- #
ROBOT_ID = "202610038037"          # 参赛队号，必须与当前登录的队号一致
BASE_URL = DEFAULT_BASE_URL        # 模拟器地址 http://127.0.0.1:2026
ARENA_ID = DEFAULT_ARENA_ID        # 场地编号 default
TIME_MARGIN_S = 30.0               # 现实时间安全余量（秒）


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问题3 机器狗自动搜索定位清除")
    parser.add_argument("robot_id", nargs="?", default=None,
                        help=f"参赛队号（默认使用文件内的 ROBOT_ID={ROBOT_ID}）")
    parser.add_argument("--base-url", default=BASE_URL, help="模拟器地址")
    parser.add_argument("--arena-id", default=ARENA_ID, help="场地编号")
    parser.add_argument("--verbose", action="store_true", help="实时打印动作日志")
    parser.add_argument("--time-margin", type=float, default=TIME_MARGIN_S,
                        help="现实时间安全余量（秒）")
    parser.add_argument("--action-log", default=None,
                        help="结构化动作日志路径（默认 logs/problem3_actions_时间戳.txt）")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    robot_id = args.robot_id or ROBOT_ID

    log_dir = os.path.join(_ROOT, "logs")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    action_log = args.action_log or os.path.join(log_dir, f"problem3_actions_{stamp}.txt")

    print(f"使用队号 {robot_id}，模拟器 {args.base_url}，场地 {args.arena_id}")
    robot = RobotClient(
        robot_id,
        base_url=args.base_url,
        arena_id=args.arena_id,
        verbose=False,
    )
    cfg = StrategyConfig(verbose=args.verbose, log_path=action_log,
                         time_margin_s=args.time_margin)
    strategy = Problem3Strategy(robot, cfg)

    try:
        summary = strategy.run()
    except RequestRejected as error:
        print(f"请求被模拟器拒绝（动作未生效，虚拟时钟未推进）：{error}")
        return
    except HttpError as error:
        print(f"HTTP 错误：{error}")
        return
    except TransportError as error:
        print(f"连不上模拟器——请确认已登录、已开始测试、5 秒倒计时已结束：{error}")
        return
    except NotInSessionError as error:
        print(f"会话状态错误：{error}")
        return
    finally:
        robot.close()

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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
