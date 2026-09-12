"""问题 4 正式入口：把 ``client.RobotClient`` 与 ``Problem4Strategy`` 串起来。

队号等参数与 problem3 一致，直接写在文件顶部，直接运行即可::

    python problem4/problem4_main.py
    python problem4/problem4_main.py --verbose        # 打印通信请求/响应

也可以临时用命令行覆盖（可选）::

    python problem4/problem4_main.py 202610038037 --base-url http://127.0.0.1:2026

前置条件：模拟器已登录、已开始"问题 4 演练测试/正式测试"、5 秒倒计时结束、接口就绪。
"""

from __future__ import annotations

import argparse
import os
import sys

# 让脚本既能 `python problem4/problem4_main.py` 直接运行，也能被导入
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_P3 = os.path.join(_ROOT, "problem3")
for _path in (_ROOT, _HERE, _P3):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from client import (  # noqa: E402
    DEFAULT_ARENA_ID,
    DEFAULT_BASE_URL,
    HttpError,
    NotInSessionError,
    RequestRejected,
    RobotClient,
    TransportError,
)
from problem4_directional_simulation import Problem4Strategy  # noqa: E402

# --------------------------------------------------------------------------- #
# 运行参数（与 problem3 一致，直接改这里即可）
# --------------------------------------------------------------------------- #
ROBOT_ID = "202610038037"          # 参赛队号，必须与当前登录的队号一致
BASE_URL = DEFAULT_BASE_URL        # 模拟器地址 http://127.0.0.1:2026
ARENA_ID = DEFAULT_ARENA_ID        # 场地编号 default
TIME_MARGIN_S = 30.0        # 现实时间安全余量（秒）


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问题4 定向干扰源自动搜索定位清除")
    parser.add_argument("robot_id", nargs="?", default=None,
                        help=f"参赛队号（默认使用文件内的 ROBOT_ID={ROBOT_ID}）")
    parser.add_argument("--base-url", default=BASE_URL, help="模拟器地址")
    parser.add_argument("--arena-id", default=ARENA_ID, help="场地编号")
    parser.add_argument("--verbose", action="store_true",
                        help="打印通信层请求/响应日志")
    parser.add_argument("--time-margin", type=float, default=TIME_MARGIN_S,
                        help="现实时间安全余量（秒）")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    robot_id = args.robot_id or ROBOT_ID

    print(f"使用队号 {robot_id}，模拟器 {args.base_url}，场地 {args.arena_id}")
    robot = RobotClient(
        robot_id,
        base_url=args.base_url,
        arena_id=args.arena_id,
        verbose=args.verbose,
    )
    strategy = Problem4Strategy(robot, time_margin_s=args.time_margin)

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

    stats = summary["stats"]
    total_time_s = float(summary["virtual_time_s"])
    survey_time_s = float(summary["survey_time_s"])
    measured = int(summary["detected"])
    cleared = int(summary["cleared"])
    print("=" * 64)
    print("问题4 运行结果")
    print("=" * 64)
    print(
        f"巡检点：实际访问 {summary['survey_station_count']}/"
        f"{summary['survey_planned_station_count']} 个"
        "（圆心 + 内外两圈各 12 点）"
    )
    print(f"巡检-清除联合滚动调度：{summary['dynamic_survey_route']}")
    print(f"干扰源：检测到 {measured} 个，已清除 {cleared} 个")
    directional = list(summary["confirmed_directional_channels"])
    others = list(summary["not_confirmed_directional_channels"])
    print(f"定向干扰源（已确认）：{len(directional)} 个 {directional}")
    print(f"全向或未被判定为定向：{len(others)} 个 {others}")
    print(f"全部清除：{summary['complete']}")
    if summary["incomplete_reason"]:
        print(f"未完成原因：{summary['incomplete_reason']}")
    print(f"总虚拟时间：{total_time_s / 60.0:.2f} min （{total_time_s:.1f} s）")
    if cleared > 0:
        print(f"平均单源定位清除时间：{total_time_s / cleared:.2f} s")
    else:
        print("平均单源定位清除时间：无（本局没有清除任何干扰源）")
    print(f"巡检结束时刻：{survey_time_s / 60.0:.2f} min")
    # print(f"收尾 TSP 计入归原点代价：{summary.get('route_end_at_origin', True)}")
    print(f"检测 {int(float(stats['measures']))} 次，"
          f"清除 {int(float(stats['clears']))} 次"
          f"（成功 {int(float(stats['clear_success']))} 次），"
          f"补测 {int(float(stats['probes']))} 次")
    print(f"巡检途中复测 {int(float(stats.get('survey_remeasures', 0.0)))} 次，"
          f"巡检边插入清除 {int(float(stats.get('survey_inserted_clears', 0.0)))} 次")
    print(f"已收敛频道跳过巡检复测 {int(float(stats.get('survey_localized_skips', 0.0)))} 次")
    # print(f"收尾阶段动态融合复测 {int(float(stats.get('finish_shared_remeasures', 0.0)))} 次")
    # print(f"定向遮挡约束 {int(float(stats.get('directional_constraints', 0.0)))} 条，"
    #       f"补测排序调整 {int(float(stats.get('directional_probe_reorders', 0.0)))} 次")
    # print(f"弱空频道：标记 {int(float(stats.get('weak_unknown_marked', 0.0)))} 个，"
    #       f"巡检跳过 {int(float(stats.get('weak_unknown_survey_skips', 0.0)))} 次，"
    #       f"收尾复核 {int(float(stats.get('weak_unknown_final_probes', 0.0)))} 次，"
    #       f"复核找回 {int(float(stats.get('weak_unknown_recovered', 0.0)))} 个")
    # print(f"定向遮挡导致的无信号：{summary['probe_no_signal_count']} 次（补测阶段）")
    print(f"网格兜底：{summary['grid_fallback_count']} 次，"
          f"最终可行域半径上界：{float(summary['max_final_radius_m']):.2f} m")
    print(f"通信日志：{robot.log_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
