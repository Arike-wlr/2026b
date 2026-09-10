"""冒烟测试（放行前自检）：用 RobotClient 跑一遍附件 2 第 10 节的 6 步计时示例。

用途：
    1. 确认模拟器已登录、测试已开始、机器狗接口已就绪（倒计时结束之后）；
    2. 确认 robot_id 与当前登录参赛队号一致；
    3. 校验本地维护的虚拟时钟与模拟器返回的 virtual_time_s 一致。

预期结果（文档表 13）：
    步骤  指令              位置        频道   虚拟时刻
    1     /enter            -           -      0
    2     /measure          (300,400)   1      105   = 500/5 + 0 + 5
    3     /measure          (300,400)   2      111   = 0 + 1 + 5
    4     /clear            (300,0)     3      194   = 400/5 + 3
    5     /measure          (300,0)     2      199   = 0 + 0 + 5
    6     /exit             -           -      199

用法（在任意目录下均可，脚本会自动把项目根目录加入模块搜索路径）：
    python test/smoke_test.py                  # 使用默认队号
    python test/smoke_test.py 202610038037     # 指定参赛队号
"""

import os
import sys

# 让测试脚本能导入项目根目录下的 client.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import (  # noqa: E402
    HttpError,
    NotInSessionError,
    RequestRejected,
    RobotClient,
    TransportError,
)

sys.stdout.reconfigure(encoding="utf-8")

ROBOT_ID = sys.argv[1] if len(sys.argv) > 1 else "202610038037"
BASE_URL = "http://127.0.0.1:2026"

# 每一步执行完之后的预期虚拟时刻
EXPECTED_VIRTUAL_TIME = [0, 105, 111, 194, 199, 199]


def main():
    robot = RobotClient(ROBOT_ID, base_url=BASE_URL, verbose=False)
    steps = []

    def record(name, virtual_time, detail):
        steps.append((name, virtual_time, detail))

    try:
        enter = robot.enter()
        print(f"/enter 成功，本局可用现实时间 {enter['remaining_real_duration_s']} 秒")
        print(f"日志文件：{robot.log_path}\n")
        record("/enter", robot.virtual_time_s, "-")

        # 步骤 2：从 (0,0) 到 (300,400)，距离 500 米；频道 1 与初始频道相同，不切换
        res = robot.measure(300, 400, 1)
        record("/measure (300,400) ch1", robot.virtual_time_s, res.result)

        # 步骤 3：原地不动，频道 1 -> 2，产生 1 秒切换耗时
        res = robot.measure(300, 400, 2)
        record("/measure (300,400) ch2", robot.virtual_time_s, res.result)

        # 步骤 4：移动到 (300,0)，距离 400 米；ch3 是目标频道，不切测向机频道
        cleared = robot.clear(300, 0, 3)
        record("/clear (300,0) ch3", robot.virtual_time_s,
               "success" if cleared.cleared else "no_target_in_range")

        # 步骤 5：原地对频道 2 检测，测向机仍停留在步骤 3 的频道 2，无切换耗时
        res = robot.measure(300, 0, 2)
        record("/measure (300,0) ch2", robot.virtual_time_s, res.result)

        reason = robot.exit()
        record("/exit", robot.virtual_time_s, reason)
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

    print(f"{'步骤':<4}{'指令':<24}{'实际虚拟时刻':>14}{'预期虚拟时刻':>14}{'结果':>12}")
    print("-" * 70)
    ok = True
    for index, (name, virtual_time, detail) in enumerate(steps):
        expected = EXPECTED_VIRTUAL_TIME[index]
        match = abs(virtual_time - expected) < 1e-6
        ok = ok and match
        flag = "OK" if match else "不一致"
        print(f"{index + 1:<4}{name:<24}{virtual_time:>14.3f}{expected:>14.3f}{detail + ' / ' + flag:>12}")

    print("-" * 70)
    print("全部一致，通信层工作正常。" if ok else "存在不一致，请检查日志与模拟器设置。")


if __name__ == "__main__":
    main()
