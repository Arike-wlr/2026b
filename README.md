# CUMCM 2026 B 题 · 机器狗通信层

无线电干扰源环境模拟器的机器狗端代码。目前只实现**通信层**（与模拟器的 HTTP+JSON 交互、状态维护、日志），
搜索与清除策略在这一层之上编写，不需要再关心 HTTP 细节。

协议依据：`附件2.docx`《模拟器通信接口说明及编程指南》；
规则依据：`B题.pdf` 正文与附录、`附件1.docx`《模拟器使用说明》。

---

## 1. 目录结构

```
d:/CUMCMB/
├── client.py                  ★ 主要代码：通信客户端 RobotClient（协议层，核心）
├── problem 1~4/               ★ 主要代码：四个问题
├── B题.pdf                     题目正文（问题 1-4、附录 1-4）
├── 先读下载说明.pdf             模拟器下载地址与提取码
├── 附件/
│   ├── 附件1.docx              模拟器使用说明（动作规则、计时规则、操作指南）
│   └── 附件2.docx              通信接口说明及编程指南（字段、状态码、示例程序）
├── test/                       测试目录（只放测试脚本，不参与正式运行）
│   ├── smoke_test.py           冒烟测试：跑通文档第 10 节的 6 步计时示例
│   ├── enter_exit.py           早期脚本：只测 /enter + /exit（已被 smoke_test.py 取代）
│   └── test_all_interfaces.py  早期脚本：一次性跑完 4 条指令（已被 smoke_test.py 取代）
├── logs/                       client.py 直接运行时的日志目录（robot_年月日_时分秒.txt）
├── problem3/q3_logs/           问题3 每次联网运行的完整归档（含表1结果）
└── problem4/q4_logs/           问题4 每次联网运行的完整归档（含表1结果）
```

约定：**主要代码一律放在项目根目录**，`test/` 只放放行前的测试脚本，正式测试运行时不依赖它。
测试脚本通过 `sys.path.insert(0, 项目根目录)` 导入根目录下的模块，所以在任意工作目录下都能跑。

`test/enter_exit.py` 与 `test/test_all_interfaces.py` 保留仅作对照，它们暴露的问题已在 `client.py`
中修正（`request_id` 会重复导致 409、没有维护测向机当前频道、没有解析业务结果、没有使用
`remaining_real_duration_s`），**新代码不要用它们**。

---

## 2. 快速开始

### 2.1 前置条件

1. 模拟器已启动并完成**在线登录**（队号 + 队员 1 信息 + 密码）；
2. 在模拟器中选择「问题 3 演练测试」或「问题 4 演练测试」，点确认开始；
3. 等待数据准备 → **5 秒倒计时结束**，界面提示机器狗接口就绪。

> 倒计时期间、非测试期间、测试结束后接口都不开放，此时请求会直接连接失败。
> 日志里出现 `10053 / 10054 Connection aborted/reset` 基本都是这个原因，不是代码问题。

### 2.2 跑冒烟测试

```powershell
cd d:/CUMCMB
python test/smoke_test.py 202610038037   # 参数是你们的参赛队号
```

正常输出：

```
/enter 成功，本局可用现实时间 1200 秒
日志文件：d:/CUMCMB/logs/robot_20260910_194500.txt

步骤  指令                      实际虚拟时刻    预期虚拟时刻        结果
----------------------------------------------------------------------
1    /enter                           0.000             0.000   - / OK
2    /measure (300,400) ch1         105.000           105.000   no_signal / OK
3    /measure (300,400) ch2         111.000           111.000   no_signal / OK
4    /clear (300,0) ch3             194.000           194.000   no_target_in_range / OK
5    /measure (300,0) ch2           199.000           199.000   no_signal / OK
6    /exit                          199.000           199.000   user_exit / OK
----------------------------------------------------------------------
全部一致，通信层工作正常。
```

`measure_result` 是什么无所谓（案例随机），关键是**虚拟时刻对得上**，说明移动耗时、
切换频道耗时的计算与模拟器一致。

---

## 3. 协议速查

### 3.1 四条指令

全部 `POST` + JSON，`Content-Type: application/json`，本机 `http://127.0.0.1:2026`。
公共请求字段：`arena_id`（固定 `"default"`）、`robot_id`（参赛队号）、`request_id`（幂等键）。

| 指令 | 额外请求字段 | 响应关键字段 | 推进虚拟时钟 |
| --- | --- | --- | --- |
| `/enter` | — | `max_virtual_duration_s`、`max_real_duration_s`、`remaining_real_duration_s` | 否 |
| `/measure` | `position{x,y}`、`channel` | `measure_result`、`svd_deg`（仅 direction 时） | 是 |
| `/clear` | `position{x,y}`、`channel` | `clear_result` | 是 |
| `/exit` | — | `exit_reason` | 否 |

- `measure_result`：`"direction"`（有信号，附 `svd_deg`）／`"near"`（≤5 米，无示向度）／`"no_signal"`
- `clear_result`：`"success"`（20 米内清除了）／`"no_target_in_range"`
- `exit_reason`：`"user_exit"`

所有响应都带 `accepted`、`real_timestamp_ms`、`virtual_time_s`。

### 3.2 虚拟耗时

| 动作 | 耗时 |
| --- | --- |
| 移动 | 直线距离 ÷ 5（m/s） |
| 切换测向机频道 | 1 秒（**只有 `/measure` 会触发**） |
| 检测 | 5 秒 |
| 精确定位未发现目标 | 3 秒 |
| 精确定位 + 激光清除 | 5 秒 |

```
/measure 总耗时 = 移动 + (频道变了 ? 1 : 0) + 5
/clear   总耗时 = 移动 + (清除成功 ? 5 : 3)        # 不产生切换频道耗时
```

### 3.3 关键物理参数

| 参数 | 值 |
| --- | --- |
| 目标区域 | 半径 1800 米的圆，圆心为原点，x 轴向东，y 轴向北 |
| 干扰源数量 | 10 ~ 16 个（不通过接口返回） |
| 频道 | 1 ~ 20 的整数，每个频道最多一个干扰源 |
| 有效接收半径 | 1000 ~ 1500 米（每个干扰源不同，不返回） |
| 定向干扰源覆盖角 | 定向方向两侧各 90°（含边界） |
| 示向度误差 | ±1° |
| 近距离阈值 | 5 米（≤5 米返回 `near`） |
| 清除半径 | 20 米 |
| 坐标上限 | 每个分量绝对值 ≤ 2,000,000 |

### 3.4 时间限制

- `/enter` 成功后开始计**程序运行时间，上限 20 分钟**；
- 测试窗口 **25 分钟**，两者取较早者；
- **`remaining_real_duration_s` 是本局真实可用秒数（0~1200），必须用它做倒计时**，
  不能假定固定 1200 秒（晚于窗口开始 5 分钟进场时实际不足 20 分钟）；
- 虚拟世界上限 100 小时（360000 秒）；
- 检测增加的 5 秒是虚拟时间，**现实中不用等**，串行合法请求没有速率上限，但不得并发。

---

## 4. `client.py` 使用说明

### 4.1 基本用法

```python
from client import RobotClient, RequestRejected, TransportError

with RobotClient("202610038037") as robot:
    enter = robot.enter()
    print("可用现实时间：", enter["remaining_real_duration_s"])

    res = robot.measure(300, 400, 1)
    if res.has_signal:
        print("示向度：", res.svd_deg)      # 只有 direction 才有值
    elif res.is_near:
        print("距离过近，可直接清除")

    cleared = robot.clear(300, 0, 3)
    print("清除成功" if cleared.cleared else "附近没有目标")

    print("退出原因：", robot.exit())
```

### 4.2 API

| 成员 | 说明 |
| --- | --- |
| `robot.enter()` | 进入并返回完整响应（含 `remaining_real_duration_s`） |
| `robot.measure(x, y, channel)` | 返回 `MeasureResult(result, svd_deg, position, channel, virtual_time_s)` |
| `robot.clear(x, y, channel)` | 返回 `ClearResult(cleared, position, channel, virtual_time_s)` |
| `robot.exit()` | 返回 `exit_reason` |
| `robot.current_position` | 当前位置，初始 `(0.0, 0.0)` |
| `robot.current_channel` | 测向机当前频道，初始 1，**只被 `/measure` 更新** |
| `robot.virtual_time_s` | 当前虚拟时刻 |
| `robot.remaining_real_s` | 剩余现实时间（秒） |
| `robot.is_time_up(margin_s=0)` | 现实或虚拟时间是否即将耗尽，可留安全余量 |
| `robot.estimate_move_time(x, y)` | 预估移动耗时 |
| `robot.estimate_measure_cost(x, y, ch)` | 预估一次检测的虚拟耗时（含切频道判断） |
| `robot.estimate_clear_cost(x, y, found)` | 预估一次清除的虚拟耗时 |

### 4.3 异常

| 异常 | 含义 | 该怎么处理 |
| --- | --- | --- |
| `RequestRejected` | HTTP 200 但 `accepted=false`，动作未生效、时钟未推进 | 检查字段拼写/队号；这类请求不占 `request_id`，修正后可重发 |
| `HttpError` | 400/404/405/409/413/415 等非 200 | 400 多为字段或范围错误；409 是 `request_id` 复用冲突 |
| `TransportError` | 重试耗尽仍无合法响应（接口未开放/测试已结束/网络故障） | 等待或终止本轮 |
| `NotInSessionError` | 未 `enter()` 就发动作 | 流程错误 |

### 4.4 已处理的坑

1. **`request_id` 唯一**：内部自增序号生成 `enter-1` / `measure-7` / `clear-3`，
   不会重复（重复会导致 HTTP 409）；只有重试时才复用原 ID 与原请求内容——这正是协议要求的。
2. **`/clear` 不切换频道**：`current_channel` 只由成功的 `/measure` 更新，
   预估耗时与调度都依赖这一点。
3. **429 / 500 自动重试**：用同一 `request_id` 重试是幂等安全的；其余 4xx 不重试直接抛错。
4. **本地校验先行**：坐标非有限值或超范围、频道非 1~20 整数，直接抛 `ValueError`，
   不发无效请求（避免浪费一次 HTTP 400）。
5. **`accepted=false` 不污染时钟**：此时不更新 `virtual_time_s`、位置和频道。
6. **日志落盘**：`client.py` 每次运行在项目根 `logs/` 下生成独立日志文件，记录全部请求与响应
   （规范要求机器狗自行记录，模拟器界面只显示最新 1000 条）；问题3/问题4 的入口再各自建一份
   “一次测试一份归档”的目录 `problem3/q3_logs/run_<时间戳>/`、
   `problem4/q4_logs/run_<时间戳>/`，内含
   `http/robot_*.txt`（通信层原始记录）、`actions.tsv`（结构化动作日志：序号/虚拟时刻/阶段/动作/
   坐标/频道/结果/可行域半径/备注）、`result.json`（汇总指标 + 异常），以及可直接汇入论文表 1
   的 `table1_row.csv` / `table1_row.json`；演练与正式采用同一保存逻辑，失败也会落盘。

---

## 5. 写策略时的建议

策略代码放在项目根目录下的 `strategy.py`，由 `main.py` 作为入口启动，只调用第 4 节的语义方法：

```python
while not robot.is_time_up(margin_s=30):     # 留 30 秒收尾
    下一步去哪 / 测哪个频道 / 是否清除 ...
    robot.measure(...) 或 robot.clear(...)
robot.exit()                                  # 无论如何收尾时主动退出
```

要点：

- 用 `robot.remaining_real_s` 做现实倒计时，不要数 HTTP 次数；
- `no_signal` ≠ 附近没有干扰源，也可能是超出接收半径或不在定向覆盖范围内；
- 单次 `svd_deg` 有 ±1° 误差，不要当精确方位用，需要交会定位（题目问题 1/2）；
- 规划路线时优先把**同频道的检测排在一起**，能省掉 1 秒/次的切换开销；
- 清除半径 20 米，可以先测到 `near`（≤5 米）或交会定位收敛后再去清，减少空跑；
- 日志大小有 2 MB 上传上限，避免异常高频循环。

---

## 6. 常见问题

| 现象 | 原因 |
| --- | --- |
| 日志里 `10053 / 10054 Connection aborted/reset` | 倒计时未结束、接口未开放或测试已结束，模拟器直接关闭连接，属正常现象 |
| `/enter` 返回 `accepted=false` | `robot_id` 与当前登录队号不一致 / 重复调用 `/enter` / 有未声明字段 |
| HTTP 409 | 同一 `request_id` 对应了不同动作，或并发发送了不同动作——必须串行 |
| HTTP 400 | JSON 语法错、缺字段、坐标非有限值或超范围、频道不是 1~20 整数 |
| 响应里 `virtual_time_s` 是 0 | 该请求 `accepted=false`，0 不是当前虚拟时刻 |
| 检测很慢 | 每次 5 秒是虚拟时间，现实中不需要 sleep，不要自己加等待 |
