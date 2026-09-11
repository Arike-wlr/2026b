# 问题三：机器人狗搜索、定位与清除算法编程规格

## 1. 文档目的

本文档把问题三的推荐决策方案整理为可直接编程的流程。算法采用：

> 七点确定性覆盖巡检 + 多点测向可行域 + 批量路径清除 + 小区域覆盖兜底。

优化目标采用字典序：

1. 首先保证所有合法情况下均不漏检，并最终清除全部干扰源；
2. 在满足第一项目标的策略中，尽量减小面积均匀分布下的平均虚拟时间。

均匀面积分布只用于决定访问顺序，不能作为提前结束的依据。

---

## 2. 题面常量

建议在程序开头集中定义以下常量，不要在函数内部使用无含义的数字。

| 常量名 | 数值 | 单位 | 含义 |
|---|---:|---|---|
| `TARGET_RADIUS` | 1800 | m | 目标圆域半径 |
| `MIN_RECEIVE_RADIUS` | 1000 | m | 干扰源最小有效接收半径 |
| `MAX_RECEIVE_RADIUS` | 1500 | m | 干扰源最大有效接收半径 |
| `NEAR_RADIUS` | 5 | m | 返回 `near` 的距离阈值 |
| `CLEAR_RADIUS` | 20 | m | 清除半径 |
| `BEARING_ERROR` | 1 | degree | 示向度最大绝对误差 |
| `MOVE_SPEED` | 5 | m/s | 机器人移动速度 |
| `MEASURE_TIME` | 5 | s | 单次检测时间 |
| `SWITCH_TIME` | 1 | s | 切换检测频道时间 |
| `CLEAR_FAIL_TIME` | 3 | s | 清除失败操作时间 |
| `CLEAR_SUCCESS_TIME` | 5 | s | 清除成功操作时间 |
| `CHANNEL_MIN` | 1 | — | 最小频道编号 |
| `CHANNEL_MAX` | 20 | — | 最大频道编号 |
| `SOURCE_COUNT_MIN` | 10 | — | 干扰源数量下界 |
| `SOURCE_COUNT_MAX` | 16 | — | 干扰源数量上界 |
| `SURVEY_RING_RADIUS` | 1150 | m | 六边形巡检点半径 |
| `SAFE_REGION_RADIUS` | 58 | m | 可使用“两次清除法”的可行域半径上界 |

角度统一在内部使用弧度，只有接口输入、输出和日志显示时转换为度。

---

## 3. 基本数据结构

### 3.1 机器人状态

```text
RobotState:
    position: (x, y)
    current_channel: int
    virtual_time: float
    request_counter: int
```

初始状态：

```text
position = (0, 0)
current_channel = 1
virtual_time = 0
```

每次收到 `accepted=true` 的响应后，必须用响应中的 `virtual_time_s` 更新时钟。

### 3.2 频道状态

```text
ChannelState:
    channel: int
    status: UNKNOWN | DETECTED | CLEARED | EMPTY_CERTIFIED
    observations: list[Observation]
    completed_survey_points: set[int]
    feasible_region: Region
    enclosing_center: (x, y) | None
    enclosing_radius: float | None
```

状态含义：

- `UNKNOWN`：尚未收到该频道的信号，也没有完成全域排除；
- `DETECTED`：至少收到一次 `direction` 或 `near`；
- `CLEARED`：接口已经返回 `clear_result="success"`；
- `EMPTY_CERTIFIED`：七个覆盖点均已检测且始终没有发现该频道信号。

### 3.3 单次观测

```text
Observation:
    position: (x, y)
    channel: int
    result: direction | near | no_signal
    bearing: float | None
    virtual_time: float
```

不要把 `no_signal` 直接解释为频道为空。

---

## 4. 七点确定性巡检

### 4.1 巡检点坐标

中心点：

```text
P0 = (0, 0)
```

外围六点：

```text
P1 = ( 1150,       0)
P2 = (  575,  995.929)
P3 = ( -575,  995.929)
P4 = (-1150,       0)
P5 = ( -575, -995.929)
P6 = (  575, -995.929)
```

程序中应使用三角函数生成坐标，不要直接使用上面的三位小数近似值：

```text
Pk = 1150 * (cos(k*pi/3), sin(k*pi/3)), k = 0, 1, ..., 5
```

其中外围点的下标可根据程序习惯重新编号。

### 4.2 覆盖保证

目标点位于相邻两个外围检测方向的中间且位于目标圆边界时最不利。它到最近外围检测点的距离为：

```text
sqrt(1800^2 + 1150^2 - 2*1800*1150*cos(30 degree))
= 988.511 m < 1000 m
```

外围六点不能覆盖原点附近区域，所以中心点不能删除。中心点和外围六点合在一起才能保证整个目标圆域均处于至少一个1000 m接收圆内。

### 4.3 频道扫描顺序

为减少频道切换：

```text
P0:  1, 2, ..., 20
P1: 20, 19, ..., 1
P2:  1, 2, ..., 20
P3: 20, 19, ..., 1
...
```

这样每个完整检测点只需要19次频道切换。

已经 `CLEARED` 的频道可以跳过。已经 `DETECTED` 但尚未清除的频道可以继续测量，以便利用巡检点免费获得更多交会方向。

### 4.4 单点扫描伪代码

```text
function scan_station(station_id, position, channel_order):
    for channel in channel_order:
        if state[channel].status == CLEARED:
            continue

        response = measure(position, channel)
        record(response)
        state[channel].completed_survey_points.add(station_id)

        if response.result == "direction":
            state[channel].status = DETECTED
            add_direction_constraint(channel, position, response.svd_deg)

        else if response.result == "near":
            state[channel].status = DETECTED
            clear_response = clear(position, channel)
            if clear_response.result == "success":
                state[channel].status = CLEARED
            else:
                raise UnexpectedSimulatorState

        else if response.result == "no_signal":
            add_safe_exclusion(channel, disk(position, 1000))
```

---

## 5. 可行域更新

对每个已经发现但尚未清除的频道维护可能位置集合 `F[channel]`。

初始集合为：

```text
F[channel] = disk((0, 0), 1800)
```

### 5.1 `direction` 更新

在位置 `S` 获得示向度 `theta` 后：

```text
F = F intersect bearing_wedge(S, theta - 1 degree, theta + 1 degree)
F = F intersect disk(S, 1500)
F = F outside disk(S, 5)
```

其中测向扇形必须是从检测点向前延伸的射线扇形，不能当成贯穿检测点的无限直线带。

### 5.2 `near` 更新

```text
F = F intersect disk(S, 5)
```

随后直接在 `S` 点调用清除接口。

### 5.3 `no_signal` 更新

由于实际接收半径未知，只能安全排除1000 m以内区域：

```text
F = F outside disk(S, 1000)
```

不能排除1000–1500 m之间的区域。

### 5.4 清除失败更新

若在 `C` 点清除失败，则：

```text
F = F outside disk(C, 20)
```

### 5.5 推荐的几何表示

为保持100%保证，不要用随机粒子或粗网格直接代替可行域。

建议使用以下任一实现：

1. 精确多边形与圆弧布尔运算；
2. 保守外包多边形：圆用外切正多边形表示，保证计算区域始终包含真实位置；
3. 将方向扇形写成两个半平面，使用凸多边形裁剪；
4. 对排除圆单独保存，最后计算覆盖与清除点时再处理非凸区域。

核心不变量是：

```text
只要干扰源尚未清除，它的真实位置必须始终属于程序保存的可行域。
```

---

## 6. 定位判据与清除方法

### 6.1 最小包围圆

对可行域计算最小包围圆：

```text
(C, R) = minimum_enclosing_circle(F)
```

其中 `C` 为圆心，`R` 为覆盖整个可行域的半径。

### 6.2 `R <= 20` 的情况

直接调用：

```text
clear(C, channel)
```

此时可以严格保证成功。

### 6.3 `20 < R <= 58` 的两次清除法

第一步，在 `C` 点尝试清除：

```text
result = clear(C, channel)
```

若成功则结束。若失败，则真实距离满足：

```text
20 < distance(C, source) <= R
```

随后在 `C` 点测向：

```text
result = measure(C, channel)
```

正常情况下应返回 `direction`。设测得方向为 `u`，移动长度取：

```text
L = (20 + R) / (2*cos(1 degree))
Q = C + L*u
clear(Q, channel)
```

当 `R <= 58` 时，目标到 `Q` 的最坏距离不超过19.02 m，因此第二次清除保证成功。

程序仍应检查接口返回值；若理论上必然成功的清除失败，应记录完整日志并终止当前策略，避免继续使用已经失真的状态。

---

## 7. 测向不足时的补测策略

### 7.1 第一组保证接收的补测点

取该频道第一次测向位置为 `S`，第一次示向单位向量为 `u`，令 `v` 为将 `u` 逆时针旋转90度得到的单位向量。

设置：

```text
Q_plus  = S + 750*u + 600*v
Q_minus = S + 750*u - 600*v
```

由于第一次测向误差不超过1度、目标距离不超过1500 m，真实目标到这两个点的最坏距离小于约978 m，因此补测点保证处于最小接收半径1000 m以内。

执行规则：

```text
probes = [Q_plus, Q_minus]
先访问距离机器人当前位置较近的点

for Q in probes:
    response = measure(Q, channel)

    if response.result == "near":
        clear(Q, channel)
        return CLEARED

    if response.result == "direction":
        update_feasible_region()
        (C, R) = minimum_enclosing_circle(F)
        if R <= 58:
            return READY_TO_CLEAR

    if response.result == "no_signal":
        记录为理论异常，并保留后续确定性兜底
```

### 7.2 自适应横向补测

若两次补测后仍有 `R > 58`：

1. 找到可行域最长方向 `e`；
2. 令 `n` 为 `e` 的垂直方向；
3. 在最小包围圆圆心两侧选择新检测点；
4. 保证新检测点到整个可行域的最大距离不超过1000 m；
5. 每次补测后重新计算可行域和最小包围圆。

参考规则：

```text
h = clamp(950 - R, 40, 250)
Q = C + sign*h*n
sign 在 +1 和 -1 之间交替

if R + h > 1000:
    Q = C
```

不要在完全相同的位置重复测向，因为同一位置的误差固定，重复测量不会提供新信息。

---

## 8. 最终确定性兜底

如果多次补测后仍不适合使用两次清除法，则用20 m清除圆覆盖剩余可行域。

选择网格间距：

```text
grid_spacing = 20*sqrt(2)*(1 - epsilon)
```

其中 `epsilon` 可取 `1e-3`。任意点到最近网格点的距离严格小于20 m。

执行步骤：

```text
1. 计算可行域包围盒；
2. 生成覆盖包围盒的全局方格点；
3. 只保留距离可行域不超过20 m的网格点；
4. 按逐行蛇形顺序排列网格点；
5. 从离机器人当前位置较近的一端开始；
6. 依次调用 clear(point, channel)；
7. 收到 success 后停止该频道的清除任务。
```

该步骤一般不会被调用，但它保证程序不会因为测向退化而失去确定性终止手段。

---

## 9. 批量路径规划

完成巡检后，先处理需要补测的频道，再统一清除已经获得安全包围圆的频道。

### 9.1 补测任务顺序

```text
while 存在 R > 58 的频道:
    对每个频道生成下一候选补测点
    选择距离机器人当前位置最近的候选点
    执行补测并更新该频道状态
```

### 9.2 清除任务顺序

以每个频道的安全包围圆圆心作为候选清除点：

```text
route = nearest_neighbor_route(current_position, clear_centers)
route = open_path_2opt(route)
```

使用开放路径，不要求机器人返回原点。

执行清除过程中，如果第一次圆心清除失败并产生新的测量、移动任务，应完成该频道的两次清除流程后，再继续访问路径中的下一个频道。

---

## 10. 完整主流程伪代码

```text
function main():
    enter_response = POST /enter
    assert enter_response.accepted == true

    initialize robot_state
    initialize channel_state[1..20]

    stations = build_seven_survey_points(radius=1150)

    # 阶段A：确定性全域巡检
    for station_id, station in stations:
        channel_order = ascending if station_id is even else descending
        scan_station(station_id, station, channel_order)

    # 完成空频道证书
    for channel in 1..20:
        if channel_state[channel].status == UNKNOWN:
            assert all seven survey points were measured
            channel_state[channel].status = EMPTY_CERTIFIED

    # 阶段B：补充定位
    for each channel with status == DETECTED:
        update enclosing circle

    while exists detected channel with enclosing_radius > 58:
        channel = choose_nearest_next_probe()
        localize_channel(channel)

    # 阶段C：批量清除
    route = optimize_open_clear_route()

    for channel in route:
        if channel_state[channel].status == CLEARED:
            continue

        if enclosing_radius <= 20:
            clear_at_enclosing_center(channel)
        else if enclosing_radius <= 58:
            bounded_region_two_clear(channel)
        else:
            grid_cover_clear(channel)

    # 阶段D：完成证书
    assert every channel is CLEARED or EMPTY_CERTIFIED

    POST /exit
```

若已经成功清除16个干扰源，可利用题目给出的数量上限直接结束，不再为剩余频道建立空频道证书。但正常七点批量策略一般会先完成巡检，再进入清除阶段。

---

## 11. HTTP接口实现要求

### 11.1 请求规则

- 所有请求串行发送；
- 必须收到上一次完整响应后才能发送下一次动作；
- 每个新动作使用新的 `request_id`；
- 只有网络超时或连接中断、需要重试完全相同的动作时，才能复用原请求内容和 `request_id`；
- 同一个 `request_id` 不得对应不同动作；
- 同时检查HTTP状态码和响应中的 `accepted`；
- `accepted=false` 时不能把返回的 `virtual_time_s=0` 当成当前虚拟时刻。

### 11.2 位置与频道状态更新

- 合法 `/measure` 会把机器人移动到 `position`，并将当前测向频道更新为 `channel`；
- 合法 `/clear` 会把机器人移动到 `position`，但不会改变当前测向频道；
- `/clear` 的 `channel` 是待清除干扰源频道，不产生频道切换耗时；
- 测试程序不需要真的等待5秒，虚拟耗时由模拟器直接累计。

### 11.3 推荐日志字段

每个动作至少记录：

```text
sequence_id
request_id
action
position_x
position_y
channel
accepted
measure_result / clear_result
svd_deg
virtual_time_s
channel_status_after_action
feasible_radius_after_action
```

正式测试前检查日志写入速度，避免异常高频写盘。

---

## 12. 正确性不变量

程序运行过程中应持续检查以下条件：

1. 每个频道最多对应一个干扰源；
2. 已成功清除的频道不再进入定位队列；
3. 未清除真实位置始终包含在保存的可行域内；
4. 只有接口返回 `success` 才能把频道标记为 `CLEARED`；
5. 只有完成七点检测且始终无信号，才能标记为 `EMPTY_CERTIFIED`；
6. 最终每个频道必须为 `CLEARED` 或 `EMPTY_CERTIFIED`；
7. 总清除数必须位于10–16之间；
8. 机器人位置和当前频道只在合法响应后更新；
9. 所有坐标必须为有限数且绝对值不超过接口限制；
10. 程序实际运行时间必须服从 `/enter` 返回的剩余时间，而不能固定假设有1200秒。

---

## 13. 必测案例

正式连接模拟器前至少完成以下单元测试和本地仿真测试。

### 13.1 覆盖测试

- 干扰源位于原点、接收半径为1000 m：必须由中心点发现；
- 干扰源位于目标圆边界、方向位于两个相邻外围点中间：到最近检测点约988.511 m；
- 对目标圆域进行高密度角度和半径扫描，验证到七点的最小距离不超过1000 m。

### 13.2 测向测试

- 示向度接近0度和360度时正确处理角度回绕；
- 两条测向边界平行、近似平行或交于远处；
- 同一位置重复测量得到相同误差时，程序不会错误缩小可行域；
- 补测点 `Q_plus` 和 `Q_minus` 对所有合法初始距离与角误差均不超过1000 m。

### 13.3 清除测试

- 距离恰好5 m时处理 `near`；
- 距离恰好20 m时清除成功；
- 最小包围圆半径为58 m时，两次清除法最坏误差小于20 m；
- 圆心清除失败后，程序正确增加20 m排除圆；
- 网格兜底能够清除位于网格单元最不利角点的目标。

### 13.4 状态机测试

- 10、13、16个干扰源的情况；
- 空频道完成七点检测后才被判空；
- 已清除频道不会被重复清除；
- 网络超时时复用请求ID，业务拒绝时不复用已占用请求ID；
- 测试提前结束或接口关闭时安全停止并保留日志。

---

## 14. 当前本地仿真参考结果

本地仿真采用 `random_state=42`，并假设：

- 干扰源数量在整数10–16中均匀生成；
- 位置在半径1800 m圆域内按面积均匀、相互独立生成；
- 有效接收半径在1000–1500 m内均匀生成；
- 不同位置的测向误差在 `[-1 degree, 1 degree]` 内生成，同一位置误差固定；
- 频道从1–20中无放回抽取。

半径1150 m方案的100个本地案例结果：

| 指标 | 结果 |
|---|---:|
| 成功案例 | 100/100 |
| 平均干扰源数 | 13.1 |
| 平均总虚拟时间 | 79.42 min |
| 中位总虚拟时间 | 79.63 min |
| 最短总虚拟时间 | 60.94 min |
| 最长总虚拟时间 | 103.94 min |
| 平均移动时间 | 64.19 min |
| 平均检测时间 | 11.87 min |
| 平均切频时间 | 2.26 min |
| 平均清除操作时间 | 1.11 min |

这些结果是依据公开规则建立的本地仿真结果，不等同于官方模拟器成绩。正式提交前必须在官方“问题3演练测试”中重复验证。

---

## 15. 对应原型文件

当前本地规则仿真原型：

```text
src/problem3_hybrid_simulation.py
```

对应测试文件：

```text
src/test_problem3_hybrid_simulation.py
```

正式接口程序可以复用其中的覆盖点生成、可行域裁剪、最小包围圆、补测点生成、两次清除法和开放路径优化逻辑；需要将本地 `LocalSimulator` 替换为真实HTTP接口封装。
