# q3-v2.01

在v2.00上继续优化：首段共享测向并比较后续路线，连续调整补测位置。旧版本目录保留。

- 同一原100例：v1.99 **248.22 → v2.00 229.31 → v2.01 226.40 秒/源**。
- 全新300例独立验证：v2.00 **232.30 → v2.01 229.42 秒/源**，两个版本均300/300全清。
- 独立噪声100例：**228.72 → 225.44 秒/源**，均全清。
- **尚未达到220**。聚簇压力场景平均慢了2.24秒/源；规划计算也有所增加。

详细方法、分项收益、失败尝试与统计区间见 [REPORT.md](REPORT.md)。逐案数据在evidence目录。

在PowerShell进入本目录后运行：

```powershell
./run_local.ps1 -Cases 100 -RandomState 1405468406
```

脚本优先使用本机已有依赖的Codex Python。其他电脑先安装依赖：

```powershell
python -m pip install -r requirements.txt
python problem3_strategy.py --cases 100 --random-state 1405468406
python problem3_strategy.py --cases 300 --random-state 2026096200
python problem3_main.py --check
```

演练入口沿用原参数。赛方客户端已选择问题3演练、登录并开始后，才运行：

```powershell
python problem3_main.py --robot-id 你的实际队号 --confirm-problem3-practice
```

本轮所有实验均在本地完成，未消耗赛方测试机会。真实联网流程未验证。

请保留整个目录。`problem3_strategy.py`为新版入口，`problem3_v200.py`保存上一版策略，`problem3_legacy.py`保存原策略和通信实现，几何、路径及模拟器文件为必要依赖。动态加载的`coverage_strategy`已嵌入legacy模块，无需另下载。

摘要与图表统一从`evidence/summary.json`及保存的案例记录生成；冻结文件与验证结果见`evidence/FREEZE.json`、`evidence/verification.json`。
