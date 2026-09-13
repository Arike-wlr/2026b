# q4_21-v2.0

在原21点框架上增加途中补测、清除后共享测向、凸分块兜底和更快的路线计算。原目录保留。

- 原100例：471.76 → 462.95 秒/源，均全清。
- 全新300例：**468.00 → 458.73 秒/源**，总时间下降 **2.00%**，均300/300全清。
- 100例独立误差验证：总时间下降2.21%。平均改进不保证每局更快。

详细方法、计量差异、未采纳实验、压力回退见 [REPORT.md](REPORT.md)。

## 本地运行

```powershell
./run_local.ps1 -Cases 100 -RandomState 2007525743
```

其他电脑先安装依赖：

```powershell
python -m pip install -r requirements.txt
python problem4_strategy.py --cases 100 --random-state 2007525743 --log-first
python problem4_main.py --check
```

在线入口保留原参数：赛方客户端已登录、选择问题4并开始后，运行 `python problem4_main.py 你的队号`。本轮未联网，也未消耗赛方测试机会。

请保留整个目录。`problem4_baseline.py`为原策略副本；几何、模拟器、日志、客户端和路线文件都是必要依赖。默认不计返航；`--route-end-at-origin`只改变收尾路线评分，不产生返航动作。数据在 `evidence/`，可复现实验代码在 `research/`。
