"""``problem3_hybrid_simulation`` 的接口适配壳（本仓库没有这个旧模块）。

早期版本的本地仿真模块叫这个名字，现役的是 ``problem3_local_sim.LocalSimulator``
（接口与 ``client.RobotClient`` 一致）。引用旧名字的研究脚本（例如
``q3_empty_channel_strategy.py``）只要不改代码就能继续跑。

只补确实被用到的入口：``generate_sources``；其余名字请直接用
``problem3_local_sim``。**这里不包含任何策略算法。**
"""

from __future__ import annotations

from problem3_local_sim import (  # noqa: F401 - 便于旧脚本顺带引用
    ClearResult,
    LocalSimulator,
    MeasureResult,
    Source,
)


def generate_sources(seed: int, num_sources: int | None = None) -> list[Source]:
    """返回该 seed 下的干扰源列表（``.channel`` / ``.position`` / ``.receive_radius``）。

    与旧模块的语义一致：同一 seed 得到同一批源；源数默认在题面的 10~16 随机。
    """
    simulator = LocalSimulator(seed, num_sources=num_sources)
    return [simulator.sources[channel] for channel in sorted(simulator.sources)]
