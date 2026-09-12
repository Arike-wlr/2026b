"""临时校验：problem2_minimax_worst_area.region_areas 的解析面积 == shapely 求交面积。

对随机 (S2, G) 样本，用两条互相独立的路径计算“定位区域面积”：
  * 解析：region_areas(S1, THETA1, S2, G, ANGLE_ERROR_DEG)
          （内部已对 S2 侧 delta in {-1°, 0°, +1°} 取最坏面积）；
  * shapely：对同一批 delta，把两个 ±1° 扇形各表示成一个足够大的三角形再求交，
          取面积最大值作为参考。
解析侧把“无界 / 退化”区域记为 +inf。判定口径：
  - 解析有限：要求与 shapely 数值一致（相对误差 <= --tol）；
  - 解析 +inf：要求 shapely 面积也确实巨大（>$1e9$)，即两扇形近乎平行导致无界；
  - 其余组合都算反例。
全部通过退出码 0，否则打印首个反例并退出码 1。

用法：
    python problem2/_verify_area_tmp.py
    python problem2/_verify_area_tmp.py --cases 4000 --gbatch 8 --seed 7
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, "d:/CUMCMB/problem2")

from problem2_candidate_region import (  # noqa: E402
    ANGLE_ERROR_DEG,
    DIST_MAX,
    DIST_MIN,
    RECEIVE_MAX,
    S1,
    TARGET_RADIUS,
    THETA1,
)
from problem2_minimax_worst_area import region_areas  # noqa: E402

EPS = ANGLE_ERROR_DEG
FAR = 1.0e7            # shapely 三角形“远端”距离；远大于任何真实坐标尺度
UNBOUNDED_AREA = 1.0e9  # shapely 面积超过此值视为无界


def wedge(apex, phi_deg, half_deg, far=FAR):
    """以 apex 为顶点、中心方位 phi、半张角 half 的扇形（用大三角形近似边界）。"""
    a0 = math.radians(phi_deg - half_deg)
    a1 = math.radians(phi_deg + half_deg)
    return Polygon(
        [
            (apex[0], apex[1]),
            (apex[0] + far * math.cos(a0), apex[1] + far * math.sin(a0)),
            (apex[0] + far * math.cos(a1), apex[1] + far * math.sin(a1)),
        ]
    )


def shapely_worst_area(S1v, S2v, Gv, eps_deg):
    """单个 G：两扇形交集面积，对 S2 侧 delta in {-eps,0,+eps} 取最大。"""
    theta2 = math.degrees(math.atan2(Gv[1] - S2v[1], Gv[0] - S2v[0]))
    w1 = wedge(S1v, THETA1, eps_deg)
    best = 0.0
    for delta in (-eps_deg, 0.0, eps_deg):
        w2 = wedge(S2v, theta2 + delta, eps_deg)
        inter = w1.intersection(w2)
        area = 0.0 if inter.is_empty else float(inter.area)
        best = max(best, area)
    return best


def sample_G(rng, count):
    """在 S1 的 ±1° 扇形里随机采样候选干扰源（距离 [DIST_MIN, DIST_MAX]）。"""
    r = rng.uniform(DIST_MIN, DIST_MAX, size=count)
    d = rng.uniform(-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG, size=count)
    ang = np.radians(THETA1 + d)
    G = np.stack([S1[0] + r * np.cos(ang), S1[1] + r * np.sin(ang)], axis=1)
    keep = np.hypot(G[:, 0], G[:, 1]) <= TARGET_RADIUS + 1e-9
    return G[keep]


def sample_S2(rng):
    """目标圆内均匀采样一个 S2（拒绝采样）。"""
    while True:
        x = rng.uniform(-TARGET_RADIUS, TARGET_RADIUS)
        y = rng.uniform(-TARGET_RADIUS, TARGET_RADIUS)
        if math.hypot(x, y) <= TARGET_RADIUS:
            return np.array([x, y])


def main():
    parser = argparse.ArgumentParser(description="region_areas 解析面积 vs shapely 校验")
    parser.add_argument("--cases", type=int, default=2000, help="随机 S2 样本数")
    parser.add_argument("--gbatch", type=int, default=8, help="每个 S2 采样多少个 G")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--tol", type=float, default=1e-6, help="相对误差阈值")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    S1v = np.asarray(S1, dtype=float)

    n_compared = 0
    n_unbounded = 0
    n_skipped = 0
    worst_rel = 0.0
    worst_case = None

    for _ in range(args.cases):
        S2 = sample_S2(rng)
        if math.hypot(S2[0] - S1v[0], S2[1] - S1v[1]) < DIST_MIN:
            continue
        G = sample_G(rng, args.gbatch)
        # 稳健性约束：只比较 S2 能收到的 G（与 min-max 里的可行域一致）
        G = G[np.hypot(G[:, 0] - S2[0], G[:, 1] - S2[1]) <= RECEIVE_MAX]
        if G.shape[0] == 0:
            continue

        a_ana = region_areas(S1v, THETA1, S2, G, ANGLE_ERROR_DEG)
        for k in range(G.shape[0]):
            Gv = G[k]
            a_ref = shapely_worst_area(S1v, S2, Gv, ANGLE_ERROR_DEG)
            a = float(a_ana[k])

            if math.isfinite(a):
                if a_ref > UNBOUNDED_AREA:
                    print(f"[反例] 解析有限但 shapely 无界：S2={S2} G={Gv} "
                          f"解析={a:.6e} shapely={a_ref:.6e}")
                    return 1
                rel = abs(a_ref - a) / max(a, 1e-12)
                n_compared += 1
                if rel > args.tol:
                    print(f"[反例] 面积不符：S2={S2} G={Gv}\n"
                          f"       解析={a:.10e}  shapely={a_ref:.10e}  rel={rel:.3e}")
                    return 1
                if rel > worst_rel:
                    worst_rel = rel
                    worst_case = (S2.copy(), Gv.copy(), a, a_ref)
            else:
                if a_ref > UNBOUNDED_AREA:
                    n_unbounded += 1
                else:
                    n_skipped += 1
                    print(f"[反例] 解析无界但 shapely 有界：S2={S2} G={Gv} "
                          f"shapely={a_ref:.6e}")
                    return 1

    print("region_areas 解析面积 vs shapely 求交：全部通过")
    print(f"  比较样本数        : {n_compared}")
    print(f"  一致判定的无界样本: {n_unbounded}")
    print(f"  最大相对误差      : {worst_rel:.3e}")
    if worst_case is not None:
        S2c, Gc, a, a_ref = worst_case
        print(f"  最差样本          : S2=({S2c[0]:.1f},{S2c[1]:.1f}) "
              f"G=({Gc[0]:.1f},{Gc[1]:.1f}) 解析={a:.6e} shapely={a_ref:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
