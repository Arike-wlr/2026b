"""问题 2（第二个检测点）：min-max 稳健选址——“考虑最坏可能的点，使它的面积最小”。

与 problem2_candidate_region.py 的区别
=====================================
*  旧口径（覆盖式）：对每个 S2 数一数“有多少个 G 的定位区域直径 d <= 20 m”，
   让这个计数最大 —— 回答“用尽可能多的点控制在 20 m 内”。
*  本文件（稳健式 / min-max）：对每个 S2，取**所有候选干扰源 G 中最坏的那一个**
   （定位区域面积最大者，且再叠加 S2 侧 ±1° 最坏示向度误差），
   然后选择让这个“最坏面积”最小的 S2：

        S2* = argmin_{S2}  max_{G}  Area( G, S2 )

   等价于“为最难定位的那个可能目标点，争取尽可能小的定位区域”。

约束
====
1. S2 在目标区域内（|S2| <= 1800 m）；
2. 稳健性：S2 必须能收到**全部**候选 G（|S2-G| <= 1500 m），否则该候选项记为不可行
   （听不懂的点无法交会，最坏情况退化为“完全定不了位”）；
   可用 --mincover 放宽为“至少覆盖某个比例”。
3. G 侧按 ±1° 误差采样（源的方位不确定），S2 侧按 ±1° 取最坏情况。

面积计算
========
定位区域 = 两个 ±1° 扇形的交集（凸多边形）。其顶点只可能是
两组边界线的 4 个交点 + 两个检测点（当某检测点落在对方扇形内时）。
按极角排序后用鞋带公式求面积；顶点少于 3 个说明区域无界 -> 面积记 +inf。
解析结果已与 shapely 求交交叉验证（偏差 ~1e-10）。

用法
====
    python problem2_minimax_worst_area.py                  # 步长 60 m
    python problem2_minimax_worst_area.py --step 30        # 更细
    python problem2_minimax_worst_area.py --mincover 0.95  # 放宽“必须全收到”
    python problem2_minimax_worst_area.py --noshow
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem2_candidate_region import (  # noqa: E402  复用已校验的几何工具
    ANGLE_ERROR_DEG,
    DIST_MAX,
    DIST_MIN,
    RECEIVE_MAX,
    S1,
    TARGET_RADIUS,
    THETA1,
    connected_components,
    cross,
    dir_vec,
    line_intersection,
    region_bounded,
    region_vertices_single,
    sample_hypotheses,
    to_ray_frame,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "figures")
FIGURE_PATH = os.path.join(OUT_DIR, "problem2_minimax_worst_area.png")
REPORT_PATH = os.path.join(HERE, "problem2_minimax_report.md")

TOL_DEFAULT = 0.10  # 候选区域：最坏面积 <= (1+tol) * 最优最坏面积


# --------------------------------------------------------------------------- #
# 定位区域面积（向量化：一个 S2 对一批 G）
# --------------------------------------------------------------------------- #
def region_areas(S1v, theta1_deg, S2v, G, eps_deg):
    """一批 G 在给定 S2 下的定位区域面积（已对 S2 侧 ±1° 取最坏情况）。

    S1v:(2,), S2v:(2,), G:(M,2)。返回 (M,) 面积数组；区域无界/退化时置 +inf。
    """
    eps = math.radians(eps_deg)
    M = G.shape[0]
    u1m = dir_vec(math.radians(theta1_deg) - eps)
    u1p = dir_vec(math.radians(theta1_deg) + eps)

    theta2 = np.arctan2(G[:, 1] - S2v[1], G[:, 0] - S2v[0])
    apex1 = np.broadcast_to(S1v, (M, 2))
    apex2 = np.broadcast_to(S2v, (M, 2))
    idx = np.arange(M)

    best = np.zeros(M)
    for delta in (-eps, 0.0, eps):
        u2m = dir_vec(theta2 + delta - eps)
        u2p = dir_vec(theta2 + delta + eps)

        with np.errstate(divide="ignore", invalid="ignore"):
            # 6 个候选顶点：4 个边界线交点 + 两个检测点
            V = np.stack(
                [
                    line_intersection(S1v, u1m, S2v, u2m),
                    line_intersection(S1v, u1m, S2v, u2p),
                    line_intersection(S1v, u1p, S2v, u2m),
                    line_intersection(S1v, u1p, S2v, u2p),
                    apex1,
                    apex2,
                ],
                axis=1,
            )  # (M,6,2)

            valid = np.isfinite(V).all(axis=-1)
            for u, p, sign in (
                (u1m, S1v, +1.0),
                (u1p, S1v, -1.0),
                (u2m[:, None, :], S2v, +1.0),
                (u2p[:, None, :], S2v, -1.0),
            ):
                valid &= (sign * cross(u, V - p) >= -1e-6)

            cnt = valid.sum(axis=1)

            # 无效顶点用“第一个有效顶点”占位：它们与占位点角度相同，排序后相邻，
            # 在鞋带公式里贡献 0 长度边，不影响面积，也不破坏闭合。
            first = np.argmax(valid, axis=1)
            filler = V[idx, first]
            Vf = np.where(valid[..., None], V, filler[:, None, :])

            center = Vf.mean(axis=1)                      # 凸包内点，用于极角排序
            ang = np.arctan2(Vf[:, :, 1] - center[:, 1:2],
                             Vf[:, :, 0] - center[:, 0:1])
            order = np.argsort(ang, axis=1)
            Vs = np.take_along_axis(Vf, order[:, :, None], axis=1)
            nxt = np.roll(Vs, -1, axis=1)
            area = 0.5 * np.abs(
                (Vs[..., 0] * nxt[..., 1] - Vs[..., 1] * nxt[..., 0]).sum(axis=1)
            )
            bounded = region_bounded(math.radians(theta1_deg), theta2, delta, eps)
            area = np.where(bounded & (cnt >= 3), area, np.inf)

        best = np.maximum(best, area)

    return best


# --------------------------------------------------------------------------- #
# min-max 网格搜索
# --------------------------------------------------------------------------- #
def minimax_grid(G, step, min_cover):
    """返回 (xs, ys, A_worst, A_best, cover)，不可行处为 nan / inf。"""
    xs = np.arange(-TARGET_RADIUS, TARGET_RADIUS + step, step)
    ys = np.arange(-TARGET_RADIUS, TARGET_RADIUS + step, step)
    shape = (ys.size, xs.size)
    A_worst = np.full(shape, np.inf)
    A_best = np.full(shape, np.inf)
    cover = np.zeros(shape)

    S1v = np.asarray(S1, dtype=float)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if math.hypot(x, y) > TARGET_RADIUS:
                continue
            S2 = np.array([x, y])
            d = np.hypot(G[:, 0] - x, G[:, 1] - y)
            recv = d <= RECEIVE_MAX
            c = recv.mean()
            cover[iy, ix] = c
            if c < min_cover or math.hypot(x - S1v[0], y - S1v[1]) < DIST_MIN:
                continue
            a = region_areas(S1v, THETA1, S2, G[recv], ANGLE_ERROR_DEG)
            A_worst[iy, ix] = a.max()      # 最坏可能的点
            A_best[iy, ix] = a.min()       # 最好可能的点

    return xs, ys, A_worst, A_best, cover


def main():
    parser = argparse.ArgumentParser(description="问题2 min-max 稳健选址")
    parser.add_argument("--step", type=float, default=60.0, help="S2 网格步长(m)")
    parser.add_argument("--rsamples", type=int, default=160, help="G 的距离采样数")
    parser.add_argument("--dsamples", type=int, default=7, help="G 的角度误差采样数")
    parser.add_argument("--mincover", type=float, default=1.0,
                        help="要求 S2 能收到的候选 G 比例下限（1.0=必须全部收到）")
    parser.add_argument("--tol", type=float, default=TOL_DEFAULT,
                        help="候选区域阈值：最坏面积 <= (1+tol)*最优值")
    parser.add_argument("--noshow", action="store_true", help="只存图不弹窗")
    args = parser.parse_args()

    G, _ = sample_hypotheses(args.rsamples, args.dsamples)
    print(f"S1 = {(float(S1[0]), float(S1[1]))}，示向度 theta1 = {THETA1}°，误差 ±{ANGLE_ERROR_DEG}°")
    print(f"候选干扰源 G：{G.shape[0]} 个（距离 {DIST_MIN}~{DIST_MAX} m）")
    print(f"min-max 目标：min_S2 max_G 面积(G, S2)，步长 {args.step:g} m，"
          f"要求 S2 至少能收到 {args.mincover:.0%} 的候选 G\n")

    xs, ys, A_worst, A_best, cover = minimax_grid(G, args.step, args.mincover)

    feasible = np.isfinite(A_worst)
    if not feasible.any():
        print("没有任何可行 S2（放宽 --mincover 试试）。")
        return

    A_opt = A_worst[feasible].min()
    iy, ix = np.unravel_index(np.argmin(np.where(feasible, A_worst, np.inf)), A_worst.shape)
    best_S2 = (float(xs[ix]), float(ys[iy]))

    thr = A_opt * (1.0 + args.tol)
    cand = feasible & (A_worst <= thr)
    comps = connected_components(cand, xs, ys)

    print(f"可行 S2 数：{int(feasible.sum())} / {int(np.isfinite(cover).sum())}")
    print(f"min-max 最优 S2 = {best_S2}")
    print(f"  最坏面积 A* = {A_opt:,.0f} m²（等价直径 ≈ {2 * math.sqrt(A_opt / math.pi):.1f} m）")
    print(f"  最好面积     = {A_best[iy, ix]:,.0f} m²")
    u, v = to_ray_frame(*best_S2, THETA1)
    print(f"  示向度参考系 (u 沿 theta1, v 垂直) = (u={u:.0f}, v={v:.0f})")

    # 谁是最坏的那个点
    recv = np.hypot(G[:, 0] - best_S2[0], G[:, 1] - best_S2[1]) <= RECEIVE_MAX
    a_all = region_areas(np.asarray(S1, float), THETA1, np.array(best_S2), G[recv], ANGLE_ERROR_DEG)
    G_recv = G[recv]
    k = int(np.argmax(a_all))
    Gw = G_recv[k]
    rw = math.hypot(*Gw)
    print(f"  最坏点 G* = ({Gw[0]:.0f}, {Gw[1]:.0f})，距 S1 {rw:.0f} m，面积 {a_all[k]:,.0f} m²")
    print(f"  可接收 G 中：面积中位数 {np.median(a_all):,.0f} m²，"
          f"<= 2*A* 的比例 {(a_all <= 2 * A_opt).mean():.1%}")

    print(f"\n候选区域（最坏面积 <= {thr:,.0f} m²，即最优值的 {1 + args.tol:.0%}）"
          f"共 {len(comps)} 瓣：")
    for kk, pts in enumerate(comps, 1):
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        uu, vv = to_ray_frame(cx, cy, THETA1)
        ax = np.array([A_worst[int(round((cy - ys[0]) / args.step)),
                              int(round((cx - xs[0]) / args.step))]])
        print(f"  瓣 {kk}: 中心 ({cx:.0f}, {cy:.0f})，示向度坐标系 (u={uu:.0f}, v={vv:.0f})，"
              f"面积约 {len(pts) * args.step ** 2 / 1e6:.3f} km²，"
              f"该处最坏面积 ≈ {ax[0]:,.0f} m²")

    # ---------------- 作图 ----------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    logw = np.where(feasible, np.log10(np.maximum(A_worst, 1e-6)), np.nan)

    # (a) 最坏面积图
    ax = axes[0, 0]
    im = ax.imshow(logw, origin="lower", extent=extent, cmap="viridis_r", aspect="equal")
    ax.add_patch(MplCircle((0, 0), TARGET_RADIUS, fill=False, edgecolor="k", ls="--", lw=1.2))
    for s in (-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG):
        a = math.radians(THETA1 + s)
        ax.plot([S1[0], S1[0] + DIST_MAX * math.cos(a)],
                [S1[1], S1[1] + DIST_MAX * math.sin(a)], ls="--", lw=1.0, color="orange")
    ax.plot(*S1, "o", ms=11, color="red", mec="k", zorder=6)
    ax.plot(*best_S2, "*", ms=18, color="deepskyblue", mec="k", zorder=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="log10(最坏定位面积 / m²)")
    ax.set_title("(a) min-max 最坏面积（越小越好）", fontsize=13)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")

    # (b) 最好面积图
    ax = axes[0, 1]
    logb = np.where(feasible, np.log10(np.maximum(A_best, 1e-6)), np.nan)
    im = ax.imshow(logb, origin="lower", extent=extent, cmap="magma_r", aspect="equal")
    ax.add_patch(MplCircle((0, 0), TARGET_RADIUS, fill=False, edgecolor="k", ls="--", lw=1.2))
    ax.plot(*S1, "o", ms=11, color="red", mec="k", zorder=6)
    ax.plot(*best_S2, "*", ms=18, color="deepskyblue", mec="k", zorder=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="log10(最好定位面积 / m²)")
    ax.set_title("(b) 最好情形面积（参考）", fontsize=13)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")

    # (c) 候选区域放大
    ax = axes[1, 0]
    ax.add_patch(MplCircle((0, 0), TARGET_RADIUS, fill=False, edgecolor="k", ls="--",
                           lw=1.0, alpha=0.5))
    ax.contourf(xs, ys, np.where(cand, logw, np.nan), levels=12, cmap="viridis_r")
    if cand.any():
        cy_, cx_ = np.where(cand)
        pad = max(80.0, 3 * args.step)
        ax.set_xlim(xs[cx_.min()] - pad, xs[cx_.max()] + pad)
        ax.set_ylim(ys[cy_.min()] - pad, ys[cy_.max()] + pad)
    ax.plot(*S1, "o", ms=11, color="red", mec="k", zorder=6)
    for s in (-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG):
        a = math.radians(THETA1 + s)
        ax.plot([S1[0], S1[0] + DIST_MAX * math.cos(a)],
                [S1[1], S1[1] + DIST_MAX * math.sin(a)], ls="--", lw=1.2, color="orange")
    ax.plot(*best_S2, "*", ms=20, color="deepskyblue", mec="k", zorder=8,
            label=f"最优 S2({best_S2[0]:.0f},{best_S2[1]:.0f})")
    for kk, pts in enumerate(comps, 1):
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        ax.plot(cx, cy, "P", ms=12, color="magenta", mec="white", zorder=9)
        ax.annotate(f"瓣 {kk}\n({cx:.0f},{cy:.0f})", (cx, cy), textcoords="offset points",
                    xytext=(6, 10), fontsize=9.5, color="magenta", zorder=10,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="magenta", pad=1.5))
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"(c) 候选区域：最坏面积 <= {thr:,.0f} m²", fontsize=13)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, ls=":", alpha=0.5)

    # (d) 最优 S2 处“最坏点”的定位区域
    ax = axes[1, 1]
    verts = region_vertices_single(np.asarray(S1, float), THETA1, np.array(best_S2),
                                   Gw, ANGLE_ERROR_DEG)
    win = max(120.0, 2.5 * math.sqrt(max(a_all[k], 1.0)))
    ax.plot(*S1, "o", ms=10, color="red", mec="k", zorder=6)
    ax.plot(*best_S2, "*", ms=18, color="deepskyblue", mec="k", zorder=7)
    a1 = math.radians(THETA1)
    for s in (-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG):
        a = math.radians(THETA1 + s)
        ax.plot([S1[0], S1[0] + 5000 * math.cos(a)], [S1[1], S1[1] + 5000 * math.sin(a)],
                lw=1.0, color="tab:blue", alpha=0.85)
    th2 = math.atan2(Gw[1] - best_S2[1], Gw[0] - best_S2[0])
    for s in (-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG):
        a = th2 + math.radians(s)
        ax.plot([best_S2[0], best_S2[0] + 5000 * math.cos(a)],
                [best_S2[1], best_S2[1] + 5000 * math.sin(a)],
                lw=1.0, color="tab:green", alpha=0.85)
    if len(verts) >= 3:
        ax.add_patch(MplPolygon(verts, closed=True, facecolor="red", alpha=0.5,
                                edgecolor="darkred", lw=2.0, zorder=5))
    ax.plot(*Gw, "x", ms=12, color="black", zorder=8)
    ax.annotate(f"G*({Gw[0]:.0f},{Gw[1]:.0f})\n面积 {a_all[k]:,.0f} m²", Gw,
                textcoords="offset points", xytext=(8, 10), fontsize=10, zorder=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", pad=1.5))
    ax.set_xlim(Gw[0] - win, Gw[0] + win)
    ax.set_ylim(Gw[1] - win, Gw[1] + win)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(d) 最优 S2 处“最坏点”的定位区域（局部放大）", fontsize=13)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=170)
    print(f"\n图片已保存：{FIGURE_PATH}")

    write_report(args, best_S2, A_opt, A_best[iy, ix], thr, cand, comps, G, Gw, a_all, cover)

    if not args.noshow:
        plt.show()
    plt.close(fig)


def write_report(args, best_S2, A_opt, A_best, thr, cand, comps, G, Gw, a_all, cover):
    u, v = to_ray_frame(*best_S2, THETA1)
    d_equiv = 2 * math.sqrt(A_opt / math.pi)
    lines = [
        "# 问题2 第二检测点：min-max（最坏情况最小化）选址报告",
        "",
        "## 模型",
        "",
        "- 决策变量：第二个检测点 S2（|S2| <= 1800 m）；",
        "- 不确定量：干扰源真实位置 G（S1 示向度 ±1° 扇形内）与 S2 的示向度测量误差 ±1°；",
        "- 目标：min_S2 max_G Area(G, S2)，即“为最难定位的那个可能目标点，争取最小的定位区域”；",
        f"- 稳健性约束：S2 必须能收到至少 {args.mincover:.0%} 的候选 G（默认全部）；",
        "- 定位区域 = 两个 ±1° 扇形交集的凸多边形，鞋带公式求面积；无界区域记 +inf。",
        "",
        "## 结果",
        "",
        f"- min-max 最优点 S2* = ({best_S2[0]:.1f}, {best_S2[1]:.1f})，"
        f"示向度坐标系 (u={u:.1f}, v={v:.1f})；",
        f"- 最坏面积 A* = {A_opt:,.0f} m²（等效直径 ≈ {d_equiv:.1f} m），"
        f"同点最好面积 = {A_best:,.0f} m²；",
        f"- 最坏目标点 G* = ({Gw[0]:.1f}, {Gw[1]:.1f})，距 S1 {math.hypot(*Gw):.1f} m；",
        f"- 候选区域取“最坏面积 <= {thr:,.0f} m²（最优值的 {1 + args.tol:.0%}）”的 S2 集合，"
        f"共 {len(comps)} 瓣，合计约 {int(cand.sum()) * args.step ** 2 / 1e6:.3f} km²。",
        "",
        "| 瓣 | 中心(x,y) | 中心(u,v) | 面积(km²) |",
        "| --- | --- | --- | --- |",
    ]
    for k, pts in enumerate(comps, 1):
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        uu, vv = to_ray_frame(cx, cy, THETA1)
        lines.append(f"| {k} | ({cx:.0f}, {cy:.0f}) | ({uu:.0f}, {vv:.0f}) | "
                     f"{len(pts) * args.step ** 2 / 1e6:.3f} |")
    lines += [
        "",
        "## 说明与对比",
        "",
        f"- 最坏面积由距离最远的可接收目标点主导（面积 ~ r²），因此 min-max 会让 S2 "
        f"去“照顾”约 {math.hypot(*Gw):.0f} m 处的目标，而不是照顾近处目标；",
        f"- 与覆盖式口径（problem2_candidate_region.py：最大化 d<=20 m 的 G 计数）相比，"
        f"min-max 不设 20 m 硬阈值，只追求“最坏也不至于太差”；",
        "- 若 20 m 是清除命中必须的硬指标，应以覆盖式结果为准；若只要求整体稳健，用本结果。",
        "",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"报告已保存：{REPORT_PATH}")


if __name__ == "__main__":
    main()
