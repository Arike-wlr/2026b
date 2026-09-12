"""问题 2：第二个检测点 S2 的选择策略与候选区域。

建模口径（“先用尽可能多的点控制在 20 m 范围内”）
==================================================
1) 已知第一个检测点 S1 与在该点测得某**全向**干扰源 G 的示向度 theta1（误差 ±1°）。
   * 由示向度含义，G 落在以 S1 为顶点、张角 2°（=±1°）的扇形（这里是一整条带误差的射线）内；
   * 又因为 S1 能收到信号，|S1-G| <= 有效接收半径 <= 1500 m；
   * 且干扰源一定在目标区域 |G| <= 1800 m。
   我们把 G 离散成一批“候选点”（在 S1 的扇形里按距离 r 与误差 delta 采样）。

2) 再加入第二个检测点 S2 后，两点交会定位的**定位区域** = 两个 ±1° 扇形的交集（凸多边形）。
   区域的“直径” d 越小，定位越准。20 m 恰好是光学精确定位 + 激光清除的半径：
   只要 d <= 20 m，机器狗走到区域中心附近一次 /clear 就能命中。

3) 选择策略 / 候选区域：
   对每一个候选 S2，统计在全部候选 G 中，有多少个 G 满足
        “S2 能收到它（|S2-G| <= 1500）且交会定位区域直径 d <= 20 m”，
   记为 S2 的得分 score(S2)。把 score 接近最大值（默认 >= 0.9*max）的位置集合作为
   **第二个检测点的候选区域**——也就是“用尽可能多的 G 点都能控制在 20 m 以内”的位置。

实现要点
========
* 定位区域直径用**解析法**计算：区域 = 4 个半平面（两组扇形边界线）的交，凸多边形的
  顶点只可能是内层两类边界线的 4 个交点中满足全部约束者，直径 = 顶点两两距离最大值。
  这样可以对候选 G 向量化，避免 shapely 逐个求交，速度足够做网格搜索。
* 每个 G 都用“最坏情况”的 S2 示向度误差 delta2 ∈ {-1°, 0°, +1°} 取直径最大值。

用法
====
    python problem2_candidate_region.py                # 默认步长 30 m
    python problem2_candidate_region.py --step 20      # 更细（更慢）
    python problem2_candidate_region.py --dmax 30      # 放宽“20 m”阈值
    python problem2_candidate_region.py --noshow       # 只存图不弹窗
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

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
FIGURE_PATH = os.path.join(OUT_DIR, "problem2_candidate_region.png")
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "problem2_report.md")


# --------------------------------------------------------------------------- #
# 可调参数
# --------------------------------------------------------------------------- #
S1 = np.array([0.0, 0.0])      # 第一个检测点（可改成任意位置）
THETA1 = 45.0                  # 在该点测得的示向度（度，x 轴正向逆时针）

TARGET_RADIUS = 1800.0         # 目标区域半径
ANGLE_ERROR_DEG = 1.0          # 示向度误差 ±1°
RECEIVE_MAX = 1500.0           # 有效接收半径上界（未知，取上界做保守）
DIST_MIN = 5.0                 # 小于等于 5 m 会返回 near，不会有示向度
DIST_MAX = 1500.0              # 候选 G 与 S1 的最大距离

D_MAX_DEFAULT = 20.0           # 定位区域直径判据（清除半径）
COVERAGE_RATIO = 0.90          # 候选区域：score >= ratio * max_score


# --------------------------------------------------------------------------- #
# 基础几何（向量化）
# --------------------------------------------------------------------------- #
def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def dir_vec(angle_rad):
    return np.stack([np.cos(angle_rad), np.sin(angle_rad)], axis=-1)


def line_intersection(p1, u1, p2, u2):
    """直线 p1+t*u1 与 p2+s*u2 的交点，批量。
    p1:(2,), u1:(2,) 或 (N,2)；p2, u2 同形。返回 (N,2)。"""
    d = p2 - p1
    denom = cross(u1, u2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross(d, u2) / denom
    return p1 + t[..., None] * u1


def region_bounded(theta1, theta2, delta, eps):
    """两个 ±eps 扇形交集是否有界：4 条外法线角度需张满整圈（最大间隔 < pi）。

    theta1、delta、eps 为弧度标量，theta2 为弧度数组 (N,)；返回 (N,) bool。
    外法线：cross(u, P-S)>=0 取右法线 (u_y,-u_x)，<=0 取左法线 (-u_y,u_x)，
    对应角度分别为 angle(u)-90° 与 angle(u)+90°。
    """
    angles = np.stack(
        [
            np.full_like(theta2, theta1) - eps - math.pi / 2,
            np.full_like(theta2, theta1) + eps + math.pi / 2,
            theta2 + delta - eps - math.pi / 2,
            theta2 + delta + eps + math.pi / 2,
        ],
        axis=1,
    )                                   # (N,4)
    angles = np.sort(np.mod(angles, 2 * math.pi), axis=1)
    gaps = np.diff(
        np.concatenate([angles, angles[:, :1] + 2 * math.pi], axis=1), axis=1
    )
    return gaps.max(axis=1) < math.pi - 1e-9


def region_diameters(S1v, theta1, S2v, G, eps_deg):
    """批量计算 (S1,theta1) 与 (S2,G) 两点交会定位区域的直径（含 S2 侧 ±1° 最坏情况）。

    S1v:(2,), S2v:(2,), G:(N,2), theta1:度。
    返回 (N,) 直径数组；区域退化（顶点不足 2 个）时置 +inf。
    """
    eps = math.radians(eps_deg)
    N = G.shape[0]

    u1m = dir_vec(math.radians(theta1) - eps)          # (2,)
    u1p = dir_vec(math.radians(theta1) + eps)          # (2,)

    theta2 = np.arctan2(G[:, 1] - S2v[1], G[:, 0] - S2v[0])   # (N,)

    best = np.zeros(N)

    apex1 = np.broadcast_to(S1v, (N, 2))
    apex2 = np.broadcast_to(S2v, (N, 2))

    for delta2 in (-eps, 0.0, eps):
        u2m = dir_vec(theta2 + delta2 - eps)[:, None, :]      # (N,1,2) 便于广播
        u2p = dir_vec(theta2 + delta2 + eps)[:, None, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            # 6 个候选顶点：两组边界线的 4 个交点 + 两个检测点（顶点可能落在对方扇形内）
            V = np.stack(
                [
                    line_intersection(S1v, u1m, S2v, u2m[:, 0, :]),
                    line_intersection(S1v, u1m, S2v, u2p[:, 0, :]),
                    line_intersection(S1v, u1p, S2v, u2m[:, 0, :]),
                    line_intersection(S1v, u1p, S2v, u2p[:, 0, :]),
                    apex1,
                    apex2,
                ],
                axis=1,
            )  # (N,6,2)

            # 半平面约束：cross(dir(phi-eps), P-S) >= 0 且 cross(dir(phi+eps), P-S) <= 0
            valid = np.isfinite(V).all(axis=-1)
            for u, p, sign in (
                (u1m, S1v, +1.0),
                (u1p, S1v, -1.0),
                (u2m, S2v, +1.0),
                (u2p, S2v, -1.0),
            ):
                cr = cross(u, V - p)                      # (N,6)
                valid &= (sign * cr >= -1e-6)

            # 顶点两两距离，仅统计两端都有效的组合
            diff = V[:, :, None, :] - V[:, None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=-1))      # (N,6,6)
            pair_mask = valid[:, :, None] & valid[:, None, :]
            diam = np.where(pair_mask, dist, 0.0).max(axis=(1, 2))
            # 区域无界（两组边界近平行）或顶点不足 3 个 -> 视为定位失效
            ok = region_bounded(math.radians(theta1), theta2, delta2, eps) \
                & (valid.sum(axis=1) >= 3)
            diam = np.where(ok, diam, np.inf)

        best = np.maximum(best, diam)

    return best


def region_vertices_single(S1v, theta1, S2v, Gv, eps_deg):
    """单个 (S2,G) 的定位区域顶点（按凸包顺序返回），用于作图。
    区域无界或退化时返回 []。"""
    eps = math.radians(eps_deg)
    u1m = dir_vec(math.radians(theta1) - eps)
    u1p = dir_vec(math.radians(theta1) + eps)
    theta2 = math.atan2(Gv[1] - S2v[1], Gv[0] - S2v[0])
    u2m = dir_vec(theta2 - eps)
    u2p = dir_vec(theta2 + eps)

    if not region_bounded(math.radians(theta1), np.array([theta2]), 0.0, eps)[0]:
        return []

    candidates = [
        line_intersection(S1v, u1m, S2v, u2m),
        line_intersection(S1v, u1m, S2v, u2p),
        line_intersection(S1v, u1p, S2v, u2m),
        line_intersection(S1v, u1p, S2v, u2p),
        S1v,
        S2v,
    ]

    pts = []
    for P in candidates:
        ok = (
            cross(u1m, P - S1v) >= -1e-6
            and cross(u1p, P - S1v) <= 1e-6
            and cross(u2m, P - S2v) >= -1e-6
            and cross(u2p, P - S2v) <= 1e-6
        )
        if ok:
            pts.append((float(P[0]), float(P[1])))

    # 去重并按极角排序（凸多边形）
    uniq = []
    for p in pts:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 1e-6 for q in uniq):
            uniq.append(p)
    if len(uniq) >= 3:
        cx = sum(p[0] for p in uniq) / len(uniq)
        cy = sum(p[1] for p in uniq) / len(uniq)
        uniq.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return uniq


def polygon_diameter(pts):
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            best = max(best, math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
    return best


# --------------------------------------------------------------------------- #
# G 候选点采样 & 网格搜索
# --------------------------------------------------------------------------- #
def sample_hypotheses(r_count, delta_count):
    """在 S1 的 ±1° 扇形内采样候选干扰源 G。返回 (G:(N,2), r1:(N,))。"""
    rs = np.linspace(DIST_MIN, DIST_MAX, r_count)
    deltas = np.linspace(-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG, delta_count)
    rr, dd = np.meshgrid(rs, deltas, indexing="ij")
    rr = rr.ravel()
    dd = dd.ravel()
    ang = np.radians(THETA1 + dd)
    G = np.stack([S1[0] + rr * np.cos(ang), S1[1] + rr * np.sin(ang)], axis=1)
    inside = np.hypot(G[:, 0], G[:, 1]) <= TARGET_RADIUS + 1e-9
    return G[inside], rr[inside]


def score_grid(G, step, d_max):
    """在目标圆域内做网格搜索，返回 (xs, ys, score, best)。"""
    xs = np.arange(-TARGET_RADIUS, TARGET_RADIUS + step, step)
    ys = np.arange(-TARGET_RADIUS, TARGET_RADIUS + step, step)
    score = np.full((ys.size, xs.size), np.nan)

    best = {"score": -1, "S2": None, "max_d": None}

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if math.hypot(x, y) > TARGET_RADIUS:
                continue
            S2 = np.array([x, y])
            if math.hypot(x - S1[0], y - S1[1]) < DIST_MIN:
                score[iy, ix] = 0.0
                continue

            d = region_diameters(S1, THETA1, S2, G, ANGLE_ERROR_DEG)
            receivable = np.hypot(G[:, 0] - x, G[:, 1] - y) <= RECEIVE_MAX
            good = receivable & (d <= d_max)
            s = int(good.sum())
            score[iy, ix] = s

            if s > best["score"]:
                best = {"score": s, "S2": (float(x), float(y)), "max_d": d}

    return xs, ys, score, best


def connected_components(mask, xs, ys):
    """8 连通地把候选区域的网格点分成若干“瓣”，返回每瓣的点列表。"""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    comps = []
    for iy in range(height):
        for ix in range(width):
            if not mask[iy, ix] or visited[iy, ix]:
                continue
            stack = [(iy, ix)]
            visited[iy, ix] = True
            pts = []
            while stack:
                cy, cx = stack.pop()
                pts.append((xs[cx], ys[cy]))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width \
                                and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(pts)
    comps.sort(key=len, reverse=True)
    return comps


def to_ray_frame(x, y, theta_deg):
    """把 S2 坐标转到以 S1 示向度为 x 轴的参考系：u 沿示向度，v 垂直（左正右负）。"""
    t = math.radians(theta_deg)
    return x * math.cos(t) + y * math.sin(t), -x * math.sin(t) + y * math.cos(t)


# --------------------------------------------------------------------------- #
# 出图
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="问题2：第二检测点候选区域")
    parser.add_argument("--step", type=float, default=30.0, help="S2 网格步长(m)")
    parser.add_argument("--dmax", type=float, default=D_MAX_DEFAULT, help="定位区域直径阈值(m)")
    parser.add_argument("--rsamples", type=int, default=160, help="G 的距离采样数")
    parser.add_argument("--dsamples", type=int, default=7, help="G 的角度误差采样数")
    parser.add_argument("--noshow", action="store_true", help="只保存图片，不弹窗")
    args = parser.parse_args()

    G, _ = sample_hypotheses(args.rsamples, args.dsamples)
    print(f"S1 = {(float(S1[0]), float(S1[1]))}，示向度 theta1 = {THETA1}°，误差 ±{ANGLE_ERROR_DEG}°")
    print(f"候选干扰源 G：{G.shape[0]} 个（距离 {DIST_MIN}~{DIST_MAX} m，误差 ±{ANGLE_ERROR_DEG}°）")
    print(f"网格步长 {args.step} m，定位区域直径阈值 d_max = {args.dmax} m\n")

    xs, ys, score, best = score_grid(G, args.step, args.dmax)

    max_score = int(np.nanmax(score))
    thr = COVERAGE_RATIO * max_score
    cand = np.isfinite(score) & (score >= thr)
    nG = G.shape[0]

    comps = connected_components(cand, xs, ys) if max_score > 0 else []

    print(f"单个 S2 能“控制在 {args.dmax} m 内”的 G 点数上限 = {max_score} / {nG} "
          f"（覆盖率 {max_score / nG:.1%}）")
    print(f"最优 S2 = {best['S2']}，得分 {best['score']}")
    for k, pts in enumerate(comps, 1):
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        u, v = to_ray_frame(cx, cy, THETA1)
        print(f"  候选瓣 {k}：中心 ({cx:.0f}, {cy:.0f})，示向度坐标系 "
              f"(u={u:.0f}, v={v:.0f})，面积约 {len(pts) * args.step ** 2 / 1e6:.3f} km²")

    if max_score > 0:
        bx, by = best["S2"]
        d_all = region_diameters(S1, THETA1, np.array(best["S2"]), G, ANGLE_ERROR_DEG)
        recv = np.hypot(G[:, 0] - bx, G[:, 1] - by) <= RECEIVE_MAX
        d_good = d_all[recv & (d_all <= args.dmax)]
        print(f"最优 S2 处：可接收且 d<={args.dmax} 的 G 共 {d_good.size} 个，"
              f"d 中位数 {np.median(d_good):.2f} m，最大 {d_good.max():.2f} m")
        area = cand.sum() * args.step * args.step
        print(f"候选区域（score >= {thr:.0f}）网格点数 {int(cand.sum())}，"
              f"面积约 {area / 1e6:.3f} km²")
    else:
        print("警告：没有任何 S2 能把哪怕一个 G 的定位区域压到阈值以内，"
              "可放宽 --dmax 或调整候选 G 的距离范围。")

    # ---------------- 作图 ----------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    # (a) 得分热力图
    ax = axes[0]
    im = ax.imshow(score, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    ax.add_patch(MplCircle((0, 0), TARGET_RADIUS, fill=False, edgecolor="black",
                           linewidth=1.2, linestyle="--"))
    ax.plot(*S1, marker="o", ms=12, color="red", markeredgecolor="black", zorder=6)
    ax.annotate("S1", S1, textcoords="offset points", xytext=(10, 8), fontsize=12, zorder=7)
    for sign in (-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG):
        a = math.radians(THETA1 + sign)
        ax.plot([S1[0], S1[0] + DIST_MAX * math.cos(a)],
                [S1[1], S1[1] + DIST_MAX * math.sin(a)],
                ls="--", lw=1.0, color="orange", zorder=4)
    fig.colorbar(im, ax=ax, fraction=0.046, label=f"满足 d<={args.dmax:g} m 的 G 点数")
    ax.set_title(f"(a) 候选 S2 得分热力图（步长 {args.step:g} m）", fontsize=13)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")

    # (b) 候选区域（放大）
    ax = axes[1]
    ax.add_patch(MplCircle((0, 0), TARGET_RADIUS, fill=False, edgecolor="black",
                           linewidth=1.0, linestyle="--", alpha=0.5))
    ax.contourf(xs, ys, np.where(cand, score, np.nan), levels=12, cmap="autumn_r", alpha=0.85)
    if cand.any():
        iy, ix = np.where(cand)
        pad = max(60.0, 3 * args.step)
        ax.set_xlim(xs[ix.min()] - pad, xs[ix.max()] + pad)
        ax.set_ylim(ys[iy.min()] - pad, ys[iy.max()] + pad)
    ax.plot(*S1, marker="o", ms=12, color="red", markeredgecolor="black", zorder=6)
    ax.annotate("S1", S1, textcoords="offset points", xytext=(10, 8), fontsize=12, zorder=7)
    for sign in (-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG):
        a = math.radians(THETA1 + sign)
        ax.plot([S1[0], S1[0] + DIST_MAX * math.cos(a)],
                [S1[1], S1[1] + DIST_MAX * math.sin(a)],
                ls="--", lw=1.2, color="orange", zorder=4)
    if best["S2"] is not None:
        ax.plot(*best["S2"], marker="*", ms=18, color="blue",
                markeredgecolor="white", zorder=8, label=f"最优 S2{tuple(round(v) for v in best['S2'])}")
        ax.legend(loc="upper right", fontsize=10)
    for k, pts in enumerate(comps, 1):
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        ax.plot(cx, cy, marker="P", ms=13, color="magenta",
                markeredgecolor="white", zorder=9)
        ax.annotate(f"瓣 {k}\n({cx:.0f},{cy:.0f})", (cx, cy), textcoords="offset points",
                    xytext=(6, 10), fontsize=9.5, color="magenta", zorder=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="magenta", pad=1.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"(b) 第二个检测点候选区域（score >= {thr:.0f}）", fontsize=13)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, linestyle=":", alpha=0.5)

    # (c) 最优 S2 的定位区域示意（局部放大）
    ax = axes[2]
    if best["S2"] is not None and max_score > 0:
        bx, by = best["S2"]
        d_all = region_diameters(S1, THETA1, np.array(best["S2"]), G, ANGLE_ERROR_DEG)
        recv = np.hypot(G[:, 0] - bx, G[:, 1] - by) <= RECEIVE_MAX
        idx = np.where(recv & (d_all <= args.dmax))[0]
        pick = idx[np.argsort(d_all[idx])[len(idx) // 2]]
        Gv = G[pick]

        verts = region_vertices_single(S1, THETA1, np.array(best["S2"]), Gv, ANGLE_ERROR_DEG)
        cx, cy = Gv
        win = max(60.0, 4 * (polygon_diameter(verts) if len(verts) >= 3 else args.dmax))

        a1 = math.radians(THETA1)
        for sign in (-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG):
            a = math.radians(THETA1 + sign)
            ax.plot([S1[0], S1[0] + 4000 * math.cos(a)],
                    [S1[1], S1[1] + 4000 * math.sin(a)], lw=1.0, color="tab:blue", alpha=0.8)
        th2 = math.atan2(Gv[1] - by, Gv[0] - bx)
        for sign in (-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG):
            a = th2 + math.radians(sign)
            ax.plot([bx, bx + 4000 * math.cos(a)], [by, by + 4000 * math.sin(a)],
                    lw=1.0, color="tab:green", alpha=0.8)
        if len(verts) >= 3:
            ax.add_patch(MplPolygon(verts, closed=True, facecolor="red", alpha=0.55,
                                    edgecolor="darkred", linewidth=2.0, zorder=5))
        ax.plot(*Gv, marker="x", ms=12, color="black", zorder=7)
        ax.annotate(f"G({Gv[0]:.1f},{Gv[1]:.1f})", Gv, textcoords="offset points",
                    xytext=(8, 8), fontsize=10, zorder=8)
        ax.plot(*best["S2"], marker="*", ms=16, color="blue", markeredgecolor="white", zorder=7)
        ax.annotate(f"S2({bx:.0f},{by:.0f})", best["S2"], textcoords="offset points",
                    xytext=(8, -14), fontsize=10, zorder=8)
        ax.set_xlim(cx - win, cx + win)
        ax.set_ylim(cy - win, cy + win)
        dd = polygon_diameter(verts)
        ax.set_title(f"(c) 最优 S2 的定位区域局部放大（d = {dd:.2f} m <= {args.dmax:g} m）",
                     fontsize=13)
    else:
        ax.text(0.5, 0.5, "无可行候选区域", ha="center", va="center", fontsize=14)
        ax.set_title("(c) 示例", fontsize=13)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180)
    print(f"\n图片已保存：{FIGURE_PATH}")

    write_report(args, best, max_score, nG, thr, cand, xs, ys, comps)

    if not args.noshow:
        plt.show()
    plt.close(fig)


def write_report(args, best, max_score, nG, thr, cand, xs, ys, comps):
    # 理论下界：区域直径 d >= 2*min(r1,r2)*sin(1°)，故 d<=20m 要求两条腿都 <= 10/sin(1°)
    r_bound = 10.0 / math.sin(math.radians(ANGLE_ERROR_DEG))

    lines = [
        "# 问题2 第二检测点候选区域报告",
        "",
        "## 建模口径",
        "",
        f"- 第一个检测点 S1 = {(float(S1[0]), float(S1[1]))}，示向度 theta1 = {THETA1}°，误差 ±{ANGLE_ERROR_DEG}°；",
        f"- 候选干扰源 G 在 S1 的 ±1° 扇形内采样 {nG} 个点，距离范围 {DIST_MIN}~{DIST_MAX} m；",
        f"- 定位区域 = 两个 ±1° 扇形的交集，其直径记为 d；判据 d <= {args.dmax:g} m（光学精确定位/清除半径）；",
        f"- 得分 score(S2) = 使“S2 可接收(|S2-G|<={RECEIVE_MAX:g})且 d<={args.dmax:g} m”成立的 G 点数；",
        f"- 候选区域 = 得分不低于 {COVERAGE_RATIO:.0%} 最大得分（即 {thr:.0f} 分）的 S2 位置集合；",
        f"- 网格步长 {args.step:g} m，S2 侧示向度误差按 ±{ANGLE_ERROR_DEG}° 取最坏情况。",
        "",
        "## 关键结论",
        "",
        f"- 由 d >= 2*min(r1,r2)*sin(1°)，要把区域压到 {args.dmax:g} m 以内，"
        f"两条腿都必须满足 r <= {r_bound:.0f} m，"
        f"因此远离 S1 的干扰源用两点交会无法达到 {args.dmax:g} m；",
        f"- 单点得分上限：{max_score} / {nG}（覆盖率 {max_score / nG:.1%}）；",
        f"- 最优 S2：{best['S2']}，得分 {best['score']}；",
        f"- 候选区域网格点数：{int(cand.sum())}，面积约 {cand.sum() * args.step * args.step / 1e6:.3f} km²，"
        f"共 {len(comps)} 瓣（对称分布于示向度射线两侧）。",
        "",
        "## 候选区域明细（示向度参考系：u 沿 theta1，v 垂直、左正右负）",
        "",
        "| 瓣 | 中心(x,y) | 中心(u,v) | 面积(km²) |",
        "| --- | --- | --- | --- |",
    ]
    for k, pts in enumerate(comps, 1):
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        u, v = to_ray_frame(cx, cy, THETA1)
        lines.append(
            f"| {k} | ({cx:.0f}, {cy:.0f}) | ({u:.0f}, {v:.0f}) | "
            f"{len(pts) * args.step ** 2 / 1e6:.3f} |"
        )
    lines += [
        "",
        "## 选择策略",
        "",
        f"1. 在 S1 测得示向度 theta1 后，前往上表任一候选瓣的中心附近（优先 {best['S2']}）；",
        "2. 在该处对同一频道再次 /measure，得到第二个示向度；",
        f"3. 两点交会用问题1的算法求定位区域；对近距离干扰源其直径可控制在 {args.dmax:g} m 内，"
        f"直接到区域中心一次 /clear 即可命中；",
        "4. 若定位区域仍偏大（如距离较远的干扰源），应继续增加检测点，"
        "而不是强行用两点清除。",
        "",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"报告已保存：{REPORT_PATH}")


if __name__ == "__main__":
    main()
