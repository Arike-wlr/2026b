"""问题 1 示意图：交会定位法的几何示意（3 个检测点，示向度误差 ±1°）。

画出：
    1. 三个检测点 S1/S2/S3；
    2. 每个点的主射线（实线）与 ±1° 边界射线（虚线）；
    3. 三个误差扇形（半透明填充）；
    4. 三个扇形的交集多边形（红色醒目边框）—— 即定位区域，并在插图中放大；
    5. 该多边形的直径线段，以及"以直径为直径"的圆；
    6. 额外画出最小覆盖圆（用于回答问题 1 第二问"能否覆盖"）。

终端同时打印：定位区域顶点、直径 d、最小覆盖圆半径 R、结论。

用法：
    python problem1_figure.py           # 使用下面 DETECTIONS 里给定的示向度
    python problem1_figure.py --fix      # 若三者不相交，自动把最后一个点的示向度
                                         # 修正为"指向前两条主射线交点"的真实方位角
    python problem1_figure.py --noshow   # 只保存图片，不弹窗
"""

import math
import os
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon as MplPolygon
from shapely.geometry import Polygon

sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
# (名称, x, y, 示向度°)  示向度 = 从检测点指向干扰源的方位角，x 轴正向逆时针，范围 [0,360)
DETECTIONS = [
    ("S1", 300.0, 400.0, 45.0),
    ("S2", 700.0, 100.0, 135.0),
    ("S3", 500.0, 800.0, 270.0),
]

ERROR_DEG = 1.0        # 示向度误差 ±1°
RAY_LENGTH = 1600.0    # 扇形 / 射线的绘制半径（米）
ARC_STEPS = 180        # 扇形圆弧离散段数

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OUT_NAME = "problem1_intersection.png"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
LABEL_OFFSET = {"S1": (12, -30), "S2": (12, 10), "S3": (12, 10)}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def bearing_deg(x0, y0, x1, y1):
    """从 (x0,y0) 指向 (x1,y1) 的方位角（度，[0,360)）。"""
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360.0


def sector_polygon(x, y, theta_deg, err_deg, radius, steps=ARC_STEPS):
    """以 (x,y) 为顶点、方位角 theta±err、半径 radius 的扇形多边形。"""
    a0 = math.radians(theta_deg - err_deg)
    a1 = math.radians(theta_deg + err_deg)
    pts = [(x, y)]
    for i in range(steps + 1):
        a = a0 + (a1 - a0) * i / steps
        pts.append((x + radius * math.cos(a), y + radius * math.sin(a)))
    return Polygon(pts)


def ray_end(x, y, theta_deg, length):
    a = math.radians(theta_deg)
    return (x + length * math.cos(a), y + length * math.sin(a))


def ray_ray_intersection(p1, d1, p2, d2):
    """两条射线 p1+t*d1 与 p2+s*d2 的交点；平行或无正向交点返回 None。"""
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return None
    wx, wy = p2[0] - p1[0], p2[1] - p1[1]
    t = (wx * d2[1] - wy * d2[0]) / cross
    s = (wx * d1[1] - wy * d1[0]) / cross
    if t < 0 or s < 0:
        return None
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def polygon_diameter(vertices):
    """凸多边形直径：顶点两两距离的最大值，返回 (d, A, B)。

    凸集的直径一定在极点（顶点）处取到，故枚举顶点即可。
    """
    best_d, best_pair = 0.0, (vertices[0], vertices[0])
    n = len(vertices)
    for i in range(n):
        xi, yi = vertices[i]
        for j in range(i + 1, n):
            xj, yj = vertices[j]
            dist = math.hypot(xi - xj, yi - yj)
            if dist > best_d:
                best_d, best_pair = dist, (vertices[i], vertices[j])
    return best_d, best_pair[0], best_pair[1]


def _circle_two(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0,
            math.hypot(a[0] - b[0], a[1] - b[1]) / 2.0)


def _circle_three(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy)
          + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx)
          + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy, math.hypot(ax - ux, ay - uy))


def _outside(p, circle):
    return math.hypot(p[0] - circle[0], p[1] - circle[1]) > circle[2] + 1e-9


def _mec_two_boundary(points, q1, q2):
    circle = _circle_two(q1, q2)
    for p in points:
        if _outside(p, circle):
            cand = _circle_three(q1, q2, p)
            if cand is not None:
                circle = cand
    return circle


def _mec_one_boundary(points, q):
    circle = (q[0], q[1], 0.0)
    for i, p in enumerate(points):
        if _outside(p, circle):
            if circle[2] == 0.0:
                circle = _circle_two(p, q)
            else:
                circle = _mec_two_boundary(points[:i], p, q)
    return circle


def minimum_enclosing_circle(points):
    """Welzl 随机增量算法，返回 (cx, cy, r)。"""
    pts = list(points)
    random.shuffle(pts)
    circle = (0.0, 0.0, -1.0)
    for i, p in enumerate(pts):
        if circle[2] < 0 or _outside(p, circle):
            circle = _mec_one_boundary(pts[:i], p)
    return circle


def largest_polygon(geom):
    """从交集结果里取出面积最大的 Polygon；没有则返回 None。"""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
        return max(polys, key=lambda g: g.area) if polys else None
    return None


# --------------------------------------------------------------------------- #
# 绘制工具
# --------------------------------------------------------------------------- #
def draw_sectors(target, detections, sectors):
    """在 target 坐标系上画三个扇形及其主射线 / 边界射线。"""
    for idx, (name, x, y, theta) in enumerate(detections):
        color = COLORS[idx % len(COLORS)]
        sx, sy = sectors[idx].exterior.xy
        target.fill(sx, sy, color=color, alpha=0.15, zorder=1)

        for sign in (-ERROR_DEG, ERROR_DEG):
            ex, ey = ray_end(x, y, theta + sign, RAY_LENGTH)
            target.plot([x, ex], [y, ey], ls="--", lw=1.0, color=color,
                        alpha=0.85, zorder=2)

        mx, my = ray_end(x, y, theta, RAY_LENGTH)
        target.plot([x, mx], [y, my], ls="-", lw=1.8, color=color, zorder=3)


def draw_region(target, verts, d, a, b, mid, r_min, show_labels=True):
    """画定位区域、直径、以 d 为直径的圆、最小覆盖圆。"""
    target.add_patch(MplPolygon(verts, closed=True, facecolor="red", alpha=0.22,
                                edgecolor="red", linewidth=2.4, zorder=5))
    target.plot([a[0], b[0]], [a[1], b[1]], color="black",
                lw=2.2 if show_labels else 1.4, zorder=8)
    target.add_patch(Circle(mid, d / 2.0, fill=False, edgecolor="darkorange",
                            linestyle="--", lw=2.0, zorder=7))
    target.add_patch(Circle((r_min[0], r_min[1]), r_min[2], fill=False,
                            edgecolor="green", linestyle=":", lw=1.8, zorder=7))


def build_legend(covered, d, r_min):
    handles = [
        Line2D([], [], color=COLORS[0], ls="-", lw=1.8, label="主射线（示向度）"),
        Line2D([], [], color=COLORS[0], ls="--", lw=1.0, label="示向度边界 ±1°"),
        Line2D([], [], color="gray", marker="o", ls="", ms=8,
               markeredgecolor="black", label="检测点"),
        Patch(facecolor="red", alpha=0.22, edgecolor="red",
              lw=2.4, label="定位区域（三扇形交集）"),
        Line2D([], [], color="black", lw=2.2, label=f"直径 d = {d:.2f} m"),
        Line2D([], [], color="darkorange", ls="--", lw=2.0,
               label=f"以 d 为直径的圆 R = d/2 = {d/2:.2f} m"),
        Line2D([], [], color="green", ls=":", lw=1.8,
               label=f"最小覆盖圆 R = {r_min:.2f} m（{'能' if covered else '不能'}覆盖）"),
    ]
    return handles


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    auto_fix = "--fix" in sys.argv
    show = "--noshow" not in sys.argv

    detections = [list(d) for d in DETECTIONS]

    # 三者不相交时的自动修正：把最后一个点的示向度改成"指向前两条主射线交点"
    if auto_fix and len(detections) >= 3:
        n1, x1, y1, t1 = detections[0]
        n2, x2, y2, t2 = detections[1]
        n3, x3, y3, _ = detections[-1]
        d1 = (math.cos(math.radians(t1)), math.sin(math.radians(t1)))
        d2 = (math.cos(math.radians(t2)), math.sin(math.radians(t2)))
        g = ray_ray_intersection((x1, y1), d1, (x2, y2), d2)
        if g is not None:
            detections[-1][3] = bearing_deg(x3, y3, g[0], g[1])
            print(f"[修正] 前两条主射线交于 G=({g[0]:.2f}, {g[1]:.2f})，"
                  f"{n3} 的示向度改为 {detections[-1][3]:.4f}°\n")

    sectors = [sector_polygon(x, y, t, ERROR_DEG, RAY_LENGTH)
               for _, x, y, t in detections]
    inter = sectors[0]
    for s in sectors[1:]:
        inter = inter.intersection(s)
    poly = largest_polygon(inter)

    # ---------------------------------------------------------------- 出图
    fig, ax = plt.subplots(figsize=(11, 9))
    draw_sectors(ax, detections, sectors)

    for idx, (name, x, y, theta) in enumerate(detections):
        color = COLORS[idx % len(COLORS)]
        ax.plot(x, y, marker="o", ms=9, color=color, markeredgecolor="black",
                markeredgewidth=1.2, zorder=6)
        ax.annotate(f"{name}({x:g},{y:g})\nθ={theta:.4g}°", (x, y),
                    textcoords="offset points",
                    xytext=LABEL_OFFSET.get(name, (12, 10)),
                    fontsize=10, zorder=7,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))

    if poly is None:
        ax.set_title("交会定位示意：三个 ±1° 误差扇形无公共交集（当前示向度不自洽）",
                     fontsize=13)
        print("三个扇形没有公共交集——给定的三个示向度互不自洽，定位区域为空。")
        print("可尝试：python problem1_figure.py --fix")
    else:
        verts = list(poly.exterior.coords)[:-1]
        d, a, b = polygon_diameter(verts)
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        cx, cy, r_min = minimum_enclosing_circle(verts)
        covered = r_min <= d / 2.0 + 1e-6

        draw_region(ax, verts, d, a, b, mid, (cx, cy, r_min))

        # 主图上用一个小红圈指出定位区域位置（真实尺寸太小，细节见插图）
        ax.add_patch(Circle((cx, cy), 30.0, fill=False, edgecolor="red",
                            lw=1.6, zorder=10))

        # ------- 放大插图：看清定位区域、直径与两个圆 -------
        axins = ax.inset_axes([0.58, 0.60, 0.40, 0.32])
        draw_sectors(axins, detections, sectors)
        draw_region(axins, verts, d, a, b, mid, (cx, cy, r_min), show_labels=False)
        bx0, by0, bx1, by1 = poly.bounds
        half = max(bx1 - bx0, by1 - by0) * 0.75 + 4.0
        mx0, my0 = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        axins.set_xlim(mx0 - half, mx0 + half)
        axins.set_ylim(my0 - half, my0 + half)
        axins.set_aspect("equal", adjustable="box")
        axins.grid(True, ls=":", alpha=0.5)
        axins.tick_params(labelsize=7)
        axins.set_title("定位区域放大", fontsize=9)
        ax.indicate_inset_zoom(axins, edgecolor="red", alpha=0.9)

        ax.set_title("交会定位法示意图（3 个检测点，示向度误差 ±1°）\n"
                     f"d = {d:.2f} m，最小覆盖圆 R = {r_min:.2f} m，"
                     f"以 d 为直径的圆{'能' if covered else '不能'}覆盖定位区域",
                     fontsize=13)
        ax.legend(handles=build_legend(covered, d, r_min),
                  loc="upper left", fontsize=8.5, framealpha=0.92)

        print(f"定位区域：{len(verts)} 个顶点，面积 {poly.area:.2f} m²")
        print("顶点： " + "，".join(f"({vx:.2f}, {vy:.2f})" for vx, vy in verts))
        print(f"直径 d            = {d:.4f} m，端点 A=({a[0]:.2f},{a[1]:.2f}) "
              f"B=({b[0]:.2f},{b[1]:.2f})")
        print(f"最小覆盖圆 R      = {r_min:.4f} m，圆心 ({cx:.2f}, {cy:.2f})")
        print(f"d/2               = {d/2:.4f} m")
        print(f"Jung 上界 d/√3    = {d/math.sqrt(3):.4f} m")
        print(f"结论：以 d 为直径的圆{'能' if covered else '不能'}覆盖该定位区域")

    # ------------------------------------------------------------------ 视图
    xs_all = [x for _, x, _, _ in detections]
    ys_all = [y for _, _, y, _ in detections]
    if poly is not None:
        xs_all += [p[0] for p in poly.exterior.coords]
        ys_all += [p[1] for p in poly.exterior.coords]
    pad = 220.0
    ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
    ax.set_ylim(min(ys_all) - pad, max(ys_all) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("x (m，正东)", fontsize=11)
    ax.set_ylabel("y (m，正北)", fontsize=11)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, OUT_NAME)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"\n图片已保存：{path}")

    if show:
        plt.show()


if __name__ == "__main__":
    main()
