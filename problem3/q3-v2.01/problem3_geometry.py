"""问题 3 几何工具库。

本模块只做“保守”的几何近似，核心不变量是：

    **程序保存的可行域永远包含干扰源的真实位置。**

因此所有会引入误差的操作都朝“把区域放大”的方向取近似，具体约定：

* 圆用**外切正多边形**表示（多边形 ⊇ 圆），求交后区域只会比真实可行域大；
* 方向扇形写成两个半平面，用 Sutherland-Hodgman 对凸多边形裁剪；
* 排除圆（`no_signal` 排除的 1000 m、清除失败排除的 20 m）只记录，
  不参与凸多边形运算——排除只会“缩小”可行域，忽略它保证包含性，最多多算几步；
* 最小包围圆对凸多边形用确定性 Welzl 增量算法，结果一定覆盖多边形。

坐标一律为 numpy `(2,)` 或 `(N, 2)` 数组，单位米；角度参数用弧度还是度在函数名/文档里写明。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

TAU = 2.0 * math.pi

Point = np.ndarray
Polygon = np.ndarray

# 裁剪容差：坐标量级 ~1e3，1e-9 的绝对容差足够判断“点在边界上”
_CLIP_EPS = 1e-9
# 多边形去重容差
_DEDUP_EPS = 1e-7


# --------------------------------------------------------------------------- #
# 基础向量
# --------------------------------------------------------------------------- #
def unit_from_deg(angle_deg: float) -> Point:
    """由角度（度）生成单位向量。"""
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)], dtype=float)


def unit_from_rad(angle_rad: float) -> Point:
    return np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)


def unit(v: Sequence[float]) -> Point:
    v = np.asarray(v, dtype=float)
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-12:
        raise ValueError("零向量无法归一化")
    return v / n


def perpendicular_left(v: Sequence[float]) -> Point:
    """逆时针旋转 90°。"""
    v = np.asarray(v, dtype=float)
    return np.array([-v[1], v[0]], dtype=float)


def wrap_angle_deg(angle_deg: float) -> float:
    """把角度规整到 [0, 360)。"""
    return angle_deg % 360.0


def signed_angle_diff_deg(a: float, b: float) -> float:
    """a - b 规整到 (-180, 180]，用于比较示向度。"""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


# --------------------------------------------------------------------------- #
# 圆与多边形
# --------------------------------------------------------------------------- #
def circumscribed_polygon(center: Sequence[float], radius: float, n: int = 128) -> Polygon:
    """半径 radius 的圆的外切正 n 边形（多边形包含整个圆）。

    顶点到圆心距离为 radius / cos(pi/n)，n 越大越贴合圆。
    """
    center = np.asarray(center, dtype=float)
    factor = 1.0 / math.cos(math.pi / n)
    ang = TAU * np.arange(n) / n
    ring = factor * radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return center + ring


def clip_halfplane(poly: Polygon, a: float, b: float, c: float) -> Polygon:
    """用半平面 ``a*x + b*y + c >= 0`` 裁剪凸多边形（Sutherland-Hodgman）。"""
    if poly is None or len(poly) == 0:
        return np.empty((0, 2))

    out: List[Point] = []
    m = len(poly)
    for i in range(m):
        p = poly[i]
        q = poly[(i + 1) % m]
        fp = a * p[0] + b * p[1] + c
        fq = a * q[0] + b * q[1] + c
        p_in = fp >= -_CLIP_EPS
        q_in = fq >= -_CLIP_EPS
        if p_in:
            out.append(p)
        if p_in != q_in:
            denom = fp - fq
            if abs(denom) > 1e-15:
                t = fp / denom
                out.append(p + t * (q - p))
    return _clean_polygon(out)


def clip_wedge(poly: Polygon, apex: Sequence[float], theta_rad: float,
               half_rad: float) -> Polygon:
    """保留以 apex 为顶点、沿 theta_rad、张角 ±half_rad 的**前向射线扇形**。

    扇形 = 两个半平面的交，天然是“从测点向前延伸”的射线扇形，
    而不是穿过测点的无限直线带。
    """
    apex = np.asarray(apex, dtype=float)
    u_minus = unit_from_rad(theta_rad - half_rad)
    u_plus = unit_from_rad(theta_rad + half_rad)

    # cross(u_minus, P-apex) >= 0
    poly = clip_halfplane(
        poly,
        -u_minus[1], u_minus[0],
        u_minus[1] * apex[0] - u_minus[0] * apex[1],
    )
    # cross(u_plus, P-apex) <= 0  ⇔  -cross(u_plus, P-apex) >= 0
    poly = clip_halfplane(
        poly,
        u_plus[1], -u_plus[0],
        -(u_plus[1] * apex[0] - u_plus[0] * apex[1]),
    )
    return poly


def intersect_convex(poly_a: Polygon, poly_b: Polygon) -> Polygon:
    """两个逆时针凸多边形的交（用 poly_b 的每条边裁剪 poly_a）。"""
    if poly_a is None or poly_b is None or len(poly_a) == 0 or len(poly_b) == 0:
        return np.empty((0, 2))

    out = poly_a
    m = len(poly_b)
    for i in range(m):
        p = poly_b[i]
        q = poly_b[(i + 1) % m]
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        # 逆时针多边形内部在边左侧：cross((dx,dy), X-p) >= 0
        out = clip_halfplane(out, -dy, dx, dy * p[0] - dx * p[1])
        if len(out) == 0:
            return out
    return out


def intersect_disk(poly: Polygon, center: Sequence[float], radius: float,
                   n: int = 128) -> Polygon:
    """poly 与盘(center, radius)相交，盘用外切正多边形保守表示。"""
    return intersect_convex(poly, circumscribed_polygon(center, radius, n))


def _clean_polygon(points: Sequence[Point]) -> Polygon:
    """去掉相邻重复点，并把点列表转成 (N,2) 数组。"""
    if not points:
        return np.empty((0, 2))
    kept: List[Point] = []
    for p in points:
        if not kept or math.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1]) > _DEDUP_EPS:
            kept.append(p)
    if len(kept) >= 2:
        first, last = kept[0], kept[-1]
        if math.hypot(first[0] - last[0], first[1] - last[1]) <= _DEDUP_EPS:
            kept.pop()
    if not kept:
        return np.empty((0, 2))
    return np.asarray(kept, dtype=float)


def polygon_area(poly: Polygon) -> float:
    """多边形面积（逆时针为正）。"""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly: Polygon) -> Point:
    if poly is None or len(poly) == 0:
        raise ValueError("空多边形没有质心")
    if len(poly) < 3:
        return poly.mean(axis=0)
    x = poly[:, 0]
    y = poly[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    area2 = float(cross.sum())
    if abs(area2) < 1e-12:
        return poly.mean(axis=0)
    cx = float(((x + x1) * cross).sum()) / (3.0 * area2)
    cy = float(((y + y1) * cross).sum()) / (3.0 * area2)
    return np.array([cx, cy], dtype=float)


# --------------------------------------------------------------------------- #
# 最小包围圆（确定性 Welzl 增量算法）
# --------------------------------------------------------------------------- #
def _dist_to(pt: Point, cx: float, cy: float) -> float:
    return math.hypot(pt[0] - cx, pt[1] - cy)


def _circle_from_two(p: Point, q: Point) -> Tuple[float, float, float]:
    cx = 0.5 * (p[0] + q[0])
    cy = 0.5 * (p[1] + q[1])
    return cx, cy, math.hypot(p[0] - q[0], p[1] - q[1]) * 0.5


def _circle_from_three(p: Point, q: Point, r: Point) -> Tuple[float, float, float]:
    ax, ay = float(p[0]), float(p[1])
    bx, by = float(q[0]), float(q[1])
    cx, cy = float(r[0]), float(r[1])
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        # 三点共线：退化为覆盖这三点所需的最小圆（由最远两点决定）
        best = _circle_from_two(p, q)
        best = min(
            [best, _circle_from_two(p, r), _circle_from_two(q, r)],
            key=lambda c: c[2],
        )
        cands = [best, _circle_from_two(p, q), _circle_from_two(p, r), _circle_from_two(q, r)]
        valid = [c for c in cands if _dist_to(p, c[0], c[1]) <= c[2] + 1e-7
                 and _dist_to(q, c[0], c[1]) <= c[2] + 1e-7
                 and _dist_to(r, c[0], c[1]) <= c[2] + 1e-7]
        return min(valid, key=lambda c: c[2])

    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def min_enclosing_circle(points: Sequence[Point]) -> Optional[Tuple[Point, float]]:
    """返回覆盖全部点的最小圆 (center, radius)；无点返回 None。"""
    if points is None or len(points) == 0:
        return None
    pts = [(float(p[0]), float(p[1])) for p in points]

    cx = cy = 0.0
    radius = -1.0
    for i, p in enumerate(pts):
        if radius < 0 or _dist_to(p, cx, cy) > radius + 1e-7:
            cx, cy, radius = p[0], p[1], 0.0
            for j in range(i):
                q = pts[j]
                if _dist_to(q, cx, cy) > radius + 1e-7:
                    cx, cy, radius = _circle_from_two(p, q)
                    for k in range(j):
                        r = pts[k]
                        if _dist_to(r, cx, cy) > radius + 1e-7:
                            cx, cy, radius = _circle_from_three(p, q, r)
    return np.array([cx, cy], dtype=float), float(radius)


def farthest_pair(poly: Polygon) -> Tuple[Point, float]:
    """凸多边形上距离最远的一对顶点：返回 (单位方向, 距离)。"""
    if poly is None or len(poly) < 2:
        return np.array([1.0, 0.0]), 0.0
    best_d = -1.0
    best = (poly[0], poly[0])
    pts = poly
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d > best_d:
                best_d = d
                best = (pts[i], pts[j])
    direction = best[1] - best[0]
    if math.hypot(direction[0], direction[1]) < 1e-12:
        return np.array([1.0, 0.0]), 0.0
    return unit(direction), float(best_d)


# --------------------------------------------------------------------------- #
# 网格覆盖与路径
# --------------------------------------------------------------------------- #
def grid_cover_points(poly: Polygon, spacing: float,
                      inside_only: bool = False) -> List[Tuple[float, float]]:
    """生成覆盖 poly 包围盒的方格点（逐行蛇形顺序），保证覆盖整个包围盒。

    spacing 取 ``20*sqrt(2)*(1-eps)`` 时，包围盒内任意点到最近方格点距离严格小于 20 m。
    ``inside_only=False`` 时返回包围盒内全部方格点——这样无论可行域形状如何，
    覆盖性都由“包围盒被完整覆盖”保证。
    """
    if poly is None or len(poly) == 0:
        return []
    xs_min, ys_min = poly.min(axis=0)
    xs_max, ys_max = poly.max(axis=0)

    nx = int(math.floor((xs_max - xs_min) / spacing)) + 1
    ny = int(math.floor((ys_max - ys_min) / spacing)) + 1
    xs = xs_min + spacing * np.arange(nx + 1)
    ys = ys_min + spacing * np.arange(ny + 1)
    xs[-1] = max(xs[-1], xs_max)
    ys[-1] = max(ys[-1], ys_max)

    pts: List[Tuple[float, float]] = []
    for iy, y in enumerate(ys):
        row = xs if iy % 2 == 0 else xs[::-1]
        for x in row:
            if inside_only and not point_in_polygon((x, y), poly):
                continue
            pts.append((float(x), float(y)))
    return pts


def point_in_polygon(pt: Sequence[float], poly: Polygon) -> bool:
    """射线法判断点是否在（凸）多边形内或边界上。"""
    if poly is None or len(poly) < 3:
        return False
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def distance_point_to_polygon(pt: Sequence[float], poly: Polygon) -> float:
    """点到多边形（边界与内部）的距离，点在内部返回 0。"""
    if poly is None or len(poly) == 0:
        return float("inf")
    if point_in_polygon(pt, poly):
        return 0.0
    px, py = float(pt[0]), float(pt[1])
    best = float("inf")
    n = len(poly)
    for i in range(n):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-12:
            d = math.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
            d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        best = min(best, d)
    return best


def nearest_neighbor_route(start: Sequence[float],
                           points: Sequence[Sequence[float]]) -> List[int]:
    """最近邻顺序，返回 points 的下标排列。"""
    n = len(points)
    if n == 0:
        return []
    remaining = list(range(n))
    route: List[int] = []
    cur = np.asarray(start, dtype=float)
    while remaining:
        idx = min(
            remaining,
            key=lambda i: math.hypot(points[i][0] - cur[0], points[i][1] - cur[1]),
        )
        route.append(idx)
        cur = np.asarray(points[idx], dtype=float)
        remaining.remove(idx)
    return route


def route_length(route: Sequence[int], start: Sequence[float],
                 points: Sequence[Sequence[float]]) -> float:
    if not route:
        return 0.0
    total = 0.0
    cur = np.asarray(start, dtype=float)
    for i in route:
        nxt = np.asarray(points[i], dtype=float)
        total += math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
        cur = nxt
    return total


def two_opt_open(route: List[int], start: Sequence[float],
                 points: Sequence[Sequence[float]]) -> List[int]:
    """开放路径 2-opt：不要求回到起点。"""
    if len(route) < 3:
        return route
    best = route[:]
    best_len = route_length(best, start, points)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for k in range(i + 1, len(best)):
                candidate = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                cand_len = route_length(candidate, start, points)
                if cand_len + 1e-9 < best_len:
                    best, best_len = candidate, cand_len
                    improved = True
    return best
