"""问题 3 单元测试与本地仿真验证（对应规格第 13 节“必测案例”）。

运行::

    python problem3/test_problem3.py            # 直接运行（unittest 主程序）
    python -m pytest problem3/test_problem3.py  # 如安装了 pytest
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from problem3_geometry import (  # noqa: E402
    clip_wedge,
    circumscribed_polygon,
    grid_cover_points,
    intersect_convex,
    min_enclosing_circle,
    point_in_polygon,
    polygon_area,
)
from problem3_local_sim import LocalSimulator, run_case  # noqa: E402
from problem3_strategy import (  # noqa: E402
    BEARING_ERROR_DEG,
    CLEAR_RADIUS,
    MAX_RECEIVE_RADIUS,
    MIN_RECEIVE_RADIUS,
    NEAR_RADIUS,
    PROBE_BACKUP_FORWARD,
    PROBE_BACKUP_SIDE,
    PROBE_FORWARD_RATIO,
    PROBE_R_EST_MAX,
    PROBE_R_EST_MIN,
    PROBE_SIDE_MAX,
    PROBE_SIDE_RATIO,
    SAFE_REGION_RADIUS,
    TARGET_RADIUS,
    ChannelState,
    Problem3Strategy,
    build_survey_points,
    GRID_SPACING,
)


# --------------------------------------------------------------------------- #
# 13.1 覆盖测试
# --------------------------------------------------------------------------- #
class TestCoverage(unittest.TestCase):
    def setUp(self) -> None:
        self.points = build_survey_points()

    def test_origin_found_by_center(self) -> None:
        d = min(math.hypot(p[0], p[1]) for p in self.points)
        self.assertLessEqual(d, MIN_RECEIVE_RADIUS)

    def test_worst_boundary_midpoint(self) -> None:
        # 边界上、位于相邻两个外围点中间的“最不利点”
        ang = math.radians(30.0)
        g = np.array([TARGET_RADIUS * math.cos(ang), TARGET_RADIUS * math.sin(ang)])
        d = min(math.hypot(g[0] - p[0], g[1] - p[1]) for p in self.points)
        self.assertAlmostEqual(d, 988.511, delta=0.05)
        self.assertLess(d, MIN_RECEIVE_RADIUS)

    def test_dense_scan_whole_disk(self) -> None:
        # 对目标圆域做高密度角度/半径扫描
        radii = np.linspace(0.0, TARGET_RADIUS, 240)
        angles = np.linspace(0.0, 2.0 * math.pi, 1440, endpoint=False)
        rr, aa = np.meshgrid(radii, angles, indexing="ij")
        pts = np.stack([rr * np.cos(aa), rr * np.sin(aa)], axis=-1).reshape(-1, 2)
        pts_arr = np.asarray([[p[0], p[1]] for p in self.points])
        diff = pts[:, None, :] - pts_arr[None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        min_dist = dist.min(axis=1)
        self.assertLessEqual(float(min_dist.max()), MIN_RECEIVE_RADIUS + 1e-6)


# --------------------------------------------------------------------------- #
# 13.2 测向测试
# --------------------------------------------------------------------------- #
class TestBearing(unittest.TestCase):
    def test_wedge_wraparound(self) -> None:
        # theta=1°，±2° 覆盖 [359°, 3°]；检查 359.5° 在内、10° 在外
        poly = circumscribed_polygon((0.0, 0.0), TARGET_RADIUS)
        wedge = clip_wedge(poly, np.array([0.0, 0.0]), math.radians(1.0),
                           math.radians(2.0))
        self.assertTrue(polygon_area(wedge) > 0)
        inside = np.array([100.0 * math.cos(math.radians(359.5)),
                           100.0 * math.sin(math.radians(359.5))])
        outside = np.array([100.0 * math.cos(math.radians(10.0)),
                            100.0 * math.sin(math.radians(10.0))])
        self.assertTrue(point_in_polygon(inside, wedge))
        self.assertFalse(point_in_polygon(outside, wedge))

    def test_wedge_is_ray_not_line(self) -> None:
        # 前向扇形不应包含测点后方的点
        poly = circumscribed_polygon((0.0, 0.0), TARGET_RADIUS)
        wedge = clip_wedge(poly, np.array([0.0, 0.0]), 0.0, math.radians(1.0))
        behind = np.array([-100.0, 0.0])
        self.assertFalse(point_in_polygon(behind, wedge))

    def test_near_parallel_wedges(self) -> None:
        # 两个几乎平行、略错开的测向扇形：只在远处相交，区域细长；不应崩溃且半径有限
        poly = circumscribed_polygon((0.0, 0.0), TARGET_RADIUS)
        w1 = clip_wedge(poly, np.array([0.0, 0.0]), 0.0, math.radians(1.0))
        w2 = clip_wedge(w1, np.array([0.0, 1.0]), 0.0, math.radians(1.0))
        self.assertGreater(polygon_area(w2), 0.0)
        c, r = min_enclosing_circle(w2)
        self.assertTrue(np.isfinite(r))

    def test_parallel_wedges_far_apart_empty(self) -> None:
        # 严格平行、错开太远的两个扇形没有交集：应优雅返回空区域
        poly = circumscribed_polygon((0.0, 0.0), TARGET_RADIUS)
        w1 = clip_wedge(poly, np.array([0.0, 0.0]), 0.0, math.radians(1.0))
        w2 = clip_wedge(w1, np.array([0.0, 500.0]), 0.0, math.radians(1.0))
        self.assertEqual(polygon_area(w2), 0.0)
        self.assertIsNone(min_enclosing_circle(w2))

    def test_repeat_measure_does_not_shrink(self) -> None:
        # 同一点重复测向得到相同误差 → 第二次裁剪不应改变区域
        f0 = circumscribed_polygon((0.0, 0.0), TARGET_RADIUS)
        S = np.array([500.0, 200.0])
        theta = math.radians(37.0)
        f1 = clip_wedge(f0, S, theta, math.radians(BEARING_ERROR_DEG))
        f2 = clip_wedge(f1, S, theta, math.radians(BEARING_ERROR_DEG))
        self.assertAlmostEqual(polygon_area(f1), polygon_area(f2), places=6)

    def test_q2_probe_points_use_dynamic_pull_aside(self) -> None:
        # 问题2思想不是固定坐标，而是估计距离后侧向拉偏，形成接近 90° 的交会角。
        sim = LocalSimulator(seed=1, num_sources=10)
        strategy = Problem3Strategy(sim)
        cs = ChannelState(channel=1)
        S = np.array([0.0, 0.0])
        u = np.array([1.0, 0.0])
        cs.first_direction = (S, u)
        cs.enclosing = (np.array([500.0, 0.0]), 120.0)

        plan = strategy._build_q2_probe_plan(cs, np.array([0.0, 0.0]))  # noqa: SLF001
        q2_candidates = plan[:2]
        forward_values = sorted(round(float(point[0]), 6) for point in q2_candidates)
        side_values = sorted(round(abs(float(point[1])), 6) for point in q2_candidates)

        r_est = 500.0
        self.assertEqual(forward_values, [r_est * PROBE_FORWARD_RATIO] * 2)
        self.assertEqual(side_values, [r_est * PROBE_SIDE_RATIO] * 2)
        self.assertLessEqual(side_values[0], PROBE_SIDE_MAX)
        self.assertGreaterEqual(r_est, PROBE_R_EST_MIN)
        self.assertLessEqual(r_est, PROBE_R_EST_MAX)

    def test_backup_probe_points_always_receivable(self) -> None:
        # 备用补测点承担全距离接收保证：首次测向位置 S、真实距离 r∈(5,1500]、
        # 测向误差 e∈[-1°,1°] 时，Q_plus / Q_minus 到干扰源的最坏距离必须 < 1000 m。
        worst = 0.0
        for r in np.linspace(NEAR_RADIUS + 1e-6, MAX_RECEIVE_RADIUS, 400):
            for e_deg in np.linspace(-BEARING_ERROR_DEG, BEARING_ERROR_DEG, 21):
                u = np.array([math.cos(math.radians(e_deg)),
                              math.sin(math.radians(e_deg))])
                v = np.array([-u[1], u[0]])
                source = np.array([r, 0.0])          # 真实方向取 +x
                for sign in (1.0, -1.0):
                    Q = (PROBE_BACKUP_FORWARD * u + sign * PROBE_BACKUP_SIDE * v)
                    worst = max(worst, float(np.hypot(*(source - Q))))
        self.assertLess(worst, MIN_RECEIVE_RADIUS)

    def test_min_enclosing_circle_known(self) -> None:
        pts = [np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, 10.0])]
        c, r = min_enclosing_circle(pts)
        self.assertAlmostEqual(r, math.hypot(5.0, 5.0), places=6)
        self.assertAlmostEqual(float(c[0]), 5.0, places=6)
        self.assertAlmostEqual(float(c[1]), 5.0, places=6)


# --------------------------------------------------------------------------- #
# 13.3 清除测试
# --------------------------------------------------------------------------- #
class TestClear(unittest.TestCase):
    def _make_sim(self, source_pos, receive=1400.0):
        sim = LocalSimulator(seed=1, num_sources=1)
        sim.sources = {1: type(list(sim.sources.values())[0])(
            channel=1, position=np.asarray(source_pos, dtype=float),
            receive_radius=receive)}
        sim.enter()
        return sim

    def test_near_at_exactly_5m(self) -> None:
        sim = self._make_sim([5.0, 0.0])
        res = sim.measure(0.0, 0.0, 1)
        self.assertEqual(res.result, "near")

    def test_direction_just_beyond_5m(self) -> None:
        sim = self._make_sim([5.01, 0.0])
        res = sim.measure(0.0, 0.0, 1)
        self.assertEqual(res.result, "direction")

    def test_clear_success_at_exactly_20m(self) -> None:
        sim = self._make_sim([20.0, 0.0])
        res = sim.clear(0.0, 0.0, 1)
        self.assertTrue(res.cleared)

    def test_clear_failure_beyond_20m(self) -> None:
        sim = self._make_sim([20.01, 0.0])
        res = sim.clear(0.0, 0.0, 1)
        self.assertFalse(res.cleared)

    def test_two_clear_worst_case_within_20m(self) -> None:
        # R=58 时两次清除法最坏误差 < 20 m
        R = SAFE_REGION_RADIUS
        L = (CLEAR_RADIUS + R) / (2.0 * math.cos(math.radians(BEARING_ERROR_DEG)))
        worst = 0.0
        for r in np.linspace(CLEAR_RADIUS + 1e-6, R, 2000):
            for e in (-math.radians(BEARING_ERROR_DEG), math.radians(BEARING_ERROR_DEG)):
                d2 = L * L + r * r - 2.0 * L * r * math.cos(e)
                worst = max(worst, math.sqrt(max(d2, 0.0)))
        self.assertLess(worst, CLEAR_RADIUS)

    def test_grid_covers_worst_corner(self) -> None:
        poly = circumscribed_polygon((0.0, 0.0), SAFE_REGION_RADIUS)
        grid = np.asarray(grid_cover_points(poly, GRID_SPACING))
        self.assertGreater(len(grid), 0)
        # 网格单元最不利角点（偏移半个间距）到最近网格点的距离必须 < 20 m
        xs_min, ys_min = poly.min(axis=0)
        corners = np.array([
            [xs_min + GRID_SPACING / 2, ys_min + GRID_SPACING / 2],
            [xs_min + GRID_SPACING / 2, ys_min - GRID_SPACING / 2],
        ])
        diff = corners[:, None, :] - grid[None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        self.assertLess(float(dist.min(axis=1).max()), CLEAR_RADIUS)


# --------------------------------------------------------------------------- #
# 13.4 状态机 / 端到端
# --------------------------------------------------------------------------- #
class TestStateMachine(unittest.TestCase):
    def test_survey_station_count(self) -> None:
        self.assertEqual(len(build_survey_points()), 7)

    def test_batch_small_medium_large(self) -> None:
        for n in (10, 13, 16):
            summary = run_case(seed=1000 + n, num_sources=n)
            self.assertEqual(summary["source_cleared"], n,
                             f"{n} 个干扰源未全部清除")
            self.assertTrue(summary["success"])
            # 已清除频道不会被重复清除：成功清除次数 == 干扰源总数
            self.assertEqual(summary["stats"]["clear_success"], n)

    def test_empty_channels_certified(self) -> None:
        # 10 个干扰源 → 10 个空频道，必须全部完成七点检测后判空
        summary = run_case(seed=2026, num_sources=10)
        self.assertEqual(summary["empty_certified"], 20 - 10)
        self.assertEqual(summary["unknown"], 0)
        self.assertTrue(summary["all_resolved"])

    def test_local_batch_no_failure(self) -> None:
        # 小批量快速回归：全部成功率应为 100%
        ok = sum(1 for i in range(12) if run_case(seed=500 + i)["success"])
        self.assertEqual(ok, 12)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
