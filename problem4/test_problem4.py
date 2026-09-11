"""问题 4 单元测试与随机场景验证（对应 ``problem3/test_problem3.py`` 的写法）。

覆盖四类内容：

1. 25 点巡检方案的几何保证（覆盖性、错开角、边长、任意发射方向必检出）；
2. 定向源的物理语义（90° 半圆含边界、遮挡只产生 ``no_signal``）；
3. 清除半径与三角格点兜底覆盖；
4. 端到端随机场景：**每个案例恰好 10 个源**，位置、接收半径、全向/定向搭配
   与发射方向都在运行时随机生成。

运行::

    python problem4/test_problem4.py            # 每次运行随机生成一批场景
    python -m pytest problem4/test_problem4.py  # 如安装了 pytest

复现某一次运行（首行会打印本次种子）::

    PowerShell:  $env:PROBLEM4_TEST_SEED=123456; python problem4/test_problem4.py
    bash:        PROBLEM4_TEST_SEED=123456 python problem4/test_problem4.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "problem3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from problem3_geometry import (  # noqa: E402
    circumscribed_polygon,
    min_enclosing_circle,
    point_in_polygon,
    polygon_area,
)
from problem3_strategy import (  # noqa: E402
    BEARING_ERROR_DEG,
    CLEAR_RADIUS,
    MAX_RECEIVE_RADIUS,
    MIN_RECEIVE_RADIUS,
    TARGET_RADIUS,
)
from problem4_directional_simulation import (  # noqa: E402
    COVER_LATTICE_SIDE_M,
    DODECAGON_INTERLEAVE_ANGLE_DEG,
    INNER_DODECAGON_RADIUS_M,
    OUTER_DODECAGON_RADIUS_M,
    SAFE_LOCALIZATION_RADIUS_M,
    DirectionalLocalSimulator,
    DirectionObservation,
    MixedSource,
    Problem4Strategy,
    clear_by_oriented_triangular_cover,
    concentric_dodecagon_route,
    concentric_dodecagon_stations,
    feasible_polygon,
    generate_mixed_sources,
    run_problem4_case,
)

# 端到端随机场景：每个案例恰好 10 个源
RANDOM_CASE_SOURCE_COUNT = 10
RANDOM_CASE_COUNT = 5


def _resolve_seed() -> int:
    """默认每次运行取一个新的随机种子；设置环境变量即可复现。"""
    from_env = os.environ.get("PROBLEM4_TEST_SEED")
    if from_env:
        return int(from_env)
    return int(np.random.default_rng().integers(0, 2**31 - 1))


RANDOM_SEED = _resolve_seed()


def setUpModule() -> None:
    print(
        f"\n[问题4测试] 本次随机种子 {RANDOM_SEED}"
        f"（复现：PROBLEM4_TEST_SEED={RANDOM_SEED}）"
    )


def make_simulator(
    specs: list[tuple[int, float, float, float, bool, float | None]],
    error_seed: int = 0,
) -> DirectionalLocalSimulator:
    """specs 每项为 (channel, x, y, receive_radius, directional, emission_deg)。"""
    sources = [
        MixedSource(
            channel=channel,
            position=np.array([x, y], dtype=float),
            receive_radius=receive_radius,
            directional=directional,
            emission_angle_rad=(
                None if emission_deg is None else math.radians(emission_deg)
            ),
        )
        for channel, x, y, receive_radius, directional, emission_deg in specs
    ]
    return DirectionalLocalSimulator(sources, error_seed=error_seed)


# --------------------------------------------------------------------------- #
# 1. 25 点巡检方案的几何保证
# --------------------------------------------------------------------------- #
class TestSurveyPlan(unittest.TestCase):
    def setUp(self) -> None:
        self.stations = concentric_dodecagon_stations()
        self.inner = self.stations[1:13]
        self.outer = self.stations[13:25]

    def test_station_count(self) -> None:
        self.assertEqual(len(self.stations), 25)
        self.assertEqual(len(self.inner), 12)
        self.assertEqual(len(self.outer), 12)

    def test_center_station_at_origin(self) -> None:
        self.assertEqual(self.stations[0], (0.0, 0.0))

    def test_inner_radius(self) -> None:
        for point in self.inner:
            self.assertAlmostEqual(
                math.hypot(point[0], point[1]),
                INNER_DODECAGON_RADIUS_M,
                places=6,
            )

    def test_outer_dodecagon_inradius_is_target_radius(self) -> None:
        # 外圈内切圆正好是半径 1800 m 的目标圆
        inradius = OUTER_DODECAGON_RADIUS_M * math.cos(
            math.radians(180.0 / 12.0)
        )
        self.assertAlmostEqual(inradius, TARGET_RADIUS, places=6)
        for point in self.outer:
            self.assertAlmostEqual(
                math.hypot(point[0], point[1]),
                OUTER_DODECAGON_RADIUS_M,
                places=6,
            )

    def test_two_rings_interleaved_by_15_degrees(self) -> None:
        self.assertAlmostEqual(DODECAGON_INTERLEAVE_ANGLE_DEG, 15.0, places=9)
        inner_angles = sorted(
            math.degrees(math.atan2(p[1], p[0])) % 360.0 for p in self.inner
        )
        outer_angles = sorted(
            math.degrees(math.atan2(p[1], p[0])) % 360.0 for p in self.outer
        )
        for inner_angle, outer_angle in zip(inner_angles, outer_angles):
            offset = (inner_angle - outer_angle) % 30.0
            self.assertAlmostEqual(
                min(offset, 30.0 - offset),
                DODECAGON_INTERLEAVE_ANGLE_DEG,
                places=6,
            )

    def test_triangulation_edges_within_receive_radius(self) -> None:
        def ring_edge(ring: list[tuple[float, float]]) -> float:
            return max(
                math.hypot(
                    ring[index][0] - ring[(index + 1) % len(ring)][0],
                    ring[index][1] - ring[(index + 1) % len(ring)][1],
                )
                for index in range(len(ring))
            )

        self.assertLessEqual(ring_edge(self.inner), MIN_RECEIVE_RADIUS)
        self.assertLessEqual(ring_edge(self.outer), MIN_RECEIVE_RADIUS)
        for point in self.inner:
            nearest = min(
                math.hypot(point[0] - other[0], point[1] - other[1])
                for other in self.outer
            )
            self.assertLessEqual(nearest, MIN_RECEIVE_RADIUS)

    def test_dense_scan_whole_disk(self) -> None:
        # 目标圆域内任意点到最近巡检点的距离必须不超过最小接收半径
        radii = np.linspace(0.0, TARGET_RADIUS, 121)
        angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
        grid_r, grid_a = np.meshgrid(radii, angles, indexing="ij")
        points = np.stack(
            [grid_r * np.cos(grid_a), grid_r * np.sin(grid_a)], axis=-1
        ).reshape(-1, 2)
        stations = np.asarray(self.stations, dtype=float)
        diff = points[:, None, :] - stations[None, :, :]
        nearest = np.hypot(diff[..., 0], diff[..., 1]).min(axis=1)
        self.assertLessEqual(float(nearest.max()), MIN_RECEIVE_RADIUS)
        # 设计余量：实测最坏点（目标圆边界）也只有约 568 m
        self.assertLessEqual(float(nearest.max()), 600.0)

    def test_any_emission_direction_is_detected(self) -> None:
        """任意位置 + 任意发射方向的定向源，25 点中必有一点落在其前半圆内。

        对给定测点 G，只有距离不超过 1000 m 的巡检点才可能收到信号；
        每个这样的点在 G 处张成一个 ±90° 的可视方向区间。整圈方向都被
        覆盖 ⇔ 这些方向的最大空隙不超过 180°。
        """
        worst_gap = 0.0
        for radius in np.linspace(0.0, TARGET_RADIUS, 37):
            for angle in np.linspace(0.0, 2.0 * math.pi, 73, endpoint=False):
                gx = radius * math.cos(angle)
                gy = radius * math.sin(angle)
                directions = sorted(
                    math.degrees(math.atan2(p[1] - gy, p[0] - gx)) % 360.0
                    for p in self.stations
                    if math.hypot(p[0] - gx, p[1] - gy)
                    <= MIN_RECEIVE_RADIUS + 1e-9
                )
                self.assertTrue(directions, "存在连一个巡检点都覆盖不到的位置")
                gaps = [
                    (directions[(index + 1) % len(directions)] - directions[index])
                    % 360.0
                    for index in range(len(directions))
                ]
                worst_gap = max(worst_gap, max(gaps))
        self.assertLessEqual(worst_gap, 180.0 + 1e-6)

    def test_route_starts_at_origin_and_visits_all_stations(self) -> None:
        route = concentric_dodecagon_route()
        self.assertEqual(len(route), 25)
        self.assertEqual(route[0], (0.0, 0.0))
        self.assertEqual(
            sorted(route), sorted(concentric_dodecagon_stations())
        )


# --------------------------------------------------------------------------- #
# 2. 定向源的物理语义
# --------------------------------------------------------------------------- #
class TestDirectionalSemantics(unittest.TestCase):
    def test_visible_from_front_half(self) -> None:
        # 源在 (500, 0)，发射方向 +x：前方 (600, 0) 必须能测到
        sim = make_simulator([(1, 500.0, 0.0, 1000.0, True, 0.0)])
        sim.enter()
        result = sim.measure(600.0, 0.0, 1)
        self.assertEqual(result.result, "direction")
        self.assertAlmostEqual(float(result.svd_deg), 180.0, delta=2.0)

    def test_blocked_behind_source(self) -> None:
        # 后方的 (0, 0) 距源仅 500 m（在接收半径内），仍应被遮挡
        sim = make_simulator([(1, 500.0, 0.0, 1000.0, True, 0.0)])
        sim.enter()
        result = sim.measure(0.0, 0.0, 1)
        self.assertEqual(result.result, "no_signal")
        self.assertEqual(sim.directional_block_count, 1)
        self.assertEqual(sim.range_no_signal_count, 0)

    def test_boundary_90_degrees_is_visible(self) -> None:
        # 题面：定向覆盖为两侧各 90°（含边界）
        sim = make_simulator([(1, 500.0, 0.0, 1000.0, True, 0.0)])
        sim.enter()
        result = sim.measure(500.0, 500.0, 1)
        self.assertEqual(result.result, "direction")
        self.assertEqual(sim.directional_block_count, 0)

    def test_range_checked_before_blocking(self) -> None:
        # 既在背后又超出接收半径：应记在"超距"，不是"定向遮挡"
        sim = make_simulator([(1, 500.0, 0.0, 1000.0, True, 0.0)])
        sim.enter()
        result = sim.measure(-600.0, 0.0, 1)
        self.assertEqual(result.result, "no_signal")
        self.assertEqual(sim.range_no_signal_count, 1)
        self.assertEqual(sim.directional_block_count, 0)

    def test_omnidirectional_never_blocked(self) -> None:
        sim = make_simulator([(1, 500.0, 0.0, 1000.0, False, None)])
        sim.enter()
        result = sim.measure(0.0, 0.0, 1)
        self.assertEqual(result.result, "direction")
        self.assertEqual(sim.directional_block_count, 0)

    def test_near_when_standing_on_source(self) -> None:
        sim = make_simulator([(1, 500.0, 0.0, 1000.0, True, 0.0)])
        sim.enter()
        result = sim.measure(500.0, 0.0, 1)
        self.assertEqual(result.result, "near")
        self.assertIsNone(result.svd_deg)

    def test_clear_success_at_exactly_20m(self) -> None:
        sim = make_simulator([(1, 20.0, 0.0, 1400.0, False, None)])
        sim.enter()
        self.assertTrue(sim.clear(0.0, 0.0, 1).cleared)

    def test_clear_failure_beyond_20m(self) -> None:
        sim = make_simulator([(1, 20.01, 0.0, 1400.0, False, None)])
        sim.enter()
        self.assertFalse(sim.clear(0.0, 0.0, 1).cleared)

    def test_no_signal_does_not_shrink_feasible_region(self) -> None:
        # no_signal 可能来自定向遮挡，策略必须忽略它，可行域只由 direction 裁剪
        strategy = Problem4Strategy(make_simulator([]))
        strategy.observations[1] = [DirectionObservation((0.0, 0.0), 120.0)]
        before = feasible_polygon(strategy.observations[1])
        strategy._record_measurement(1, (0.0, 0.0), "no_signal", None)  # noqa: SLF001
        self.assertEqual(len(strategy.observations[1]), 1)
        self.assertAlmostEqual(
            polygon_area(feasible_polygon(strategy.observations[1])),
            polygon_area(before),
            places=6,
        )

    def test_direction_observation_always_contains_true_source(self) -> None:
        # 可行域必须包含真实源：单次测向 + 1° 误差的楔形裁剪
        sim = make_simulator([(1, 900.0, 400.0, 1400.0, False, None)])
        sim.enter()
        observation = sim.measure(0.0, 0.0, 1)
        polygon = feasible_polygon(
            [DirectionObservation((0.0, 0.0), float(observation.svd_deg))]
        )
        enclosing = min_enclosing_circle(polygon)
        self.assertIsNotNone(enclosing)
        center, _ = enclosing
        self.assertLessEqual(
            math.hypot(center[0] - 900.0, center[1] - 400.0), MAX_RECEIVE_RADIUS
        )
        self.assertTrue(point_in_polygon((900.0, 400.0), polygon))

    def test_certify_directional_requires_full_containment(self) -> None:
        strategy = Problem4Strategy(make_simulator([]))
        strategy.observations[1] = [DirectionObservation((0.0, 0.0), 0.0)]
        # 楔形最远顶点在 1500 m 处：测点离它 800 m → 可确认定向
        strategy._certify_directional(1, (700.0, 0.0))  # noqa: SLF001
        self.assertIn(1, strategy.certified_directional)

        strategy.observations[2] = [DirectionObservation((0.0, 0.0), 0.0)]
        strategy._certify_directional(2, (-200.0, 0.0))  # noqa: SLF001
        self.assertNotIn(2, strategy.certified_directional)


# --------------------------------------------------------------------------- #
# 3. 清除半径与兜底覆盖
# --------------------------------------------------------------------------- #
class TestClearing(unittest.TestCase):
    def test_two_clear_worst_case_within_20m(self) -> None:
        # R = 58 时两次清除法的最坏偏差仍严格小于 20 m
        worst = 0.0
        for r in np.linspace(CLEAR_RADIUS + 1e-6, SAFE_LOCALIZATION_RADIUS_M, 2000):
            for error in (
                -math.radians(BEARING_ERROR_DEG),
                math.radians(BEARING_ERROR_DEG),
            ):
                travel = (CLEAR_RADIUS + r) / (
                    2.0 * math.cos(math.radians(BEARING_ERROR_DEG))
                )
                squared = (
                    travel * travel
                    + r * r
                    - 2.0 * travel * r * math.cos(error)
                )
                worst = max(worst, math.sqrt(max(squared, 0.0)))
        self.assertLess(worst, CLEAR_RADIUS)

    def test_cover_lattice_covering_radius_below_clear_radius(self) -> None:
        self.assertLess(COVER_LATTICE_SIDE_M / math.sqrt(3.0), CLEAR_RADIUS)

    def test_grid_fallback_clears_source(self) -> None:
        sim = make_simulator([(1, 100.0, 50.0, 1400.0, False, None)])
        sim.enter()
        polygon = circumscribed_polygon((100.0, 50.0), 300.0)
        clear_by_oriented_triangular_cover(sim, 1, polygon)
        self.assertTrue(sim.sources[1].cleared)


# --------------------------------------------------------------------------- #
# 4. 端到端随机场景（恰好 10 个源，每次运行重新随机）
# --------------------------------------------------------------------------- #
class TestRandomCases(unittest.TestCase):
    def test_random_ten_source_cases(self) -> None:
        for index in range(RANDOM_CASE_COUNT):
            with self.subTest(case=index):
                result = run_problem4_case(
                    case_id=index + 1,
                    seed=RANDOM_SEED * 100 + index,
                    source_count=RANDOM_CASE_SOURCE_COUNT,
                )
                self.assertEqual(result.source_count, RANDOM_CASE_SOURCE_COUNT)
                self.assertEqual(result.cleared_count, RANDOM_CASE_SOURCE_COUNT)
                self.assertTrue(result.success)
                # 10 源不触发“已发现 16 个”提前结束，必须完整访问全部安全点。
                self.assertEqual(result.survey_station_count, 25)
                # 定向判定不允许出现假阳性
                self.assertLessEqual(
                    result.confirmed_directional_count, result.directional_count
                )
                self.assertEqual(
                    result.directional_count + result.omnidirectional_count,
                    RANDOM_CASE_SOURCE_COUNT,
                )
                self.assertGreaterEqual(result.directional_count, 1)
                self.assertGreaterEqual(result.omnidirectional_count, 1)

    def test_source_mix_and_layout_are_random(self) -> None:
        first = generate_mixed_sources(RANDOM_SEED, source_count=10)
        second = generate_mixed_sources(RANDOM_SEED + 1, source_count=10)

        def signature(sources: list[MixedSource]) -> tuple:
            return tuple(
                sorted(
                    (
                        source.channel,
                        round(float(source.position[0]), 6),
                        round(float(source.position[1]), 6),
                        round(source.receive_radius, 6),
                        source.directional,
                    )
                    for source in sources
                )
            )

        self.assertNotEqual(signature(first), signature(second))
        for sources in (first, second):
            self.assertEqual(len(sources), 10)
            self.assertEqual(len({s.channel for s in sources}), 10)
            for source in sources:
                self.assertLessEqual(
                    math.hypot(
                        float(source.position[0]), float(source.position[1])
                    ),
                    TARGET_RADIUS + 1e-9,
                )
                self.assertGreaterEqual(
                    source.receive_radius, MIN_RECEIVE_RADIUS - 1e-9
                )
                self.assertLessEqual(
                    source.receive_radius, MAX_RECEIVE_RADIUS + 1e-9
                )
                self.assertEqual(
                    source.directional,
                    source.emission_angle_rad is not None,
                )

    def test_same_seed_reproduces_same_run(self) -> None:
        seed = RANDOM_SEED + 7
        first = run_problem4_case(case_id=1, seed=seed, source_count=10)
        second = run_problem4_case(case_id=1, seed=seed, source_count=10)
        self.assertEqual(first.total_time_s, second.total_time_s)
        self.assertEqual(first.directional_count, second.directional_count)
        self.assertEqual(first.measurement_count, second.measurement_count)

    def test_pure_directional_batch(self) -> None:
        result = run_problem4_case(
            case_id=1,
            seed=RANDOM_SEED + 11,
            directional_probability=1.0,
            force_mixed=False,
            source_count=10,
        )
        self.assertEqual(result.directional_count, 10)
        self.assertEqual(result.omnidirectional_count, 0)
        self.assertTrue(result.success)

    def test_pure_omnidirectional_batch(self) -> None:
        result = run_problem4_case(
            case_id=1,
            seed=RANDOM_SEED + 12,
            directional_probability=0.0,
            force_mixed=False,
            source_count=10,
        )
        self.assertEqual(result.directional_count, 0)
        self.assertEqual(result.omnidirectional_count, 10)
        self.assertTrue(result.success)

    def test_source_count_default_stays_in_problem_range(self) -> None:
        for index in range(3):
            sources = generate_mixed_sources(RANDOM_SEED + index)
            self.assertGreaterEqual(len(sources), 10)
            self.assertLessEqual(len(sources), 16)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
