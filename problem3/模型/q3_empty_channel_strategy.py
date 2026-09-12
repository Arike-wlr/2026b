"""Low-source empty-channel coverage experiments for q3_practice_v2.

The only changed decision is the minimum residual area required to measure an
unknown channel at a task stop that the robot already visits.  Actual maps,
forced residual searches, source localization and all completion certificates
remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from problem3_hybrid_simulation import generate_sources
from q3_practice_v2_local_benchmark import (
    DEFAULT_EXTERNAL_SOURCE,
    load_external_implementation,
    run_case,
    summarize,
    write_results,
)


DEFAULT_PIGGYBACK_GAIN_M2 = 1_000_000.0
DEFAULT_HEX_RING_RADIUS_M = 1_125.0
DISABLED_PIGGYBACK_GAIN_M2 = 1.0e30


def build_empty_channel_strategy(
    base_strategy,
    gain_threshold_m2: float,
    always_finish_channel: bool = True,
    resume_piggyback_at_known_sources: int | None = None,
    resumed_gain_threshold_m2: float = DEFAULT_PIGGYBACK_GAIN_M2,
    resume_by_search_stop: int | None = None,
):
    from coverage_strategy import reception

    class EmptyChannelStrategy(base_strategy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._adaptive_resume_latched = False
            self._adaptive_resume_known_sources = None
            self._adaptive_resume_search_stops = None

        def _adaptive_resume_active(self):
            if self._adaptive_resume_latched:
                return True
            if resume_piggyback_at_known_sources is None:
                return False
            known_sources = self._known_source_count()
            if known_sources < resume_piggyback_at_known_sources:
                return False
            if resume_by_search_stop is not None and self.search_stops > resume_by_search_stop:
                return False
            self._adaptive_resume_latched = True
            self._adaptive_resume_known_sources = known_sources
            self._adaptive_resume_search_stops = self.search_stops
            return True

        def scan(self, point, force=False):
            if force:
                return super().scan(point, force=True)

            unknown = self.unknown()
            if not unknown:
                return False
            disk = reception(point)
            current = self.robot.current_channel
            resume_active = self._adaptive_resume_active()
            active_threshold = (
                resumed_gain_threshold_m2 if resume_active else gain_threshold_m2
            )
            candidates = []
            for channel in unknown:
                if self.channels[channel].status != "UNKNOWN":
                    continue
                remaining = self.map.remaining[channel]
                gain = remaining.intersection(disk).area
                completes = not remaining.is_empty and remaining.difference(disk).is_empty
                if (always_finish_channel and completes) or gain >= active_threshold:
                    candidates.append((channel != current, channel, gain, completes))

            changed = False
            for _, channel, gain, completes in sorted(candidates):
                if self.channels[channel].status != "UNKNOWN":
                    continue
                self.sense(channel, point, "PIGGYBACK")
                self.piggyback_measures += 1
                self.piggyback_area += gain
                changed = True
            return changed

        def run(self):
            result = super().run()
            result["adaptive_resume"] = {
                "latched": self._adaptive_resume_latched,
                "known_sources": self._adaptive_resume_known_sources,
                "search_stops": self._adaptive_resume_search_stops,
            }
            return result

    return EmptyChannelStrategy


def build_hex_cover_strategy(
    base_strategy,
    gain_threshold_m2: float,
    always_finish_channel: bool = False,
    ring_radius_m: float = DEFAULT_HEX_RING_RADIUS_M,
    resume_piggyback_at_known_sources: int | None = None,
    resumed_gain_threshold_m2: float = DEFAULT_PIGGYBACK_GAIN_M2,
    resume_dynamic_search: bool = False,
    resume_by_search_stop: int | None = None,
    enable_marginal_scan: bool = False,
    marginal_scan_margin_s: float = 2.0,
    enable_neighborhood_cover: bool = False,
    neighborhood_iterations: int = 4,
):
    """Add a certified center-plus-six coverage backbone.

    The center is already force-scanned by the parent strategy.  Six sites on
    the regular hexagon cover the full search domain.  At each replanning step
    all 64 subsets are checked against the *actual* per-channel residual maps;
    this can omit a site already made redundant by real measurements.  If a
    numerical or geometric residual remains after the backbone is exhausted,
    the parent's certified residual planner is used unchanged.
    """
    from coverage_strategy import Task, reception
    from problem3_geometry import nearest_neighbor_route, route_length, two_opt_open
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import nearest_points, unary_union

    threshold_strategy = build_empty_channel_strategy(
        base_strategy,
        gain_threshold_m2,
        always_finish_channel=always_finish_channel,
        resume_piggyback_at_known_sources=resume_piggyback_at_known_sources,
        resumed_gain_threshold_m2=resumed_gain_threshold_m2,
        resume_by_search_stop=resume_by_search_stop,
    )

    class HexCoverStrategy(threshold_strategy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._hex_points = None
            self._hex_disks = None
            self._hex_orientation_rad = None
            self._marginal_scan_evaluations = 0
            self._marginal_scan_accepts = 0
            self._marginal_scan_planned_saving_s = 0.0
            self._cover_patches = None
            self._cover_allowed = None
            self._cover_neighborhood_points = None
            self._cover_neighborhood_updates = 0
            self._cover_neighborhood_planned_saving_m = 0.0

        def _route_distance(self, points):
            if not points:
                return 0.0
            start = self.robot.current_position
            order = nearest_neighbor_route(start, points)
            order = two_opt_open(order, start, points)
            return route_length(order, start, points)

        def _initialize_hex_cover(self, tasks):
            if self._hex_points is not None:
                return
            sector = math.pi / 3.0
            candidate_angles = [k * math.pi / 36.0 for k in range(12)]
            candidate_angles.extend(
                math.atan2(float(task.point[1]), float(task.point[0])) % sector
                for task in tasks
            )
            task_points = [np.asarray(task.point, dtype=float) for task in tasks]
            best = None
            for angle in candidate_angles:
                ring = [
                    np.array(
                        [
                            ring_radius_m * math.cos(angle + k * sector),
                            ring_radius_m * math.sin(angle + k * sector),
                        ]
                    )
                    for k in range(6)
                ]
                cover = unary_union([reception((0.0, 0.0))] + [reception(p) for p in ring])
                if not self.map.domain.difference(cover).is_empty:
                    continue
                score = self._route_distance(task_points + ring)
                key = (score, angle)
                if best is None or key < best[0]:
                    best = (key, angle, ring)
            if best is None:
                raise RuntimeError("The requested hexagon radius does not certify full coverage")
            _, self._hex_orientation_rad, self._hex_points = best
            self._hex_disks = [reception(point) for point in self._hex_points]
            if enable_neighborhood_cover:
                self._initialize_cover_neighborhoods()

        def _initialize_cover_neighborhoods(self):
            safe_radius_m = 999.5
            far_radius_m = 4_000.0
            sector = math.pi / 3.0
            center_disk = reception((0.0, 0.0))
            self._cover_patches = []
            self._cover_allowed = []
            self._cover_neighborhood_points = []
            for index, fixed_point in enumerate(self._hex_points):
                angle = self._hex_orientation_rad + index * sector
                wedge = Polygon(
                    [
                        (0.0, 0.0),
                        (
                            far_radius_m * math.cos(angle - sector / 2.0),
                            far_radius_m * math.sin(angle - sector / 2.0),
                        ),
                        (
                            far_radius_m * math.cos(angle + sector / 2.0),
                            far_radius_m * math.sin(angle + sector / 2.0),
                        ),
                    ]
                )
                patch = self.map.domain.intersection(wedge).difference(center_disk).convex_hull
                vertices = list(patch.exterior.coords)[:-1]
                allowed = Point(vertices[0]).buffer(safe_radius_m, quad_segs=32)
                for vertex in vertices[1:]:
                    allowed = allowed.intersection(
                        Point(vertex).buffer(safe_radius_m, quad_segs=32)
                    )
                if allowed.is_empty:
                    raise RuntimeError("A certified outer-sector scan neighborhood is empty")
                if not patch.difference(reception(fixed_point)).is_empty:
                    raise RuntimeError("The fixed hexagon point does not cover its assigned sector")
                self._cover_patches.append(patch)
                self._cover_allowed.append(allowed)
                self._cover_neighborhood_points.append(np.asarray(fixed_point, dtype=float))

        @staticmethod
        def _point_from_geometry(geometry):
            return np.array([float(geometry.x), float(geometry.y)])

        def _optimized_neighborhood_searches(self, tasks):
            remaining = {channel: self.map.remaining[channel] for channel in self.unknown()}
            required = [
                index
                for index, patch in enumerate(self._cover_patches)
                if any(shape.intersection(patch).area > 1e-8 for shape in remaining.values())
            ]
            if not required:
                return []

            fixed_task_points = [np.asarray(task.point, dtype=float) for task in tasks]
            points = {
                index: np.asarray(self._cover_neighborhood_points[index], dtype=float).copy()
                for index in required
            }
            original_points = {
                index: np.asarray(self._hex_points[index], dtype=float).copy()
                for index in required
            }
            fixed_route_points = fixed_task_points + [original_points[index] for index in required]
            fixed_route_m = self._route_distance(fixed_route_points)

            for _ in range(max(1, neighborhood_iterations)):
                combined = fixed_task_points + [points[index] for index in required]
                order = nearest_neighbor_route(self.robot.current_position, combined)
                order = two_opt_open(order, self.robot.current_position, combined)
                order_position = {node: position for position, node in enumerate(order)}
                for local_index, sector_index in enumerate(required):
                    node = len(fixed_task_points) + local_index
                    position = order_position[node]
                    previous = (
                        np.asarray(self.robot.current_position, dtype=float)
                        if position == 0
                        else np.asarray(combined[order[position - 1]], dtype=float)
                    )
                    next_point = (
                        None
                        if position + 1 == len(order)
                        else np.asarray(combined[order[position + 1]], dtype=float)
                    )
                    allowed = self._cover_allowed[sector_index]
                    candidates = [points[sector_index]]
                    previous_geometry = Point(float(previous[0]), float(previous[1]))
                    candidates.append(
                        self._point_from_geometry(nearest_points(allowed, previous_geometry)[0])
                    )
                    if next_point is not None:
                        next_geometry = Point(float(next_point[0]), float(next_point[1]))
                        candidates.append(
                            self._point_from_geometry(nearest_points(allowed, next_geometry)[0])
                        )
                        segment = LineString([previous, next_point])
                        candidates.append(
                            self._point_from_geometry(nearest_points(allowed, segment)[0])
                        )

                    def local_cost(candidate):
                        value = float(np.linalg.norm(candidate - previous))
                        if next_point is not None:
                            value += float(np.linalg.norm(next_point - candidate))
                        return value

                    points[sector_index] = min(candidates, key=local_cost)

            optimized = []
            for index in required:
                point = points[index]
                if not self._cover_patches[index].difference(reception(point)).is_empty:
                    point = original_points[index]
                if not self._cover_patches[index].difference(reception(point)).is_empty:
                    raise RuntimeError("Optimized scan point lost its sector coverage certificate")
                if np.linalg.norm(point - self._cover_neighborhood_points[index]) > 1e-7:
                    self._cover_neighborhood_updates += 1
                self._cover_neighborhood_points[index] = point.copy()
                optimized.append(point)

            optimized_route_m = self._route_distance(fixed_task_points + optimized)
            self._cover_neighborhood_planned_saving_m += max(
                0.0,
                fixed_route_m - optimized_route_m,
            )
            return [Task("search", point.copy()) for point in optimized]

        def _search_measurement_count(self, chosen, remaining):
            return sum(
                remaining[channel].intersection(self._hex_disks[index]).area > 1e-8
                for index in chosen
                for channel in remaining
            )

        def _minimum_certifying_hex_subset(self, tasks, remaining=None):
            if remaining is None:
                unknown = self.unknown()
                remaining = {channel: self.map.remaining[channel] for channel in unknown}
            else:
                unknown = list(remaining)
            if not unknown:
                return []
            self._initialize_hex_cover(tasks)
            best = None
            task_points = [np.asarray(task.point, dtype=float) for task in tasks]
            for mask in range(1 << 6):
                chosen = [index for index in range(6) if mask & (1 << index)]
                if any(
                    not any(
                        remaining[channel].intersection(self._hex_disks[index]).area > 1e-8
                        for channel in unknown
                    )
                    for index in chosen
                ):
                    # A heuristic open-route score can otherwise retain a point
                    # that is geometrically redundant.  The execution layer is
                    # correct to reject such a no-progress forced scan.
                    continue
                if chosen:
                    future_cover = unary_union([self._hex_disks[index] for index in chosen])
                    certifies = all(
                        remaining[channel].difference(future_cover).is_empty
                        for channel in unknown
                    )
                else:
                    certifies = all(remaining[channel].is_empty for channel in unknown)
                if not certifies:
                    continue
                points = task_points + [self._hex_points[index] for index in chosen]
                # Action cost breaks route-length ties in favour of fewer forced
                # all-channel scans.  The actual execution still re-plans after
                # every observation.
                score = (
                    self._route_distance(points)
                    + 30.0 * self._search_measurement_count(chosen, remaining)
                )
                key = (score, len(chosen), tuple(chosen))
                if best is None or key < best[0]:
                    best = (key, chosen)
            return None if best is None else best[1]

        def _hex_plan_score(self, tasks, remaining):
            chosen = self._minimum_certifying_hex_subset(tasks, remaining)
            if chosen is None:
                return math.inf
            points = [np.asarray(task.point, dtype=float) for task in tasks]
            points.extend(self._hex_points[index] for index in chosen)
            return (
                self._route_distance(points)
                + 30.0 * self._search_measurement_count(chosen, remaining)
            )

        def _marginal_scan(self, point):
            unknown = self.unknown()
            if not unknown:
                return False
            tasks = self.target_tasks()
            self._initialize_hex_cover(tasks)
            disk = reception(point)
            remaining = {channel: self.map.remaining[channel] for channel in unknown}
            eligible = [
                channel
                for channel in unknown
                if remaining[channel].intersection(disk).area > 1e-8
            ]
            if not eligible:
                return False

            groups = {}
            for channel in eligible:
                groups.setdefault(remaining[channel].wkb, []).append(channel)
            candidate_groups = list(groups.values())
            if len(candidate_groups) > 1:
                candidate_groups.append(eligible)

            baseline_score = self._hex_plan_score(tasks, remaining)
            current_channel = self.robot.current_channel
            best = None
            for group in candidate_groups:
                hypothetical = dict(remaining)
                for channel in group:
                    hypothetical[channel] = hypothetical[channel].difference(disk)
                future_score = self._hex_plan_score(tasks, hypothetical)
                switch_count = len(group) - (1 if current_channel in group else 0)
                immediate_s = 5.0 * len(group) + switch_count
                planned_saving_s = (baseline_score - future_score) / 5.0 - immediate_s
                self._marginal_scan_evaluations += 1
                key = (planned_saving_s, -len(group))
                if best is None or key > best[0]:
                    best = (key, list(group), planned_saving_s)

            if best is None or best[2] <= marginal_scan_margin_s:
                return False
            selected = best[1]
            changed = False
            for channel in sorted(
                selected,
                key=lambda candidate: (candidate != self.robot.current_channel, candidate),
            ):
                if self.channels[channel].status != "UNKNOWN":
                    continue
                gain = self.map.gain(channel, point)
                if gain <= 1e-8:
                    continue
                self.sense(channel, point, "MARGINAL_PIGGYBACK")
                self.piggyback_measures += 1
                self.piggyback_area += gain
                changed = True
            if changed:
                self._marginal_scan_accepts += 1
                self._marginal_scan_planned_saving_s += best[2]
            return changed

        def scan(self, point, force=False):
            if force or not enable_marginal_scan:
                return super().scan(point, force=force)
            return self._marginal_scan(point)

        def planned_searches(self, tasks):
            need = self.remaining_shape()
            if need.is_empty:
                return []
            if (
                resume_dynamic_search
                and self._adaptive_resume_active()
            ):
                return super().planned_searches(tasks)
            if enable_neighborhood_cover:
                self._initialize_hex_cover(tasks)
                searches = self._optimized_neighborhood_searches(tasks)
                if searches:
                    return searches
            chosen = self._minimum_certifying_hex_subset(tasks)
            if chosen is not None:
                return [Task("search", self._hex_points[index].copy()) for index in chosen]
            return super().planned_searches(tasks)

        def run(self):
            result = super().run()
            result["hex_cover"] = {
                "ring_radius_m": ring_radius_m,
                "orientation_rad": self._hex_orientation_rad,
                "marginal_scan_evaluations": self._marginal_scan_evaluations,
                "marginal_scan_accepts": self._marginal_scan_accepts,
                "marginal_scan_planned_saving_s": self._marginal_scan_planned_saving_s,
                "neighborhood_point_updates": self._cover_neighborhood_updates,
                "neighborhood_planned_saving_m": self._cover_neighborhood_planned_saving_m,
            }
            return result

    return HexCoverStrategy


def selected_seeds(
    start: int,
    count: int,
    minimum_source_count: int,
    maximum_source_count: int,
) -> list[int]:
    seeds = []
    seed = start
    while len(seeds) < count:
        source_count = len(generate_sources(seed))
        if minimum_source_count <= source_count <= maximum_source_count:
            seeds.append(seed)
        seed += 1
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-source-count", type=int, default=10)
    parser.add_argument("--max-source-count", type=int, default=12)
    parser.add_argument("--gain-threshold", type=float, default=DEFAULT_PIGGYBACK_GAIN_M2)
    parser.add_argument(
        "--disable-finish-exception",
        action="store_true",
        help="Disable the rule that always measures a channel if this stop finishes its residual area.",
    )
    parser.add_argument(
        "--hex-cover",
        action="store_true",
        help="Use the certified center-plus-six coverage backbone.",
    )
    parser.add_argument(
        "--disable-piggyback",
        action="store_true",
        help="Batch unknown-channel coverage only at forced search sites.",
    )
    parser.add_argument("--ring-radius", type=float, default=DEFAULT_HEX_RING_RADIUS_M)
    parser.add_argument(
        "--resume-piggyback-at-known-sources",
        type=int,
        default=None,
        help="Resume selective piggyback scans after this many sources are confirmed.",
    )
    parser.add_argument(
        "--resumed-gain-threshold",
        type=float,
        default=DEFAULT_PIGGYBACK_GAIN_M2,
    )
    parser.add_argument(
        "--resume-dynamic-search",
        action="store_true",
        help="Resume the parent residual-search planner together with piggyback scans.",
    )
    parser.add_argument(
        "--resume-by-search-stop",
        type=int,
        default=None,
        help="Only latch adaptive mode if the known-source trigger is reached this early.",
    )
    parser.add_argument(
        "--marginal-scan",
        action="store_true",
        help="Accept piggyback scans only when the recomputed certified plan is cheaper.",
    )
    parser.add_argument("--marginal-scan-margin", type=float, default=2.0)
    parser.add_argument(
        "--neighborhood-cover",
        action="store_true",
        help="Choose each outer-sector scan inside its certified feasible neighborhood.",
    )
    parser.add_argument("--neighborhood-iterations", type=int, default=4)
    parser.add_argument("--source", type=Path, default=DEFAULT_EXTERNAL_SOURCE)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/tables/q3_empty_channel"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategy_types = load_external_implementation(args.source)
    gain_threshold = (
        DISABLED_PIGGYBACK_GAIN_M2 if args.disable_piggyback else args.gain_threshold
    )
    always_finish_channel = not (
        args.disable_finish_exception or args.disable_piggyback
    )
    if args.hex_cover:
        Strategy = build_hex_cover_strategy(
            strategy_types[1],
            gain_threshold,
            always_finish_channel=always_finish_channel,
            ring_radius_m=args.ring_radius,
            resume_piggyback_at_known_sources=args.resume_piggyback_at_known_sources,
            resumed_gain_threshold_m2=args.resumed_gain_threshold,
            resume_dynamic_search=args.resume_dynamic_search,
            resume_by_search_stop=args.resume_by_search_stop,
            enable_marginal_scan=args.marginal_scan,
            marginal_scan_margin_s=args.marginal_scan_margin,
            enable_neighborhood_cover=args.neighborhood_cover,
            neighborhood_iterations=args.neighborhood_iterations,
        )
    else:
        Strategy = build_empty_channel_strategy(
            strategy_types[1],
            gain_threshold,
            always_finish_channel=always_finish_channel,
            resume_piggyback_at_known_sources=args.resume_piggyback_at_known_sources,
            resumed_gain_threshold_m2=args.resumed_gain_threshold,
            resume_by_search_stop=args.resume_by_search_stop,
        )
    seeds = selected_seeds(
        args.random_state,
        args.cases,
        args.min_source_count,
        args.max_source_count,
    )
    results = [
        run_case(index + 1, seed, strategy_types, strategy_class=Strategy)
        for index, seed in enumerate(seeds)
    ]
    summary = summarize(results, args.random_state)
    summary["strategy"] = (
        "q3_empty_channel_hex_cover" if args.hex_cover
        else "q3_empty_channel_piggyback_threshold"
    )
    summary["source_path"] = str(args.source.resolve())
    summary["gain_threshold_m2"] = gain_threshold
    summary["always_finish_channel"] = always_finish_channel
    summary["piggyback_disabled"] = args.disable_piggyback
    summary["hex_cover"] = args.hex_cover
    summary["ring_radius_m"] = args.ring_radius
    summary["resume_piggyback_at_known_sources"] = args.resume_piggyback_at_known_sources
    summary["resumed_gain_threshold_m2"] = args.resumed_gain_threshold
    summary["resume_dynamic_search"] = args.resume_dynamic_search
    summary["resume_by_search_stop"] = args.resume_by_search_stop
    summary["marginal_scan"] = args.marginal_scan
    summary["marginal_scan_margin_s"] = args.marginal_scan_margin
    summary["neighborhood_cover"] = args.neighborhood_cover
    summary["neighborhood_iterations"] = args.neighborhood_iterations
    summary["minimum_source_count"] = args.min_source_count
    summary["maximum_source_count"] = args.max_source_count
    summary["selected_seeds"] = seeds
    write_results(results, summary, args.output_prefix)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
