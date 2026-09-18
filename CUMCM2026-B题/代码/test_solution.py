"""Deterministic regression tests for the B-question models and protocol client."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np

from geometry import (
    bearing_region,
    bearing_region_bounds,
    bearing_sector_polygon,
    clip_convex_polygon_halfplane,
    convex_hull,
    cross,
    diameter,
    diameter_circle_certificate,
    min_enclosing_circle,
    optimal_second_point,
    point_in_convex_polygon,
    q3_cover_points,
    q3_minimax_ring_points,
    q3_ring_cover_points,
    q4_certified21_cover_points,
    q4_compact_cover_points,
    q4_compact_cover_triangles,
    q4_triangular_cover_points,
    second_point_candidate_region,
)
from offline_sim import JammerWorld
from robot import (
    HttpClient,
    Observation,
    Robot,
    RobotConfig,
    _OfflineClient,
    _multistart_open_order,
    run_offline,
)


class GeometryTests(unittest.TestCase):
    def test_diameter_handles_degenerate_polygon(self):
        self.assertEqual(diameter(np.array([[3.0, -2.0]])), 0.0)
        self.assertEqual(diameter(np.array([[3.0, -2.0], [3.0, -2.0]])), 0.0)

    def test_convex_membership_handles_degenerate_sets(self):
        self.assertFalse(point_in_convex_polygon((100.0, 100.0), np.array([[0.0, 0.0]])))
        self.assertFalse(
            point_in_convex_polygon(
                (100.0, 100.0),
                np.array([[0.0, 0.0], [0.0, 0.0]]),
            )
        )
        segment = np.array([[0.0, 0.0], [1.0, 0.0]])
        self.assertFalse(point_in_convex_polygon((2.0, 0.0), segment))
        self.assertTrue(point_in_convex_polygon((0.5, 0.0), segment))
        self.assertFalse(point_in_convex_polygon((0.5, 1.0), segment))

    def test_q1_diameter_circle_certificate_has_checkable_witness(self):
        source = np.array([420.0, 680.0])
        observers = [
            np.array([-500.0, 0.0]),
            np.array([800.0, -200.0]),
            np.array([-200.0, 1100.0]),
        ]
        errors = (-0.73, 0.42, -0.18)
        readings = []
        for observer, error in zip(observers, errors):
            theta = math.atan2(source[1] - observer[1], source[0] - observer[0])
            readings.append((observer, theta + math.radians(error)))
        region = bearing_region(
            readings,
            1800.0,
            1.0,
            [(observer, 1500.0) for observer in observers],
        )
        certificate = diameter_circle_certificate(region)
        left, right = certificate["diameter_endpoints"]
        midpoint = certificate["diameter_center"]
        witness = certificate["witness"]
        self.assertAlmostEqual(float(np.linalg.norm(left - right)), certificate["diameter"], places=8)
        self.assertAlmostEqual(float(np.linalg.norm((left + right) / 2.0 - midpoint)), 0.0, places=8)
        self.assertGreater(float(np.linalg.norm(witness - midpoint)), certificate["diameter_radius"])
        self.assertFalse(certificate["covers"])
        self.assertGreater(certificate["overflow"], 4.0)

    def test_q1_pure_sector_intersection_reports_bounded_empty_and_unbounded(self):
        source = np.array([420.0, 680.0])
        observers = [
            np.array([-500.0, 0.0]),
            np.array([800.0, -200.0]),
            np.array([-200.0, 1100.0]),
        ]
        errors = (-0.73, 0.42, -0.18)
        readings = []
        for observer, error in zip(observers, errors):
            theta = math.atan2(source[1] - observer[1], source[0] - observer[0])
            readings.append((observer, theta + math.radians(error)))
        bounded, bounded_status = bearing_sector_polygon(readings, 1.0)
        self.assertEqual(bounded_status, "polygon")
        self.assertTrue(point_in_convex_polygon(source, bounded, tol=1e-6))

        _, unbounded_status = bearing_sector_polygon(
            [((0.0, 0.0), 0.0), ((10.0, 0.0), 0.0)],
            1.0,
        )
        self.assertEqual(unbounded_status, "unbounded")
        empty, empty_status = bearing_sector_polygon(
            [((0.0, 0.0), 0.0), ((-100.0, 0.0), math.pi)],
            1.0,
        )
        self.assertEqual(empty_status, "empty")
        self.assertEqual(len(empty), 0)

    def test_q1_inner_outer_disk_bounds_tighten(self):
        source = np.array([1798.0, 0.0])
        observers = [
            np.array([500.0, 0.0]),
            np.array([800.0, -800.0]),
            np.array([800.0, 800.0]),
        ]
        errors = (0.8, -0.7, 0.3)
        readings = []
        for observer, error in zip(observers, errors):
            theta = math.atan2(source[1] - observer[1], source[0] - observer[0])
            readings.append((observer, theta + math.radians(error)))
        disks = [(observer, 1500.0) for observer in observers]
        gaps = []
        for sides in (360, 720):
            inner, outer = bearing_region_bounds(readings, 1800.0, 1.0, disks, sides)
            self.assertGreater(len(inner), 0)
            self.assertGreater(len(outer), 0)
            self.assertTrue(
                all(point_in_convex_polygon(point, outer, tol=1e-6) for point in inner)
            )
            diameter_gap = diameter(outer) - diameter(inner)
            radius_gap = min_enclosing_circle(outer).radius - min_enclosing_circle(inner).radius
            self.assertGreaterEqual(diameter_gap, -1e-8)
            self.assertGreaterEqual(radius_gap, -1e-8)
            gaps.append((diameter_gap, radius_gap))
        self.assertLess(gaps[1][0], gaps[0][0])
        self.assertLess(gaps[1][1], gaps[0][1])

    def test_q3_cover_is_below_minimum_receive_radius(self):
        points = q3_cover_points()
        worst = 0.0
        for radius in np.linspace(0.0, 1800.0, 361):
            for angle in np.linspace(0.0, 2.0 * math.pi, 1440, endpoint=False):
                source = radius * np.array([math.cos(angle), math.sin(angle)])
                worst = max(worst, float(np.min(np.linalg.norm(points - source, axis=1))))
        self.assertLess(worst, 1000.0)

    def test_q3_optimized_ring_covers_target_disk(self):
        for sides in (6, 7, 8, 10, 12, 14, 16):
            points = q3_ring_cover_points(sides=sides)
            worst = 0.0
            for radius in np.linspace(0.0, 1800.0, 181):
                for angle in np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False):
                    source = radius * np.array([math.cos(angle), math.sin(angle)])
                    worst = max(worst, float(np.min(np.linalg.norm(points - source, axis=1))))
            self.assertLess(worst, 1000.0)

    def test_q3_optimized_ring_has_continuous_cover_certificate(self):
        target_radius = 1800.0
        receive_radius = 1000.0
        for sides in (6, 7):
            points = q3_ring_cover_points(
                target_radius=target_radius,
                min_receive_radius=receive_radius,
                sides=sides,
            )
            ring_radius = float(np.linalg.norm(points[1]))
            half_step = math.pi / sides
            boundary_gap = math.sqrt(
                target_radius**2
                + ring_radius**2
                - 2.0 * target_radius * ring_radius * math.cos(half_step)
            )

            # The center covers r <= receive_radius. Beyond that radius,
            # distance to the nearest ring point is largest at the target
            # boundary and sector bisector.
            self.assertGreater(receive_radius, ring_radius * math.cos(half_step))
            self.assertLess(boundary_gap, receive_radius)

    def test_q3_minimax_ring_covers_target_disk_without_origin(self):
        points = q3_minimax_ring_points()
        self.assertEqual(len(points), 7)
        worst = 0.0
        for radius in np.linspace(0.0, 1800.0, 361):
            for angle in np.linspace(0.0, 2.0 * math.pi, 1440, endpoint=False):
                source = radius * np.array([math.cos(angle), math.sin(angle)])
                worst = max(worst, float(np.min(np.linalg.norm(points - source, axis=1))))
        self.assertLess(worst, 1000.0)
        with self.assertRaises(ValueError):
            q3_minimax_ring_points(sides=6)

    def test_q3_range_order_constraints_keep_the_true_source(self):
        with tempfile.TemporaryDirectory() as directory:
            world = JammerWorld.random_case(
                7041,
                3,
                count=10,
                receive_radius_range=(1000.0, 1000.0),
                boundary_bias=True,
            )
            robot = Robot(
                _OfflineClient(world, Path(directory) / "events.jsonl"),
                RobotConfig(problem=3, q3_range_order_constraints=True),
                Path(directory),
            )
            robot._post("/enter")
            source = world.sources[0]
            for point in robot._scan_points():
                robot.measure(point, source.channel)
            region, _, _ = robot._region_for(source.channel)
            self.assertGreater(len(region), 0)
            self.assertTrue(
                point_in_convex_polygon(source.position, region, tol=1e-6)
            )

    def test_q4_mesh_positive_spans_every_sampled_source(self):
        points = q4_triangular_cover_points()
        self.assertEqual(len(points), 31)
        for x in np.arange(-1800.0, 1800.1, 60.0):
            for y in np.arange(-1800.0, 1800.1, 60.0):
                source = np.array([x + 0.137, y + 0.271])
                if np.linalg.norm(source) > 1800.0:
                    continue
                vectors = points - source
                close = vectors[np.linalg.norm(vectors, axis=1) <= 1000.0 + 1e-8]
                self.assertGreaterEqual(len(close), 3)
                angles = sorted(math.atan2(v[1], v[0]) % (2.0 * math.pi) for v in close if np.linalg.norm(v) > 1e-8)
                gaps = [angles[(i + 1) % len(angles)] - angles[i] for i in range(len(angles))]
                gaps[-1] += 2.0 * math.pi
                self.assertLessEqual(max(gaps), math.pi + 1e-8)

    def test_q4_compact_cover_positive_spans_every_sampled_source(self):
        points = q4_compact_cover_points()
        self.assertEqual(len(points), 25)
        for x in np.arange(-1800.0, 1800.1, 40.0):
            for y in np.arange(-1800.0, 1800.1, 40.0):
                source = np.array([x + 0.137, y + 0.271])
                if np.linalg.norm(source) > 1800.0:
                    continue
                vectors = points - source
                close = vectors[np.linalg.norm(vectors, axis=1) <= 1000.0 + 1e-8]
                self.assertGreaterEqual(len(close), 3)
                angles = sorted(
                    math.atan2(v[1], v[0]) % (2.0 * math.pi)
                    for v in close
                    if np.linalg.norm(v) > 1e-8
                )
                gaps = [angles[(i + 1) % len(angles)] - angles[i] for i in range(len(angles))]
                gaps[-1] += 2.0 * math.pi
                self.assertLessEqual(max(gaps), math.pi + 1e-8)

    def test_q4_certified21_cover_positive_spans_every_sampled_source(self):
        points = q4_certified21_cover_points()
        self.assertEqual(len(points), 21)
        for x in np.arange(-1800.0, 1800.1, 40.0):
            for y in np.arange(-1800.0, 1800.1, 40.0):
                source = np.array([x + 0.137, y + 0.271])
                if np.linalg.norm(source) > 1800.0:
                    continue
                vectors = points - source
                close = vectors[np.linalg.norm(vectors, axis=1) <= 997.0 + 1e-8]
                self.assertGreaterEqual(len(close), 2)
                angles = sorted(
                    math.atan2(v[1], v[0]) % (2.0 * math.pi)
                    for v in close
                    if np.linalg.norm(v) > 1e-8
                )
                gaps = [angles[(i + 1) % len(angles)] - angles[i] for i in range(len(angles))]
                gaps[-1] += 2.0 * math.pi
                self.assertLessEqual(max(gaps), math.pi + 1e-8)

    def test_q4_symmetric_probe_pair_preserves_directional_detection(self):
        """At least one guaranteed-reception probe sees every source beyond the split."""
        error = math.radians(1.01)
        upper = 1500.0
        split = 750.0
        lateral = upper * math.tan(error) + 0.5
        probes = (
            np.array([split, lateral], dtype=float),
            np.array([split, -lateral], dtype=float),
        )
        for delta in np.linspace(-error, error, 9):
            minimum_radius = split / math.cos(delta)
            for radius in np.linspace(minimum_radius, 1500.0, 9):
                source = radius * np.array([math.cos(delta), math.sin(delta)])
                self.assertLessEqual(max(np.linalg.norm(probe - source) for probe in probes), 998.0)
                for orientation in np.linspace(0.0, 2.0 * math.pi, 721):
                    normal = np.array([math.cos(orientation), math.sin(orientation)])
                    if float(np.dot(normal, -source)) < -1e-9:
                        continue
                    visible = max(float(np.dot(normal, probe - source)) for probe in probes)
                    self.assertGreaterEqual(visible, -1e-8)

    def test_q4_recovery_halfplane_is_applied_to_future_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(None, RobotConfig(problem=4), Path(directory))
            robot.obs.append(Observation((0.0, 0.0), 1, "direction", 0.0, 0.0))
            robot.region_halfplanes[1] = [(np.array([-1.0, 0.0]), -500.0)]
            region, _, _ = robot._region_for(1)
        self.assertGreater(len(region), 0)
        self.assertLessEqual(float(np.max(region[:, 0])), 500.0 + 1e-7)

    def test_q4_optimized_policy_is_the_default(self):
        config = RobotConfig(problem=4)
        self.assertEqual(config.q4_layout, "compact")
        self.assertTrue(config.q4_symmetric_recovery)
        self.assertEqual(config.q4_information_floor_fraction, 0.10)
        self.assertTrue(config.q4_skip_impossible_rechecks)
        self.assertTrue(config.q4_local_positive_cap)
        self.assertFalse(config.q4_local_triangle_support)
        self.assertEqual(config.q4_estimated_clear_radius, 100.0)
        self.assertEqual(config.q4_estimated_clear_attempts, 2)
        self.assertTrue(config.q4_shadow_bisection)
        self.assertTrue(config.q4_scan_estimated_clear)
        self.assertEqual(config.q4_scan_clear_max_detour, 200.0)
        self.assertFalse(config.q4_integrated_scan_clear)
        self.assertTrue(config.q4_joint_constraints)
        self.assertTrue(config.q4_time_lookahead)
        self.assertTrue(config.q4_safe_clear_route)

    def test_q3_optimized_policy_is_the_default(self):
        config = RobotConfig(problem=3)
        self.assertEqual(config.q3_layout, "minimax")
        self.assertEqual(config.q3_ring_sides, 7)
        self.assertEqual(config.q3_estimated_clear_radius, 100.0)
        self.assertEqual(config.q3_estimated_clear_attempts, 1)
        self.assertTrue(config.q3_scan_estimated_clear)
        self.assertEqual(config.q3_scan_clear_max_detour, 150.0)
        self.assertTrue(config.q3_multistart_localization_route)

    def test_q3_fast_path_retains_complete_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_offline(
                12001,
                3,
                Path(directory),
                world_options={
                    "count": 10,
                    "receive_radius_range": (1000.0, 1000.0),
                    "boundary_bias": True,
                },
            )
        self.assertEqual(result["coverage"], 1.0)
        self.assertIsNone(result["failure"])

    def test_q4_default_robot_uses_certified21_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            world = JammerWorld.random_case(20260912, 4)
            robot = Robot(
                _OfflineClient(world, Path(directory) / "events.jsonl"),
                RobotConfig(problem=4),
                Path(directory),
            )
            self.assertEqual(len(robot._scan_points()), 21)

    def test_q4_sparse_positive_triangle_support_contains_source(self):
        points = q4_compact_cover_points()
        triangles = q4_compact_cover_triangles()
        rng = np.random.default_rng(20260912)
        for _ in range(800):
            radius = math.sqrt(float(rng.random())) * 1800.0
            angle = float(rng.random()) * 2.0 * math.pi
            source = radius * np.array([math.cos(angle), math.sin(angle)])
            orientation = float(rng.random()) * 2.0 * math.pi
            normal = np.array([math.cos(orientation), math.sin(orientation)])
            vectors = points - source
            visible = points[
                (np.linalg.norm(vectors, axis=1) <= 1000.0 + 1e-9)
                & ((vectors @ normal) >= -1e-9)
            ]
            self.assertGreaterEqual(len(visible), 1)
            if len(visible) >= 3:
                continue
            incident = [
                triangle
                for triangle in triangles
                if any(
                    float(np.linalg.norm(vertex - positive)) < 1e-6
                    for vertex in triangle
                    for positive in visible
                )
            ]
            support = convex_hull(np.vstack(incident))
            self.assertTrue(point_in_convex_polygon(source, support, tol=1e-6))

    def test_q4_local_positive_cap_keeps_union_of_possible_near_bearings(self):
        truth = np.array([700.0, 200.0])
        observers = [np.array([-500.0, 200.0]), np.array([700.0, -600.0])]
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(None, RobotConfig(problem=4), Path(directory))
            for observer in observers:
                angle = math.degrees(math.atan2(*(truth - observer)[::-1]))
                robot.obs.append(
                    Observation(tuple(observer), 1, "direction", angle, 0.0)
                )
            robot.local_positive_cap_channels.add(1)
            region, _, _ = robot._region_for(1)
        self.assertTrue(point_in_convex_polygon(truth, region, tol=1e-6))

    def test_q4_positive_observer_segment_extends_safe_candidate_set(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(
                None,
                RobotConfig(problem=4, q4_positive_hull_views=True),
                Path(directory),
            )
            truth = np.zeros(2)
            observers = (np.array([1400.0, -100.0]), np.array([1400.0, 100.0]))
            for observer in observers:
                angle = math.degrees(math.atan2(*(truth - observer)[::-1]))
                robot.obs.append(Observation(tuple(observer), 1, "direction", angle, 0.0))
            region, _, _ = robot._region_for(1)
            candidates = robot._view_candidates(1, region)
        extended = [
            np.asarray(candidate)
            for candidate in candidates
            if abs(candidate[0] - 1400.0) < 1e-6 and abs(candidate[1]) < 100.0
        ]
        self.assertGreater(len(extended), 0)
        self.assertTrue(
            any(float(np.max(np.linalg.norm(region - candidate, axis=1))) > 998.0 for candidate in extended)
        )

    def test_q4_bearing_center_estimate_recovers_exact_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(None, RobotConfig(problem=4), Path(directory))
            truth = np.array([320.0, -270.0])
            for observer in (np.array([-700.0, 100.0]), np.array([600.0, 800.0]), np.array([900.0, -500.0])):
                angle = math.degrees(math.atan2(*(truth - observer)[::-1]))
                robot.obs.append(Observation(tuple(observer), 1, "direction", angle, 0.0))
            region, _, _ = robot._region_for(1)
            estimate = robot._bearing_center_estimate(1, region)
        self.assertLess(float(np.linalg.norm(estimate - truth)), 1e-6)

    def test_q4_one_bearing_posterior_stays_in_feasible_region(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(
                None,
                RobotConfig(problem=4, q4_one_bearing_posterior="mean"),
                Path(directory),
            )
            robot.obs.append(Observation((0.0, 0.0), 1, "direction", 0.0, 0.0))
            for point in ((0.0, 930.0), (0.0, -930.0), (-930.0, 0.0)):
                robot.obs.append(Observation(point, 1, "no_signal", None, 0.0))
            region, _, _ = robot._region_for(1)
            estimate = robot._one_bearing_posterior_estimate(1, region)
        self.assertTrue(point_in_convex_polygon(estimate, region, tol=1e-6))

    def test_q4_shadow_bisection_recovers_a_visible_probe(self):
        class Client:
            def __init__(self, world):
                self.world = world
                self.counter = 0

            def post(self, path, body):
                self.counter += 1
                request_id = f"shadow-{self.counter}"
                if path == "/measure":
                    return self.world.measure(body["position"], body["channel"], request_id)
                if path == "/clear":
                    return self.world.clear(body["position"], body["channel"], request_id)
                raise ValueError(path)

        from offline_sim import Source

        source = Source(1, 0.0, 0.0, 1000.0, directional_angle=0.0)
        world = JammerWorld([source], seed=7)
        world.entered = True
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(
                Client(world),
                RobotConfig(problem=4, q4_shadow_bisection_rounds=4),
                Path(directory),
            )
            robot.obs.append(Observation((600.0, 0.0), 1, "direction", 180.0, 0.0))
            cleared, new_bearing = robot._q4_recover_from_shadow(1, (-100.0, 0.0))
        self.assertFalse(cleared)
        self.assertTrue(new_bearing)

    def test_multistart_open_route_is_not_worse_than_nearest_first(self):
        points = {
            1: np.array([8.0, 0.0]),
            2: np.array([9.0, 8.0]),
            3: np.array([0.0, 9.0]),
            4: np.array([-7.0, 1.0]),
        }
        route = _multistart_open_order(points, (0.0, 0.0))
        self.assertEqual(set(route), set(points))

        def length(order):
            total = 0.0
            previous = np.zeros(2)
            for stop in order:
                total += float(np.linalg.norm(points[stop] - previous))
                previous = points[stop]
            return total

        self.assertLessEqual(length(route), length([4, 1, 2, 3]) + 1e-9)

    def test_bounded_error_region_contains_true_source(self):
        rng = np.random.default_rng(2026)
        for _ in range(30):
            radius = math.sqrt(float(rng.random())) * 1790.0
            angle = float(rng.random()) * 2.0 * math.pi
            source = radius * np.array([math.cos(angle), math.sin(angle)])
            observers = []
            errors = []
            while len(observers) < 3:
                observer = rng.uniform(-1200.0, 1200.0, 2)
                if np.linalg.norm(source - observer) <= 1490.0:
                    observers.append(observer)
                    errors.append(float(rng.uniform(-1.0, 1.0)))
            readings = []
            for observer, error in zip(observers, errors):
                true = math.atan2(source[1] - observer[1], source[0] - observer[0])
                readings.append((observer, true + math.radians(error)))
            region = bearing_region(readings, 1800.0, 1.0, [(observer, 1500.0) for observer in observers])
            self.assertTrue(point_in_convex_polygon(source, region, tol=1e-6))

    def test_enclosing_circle_and_diameter_bounds(self):
        rng = np.random.default_rng(17)
        points = rng.normal(size=(30, 2)) * np.array([3.0, 1.0])
        circle = min_enclosing_circle(points)
        self.assertTrue(np.all(np.linalg.norm(points - circle.center, axis=1) <= circle.radius + 1e-7))
        self.assertGreaterEqual(circle.radius + 1e-8, diameter(points) / 2.0)

    def test_q2_candidate_guarantees_reception_and_crossing_angle(self):
        region, candidates, _ = second_point_candidate_region((0.0, 0.0), 0.0)
        best, _ = optimal_second_point((0.0, 0.0), 0.0)
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(float(np.max(np.linalg.norm(region - best, axis=1))), 999.0 + 1e-7)
        worst_angle = 90.0
        for radius in np.linspace(5.01, 1500.0, 500):
            for error in np.linspace(-1.0, 1.0, 21):
                source = radius * np.array([math.cos(math.radians(error)), math.sin(math.radians(error))])
                a, b = source, source - best
                sine = abs(cross(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
                worst_angle = min(worst_angle, math.degrees(math.asin(min(1.0, sine))))
                self.assertLessEqual(np.linalg.norm(source - best), 1000.0)
        self.assertGreaterEqual(worst_angle, 30.0)

    def test_q3_entry_candidates_preserve_guaranteed_reception(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(
                None,
                RobotConfig(
                    problem=3,
                    q3_entry_candidates=True,
                    q3_information_floor_fraction=0.1,
                ),
                Path(directory),
            )
            robot.position = (300.0, 900.0)
            robot.obs.append(
                Observation((0.0, 0.0), 1, "direction", 0.0, 0.0)
            )
            region, _, _ = robot._region_for(1)
            candidates = robot._view_candidates(1, region)
        self.assertGreater(len(candidates), 0)
        for candidate in candidates:
            maximum = float(
                np.max(np.linalg.norm(region - np.asarray(candidate), axis=1))
            )
            self.assertLessEqual(maximum, 998.0 + 1e-7)


class SimulatorAndProtocolTests(unittest.TestCase):
    def test_offline_coverage_uses_simulator_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_offline(2, 4, Path(directory))
        self.assertEqual(result["cleared_count"], result["simulator_cleared_count"])
        self.assertEqual(result["coverage"], 1.0)

    def test_location_error_is_repeatable_across_world_instances(self):
        worlds = [JammerWorld.random_case(31, 4) for _ in range(2)]
        source = worlds[0].sources[0]
        point = {"x": source.x + 100.0, "y": source.y}
        values = [world._fixed_error_deg(world.sources[0], point) for world in worlds]
        self.assertEqual(values[0], values[1])

    def test_http_retry_reuses_identical_payload_and_request_id(self):
        requests: list[dict] = []

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return b'{"accepted":true,"virtual_time_s":0}'

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            if len(requests) == 1:
                raise urllib.error.URLError("response lost")
            return Response()

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "log.jsonl"
            client = HttpClient("http://127.0.0.1:2026", "runtime-id", log_path)
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                client.post("/enter", {}, retries=2)
            log_text = log_path.read_text(encoding="utf-8")
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[0]["request_id"], requests[1]["request_id"])
        self.assertNotIn("runtime-id", log_text)
        self.assertIn("<ROBOT_ID>", log_text)

    def test_q4_adaptive_discovery_keeps_directional_channel_active(self):
        """A found Q4 channel remains active until its bearing target is met."""
        class Client:
            def __init__(self, world):
                self.world = world
                self.counter = 0

            def post(self, path, body):
                self.counter += 1
                request_id = f"t-{self.counter}"
                if path == "/enter":
                    return self.world.enter(request_id)
                if path == "/measure":
                    return self.world.measure(body["position"], body["channel"], request_id)
                if path == "/clear":
                    return self.world.clear(body["position"], body["channel"], request_id)
                return self.world.exit(request_id)

        world = JammerWorld.random_case(
            91,
            4,
            directional_probability=1.0,
            count=10,
            receive_radius_range=(1000.0, 1000.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(Client(world), RobotConfig(problem=4), Path(directory))
            robot._post("/enter")
            points = robot._scan_points()
            robot._discovery_scan(points)
            truth_channels = {source.channel for source in world.sources}
            self.assertEqual(robot.discovered, truth_channels)
            self.assertEqual(robot.absent, set(range(1, 21)) - truth_channels)
            self.assertEqual(robot.discovered | robot.absent, set(range(1, 21)))
            self.assertEqual(robot.discovered & robot.absent, set())


class Q3ImprovementTests(unittest.TestCase):
    def test_degenerate_membership_and_distance_are_bounded(self):
        from geometry import point_polygon_distance
        for vertices in (
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]),
        ):
            self.assertFalse(point_in_convex_polygon((2000.0, 0.0), vertices))
            self.assertGreater(point_polygon_distance((2000.0, 0.0), vertices), 1900.0)
        self.assertTrue(point_in_convex_polygon((0.5, 0.0), np.array([[0.0, 0.0], [1.0, 0.0]])))

    def test_q3_skip_preserves_unknown_channel_scan(self):
        class Client:
            def __init__(self):
                self.measured = []

            def post(self, path, body):
                self.measured.append(body["channel"])
                return {"accepted": True, "measure_result": "no_signal", "virtual_time_s": 6.0}

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            robot = Robot(client, RobotConfig(problem=3, q3_skip_impossible_rechecks=True), directory)
            robot.obs.append(Observation((0.0, 0.0), 1, "direction", 0.0, 0.0))
            robot.discovered.add(1)
            counts = {1: 1}
            robot._measure_discovery_point((-1700.0, 0.0), {2}, counts, 2)
            self.assertEqual(client.measured, [2])
            self.assertEqual(robot.skipped_rechecks, 1)
            self.assertIn((-1700.0, 0.0), robot.inferred_no_signal[1])
            self.assertEqual(len([o for o in robot.obs if o.channel == 1]), 1)

    def test_cache_invalidates_for_observations_configuration_and_inferences(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(None, RobotConfig(problem=3, q3_range_order_constraints=True), directory)
            robot.obs.append(Observation((0.0, 0.0), 1, "direction", 0.0, 0.0))
            first, _, _ = robot._region_for(1)
            robot._region_for(1)
            self.assertEqual(robot.region_cache_hits, 1)
            robot.inferred_no_signal[1] = {(600.0, 0.0)}
            shrunk, _, _ = robot._region_for(1)
            self.assertLessEqual(float(np.max(shrunk[:, 0])), 300.0 + 1e-6)
            self.assertGreater(float(np.max(first[:, 0])), 1000.0)
            robot.cfg.max_error_deg = 1.02
            robot._region_for(1)
            self.assertEqual(robot.region_cache_misses, 3)
            robot.obs.append(Observation((-10.0, 100.0), 1, "direction", 340.0, 0.0))
            robot._region_for(1)
            self.assertEqual(robot.region_cache_misses, 4)

    def test_service_point_is_certified_and_no_longer_than_center(self):
        rng = np.random.default_rng(6071)
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(None, RobotConfig(problem=3, q3_service_region=True), directory)
            for _ in range(12):
                region = convex_hull(rng.uniform(-6.0, 6.0, size=(8, 2)))
                circle = min_enclosing_circle(region)
                robot.position = (-100.0, 25.0)
                following = np.array([80.0, 40.0])
                point = robot._certified_service_point(region, circle, following)
                self.assertLessEqual(float(np.max(np.linalg.norm(region - point, axis=1))), 19.5 + 1e-7)
                cost = lambda q: np.linalg.norm(q - robot.position) + np.linalg.norm(q - following)
                self.assertLessEqual(float(cost(point)), float(cost(circle.center)) + 1e-7)

    def test_low_budget_preserves_reception_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(None, RobotConfig(problem=3), directory)
            robot.obs.append(Observation((0.0, 0.0), 1, "direction", 0.0, 0.0))
            robot.deadline_monotonic = 110.0
            with patch("robot.time.monotonic", return_value=100.0):
                region, _, _ = robot._region_for(1)
                candidates = robot._view_candidates(1, region)
            self.assertTrue(robot.budget_degraded)
            self.assertGreater(len(candidates), 0)
            for point in candidates:
                self.assertLessEqual(float(np.max(np.linalg.norm(region - point, axis=1))), 998.0 + 1e-7)

    def test_exit_is_allowed_after_action_limit(self):
        class Client:
            def post(self, path, body):
                return {"accepted": True, "virtual_time_s": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(Client(), RobotConfig(max_actions=2), directory)
            robot._post("/enter")
            with self.assertRaises(RuntimeError):
                robot._post("/measure", channel=1, position={"x": 0, "y": 0})
            self.assertTrue(robot._post("/exit")["accepted"])

    def test_exhausted_budget_is_not_reported_as_complete(self):
        class Client:
            def __init__(self):
                self.paths = []

            def post(self, path, body):
                self.paths.append(path)
                return {"accepted": True, "virtual_time_s": 0.0, "remaining_real_duration_s": 1.0}

        with tempfile.TemporaryDirectory() as directory:
            client = Client()
            robot = Robot(client, RobotConfig(problem=3), directory)
            with patch("robot.time.monotonic", return_value=100.0):
                result = robot.run()
            self.assertEqual(client.paths, ["/enter", "/exit"])
            self.assertIsNotNone(result["failure"])
            self.assertFalse(result["completion_certified_by_policy"])
            self.assertFalse(result["discovery_complete"])

    def test_invalid_layout_still_exits_and_reports_failure(self):
        world = JammerWorld.random_case(6099, 3)
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(_OfflineClient(world, Path(directory) / "events.jsonl"), RobotConfig(q3_layout="invalid"), directory)
            result = robot.run()
            self.assertFalse(world.entered)
            self.assertIsNotNone(result["failure"])
            self.assertFalse(result["completion_certified_by_policy"])

    def test_rejected_enter_produces_failure_summary(self):
        class Client:
            def post(self, path, body):
                return {"accepted": False}
        with tempfile.TemporaryDirectory() as directory:
            result = Robot(Client(), RobotConfig(), directory).run()
        self.assertIsNotNone(result["failure"])
        self.assertFalse(result["completion_certified_by_policy"])

    def test_deadline_stops_http_without_transmission(self):
        with tempfile.TemporaryDirectory() as directory:
            client = HttpClient("http://127.0.0.1:2026", "test-id", Path(directory) / "events.jsonl")
            client.set_deadline(99.0)
            with patch("robot.time.monotonic", return_value=100.0), patch("urllib.request.urlopen") as urlopen:
                with self.assertRaises(TimeoutError):
                    client.post("/measure", {"channel": 1})
                urlopen.assert_not_called()

    def test_clear_does_not_switch_measurement_channel(self):
        from offline_sim import Source
        world = JammerWorld([Source(2, 10.0, 0.0, 1000.0)])
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(_OfflineClient(world, Path(directory) / "events.jsonl"), RobotConfig(), directory)
            robot._post("/enter")
            robot.clear((10.0, 0.0), 2)
            self.assertEqual(robot.channel, 1)
            self.assertEqual(robot.switch_count, 0)
            self.assertEqual(robot.clear_success_count, 1)
            self.assertAlmostEqual(robot.last_virtual_time, 7.0)

    def test_joint_time_order_is_complete_and_preserves_region(self):
        with tempfile.TemporaryDirectory() as directory:
            robot = Robot(None, RobotConfig(q3_joint_time_routing=True, q3_service_region=True), directory)
            for ch, target in ((1, np.array([100.0, 0.0])), (2, np.array([400.0, 200.0]))):
                robot.discovered.add(ch)
                for observer in (np.array([-400.0, 100.0]), np.array([300.0, -500.0])):
                    angle = math.degrees(math.atan2(*(target - observer)[::-1]))
                    robot.obs.append(Observation(tuple(observer), ch, "direction", angle, 0.0))
            regions = {ch: robot._region_for(ch)[0].copy() for ch in robot.discovered}
            order = robot._localization_order()
            self.assertEqual(set(order), {1, 2})
            for ch in regions:
                self.assertTrue(np.array_equal(regions[ch], robot._region_for(ch)[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
