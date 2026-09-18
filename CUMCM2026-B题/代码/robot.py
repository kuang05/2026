"""Guaranteed-coverage search and bounded-error localization for CUMCM 2026 B.

The same policy runs against the supplied HTTP simulator and the deterministic
offline regression world. New actions are strictly serial, and a network
retry reuses the complete original payload and request_id.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from geometry import (
    EPS,
    bearing_region,
    clip_convex_polygon_halfplane,
    convex_hull,
    cross,
    diameter,
    intersect_convex_polygons,
    min_enclosing_circle,
    point_in_convex_polygon,
    point_polygon_distance,
    q3_cover_points,
    q3_minimax_ring_points,
    q3_ring_cover_points,
    q3_custom_ring_points,
    q4_certified21_cover_points,
    q4_compact_cover_points,
    q4_compact_cover_triangles,
    q4_triangular_cover_points,
    unit,
)


@dataclass
class Observation:
    position: tuple[float, float]
    channel: int
    result: str
    angle_deg: float | None
    virtual_time: float


@dataclass
class RobotConfig:
    problem: int = 3
    target_radius: float = 1800.0
    max_error_deg: float = 1.01
    min_receive_radius: float = 1000.0
    max_receive_radius: float = 1500.0
    clear_radius: float = 20.0
    clear_guarantee_radius: float = 19.5
    q3_ring_radius: float = 1150.0
    q3_custom_ring: bool = False
    q3_include_center: bool = True
    q3_layout: str = "minimax"
    q3_ring_sides: int = 7
    q3_min_discovery_directions: int = 2
    q3_view_travel_weight: float = 0.2
    q3_entry_candidates: bool = True
    q3_information_floor_fraction: float = 0.1
    # For an omnidirectional source, a positive/negative pair on one channel
    # gives the valid ordering d(positive, source) < d(negative, source).
    # Kept as an explicit experiment switch; the conservative default is off.
    q3_range_order_constraints: bool = False
    q3_estimated_clear_radius: float = 100.0
    q3_estimated_clear_attempts: int = 1
    q3_measure_after_failed_estimate: bool = True
    q3_scan_estimated_clear: bool = True
    q3_scan_clear_max_detour: float = 150.0
    q3_multistart_localization_route: bool = True
    q3_skip_impossible_rechecks: bool = True
    # Independently switchable experiments; defaults are selected on held-out cases.
    q3_service_region: bool = True
    q3_joint_time_routing: bool = False
    q3_joint_candidate_limit: int = 4
    budget_aware: bool = True
    exit_reserve_s: float = 2.0
    planning_guard_s: float = 30.0
    q4_mesh_edge: float = 980.0
    q4_layout: str = "compact"
    # A directional source may be visible from only one mesh vertex.  During
    # Q4 discovery keeps a found channel active until enough positive bearings
    # are available. Three is the measured time optimum in the matched-case
    # benchmark; the deterministic fallback remains valid when fewer are seen.
    q4_min_discovery_directions: int = 3
    q4_symmetric_recovery: bool = True
    q4_recovery_rounds: int = 6
    q4_information_floor_fraction: float = 0.10
    q4_integrated_scan_clear: bool = False
    q4_speculative_clear_radius: float = 19.5
    q4_skip_impossible_rechecks: bool = True
    q4_local_positive_cap: bool = True
    q4_local_triangle_support: bool = False
    q4_positive_hull_views: bool = False
    q4_estimated_clear_radius: float = 100.0
    q4_estimated_clear_attempts: int = 2
    q4_estimator: str = "pairwise"
    q4_measure_after_failed_estimate: bool = True
    q4_shadow_bisection: bool = True
    q4_shadow_bisection_rounds: int = 4
    q4_scan_estimated_clear: bool = True
    q4_scan_clear_max_detour: float = 200.0
    q4_scan_shadow_recovery: bool = False
    q4_scan_clear_lookahead: bool = False
    q4_scan_clear_success_probability: float = 0.90
    q4_adaptive_outer_route: bool = False
    q4_estimate_route_centers: bool = False
    q4_multistart_localization_route: bool = False
    q4_one_bearing_posterior: str = "off"
    q4_one_bearing_initial_probe: bool = True
    # Stop collecting additional bearings as soon as the robust feasible
    # region is small enough for a guaranteed clear.  The fixed discovery
    # count remains the minimum search-stage requirement.
    adaptive_clear_stop: bool = True
    max_source_count: int = 16
    max_view_attempts: int = 8
    max_actions: int = 10000
    real_time_reserve_s: float = 3.0
    # Q4 refinements are separately switchable for matched ablation tests.
    q4_joint_constraints: bool = True
    q4_time_lookahead: bool = True
    q4_safe_clear_route: bool = True
    q4_planning_budget_s: float = 1.0
    q4_constraint_cells: int = 96


class RuntimeBudgetExceeded(TimeoutError):
    """The remaining real-time budget is reserved for orderly exit."""


class HttpClient:
    """Strict serial JSON client with protocol-correct idempotent retries."""

    def __init__(self, base_url: str, robot_id: str, log_path: str | Path):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.run_tag = str(int(time.time() * 1000))
        self.last_response: dict[str, Any] | None = None
        self.deadline: float | None = None
        self.exit_reserve_s = 0.0

    def set_deadline(self, deadline: float | None, reserve_s: float = 0.0) -> None:
        self.deadline = deadline
        self.exit_reserve_s = max(0.0, reserve_s)

    def _remaining(self, path: str) -> float:
        if self.deadline is None:
            return math.inf
        reserve = 0.0 if path == "/exit" else self.exit_reserve_s
        remaining = self.deadline - reserve - time.monotonic()
        if remaining <= 0.02:
            raise RuntimeBudgetExceeded(f"real-time budget exhausted before {path}")
        return remaining

    @contextmanager
    def _open_response(self, request, timeout, path):
        if self.deadline is None:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                yield response
            return

        deadline = self.deadline - (0.0 if path == "/exit" else self.exit_reserve_s)
        state = {"socket": None, "expired": False}
        lock = threading.Lock()

        def expire():
            with lock:
                state["expired"] = True
                connected = state["socket"]
                if connected is not None:
                    try:
                        connected.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        def factory(connection_type):
            def create(*args, **kwargs):
                connection = connection_type(*args, **kwargs)
                original_connect = connection._create_connection

                def connect(*connect_args, **connect_kwargs):
                    connected = original_connect(*connect_args, **connect_kwargs)
                    with lock:
                        if state["expired"] or time.monotonic() >= deadline:
                            connected.close()
                            raise RuntimeBudgetExceeded(
                                "request deadline expired during connect"
                            )
                        old = state["socket"]
                        if old is not None:
                            old.close()
                        state["socket"] = connected.dup()
                    return connected

                connection._create_connection = connect
                return connection

            return create

        http_factory = factory(http.client.HTTPConnection)
        https_factory = factory(http.client.HTTPSConnection)

        class TimedHTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, req):
                return self.do_open(http_factory, req)

        class TimedHTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, req):
                return self.do_open(https_factory, req, context=self._context)

        timer = threading.Timer(max(0.0, deadline - time.monotonic()), expire)
        timer.daemon = True
        timer.start()
        try:
            opener = urllib.request.build_opener(TimedHTTPHandler(), TimedHTTPSHandler())
            with opener.open(request, timeout=timeout) as response:
                yield response
                if state["expired"]:
                    raise RuntimeBudgetExceeded(
                        "request exceeded its absolute deadline"
                    )
        except Exception as error:
            if state["expired"]:
                raise RuntimeBudgetExceeded(
                    "request exceeded its absolute deadline"
                ) from error
            raise
        finally:
            timer.cancel()
            timer.join()
            with lock:
                if state["socket"] is not None:
                    state["socket"].close()

    def _write_log(self, record: dict[str, Any]) -> None:
        record = dict(record)
        if "request" in record:
            request = dict(record["request"])
            if "robot_id" in request:
                request["robot_id"] = "<ROBOT_ID>"
            record["request"] = request
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def post(self, path: str, body: dict[str, Any], retries: int = 4):
        if retries < 1:
            raise ValueError("retries must be at least one")
        self.counter += 1
        request_id = body.get("request_id") or f"{path.strip('/')}-{self.run_tag}-{self.counter}"
        payload = {"arena_id": "default", "robot_id": self.robot_id, **body, "request_id": request_id}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

        for attempt in range(1, retries + 1):
            timeout = min(8.0, self._remaining(path) - 0.01)
            request = urllib.request.Request(
                self.base_url + path,
                data=encoded,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with self._open_response(request, timeout, path) as response_stream:
                    response = json.loads(response_stream.read().decode("utf-8"))
                    status = int(response_stream.status)
                self.last_response = response
                self._write_log(
                    {"wall_time": time.time(), "path": path, "attempt": attempt, "http_status": status,
                     "request": payload, "response": response}
                )
                return response
            except urllib.error.HTTPError as error:
                response = {"http_status": error.code, "reason": str(error.reason)}
                error.close()
                self._write_log(
                    {"wall_time": time.time(), "path": path, "attempt": attempt, "http_status": error.code,
                     "request": payload, "response": response}
                )
                raise RuntimeError(f"HTTP {error.code} for {path}: {response}") from error
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
            ) as error:
                self._write_log(
                    {"wall_time": time.time(), "path": path, "attempt": attempt,
                     "request": payload, "transport_error": repr(error)}
                )
                if attempt == retries:
                    raise
                delay = 0.2 * attempt
                if self._remaining(path) <= delay + 0.02:
                    raise RuntimeBudgetExceeded("insufficient budget for retry") from error
                time.sleep(delay)
        raise RuntimeError("unreachable retry state")


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _open_route(points: Sequence[Sequence[float]], start: Sequence[float] = (0.0, 0.0)) -> list[tuple[float, float]]:
    """Nearest-neighbour route followed by deterministic open-path 2-opt."""
    remaining = [tuple(map(float, point)) for point in points]
    if not remaining:
        return []
    first_index = min(range(len(remaining)), key=lambda i: (_distance(start, remaining[i]), remaining[i]))
    route = [remaining.pop(first_index)]
    while remaining:
        index = min(range(len(remaining)), key=lambda i: (_distance(route[-1], remaining[i]), remaining[i]))
        route.append(remaining.pop(index))

    improved = True
    passes = 0
    while improved and passes < 12:
        improved = False
        passes += 1
        for i in range(0, len(route) - 1):
            for j in range(i + 1, len(route)):
                left = start if i == 0 else route[i - 1]
                before = _distance(left, route[i])
                after = _distance(left, route[j])
                if j + 1 < len(route):
                    before += _distance(route[j], route[j + 1])
                    after += _distance(route[i], route[j + 1])
                if after + 1e-8 < before:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
    return route


def _route_length(points: Sequence[Sequence[float]], start: Sequence[float]) -> float:
    route = _open_route(points, start=start)
    return sum(
        _distance(start if index == 0 else route[index - 1], point)
        for index, point in enumerate(route)
    )


def _joint_route_first(
    tasks: Sequence[tuple[str, int, Sequence[float]]],
    start: Sequence[float],
) -> int:
    """Return the first task in a deterministic open-path route."""
    if not tasks:
        raise ValueError("at least one route task is required")
    points = [tuple(map(float, task[2])) for task in tasks]
    remaining = list(range(len(tasks)))

    def tie_key(index: int) -> tuple[float, str, int]:
        return (_distance(start, points[index]), tasks[index][0], tasks[index][1])

    first = min(remaining, key=tie_key)
    route = [first]
    remaining.remove(first)
    while remaining:
        previous = points[route[-1]]
        chosen = min(
            remaining,
            key=lambda index: (
                _distance(previous, points[index]),
                tasks[index][0],
                tasks[index][1],
            ),
        )
        route.append(chosen)
        remaining.remove(chosen)

    improved = True
    passes = 0
    while improved and passes < 12:
        improved = False
        passes += 1
        for i in range(0, len(route) - 1):
            for j in range(i + 1, len(route)):
                left = start if i == 0 else points[route[i - 1]]
                before = _distance(left, points[route[i]])
                after = _distance(left, points[route[j]])
                if j + 1 < len(route):
                    before += _distance(points[route[j]], points[route[j + 1]])
                    after += _distance(points[route[i]], points[route[j + 1]])
                if after + 1e-8 < before:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    improved = True
    return route[0]


def _multistart_open_order(
    points: dict[int, np.ndarray],
    start: Sequence[float],
) -> list[int]:
    """Build a deterministic 2-opt open route from every possible first stop."""
    if not points:
        return []
    start_point = np.asarray(start, dtype=float)

    def distance(left: int | None, right: int) -> float:
        origin = start_point if left is None else points[left]
        return float(np.linalg.norm(origin - points[right]))

    def improve(route: list[int]) -> list[int]:
        changed = True
        passes = 0
        while changed and passes < 12:
            changed = False
            passes += 1
            for left_index in range(len(route) - 1):
                for right_index in range(left_index + 1, len(route)):
                    left = None if left_index == 0 else route[left_index - 1]
                    before = distance(left, route[left_index])
                    after = distance(left, route[right_index])
                    if right_index + 1 < len(route):
                        next_stop = route[right_index + 1]
                        before += distance(route[right_index], next_stop)
                        after += distance(route[left_index], next_stop)
                    if after + 1e-8 < before:
                        route[left_index : right_index + 1] = reversed(
                            route[left_index : right_index + 1]
                        )
                        changed = True
        return route

    def route_length(route: list[int]) -> float:
        return sum(
            distance(None if index == 0 else route[index - 1], stop)
            for index, stop in enumerate(route)
        )

    routes = []
    for first in sorted(points):
        route = [first]
        remaining = set(points) - {first}
        while remaining:
            chosen = min(
                remaining,
                key=lambda stop: (distance(route[-1], stop), stop),
            )
            route.append(chosen)
            remaining.remove(chosen)
        improved = improve(route)
        routes.append((route_length(improved), improved))
    return min(routes, key=lambda item: (item[0], item[1]))[1]


class Robot:
    def __init__(self, client: Any, config: RobotConfig, result_dir: str | Path):
        self.client = client
        self.cfg = config
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.position = (0.0, 0.0)
        self.channel = 1
        self.obs: list[Observation] = []
        self.discovered: set[int] = set()
        self.cleared: set[int] = set()
        self.absent: set[int] = set()
        self.action_count = 0
        self.last_virtual_time = 0.0
        self.failed_clear_points: dict[int, list[tuple[float, float]]] = {}
        self.region_halfplanes: dict[int, list[tuple[np.ndarray, float]]] = {}
        self.local_positive_cap_channels: set[int] = set()
        self.discovery_support_polygons: dict[int, np.ndarray] = {}
        self.estimated_clear_attempts: dict[int, int] = {}
        self.estimated_clear_bearing_counts: dict[int, int] = {}
        self.one_bearing_posterior_attempted: set[int] = set()
        self.start_wall_time = 0.0
        self.remaining_real_duration_s: float | None = None
        self.real_deadline: float | None = None
        self.exit_reserve_s = 0.0
        self.next_task_hint: np.ndarray | None = None
        self.refinement_stats = {
            "constraint_updates": 0,
            "constraint_shrinks": 0,
            "lookahead_choices": 0,
            "safe_clear_choices": 0,
            "planning_fallbacks": 0,
        }
        self._constraint_cache: dict[int, tuple[Any, np.ndarray]] = {}
        self.deadline_monotonic: float | None = None
        self.inferred_no_signal: dict[int, set[tuple[float, float]]] = {}
        self.skipped_rechecks = 0
        self.region_cache_hits = 0
        self.region_cache_misses = 0
        self._region_cache: dict[int, tuple[Any, Any]] = {}
        self.budget_degraded = False
        self.discovery_complete = False
        self.service_point_uses = 0
        self.joint_order_changes = 0
        self._q3_following_hint: np.ndarray | None = None
        self.distance_m = 0.0
        self.measure_count = 0
        self.switch_count = 0
        self.clear_success_count = 0
        self.clear_failure_count = 0

    def _check_budget(self) -> None:
        if (
            self.cfg.budget_aware
            and self.real_deadline is not None
            and time.monotonic() >= self.real_deadline - self.exit_reserve_s
        ):
            raise RuntimeBudgetExceeded("real-time action budget exhausted; exiting")

    def _planning_deadline(self) -> float:
        self._check_budget()
        deadline = time.monotonic() + max(0.001, self.cfg.q4_planning_budget_s)
        if self.cfg.budget_aware and self.real_deadline is not None:
            deadline = min(deadline, self.real_deadline - self.exit_reserve_s)
        return deadline

    def _remaining_wall_time(self) -> float:
        if self.deadline_monotonic is None:
            return math.inf
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def _low_budget(self) -> bool:
        low = self.cfg.budget_aware and self._remaining_wall_time() <= self.cfg.planning_guard_s
        self.budget_degraded = self.budget_degraded or low
        return low

    @property
    def deadline_monotonic(self) -> float | None:
        """Compatibility alias for the shared absolute real-time deadline."""
        return self.real_deadline

    @deadline_monotonic.setter
    def deadline_monotonic(self, value: float | None) -> None:
        self.real_deadline = value

    def _post(self, path: str, **body):
        if path != "/exit":
            self._check_budget()
        if path != "/exit" and self.action_count >= max(1, self.cfg.max_actions - 1):
            raise RuntimeError(f"action limit {self.cfg.max_actions} reached; exit reserved")
        self.action_count += 1
        response = self.client.post(path, body)
        if response.get("accepted") is not True:
            raise RuntimeError(f"rejected {path}: {response}")
        self.last_virtual_time = float(response.get("virtual_time_s", self.last_virtual_time))
        if "position" in body:
            new_position = (float(body["position"]["x"]), float(body["position"]["y"]))
            self.distance_m += _distance(self.position, new_position)
            self.position = new_position
        if path == "/clear":
            self.clear_success_count += int(response.get("clear_result") == "success")
            self.clear_failure_count += int(response.get("clear_result") != "success")
        if path == "/measure":
            self.measure_count += 1
            self.switch_count += int(self.channel != int(body["channel"]))
            self.channel = int(body["channel"])
            self.obs.append(
                Observation(
                    self.position,
                    self.channel,
                    str(response["measure_result"]),
                    float(response["svd_deg"]) if "svd_deg" in response else None,
                    self.last_virtual_time,
                )
            )
        return response

    def measure(self, position: Sequence[float], channel: int):
        return self._post(
            "/measure",
            position={"x": float(position[0]), "y": float(position[1])},
            channel=int(channel),
        )

    def clear(self, position: Sequence[float], channel: int):
        point = (float(position[0]), float(position[1]))
        response = self._post(
            "/clear", position={"x": point[0], "y": point[1]}, channel=int(channel)
        )
        if response.get("clear_result") == "success":
            self.cleared.add(int(channel))
        else:
            self.failed_clear_points.setdefault(int(channel), []).append(point)
        return response

    def _scan_points(self) -> list[tuple[float, float]]:
        if self.cfg.problem == 3:
            if self.cfg.q3_custom_ring:
                points = q3_custom_ring_points(
                    self.cfg.q3_ring_radius,
                    self.cfg.q3_ring_sides,
                    self.cfg.q3_include_center,
                )
            elif self.cfg.q3_layout == "legacy":
                points = q3_cover_points(self.cfg.q3_ring_radius)
            elif self.cfg.q3_layout == "ring":
                points = q3_ring_cover_points(
                    self.cfg.target_radius,
                    self.cfg.min_receive_radius,
                    self.cfg.q3_ring_sides,
                )
            elif self.cfg.q3_layout == "minimax":
                points = q3_minimax_ring_points(
                    self.cfg.target_radius,
                    self.cfg.min_receive_radius,
                    self.cfg.q3_ring_sides,
                )
            else:
                raise ValueError(f"unknown Q3 scan layout: {self.cfg.q3_layout}")
            return _open_route(points, start=(0.0, 0.0))
        if self.cfg.q4_layout in {"compact", "certified21"}:
            mesh = q4_certified21_cover_points(
                self.cfg.target_radius,
                self.cfg.min_receive_radius,
            )
        elif self.cfg.q4_layout == "compact25":
            mesh = q4_compact_cover_points(
                self.cfg.target_radius,
                self.cfg.min_receive_radius,
            )
        elif self.cfg.q4_layout == "legacy":
            mesh = q4_triangular_cover_points(self.cfg.target_radius, self.cfg.q4_mesh_edge)
        else:
            raise ValueError(f"unknown Q4 scan layout: {self.cfg.q4_layout}")
        return _open_route(mesh, start=(0.0, 0.0))

    def _channel_order(self, channels: set[int]) -> list[int]:
        if self.channel in channels:
            return [self.channel] + sorted(channels - {self.channel})
        return sorted(channels)

    def _direction_observations(self, channel: int) -> list[Observation]:
        return [o for o in self.obs if o.channel == channel and o.angle_deg is not None]

    def _region_for(self, channel: int) -> tuple[np.ndarray, float, Any]:
        """Cache by all constraint inputs, never by observation count alone."""
        support = self.discovery_support_polygons.get(channel)
        key = (
            tuple((o.position, o.result, o.angle_deg) for o in self.obs if o.channel == channel),
            tuple(sorted(self.inferred_no_signal.get(channel, set()))),
            tuple((tuple(np.asarray(a, dtype=float)), float(b))
                  for a, b in self.region_halfplanes.get(channel, [])),
            None if support is None else np.asarray(support, dtype=float).tobytes(),
            channel in self.local_positive_cap_channels,
            self.cfg.problem, self.cfg.q3_range_order_constraints,
            self.cfg.target_radius, self.cfg.max_error_deg,
            self.cfg.min_receive_radius, self.cfg.max_receive_radius,
            tuple(self.failed_clear_points.get(channel, [])),
            self.cfg.q4_joint_constraints, self.cfg.q4_constraint_cells,
            self.cfg.q4_planning_budget_s, self.cfg.clear_radius,
        )
        cached = self._region_cache.get(channel)
        if cached is not None and cached[0] == key:
            self.region_cache_hits += 1
            return cached[1]
        self.region_cache_misses += 1
        value = self._compute_region_for(channel)
        self._region_cache[channel] = (key, value)
        return value

    def _compute_region_for(self, channel: int) -> tuple[np.ndarray, float, Any]:
        observations = self._direction_observations(channel)
        if not observations:
            return np.empty((0, 2)), math.inf, None
        bearing_data = [(o.position, math.radians(float(o.angle_deg))) for o in observations]
        if channel in self.local_positive_cap_channels:
            components = []
            for capped_index in range(len(observations)):
                receive_disks = [
                    (
                        observation.position,
                        self.cfg.min_receive_radius
                        if index == capped_index
                        else self.cfg.max_receive_radius,
                    )
                    for index, observation in enumerate(observations)
                ]
                component = bearing_region(
                    bearing_data,
                    target_radius=self.cfg.target_radius,
                    error_deg=self.cfg.max_error_deg,
                    extra_disks=receive_disks,
                    n_disk=360,
                )
                if len(component):
                    components.append(component)
            region = (
                convex_hull(np.vstack(components))
                if components
                else np.empty((0, 2))
            )
        else:
            receive_disks = [
                (o.position, self.cfg.max_receive_radius)
                for o in observations
            ]
            region = bearing_region(
                bearing_data,
                target_radius=self.cfg.target_radius,
                error_deg=self.cfg.max_error_deg,
                extra_disks=receive_disks,
                n_disk=360,
            )
        support = self.discovery_support_polygons.get(channel)
        if support is not None and len(support):
            region = intersect_convex_polygons(region, support)
        if self.cfg.problem == 3 and self.cfg.q3_range_order_constraints:
            # With an omnidirectional source and one fixed reception radius,
            # every positive observer p is closer than every no-signal
            # observer n. Squared distances give the exact half-plane below.
            positive_points = [
                np.asarray(observation.position, dtype=float)
                for observation in self.obs
                if observation.channel == channel
                and observation.result in {"direction", "near"}
            ]
            negative_points = [
                np.asarray(observation.position, dtype=float)
                for observation in self.obs
                if observation.channel == channel
                and observation.result == "no_signal"
            ]
            # These are proved no-signal locations, not fabricated observations.
            negative_points.extend(np.asarray(point, dtype=float)
                                   for point in sorted(self.inferred_no_signal.get(channel, set())))
            for positive in positive_points:
                for negative in negative_points:
                    normal = positive - negative
                    bound = 0.5 * float(
                        np.dot(positive, positive) - np.dot(negative, negative)
                    )
                    region = clip_convex_polygon_halfplane(region, normal, bound)
                    if len(region) == 0:
                        return region, math.inf, None
        for normal, bound in self.region_halfplanes.get(channel, []):
            region = clip_convex_polygon_halfplane(region, normal, bound)
            if len(region) == 0:
                return region, math.inf, None
        if len(region) == 0:
            return region, math.inf, None
        if self.cfg.problem == 4 and self.cfg.q4_joint_constraints and len(region) >= 3:
            from q4_refinement import refine_region

            relevant = [o for o in self.obs if o.channel == channel]
            failed = self.failed_clear_points.get(channel, [])
            signature = (
                tuple((o.position, o.result, o.angle_deg) for o in relevant),
                tuple(failed),
                region.tobytes(),
                self.cfg.min_receive_radius,
                self.cfg.max_receive_radius,
                self.cfg.clear_radius,
                self.cfg.q4_constraint_cells,
                self.cfg.q4_planning_budget_s,
            )
            cached = self._constraint_cache.get(channel)
            if cached is not None and cached[0] == signature:
                region = cached[1].copy()
            else:
                refined = refine_region(
                    region,
                    relevant,
                    failed,
                    min_radius=self.cfg.min_receive_radius,
                    max_radius=self.cfg.max_receive_radius,
                    clear_radius=self.cfg.clear_radius,
                    max_cells=self.cfg.q4_constraint_cells,
                    deadline=self._planning_deadline(),
                )
                self.refinement_stats["constraint_updates"] += 1
                if refined.timed_out:
                    self.refinement_stats["planning_fallbacks"] += 1
                elif refined.rejected_cells:
                    self.refinement_stats["constraint_shrinks"] += 1
                region = refined.region
                self._constraint_cache[channel] = (signature, region.copy())
        circle = min_enclosing_circle(region)
        return region, diameter(region), circle

    def _view_candidates(
        self,
        channel: int,
        region: np.ndarray,
        prefer_travel: bool = True,
    ) -> list[tuple[float, float]]:
        candidates = self._geometry_view_candidates(channel, region, prefer_travel)
        if self.cfg.problem != 4 or not self.cfg.q4_time_lookahead or not candidates:
            return candidates
        from q4_refinement import rank_measurements

        ranked, changed = rank_measurements(
            candidates,
            region,
            [o for o in self.obs if o.channel == channel],
            self.position,
            self.failed_clear_points.get(channel, []),
            error_deg=self.cfg.max_error_deg,
            switch_cost=float(self.channel != channel),
            deadline=self._planning_deadline(),
        )
        self.refinement_stats["lookahead_choices"] += int(changed)
        return ranked

    def _geometry_view_candidates(
        self,
        channel: int,
        region: np.ndarray,
        prefer_travel: bool = True,
    ) -> list[tuple[float, float]]:
        observations = self._direction_observations(channel)
        if not observations or len(region) == 0:
            return []
        tried = [np.asarray(o.position, dtype=float) for o in self.obs if o.channel == channel]
        vertices = np.asarray(region, dtype=float)
        samples = [p for p in vertices]
        samples.extend((vertices[i] + vertices[(i + 1) % len(vertices)]) / 2.0 for i in range(len(vertices)))
        samples.append(np.mean(vertices, axis=0))
        samples_array = np.asarray(samples)

        raw: list[np.ndarray] = []
        guaranteed_positive: list[np.ndarray] = []
        circle = min_enclosing_circle(vertices)
        raw.append(circle.center.copy())
        if len(observations) == 1:
            first = observations[0]
            p = np.asarray(first.position, dtype=float)
            u = unit(math.radians(float(first.angle_deg)))
            normal = np.array([-u[1], u[0]])
            projection = (vertices - p) @ u
            for fraction in (0.42, 0.50, 0.58):
                anchor = p + (float(np.min(projection)) + fraction * float(np.ptp(projection))) * u
                for offset in (650.0, 600.0, 550.0, 475.0, 400.0, 300.0, 200.0):
                    raw.extend((anchor + offset * normal, anchor - offset * normal))

        # A reception disk and a directional emitting half-plane are both
        # convex.  Therefore every convex combination of two positive
        # observers is guaranteed to remain receivable for the true source,
        # even when its actual reception radius is greater than the known
        # 1000 m lower bound.
        if self.cfg.problem == 4 and self.cfg.q4_positive_hull_views and len(observations) >= 2:
            for left_index in range(len(existing_positions := [np.asarray(o.position, dtype=float) for o in observations])):
                for right_index in range(left_index + 1, len(existing_positions)):
                    left = existing_positions[left_index]
                    right = existing_positions[right_index]
                    for fraction in (0.20, 0.35, 0.50, 0.65, 0.80):
                        guaranteed_positive.append((1.0 - fraction) * left + fraction * right)
            guaranteed_positive.append(np.mean(existing_positions, axis=0))
            raw.extend(guaranteed_positive)

        for radius in (120.0, 250.0, 450.0, 650.0, 850.0):
            for k in range(24):
                raw.append(circle.center + radius * unit(2.0 * math.pi * k / 24.0))

        current = np.asarray(self.position, dtype=float)
        reception_limit = self.cfg.min_receive_radius - 2.0
        if self.cfg.problem == 3 and self.cfg.q3_entry_candidates:
            feasible_seeds = [
                candidate
                for candidate in raw
                if float(np.max(np.linalg.norm(vertices - candidate, axis=1)))
                <= reception_limit
            ]
            current_max_distance = float(
                np.max(np.linalg.norm(vertices - current, axis=1))
            )
            if current_max_distance <= reception_limit:
                raw.append(current.copy())
            else:
                # The guaranteed-reception set is convex. Enter it along
                # informative rays instead of travelling to their far ends.
                for target in feasible_seeds:
                    low, high = 0.0, 1.0
                    for _ in range(36):
                        fraction = 0.5 * (low + high)
                        candidate = current + fraction * (target - current)
                        maximum = float(
                            np.max(np.linalg.norm(vertices - candidate, axis=1))
                        )
                        if maximum <= reception_limit:
                            high = fraction
                        else:
                            low = fraction
                    fraction = min(1.0, high + 1e-8)
                    raw.append(current + fraction * (target - current))

        unique: list[np.ndarray] = []
        for candidate in raw:
            if any(np.linalg.norm(candidate - prior) < 1.0 for prior in tried):
                continue
            if any(np.linalg.norm(candidate - prior) < 1e-6 for prior in unique):
                continue
            safe_by_positive_hull = any(
                np.linalg.norm(candidate - safe) < 1e-6
                for safe in guaranteed_positive
            )
            if (
                not safe_by_positive_hull
                and float(np.max(np.linalg.norm(vertices - candidate, axis=1))) > reception_limit
            ):
                continue
            unique.append(candidate)

        if self._low_budget():
            # The candidate feasibility tests remain unchanged; only expensive
            # information ranking is replaced by nearest certified reception.
            unique.sort(key=lambda p: (_distance(p, self.position), float(p[0]), float(p[1])))
            return [tuple(map(float, p)) for p in unique]

        existing_positions = [np.asarray(o.position, dtype=float) for o in observations]

        def information_score(candidate: np.ndarray) -> float:
            worst_information = math.inf
            for source in samples_array:
                information = np.zeros((2, 2), dtype=float)
                for observer in existing_positions + [candidate]:
                    vector = source - observer
                    distance_value = max(5.0, float(np.linalg.norm(vector)))
                    normal = np.array([-vector[1], vector[0]]) / distance_value
                    information += np.outer(normal, normal) / (distance_value * distance_value)
                worst_information = min(worst_information, float(np.linalg.eigvalsh(information)[0]))
            return max(0.0, worst_information)

        scored = [
            (
                point,
                information_score(point),
                float(np.linalg.norm(point - np.asarray(self.position))),
            )
            for point in unique
        ]
        if self.cfg.problem == 3 and self.cfg.q3_information_floor_fraction > 0.0:
            best_information = max((item[1] for item in scored), default=0.0)
            floor = self.cfg.q3_information_floor_fraction * best_information
            eligible = [item for item in scored if item[1] + EPS >= floor]
            rejected = [item for item in scored if item[1] + EPS < floor]
            eligible.sort(key=lambda item: (item[2], -item[1], item[0][0], item[0][1]))
            rejected.sort(key=lambda item: (-item[1], item[2], item[0][0], item[0][1]))
            return [tuple(map(float, item[0])) for item in eligible + rejected]

        if (
            self.cfg.problem == 4
            and prefer_travel
            and self.cfg.q4_information_floor_fraction > 0.0
        ):
            best_information = max((item[1] for item in scored), default=0.0)
            floor = self.cfg.q4_information_floor_fraction * best_information
            eligible = [item for item in scored if item[1] + EPS >= floor]
            rejected = [item for item in scored if item[1] + EPS < floor]
            eligible.sort(key=lambda item: (item[2], -item[1], item[0][0], item[0][1]))
            rejected.sort(key=lambda item: (-item[1], item[2], item[0][0], item[0][1]))
            return [tuple(map(float, item[0])) for item in eligible + rejected]

        def score(item: tuple[np.ndarray, float, float]) -> float:
            _, worst_information, travel = item
            travel_weight = (
                self.cfg.q3_view_travel_weight
                if self.cfg.problem == 3
                else 2e-4
            )
            travel_penalty = travel_weight * travel
            return 1e6 * worst_information - travel_penalty

        return [
            tuple(map(float, item[0]))
            for item in sorted(scored, key=score, reverse=True)
        ]

    def _one_bearing_posterior_estimate(
        self,
        channel: int,
        region: np.ndarray,
    ) -> np.ndarray:
        """Estimate range along one bearing from the scan-stage shadow pattern."""
        observations = self._direction_observations(channel)
        if len(observations) != 1:
            return min_enclosing_circle(region).center.copy()
        mode = self.cfg.q4_one_bearing_posterior
        if mode not in {"map", "mean"}:
            raise ValueError(f"unknown one-bearing posterior mode: {mode}")

        positive = observations[0]
        origin = np.asarray(positive.position, dtype=float)
        measured = math.radians(float(positive.angle_deg))
        negative_points = np.asarray(
            [
                observation.position
                for observation in self.obs
                if observation.channel == channel
                and observation.result == "no_signal"
            ],
            dtype=float,
        )
        orientations = 2.0 * math.pi * np.arange(72) / 72.0
        orientation_vectors = np.column_stack(
            (np.cos(orientations), np.sin(orientations))
        )
        receive_radii = np.linspace(
            self.cfg.min_receive_radius,
            self.cfg.max_receive_radius,
            11,
        )
        candidates: list[np.ndarray] = []
        weights: list[float] = []
        for error_deg in np.linspace(-self.cfg.max_error_deg, self.cfg.max_error_deg, 9):
            direction = unit(measured + math.radians(float(error_deg)))
            for source_distance in np.linspace(10.0, self.cfg.max_receive_radius, 150):
                source = origin + source_distance * direction
                if not point_in_convex_polygon(source, region, tol=1e-7):
                    continue
                positive_mask = orientation_vectors @ (origin - source) >= -1e-9
                if len(negative_points):
                    negative_vectors = negative_points - source
                    negative_distances = np.linalg.norm(negative_vectors, axis=1)
                else:
                    negative_vectors = np.empty((0, 2))
                    negative_distances = np.empty(0)
                weight = 0.0
                for receive_radius in receive_radii:
                    if source_distance > receive_radius + 1e-9:
                        continue
                    mask = positive_mask.copy()
                    if len(negative_points):
                        close = negative_vectors[
                            negative_distances <= receive_radius + 1e-9
                        ]
                        if len(close):
                            mask &= np.all(
                                orientation_vectors @ close.T < 1e-9,
                                axis=1,
                            )
                    weight += float(np.mean(mask))
                weight *= max(source_distance, 1.0)
                if weight > 0.0:
                    candidates.append(source)
                    weights.append(weight)
        if not candidates:
            return min_enclosing_circle(region).center.copy()
        candidate_array = np.asarray(candidates)
        weight_array = np.asarray(weights)
        if mode == "map":
            return candidate_array[int(np.argmax(weight_array))].copy()
        return np.average(candidate_array, axis=0, weights=weight_array)

    @staticmethod
    def _polygon_centroid(region: np.ndarray) -> np.ndarray:
        """Return the area centroid of a nonempty convex polygon."""
        if len(region) < 3:
            return np.mean(region, axis=0)
        following = np.roll(region, -1, axis=0)
        signed_cross = region[:, 0] * following[:, 1] - following[:, 0] * region[:, 1]
        cross_sum = float(np.sum(signed_cross))
        if abs(cross_sum) <= EPS:
            return np.mean(region, axis=0)
        return np.sum((region + following) * signed_cross[:, None], axis=0) / (3.0 * cross_sum)

    @staticmethod
    def _project_to_region(point: np.ndarray, region: np.ndarray) -> np.ndarray:
        """Project a point onto a convex polygon when an estimate leaves it."""
        if point_in_convex_polygon(point, region, tol=1e-8):
            return point.copy()
        best_point = region[0].copy()
        best_distance = math.inf
        for index, left in enumerate(region):
            right = region[(index + 1) % len(region)]
            edge = right - left
            scale = float(np.dot(edge, edge))
            fraction = 0.0 if scale <= EPS else float(np.dot(point - left, edge) / scale)
            candidate = left + min(1.0, max(0.0, fraction)) * edge
            distance_value = float(np.linalg.norm(point - candidate))
            if distance_value < best_distance:
                best_distance = distance_value
                best_point = candidate
        return best_point

    def _bearing_center_estimate(self, channel: int, region: np.ndarray) -> np.ndarray:
        """Estimate the source from central bearing-line intersections.

        Only intersections inside the bounded-error feasible region are used.
        Squared crossing-angle weights suppress nearly parallel line pairs.
        This is an expected-time fast path; the region itself remains the
        deterministic certificate used by all fallback logic.
        """
        observations = self._direction_observations(channel)
        intersections: list[np.ndarray] = []
        weights: list[float] = []
        for left_index in range(len(observations)):
            left = observations[left_index]
            left_origin = np.asarray(left.position, dtype=float)
            left_direction = unit(math.radians(float(left.angle_deg)))
            for right_index in range(left_index + 1, len(observations)):
                right = observations[right_index]
                right_origin = np.asarray(right.position, dtype=float)
                right_direction = unit(math.radians(float(right.angle_deg)))
                denominator = cross(left_direction, right_direction)
                if abs(denominator) < math.sin(math.radians(3.0)):
                    continue
                delta = right_origin - left_origin
                left_distance = cross(delta, right_direction) / denominator
                right_distance = cross(delta, left_direction) / denominator
                if left_distance < 0.0 or right_distance < 0.0:
                    continue
                candidate = left_origin + left_distance * left_direction
                if not point_in_convex_polygon(candidate, region, tol=1e-7):
                    continue
                intersections.append(candidate)
                weights.append(float(denominator * denominator))
        pairwise = (
            np.average(np.asarray(intersections), axis=0, weights=np.asarray(weights))
            if intersections
            else min_enclosing_circle(region).center.copy()
        )
        mode = self.cfg.q4_estimator
        if mode == "pairwise":
            return pairwise

        centroid = self._polygon_centroid(region)
        if mode == "centroid":
            return centroid
        if mode == "blend":
            return self._project_to_region(0.75 * pairwise + 0.25 * centroid, region)
        if mode != "angular_irls":
            raise ValueError(f"unknown Q4 bearing estimator: {mode}")

        # For a small angular error, perpendicular line residual divided by
        # observer range is the angular residual. Iteratively updating those
        # ranges gives a lightweight bearing-only nonlinear least-squares fit.
        estimate = pairwise.copy()
        normals = []
        origins = []
        for observation in observations:
            direction = unit(math.radians(float(observation.angle_deg)))
            normals.append(np.array([-direction[1], direction[0]], dtype=float))
            origins.append(np.asarray(observation.position, dtype=float))
        normal_matrix = np.asarray(normals)
        origin_matrix = np.asarray(origins)
        right_hand_side = np.sum(normal_matrix * origin_matrix, axis=1)
        for _ in range(12):
            ranges = np.maximum(20.0, np.linalg.norm(estimate - origin_matrix, axis=1))
            weighted = normal_matrix / ranges[:, None]
            target = right_hand_side / ranges
            candidate, *_ = np.linalg.lstsq(weighted, target, rcond=None)
            candidate = self._project_to_region(candidate, region)
            if float(np.linalg.norm(candidate - estimate)) <= 1e-5:
                estimate = candidate
                break
            estimate = candidate
        return estimate

    def _q4_recover_from_shadow(
        self,
        channel: int,
        negative_point: Sequence[float],
    ) -> tuple[bool, bool]:
        """Bisect from a known shadow point toward a positive observer.

        The segment stays inside the actual reception disk.  Its positive end
        is in the emitting half-plane, so finite bisection either obtains a
        fresh bearing or moves monotonically toward a point known to receive.
        Returns ``(cleared, new_bearing)``.
        """
        observations = self._direction_observations(channel)
        if not observations:
            return False, False
        negative = np.asarray(negative_point, dtype=float)
        positive = min(
            (np.asarray(observation.position, dtype=float) for observation in observations),
            key=lambda point: float(np.linalg.norm(point - negative)),
        )
        for _ in range(max(1, int(self.cfg.q4_shadow_bisection_rounds))):
            probe = 0.5 * (negative + positive)
            response = self.measure(probe, channel)
            result = response.get("measure_result")
            if result == "near":
                return self.clear(probe, channel).get("clear_result") == "success", False
            if result == "direction":
                return False, True
            negative = probe
        return False, False

    def _certified_service_point(
        self, region: np.ndarray, circle: Any, following: Sequence[float] | None = None
    ) -> np.ndarray:
        """Choose a shorter-route point in the intersection of clear disks.

        Convex line searches generate candidates, not a claim of global SOCP
        optimality. Every returned point is checked against ALL region vertices.
        The original enclosing-circle center is always a feasible fallback.
        """
        center = np.asarray(circle.center, dtype=float)
        radius = self.cfg.clear_guarantee_radius
        if circle.radius > radius or self._low_budget():
            return center.copy()
        vertices = np.asarray(region, dtype=float)
        if not len(vertices) or float(np.max(np.linalg.norm(vertices - center, axis=1))) > radius + 1e-8:
            return center.copy()
        start = np.asarray(self.position, dtype=float)
        end = None if following is None else np.asarray(following, dtype=float)

        def cost(point: np.ndarray) -> float:
            return float(np.linalg.norm(point - start)) + (
                0.0 if end is None else float(np.linalg.norm(point - end))
            )

        directions = [start - center]
        if end is not None:
            directions.extend([end - center, 0.5 * (start + end) - center])
        directions.extend(unit(2.0 * math.pi * k / 16.0) for k in range(16))
        best, best_cost = center.copy(), cost(center)
        for vector in directions:
            norm = float(np.linalg.norm(vector))
            if norm < 1e-10:
                continue
            direction = vector / norm
            # Along each ray, circle constraints give an analytic interval.
            delta = center - vertices
            projection = delta @ direction
            discriminants = np.maximum(0.0, projection**2 + radius**2 - np.sum(delta**2, axis=1))
            maximum = max(0.0, float(np.min(-projection + np.sqrt(discriminants))))
            if maximum <= 1e-8:
                continue
            if end is None:
                distance_along = min(maximum, max(0.0, float(np.dot(start - center, direction))))
            else:
                left, right = 0.0, maximum
                for _ in range(30):
                    one = left + (right - left) / 3.0
                    two = right - (right - left) / 3.0
                    if cost(center + one * direction) <= cost(center + two * direction):
                        right = two
                    else:
                        left = one
                distance_along = 0.5 * (left + right)
            point = center + distance_along * direction
            value = cost(point)
            if value + 1e-7 < best_cost and float(np.max(np.linalg.norm(vertices - point, axis=1))) <= radius + 1e-8:
                best, best_cost = point, value
        return best

    def _clear_if_certain(self, channel: int, circle: Any) -> bool:
        if circle is None or circle.radius > self.cfg.clear_guarantee_radius:
            return False
        point = circle.center
        if self.cfg.problem == 4 and self.cfg.q4_safe_clear_route:
            from q4_refinement import safe_clear_point

            region, _, _ = self._region_for(channel)
            if len(region):
                candidate = safe_clear_point(
                    region,
                    self.position,
                    self.next_task_hint,
                    radius=self.cfg.clear_guarantee_radius,
                    reference=circle.center,
                    deadline=self._planning_deadline(),
                )
                if candidate is not None:
                    self.refinement_stats["safe_clear_choices"] += int(
                        _distance(candidate, point) > 1e-5
                    )
                    point = candidate
        if self.cfg.problem == 3 and self.cfg.q3_service_region:
            region, _, _ = self._region_for(channel)
            point = self._certified_service_point(region, circle, self._q3_following_hint)
            if _distance(point, circle.center) > 1e-6:
                self.service_point_uses += 1
        response = self.clear(point, channel)
        return response.get("clear_result") == "success"

    def _clear_cover_points(self, region: np.ndarray) -> list[tuple[float, float]]:
        """Triangular-lattice cover whose covering radius is below 20 m."""
        cover_radius = self.cfg.clear_guarantee_radius
        edge = math.sqrt(3.0) * cover_radius * 0.98
        vertical = math.sqrt(3.0) * edge / 2.0
        lower = np.min(region, axis=0) - cover_radius
        upper = np.max(region, axis=0) + cover_radius
        j_min = int(math.floor(lower[1] / vertical)) - 1
        j_max = int(math.ceil(upper[1] / vertical)) + 1
        points: list[tuple[float, float]] = []
        for j in range(j_min, j_max + 1):
            self._check_budget()
            y = j * vertical
            shift = 0.5 * edge * j
            i_min = int(math.floor((lower[0] - shift) / edge)) - 1
            i_max = int(math.ceil((upper[0] - shift) / edge)) + 1
            for i in range(i_min, i_max + 1):
                point = np.array([i * edge + shift, y], dtype=float)
                if point_polygon_distance(point, region) <= cover_radius + 1e-7:
                    points.append((float(point[0]), float(point[1])))
        return _open_route(points, start=self.position)

    def _fallback_clear(self, channel: int, region: np.ndarray) -> bool:
        points = self._clear_cover_points(region)
        for point in points:
            if any(_distance(point, old) < 1e-6 for old in self.failed_clear_points.get(channel, [])):
                continue
            response = self.clear(point, channel)
            if response.get("clear_result") == "success":
                return True
        return False

    def _q4_recover_single_bearing(self, channel: int, region: np.ndarray) -> tuple[bool, np.ndarray]:
        """Resolve a one-bearing directional source by symmetric bisection.

        The first positive observer is inside the source's emitting half-plane.
        At a chosen ray split, two lateral probes bracket every orientation
        compatible with that observation.  Both probes are also required to
        lie within the guaranteed 1000 m reception distance of the complete
        feasible region.  Therefore two no-signal responses safely imply that
        the source lies before the split, while either positive response adds
        a second bearing without blind clear attempts.
        """
        observations = self._direction_observations(channel)
        if len(observations) != 1 or len(region) == 0:
            return False, region

        first = observations[0]
        origin = np.asarray(first.position, dtype=float)
        direction = unit(math.radians(float(first.angle_deg)))
        normal = np.array([-direction[1], direction[0]], dtype=float)
        narrowed = np.asarray(region, dtype=float)
        reception_limit = self.cfg.min_receive_radius - 2.0
        tangent = math.tan(math.radians(self.cfg.max_error_deg))

        for _ in range(max(1, int(self.cfg.q4_recovery_rounds))):
            circle = min_enclosing_circle(narrowed)
            if circle.radius <= self.cfg.clear_guarantee_radius:
                return self._clear_if_certain(channel, circle), narrowed

            projections = (narrowed - origin) @ direction
            lower = max(0.0, float(np.min(projections)))
            upper = float(np.max(projections))
            if upper <= lower + 1e-7:
                break
            split = 0.5 * (lower + upper)

            # For any feasible source q=t*u+s*v with t>=split, the bearing
            # wedge gives |s|<=upper*tan(error).  This lateral separation
            # guarantees at least one probe remains in every emitting
            # half-plane that contains the original positive observer.
            lateral = upper * tangent + 0.5
            midpoint = origin + split * direction
            probes = [midpoint + lateral * normal, midpoint - lateral * normal]
            if any(
                float(np.max(np.linalg.norm(narrowed - probe, axis=1))) > reception_limit
                for probe in probes
            ):
                break
            probes.sort(key=lambda probe: float(np.linalg.norm(probe - np.asarray(self.position))))

            positive = False
            for probe in probes:
                response = self.measure(probe, channel)
                result = response.get("measure_result")
                if result == "near":
                    return self.clear(probe, channel).get("clear_result") == "success", narrowed
                if result == "direction":
                    positive = True
                    break
            if positive:
                return False, narrowed

            # Both guaranteed-reception probes returned no_signal.  The
            # positive-span argument above makes t < split a hard constraint.
            bound = -split - float(np.dot(direction, origin))
            self.region_halfplanes.setdefault(channel, []).append((-direction.copy(), bound))
            narrowed = clip_convex_polygon_halfplane(narrowed, -direction, bound)
            if len(narrowed) == 0:
                return False, region

        circle = min_enclosing_circle(narrowed)
        if circle.radius <= self.cfg.clear_guarantee_radius:
            return self._clear_if_certain(channel, circle), narrowed
        return False, narrowed

    def _localize_and_clear(self, channel: int) -> bool:
        region, _, _ = self._region_for(channel)
        if (
            self.cfg.problem == 4
            and self.cfg.q4_symmetric_recovery
            and len(self._direction_observations(channel)) == 1
            and len(region) > 0
        ):
            posterior_used = (
                self.cfg.q4_one_bearing_posterior != "off"
                and channel not in self.one_bearing_posterior_attempted
            )
            if posterior_used:
                self.one_bearing_posterior_attempted.add(channel)
                estimate = self._one_bearing_posterior_estimate(channel, region)
                if self.clear(estimate, channel).get("clear_result") == "success":
                    return True
                response = self.measure(estimate, channel)
                if response.get("measure_result") == "near":
                    return self.clear(estimate, channel).get("clear_result") == "success"
                if response.get("measure_result") == "direction":
                    region, _, _ = self._region_for(channel)
            elif self.cfg.q4_one_bearing_initial_probe:
                # Preserve the legacy fast path when the posterior experiment
                # is disabled: one high-information probe often gets a second
                # bearing immediately.
                candidates = self._view_candidates(channel, region)
                if candidates:
                    response = self.measure(candidates[0], channel)
                    if response.get("measure_result") == "near":
                        return self.clear(candidates[0], channel).get("clear_result") == "success"
                    if response.get("measure_result") == "direction":
                        region, _, _ = self._region_for(channel)
            cleared, narrowed = self._q4_recover_single_bearing(channel, region)
            if cleared:
                return True
            if len(self._direction_observations(channel)) == 1:
                return self._fallback_clear(channel, narrowed)

        consecutive_no_signal = 0
        for _ in range(self.cfg.max_view_attempts):
            region, _, circle = self._region_for(channel)
            if len(region) == 0:
                return False
            if self._clear_if_certain(channel, circle):
                return True
            estimated_clear_radius = (
                self.cfg.q4_estimated_clear_radius
                if self.cfg.problem == 4
                else self.cfg.q3_estimated_clear_radius
            )
            estimated_clear_attempts = (
                self.cfg.q4_estimated_clear_attempts
                if self.cfg.problem == 4
                else self.cfg.q3_estimated_clear_attempts
            )
            measure_after_failed_estimate = (
                self.cfg.q4_measure_after_failed_estimate
                if self.cfg.problem == 4
                else self.cfg.q3_measure_after_failed_estimate
            )
            if (
                estimated_clear_radius > 0.0
                and circle.radius <= estimated_clear_radius
                and len(self._direction_observations(channel)) >= 2
                and self.estimated_clear_attempts.get(channel, 0)
                < estimated_clear_attempts
                and self.estimated_clear_bearing_counts.get(channel)
                != len(self._direction_observations(channel))
            ):
                estimate = self._bearing_center_estimate(channel, region)
                self.estimated_clear_attempts[channel] = (
                    self.estimated_clear_attempts.get(channel, 0) + 1
                )
                self.estimated_clear_bearing_counts[channel] = len(
                    self._direction_observations(channel)
                )
                if self.clear(estimate, channel).get("clear_result") == "success":
                    return True
                if measure_after_failed_estimate:
                    response = self.measure(estimate, channel)
                    if response.get("measure_result") == "near":
                        return self.clear(estimate, channel).get("clear_result") == "success"
                    if (
                        self.cfg.problem == 4
                        and response.get("measure_result") == "no_signal"
                        and self.cfg.q4_shadow_bisection
                        and 2.0 * circle.radius < self.cfg.min_receive_radius - 2.0
                    ):
                        cleared, new_bearing = self._q4_recover_from_shadow(
                            channel, estimate
                        )
                        if cleared:
                            return True
                        consecutive_no_signal = 0 if new_bearing else consecutive_no_signal + 1
                        continue
                    consecutive_no_signal = (
                        0 if response.get("measure_result") == "direction" else consecutive_no_signal + 1
                    )
                    continue
            speculative_radius = (
                self.cfg.q4_speculative_clear_radius
                if self.cfg.problem == 4
                else 30.0
            )
            if circle.radius <= speculative_radius and not any(
                _distance(circle.center, old) < 1e-6 for old in self.failed_clear_points.get(channel, [])
            ):
                if self.clear(circle.center, channel).get("clear_result") == "success":
                    return True

            candidates = self._view_candidates(
                channel,
                region,
                prefer_travel=consecutive_no_signal == 0,
            )
            if not candidates:
                break
            response = self.measure(candidates[0], channel)
            if response.get("measure_result") == "near":
                return self.clear(candidates[0], channel).get("clear_result") == "success"
            if response.get("measure_result") == "direction":
                consecutive_no_signal = 0
            else:
                consecutive_no_signal += 1

        region, _, _ = self._region_for(channel)
        return len(region) > 0 and self._fallback_clear(channel, region)

    def _measure_discovery_point(
        self,
        point: tuple[float, float],
        unresolved: set[int],
        positive_counts: dict[int, int],
        target_directions: int,
    ) -> None:
        if target_directions > 1:
            active = set(unresolved)
            for channel in self.discovered - self.cleared:
                if positive_counts.get(channel, 0) >= target_directions:
                    continue
                skip_impossible = (
                    self.cfg.problem == 4 and self.cfg.q4_skip_impossible_rechecks
                ) or (
                    self.cfg.problem == 3 and self.cfg.q3_skip_impossible_rechecks
                )
                if skip_impossible:
                    region, _, _ = self._region_for(channel)
                    if (
                        len(region)
                        and point_polygon_distance(point, region)
                        > self.cfg.max_receive_radius + 1.0
                    ):
                        self.skipped_rechecks += 1
                        if self.cfg.problem == 3:
                            self.inferred_no_signal.setdefault(channel, set()).add(tuple(map(float, point)))
                        continue
                active.add(channel)
            channels = self._channel_order(active)
        else:
            channels = self._channel_order(unresolved)

        for channel in channels:
            response = self.measure(point, channel)
            result = response.get("measure_result")
            if result == "near":
                if self.clear(point, channel).get("clear_result") == "success":
                    self.discovered.add(channel)
                    unresolved.discard(channel)
                    positive_counts[channel] = target_directions
            elif result == "direction":
                self.discovered.add(channel)
                unresolved.discard(channel)
                positive_counts[channel] = positive_counts.get(channel, 0) + 1
                if self.cfg.adaptive_clear_stop and positive_counts[channel] >= 2:
                    region, _, circle = self._region_for(channel)
                    if (
                        len(region)
                        and circle is not None
                        and circle.radius <= self.cfg.clear_guarantee_radius
                    ):
                        positive_counts[channel] = target_directions

    def _opportunistic_q4_scan_clear(
        self,
        next_point: tuple[float, float] | None,
        remaining_scan_points: Sequence[tuple[float, float]],
        positive_counts: dict[int, int],
        target_directions: int,
    ) -> None:
        """Insert one high-confidence clear when it adds little scan detour."""
        if self.cfg.problem == 4:
            enabled = self.cfg.q4_scan_estimated_clear
            estimated_radius = self.cfg.q4_estimated_clear_radius
            attempt_limit = self.cfg.q4_estimated_clear_attempts
            detour_limit = self.cfg.q4_scan_clear_max_detour
            measure_after_failure = self.cfg.q4_measure_after_failed_estimate
        else:
            enabled = self.cfg.q3_scan_estimated_clear
            estimated_radius = self.cfg.q3_estimated_clear_radius
            attempt_limit = self.cfg.q3_estimated_clear_attempts
            detour_limit = self.cfg.q3_scan_clear_max_detour
            measure_after_failure = self.cfg.q3_measure_after_failed_estimate
        if not enabled or estimated_radius <= 0.0:
            return

        current = np.asarray(self.position, dtype=float)
        following = np.asarray(next_point, dtype=float) if next_point is not None else None
        candidates: list[tuple[float, int, np.ndarray]] = []
        for channel in sorted(self.discovered - self.cleared):
            observations = self._direction_observations(channel)
            if len(observations) < 2:
                continue
            if self.estimated_clear_attempts.get(channel, 0) >= attempt_limit:
                continue
            if self.estimated_clear_bearing_counts.get(channel) == len(observations):
                continue
            region, _, circle = self._region_for(channel)
            if (
                len(region) == 0
                or circle is None
                or circle.radius > estimated_radius
            ):
                continue
            estimate = self._bearing_center_estimate(channel, region)
            if (
                self.cfg.problem == 4
                and self.cfg.q4_safe_clear_route
                and circle.radius <= self.cfg.clear_guarantee_radius
            ):
                from q4_refinement import safe_clear_point

                safe_point = safe_clear_point(
                    region,
                    current,
                    following,
                    radius=self.cfg.clear_guarantee_radius,
                    deadline=self._planning_deadline(),
                )
                if safe_point is not None:
                    estimate = safe_point
            if self.cfg.problem == 3 and self.cfg.q3_service_region and circle.radius <= self.cfg.clear_guarantee_radius:
                estimate = self._certified_service_point(region, circle, next_point)
            if any(
                _distance(estimate, old) < 1e-6
                for old in self.failed_clear_points.get(channel, [])
            ):
                continue
            detour = float(np.linalg.norm(estimate - current))
            if following is not None:
                detour += float(np.linalg.norm(following - estimate))
                detour -= float(np.linalg.norm(following - current))
            if detour <= detour_limit + EPS:
                candidates.append((detour, channel, estimate))

        if not candidates:
            return
        if self.cfg.problem == 4 and self.cfg.q4_scan_clear_lookahead:
            future_start = (
                np.asarray(remaining_scan_points[-1], dtype=float)
                if remaining_scan_points
                else current
            )
            full_future_length = _route_length(
                [item[2] for item in candidates], future_start
            )
            ranked = []
            for detour, candidate_channel, candidate_estimate in candidates:
                other_centers = [
                    item[2] for item in candidates if item[1] != candidate_channel
                ]
                marginal_future = full_future_length - _route_length(
                    other_centers, future_start
                )
                expected_saving = (
                    self.cfg.q4_scan_clear_success_probability * marginal_future
                    - detour
                )
                ranked.append(
                    (expected_saving, -detour, -candidate_channel, candidate_estimate)
                )
            expected_saving, _, negative_channel, estimate = max(ranked)
            if expected_saving <= EPS:
                return
            channel = -negative_channel
        else:
            _, channel, estimate = min(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
        self.estimated_clear_attempts[channel] = (
            self.estimated_clear_attempts.get(channel, 0) + 1
        )
        self.estimated_clear_bearing_counts[channel] = len(
            self._direction_observations(channel)
        )
        if self.clear(estimate, channel).get("clear_result") == "success":
            positive_counts[channel] = target_directions
            return
        if not measure_after_failure:
            return
        response = self.measure(estimate, channel)
        result = response.get("measure_result")
        if result == "near":
            if self.clear(estimate, channel).get("clear_result") == "success":
                positive_counts[channel] = target_directions
        elif result == "direction":
            positive_counts[channel] = min(
                target_directions,
                positive_counts.get(channel, 0) + 1,
            )
        elif (
            result == "no_signal"
            and self.cfg.problem == 4
            and self.cfg.q4_shadow_bisection
            and self.cfg.q4_scan_shadow_recovery
            and 2.0 * circle.radius < self.cfg.min_receive_radius - 2.0
        ):
            cleared, new_bearing = self._q4_recover_from_shadow(channel, estimate)
            if cleared:
                positive_counts[channel] = target_directions
            elif new_bearing:
                positive_counts[channel] = min(
                    target_directions,
                    len(self._direction_observations(channel)),
                )

    def _integrated_q4_discovery_scan(
        self,
        scan_points: list[tuple[float, float]],
    ) -> None:
        """Insert provable clears without changing the coverage-point order."""
        unresolved = set(range(1, 21))
        positive_counts: dict[int, int] = {}
        target_directions = self.cfg.q4_min_discovery_directions
        remaining = list(enumerate(scan_points))

        while True:
            if len(self.discovered) >= self.cfg.max_source_count:
                unresolved.clear()
                remaining.clear()

            clear_tasks: list[tuple[str, int, tuple[float, float]]] = []
            for channel in sorted(self.discovered - self.cleared):
                region, _, circle = self._region_for(channel)
                if (
                    len(region)
                    and circle is not None
                    and circle.radius <= self.cfg.clear_guarantee_radius
                    and not any(
                        _distance(circle.center, old) < 1e-6
                        for old in self.failed_clear_points.get(channel, [])
                    )
                ):
                    clear_tasks.append(
                        ("clear", channel, tuple(map(float, circle.center)))
                    )

            if not remaining and not clear_tasks:
                break

            chosen_clear: tuple[str, int, tuple[float, float]] | None = None
            if clear_tasks and not remaining:
                chosen_clear = clear_tasks[
                    _joint_route_first(clear_tasks, self.position)
                ]
            elif clear_tasks:
                base_points = [
                    tuple(map(float, point))
                    for _, point in remaining
                ]
                base_path = [tuple(map(float, self.position)), *base_points]
                eligible: list[
                    tuple[float, float, tuple[str, int, tuple[float, float]]]
                ] = []
                for task in clear_tasks:
                    center = task[2]
                    insertion_costs = []
                    for edge_index in range(len(base_path) - 1):
                        left = base_path[edge_index]
                        right = base_path[edge_index + 1]
                        insertion_costs.append(
                            _distance(left, center)
                            + _distance(center, right)
                            - _distance(left, right)
                        )
                    insertion_costs.append(_distance(base_path[-1], center))
                    best_edge = min(
                        range(len(insertion_costs)),
                        key=lambda index: (insertion_costs[index], index),
                    )
                    if best_edge == 0:
                        eligible.append(
                            (
                                insertion_costs[0],
                                _distance(self.position, center),
                                task,
                            )
                        )
                if eligible:
                    chosen_clear = min(
                        eligible,
                        key=lambda item: (item[0], item[1], item[2][1]),
                    )[2]

            if chosen_clear is not None:
                _, identifier, point = chosen_clear
                self.clear(point, identifier)
                continue

            _, point = remaining.pop(0)
            self._measure_discovery_point(
                point,
                unresolved,
                positive_counts,
                target_directions,
            )

        self.absent = set(range(1, 21)) - self.discovered

    def _plan_q4_outer_route(
        self,
        outer_points: Sequence[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Choose the outer-ring gap using the current localization backlog."""
        if len(outer_points) < 3:
            return [tuple(map(float, point)) for point in outer_points]
        ring = sorted(
            (tuple(map(float, point)) for point in outer_points),
            key=lambda point: math.atan2(point[1], point[0]),
        )
        centers = []
        for channel in sorted(self.discovered - self.cleared):
            region, _, circle = self._region_for(channel)
            if len(region) == 0 or circle is None:
                continue
            if (
                self.cfg.q4_estimated_clear_radius > 0.0
                and circle.radius <= self.cfg.q4_estimated_clear_radius
                and len(self._direction_observations(channel)) >= 2
            ):
                centers.append(self._bearing_center_estimate(channel, region))
            else:
                centers.append(circle.center)

        candidates = []
        for start_index in range(len(ring)):
            for step in (-1, 1):
                order = [
                    ring[(start_index + step * offset) % len(ring)]
                    for offset in range(len(ring))
                ]
                scan_length = _distance(self.position, order[0]) + sum(
                    _distance(order[index - 1], order[index])
                    for index in range(1, len(order))
                )
                future_length = _route_length(centers, order[-1])
                candidates.append((scan_length + future_length, scan_length, order))
        return min(candidates, key=lambda item: (item[0], item[1], item[2]))[2]

    def _discovery_scan(self, scan_points: list[tuple[float, float]]) -> None:
        if self.cfg.problem == 4 and self.cfg.q4_integrated_scan_clear:
            self._integrated_q4_discovery_scan(scan_points)
            return

        unresolved = set(range(1, 21))
        positive_counts: dict[int, int] = {}
        target_directions = (
            self.cfg.q4_min_discovery_directions
            if self.cfg.problem == 4
            else self.cfg.q3_min_discovery_directions
        )
        visited_count = 0
        for point_index, point in enumerate(scan_points):
            if len(self.discovered) >= self.cfg.max_source_count:
                unresolved.clear()
            self._measure_discovery_point(
                point,
                unresolved,
                positive_counts,
                target_directions,
            )
            next_point = (
                scan_points[point_index + 1]
                if point_index + 1 < len(scan_points)
                else None
            )
            if self.cfg.problem in {3, 4}:
                self._opportunistic_q4_scan_clear(
                    next_point,
                    scan_points[point_index + 1 :],
                    positive_counts,
                    target_directions,
                )
            visited_count += 1
            if len(self.discovered) >= self.cfg.max_source_count:
                break
            if (
                self.cfg.problem == 4
                and self.cfg.q4_adaptive_outer_route
                and self.cfg.q4_layout in {"compact", "certified21", "compact25"}
                and len(scan_points) >= 13
                and visited_count == len(scan_points) - 12
            ):
                scan_points[point_index + 1 :] = self._plan_q4_outer_route(
                    scan_points[point_index + 1 :]
                )
        self.absent = set(range(1, 21)) - self.discovered
        if self.cfg.problem == 4:
            sparse_channels = {
                channel
                for channel in self.discovered - self.cleared
                if 0 < positive_counts.get(channel, 0) < target_directions
            }
            if (
                self.cfg.q4_local_positive_cap
                and visited_count == len(scan_points)
            ):
                self.local_positive_cap_channels.update(sparse_channels)
            if (
                self.cfg.q4_local_triangle_support
                and self.cfg.q4_layout == "compact25"
            ):
                triangles = q4_compact_cover_triangles(
                    self.cfg.target_radius,
                    self.cfg.min_receive_radius,
                )
                for channel in sparse_channels:
                    region, _, _ = self._region_for(channel)
                    channel_observations = [
                        observation
                        for observation in self.obs
                        if observation.channel == channel
                    ]
                    allowed_components = []
                    for triangle in triangles:
                        triangle = convex_hull(triangle)
                        component = intersect_convex_polygons(region, triangle)
                        if len(component) == 0:
                            continue
                        vertex_results = []
                        for vertex in triangle:
                            match = next(
                                (
                                    observation.result
                                    for observation in channel_observations
                                    if _distance(observation.position, vertex) < 1e-5
                                ),
                                None,
                            )
                            vertex_results.append(match)
                        if all(result == "no_signal" for result in vertex_results):
                            continue
                        allowed_components.append(component)
                    if allowed_components:
                        self.discovery_support_polygons[channel] = convex_hull(
                            np.vstack(allowed_components)
                        )

    def _q3_time_order(self, original: list[int], centers: dict[int, np.ndarray]) -> list[int]:
        """One-step completion-time proxy using the actual next action point.

        Measures/switches/failed-estimate recovery and travel are counted in
        seconds. Future centers remain a proxy, not a probabilistic forecast or
        worst-case time certificate. Replan after each completed source.
        """
        if len(original) < 2 or self._low_budget():
            return original
        nearest = sorted(centers, key=lambda ch: (_distance(self.position, centers[ch]), ch))
        choices = list(dict.fromkeys([original[0]] + nearest))[:max(1, self.cfg.q3_joint_candidate_limit)]
        ranked = []
        for channel in choices:
            if self._low_budget():
                return original
            region, _, circle = self._region_for(channel)
            if not len(region) or circle is None:
                continue
            tail = [ch for ch in original if ch != channel]
            following = centers.get(tail[0]) if tail else None
            if circle.radius <= self.cfg.clear_guarantee_radius:
                point = (
                    self._certified_service_point(region, circle, following)
                    if self.cfg.q3_service_region else circle.center
                )
                finish = point
                action_cost = 5.0
            elif (
                circle.radius <= self.cfg.q3_estimated_clear_radius
                and len(self._direction_observations(channel)) >= 2
                and self.estimated_clear_attempts.get(channel, 0) < self.cfg.q3_estimated_clear_attempts
                and self.estimated_clear_bearing_counts.get(channel) != len(self._direction_observations(channel))
            ):
                point = self._bearing_center_estimate(channel, region)
                finish = circle.center
                # Explicit failed-trial + remeasurement + eventual clear proxy;
                # no uncalibrated success probability is assumed.
                action_cost = 3.0 + 5.0 + float(self.channel != channel) + 5.0
            else:
                candidates = self._view_candidates(channel, region)
                if not candidates:
                    continue
                point, finish = np.asarray(candidates[0]), circle.center
                action_cost = 5.0 + float(self.channel != channel) + 5.0
            length = _distance(self.position, point) + _distance(point, finish)
            previous = finish
            for other in tail:
                if other in centers:
                    length += _distance(previous, centers[other])
                    previous = centers[other]
            ranked.append((length / 5.0 + action_cost, original.index(channel), channel, tail))
        if not ranked:
            return original
        _, _, first, tail = min(ranked, key=lambda item: item[:3])
        self.joint_order_changes += int(first != original[0])
        return [first] + tail

    def _localization_order(self) -> list[int]:
        remaining = self.discovered - self.cleared
        centers: dict[int, np.ndarray] = {}
        for channel in remaining:
            region, _, circle = self._region_for(channel)
            if len(region) and circle is not None:
                if (
                    self.cfg.problem == 4
                    and self.cfg.q4_estimate_route_centers
                    and self.cfg.q4_estimated_clear_radius > 0.0
                    and circle.radius <= self.cfg.q4_estimated_clear_radius
                    and len(self._direction_observations(channel)) >= 2
                ):
                    centers[channel] = self._bearing_center_estimate(channel, region)
                else:
                    centers[channel] = circle.center
        if not centers:
            return sorted(remaining)
        if self._low_budget():
            return sorted(remaining, key=lambda ch: (_distance(self.position, centers[ch]) if ch in centers else math.inf, ch))

        if (
            self.cfg.problem == 4
            and self.cfg.q4_multistart_localization_route
        ) or (
            self.cfg.problem == 3
            and self.cfg.q3_multistart_localization_route
        ):
            order = _multistart_open_order(centers, self.position)
            order.extend(sorted(remaining - set(centers)))
            return self._q3_time_order(order, centers) if self.cfg.problem == 3 and self.cfg.q3_joint_time_routing else order

        position = np.asarray(self.position, dtype=float)

        unvisited = set(centers)
        order: list[int] = []
        while unvisited:
            chosen = min(
                unvisited,
                key=lambda channel: (np.linalg.norm(centers[channel] - position), channel),
            )
            order.append(chosen)
            position = centers[chosen]
            unvisited.remove(chosen)

        improved = True
        passes = 0
        start = np.asarray(self.position, dtype=float)
        while improved and passes < 12:
            improved = False
            passes += 1
            for i in range(0, len(order) - 1):
                for j in range(i + 1, len(order)):
                    left = start if i == 0 else centers[order[i - 1]]
                    before = float(np.linalg.norm(left - centers[order[i]]))
                    after = float(np.linalg.norm(left - centers[order[j]]))
                    if j + 1 < len(order):
                        before += float(np.linalg.norm(centers[order[j]] - centers[order[j + 1]]))
                        after += float(np.linalg.norm(centers[order[i]] - centers[order[j + 1]]))
                    if after + 1e-8 < before:
                        order[i:j + 1] = reversed(order[i:j + 1])
                        improved = True

        order.extend(sorted(remaining - set(centers)))
        return self._q3_time_order(order, centers) if self.cfg.problem == 3 and self.cfg.q3_joint_time_routing else order

    def run(self):
        self.start_wall_time = time.monotonic()
        scan_points: list[tuple[float, float]] = []
        failure: str | None = None
        exit_failure: str | None = None
        enter_attempted = False
        exit_succeeded = False
        initial_scan_points: list[tuple[float, float]] = []
        try:
            if self.cfg.max_actions < 1:
                raise ValueError("max_actions must be positive")
            if not (
                0.0
                < self.cfg.clear_guarantee_radius
                <= self.cfg.clear_radius
                <= 20.0
            ):
                raise ValueError(
                    "clear radii must satisfy 0 < guarantee <= clear <= 20"
                )
            if (
                not math.isfinite(self.cfg.real_time_reserve_s)
                or self.cfg.real_time_reserve_s < 0
            ):
                raise ValueError("real_time_reserve_s must be finite and nonnegative")
            scan_points = self._scan_points()
            initial_scan_points = list(scan_points)
            enter_attempted = True
            enter_started = time.monotonic()
            enter_response = self._post("/enter")
            if "remaining_real_duration_s" in enter_response:
                self.remaining_real_duration_s = float(enter_response["remaining_real_duration_s"])
                if (
                    not math.isfinite(self.remaining_real_duration_s)
                    or self.remaining_real_duration_s <= 0
                ):
                    raise ValueError("invalid remaining_real_duration_s")
                self.real_deadline = enter_started + self.remaining_real_duration_s
                self.exit_reserve_s = (
                    max(0.0, self.cfg.exit_reserve_s)
                    if self.cfg.problem == 3
                    else min(
                        self.cfg.real_time_reserve_s,
                        self.remaining_real_duration_s / 5.0,
                    )
                )
                if hasattr(self.client, "set_deadline"):
                    self.client.set_deadline(
                        self.real_deadline if self.cfg.budget_aware else None,
                        self.exit_reserve_s,
                    )
            self._check_budget()
            self._discovery_scan(scan_points)
            self.discovery_complete = True
            while self.discovered - self.cleared:
                self._check_budget()
                order = self._localization_order()
                channel = order[0]
                self.next_task_hint = None
                if (
                    self.cfg.problem == 4
                    and self.cfg.q4_safe_clear_route
                    and len(order) > 1
                ):
                    _, _, following = self._region_for(order[1])
                    if following is not None:
                        self.next_task_hint = following.center.copy()
                self._q3_following_hint = None
                if self.cfg.problem == 3 and self.cfg.q3_service_region and len(order) > 1:
                    _, _, next_circle = self._region_for(order[1])
                    if next_circle is not None:
                        self._q3_following_hint = next_circle.center.copy()
                if not self._localize_and_clear(channel):
                    failure = f"failed to clear discovered channel {channel}"
                    break
        except Exception as error:
            failure = repr(error)
        finally:
            if enter_attempted:
                try:
                    self._post("/exit")
                    exit_succeeded = True
                except Exception as error:
                    exit_failure = repr(error)
                    if failure is None:
                        failure = f"exit failed: {error!r}"

        wall_runtime = time.monotonic() - self.start_wall_time
        summary = {
            "problem": self.cfg.problem,
            "coverage_method": (
                (
                    "7-point disk cover"
                    if self.cfg.q3_layout == "legacy"
                    else (
                        f"{self.cfg.q3_ring_sides}-point minimax ring cover"
                        if self.cfg.q3_layout == "minimax"
                        else f"{self.cfg.q3_ring_sides + 1}-point optimized ring cover"
                    )
                )
                if self.cfg.problem == 3
                else (
                    "21-point continuously certified positive-spanning cover"
                    if self.cfg.q4_layout in {"compact", "certified21"}
                    else (
                        "25-point compact positive-spanning triangulation"
                        if self.cfg.q4_layout == "compact25"
                        else "980 m triangular positive-spanning lattice"
                    )
                )
            ),
            "scan_point_count": len(scan_points),
            "q3_policy": (
                {
                    "layout": self.cfg.q3_layout,
                    "ring_sides": self.cfg.q3_ring_sides,
                    "discovery_directions": self.cfg.q3_min_discovery_directions,
                    "entry_candidates": self.cfg.q3_entry_candidates,
                    "information_floor_fraction": self.cfg.q3_information_floor_fraction,
                    "range_order_constraints": self.cfg.q3_range_order_constraints,
                    "estimated_clear_radius": self.cfg.q3_estimated_clear_radius,
                    "estimated_clear_attempts": self.cfg.q3_estimated_clear_attempts,
                    "scan_estimated_clear": self.cfg.q3_scan_estimated_clear,
                    "scan_clear_max_detour": self.cfg.q3_scan_clear_max_detour,
                    "multistart_localization_route": self.cfg.q3_multistart_localization_route,
                    "skip_impossible_rechecks": self.cfg.q3_skip_impossible_rechecks,
                    "service_region": self.cfg.q3_service_region,
                    "joint_time_routing": self.cfg.q3_joint_time_routing,
                }
                if self.cfg.problem == 3
                else None
            ),
            "q4_policy": (
                {
                    "layout": self.cfg.q4_layout,
                    "discovery_directions": self.cfg.q4_min_discovery_directions,
                    "symmetric_recovery": self.cfg.q4_symmetric_recovery,
                    "information_floor_fraction": self.cfg.q4_information_floor_fraction,
                    "integrated_scan_clear": self.cfg.q4_integrated_scan_clear,
                    "positive_hull_views": self.cfg.q4_positive_hull_views,
                    "estimated_clear_radius": self.cfg.q4_estimated_clear_radius,
                    "estimated_clear_attempts": self.cfg.q4_estimated_clear_attempts,
                    "estimator": self.cfg.q4_estimator,
                    "shadow_bisection": self.cfg.q4_shadow_bisection,
                    "scan_estimated_clear": self.cfg.q4_scan_estimated_clear,
                    "scan_clear_max_detour": self.cfg.q4_scan_clear_max_detour,
                    "scan_shadow_recovery": self.cfg.q4_scan_shadow_recovery,
                    "scan_clear_lookahead": self.cfg.q4_scan_clear_lookahead,
                    "adaptive_outer_route": self.cfg.q4_adaptive_outer_route,
                    "estimate_route_centers": self.cfg.q4_estimate_route_centers,
                    "multistart_localization_route": self.cfg.q4_multistart_localization_route,
                    "one_bearing_posterior": self.cfg.q4_one_bearing_posterior,
                    "one_bearing_initial_probe": self.cfg.q4_one_bearing_initial_probe,
                    "adaptive_clear_stop": self.cfg.adaptive_clear_stop,
                    "joint_constraints": self.cfg.q4_joint_constraints,
                    "time_lookahead": self.cfg.q4_time_lookahead,
                    "safe_clear_route": self.cfg.q4_safe_clear_route,
                }
                if self.cfg.problem == 4
                else None
            ),
            "discovered_channels": sorted(self.discovered),
            "absent_channels": sorted(self.absent),
            "cleared_channels": sorted(self.cleared),
            "cleared_count": len(self.cleared),
            "actions": self.action_count,
            "virtual_time_s": self.last_virtual_time,
            "program_runtime_s": wall_runtime,
            "failure": failure,
            "exit_succeeded": exit_succeeded,
            "exit_failure": exit_failure,
            "remaining_real_duration_s": self.remaining_real_duration_s,
            "config": asdict(self.cfg),
            "initial_scan_points": initial_scan_points,
            "config_sha256": hashlib.sha256(
                json.dumps(asdict(self.cfg), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "scan_points_sha256": hashlib.sha256(
                json.dumps(initial_scan_points, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256(
                    (Path(__file__).parent / name).read_bytes()
                ).hexdigest()
                for name in (
                    "robot.py",
                    "geometry.py",
                    "offline_sim.py",
                    "q4_refinement.py",
                )
                if (Path(__file__).parent / name).is_file()
            },
            "refinement_stats": dict(self.refinement_stats),
            "completion_certified_by_policy": self.discovery_complete and not (self.discovered - self.cleared) and failure is None,
            "discovery_complete": self.discovery_complete,
            "remaining_real_duration_s_at_entry": self.remaining_real_duration_s,
            "budget_degraded": self.budget_degraded,
            "skipped_rechecks": self.skipped_rechecks,
            "inferred_no_signal": {str(ch): sorted(points) for ch, points in self.inferred_no_signal.items()},
            "region_cache_hits": self.region_cache_hits,
            "region_cache_misses": self.region_cache_misses,
            "service_point_uses": self.service_point_uses,
            "joint_order_changes": self.joint_order_changes,
            "cost_breakdown": {
                "distance_m": self.distance_m,
                "movement_s": self.distance_m / 5.0,
                "measurement_s": 5.0 * self.measure_count,
                "switch_s": self.switch_count,
                "clear_s": 5.0 * self.clear_success_count + 3.0 * self.clear_failure_count,
                "measure_count": self.measure_count,
                "clear_failures": self.clear_failure_count,
            },
            "full_config": asdict(self.cfg),
        }
        (self.result_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


class _OfflineClient:
    def __init__(self, world, log_path: str | Path):
        self.world = world
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def post(self, path: str, body: dict[str, Any]):
        self.counter += 1
        request_body = dict(body)
        request_id = request_body.setdefault("request_id", f"{path.strip('/')}-{self.counter}")
        if path == "/enter":
            response = self.world.enter(request_id)
        elif path == "/measure":
            response = self.world.measure(request_body["position"], request_body["channel"], request_id)
        elif path == "/clear":
            response = self.world.clear(request_body["position"], request_body["channel"], request_id)
        elif path == "/exit":
            response = self.world.exit(request_id)
        else:
            raise ValueError(path)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"path": path, "request": request_body, "response": response}, ensure_ascii=False) + "\n")
        return response


def run_offline(
    seed: int,
    problem: int,
    out_dir: str | Path,
    world_options: dict[str, Any] | None = None,
    robot_options: dict[str, Any] | None = None,
):
    from offline_sim import JammerWorld

    world = JammerWorld.random_case(seed, problem, **(world_options or {}))
    client = _OfflineClient(world, Path(out_dir) / "events.jsonl")
    robot = Robot(client, RobotConfig(problem=problem, **(robot_options or {})), out_dir)
    result = robot.run()
    result["total_sources"] = len(world.sources)
    result["source_directional"] = sum(source.directional for source in world.sources)
    simulator_cleared_count = world.cleared_count
    result["simulator_cleared_count"] = simulator_cleared_count
    result["coverage"] = simulator_cleared_count / len(world.sources)
    result["average_localization_clear_time_s"] = world.virtual_time / max(1, simulator_cleared_count)
    if simulator_cleared_count != result["cleared_count"] and not result["failure"]:
        result["failure"] = (
            "robot/world cleared-state mismatch: "
            f"robot={result['cleared_count']}, simulator={simulator_cleared_count}"
        )
    world.save(Path(out_dir) / "world.json")
    (Path(out_dir) / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int, default=3, choices=[3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/offline")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument(
        "--q3-layout",
        choices=["legacy", "ring", "minimax"],
        default=RobotConfig.q3_layout,
    )
    parser.add_argument("--q3-ring-sides", type=int, default=7)
    parser.add_argument("--q3-directions", type=int, default=2)
    parser.add_argument("--q3-view-travel-weight", type=float, default=0.2)
    parser.add_argument(
        "--q3-entry-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--q3-information-floor-fraction", type=float, default=0.1)
    parser.add_argument("--q3-skip-impossible-rechecks", action=argparse.BooleanOptionalAction, default=RobotConfig.q3_skip_impossible_rechecks)
    parser.add_argument("--q3-service-region", action=argparse.BooleanOptionalAction, default=RobotConfig.q3_service_region)
    parser.add_argument("--q3-joint-time-routing", action=argparse.BooleanOptionalAction, default=RobotConfig.q3_joint_time_routing)
    parser.add_argument("--budget-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--q3-range-order-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use valid positive/no-signal distance-order constraints for Q3",
    )
    parser.add_argument(
        "--q4-layout",
        choices=["compact", "certified21", "compact25", "legacy"],
        default="compact",
    )
    parser.add_argument(
        "--q4-profile",
        choices=["default", "baseline", "safe", "joint", "joint-safe", "combined"],
        default="default",
        help="ablatable Q4 refinement profile; baseline keeps reliability fixes",
    )
    parser.add_argument(
        "--adaptive-clear-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop adaptive localization immediately when a guaranteed clear is possible",
    )
    parser.add_argument(
        "--robot-id",
        default=None,
        help="current simulator login team number; required for --online and intentionally not embedded in source",
    )
    args = parser.parse_args()
    profile_options = {} if args.q4_profile == "default" else {
        "q4_joint_constraints": args.q4_profile in {"joint", "joint-safe", "combined"},
        "q4_safe_clear_route": args.q4_profile in {"safe", "joint-safe", "combined"},
        "q4_time_lookahead": args.q4_profile == "combined",
    }

    if args.online:
        if not args.robot_id:
            parser.error("--robot-id is required when --online is selected")
        client = HttpClient(args.base_url, args.robot_id, Path(args.out) / "events.jsonl")
        result = Robot(
            client,
            RobotConfig(
                problem=args.problem,
                q3_layout=args.q3_layout,
                q3_ring_sides=args.q3_ring_sides,
                q3_min_discovery_directions=args.q3_directions,
                q3_view_travel_weight=args.q3_view_travel_weight,
                q3_entry_candidates=args.q3_entry_candidates,
                q3_information_floor_fraction=args.q3_information_floor_fraction,
                q3_range_order_constraints=args.q3_range_order_constraints,
                q3_skip_impossible_rechecks=args.q3_skip_impossible_rechecks,
                q3_service_region=args.q3_service_region,
                q3_joint_time_routing=args.q3_joint_time_routing,
                budget_aware=args.budget_aware,
                q4_layout=args.q4_layout,
                adaptive_clear_stop=args.adaptive_clear_stop,
                **profile_options,
            ),
            args.out,
        ).run()
    else:
        result = run_offline(
            args.seed,
            args.problem,
            args.out,
            robot_options={
                "q3_layout": args.q3_layout,
                "q3_ring_sides": args.q3_ring_sides,
                "q3_min_discovery_directions": args.q3_directions,
                "q3_view_travel_weight": args.q3_view_travel_weight,
                "q3_entry_candidates": args.q3_entry_candidates,
                "q3_information_floor_fraction": args.q3_information_floor_fraction,
                "q3_range_order_constraints": args.q3_range_order_constraints,
                "q3_skip_impossible_rechecks": args.q3_skip_impossible_rechecks,
                "q3_service_region": args.q3_service_region,
                "q3_joint_time_routing": args.q3_joint_time_routing,
                "budget_aware": args.budget_aware,
                "q4_layout": args.q4_layout,
                "adaptive_clear_stop": args.adaptive_clear_stop,
                **profile_options,
            },
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
