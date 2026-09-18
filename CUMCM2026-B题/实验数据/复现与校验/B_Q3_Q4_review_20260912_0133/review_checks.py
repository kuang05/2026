"""Read-only audit of the current B Q3/Q4 solver; results go to this folder only.

Does not contact the official simulator, consume test attempts, change source code,
or read hidden official case truth. Existing JSON event logs are assessed as evidence.
Run with an isolated Python environment containing NumPy:
  python -B review_checks.py --part geometry
  python -B review_checks.py --part benchmarks
  python -B review_checks.py --part logs
"""
from __future__ import annotations
import argparse
import dataclasses
import datetime as dt
import hashlib
import http.client
import itertools
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "CUMCM2026Problems" / "B题" / "solve"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
import numpy as np
import geometry as G
import robot as R
from offline_sim import JammerWorld


def save(name, obj):
    path = OUT / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def manifest():
    names = ["robot.py", "geometry.py", "offline_sim.py", "test_solution.py", "benchmark_q3.py", "benchmark_q4.py", "run_experiments.py"]
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "source_root": str(SOURCE),
        "config": dataclasses.asdict(R.RobotConfig()),
        "source_files": [{"name": n, "sha256": hashlib.sha256((SOURCE / n).read_bytes()).hexdigest(), "size": (SOURCE / n).stat().st_size} for n in names],
    }


def oracle_circle(points):
    """Independent exhaustive support-set oracle, only for small test sets."""
    pts = np.asarray(points, float)
    best = None
    candidates = [(p, 0.0) for p in pts]
    for p, q in itertools.combinations(pts, 2):
        c = (p + q) / 2
        candidates.append((c, float(np.linalg.norm(p - q) / 2)))
    for p, q, r in itertools.combinations(pts, 3):
        a = np.vstack([q - p, r - p]) * 2
        if abs(np.linalg.det(a)) < 1e-18:
            continue
        b = np.array([np.dot(q - p, q - p), np.dot(r - p, r - p)])
        c = p + np.linalg.solve(a, b)
        candidates.append((c, float(np.linalg.norm(c - p))))
    for c, radius in candidates:
        if np.max(np.linalg.norm(pts - c, axis=1)) <= radius + 1e-7:
            if best is None or radius < best[1]:
                best = (c, radius)
    if best is None:
        raise AssertionError("No oracle support circle")
    return best


def geometry_checks():
    results = {"manifest": manifest()}
    rng = np.random.default_rng(9122026)
    max_radius_delta = 0.0
    max_violation = 0.0
    failures = []
    for k in range(260):
        n = int(rng.integers(3, 11))
        if k < 200:
            points = rng.normal(size=(n, 2)) * rng.uniform(1, 1000)
        else:
            points = np.column_stack([rng.uniform(-500, 500, n), rng.normal(size=n) * 1e-7]) + np.array([1500, 700])
        if k % 11 == 0:
            points = np.vstack([points, points[0]])
        current = G.min_enclosing_circle(points)
        _, expected = oracle_circle(points)
        delta = abs(current.radius - expected)
        violation = float(np.max(np.linalg.norm(points - current.center, axis=1)) - current.radius)
        max_radius_delta = max(max_radius_delta, delta)
        max_violation = max(max_violation, violation)
        if delta > 1e-5 or violation > 1e-7:
            failures.append({"case": k, "radius_delta": delta, "violation": violation})
    results["enclosing_circle_oracle"] = {"cases": 260, "failures": failures, "max_radius_difference_m": max_radius_delta, "max_containment_violation_m": max_violation}

    containment_failures = []
    radius_differences = []
    certified_clears = 0
    max_certified_error = 0.0
    for k in range(70):
        angle = float(rng.uniform(0, 2 * math.pi))
        radius = 1800.0 if k % 3 == 0 else float(rng.uniform(0, 1800))
        source = radius * G.unit(angle)
        count = (1, 2, 3, 5)[k % 4]
        readings, disks = [], []
        for j in range(count):
            direction = (0.00001 * j if k % 7 == 0 else float(rng.uniform(0, 2 * math.pi)))
            distance = 1500.0 if j == 0 else float(rng.uniform(40, 1450))
            observer = source - distance * G.unit(direction)
            error = (-1.0, 1.0, -0.995, 0.995, 0.0)[(k + j) % 5]
            reading = math.radians(round((math.degrees(direction) + error) % 360, 2))
            readings.append((observer, reading))
            disks.append((observer, 1500.0))
        circles = []
        for n_disk in (360, 720, 1440):
            region = G.bearing_region(readings, 1800, 1.01, disks, n_disk)
            ok = len(region) > 0 and G.point_in_convex_polygon(source, region, tol=1e-5)
            if not ok:
                containment_failures.append({"case": k, "n_disk": n_disk})
            c = G.min_enclosing_circle(region)
            circles.append(c.radius)
            if c.radius <= 19.5:
                certified_clears += 1
                max_certified_error = max(max_certified_error, float(np.linalg.norm(c.center - source)))
        if all(math.isfinite(x) for x in circles):
            radius_differences.append(abs(circles[0] - circles[-1]))
    results["bounded_regions"] = {"base_cases": 70, "disk_resolutions": [360, 720, 1440], "evaluations": 210, "source_containment_failures": containment_failures, "certified_clear_evaluations": certified_clears, "max_certified_center_error_m": max_certified_error, "max_radius_difference_360_vs_1440_m": max(radius_differences), "disk_outer_radial_error_1800_at_360_m": 1800 * (1 / math.cos(math.pi / 360) - 1), "note": "Finite tests, not a proof of a global post-intersection error bound."}

    points3 = G.q3_ring_cover_points(sides=7)
    rho = float(np.linalg.norm(points3[1]))
    alpha = math.pi / 7
    boundary_distance = math.sqrt(1800**2 + rho**2 - 2 * 1800 * rho * math.cos(alpha))
    transition = rho / (2 * math.cos(alpha))
    points4 = G.q4_compact_cover_points()
    ro = float(np.linalg.norm(points4[-1]))
    bridge = math.sqrt(930**2 + ro**2 - 2 * 930 * ro * math.cos(math.pi / 12))
    edges = [930.0, 2 * 930 * math.sin(math.pi / 12), 2 * ro * math.sin(math.pi / 12), bridge]
    results["coverage_geometry"] = {"q3_points": len(points3), "q3_ring_radius_m": rho, "q3_continuous_worst_nearest_m": max(transition, boundary_distance), "q3_min_receive_margin_m": 1000 - max(transition, boundary_distance), "q4_points": len(points4), "q4_outer_radius_m": ro, "q4_longest_triangulation_edge_m": max(edges), "q4_margin_m": 1000 - max(edges), "q3_count_from_current_experiment_reporting_expression": len(G.q3_ring_cover_points())}

    robot = R.Robot(None, R.RobotConfig(problem=4), OUT / "fallback_check")
    robot.obs = [R.Observation((0.0, 0.0), 1, "direction", 0.0, 0.0)]
    region, _, _ = robot._region_for(1)
    cover = np.asarray(robot._clear_cover_points(region))
    worst = 0.0
    count = 0
    for radius in np.linspace(0, 1500, 301):
        for angle in np.linspace(-1.005, 1.005, 31):
            source = radius * G.unit(math.radians(angle))
            worst = max(worst, float(np.min(np.linalg.norm(cover - source, axis=1))))
            count += 1
    results["one_bearing_fallback"] = {"grid_points": len(cover), "sampled_true_locations": count, "max_nearest_clear_point_distance_m": worst, "pass_under_20_m": worst <= 20, "note": "One representative long thin uncertainty set; not a full worst-case resource proof."}

    # Controlled network fault: a response body truncated after the server acted.
    attempts = []
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            raise http.client.IncompleteRead(b'{"accepted":', 10)
    def urlopen(request, timeout):
        attempts.append(request.data.decode("utf-8"))
        return Response()
    client = R.HttpClient("http://127.0.0.1:2026", "audit-stub", OUT / "incomplete_read_probe.jsonl")
    exception = None
    with patch("urllib.request.urlopen", side_effect=urlopen):
        try:
            client.post("/enter", {}, retries=4)
        except Exception as e:
            exception = type(e).__name__
    results["incomplete_response_probe"] = {"attempts": len(attempts), "exception": exception, "configured_retries": 4, "interpretation": "An IncompleteRead is not retried by the current exception handler; no real network request was sent."}

    # Controlled deadline fault: time elapses after an accepted /enter.
    deadline_actions = []
    clock = {"now": 0.0}
    class DeadlineClient:
        def post(self, path, body):
            deadline_actions.append({"path": path, "clock_s": clock["now"]})
            if path == "/enter":
                return {"accepted": True, "virtual_time_s": 0, "remaining_real_duration_s": 0.1}
            return {"accepted": True, "virtual_time_s": 5, "measure_result": "no_signal", "exit_reason": "user_exit"}
    class DeadlineRobot(R.Robot):
        def _scan_points(self):
            clock["now"] = 2.0
            return [(0.0, 0.0)]
        def _discovery_scan(self, _):
            self.measure((0.0, 0.0), 1)
    with patch("robot.time.monotonic", side_effect=lambda: clock["now"]):
        bot = DeadlineRobot(DeadlineClient(), R.RobotConfig(), OUT / "deadline_probe")
        response = bot.run()
    results["deadline_probe"] = {"remaining_budget_s": 0.1, "reported_elapsed_s": response["program_runtime_s"], "actions": deadline_actions, "deadline_detected": response["failure"] is not None, "interpretation": "Controlled clock/stub probe; the controller stores the budget but does not check it before new actions. The official server enforces its own deadline."}
    save("geometry_validation.json", results)
    print(json.dumps({k: v for k, v in results.items() if k != "manifest"}, ensure_ascii=False, indent=2), flush=True)


class TracedRobot(R.Robot):
    def __init__(self, client, config, result_dir):
        super().__init__(client, config, result_dir)
        self.audit = {"certified_attempts": 0, "certified_failures": 0, "speculative_attempts": 0, "speculative_failures": 0, "near_attempts": 0, "near_failures": 0, "fallback_attempts": 0, "fallback_invocations": 0, "discovery_bearing_counts": {}}
        self.clear_context = None
    def _clear_if_certain(self, channel, circle):
        previous = self.clear_context
        self.clear_context = "certified"
        try:
            return super()._clear_if_certain(channel, circle)
        finally:
            self.clear_context = previous
    def _fallback_clear(self, channel, region):
        self.audit["fallback_invocations"] += 1
        previous = self.clear_context
        self.clear_context = "fallback"
        try:
            return super()._fallback_clear(channel, region)
        finally:
            self.clear_context = previous
    def clear(self, position, channel):
        near = bool(self.obs and self.obs[-1].channel == channel and self.obs[-1].result == "near" and math.dist(self.obs[-1].position, position) < 1e-8)
        context = self.clear_context or ("near" if near else "speculative")
        self.audit[context + "_attempts"] += 1
        response = super().clear(position, channel)
        key = context + "_failures"
        if response.get("clear_result") != "success":
            self.audit[key] = self.audit.get(key, 0) + 1
        return response
    def _discovery_scan(self, scan_points):
        super()._discovery_scan(scan_points)
        self.audit["discovery_bearing_counts"] = {str(c): len(self._direction_observations(c)) for c in sorted(self.discovered)}
        self.audit["discovery_actions"] = self.action_count
        self.audit["discovery_virtual_time_s"] = self.last_virtual_time


def accepted_trace(events):
    position = (0.0, 0.0)
    channel = 1
    distance = 0.0
    measures = switches = successes = failed_clears = near = no_signal = 0
    counted = set()
    coordinates_ok = True
    for event in events:
        path = event["path"]
        request, response = event["request"], event["response"]
        if response.get("accepted") is not True:
            continue
        rid = request.get("request_id")
        if rid in counted:
            continue
        counted.add(rid)
        p = request.get("position")
        if p is not None:
            next_position = (float(p["x"]), float(p["y"]))
            coordinates_ok &= all(math.isfinite(x) and abs(x) <= 2000000 for x in next_position)
            distance += math.dist(position, next_position)
            position = next_position
        if path == "/measure":
            measures += 1
            switches += request["channel"] != channel
            channel = request["channel"]
            near += response.get("measure_result") == "near"
            no_signal += response.get("measure_result") == "no_signal"
        elif path == "/clear":
            successes += response.get("clear_result") == "success"
            failed_clears += response.get("clear_result") != "success"
    total = distance / 5 + 5 * measures + switches + 5 * successes + 3 * failed_clears
    return {"distance_m": distance, "movement_time_s": distance / 5, "measures": measures, "switches": switches, "successful_clears": successes, "failed_clears": failed_clears, "near_measurements": near, "no_signal_measurements": no_signal, "reconstructed_virtual_time_s": total, "coordinates_valid": bool(coordinates_ok)}


def run_one(problem, scenario, seed, policy, options, world_options):
    path = OUT / "fresh_runs" / f"p{problem}_{scenario}_{seed}_{policy}"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite an existing audit run: {path}")
    world = JammerWorld.random_case(seed, problem, **world_options)
    bot = TracedRobot(R._OfflineClient(world, path / "events.jsonl"), R.RobotConfig(problem=problem, **options), path)
    result = bot.run()
    events = world.events
    trace = accepted_trace(events)
    truth = {s.channel: s for s in world.sources}
    clear_distances = []
    for event in events:
        if event["path"] == "/clear" and event["response"].get("clear_result") == "success":
            request = event["request"]
            source = truth[request["channel"]]
            clear_distances.append(math.dist((request["position"]["x"], request["position"]["y"]), source.position))
    result.update({"problem": problem, "scenario": scenario, "seed": seed, "policy": policy, "total_sources": len(world.sources), "directional_sources": sum(s.directional for s in world.sources), "coverage": len(bot.cleared) / len(world.sources), "average_time_s": result["virtual_time_s"] / max(1, result["cleared_count"]), "trace": trace, "time_reconstruction_residual_s": result["virtual_time_s"] - trace["reconstructed_virtual_time_s"], "max_successful_clear_distance_m": max(clear_distances, default=0), "audit": bot.audit})
    save(str(path.relative_to(OUT) / "audit.json"), result)
    world.save(path / "world.json")
    print(f"p{problem} {scenario} seed={seed} {policy}: coverage={result['coverage']:.3f} T={result['virtual_time_s']:.3f} wall={result['program_runtime_s']:.3f}", flush=True)
    return result


def benchmark_checks():
    policies = {
        3: {
            "legacy": {"q3_layout": "legacy", "q3_min_discovery_directions": 1, "q3_view_travel_weight": 2e-4, "q3_entry_candidates": False, "q3_information_floor_fraction": 0.0},
            "current": {},
        },
        4: {
            "legacy": {"q4_layout": "legacy", "q4_min_discovery_directions": 2},
            "current": {},
        },
    }
    rows = []
    for problem in (3, 4):
        for scenario, seeds in (("random", list(range(1001, 1007))), ("stress", [2001, 2002, 2003])):
            options = {} if scenario == "random" else {"count": 16, "receive_radius_range": (1000.0, 1000.0), "boundary_bias": True}
            if scenario == "stress" and problem == 4:
                options["directional_probability"] = 1.0
            for seed in seeds:
                # Alternation reduces a fixed execution-order bias in wall times.
                order = list(policies[problem]) if seed % 2 else list(reversed(policies[problem]))
                for policy in order:
                    rows.append(run_one(problem, scenario, seed, policy, policies[problem][policy], options))
    groups = []
    for problem, scenario in itertools.product((3, 4), ("random", "stress")):
        subset = [r for r in rows if r["problem"] == problem and r["scenario"] == scenario]
        policies_summary = {}
        for policy in ("legacy", "current"):
            current = [r for r in subset if r["policy"] == policy]
            policies_summary[policy] = {
                "runs": len(current), "all_cleared_runs": sum(r["coverage"] == 1 and r["failure"] is None for r in current),
                "coverage_min": min(r["coverage"] for r in current),
                "virtual_time_mean_s": statistics.mean(r["virtual_time_s"] for r in current),
                "virtual_time_max_s": max(r["virtual_time_s"] for r in current),
                "average_time_mean_s": statistics.mean(r["average_time_s"] for r in current),
                "wall_runtime_mean_s": statistics.mean(r["program_runtime_s"] for r in current),
                "wall_runtime_max_s": max(r["program_runtime_s"] for r in current),
                "actions_mean": statistics.mean(r["actions"] for r in current),
                "movement_share_mean": statistics.mean(r["trace"]["movement_time_s"] / r["virtual_time_s"] for r in current),
                "failed_clears": sum(r["trace"]["failed_clears"] for r in current),
                "fallback_invocations": sum(r["audit"]["fallback_invocations"] for r in current),
                "certified_clear_failures": sum(r["audit"]["certified_failures"] for r in current),
                "max_successful_clear_distance_m": max(r["max_successful_clear_distance_m"] for r in current),
                "max_time_reconstruction_residual_s": max(abs(r["time_reconstruction_residual_s"]) for r in current),
            }
        differences = []
        for seed in sorted({r["seed"] for r in subset}):
            pair = {r["policy"]: r for r in subset if r["seed"] == seed}
            differences.append((pair["legacy"]["virtual_time_s"] - pair["current"]["virtual_time_s"]) / pair["legacy"]["virtual_time_s"])
        groups.append({"problem": problem, "scenario": scenario, "policies": policies_summary, "paired_time_reduction_mean": statistics.mean(differences), "paired_time_reduction_min": min(differences), "current_faster_cases": sum(x > 0 for x in differences), "paired_cases": len(differences)})
    # One fixed-case repeat per question verifies deterministic actions, not accuracy.
    repeatability = []
    for problem in (3, 4):
        repeated = run_one(problem, "repeat", 1001, "current", {}, {})
        original = next(r for r in rows if r["problem"] == problem and r["scenario"] == "random" and r["seed"] == 1001 and r["policy"] == "current")
        base = OUT / "fresh_runs"
        a = base / f"p{problem}_random_1001_current" / "events.jsonl"
        b = base / f"p{problem}_repeat_1001_current" / "events.jsonl"
        repeatability.append({"problem": problem, "seed": 1001, "event_logs_byte_identical": a.read_bytes() == b.read_bytes(), "virtual_time_difference_s": repeated["virtual_time_s"] - original["virtual_time_s"]})
    result = {"manifest": manifest(), "status": "fresh_synthetic_audit_not_official_scores", "design": {"random_seeds": list(range(1001, 1007)), "stress_seeds": [2001, 2002, 2003], "paired_runs": 36, "repeat_runs": 2, "warning": "Small independent audit sample; not a confidence guarantee, a global optimum certificate, or proof of the official case distribution."}, "groups": groups, "repeatability": repeatability}
    save("fresh_benchmark_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def log_checks():
    dirs = [SOURCE / "results" / "practice_p3_20260911_231309", SOURCE / "results" / "practice_p4_optimized_20260911_234058"]
    result = {"manifest": manifest(), "files": []}
    for directory in dirs:
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        saved_summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        sessions, session = [], []
        for event in events:
            if event.get("path") == "/enter" and event.get("response", {}).get("accepted") is True:
                # A transport retry with the same ID must not start a new session.
                if session and event["request"].get("request_id") != session[0]["request"].get("request_id"):
                    sessions.append(session)
                    session = []
            session.append(event)
            if event.get("path") == "/exit" and event.get("response", {}).get("accepted") is True:
                sessions.append(session)
                session = []
        if session:
            sessions.append(session)
        summaries = []
        for i, s in enumerate(sessions):
            completed = [e for e in s if "response" in e]
            trace = accepted_trace(completed)
            last = next((e["response"] for e in reversed(completed) if e["response"].get("accepted") is True), {})
            first = next((e["response"] for e in completed if e.get("path") == "/enter" and e["response"].get("accepted") is True), {})
            began = first.get("real_timestamp_ms")
            ended = last.get("real_timestamp_ms")
            stamp = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat() if ms else None
            summaries.append({"session_index": i + 1, "events": len(s), "begin_utc": stamp(began), "end_utc": stamp(ended), "reported_program_duration_s": (ended - began) / 1000 if began and ended else None, "has_successful_exit": any(e.get("path") == "/exit" and e.get("response", {}).get("accepted") is True for e in s), "virtual_time_s": last.get("virtual_time_s"), "matches_saved_summary": abs(float(last.get("virtual_time_s", -1)) - saved_summary["virtual_time_s"]) < 1e-5 and trace["successful_clears"] == saved_summary["cleared_count"], "trace": trace})
        result["files"].append({"directory": str(directory.relative_to(ROOT)), "total_events": len(events), "session_count": len(summaries), "saved_summary": saved_summary, "sessions": summaries})
    save("practice_log_audit.json", result)
    print(json.dumps({"files": result["files"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["geometry", "benchmarks", "logs"], required=True)
    args = parser.parse_args()
    {"geometry": geometry_checks, "benchmarks": benchmark_checks, "logs": log_checks}[args.part]()
