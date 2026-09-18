"""Independent-seed Q3 policy ablations; synthetic evidence, never official scores.

No online clients are used. A frozen pre-change source directory supplies the
original policy. Every run receives a fresh result directory and identical
hidden source data within its matched-case group. Source truth is used only by
the simulator and post-run metrics, never by Robot decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
POLICIES = {
    "original": {},
    # Keep ablation groups on the former center-plus-ring layout.  Only the
    # selected policy may inherit the current default layout, so layout gains
    # are not mislabeled as pruning or service-point gains.
    "cache_only": {"q3_layout": "ring", "q3_skip_impossible_rechecks": False, "q3_service_region": False, "q3_joint_time_routing": False},
    "prune": {"q3_layout": "ring", "q3_skip_impossible_rechecks": True, "q3_service_region": False, "q3_joint_time_routing": False},
    "service": {"q3_layout": "ring", "q3_skip_impossible_rechecks": True, "q3_service_region": True, "q3_joint_time_routing": False},
    "joint": {"q3_layout": "ring", "q3_skip_impossible_rechecks": True, "q3_service_region": True, "q3_joint_time_routing": True},
    "selected": None,
}
SCENARIOS = (
    "random", "n10_boundary", "n13_boundary", "n16_boundary", "n10_uniform",
    "n16_uniform", "plus_one", "minus_one", "smooth_error", "narrow_angles",
)


def load_modules(root: Path):
    for name in ("robot", "geometry", "offline_sim"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    try:
        importlib.invalidate_caches()
        policy = importlib.import_module("robot")
        simulator = importlib.import_module("offline_sim")
    finally:
        sys.path.pop(0)
    return policy, simulator


def error_plus(self, source, position):
    return 1.0


def error_minus(self, source, position):
    return -1.0


def error_smooth(self, source, position):
    return math.sin(position["x"] / 400.0 + position["y"] / 600.0 + source.channel / 4.0)


def run_one(job):
    import types
    source_root, baseline_root, output_root, scenario, seed, label = job
    policy, simulator = load_modules(Path(baseline_root if label == "original" else source_root))
    options = {}
    if scenario != "random":
        count = 10 if scenario.startswith("n10") else (16 if scenario.startswith("n16") else 13)
        options = {"count": count, "receive_radius_range": (1000.0, 1000.0),
                   "boundary_bias": "uniform" not in scenario}
    world = simulator.JammerWorld.random_case(seed, 3, **options)
    if scenario == "plus_one":
        world._fixed_error_deg = types.MethodType(error_plus, world)
    elif scenario == "minus_one":
        world._fixed_error_deg = types.MethodType(error_minus, world)
    elif scenario == "smooth_error":
        world._fixed_error_deg = types.MethodType(error_smooth, world)
    elif scenario == "narrow_angles":
        # Source layout with narrow bearing spread and boundary positions.
        rng = np.random.default_rng(seed)
        for source in world.sources:
            radius = float(rng.uniform(1450.0, 1800.0))
            angle = float(rng.uniform(-0.03, 0.03)) + (math.pi if source.channel % 2 else 0.0)
            source.x, source.y = radius * math.cos(angle), radius * math.sin(angle)
    hidden = [asdict(source) for source in world.sources]
    world_hash = hashlib.sha256(json.dumps(hidden, sort_keys=True).encode()).hexdigest()
    directory = Path(output_root) / "runs" / scenario / f"{label}_{seed}"
    if directory.exists():
        raise FileExistsError(f"Fresh run directory required: {directory}")
    config = policy.RobotConfig(problem=3, **(POLICIES[label] or {}))
    robot = policy.Robot(policy._OfflineClient(world, directory / "events.jsonl"), config, directory)
    started = time.perf_counter()
    result = robot.run()
    runtime = time.perf_counter() - started
    distance = 0.0
    previous = (0.0, 0.0)
    measured = switched = failed = cleared = 0
    channel = 1
    for event in world.events:
        req, response = event["request"], event["response"]
        if "position" in req:
            point = (req["position"]["x"], req["position"]["y"])
            distance += math.dist(previous, point)
            previous = point
        if event["path"] == "/measure":
            measured += 1
            switched += int(channel != req["channel"])
            channel = req["channel"]
        elif event["path"] == "/clear":
            cleared += int(response.get("clear_result") == "success")
            failed += int(response.get("clear_result") != "success")
    reconstructed = distance / 5.0 + 5.0 * measured + switched + 5.0 * cleared + 3.0 * failed
    if abs(reconstructed - world.virtual_time) > 1e-6:
        raise AssertionError("Trace accounting does not match simulator time")
    row = {
        "scenario": scenario, "seed": seed, "policy": label, "world_hash": world_hash,
        "count": len(world.sources), "cleared": world.cleared_count,
        "coverage": world.cleared_count / len(world.sources),
        "time_s": world.virtual_time, "per_source_s": world.virtual_time / max(1, world.cleared_count),
        "distance_m": distance, "measures": measured, "switches": switched, "failed_clears": failed,
        "actions": result["actions"], "runtime_s": runtime, "failure": result["failure"],
        "skipped_rechecks": result.get("skipped_rechecks", 0),
        "service_point_uses": result.get("service_point_uses", 0),
        "joint_order_changes": result.get("joint_order_changes", 0),
        "region_cache_hits": result.get("region_cache_hits", 0),
        "config": asdict(config),
    }
    (directory / "case_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def summary(rows, policies):
    bases = {(r["scenario"], r["seed"]): r for r in rows if r["policy"] == "original"}
    rng = np.random.default_rng(710317)
    overall = {}
    for label in policies:
        values = [r for r in rows if r["policy"] == label]
        gains = np.array([1.0 - r["time_s"] / bases[(r["scenario"], r["seed"])]["time_s"] for r in values])
        if any(r["world_hash"] != bases[(r["scenario"], r["seed"])]["world_hash"] for r in values):
            raise AssertionError("Paired cases have different hidden sources")
        # Stratified paired bootstrap: keep each stratum's weight fixed.
        strata = [np.array([1.0 - r["time_s"] / bases[(r["scenario"], r["seed"])]["time_s"]
                           for r in values if r["scenario"] == name]) for name in SCENARIOS]
        means = np.zeros(10000)
        for group in strata:
            if len(group):
                indices = rng.integers(0, len(group), size=(10000, len(group)))
                means += group[indices].sum(axis=1) / len(values)
        overall[label] = {
            "cases": len(values), "coverage_min": min(r["coverage"] for r in values),
            "failures": sum(bool(r["failure"]) for r in values),
            "paired_mean_reduction": float(gains.mean()),
            "paired_median_reduction": float(np.median(gains)),
            "paired_min_reduction": float(gains.min()),
            "paired_bootstrap95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
            "faster_cases": int(np.sum(gains > 1e-10)),
            "equal_cases": int(np.sum(np.abs(gains) <= 1e-10)),
            "mean_time_s": statistics.mean(r["time_s"] for r in values),
            "mean_per_source_s": statistics.mean(r["per_source_s"] for r in values),
            "median_per_source_s": statistics.median(r["per_source_s"] for r in values),
            "p95_per_source_s": float(np.quantile([r["per_source_s"] for r in values], 0.95)),
            "max_per_source_s": max(r["per_source_s"] for r in values),
            "mean_distance_m": statistics.mean(r["distance_m"] for r in values),
            "mean_measures": statistics.mean(r["measures"] for r in values),
            "mean_failed_clears": statistics.mean(r["failed_clears"] for r in values),
            "mean_runtime_s": statistics.mean(r["runtime_s"] for r in values),
            "max_runtime_s": max(r["runtime_s"] for r in values),
            "mean_skipped_rechecks": statistics.mean(r["skipped_rechecks"] for r in values),
            "mean_service_point_uses": statistics.mean(r["service_point_uses"] for r in values),
            "mean_joint_order_changes": statistics.mean(r["joint_order_changes"] for r in values),
        }
    strata_results = {}
    for name in SCENARIOS:
        strata_results[name] = {}
        for label in policies:
            subset = [r for r in rows if r["scenario"] == name and r["policy"] == label]
            if subset:
                strata_results[name][label] = {
                    "cases": len(subset), "coverage_min": min(r["coverage"] for r in subset),
                    "mean_reduction": statistics.mean(1.0 - r["time_s"] / bases[(r["scenario"], r["seed"])]["time_s"] for r in subset),
                    "mean_per_source_s": statistics.mean(r["per_source_s"] for r in subset),
                }
    return {
        "data_status": "offline_synthetic_independent_strata_not_official_scores",
        "overall": overall, "strata": strata_results,
        "unique_world_count": len({r["world_hash"] for r in bases.values()}),
        "unique_seed_count": len({r["seed"] for r in bases.values()}),
        "full_configs": {label: next(r["config"] for r in rows if r["policy"] == label) for label in policies},
        "note": "Stratified paired bootstrap is conditional on these synthetic stress strata; it is not an official-score guarantee. Concurrent runtime figures are machine/load dependent.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "holdout"), required=True)
    parser.add_argument("--per-stratum", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="optional fresh seed block; defaults preserve the original screen/holdout sets",
    )
    parser.add_argument("--policies", nargs="+", choices=tuple(POLICIES), default=["original", "prune", "service", "joint"])
    args = parser.parse_args()
    if "original" not in args.policies or args.per_stratum < 1:
        parser.error("Include original and at least one case per stratum")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Use a new empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    base_seed = args.seed_start if args.seed_start is not None else (
        730000 if args.phase == "screen" else 840000
    )
    jobs = [(str(ROOT), str(args.baseline_root.resolve()), str(args.out.resolve()), scenario,
             base_seed + index * 1000 + replicate, label)
            for index, scenario in enumerate(SCENARIOS)
            for replicate in range(args.per_stratum) for label in args.policies]
    hashes_before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ("robot.py", "geometry.py", "offline_sim.py", "benchmark_q3_improvements.py")}
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if completed % 10 == 0 or row["failure"] or row["coverage"] < 1.0:
                print(json.dumps({"completed": completed, "total": len(jobs), "last": row["policy"],
                                  "coverage": row["coverage"], "failure": row["failure"]}), flush=True)
    rows.sort(key=lambda r: (r["scenario"], r["seed"], r["policy"]))
    result = summary(rows, args.policies)
    result.update({"phase": args.phase, "created_at": datetime.now().astimezone().isoformat(),
                   "source_sha256": hashes_before,
                   "source_unchanged_during_run": all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in hashes_before.items()),
                   "baseline_root": str(args.baseline_root.resolve())})
    (args.out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2), flush=True)
    if not result["source_unchanged_during_run"]:
        raise SystemExit("Source changed during experiment; do not freeze mixed-version results")
    if any(v["failures"] or v["coverage_min"] < 1.0 for v in result["overall"].values()):
        raise SystemExit("A case failed; inspect rows before adopting any policy")


if __name__ == "__main__":
    main()
