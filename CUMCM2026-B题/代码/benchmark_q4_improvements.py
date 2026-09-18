"""Paired, offline-only validation for the final Q4 refinement profile."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICIES = {
    "baseline": {
        "q4_joint_constraints": False,
        "q4_time_lookahead": False,
        "q4_safe_clear_route": False,
    },
    "safe": {
        "q4_joint_constraints": False,
        "q4_time_lookahead": False,
        "q4_safe_clear_route": True,
    },
    "joint": {
        "q4_joint_constraints": True,
        "q4_time_lookahead": False,
        "q4_safe_clear_route": False,
    },
    "joint-safe": {
        "q4_joint_constraints": True,
        "q4_time_lookahead": False,
        "q4_safe_clear_route": True,
    },
    "combined": {
        "q4_joint_constraints": True,
        "q4_time_lookahead": True,
        "q4_safe_clear_route": True,
    },
}


def _position_hash_error(seed: int, position: dict[str, float]) -> float:
    key = (
        f"{seed}|{float(position['x']):.6f}|{float(position['y']):.6f}"
    ).encode("ascii")
    raw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return 2.0 * (raw / (2**64 - 1)) - 1.0


def _apply_error_profile(world, profile: str) -> None:
    if profile == "channel_hash":
        return
    if profile == "position_hash":
        world._fixed_error_deg = (
            lambda source, position: _position_hash_error(world.seed, position)
        )
    elif profile == "smooth_shared":
        world._fixed_error_deg = lambda source, position: (
            math.sin(position["x"] / 300.0) * math.cos(position["y"] / 500.0)
        )
    elif profile == "positive_endpoint":
        world._fixed_error_deg = lambda source, position: 1.0
    elif profile == "negative_endpoint":
        world._fixed_error_deg = lambda source, position: -1.0
    else:
        raise ValueError(f"unknown error profile: {profile}")


def _scenario_specs(stage: str) -> list[dict]:
    specs: list[tuple[str, dict, str]] = []
    if stage == "replay":
        specs.extend(("random", {}, "channel_hash") for _ in range(10))
        profiles = (
            "channel_hash",
            "channel_hash",
            "positive_endpoint",
            "negative_endpoint",
            "smooth_shared",
        )
        for count in (10, 13, 16):
            for source_type, probability in (
                ("omni", 0.0),
                ("mixed", 0.57),
                ("directional", 1.0),
            ):
                for profile in profiles:
                    specs.append(
                        (
                            f"n{count}_{source_type}_boundary",
                            {
                                "count": count,
                                "directional_probability": probability,
                                "receive_radius_range": [1000, 1000],
                                "boundary_bias": True,
                            },
                            profile,
                        )
                    )
        start = 96123000
    else:
        for index in range(6):
            specs.append(
                (
                    "random_shared_field",
                    {},
                    "position_hash" if index % 2 == 0 else "smooth_shared",
                )
            )
        for count in (10, 13, 16):
            for source_type, probability in (
                ("omni", 0.0),
                ("mixed", 0.57),
                ("directional", 1.0),
            ):
                for profile in ("position_hash", "smooth_shared"):
                    specs.append(
                        (
                            f"n{count}_{source_type}_shared_boundary",
                            {
                                "count": count,
                                "directional_probability": probability,
                                "receive_radius_range": [1000, 1000],
                                "boundary_bias": True,
                            },
                            profile,
                        )
                    )
        start = 97123000
    return [
        {
            "case": index,
            "seed": start + index * 41,
            "scenario": name,
            "world": options,
            "error_profile": profile,
        }
        for index, (name, options, profile) in enumerate(specs)
    ]


def _worker(job: dict) -> dict:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    import numpy as np

    from geometry import point_in_convex_polygon
    from offline_sim import JammerWorld
    from robot import Robot, RobotConfig, _OfflineClient

    output = Path(job["out"])
    output.mkdir(parents=True, exist_ok=False)
    world = JammerWorld.random_case(job["seed"], 4, **job["world"])
    _apply_error_profile(world, job["error_profile"])
    truth = {source.channel: source for source in world.sources}
    truth_violations = []

    class VerifiedRobot(Robot):
        def _region_for(self, channel):
            region, region_diameter, circle = super()._region_for(channel)
            if (
                self._direction_observations(channel)
                and channel in truth
                and channel not in self.cleared
                and (
                    not len(region)
                    or not point_in_convex_polygon(
                        truth[channel].position,
                        region,
                        tol=1e-5,
                    )
                )
            ):
                truth_violations.append(
                    {"channel": channel, "actions": self.action_count}
                )
            return region, region_diameter, circle

    robot = VerifiedRobot(
        _OfflineClient(world, output / "events.jsonl"),
        RobotConfig(problem=4, **POLICIES[job["policy"]]),
        output,
    )
    result = robot.run()
    position = (0.0, 0.0)
    distance = 0.0
    for event in world.events:
        request = event["request"]
        if "position" in request:
            point = request["position"]
            next_position = (point["x"], point["y"])
            distance += math.dist(position, next_position)
            position = next_position
    record = {
        "case": job["case"],
        "seed": job["seed"],
        "scenario": job["scenario"],
        "error_profile": job["error_profile"],
        "policy": job["policy"],
        "world_options": job["world"],
        "world_sha256": hashlib.sha256(
            json.dumps(
                [asdict(source) for source in world.sources],
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
        "config": asdict(robot.cfg),
        "coverage": world.cleared_count / len(world.sources),
        "sources": len(world.sources),
        "discovery_exact": robot.discovered == set(truth),
        "truth_violations": truth_violations,
        "failure": result["failure"],
        "virtual_time_s": world.virtual_time,
        "time_per_source_s": world.virtual_time / len(world.sources),
        "distance_m": distance,
        "measures": sum(e["path"] == "/measure" for e in world.events),
        "no_signal": sum(
            e["response"].get("measure_result") == "no_signal"
            for e in world.events
        ),
        "failed_clears": sum(
            e["response"].get("clear_result") == "no_target_in_range"
            for e in world.events
        ),
        "wall_runtime_s": result["program_runtime_s"],
        "refinement_stats": result.get("refinement_stats", {}),
        "source_sha256": result.get("source_sha256", {}),
    }
    (output / "validation.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def _run_subprocess(job: dict) -> dict:
    process = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", json.dumps(job)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        timeout=240,
    )
    if process.returncode:
        raise RuntimeError(
            f"{job['policy']} case {job['case']}: "
            f"{process.stderr[-3000:]} {process.stdout[-1000:]}"
        )
    return json.loads(process.stdout)


def _policy_summary(group: list[dict], baseline: dict[int, dict]) -> dict:
    import numpy as np

    ordered = sorted(group, key=lambda row: row["case"])
    reductions = np.asarray(
        [
            (baseline[row["case"]]["virtual_time_s"] - row["virtual_time_s"])
            / baseline[row["case"]]["virtual_time_s"]
            for row in ordered
        ]
    )
    rng = np.random.default_rng(20260912)
    bootstrap = np.mean(
        rng.choice(reductions, size=(10000, len(reductions)), replace=True),
        axis=1,
    )
    return {
        "cases": len(ordered),
        "coverage_min": min(row["coverage"] for row in ordered),
        "failure_runs": sum(bool(row["failure"]) for row in ordered),
        "discovery_mismatches": sum(not row["discovery_exact"] for row in ordered),
        "truth_violations": sum(len(row["truth_violations"]) for row in ordered),
        "mean_virtual_time_s": float(
            np.mean([row["virtual_time_s"] for row in ordered])
        ),
        "mean_time_per_source_s": float(
            np.mean([row["time_per_source_s"] for row in ordered])
        ),
        "p95_time_per_source_s": float(
            np.percentile([row["time_per_source_s"] for row in ordered], 95)
        ),
        "mean_distance_m": float(np.mean([row["distance_m"] for row in ordered])),
        "mean_measures": float(np.mean([row["measures"] for row in ordered])),
        "mean_failed_clears": float(
            np.mean([row["failed_clears"] for row in ordered])
        ),
        "mean_wall_runtime_s": float(
            np.mean([row["wall_runtime_s"] for row in ordered])
        ),
        "max_wall_runtime_s": max(row["wall_runtime_s"] for row in ordered),
        "paired_reduction_mean": float(np.mean(reductions)),
        "paired_reduction_median": float(np.median(reductions)),
        "paired_reduction_min": float(np.min(reductions)),
        "paired_reduction_max": float(np.max(reductions)),
        "paired_bootstrap_ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "faster_cases": int(np.sum(reductions > 1e-10)),
        "slower_cases": int(np.sum(reductions < -1e-10)),
    }


def _summarize(rows: list[dict], stage: str) -> dict:
    baseline = {row["case"]: row for row in rows if row["policy"] == "baseline"}
    policies = {}
    for policy in sorted({row["policy"] for row in rows}):
        policies[policy] = _policy_summary(
            [row for row in rows if row["policy"] == policy],
            baseline,
        )
    pairs_match = all(
        row["world_sha256"] == baseline[row["case"]]["world_sha256"]
        for row in rows
    )
    return {
        "stage": stage,
        "data_status": "offline_synthetic_independent_seeds_not_official",
        "paired_worlds_match": pairs_match,
        "policies": policies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--stage", choices=("replay", "independent"), default="independent")
    parser.add_argument("--policies", nargs="+", choices=tuple(POLICIES), default=list(POLICIES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/q4_improvement_independent")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(json.loads(args.worker)), ensure_ascii=False))
        return
    if "baseline" not in args.policies:
        parser.error("baseline is required for paired comparison")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            **case,
            "policy": policy,
            "out": str(output / f"case{case['case']:03d}_{policy}"),
        }
        for case in _scenario_specs(args.stage)
        for policy in args.policies
    ]
    rows = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_subprocess, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{len(rows)}/{len(jobs)} {row['policy']} case={row['case']} "
                f"coverage={row['coverage']:.3f} time={row['virtual_time_s']:.2f}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["case"], row["policy"]))
    (output / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = _summarize(rows, args.stage)
    summary["validation_wall_runtime_s"] = time.monotonic() - started
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    selected = summary["policies"].get("combined")
    if not summary["paired_worlds_match"] or any(
        metrics["coverage_min"] < 1.0
        or metrics["failure_runs"]
        or metrics["discovery_mismatches"]
        or metrics["truth_violations"]
        for metrics in summary["policies"].values()
    ):
        raise SystemExit(2)
    if selected is not None and selected["paired_reduction_mean"] <= 0.0:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
