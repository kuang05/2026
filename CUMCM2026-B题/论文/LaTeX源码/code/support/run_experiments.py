"""Reproducible offline experiments for Questions 1--4.

Offline worlds are deliberately labeled synthetic.  They validate the policy
and geometry but are never substituted for the three official simulator runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from geometry import (
    bearing_region,
    cross,
    diameter,
    diameter_circle_certificate,
    min_enclosing_circle,
    optimal_second_point,
    q3_cover_points,
    q4_compact_cover_points,
    q4_triangular_cover_points,
    second_point_candidate_region,
)
from robot import run_offline


def _q1_metrics() -> dict:
    source = np.array([420.0, 680.0])
    # Keep every observation within the largest possible 1500 m reception
    # disk, so this is a feasible Q1 instance rather than an empty intersection.
    observers = [np.array([-500.0, 0.0]), np.array([800.0, -200.0]), np.array([-200.0, 1100.0])]
    errors_deg = [-0.73, 0.42, -0.18]
    readings = []
    for observer, error in zip(observers, errors_deg):
        true_angle = math.atan2(source[1] - observer[1], source[0] - observer[0])
        readings.append((observer, true_angle + math.radians(error)))
    region = bearing_region(
        readings,
        target_radius=1800.0,
        error_deg=1.0,
        extra_disks=[(observer, 1500.0) for observer in observers],
        n_disk=720,
    )
    certificate = diameter_circle_certificate(region)
    circle = certificate["min_enclosing_circle"]
    engineering_region = bearing_region(
        readings,
        target_radius=1800.0,
        error_deg=1.01,
        extra_disks=[(observer, 1500.0) for observer in observers],
        n_disk=720,
    )
    engineering_circle = min_enclosing_circle(engineering_region)
    return {
        "error_bound_deg": 1.0,
        "observer_count": len(observers),
        "region_vertex_count": int(len(region)),
        "diameter_m": float(diameter(region)),
        "min_enclosing_radius_m": float(circle.radius),
        "diameter_circle_covers_region": bool(certificate["covers"]),
        "diameter_circle_overflow_m": float(certificate["overflow"]),
        "engineering_1_01_deg": {
            "diameter_m": float(diameter(engineering_region)),
            "min_enclosing_radius_m": float(engineering_circle.radius),
        },
        "source_in_region": bool(
            len(region) > 0
            and np.all(
                [
                    cross(region[(i + 1) % len(region)] - region[i], source - region[i]) >= -1e-6
                    for i in range(len(region))
                ]
            )
        ),
    }


def _q2_metrics() -> dict:
    rows = []
    for bearing_deg in range(0, 360, 45):
        theta = math.radians(bearing_deg)
        region, candidates, scores = second_point_candidate_region((0.0, 0.0), theta, grid_step=40.0)
        best, _ = optimal_second_point((0.0, 0.0), theta, grid_step=40.0)
        rows.append(
            {
                "first_bearing_deg": bearing_deg,
                "first_region_vertices": int(len(region)),
                "candidate_count": int(len(candidates)),
                "best_x_m": float(best[0]),
                "best_y_m": float(best[1]),
                "best_score": float(np.max(scores)),
            }
        )
    return {
        "rows": rows,
        "candidate_rule": "within 1000 m of every first-reading feasible point and worst crossing angle >=30 deg",
    }


def _run_policy_experiments(out: Path, seeds: list[int]) -> list[dict]:
    rows: list[dict] = []
    for problem in (3, 4):
        for seed in seeds:
            run_dir = out / "offline_runs" / f"p{problem}_{seed:04d}"
            result = run_offline(seed, problem, run_dir)
            rows.append(
                {
                    "scenario": "random",
                    "problem": problem,
                    "seed": seed,
                    "total_sources": result["total_sources"],
                    "directional_sources": result["source_directional"],
                    "cleared_count": result["cleared_count"],
                    "coverage": result["coverage"],
                    "virtual_time_s": result["virtual_time_s"],
                    "average_time_s": result["average_localization_clear_time_s"],
                    "actions": result["actions"],
                    "failure": result["failure"] or "",
                }
            )

    # Adversarial stress cases: every source is directional, all receive
    # radii are the lower bound, and radial draws are boundary-biased.  These
    # belong to Q4 only; keeping them outside the problem loop prevents the
    # same run from being counted once under Q3 and once under Q4.
    for seed in seeds[: min(5, len(seeds))]:
        run_dir = out / "offline_runs" / f"p4_stress_{seed:04d}"
        result = run_offline(
            seed,
            4,
            run_dir,
            {
                "directional_probability": 1.0,
                "count": 16,
                "receive_radius_range": (1000.0, 1000.0),
                "boundary_bias": True,
            },
        )
        rows.append(
            {
                "scenario": "adversarial_all_directional_min_radius",
                "problem": 4,
                "seed": seed,
                "total_sources": result["total_sources"],
                "directional_sources": result["source_directional"],
                "cleared_count": result["cleared_count"],
                "coverage": result["coverage"],
                "virtual_time_s": result["virtual_time_s"],
                "average_time_s": result["average_localization_clear_time_s"],
                "actions": result["actions"],
                "failure": result["failure"] or "",
            }
        )
    return rows


def _write_policy_summary(out: Path, rows: list[dict]) -> dict:
    csv_path = out / "monte_carlo.csv"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for key in sorted({(row["scenario"], row["problem"]) for row in rows}):
        subset = [row for row in rows if (row["scenario"], row["problem"]) == key]
        coverages = [float(row["coverage"]) for row in subset]
        times = [float(row["average_time_s"]) for row in subset]
        grouped[f"{key[0]}_p{key[1]}"] = {
            "runs": len(subset),
            "coverage_mean": statistics.mean(coverages),
            "coverage_min": min(coverages),
            "coverage_sd": statistics.pstdev(coverages) if len(coverages) > 1 else 0.0,
            "average_time_mean_s": statistics.mean(times),
            "average_time_sd_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
            "failure_runs": sum(bool(row["failure"]) for row in subset),
        }
    summary = {"data_status": "offline_synthetic_regression_only", "groups": grouped}
    (out / "monte_carlo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/experiments")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    rows = _run_policy_experiments(out, seeds)
    summary = _write_policy_summary(out, rows)
    q1 = _q1_metrics()
    q2 = _q2_metrics()
    geometry = {
        "q3_scan_point_count": int(len(q3_cover_points())),
        "q4_scan_point_count": int(len(q4_compact_cover_points())),
        "q4_legacy_scan_point_count": int(len(q4_triangular_cover_points())),
    }
    (out / "q1_metrics.json").write_text(json.dumps(q1, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "q2_metrics.json").write_text(json.dumps(q2, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "geometry_metrics.json").write_text(json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "q1": q1, "q2": q2, "geometry": geometry}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
