"""Bounded-cost Q4 refinements that never use simulator truth.

The guarantee layer keeps conservative cells with one shared reception radius
and one shared emission orientation per source. Finite hypotheses are used only
to rank actions; they never authorize a clear or remove a feasible source.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from geometry import clip_convex_polygon_halfplane, convex_hull, min_enclosing_circle

TAU = 2.0 * math.pi
GEOM_TOL = 1e-6


def _arc(center: float, half_width: float) -> list[tuple[float, float]]:
    if half_width >= math.pi:
        return [(0.0, TAU)]
    lo = (center - half_width) % TAU
    hi = (center + half_width) % TAU
    return [(lo, hi)] if lo <= hi else [(0.0, hi), (lo, TAU)]


def _intersect_arcs(left, right):
    return [
        (max(a, c), min(b, d))
        for a, b in left
        for c, d in right
        if max(a, c) <= min(b, d) + 1e-12
    ]


def _orientation_arcs(center, radius, positive, shadow):
    """Return an existential relaxation over a ball containing a position cell."""
    arcs = [(0.0, TAU)]
    for sign, observers in ((1.0, positive), (-1.0, shadow)):
        for observer in observers:
            vector = sign * (observer - center)
            distance = float(np.linalg.norm(vector))
            if distance <= radius + GEOM_TOL:
                continue
            half_width = math.acos(
                max(-1.0, min(1.0, -(radius + GEOM_TOL) / distance))
            )
            arcs = _intersect_arcs(
                arcs,
                _arc(math.atan2(vector[1], vector[0]), half_width),
            )
            if not arcs:
                return []
    return arcs


def _arrays(observations):
    positive = np.asarray(
        [o.position for o in observations if o.result in {"direction", "near"}],
        dtype=float,
    ).reshape((-1, 2))
    negative = np.asarray(
        [o.position for o in observations if o.result == "no_signal"],
        dtype=float,
    ).reshape((-1, 2))
    return positive, negative


def cell_can_contain_source(
    cell,
    positive,
    negative,
    failed,
    min_radius=1000.0,
    max_radius=1500.0,
    clear_radius=20.0,
):
    """Return False only when a cell is impossible for both source types.

    True is a relaxation, not a statement that every point in the cell is
    feasible. Distance and angle tests are widened across the cell so a true
    source cannot be rejected by this predicate.
    """
    if not len(cell):
        return False
    center = (np.min(cell, axis=0) + np.max(cell, axis=0)) / 2.0
    radius = float(np.max(np.linalg.norm(cell - center, axis=1)))
    if len(failed):
        maximums = np.max(
            np.linalg.norm(cell[None, :, :] - failed[:, None, :], axis=2),
            axis=1,
        )
        if np.any(maximums < clear_radius - GEOM_TOL):
            return False
    if not len(positive):
        return True
    positive_lower = np.maximum(
        0.0,
        np.linalg.norm(positive - center, axis=1) - radius - GEOM_TOL,
    )
    shared_radius_lower = max(min_radius, float(np.max(positive_lower)))
    if shared_radius_lower > max_radius + GEOM_TOL:
        return False
    if not len(negative):
        return True
    negative_upper = (
        np.max(
            np.linalg.norm(cell[None, :, :] - negative[:, None, :], axis=2),
            axis=1,
        )
        + GEOM_TOL
    )
    forced_shadow = negative[negative_upper < shared_radius_lower - GEOM_TOL]
    if not len(forced_shadow):
        return True
    return bool(_orientation_arcs(center, radius, positive, forced_shadow))


@dataclass
class JointRefinement:
    region: np.ndarray
    cells: list[np.ndarray]
    rejected_cells: int = 0
    timed_out: bool = False


def refine_region(
    region,
    observations,
    failed_points,
    *,
    min_radius=1000.0,
    max_radius=1500.0,
    clear_radius=20.0,
    max_cells=96,
    deadline=math.inf,
):
    """Remove certified-impossible cells and return an outer convex hull.

    Unexamined cells are retained. A deadline hit returns the original region,
    so incomplete refinement cannot weaken the fallback guarantee.
    """
    original = np.asarray(region, dtype=float)
    if len(original) < 3:
        return JointRefinement(original, [original])
    positive, negative = _arrays(observations)
    failed = np.asarray(failed_points, dtype=float).reshape((-1, 2))
    if not len(negative) and not len(failed):
        return JointRefinement(original, [original])
    pending = [original]
    kept = []
    rejected = 0
    splits = 0
    while pending:
        if time.monotonic() > deadline:
            return JointRefinement(original, [original], rejected, True)
        index = max(
            range(len(pending)),
            key=lambda i: float(np.max(np.ptp(pending[i], axis=0))),
        )
        cell = pending.pop(index)
        if not cell_can_contain_source(
            cell,
            positive,
            negative,
            failed,
            min_radius,
            max_radius,
            clear_radius,
        ):
            rejected += 1
            continue
        widths = np.ptp(cell, axis=0)
        axis = int(np.argmax(widths))
        if splits >= max_cells or widths[axis] <= 8.0:
            kept.append(cell)
            continue
        cut = 0.5 * (float(np.min(cell[:, axis])) + float(np.max(cell[:, axis])))
        normal = np.zeros(2)
        normal[axis] = 1.0
        halves = (
            clip_convex_polygon_halfplane(cell, normal, cut),
            clip_convex_polygon_halfplane(cell, -normal, -cut),
        )
        pending.extend(half for half in halves if len(half))
        splits += 1
    if not kept:
        return JointRefinement(original, [original], rejected)
    return JointRefinement(convex_hull(np.vstack(kept)), kept, rejected)


def safe_clear_point(
    region,
    current,
    following=None,
    *,
    radius=19.5,
    reference=None,
    deadline=math.inf,
):
    """Choose a verified safe point no worse than a safe reference route."""
    vertices = np.asarray(region, dtype=float)
    current = np.asarray(current, dtype=float)
    circle = min_enclosing_circle(vertices)
    reference = circle.center.copy() if reference is None else np.asarray(reference, dtype=float)
    if circle.radius > radius + 1e-8:
        return None
    following = None if following is None else np.asarray(following, dtype=float)

    def safe(point):
        return float(np.max(np.linalg.norm(vertices - point, axis=1))) <= radius + 1e-8

    def cost(point):
        value = float(np.linalg.norm(current - point))
        if following is not None:
            value += float(np.linalg.norm(following - point))
        return value

    if not safe(reference):
        reference = circle.center.copy()
    if not safe(reference):
        return None
    candidates = [reference]
    if safe(current):
        candidates.append(current)
    if following is not None:
        vector = following - current
        quadratic = float(np.dot(vector, vector))
        if quadratic > 1e-12:
            lo, hi = 0.0, 1.0
            for vertex in vertices:
                delta = current - vertex
                linear = 2.0 * float(np.dot(delta, vector))
                constant = float(np.dot(delta, delta)) - radius * radius
                discriminant = linear * linear - 4.0 * quadratic * constant
                if discriminant < 0.0:
                    lo, hi = 1.0, 0.0
                    break
                root = math.sqrt(max(0.0, discriminant))
                lo = max(lo, (-linear - root) / (2.0 * quadratic))
                hi = min(hi, (-linear + root) / (2.0 * quadratic))
            if lo <= hi:
                point = current + (0.5 * (lo + hi)) * vector
                if safe(point):
                    return point
    for vertex in vertices:
        vector = current - vertex
        distance = float(np.linalg.norm(vector))
        if distance > 1e-12:
            candidate = vertex + radius * vector / distance
            if safe(candidate):
                candidates.append(candidate)
    for i, left in enumerate(vertices):
        if time.monotonic() > deadline:
            return reference
        for right in vertices[:i]:
            vector = right - left
            distance = float(np.linalg.norm(vector))
            if distance <= 1e-9 or distance > 2 * radius:
                continue
            midpoint = (left + right) / 2.0
            height = math.sqrt(max(0.0, radius * radius - distance * distance / 4.0))
            normal = np.array([-vector[1], vector[0]]) / distance
            for candidate in (midpoint + height * normal, midpoint - height * normal):
                if safe(candidate):
                    candidates.append(candidate)
    gradient = (reference - current) / max(
        1e-9,
        float(np.linalg.norm(reference - current)),
    )
    if following is not None:
        gradient += (reference - following) / max(
            1e-9,
            float(np.linalg.norm(reference - following)),
        )
    if np.linalg.norm(gradient) > 1e-9:
        direction = -gradient / np.linalg.norm(gradient)
        lo, hi = 0.0, 2.0 * radius
        for _ in range(32):
            middle = (lo + hi) / 2.0
            if safe(reference + middle * direction):
                lo = middle
            else:
                hi = middle
        candidates.append(reference + lo * direction)
    return min(candidates, key=cost).copy()


@dataclass
class SourceHypothesis:
    point: np.ndarray
    radius: float
    orientation_arcs: list[tuple[float, float]]
    omni: bool
    weight: float


def build_hypotheses(region, observations, failed_points=(), max_points=25):
    """Build finite scenarios used only as a time-cost proxy."""
    region = np.asarray(region, dtype=float)
    positive, negative = _arrays(observations)
    failed = np.asarray(failed_points, dtype=float).reshape((-1, 2))
    center = np.mean(region, axis=0)
    samples = [center]
    for vertex in region:
        for fraction in (0.25, 0.60, 0.90):
            samples.append(center + fraction * (vertex - center))
    if len(samples) > max_points:
        indices = np.linspace(0, len(samples) - 1, max_points, dtype=int)
        samples = [samples[index] for index in indices]
    hypotheses = []
    for point in samples:
        if len(failed) and np.any(
            np.linalg.norm(failed - point, axis=1) < 20.0 - GEOM_TOL
        ):
            continue
        positive_distances = np.linalg.norm(positive - point, axis=1)
        lo = max(1000.0, float(np.max(positive_distances, initial=0.0)))
        if lo > 1500.0:
            continue
        negative_distances = np.linalg.norm(negative - point, axis=1)
        radii = (lo + (1500.0 - lo) * np.array([1 / 6, 0.5, 5 / 6])).tolist()
        for radius in radii:
            close_negative = negative[negative_distances <= radius + GEOM_TOL]
            if not len(close_negative):
                hypotheses.append(SourceHypothesis(point, radius, [], True, 0.5))
            arcs = _orientation_arcs(point, 0.0, positive, close_negative)
            arc_length = sum(max(0.0, b - a) for a, b in arcs)
            if arcs and arc_length > 1e-9:
                hypotheses.append(
                    SourceHypothesis(point, radius, arcs, False, 0.5 * arc_length / TAU)
                )
    if hypotheses:
        total = sum(h.weight for h in hypotheses)
        for hypothesis in hypotheses:
            hypothesis.weight /= total
    return hypotheses


def visibility_probability(hypothesis, observer):
    vector = np.asarray(observer) - hypothesis.point
    distance = float(np.linalg.norm(vector))
    if distance > hypothesis.radius + 1e-8:
        return 0.0
    if hypothesis.omni or distance <= 1e-9:
        return 1.0
    visible_arc = _arc(math.atan2(vector[1], vector[0]), math.pi / 2)
    denominator = sum(max(0.0, b - a) for a, b in hypothesis.orientation_arcs)
    overlap = _intersect_arcs(hypothesis.orientation_arcs, visible_arc)
    return min(
        1.0,
        sum(max(0.0, b - a) for a, b in overlap) / max(denominator, 1e-12),
    )


def rank_measurements(
    candidates,
    region,
    observations,
    current,
    failed_points=(),
    *,
    error_deg=1.01,
    switch_cost=0.0,
    deadline=math.inf,
):
    """Rank robust candidates with a bounded two-stage time proxy."""
    original = list(candidates)
    if not original or len(region) < 3:
        return original, False
    hypotheses = build_hypotheses(region, observations, failed_points)
    if not hypotheses:
        return original, False
    sources = np.asarray([h.point for h in hypotheses])
    weights = np.asarray([h.weight for h in hypotheses])
    positive, _ = _arrays(observations)
    options = original[:6]
    center = min_enclosing_circle(region).center
    if not any(np.linalg.norm(np.asarray(p) - center) < 1.0 for p in options):
        untried = all(
            np.linalg.norm(np.asarray(o.position) - center) >= 1.0
            for o in observations
        )
        if (
            untried
            and float(np.max(np.linalg.norm(region - center, axis=1))) <= 998.0
        ):
            options.append(tuple(map(float, center)))
    scored = []
    for candidate in options:
        if time.monotonic() > deadline:
            return original, False
        point = np.asarray(candidate, dtype=float)
        distance = np.linalg.norm(sources - point, axis=1)
        visible = np.asarray(
            [visibility_probability(hypothesis, point) for hypothesis in hypotheses]
        )
        response_weight = weights * visible
        probability = float(np.sum(response_weight))
        angles = np.arctan2(sources[:, 1] - point[1], sources[:, 0] - point[0])
        residual = (angles[:, None] - angles[None, :] + math.pi) % TAU - math.pi
        compatible = np.abs(residual) <= math.radians(2.0 * error_deg)
        conditional = compatible * response_weight[None, :]
        mass = conditional.sum(axis=1)
        estimate = conditional @ sources / np.maximum(mass[:, None], 1e-12)
        errors = np.linalg.norm(sources - estimate, axis=1)
        clear_cost = np.linalg.norm(estimate - point, axis=1) / 5.0 + 5.0
        clear_cost += (errors > 19.5) * (
            8.0 + np.minimum(400.0, 2.0 * errors) / 5.0
        )
        clear_cost[distance <= 5.0] = 5.0
        positive_cost = float(np.sum(response_weight * clear_cost))
        anchor_distance = min(
            (float(np.linalg.norm(p - point)) for p in positive),
            default=500.0,
        )
        miss_cost = (
            6.0
            + 0.5 * anchor_distance / 5.0
            + float(np.average(distance, weights=weights)) / 5.0
            + 5.0
        )
        score = (
            float(np.linalg.norm(point - np.asarray(current))) / 5.0
            + 5.0
            + switch_cost
        )
        score += positive_cost + (1.0 - probability) * miss_cost
        scored.append((score, candidate))
    baseline_score = scored[0][0]
    best_score, best = min(scored, key=lambda item: item[0])
    if best_score + 3.0 >= baseline_score:
        return original, False
    best_tuple = tuple(map(float, best))
    return [best_tuple] + [
        point
        for point in original
        if np.linalg.norm(np.asarray(point) - best_tuple) > 1e-7
    ], True
