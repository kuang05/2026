"""Geometry utilities for bearing-only localization with bounded angular error."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-10


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def unit(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=float)


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def bearing(p: Sequence[float], q: Sequence[float]) -> float:
    return math.atan2(q[1] - p[1], q[0] - p[0])


def line_intersection(p: np.ndarray, u: np.ndarray, q: np.ndarray, v: np.ndarray):
    den = cross(u, v)
    if abs(den) < EPS:
        return None
    t = cross(q - p, v) / den
    return p + t * u


def _clip_halfplane(poly: list[np.ndarray], a: np.ndarray, b: float, tol=1e-9):
    """Keep a dot x >= b, using Sutherland-Hodgman clipping."""
    if not poly:
        return []
    out: list[np.ndarray] = []
    for p, q in zip(poly, poly[1:] + poly[:1]):
        fp = float(np.dot(a, p) - b)
        fq = float(np.dot(a, q) - b)
        inp, inq = fp >= -tol, fq >= -tol
        if inp:
            out.append(p)
        if inp != inq:
            den = fp - fq
            if abs(den) > EPS:
                lam = fp / den
                out.append(p + lam * (q - p))
    return out


def clip_convex_polygon_halfplane(
    poly: np.ndarray,
    a: Sequence[float],
    b: float,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return the part of a convex polygon satisfying ``a dot x >= b``."""
    clipped = _clip_halfplane(
        [np.asarray(point, dtype=float) for point in np.asarray(poly, dtype=float)],
        np.asarray(a, dtype=float),
        float(b),
        tol,
    )
    if not clipped:
        return np.empty((0, 2), dtype=float)
    return np.asarray(clipped, dtype=float)


def sector_halfplanes(observer: Sequence[float], theta: float, error_deg: float = 1.0):
    """Return (a,b) constraints a dot z >= b for a forward angular sector."""
    p = np.asarray(observer, dtype=float)
    d = math.radians(error_deg)
    lo, hi = theta - d, theta + d
    u_lo, u_hi = unit(lo), unit(hi)
    # cross(u_lo,z-p)>=0 -> (-u_lo_y,u_lo_x) dot z >= cross(u_lo,p)
    a1 = np.array([-u_lo[1], u_lo[0]])
    b1 = float(np.dot(a1, p))
    # cross(u_hi,z-p)<=0 -> (u_hi_y,-u_hi_x) dot z >= -cross(u_hi,p)
    a2 = np.array([u_hi[1], -u_hi[0]])
    b2 = float(np.dot(a2, p))
    return [(a1, b1), (a2, b2)]


def bearing_sector_polygon(
    observations: Iterable[tuple[Sequence[float], float]],
    error_deg: float = 1.0,
    tol: float = 1e-8,
) -> tuple[np.ndarray, str]:
    """Intersect only bearing half-planes and classify the resulting set.

    A bounded nonempty intersection is returned by its convex-hull vertices.
    For an unbounded intersection, the returned finite vertices are diagnostic
    only; callers should add the official target disk before computing a
    physical diameter.
    """
    if not (0.0 < error_deg < 90.0):
        raise ValueError("error_deg must lie strictly between 0 and 90 degrees")
    halfplanes = [
        (np.asarray(normal, dtype=float), float(offset))
        for observer, theta in observations
        for normal, offset in sector_halfplanes(observer, theta, error_deg)
    ]
    if not halfplanes:
        return np.empty((0, 2), dtype=float), "unbounded"

    candidates: list[np.ndarray] = []
    for i, (left_normal, left_offset) in enumerate(halfplanes):
        for right_normal, right_offset in halfplanes[:i]:
            matrix = np.vstack([left_normal, right_normal])
            if abs(float(np.linalg.det(matrix))) <= EPS:
                continue
            point = np.linalg.solve(matrix, np.array([left_offset, right_offset]))
            if all(float(np.dot(normal, point) - offset) >= -tol for normal, offset in halfplanes):
                candidates.append(point)

    normals = np.asarray([normal for normal, _ in halfplanes])
    normal_angles = np.sort(np.mod(np.arctan2(normals[:, 1], normals[:, 0]), 2.0 * math.pi))
    circular_gaps = np.diff(np.r_[normal_angles, normal_angles[0] + 2.0 * math.pi])
    unbounded = float(np.max(circular_gaps)) >= math.pi - 1e-12

    if not candidates:
        # Bearing sectors are pointed wedges.  Within this domain, a feasible
        # nonempty intersection has at least one pairwise boundary vertex.
        return np.empty((0, 2), dtype=float), "empty"
    hull = convex_hull(np.asarray(candidates, dtype=float))
    if unbounded:
        return hull, "unbounded"
    if len(hull) == 1:
        return hull, "point"
    if len(hull) == 2:
        return hull, "segment"
    return hull, "polygon"


def disk_polygon(radius: float, n: int = 720, circumscribed: bool = True) -> list[np.ndarray]:
    """Return a regular polygon approximation of a disk.

    Feasible regions are used to decide whether a 20 m clear is guaranteed.
    The default polygon therefore circumscribes the disk, so approximation
    error can enlarge the uncertainty set but can never exclude the source.
    """
    if n < 8:
        raise ValueError("a disk approximation needs at least 8 sides")
    scale = radius / math.cos(math.pi / n) if circumscribed else radius
    return [scale * unit(2 * math.pi * k / n) for k in range(n)]


def bearing_region(
    observations: Iterable[tuple[Sequence[float], float]],
    target_radius: float = 1800.0,
    error_deg: float = 1.0,
    extra_disks: Iterable[tuple[Sequence[float], float]] = (),
    n_disk: int = 720,
    circumscribed_disks: bool = True,
) -> np.ndarray:
    """Intersect angular sectors with polygonal target/reception disks.

    Circumscribed polygons give a conservative outer feasible region.  Setting
    ``circumscribed_disks`` to false gives a contained inner region, which can
    be paired with the outer region to audit circular-boundary approximation.
    """
    poly = disk_polygon(target_radius, n_disk, circumscribed=circumscribed_disks)
    for p, theta in observations:
        for a, b in sector_halfplanes(p, theta, error_deg):
            poly = _clip_halfplane(poly, a, b)
            if not poly:
                return np.empty((0, 2))
    for center, radius in extra_disks:
        # A polygonal disk is enough for planning; increase n_disk for final audit.
        c = np.asarray(center, dtype=float)
        disk = [
            c + p
            for p in disk_polygon(
                radius,
                n_disk,
                circumscribed=circumscribed_disks,
            )
        ]
        # Intersect two convex polygons by clipping each edge of the disk polygon.
        for p, q in zip(disk, disk[1:] + disk[:1]):
            edge = q - p
            # interior is left of CCW edge: cross(edge,z-p)>=0
            a = np.array([-edge[1], edge[0]])
            b = float(np.dot(a, p))
            poly = _clip_halfplane(poly, a, b)
            if not poly:
                return np.empty((0, 2))
    return np.asarray(poly, dtype=float)


def bearing_region_bounds(
    observations: Iterable[tuple[Sequence[float], float]],
    target_radius: float = 1800.0,
    error_deg: float = 1.0,
    extra_disks: Iterable[tuple[Sequence[float], float]] = (),
    n_disk: int = 720,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contained and conservative polygonal bounds for the feasible set."""
    readings = [(np.asarray(point, dtype=float), float(theta)) for point, theta in observations]
    disks = [(np.asarray(center, dtype=float), float(radius)) for center, radius in extra_disks]
    inner = bearing_region(
        readings,
        target_radius,
        error_deg,
        disks,
        n_disk,
        circumscribed_disks=False,
    )
    outer = bearing_region(
        readings,
        target_radius,
        error_deg,
        disks,
        n_disk,
        circumscribed_disks=True,
    )
    return inner, outer


def convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points.copy()
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)

    def orient(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and orient(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and orient(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def intersect_convex_polygons(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Intersect two counter-clockwise convex polygons."""
    poly = [np.asarray(point, dtype=float) for point in np.asarray(subject)]
    clip = [np.asarray(point, dtype=float) for point in np.asarray(clipper)]
    if not poly or not clip:
        return np.empty((0, 2), dtype=float)
    for left, right in zip(clip, clip[1:] + clip[:1]):
        edge = right - left
        normal = np.array([-edge[1], edge[0]], dtype=float)
        poly = _clip_halfplane(poly, normal, float(np.dot(normal, left)))
        if not poly:
            return np.empty((0, 2), dtype=float)
    return np.asarray(poly, dtype=float)


def diameter(poly: np.ndarray) -> float:
    if len(poly) < 2:
        return 0.0
    hull = convex_hull(poly)
    if len(hull) < 2:
        return 0.0
    return float(max(np.linalg.norm(hull[i] - hull[j]) for i in range(len(hull)) for j in range(i)))


@dataclass
class Circle:
    center: np.ndarray
    radius: float


def _circle_two(a: np.ndarray, b: np.ndarray) -> Circle:
    return Circle((a + b) / 2.0, float(np.linalg.norm(a - b) / 2.0))


def _circle_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Circle | None:
    d = 2 * cross(b - a, c - a)
    if abs(d) < 1e-12:
        return None
    aa, bb, cc = np.dot(a, a), np.dot(b, b), np.dot(c, c)
    ux = (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1])) / d
    uy = (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0])) / d
    o = np.array([ux, uy])
    return Circle(o, float(np.linalg.norm(o - a)))


def _contains(circle: Circle, point: np.ndarray, tol: float = 1e-7) -> bool:
    return float(np.linalg.norm(point - circle.center)) <= circle.radius + tol


def _circle_collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Circle:
    pairs = [(a, b), (a, c), (b, c)]
    x, y = max(pairs, key=lambda pair: np.linalg.norm(pair[0] - pair[1]))
    return _circle_two(x, y)


def min_enclosing_circle(points: np.ndarray) -> Circle:
    """Smallest enclosing circle of a finite point set.

    This is the deterministic incremental form of the standard randomized
    algorithm.  It is exact up to floating-point tolerance and avoids the
    quartic candidate enumeration used by the first prototype.
    """
    pts = convex_hull(np.asarray(points, dtype=float))
    if len(pts) == 0:
        return Circle(np.zeros(2), math.inf)

    circle: Circle | None = None
    for i, p in enumerate(pts):
        if circle is not None and _contains(circle, p):
            continue
        circle = Circle(p.copy(), 0.0)
        for j, q in enumerate(pts[:i]):
            if _contains(circle, q):
                continue
            circle = _circle_two(p, q)
            for r in pts[:j]:
                if _contains(circle, r):
                    continue
                candidate = _circle_three(p, q, r)
                circle = candidate if candidate is not None else _circle_collinear(p, q, r)
    assert circle is not None
    return circle


def diameter_circle_certificate(points: np.ndarray, tol: float = 1e-7) -> dict[str, object]:
    """Build a directly checkable certificate for the diameter-circle question."""
    hull = convex_hull(np.asarray(points, dtype=float))
    if len(hull) == 0:
        return {
            "status": "empty",
            "diameter": 0.0,
            "diameter_endpoints": None,
            "diameter_center": None,
            "diameter_radius": 0.0,
            "witness": None,
            "witness_distance": math.inf,
            "overflow": math.inf,
            "min_enclosing_circle": min_enclosing_circle(hull),
            "support_points": np.empty((0, 2), dtype=float),
            "covers": None,
            "circle_inflation_ratio": None,
        }

    if len(hull) == 1:
        circle = min_enclosing_circle(hull)
        return {
            "status": "point",
            "diameter": 0.0,
            "diameter_endpoints": (hull[0].copy(), hull[0].copy()),
            "diameter_center": hull[0].copy(),
            "diameter_radius": 0.0,
            "witness": hull[0].copy(),
            "witness_distance": 0.0,
            "overflow": 0.0,
            "min_enclosing_circle": circle,
            "support_points": hull.copy(),
            "covers": True,
            "circle_inflation_ratio": None,
        }

    distance_value, left_index, right_index = max(
        (
            float(np.linalg.norm(hull[i] - hull[j])),
            i,
            j,
        )
        for i in range(len(hull))
        for j in range(i)
    )
    left, right = hull[left_index].copy(), hull[right_index].copy()
    midpoint = (left + right) / 2.0
    distances = np.linalg.norm(hull - midpoint, axis=1)
    witness_index = int(np.argmax(distances))
    witness_distance = float(distances[witness_index])
    diameter_radius = distance_value / 2.0
    overflow = witness_distance - diameter_radius
    circle = min_enclosing_circle(hull)
    support_tol = max(tol, 1e-7)
    support = hull[
        np.abs(np.linalg.norm(hull - circle.center, axis=1) - circle.radius)
        <= support_tol
    ]
    return {
        "status": "segment" if len(hull) == 2 else "polygon",
        "diameter": distance_value,
        "diameter_endpoints": (left, right),
        "diameter_center": midpoint,
        "diameter_radius": diameter_radius,
        "witness": hull[witness_index].copy(),
        "witness_distance": witness_distance,
        "overflow": overflow,
        "min_enclosing_circle": circle,
        "support_points": support,
        "covers": overflow <= tol,
        "circle_inflation_ratio": 2.0 * circle.radius / distance_value,
    }


def robust_estimate(observations, target_radius=1800.0, error_deg=1.0):
    poly = bearing_region(observations, target_radius, error_deg)
    return poly, min_enclosing_circle(poly), diameter(poly)


def point_in_convex_polygon(point: Sequence[float], poly: np.ndarray, tol: float = 1e-8) -> bool:
    """Membership of a CCW convex polygon, including point/segment limits."""
    vertices = np.asarray(poly, dtype=float)
    if len(vertices) == 0:
        return False
    q = np.asarray(point, dtype=float)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(vertices)):
        return False
    if len(vertices) == 1:
        return float(np.linalg.norm(q - vertices[0])) <= tol
    if len(vertices) == 2:
        return point_segment_distance(q, vertices[0], vertices[1]) <= tol
    # Clipping can leave repeated or collinear vertices. Cross products alone
    # would otherwise accept an entire line (or plane) as inside that set.
    shifted = vertices - vertices[0]
    area2 = float(np.sum(shifted[:, 0] * np.roll(shifted[:, 1], -1)
                         - shifted[:, 1] * np.roll(shifted[:, 0], -1)))
    if abs(area2) <= EPS:
        return min(point_segment_distance(q, vertices[i], vertices[(i + 1) % len(vertices)])
                   for i in range(len(vertices))) <= tol
    return all(cross(vertices[(i + 1) % len(vertices)] - vertices[i], q - vertices[i]) >= -tol
               for i in range(len(vertices)))


def point_segment_distance(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    p, x, y = map(lambda z: np.asarray(z, dtype=float), (point, a, b))
    d = y - x
    if float(np.dot(d, d)) <= EPS:
        return float(np.linalg.norm(p - x))
    t = min(1.0, max(0.0, float(np.dot(p - x, d) / np.dot(d, d))))
    return float(np.linalg.norm(p - (x + t * d)))


def point_polygon_distance(point: Sequence[float], poly: np.ndarray) -> float:
    if len(poly) == 0:
        return math.inf
    if point_in_convex_polygon(point, poly):
        return 0.0
    return min(point_segment_distance(point, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def q3_cover_points(ring_radius: float = 1150.0) -> np.ndarray:
    """Seven-point cover of the radius-1800 arena by radius-1000 disks."""
    points = [np.zeros(2)]
    points.extend(ring_radius * unit(k * math.pi / 3.0) for k in range(6))
    return np.asarray(points, dtype=float)


def q3_ring_cover_points(
    target_radius: float = 1800.0,
    min_receive_radius: float = 1000.0,
    sides: int = 12,
    boundary_margin: float = 2.0,
) -> np.ndarray:
    """Center plus a regular ring with a continuous disk-cover certificate.

    The center covers source radii through ``min_receive_radius``. In each
    outer angular sector, distance to the nearest ring point is maximized at
    the arena boundary and sector bisector. The closed-form radius below
    places that worst point strictly inside the minimum reception disk.
    """
    if sides < 3:
        raise ValueError("sides must be at least 3")
    if not (0.0 <= boundary_margin < min_receive_radius):
        raise ValueError("boundary_margin must lie inside the reception radius")
    half_step = math.pi / sides
    discriminant = min_receive_radius**2 - target_radius**2 * math.sin(half_step) ** 2
    if discriminant <= 0.0:
        raise ValueError("too few ring points to cover the target boundary")
    ring_radius = (
        target_radius * math.cos(half_step)
        - math.sqrt(discriminant)
        + boundary_margin
    )
    boundary_gap = math.sqrt(
        target_radius**2
        + ring_radius**2
        - 2.0 * target_radius * ring_radius * math.cos(half_step)
    )
    if boundary_gap >= min_receive_radius:
        raise ValueError("ring construction does not cover the target boundary")
    points = [np.zeros(2)]
    points.extend(ring_radius * unit(2.0 * math.pi * k / sides) for k in range(sides))
    return np.asarray(points, dtype=float)


def q3_custom_ring_points(
    ring_radius: float,
    sides: int,
    include_center: bool = True,
) -> np.ndarray:
    """Construct a regular Q3 ring for time-objective screening.

    Coverage is audited by the experiment driver; the online policy still
    rejects configurations that fail its explicit geometric audit.
    """
    if sides < 3 or ring_radius <= 0.0:
        raise ValueError("invalid custom ring")
    points = []
    if include_center:
        points.append(np.zeros(2))
    points.extend(
        ring_radius * unit(2.0 * math.pi * k / sides)
        for k in range(sides)
    )
    return np.asarray(points, dtype=float)


def q3_minimax_ring_points(
    target_radius: float = 1800.0,
    min_receive_radius: float = 1000.0,
    sides: int = 7,
) -> np.ndarray:
    """Regular ring minimizing the worst center-or-boundary cover distance."""
    if sides < 3:
        raise ValueError("sides must be at least 3")
    half_step = math.pi / sides
    ring_radius = target_radius / (2.0 * math.cos(half_step))
    boundary_gap = math.sqrt(
        target_radius**2
        + ring_radius**2
        - 2.0 * target_radius * ring_radius * math.cos(half_step)
    )
    if max(ring_radius, boundary_gap) >= min_receive_radius:
        raise ValueError("ring has too few points to cover the target disk")
    return np.asarray(
        [ring_radius * unit(2.0 * math.pi * k / sides) for k in range(sides)],
        dtype=float,
    )


def _triangle_intersects_origin_disk(points: list[np.ndarray], radius: float) -> bool:
    if any(float(np.linalg.norm(p)) <= radius + 1e-9 for p in points):
        return True
    if any(point_segment_distance((0.0, 0.0), points[i], points[(i + 1) % 3]) <= radius + 1e-9 for i in range(3)):
        return True
    signs = [cross(points[(i + 1) % 3] - points[i], -points[i]) for i in range(3)]
    return all(v >= -1e-9 for v in signs) or all(v <= 1e-9 for v in signs)


def q4_triangular_cover_points(target_radius: float = 1800.0, edge: float = 980.0) -> np.ndarray:
    """Vertices of every triangular-lattice cell intersecting the arena.

    Each retained triangle has side length ``edge < 1000``.  A source in that
    triangle is within 1000 m of all three vertices and is in their convex
    hull; consequently every closed directional half-plane through the source
    contains at least one detectable vertex.
    """
    if not (0.0 < edge < 1000.0):
        raise ValueError("edge must be strictly between 0 and 1000 m")
    vertical = math.sqrt(3.0) * edge / 2.0
    span = int(math.ceil((target_radius + edge) / min(edge, vertical))) + 3

    def lattice(i: int, j: int) -> np.ndarray:
        return np.array([edge * (i + 0.5 * j), vertical * j], dtype=float)

    used: set[tuple[int, int]] = set()
    for i in range(-2 * span, 2 * span + 1):
        for j in range(-2 * span, 2 * span + 1):
            triangles = [
                ((i, j), (i + 1, j), (i, j + 1)),
                ((i + 1, j + 1), (i + 1, j), (i, j + 1)),
            ]
            for triangle in triangles:
                vertices = [lattice(*index) for index in triangle]
                if _triangle_intersects_origin_disk(vertices, target_radius):
                    used.update(triangle)
    return np.asarray([lattice(i, j) for i, j in sorted(used, key=lambda z: (z[1], z[0]))])


def q4_certified21_cover_points(
    target_radius: float = 1800.0,
    min_receive_radius: float = 1000.0,
    boundary_margin_fraction: float = 0.00025,
) -> np.ndarray:
    """Return the continuously certified 21-point Q4 discovery cover.

    For the official ratio ``target_radius / min_receive_radius <= 1.8``, the
    origin, an eight-point ring at 0.996 of the reception radius, and a
    circumscribed twelve-point boundary ring positively span every source in
    the target disk using points within 0.997 of the reception radius.  Hence
    every closed 180-degree emitting half-plane contains a detectable point.
    The 0.003 reception margin absorbs the numerical and boundary tolerances.
    """
    if min_receive_radius <= 0.0:
        raise ValueError("min_receive_radius must be positive")
    if target_radius <= 0.0:
        raise ValueError("target_radius must be positive")
    if target_radius > 1.8 * min_receive_radius + 1e-9:
        raise ValueError(
            "certified 21-point cover requires target_radius / "
            "min_receive_radius <= 1.8"
        )
    if boundary_margin_fraction < 0.0:
        raise ValueError("boundary_margin_fraction must be nonnegative")

    inner_radius = 0.996 * min_receive_radius
    boundary_margin = boundary_margin_fraction * min_receive_radius
    outer_radius = (target_radius + boundary_margin) / math.cos(math.pi / 12.0)
    points = [np.zeros(2)]
    points.extend(
        inner_radius * unit(2.0 * math.pi * k / 8.0)
        for k in range(8)
    )
    points.extend(
        outer_radius * unit(2.0 * math.pi * k / 12.0)
        for k in range(12)
    )
    return np.asarray(points, dtype=float)


def q4_compact_cover_points(
    target_radius: float = 1800.0,
    min_receive_radius: float = 1000.0,
    sides: int = 12,
    inner_radius: float = 930.0,
    boundary_margin: float = 1.0,
) -> np.ndarray:
    """Return a compact triangulation that guarantees directional discovery.

    The outer regular polygon circumscribes the target disk. Alternating inner
    and outer vertices, together with the origin, triangulate that polygon.
    Every edge in the default construction is shorter than the guaranteed
    reception radius. Thus every source is positively spanned by three nearby
    scan points, so every directional 180-degree half-plane contains one.
    """
    if sides < 3:
        raise ValueError("sides must be at least 3")
    if min_receive_radius <= 0.0:
        raise ValueError("min_receive_radius must be positive")
    if not (0.0 < inner_radius < min_receive_radius):
        raise ValueError("inner_radius must lie inside the reception radius")
    if boundary_margin < 0.0:
        raise ValueError("boundary_margin must be nonnegative")

    half_step = math.pi / sides
    outer_radius = (target_radius + boundary_margin) / math.cos(half_step)
    inner_edge = 2.0 * inner_radius * math.sin(half_step)
    outer_edge = 2.0 * outer_radius * math.sin(half_step)
    bridge = math.sqrt(
        inner_radius * inner_radius
        + outer_radius * outer_radius
        - 2.0 * inner_radius * outer_radius * math.cos(half_step)
    )
    longest_edge = max(inner_radius, inner_edge, outer_edge, bridge)
    if longest_edge >= min_receive_radius:
        raise ValueError(
            "compact cover is not guaranteed: longest triangle edge "
            f"{longest_edge:.6f} m must be below {min_receive_radius:.6f} m"
        )

    points = [np.zeros(2)]
    points.extend(inner_radius * unit(2.0 * math.pi * k / sides) for k in range(sides))
    points.extend(
        outer_radius * unit(2.0 * math.pi * (k + 0.5) / sides)
        for k in range(sides)
    )
    return np.asarray(points, dtype=float)


def q4_compact_cover_triangles(
    target_radius: float = 1800.0,
    min_receive_radius: float = 1000.0,
    sides: int = 12,
    inner_radius: float = 930.0,
    boundary_margin: float = 1.0,
) -> list[np.ndarray]:
    """Return the short-edge triangles underlying the compact Q4 cover."""
    points = q4_compact_cover_points(
        target_radius,
        min_receive_radius,
        sides,
        inner_radius,
        boundary_margin,
    )
    center = 0
    inner_start = 1
    outer_start = 1 + sides
    triangles = []
    for index in range(sides):
        inner = inner_start + index
        inner_next = inner_start + (index + 1) % sides
        outer = outer_start + index
        outer_previous = outer_start + (index - 1) % sides
        triangles.extend(
            (
                points[[center, inner, inner_next]],
                points[[inner, inner_next, outer]],
                points[[inner, outer_previous, outer]],
            )
        )
    return triangles


def _source_samples(region: np.ndarray, first: np.ndarray) -> np.ndarray:
    if len(region) == 0:
        return region
    samples = [p for p in region]
    samples.extend((region[i] + region[(i + 1) % len(region)]) / 2.0 for i in range(len(region)))
    samples.append(np.mean(region, axis=0))
    return np.asarray([p for p in samples if np.linalg.norm(p - first) > 5.0 + 1e-6])


def second_point_candidate_region(
    first: Sequence[float],
    theta: float,
    target_radius: float = 1800.0,
    error_deg: float = 1.0,
    min_receive_radius: float = 1000.0,
    max_receive_radius: float = 1500.0,
    grid_step: float = 40.0,
    min_cross_angle_deg: float = 30.0,
    min_separation: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the robust Q2 candidate set and a quality score for each point.

    A candidate must receive the source for every location compatible with the
    first reading (using the guaranteed 1000 m radius), and its worst sampled
    crossing angle must exceed ``min_cross_angle_deg``.  Scores reward angular
    conditioning and mildly penalize travel.
    """
    p = np.asarray(first, dtype=float)
    region = bearing_region(
        [(p, theta)],
        target_radius=target_radius,
        error_deg=error_deg,
        extra_disks=[(p, max_receive_radius)],
    )
    if len(region) == 0:
        return region, np.empty((0, 2)), np.empty(0)
    samples = _source_samples(region, p)
    lower = np.min(region, axis=0) - min_receive_radius
    upper = np.max(region, axis=0) + min_receive_radius
    candidates: list[np.ndarray] = []
    scores: list[float] = []
    for x in np.arange(lower[0], upper[0] + 0.5 * grid_step, grid_step):
        for y in np.arange(lower[1], upper[1] + 0.5 * grid_step, grid_step):
            q = np.array([x, y], dtype=float)
            travel = float(np.linalg.norm(q - p))
            if travel < min_separation:
                continue
            if float(np.max(np.linalg.norm(region - q, axis=1))) > min_receive_radius - 1.0:
                continue
            angles = []
            for source in samples:
                a, b = source - p, source - q
                den = float(np.linalg.norm(a) * np.linalg.norm(b))
                if den <= EPS:
                    angles.append(math.pi / 2.0)
                else:
                    angles.append(math.asin(min(1.0, abs(cross(a, b)) / den)))
            worst = min(angles, default=0.0)
            if worst + 1e-12 < math.radians(min_cross_angle_deg):
                continue
            candidates.append(q)
            scores.append(math.degrees(worst) - 0.002 * travel)
    return region, np.asarray(candidates), np.asarray(scores)


def optimal_second_point(
    first: Sequence[float], theta: float, target_radius: float = 1800.0, grid_step: float = 40.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the best robust second point and all feasible Q2 candidates."""
    _, candidates, scores = second_point_candidate_region(first, theta, target_radius, grid_step=grid_step)
    if len(candidates) == 0:
        raise ValueError("no robust second detection point exists for this reading")
    return candidates[int(np.argmax(scores))], candidates


if __name__ == "__main__":
    obs = [((0.0, 0.0), math.radians(20)), ((300.0, 0.0), math.radians(100))]
    p, c, d = robust_estimate(obs)
    print(len(p), c.center, c.radius, d)
