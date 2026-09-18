"""Deterministic offline simulator matching the official B-question rules.

This is not used to claim official scores.  It is a regression harness for the
robot policy before connecting to the supplied simulator.
"""
from __future__ import annotations

import json
import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from geometry import bearing, wrap_pi


ARENA_RADIUS = 1800.0
SPEED = 5.0


@dataclass
class Source:
    channel: int
    x: float
    y: float
    receive_radius: float
    directional_angle: float | None = None
    cleared: bool = False

    @property
    def position(self):
        return (self.x, self.y)

    @property
    def directional(self):
        return self.directional_angle is not None


class JammerWorld:
    def __init__(self, sources: list[Source], seed: int = 0, error_deg: float = 1.0):
        self.sources = sources
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self.error_deg = error_deg
        self.position = (0.0, 0.0)
        self.channel = 1
        self.virtual_time = 0.0
        self.entered = False
        self.events: list[dict[str, Any]] = []

    @classmethod
    def random_case(
        cls,
        seed: int,
        problem: int = 3,
        directional_probability: float | None = None,
        count: int | None = None,
        receive_radius_range: tuple[float, float] = (1000.0, 1500.0),
        boundary_bias: bool = False,
    ) -> "JammerWorld":
        rng = random.Random(seed)
        count = rng.randint(10, 16) if count is None else int(count)
        if not 10 <= count <= 16:
            raise ValueError("source count must be in [10, 16]")
        channels = rng.sample(range(1, 21), count)
        directional_probability = (0.45 if problem == 4 else 0.0) if directional_probability is None else directional_probability
        sources = []
        for ch in channels:
            radial_draw = rng.random() ** (0.15 if boundary_bias else 0.5)
            r = radial_draw * ARENA_RADIUS
            a = rng.random() * 2 * math.pi
            x, y = r * math.cos(a), r * math.sin(a)
            recv = rng.uniform(*receive_radius_range)
            direction = None
            if problem == 4 and rng.random() < directional_probability:
                direction = rng.random() * 2 * math.pi
            sources.append(Source(ch, x, y, recv, direction))
        return cls(sources, seed=seed)

    def _record(self, path: str, request: dict[str, Any], result: dict[str, Any], dt: float):
        self.events.append({"path": path, "request": request, "response": result, "dt": dt})

    def _base(self, request_id: str):
        return {"accepted": True, "virtual_time_s": self.virtual_time, "request_id": request_id}

    def enter(self, request_id: str = "enter-1"):
        if self.entered:
            return {"accepted": False, "virtual_time_s": 0.0}
        self.entered = True
        result = self._base(request_id)
        result.update({"max_virtual_duration_s": 360000.0, "max_real_duration_s": 1200, "remaining_real_duration_s": 1200})
        self._record("/enter", {"request_id": request_id}, result, 0.0)
        return result

    def _move(self, position):
        x, y = float(position["x"]), float(position["y"])
        d = math.hypot(x - self.position[0], y - self.position[1])
        self.position = (x, y)
        return d / SPEED

    def _source_visible(self, source: Source, position):
        d = math.hypot(position["x"] - source.x, position["y"] - source.y)
        if d > source.receive_radius + 1e-9:
            return False, d
        if source.directional:
            if d <= 1e-12:
                return True, d
            from_source = math.atan2(position["y"] - source.y, position["x"] - source.x)
            if abs(wrap_pi(from_source - source.directional_angle)) > math.pi / 2 + 1e-12:
                return False, d
        return True, d

    def _fixed_error_deg(self, source: Source, position: dict[str, float]) -> float:
        """Stable location-specific error, independent of Python hash randomization."""
        key = (
            f"{self.seed}|{source.channel}|{float(position['x']):.6f}|"
            f"{float(position['y']):.6f}"
        ).encode("ascii")
        raw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        unit_value = raw / float((1 << 64) - 1)
        return (2.0 * unit_value - 1.0) * self.error_deg

    def measure(self, position: dict[str, float], channel: int, request_id: str):
        if not self.entered:
            return {"accepted": False, "virtual_time_s": 0.0}
        move_dt = self._move(position)
        switch_dt = 1.0 if int(channel) != self.channel else 0.0
        self.channel = int(channel)
        dt = move_dt + switch_dt + 5.0
        self.virtual_time += dt
        source = next((s for s in self.sources if s.channel == int(channel) and not s.cleared), None)
        result = "no_signal"
        response = self._base(request_id)
        if source is not None:
            visible, d = self._source_visible(source, position)
            if visible:
                if d <= 5.0:
                    result = "near"
                else:
                    result = "direction"
                    # A fixed error at a fixed position, as stated in the task.
                    e = self._fixed_error_deg(source, position)
                    true = math.atan2(source.y - float(position["y"]), source.x - float(position["x"]))
                    response["svd_deg"] = round((math.degrees(true + math.radians(e))) % 360.0, 2)
        response["measure_result"] = result
        self._record("/measure", {"position": position, "channel": channel, "request_id": request_id}, response, dt)
        return response

    def clear(self, position: dict[str, float], channel: int, request_id: str):
        if not self.entered:
            return {"accepted": False, "virtual_time_s": 0.0}
        move_dt = self._move(position)
        source = next((s for s in self.sources if s.channel == int(channel) and not s.cleared), None)
        ok = source is not None and math.hypot(position["x"] - source.x, position["y"] - source.y) <= 20.0 + 1e-9
        action_dt = 5.0 if ok else 3.0
        if ok:
            source.cleared = True
        dt = move_dt + action_dt
        self.virtual_time += dt
        response = self._base(request_id)
        response["clear_result"] = "success" if ok else "no_target_in_range"
        self._record("/clear", {"position": position, "channel": channel, "request_id": request_id}, response, dt)
        return response

    def exit(self, request_id: str = "exit-1"):
        response = self._base(request_id)
        response["exit_reason"] = "user_exit"
        self._record("/exit", {"request_id": request_id}, response, 0.0)
        self.entered = False
        return response

    @property
    def cleared_count(self):
        return sum(s.cleared for s in self.sources)

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"sources": [asdict(s) for s in self.sources], "events": self.events}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    w = JammerWorld.random_case(42, 4)
    print(len(w.sources), sum(s.directional for s in w.sources))
