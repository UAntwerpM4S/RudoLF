

"""
Source: https://www.samgongustofa.is/media/log-og-reglur/COLREG-Consolidated-2018.pdf
Scenarios:
  0  "path-following"   – no dynamic TS within range
  1  "head-on"          – Rule 14
  2  "overtaking"       – Rule 13
  3  "crossing-right"   – Rule 15, 16, TS on starboard (OS give-way)
  4  "crossing-left"    – Rule 15, 17, TS on port (OS stand-on) --> Omitted, because desired behavior is "path-following"
"""

import math
import numpy as np
from typing import Optional

SCENARIO_LABELS = ["path-following", "head-on", "overtaking", "crossing-right"]
NUM_SCENARIOS = len(SCENARIO_LABELS)


def _normalize_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class ScenarioClassifier:
    def __init__(
        self,
        classify_range: float = 100.0,
        forget_range: float = 110.0,
        head_on_bearing_tol: float = math.radians(15.0),
        head_on_heading_tol: float = math.radians(15.0),
        overtaking_abaft_angle: float = math.radians(112.5),
    ):
        self.classify_range = classify_range
        self.forget_range = forget_range
        self.head_on_bearing_tol = head_on_bearing_tol
        self.head_on_heading_tol = head_on_heading_tol
        self.overtaking_abaft_angle = overtaking_abaft_angle

        self._tracked_ts: Optional[int] = None
        self._scenario: int = 0
        self._overtaking_locked: bool = False
        self._headon_locked: bool = False
        self._crossing_right_locked: bool = False

    def reset(self) -> None:
        self._tracked_ts = None
        self._scenario = 0
        self._overtaking_locked = False
        self._headon_locked = False
        self._crossing_right_locked = False

    def classify(self, os_pos: np.ndarray, os_heading: float, collision_entities: list) -> int:
        """Classify the encounter with the nearest dynamic TS. Returns scenario index."""
        ts_idx = self._select_ts(os_pos, collision_entities)

        if ts_idx is None:
            self._tracked_ts = None
            self._scenario = 0
            self._overtaking_locked = False
            self._headon_locked = False
            #self._crossing_right_locked = False
            return 0

        self._tracked_ts = ts_idx
        ts = collision_entities[ts_idx]

        if self._overtaking_locked:
            self._scenario = 2
            return 2

        if self._headon_locked:
            self._scenario = 1
            return 1

        if self._crossing_right_locked:
            self._scenario = 3
            return 3

        self._scenario = self._classify_geometry(os_pos, os_heading, ts.get_position(), ts.get_heading())

        if self._scenario == 2:
            self._overtaking_locked = True

        if self._scenario == 1:
            self._headon_locked = True

        if self._scenario == 3:
            self._crossing_right_locked = True


        return self._scenario

    def get_one_hot(self) -> np.ndarray:
        vec = np.zeros(NUM_SCENARIOS, dtype=np.float32)
        vec[self._scenario] = 1.0
        return vec

    @property
    def label(self) -> str:
        return SCENARIO_LABELS[self._scenario]

    @property
    def tracked_ts(self) -> Optional[int]:
        return self._tracked_ts

    def _select_ts(self, os_pos: np.ndarray, collision_entities: list) -> Optional[int]:
        """
        Pick nearest dynamic TS within range, with hysteresis on the tracked one.
        Hysteresis makes sure that TS don't oscillate in and out of classification range
        """
        if not collision_entities:
            return None

        # Keep current if still in forget range
        if self._tracked_ts is not None and self._tracked_ts < len(collision_entities):
            ts = collision_entities[self._tracked_ts]
            if ts.is_dynamic and ts.distance_to_point(os_pos) <= self.forget_range:
                return self._tracked_ts

        # Find nearest dynamic TS within classify range
        best_idx, best_dist = None, self.classify_range
        for i, e in enumerate(collision_entities):
            if not e.is_dynamic:
                continue
            d = e.distance_to_point(os_pos)
            if d <= best_dist:
                best_dist = d
                best_idx = i

        return best_idx

    def _classify_geometry(self, os_pos, os_heading, ts_pos, ts_heading) -> int:
        """
        Classifies the TS based on relative bearing and heading difference
        """
        dx = float(ts_pos[0] - os_pos[0])
        dy = float(ts_pos[1] - os_pos[1])

        # Relative bearing of TS from OS
        rel_bearing_os = _normalize_angle(math.atan2(dy, dx) - os_heading)

        # Heading difference
        delta_heading = abs(_normalize_angle(ts_heading - os_heading))

        # Bearing of OS from TS (for overtaking: is OS abaft TS's beam?)
        rel_bearing_ts = _normalize_angle(math.atan2(-dy, -dx) - ts_heading)

        # Head-on: TS nearly ahead, courses nearly reciprocal
        if abs(rel_bearing_os) < self.head_on_bearing_tol and delta_heading > (math.pi - self.head_on_heading_tol):
            return 1

        # Overtaking: OS is abaft TS's beam
        if abs(rel_bearing_ts) > self.overtaking_abaft_angle:
            return 2

        # Crossing
        return 3 if rel_bearing_os < 0 else 0