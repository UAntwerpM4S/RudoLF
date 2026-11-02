"""
COLREG encounter classification for the own ship (OS).

Source: https://www.samgongustofa.is/media/log-og-reglur/COLREG-Consolidated-2018.pdf
Scenarios:
  0  "path-following"   no dynamic TS within range
  1  "head-on"          Rule 14
  2  "overtaking"       Rule 13
  3  "crossing-right"   Rules 15, 16: TS on starboard (OS give-way)
  4  "crossing-left"    Rules 15, 17: TS on port (OS stand-on); metrics only, the
                        policy treats it as path-following
"""

import math
import numpy as np

from typing import Optional

SCENARIO_LABELS = ["path-following", "head-on", "overtaking", "crossing-right"]
NUM_SCENARIOS = len(SCENARIO_LABELS)
SCENARIO_INDEX = {label: i for i, label in enumerate(SCENARIO_LABELS)}

# Extended 5-way labels for metrics only (policy and reward use the 4-way space);
# they make crossing-left measurable, which the reduced space folds into path-following.
EXTENDED_SCENARIO_LABELS = ["path-following", "head-on", "overtaking", "crossing-right", "crossing-left"]
EXTENDED_SCENARIO_INDEX = {label: i for i, label in enumerate(EXTENDED_SCENARIO_LABELS)}

# Builder label to reduced label. "static" (no dynamic TS) and "crossing-left"
# (OS is stand-on: keep course and speed) both map to path-following.
_BUILDER_TO_CLASSIFIER = {
    "path-following": "path-following",
    "static": "path-following",
    "head-on": "head-on",
    "overtaking": "overtaking",
    "crossing-right": "crossing-right",
    "crossing-left": "path-following",
}

# Builder label to extended label; keeps crossing-left for confusion matrices.
_BUILDER_TO_EXTENDED = {
    "path-following": "path-following",
    "static": "path-following",
    "head-on": "head-on",
    "overtaking": "overtaking",
    "crossing-right": "crossing-right",
    "crossing-left": "crossing-left",
}


def _normalize_angle(angle: float) -> float:
    """
    Normalizes an angle to the interval [-π, π).

    Args:
        angle: Angle in radians.

    Returns:
        The normalized angle in radians.
    """

    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class ScenarioClassifier:
    """
    Classifies COLREG encounter scenarios.

    Tracks the nearest dynamic target ship, applies geometric encounter
    classification with hysteresis, and optionally overrides the result with
    ground-truth scenario labels provided by the scenario builder.
    """

    def __init__(
        self,
        classify_range: float = 750.0,
        forget_range: float = 770.0,
    ):
        """
        Initializes the scenario classifier.

        Args:
            classify_range: Maximum distance for selecting a target ship.
            forget_range: Distance beyond which the currently tracked target ship
                is forgotten.
        """

        self.classify_range = classify_range
        self.forget_range = forget_range

        self._tracked_ts: Optional[int] = None
        self._scenario: int = 0
        self._overtaking_locked: bool = False
        self._headon_locked: bool = False
        self._crossing_right_locked: bool = False
        # Ground-truth lock (ground_truth_classification=True): return the
        # builder's spawn-time scenario instead of the geometric result, so a
        # misclassification cannot corrupt reward shaping.
        self._episode_scenario_label: Optional[str] = None

        # Last geometric classification output
        self._last_geometric_scenario: int = 0            # 4-way index
        self._last_geometric_extended: int = 0            # 5-way index

        # Last collision_entities list, used to look up the tracked TS's spawn label.
        self._last_entities: Optional[list] = None

        # Path geometry from set_path_segments(), used by classify_path_projected()
        # to avoid curvature-induced misclassification.
        self._seg_starts: Optional[np.ndarray] = None
        self._seg_tangents: Optional[np.ndarray] = None
        self._seg_lengths: Optional[np.ndarray] = None

    def set_path_segments(
        self,
        seg_starts: np.ndarray,
        seg_tangents: np.ndarray,
        seg_lengths: np.ndarray,
    ) -> None:
        """
        Register the path segment geometry used by classify_path_projected.

        Expected shapes: seg_starts (N, 2), seg_tangents (N, 2) unit vectors,
        seg_lengths (N,).
        """

        self._seg_starts = np.asarray(seg_starts, dtype=np.float32)
        self._seg_tangents = np.asarray(seg_tangents, dtype=np.float32)
        self._seg_lengths = np.asarray(seg_lengths, dtype=np.float32)

    def reset(self) -> None:
        """
        Resets the classifier state for a new episode.
        """

        self._tracked_ts = None
        self._scenario = 0
        self._overtaking_locked = False
        self._headon_locked = False
        self._crossing_right_locked = False
        self._episode_scenario_label = None
        self._last_geometric_scenario = 0
        self._last_geometric_extended = 0
        self._last_entities = None

    def set_episode_scenario(self, scenario_label: Optional[str]) -> None:
        """
        Lock the classifier to a ground-truth scenario label (builder label set).

        While a TS is tracked, classification then returns this scenario instead
        of the geometric result. Pass None to release the lock.
        """

        if scenario_label is None:
            self._episode_scenario_label = None
            return

        mapped = _BUILDER_TO_CLASSIFIER.get(scenario_label)
        if mapped is None:
            raise ValueError(
                f"Unknown scenario label '{scenario_label}'. "
                f"Expected one of {list(_BUILDER_TO_CLASSIFIER.keys())}."
            )
        self._episode_scenario_label = mapped
        self._scenario = SCENARIO_INDEX[mapped]

    def classify_path_projected(
        self,
        os_pos: np.ndarray,
        os_heading: float,
        collision_entities: list,
        approach_offset: float = 20.0,
    ) -> int:
        """
        Path-aware classification.

        Replaces the OS heading and position by the heading of the path segment
        nearest the tracked TS and a virtual OS on that segment, so path
        curvature does not skew the result (head-on always gives delta=180°,
        crossings 90°). Falls back to raw geometry without set_path_segments().

        Returns:
            Reduced 4-way scenario index.
        """

        self._last_entities = collision_entities

        ts_idx = self._select_ts(os_pos, collision_entities)
        self._tracked_ts = ts_idx

        if ts_idx is None:
            return self._finalize_scenario(None)

        ts = collision_entities[ts_idx]
        ts_pos = np.asarray(ts.get_position(), dtype=np.float32)
        ts_heading = ts.get_heading()

        seg_idx, foot = self._find_nearest_segment(ts_pos)
        if seg_idx is None:
            ext = self._classify_geometry_extended(os_pos, os_heading, ts_pos, ts_heading)
        else:
            tangent = self._seg_tangents[seg_idx]
            virtual_heading = float(math.atan2(float(tangent[1]), float(tangent[0])))
            # Virtual OS approach_offset behind the TS's foot point, so the side
            # check uses the TS's path-relative side.
            virtual_os_pos = foot - tangent * float(approach_offset)
            ext = self._classify_geometry_extended(
                virtual_os_pos, virtual_heading, ts_pos, ts_heading
            )

        return self._finalize_scenario(ext)

    def _finalize_scenario(self, ext: Optional[int]) -> int:
        """
        Apply the no-TS, GT-lock and hysteresis logic to an extended geometric
        label (None if no TS is tracked) and return the reduced index.
        """

        # No TS in range gives path-following, even under a GT lock; otherwise
        # the approach phase would be routed to the wrong policy.
        if ext is None:
            self._last_geometric_scenario = 0
            self._last_geometric_extended = 0
            self._scenario = 0
            self._overtaking_locked = False
            self._headon_locked = False
            self._crossing_right_locked = False
            return 0

        self._last_geometric_extended = ext
        reduced_label = _BUILDER_TO_CLASSIFIER[EXTENDED_SCENARIO_LABELS[ext]]
        self._last_geometric_scenario = SCENARIO_INDEX[reduced_label]

        # GT lock: return the pre-committed label.
        if self._episode_scenario_label is not None:
            self._scenario = SCENARIO_INDEX[self._episode_scenario_label]
            return self._scenario

        # Geometric mode: once detected, a scenario stays locked until the TS is lost.
        if self._overtaking_locked:
            self._scenario = 2
            return 2

        if self._headon_locked:
            self._scenario = 1
            return 1

        if self._crossing_right_locked:
            self._scenario = 3
            return 3

        self._scenario = self._last_geometric_scenario

        if self._scenario == 2:
            self._overtaking_locked = True

        if self._scenario == 1:
            self._headon_locked = True

        if self._scenario == 3:
            self._crossing_right_locked = True

        return self._scenario

    def _find_nearest_segment(self, ts_pos: np.ndarray):
        """
        Return (seg_idx, foot_point) of the segment whose clamped foot point is
        nearest to ts_pos, or (None, None) without a registered path.
        """

        if self._seg_starts is None or len(self._seg_starts) == 0:
            return None, None

        ts_pos = np.asarray(ts_pos, dtype=np.float32)
        rel = ts_pos - self._seg_starts                          # (N, 2)
        t_proj = np.sum(rel * self._seg_tangents, axis=1)        # (N,)
        t_clamped = np.clip(t_proj, 0.0, self._seg_lengths)
        feet = self._seg_starts + self._seg_tangents * t_clamped[:, None]
        dists = np.linalg.norm(ts_pos - feet, axis=1)
        best = int(np.argmin(dists))
        return best, feet[best]

    def get_one_hot(self) -> np.ndarray:
        """
        Returns the current scenario as a one-hot vector.

        Returns:
            One-hot encoded scenario vector.
        """

        vec = np.zeros(NUM_SCENARIOS, dtype=np.float32)
        vec[self._scenario] = 1.0
        return vec

    @property
    def label(self) -> str:
        """
        Current reduced scenario label.
        """

        return SCENARIO_LABELS[self._scenario]

    @property
    def tracked_ts(self) -> Optional[int]:
        """
        Index of the currently tracked target ship, if any.
        """

        return self._tracked_ts

    @property
    def geometric_extended_label(self) -> str:
        """
        5-way extended geometric classification (distinguishes crossing-left).
        """

        return EXTENDED_SCENARIO_LABELS[self._last_geometric_extended]

    @property
    def ground_truth_extended_label(self) -> Optional[str]:
        """
        5-way GT label from the tracked TS's spawn_scenario_label (diagnostics).
        """

        if self._tracked_ts is None or self._last_entities is None:
            return None
        if self._tracked_ts >= len(self._last_entities):
            return None
        ts = self._last_entities[self._tracked_ts]
        raw = getattr(ts, "spawn_scenario_label", None)
        if raw is None:
            return None
        return _BUILDER_TO_EXTENDED.get(raw)

    def _select_ts(self, os_pos: np.ndarray, collision_entities: list) -> Optional[int]:
        """
        Pick the nearest dynamic TS within classify_range. The tracked TS is kept
        while within forget_range, so it does not flicker in and out of range.
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

    @staticmethod
    def _classify_geometry_extended(os_pos, os_heading: float, ts_pos, ts_heading: float) -> int:
        """
        Classify a tracked TS by heading difference, folded to [0, pi] and
        matched to the nearest canonical angle (boundaries at pi/4, 3*pi/4):

          * pi:   head-on (reciprocal courses)
          * 0:    overtaking (aligned courses)
          * pi/2: crossing; right or left depending on the side of OS the TS is on

        Never returns path-following (a TS is tracked). Returns the extended
        5-way index; the caller reduces it to the 4-way space.
        """

        # Heading difference, folded to [0, pi].
        delta_heading = abs(_normalize_angle(ts_heading - os_heading))

        dist_reciprocal    = abs(delta_heading - math.pi)
        dist_aligned       = delta_heading
        dist_perpendicular = abs(delta_heading - math.pi / 2.0)

        closest = min(dist_reciprocal, dist_aligned, dist_perpendicular)

        if closest == dist_reciprocal:
            return EXTENDED_SCENARIO_INDEX["head-on"]

        if closest == dist_aligned:
            return EXTENDED_SCENARIO_INDEX["overtaking"]

        # Crossing: rel_bearing_os is CCW-positive, so negative means starboard
        # (crossing-right), positive means port (crossing-left).
        dx = float(ts_pos[0] - os_pos[0])
        dy = float(ts_pos[1] - os_pos[1])
        rel_bearing_os = _normalize_angle(math.atan2(dy, dx) - os_heading)
        if rel_bearing_os < 0:
            return EXTENDED_SCENARIO_INDEX["crossing-right"]
        return EXTENDED_SCENARIO_INDEX["crossing-left"]
