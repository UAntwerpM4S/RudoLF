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
SCENARIO_INDEX = {label: i for i, label in enumerate(SCENARIO_LABELS)}

# Extended labels used purely for metrics/diagnostics. The policy and reward
# shaping use the reduced 4-way space above
# The extended space lets us *measure* how often the geometric classifier's
# port-side crossing branch fires (which is otherwise indistinguishable from
# a genuine path-following classification in the reduced space).
EXTENDED_SCENARIO_LABELS = ["path-following", "head-on", "overtaking", "crossing-right", "crossing-left"]
EXTENDED_SCENARIO_INDEX = {label: i for i, label in enumerate(EXTENDED_SCENARIO_LABELS)}

# Map the scenario_builder's full label set onto the classifier's reduced
# label set. "static" means no dynamic TS is present, so the agent should
# simply follow its path. "crossing-left" is treated as path-following as well
# because OS is the stand-on vessel and the desired behavior is to maintain
# course and speed.
_BUILDER_TO_CLASSIFIER = {
    "path-following": "path-following",
    "static": "path-following",
    "head-on": "head-on",
    "overtaking": "overtaking",
    "crossing-right": "crossing-right",
    "crossing-left": "path-following",
}

# Builder-label -> extended-metric-label. Unlike the reduced mapping above,
# this preserves crossing-left as a distinct class so confusion matrices
# can show e.g. head-on <-> crossing-left mix-ups.
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
        # When set, classification bypasses the geometric result for the
        # returned label and always returns this scenario. Used when
        # ground_truth_classification=True: the ScenarioBuilder commits to a
        # scenario at spawn time, so the classifier honors that choice rather
        # than risk misclassifying e.g. a head-on as crossing-right (which
        # would poison reward shaping during training).
        self._episode_scenario_label: Optional[str] = None

        # Last geometric classification output
        self._last_geometric_scenario: int = 0            # 4-way index
        self._last_geometric_extended: int = 0            # 5-way index

        # Cached reference to the last-classified collision_entities list, so
        # ground_truth_label can look up the tracked TS's spawn scenario.
        self._last_entities: Optional[list] = None

        # Path geometry (set once per env via set_path_segments). When
        # available, classify_path_projected() uses it to project OS onto
        # the path segment nearest the tracked TS before running the
        # geometric branch, eliminating curvature-induced misclassification.
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
        Lock the classifier to a ground-truth scenario.

        When a label is set, classification short-circuits the geometric
        result and always returns this scenario's index.

        Pass None to release the lock and resume geometric classification.
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
        Path-aware classification. Uses the heading of the path segment
        nearest the tracked TS in place of the OS heading, and a virtual
        OS position projected onto that segment, so curvature does not
        skew the heading-difference bucket or the port/starboard bearing
        check: a head-on scenario sees delta=180° regardless of where the
        OS is along an arc, crossings see delta=90°, etc.

        Falls back to raw-geometry classification if set_path_segments()
        has not been called.
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
            # Place the virtual OS one approach_offset behind the TS's foot
            # on the segment so the perpendicular bearing check reads the
            # TS's path-relative side rather than the OS's current side.
            virtual_os_pos = foot - tangent * float(approach_offset)
            ext = self._classify_geometry_extended(
                virtual_os_pos, virtual_heading, ts_pos, ts_heading
            )

        return self._finalize_scenario(ext)

    def _finalize_scenario(self, ext: Optional[int]) -> int:
        """
        Apply the ts-in-range, GT-lock, and hysteresis logic given the
        extended geometric label just computed (or None if no TS tracked).
        """

        # No TS in range -> path-following, regardless of any GT lock. The GT
        # lock must only override geometry during an encounter; otherwise the
        # agent would spend the approach phase routed to the wrong policy
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

        # GT lock: with a tracked TS, return the pre-committed label regardless
        # of geometry (prevents misclassification from poisoning reward shaping).
        if self._episode_scenario_label is not None:
            self._scenario = SCENARIO_INDEX[self._episode_scenario_label]
            return self._scenario

        # Geometric mode: honor scenario locks for hysteresis.
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
        Return (seg_idx, foot_point) for the path segment whose clamped
        foot-of-perpendicular to ts_pos is nearest. Returns (None, None) if
        no path is registered.
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
        5-way GT label from the tracked TS's spawn_scenario_label.

        Preserves crossing-left as a distinct class for diagnostics.
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

    @staticmethod
    def _classify_geometry_extended(os_pos, os_heading: float, ts_pos, ts_heading: float) -> int:
        """
        Classifies a tracked TS by *heading difference*. The difference is
        folded to [0, pi] and bucketed by which canonical angle it is
        nearest:

          * near pi   -> head-on   (reciprocal courses)
          * near 0    -> overtaking (aligned courses)
          * near pi/2 -> crossing  (perpendicular courses; left/right
                                    disambiguated by which side of OS
                                    the TS lies on — the one remaining
                                    use of relative bearing)

        Bucket boundaries fall naturally at pi/4 and 3*pi/4 from taking the
        nearest of the three canonical angles.

        This function never returns "path-following": the caller has already
        confirmed a TS is tracked. The "no TS in range" path-following case
        is handled in _finalize_scenario when ext is None.

        Returns an *extended* 5-way index. The reduced 4-way output
        (which collapses crossing-left to path-following per the Rule 17
        "stand-on" convention) is produced downstream via
        `_BUILDER_TO_CLASSIFIER[EXTENDED_SCENARIO_LABELS[ext]]`.
        """

        # Heading difference, folded to [0, pi].
        delta_heading = abs(_normalize_angle(ts_heading - os_heading))

        # Distance to each canonical heading-difference value.
        dist_reciprocal    = abs(delta_heading - math.pi)
        dist_aligned       = delta_heading
        dist_perpendicular = abs(delta_heading - math.pi / 2.0)

        closest = min(dist_reciprocal, dist_aligned, dist_perpendicular)

        if closest == dist_reciprocal:
            return EXTENDED_SCENARIO_INDEX["head-on"]

        if closest == dist_aligned:
            return EXTENDED_SCENARIO_INDEX["overtaking"]

        # Perpendicular bucket: TS on OS-starboard -> crossing-right,
        # TS on OS-port -> crossing-left. Math convention: rel_bearing_os
        # is CCW-positive in OS body frame, so negative means starboard.
        dx = float(ts_pos[0] - os_pos[0])
        dy = float(ts_pos[1] - os_pos[1])
        rel_bearing_os = _normalize_angle(math.atan2(dy, dx) - os_heading)
        if rel_bearing_os < 0:
            return EXTENDED_SCENARIO_INDEX["crossing-right"]
        return EXTENDED_SCENARIO_INDEX["crossing-left"]
