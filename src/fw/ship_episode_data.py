from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CollisionEvent:
    """Represents a proximity event between two ships at a specific timestep.

    A collision event is created when two ships come within a configured
    distance threshold. It may represent either a near-miss or a hard collision.

    Attributes:
        timestep: Time index at which the event occurred.
        ship_a: Identifier of the first ship.
        ship_b: Identifier of the second ship.
        distance: Euclidean distance between ships at the event timestep.
        is_collision: True if distance is below the collision threshold,
            otherwise False (near-miss).
    """

    timestep: int
    ship_a: str
    ship_b: str
    distance: float
    is_collision: bool  # True if distance < hard collision threshold

@dataclass
class ShipEpisodeData:
    """Stores all recorded data for a single ship over one episode.

    This structure contains time-series data collected during environment
    interaction, including trajectory, control actions, navigation errors,
    and collision events.

    Attributes:
        ship_id: Identifier of the ship (matches environment sub-environment key).
        episode: Zero-based episode index.

        path: Array of shape (T, 2) containing (x, y) positions.
        cross_errors: Cross-track error per timestep.
        rudder_actions: Rudder angle at each timestep.
        rudder_changes: Absolute change in rudder angle per timestep.
        heading_errors: Heading error (radians) per timestep.
        velocities: Surge velocity per timestep.
        ukc_measurements: Under-keel clearance per timestep.
        reward: Total episode reward.
        collision_events: List of detected `CollisionEvent` instances.
    """

    ship_id: str
    episode: int
    path: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    cross_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    rudder_actions: np.ndarray = field(default_factory=lambda: np.array([]))
    rudder_changes: np.ndarray = field(default_factory=lambda: np.array([]))
    heading_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    ukc_measurements: np.ndarray = field(default_factory=lambda: np.array([]))
    reward: float = 0.0
    collision_events: list[CollisionEvent] = field(default_factory=list)

# Type alias for the full data structure:
#   ship_data["ship_0"][episode_index] → ShipEpisodeData
ShipData = dict[str, list[ShipEpisodeData]]

@dataclass
class RunData:
    """Container for a full evaluation run across all ships and episodes.

    Attributes:
        ships: Dictionary mapping ship IDs to lists of `ShipEpisodeData`,
            one list per episode.
        num_episodes: Total number of episodes executed in the run.
    """

    ships: dict[str, list[ShipEpisodeData]]
    num_episodes: int

# ---------------------------------------------------------------------------
# Collision annotator
# ---------------------------------------------------------------------------

class CollisionAnnotator:
    """Detects and annotates near-miss and collision events between ships.

    This class compares ship trajectories across episodes and records spatial
    proximity events. Events are attached directly to the corresponding
    `ShipEpisodeData` objects.

    Args:
        near_miss_threshold: Distance below which a near-miss event is recorded.
        collision_threshold: Distance below which a hard collision is recorded.
    """

    def __init__(self, near_miss_threshold: float = 50.0, collision_threshold: float = 15.0):
        """Initializes the collision detector with distance thresholds.

        Args:
            near_miss_threshold: Threshold for near-miss detection.
            collision_threshold: Threshold for collision detection.
        """

        self.near_miss_threshold = near_miss_threshold
        self.collision_threshold = collision_threshold

    def annotate(self, run_data: RunData) -> None:
        """Annotates a full run with collision and near-miss events.

        Iterates over all ship pairs and compares their trajectories
        timestep-by-timestep.

        Args:
            run_data: Complete evaluation run data to annotate.

        Returns:
            None. The input object is modified in-place.
        """

        ship_data = run_data.ships
        ship_ids = list(ship_data.keys())

        if len(ship_ids) < 2:
            return  # nothing to compare in single-ship setup

        num_episodes = len(next(iter(ship_data.values())))

        for episode_idx in range(num_episodes):
            for i in range(len(ship_ids)):
                for j in range(i + 1, len(ship_ids)):
                    sid_a = ship_ids[i]
                    sid_b = ship_ids[j]
                    ep_a = ship_data[sid_a][episode_idx]
                    ep_b = ship_data[sid_b][episode_idx]
                    self._compare_paths(ep_a, ep_b)

    def _compare_paths(self, ep_a: ShipEpisodeData, ep_b: ShipEpisodeData) -> None:
        """Compares two ship episodes and records proximity events.

        Args:
            ep_a: First ship episode.
            ep_b: Second ship episode.

        Notes:
            - Only compares up to the minimum shared trajectory length.
            - Events are appended symmetrically to both episodes.
        """

        min_len = min(len(ep_a.path), len(ep_b.path))
        if min_len == 0:
            return

        for t in range(min_len):
            dist = float(np.linalg.norm(ep_a.path[t] - ep_b.path[t]))
            if dist < self.near_miss_threshold:
                is_collision = dist < self.collision_threshold
                event_a = CollisionEvent(
                    timestep=t,
                    ship_a=ep_a.ship_id,
                    ship_b=ep_b.ship_id,
                    distance=dist,
                    is_collision=is_collision,
                )
                event_b = CollisionEvent(
                    timestep=t,
                    ship_a=ep_a.ship_id,
                    ship_b=ep_b.ship_id,
                    distance=dist,
                    is_collision=is_collision,
                )
                ep_a.collision_events.append(event_a)
                ep_b.collision_events.append(event_b)
