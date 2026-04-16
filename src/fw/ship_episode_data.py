from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CollisionEvent:
    """Records a near-miss or collision between two ships at a given timestep."""
    timestep: int
    ship_a: str
    ship_b: str
    distance: float
    is_collision: bool  # True if distance < hard collision threshold

@dataclass
class ShipEpisodeData:
    """
    All recorded data for a single ship over a single episode.

    Attributes
    ----------
    ship_id     : identifier matching the key in MultiSimEnv.sub_environments
    episode     : zero-based episode index
    path        : (T, 2) array of (x, y) positions
    cross_errors: cross-track errors per step (length may be < T due to
                  checkpoint gating in the original code)
    rudder_actions     : rudder angle at each step
    rudder_changes     : absolute change in rudder angle per step
    heading_errors     : heading error (rad) per step
    velocities         : surge velocity per step
    ukc_measurements   : under-keel clearance per step
    reward             : scalar episode reward
    collision_events   : populated later by CollisionAnnotator
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
    """
    Container for a full evaluation run.
    """
    ships: dict[str, list[ShipEpisodeData]]
    num_episodes: int

# ---------------------------------------------------------------------------
# Collision annotator
# ---------------------------------------------------------------------------

class CollisionAnnotator:
    """
    Detects near-miss and collision events between ships and attaches them
    to the relevant ShipEpisodeData objects.

    Parameters
    ----------
    near_miss_threshold : distance (same units as ship_pos) below which
                          a near-miss event is recorded
    collision_threshold : distance below which a hard collision is recorded
    """

    def __init__(self, near_miss_threshold: float = 50.0, collision_threshold: float = 15.0):
        self.near_miss_threshold = near_miss_threshold
        self.collision_threshold = collision_threshold

    def annotate(self, run_data: RunData) -> None:
        """
        Annotate run_data in-place with CollisionEvent objects.

        Iterates over all pairs of ships and compares positions at each
        shared timestep within each episode.
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
        """Compare two ships' paths and record proximity events on both."""
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
