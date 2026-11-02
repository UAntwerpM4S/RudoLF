from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CollisionEvent:
    """
    Proximity event between two ships at one timestep: a near miss, or a
    collision if is_collision (distance below the collision threshold).
    """

    timestep: int
    ship_a: str
    ship_b: str
    distance: float
    is_collision: bool

    def __post_init__(self) -> None:
        """
        Raise ValueError if timestep or distance is negative.
        """

        if self.timestep < 0:
            raise ValueError(f"timestep must be non-negative, got {self.timestep}.")
        if self.distance < 0.0:
            raise ValueError(f"distance must be non-negative, got {self.distance}.")


@dataclass
class ShipEpisodeData:
    """
    Recorded time series of one ship over one episode.

    Attributes:
        ship_id: Ship identifier (the env's sub-environment key).
        episode: Zero-based episode index.
        path: (T, 2) positions.
        cross_errors, rudder_actions, rudder_changes, thrust_actions,
            thrust_changes, heading_errors (rad), velocities (surge),
            ukc_measurements: Per-timestep values.
        reward: Total episode reward.
        collision_events: Detected CollisionEvents.
    """

    ship_id: str
    episode: int
    path: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    cross_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    rudder_actions: np.ndarray = field(default_factory=lambda: np.array([]))
    rudder_changes: np.ndarray = field(default_factory=lambda: np.array([]))
    thrust_actions: np.ndarray = field(default_factory=lambda: np.array([]))
    thrust_changes: np.ndarray = field(default_factory=lambda: np.array([]))
    heading_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    ukc_measurements: np.ndarray = field(default_factory=lambda: np.array([]))
    reward: float = 0.0
    collision_events: list[CollisionEvent] = field(default_factory=list)
    # COLAV terminal info reported by the env at episode end
    collision: bool = False
    collision_type: str | None = None
    collision_side: str | None = None
    colreg_violations: int = 0
    # One record per encounter, with at least "gt" and "pred" labels (5-way
    # extended label space) plus outcome/CRI fields used by MetricsComputer.
    scenario_encounters: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Validate ship_id and episode, and coerce an empty path to shape (0, 2).

        Raises:
            ValueError: On an empty ship_id, negative episode or a path that is
                not (T, 2).
        """

        if not self.ship_id:
            raise ValueError("ship_id must be a non-empty string.")
        if self.episode < 0:
            raise ValueError(f"episode must be non-negative, got {self.episode}.")
        path = np.asarray(self.path)
        if path.ndim == 1 and path.size == 0:
            object.__setattr__(self, "path", np.empty((0, 2)))
        elif path.ndim != 2 or path.shape[1] != 2:
            if not (path.ndim == 2 and path.shape[0] == 0):
                raise ValueError(
                    f"path must have shape (T, 2), got {path.shape}."
                )


# ship_data["ship_0"][episode_index] is a ShipEpisodeData
ShipData = dict[str, list[ShipEpisodeData]]


@dataclass
class RunData:
    """
    Full evaluation run.

    Attributes:
        ships: {ship_id: [ShipEpisodeData per episode]}.
        num_episodes: Number of episodes in the run.
    """

    ships: dict[str, list[ShipEpisodeData]]
    num_episodes: int

    def __post_init__(self) -> None:
        """
        Raise ValueError if num_episodes is negative or ships is None.
        """

        if self.num_episodes < 0:
            raise ValueError(f"num_episodes must be non-negative, got {self.num_episodes}.")
        if self.ships is None:
            raise ValueError("ships must not be None.")


# ---------------------------------------------------------------------------
# Collision annotator
# ---------------------------------------------------------------------------

class CollisionAnnotator:
    """
    Record near-miss and collision events between ship pairs, attached to the
    ShipEpisodeData objects.
    """

    def __init__(self, near_miss_threshold: float = 50.0, collision_threshold: float = 15.0):
        """
        Args:
            near_miss_threshold: Distance below which an event is recorded.
            collision_threshold: Distance below which it is a collision.

        Raises:
            ValueError: If a threshold is non-positive or
                collision_threshold >= near_miss_threshold.
        """

        if near_miss_threshold <= 0 or collision_threshold <= 0:
            raise ValueError("Distance thresholds must be positive.")
        if collision_threshold >= near_miss_threshold:
            raise ValueError(
                f"collision_threshold ({collision_threshold}) must be strictly less than "
                f"near_miss_threshold ({near_miss_threshold})."
            )

        self.near_miss_threshold = near_miss_threshold
        self.collision_threshold = collision_threshold

    def annotate(self, run_data: RunData) -> None:
        """
        Compare all ship pairs per episode and add events in place.

        Raises:
            ValueError: If the ships have different episode counts.
        """

        ship_data = run_data.ships
        ship_ids = list(ship_data.keys())

        if len(ship_ids) < 2:
            return  # nothing to compare in single-ship setup

        episode_counts = {sid: len(ship_data[sid]) for sid in ship_ids}
        unique_counts = set(episode_counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                f"Inconsistent episode counts across ships: {episode_counts}. "
                f"All ships must have the same number of episodes."
            )

        num_episodes = next(iter(unique_counts))

        for episode_idx in range(num_episodes):
            for i in range(len(ship_ids)):
                for j in range(i + 1, len(ship_ids)):
                    sid_a = ship_ids[i]
                    sid_b = ship_ids[j]
                    ep_a = ship_data[sid_a][episode_idx]
                    ep_b = ship_data[sid_b][episode_idx]
                    self._compare_paths(ep_a, ep_b)

    def _compare_paths(self, ep_a: ShipEpisodeData, ep_b: ShipEpisodeData) -> None:
        """
        Record proximity events of two episodes (over their common length) in
        both episodes.
        """

        min_len = min(len(ep_a.path), len(ep_b.path))
        if min_len == 0:
            return

        path_a = ep_a.path[:min_len]
        path_b = ep_b.path[:min_len]
        dists = np.linalg.norm(path_a - path_b, axis=1)

        event_timesteps = np.flatnonzero(dists < self.near_miss_threshold)

        for t in event_timesteps:
            dist = float(dists[t])
            is_collision = dist < self.collision_threshold

            event_a = CollisionEvent(
                timestep=int(t),
                ship_a=ep_a.ship_id,
                ship_b=ep_b.ship_id,
                distance=dist,
                is_collision=is_collision,
            )
            event_b = CollisionEvent(
                timestep=int(t),
                ship_a=ep_a.ship_id,
                ship_b=ep_b.ship_id,
                distance=dist,
                is_collision=is_collision,
            )

            ep_a.collision_events.append(event_a)
            ep_b.collision_events.append(event_b)
