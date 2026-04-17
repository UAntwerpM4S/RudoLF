from __future__ import annotations

import numpy as np

from typing import Optional
from contextlib import contextmanager
from fw.output_helpers import compute_heading_error
from fw.ship_episode_data import RunData, ShipData, ShipEpisodeData


class ShipDataExtractor:
    """
    Extracts and records ship-specific data from an environment.

    This utility class is responsible for pulling relevant state and action
    information from a ship environment and appending it to a per-ship
    accumulator. It operates in a defensive manner by checking attribute
    existence before accessing them, allowing compatibility across different
    environment implementations.

    Methods:
        record_initial(ship_env, acc):
            Records the initial state of a ship at the start of an episode.

        record_step(ship_env, acc):
            Records timestep data for a ship during an episode.
    """

    @staticmethod
    def record_initial(ship_env, acc: dict) -> None:
        """
        Records the initial state of a ship environment.

        Captures the starting values for position, cross-track error, velocity,
        heading error, and initializes rudder-related signals.

        Args:
            ship_env: Environment instance representing a ship.
            acc: Dictionary accumulator where data is stored.

        Notes:
            - Rudder actions and changes are initialized to 0.0 if available.
            - Heading error is computed via `compute_heading_error` when a
              physical simulator is present.
        """

        if hasattr(ship_env, "ship_pos"):
            acc["path"].append(np.copy(ship_env.ship_pos))
        if hasattr(ship_env, "cross_error"):
            acc["cross_errors"].append(ship_env.cross_error)
        if hasattr(ship_env, "velocities"):
            acc["velocities"].append(ship_env.velocities)
        if hasattr(ship_env, "phys_sim"):
            acc["heading_errors"].append(compute_heading_error(ship_env))
        if hasattr(ship_env, "performed_action"):
            acc["rudder_actions"].append(0.0)
            acc["rudder_changes"].append(0.0)

    @staticmethod
    def record_step(ship_env, acc: dict) -> None:
        """
        Records a single timestep of ship data.

        Appends current state and action-related values to the accumulator,
        including position, cross-track error, heading error, velocities,
        rudder actions, and under-keel clearance (UKC).

        Args:
            ship_env: Environment instance representing a ship.
            acc: Dictionary accumulator where data is stored.

        Notes:
            - Rudder changes are computed as the absolute difference between
              consecutive actions.
            - UKC values are duplicated on the first step to ensure alignment
              with other timestep-based arrays.
            - Only attributes present on the environment are recorded.
        """

        if hasattr(ship_env, "ship_pos"):
            acc["path"].append(np.copy(ship_env.ship_pos))

        if hasattr(ship_env, "cross_error") and hasattr(ship_env, "checkpoint_index"):
            acc["cross_errors"].append(ship_env.cross_error)

        if hasattr(ship_env, "heading_error"):
            acc["heading_errors"].append(ship_env.heading_error)
        elif hasattr(ship_env, "phys_sim"):
            acc["heading_errors"].append(compute_heading_error(ship_env))

        if hasattr(ship_env, "velocities"):
            acc["velocities"].append(ship_env.velocities)

        if hasattr(ship_env, "performed_action"):
            curr = ship_env.performed_action[0]
            if acc["rudder_actions"]:
                prev = acc["rudder_actions"][-1]
                acc["rudder_changes"].append(abs(curr - prev))
            else:
                acc["rudder_changes"].append(0.0)
            acc["rudder_actions"].append(curr)

        if hasattr(ship_env, "ukc"):
            if len(acc["ukc_measurements"]) == 0:
                acc["ukc_measurements"].append(ship_env.ukc)
            acc["ukc_measurements"].append(ship_env.ukc)


# ---------------------------------------------------------------------------
# Episode collector
# ---------------------------------------------------------------------------

class EpisodeCollector:
    """
    Collects evaluation episodes and aggregates per-ship trajectory data.

    Executes one or more episodes using a trained agent and records timestep-level
    data (e.g., positions, errors, actions, rewards) for each ship. Supports both
    single-ship and multi-ship environments and returns structured results for
    analysis or visualization.

    Args:
        agent: Trained agent implementing `predict(state, deterministic=True)`
            (e.g., SinglePPOAgent or MultiPPOAgent).
        env: Environment compatible with the agent. Can be a single-ship
            environment or a multi-ship environment exposing `sub_environments`.
        ship_ids: Optional list of ship identifiers to track. If not provided,
            uses `env.sub_environments.keys()` when available, otherwise defaults
            to ["ship_0"].
    """

    def __init__(self, agent, env, ship_ids: Optional[list[str]] = None, airsim_bridge=None):
        """Initialize the EpisodeCollector."""
        self._agent = agent
        self._env = env
        self._airsim_bridge = airsim_bridge

        if self._airsim_bridge and self._airsim_bridge.is_connected:
            from airsim.communication.coord_transform import CoordTransform
            self._coord_transform = CoordTransform()
        else:
            self._coord_transform = None

        if ship_ids is not None:
            self._ship_ids = ship_ids
        elif hasattr(env, "sub_environments") and env.sub_environments:
            self._ship_ids = list(env.sub_environments.keys())
        else:
            self._ship_ids = ["ship_0"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, num_episodes: int = 5, live_animation: bool = False) -> RunData:
        """
        Runs multiple evaluation episodes and collects per-ship data.

        Args:
            num_episodes: Number of episodes to execute.
            live_animation: If True, temporarily enables "human" render mode
                for all ship environments during execution.

        Returns:
            RunData containing aggregated per-ship `ShipEpisodeData` for each
            episode.
        """

        ship_data: ShipData = {sid: [] for sid in self._ship_ids}

        for episode in range(num_episodes):
            episode_data = self._run_episode(episode, live_animation)
            for sid, data in episode_data.items():
                ship_data[sid].append(data)

        return RunData(
            ships=ship_data,
            num_episodes=num_episodes,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @contextmanager
    def _render_mode(self, ship_envs: dict[str, object], enabled: bool):
        """
        Temporarily enables rendering for ship environments.

        Args:
            ship_envs: Mapping of ship IDs to their environment instances.
            enabled: If True, switches environments to "human" render mode
                within the context.

        Yields:
            None.

        Notes:
            Original render modes are restored after exiting the context.
            Environments set to None are ignored.
        """

        if not enabled:
            yield
            return

        render_modes: dict[str, object] = {}

        try:
            for sid, ship_env in ship_envs.items():
                if ship_env is None:
                    continue
                render_modes[sid] = ship_env.render_mode
                ship_env.render_mode = "human"
            yield
        finally:
            for sid, ship_env in ship_envs.items():
                if ship_env is None:
                    continue
                # Only restore if we successfully captured the original mode.
                if sid in render_modes:
                    ship_env.render_mode = render_modes[sid]

    def _run_episode(self, episode: int, live_animation: bool = False) -> dict[str, ShipEpisodeData]:
        """
        Executes a single episode and collects per-ship data.

        Args:
            episode: Episode index used for bookkeeping.
            live_animation: If True, enables rendering during the episode.

        Returns:
            Dictionary mapping ship IDs to `ShipEpisodeData`.
        """

        env = self._env
        ship_ids = self._ship_ids

        # Cache ship environments once
        ship_envs = {sid: self._get_ship_env(sid) for sid in ship_ids}

        # Initialise per-ship accumulators
        accumulators: dict[str, dict] = {
            sid: self._empty_accumulator(sid, episode)
            for sid in ship_ids
        }

        # Reset environment
        state, _ = env.reset()
        done = np.zeros(len(ship_ids), dtype=bool)
        steps = 0

        # Capture initial state for each ship
        for sid, ship_env in ship_envs.items():
            if ship_env is not None:
                ShipDataExtractor.record_initial(ship_env, accumulators[sid])

        with self._render_mode(ship_envs, live_animation):
            while not np.all(done):
                action, _, _ = self._agent.predict(state=state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)

                env_done = np.logical_or(terminated, truncated)
                done = np.logical_or(done, env_done)

                steps += 1
                self._record_step(accumulators, ship_envs, reward)

                if self._coord_transform and hasattr(env, 'ship_pos') and hasattr(env, 'phys_sim'):
                    ax, ay, _ = self._coord_transform.framework_to_airsim(
                        float(env.ship_pos[0]), float(env.ship_pos[1])
                    )
                    ayaw = self._coord_transform.framework_heading_to_airsim_yaw(
                        float(env.phys_sim.state.heading)
                    )
                    self._airsim_bridge.set_pose(ax, ay, 0.0, ayaw)

                if live_animation:
                    for ship_env in ship_envs.values():
                        if ship_env is not None:
                            ship_env.render()

        # reward is accumulated inline during _record_step; nothing to finalize here

        print(f"Episode {episode + 1} finished after {steps} steps")

        return {
            sid: self._accumulator_to_episode_data(acc)
            for sid, acc in accumulators.items()
        }

    @staticmethod
    def _record_step(
            accumulators: dict[str, dict],
            ship_envs: dict[str, object],
            reward,
    ) -> None:
        """
        Records one timestep of data for each ship.

        Args:
            accumulators: Per-ship data accumulators.
            ship_envs: Mapping of ship IDs to environment instances.
            reward: Reward signal from the environment. Can be a scalar,
                sequence, or dict keyed by ship ID.

        Notes:
            - State data is recorded via `ShipDataExtractor.record_step`.
            - Rewards are accumulated as a running sum per ship.
        """

        for idx, (sid, ship_env) in enumerate(ship_envs.items()):
            if ship_env is None:
                continue

            acc = accumulators[sid]

            ShipDataExtractor.record_step(ship_env, acc)

            if isinstance(reward, dict):
                r = reward.get(sid, 0.0)
            elif isinstance(reward, (list, np.ndarray)):
                r = float(reward[idx])
            else:
                r = float(reward)  # shared scalar

            acc["reward"] = acc.get("reward", 0.0) + r

    def _get_ship_env(self, ship_id: str):
        """
        Returns the environment corresponding to a given ship.

        Args:
            ship_id: Identifier of the ship.

        Returns:
            The sub-environment for the given ship in multi-ship mode,
            otherwise the main environment.
        """

        if hasattr(self._env, "sub_environments") and self._env.sub_environments:
            return self._env.sub_environments.get(ship_id)
        return self._env  # single-ship: env IS the ship env

    @staticmethod
    def _empty_accumulator(ship_id: str, episode: int) -> dict:
        """
        Initializes an empty data accumulator for a ship.

        Args:
            ship_id: Identifier of the ship.
            episode: Episode index.

        Returns:
            Dictionary initialized with empty lists and default values for
            storing timestep data and cumulative reward.
        """

        return {
            "ship_id": ship_id,
            "episode": episode,
            "path": [],
            "cross_errors": [],
            "rudder_actions": [],
            "rudder_changes": [],
            "heading_errors": [],
            "velocities": [],
            "ukc_measurements": [],
            "reward": 0.0,
        }

    @staticmethod
    def _accumulator_to_episode_data(acc: dict) -> ShipEpisodeData:
        """
        Converts a raw accumulator into structured episode data.

        Args:
            acc: Accumulator containing collected timestep data.

        Returns:
            ShipEpisodeData with NumPy arrays and aggregated metrics.

        Notes:
            Empty path data is converted to an array of shape (0, 2).
        """

        return ShipEpisodeData(
            ship_id=acc["ship_id"],
            episode=acc["episode"],
            path=np.array(acc["path"]) if acc["path"] else np.empty((0, 2)),
            cross_errors=np.array(acc["cross_errors"]),
            rudder_actions=np.array(acc["rudder_actions"]),
            rudder_changes=np.array(acc["rudder_changes"]),
            heading_errors=np.array(acc["heading_errors"]),
            velocities=np.array(acc["velocities"]),
            ukc_measurements=np.array(acc["ukc_measurements"]),
            reward=acc.get("reward", 0.0),
        )
