from __future__ import annotations

import warnings
import numpy as np

from typing import Optional
from collections import deque
from contextlib import contextmanager
from fw.configuration.config_profile import VesselProfile
from fw.postprocessing.output_helpers import compute_heading_error
from fw.configuration.profile_applier import SimulatorApplier, ActionAdapter
from fw.postprocessing.ship_episode_data import RunData, ShipData, ShipEpisodeData


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
            acc["thrust_actions"].append(0.0)
            acc["thrust_changes"].append(0.0)

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
            curr_thrust = ship_env.performed_action[1]
            if acc["rudder_actions"]:
                prev = acc["rudder_actions"][-1]
                acc["rudder_changes"].append(abs(curr - prev))
            else:
                acc["rudder_changes"].append(0.0)
            acc["rudder_actions"].append(curr)

            if acc["thrust_actions"]:
                prev_thrust = acc["thrust_actions"][-1]
                acc["thrust_changes"].append(abs(curr_thrust - prev_thrust))
            else:
                acc["thrust_changes"].append(0.0)
            acc["thrust_actions"].append(curr_thrust)

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

    def __init__(self, agent, env, profile: VesselProfile, ship_ids: Optional[list[str]] = None, airsim_bridge=None):
        """
        Initialize the collector and resolve the ship IDs to track.

        Args:
            agent: Trained agent used to produce actions during evaluation.
            env: Environment to evaluate, either single-ship or multi-ship.
            profile: Profile that will be used for training or visualization.
            ship_ids: Optional explicit ship ID list. If omitted, the collector
                uses `env.sub_environments` when available, otherwise
                defaults to `["ship_0"]`.

        Raises:
            ValueError: If the resolved ship ID list is empty.
        """

        self._agent = agent
        self._env = env
        self._profile = profile
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

        if not self._ship_ids:
            raise ValueError("ship_ids must contain at least one entry.")

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

        Raises:
            ValueError: If num_episodes is not positive.
        """

        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}.")

        ship_data: ShipData = {sid: [] for sid in self._ship_ids}

        for episode in range(num_episodes):
            episode_data = self._run_episode(episode, live_animation)
            for sid, data in episode_data.items():
                ship_data[sid].append(data)

        return RunData(
            ships=ship_data,
            num_episodes=num_episodes,
        )

    @staticmethod
    def _transform_actions(
        profile: VesselProfile,
        actions,
    ):
        """
        Apply :meth:`transform` to every action in a batch and return the
        result in the same container type as the input.

        Supported input types:

        * `np.ndarray` of shape `(N, action_dim)` — one row per agent.
        * `tuple` of `N` individual action arrays.
        * A single `np.ndarray` of shape `(action_dim,)` — scalar
          single-agent case.

        Args:
            profile: Profile whose thrust settings to apply.
            actions: Batch of actions in any of the supported container types.

        Returns:
            Transformed actions in the same container type as *actions*.

        Raises:
            TypeError: If *actions* is not one of the supported types.
        """

        if isinstance(actions, np.ndarray) and actions.ndim == 2:
            return np.stack([
                ActionAdapter.transform(profile, actions[i])
                for i in range(actions.shape[0])
            ])

        if isinstance(actions, tuple):
            return tuple(
                ActionAdapter.transform(profile, a)
                for i, a in enumerate(actions)
            )

        if isinstance(actions, np.ndarray) and actions.ndim == 1:
            return ActionAdapter.transform(profile, actions)

        raise TypeError(
            f"ActionAdapter.transform_actions: unsupported actions type "
            f"'{type(actions).__name__}' with ndim="
            f"{getattr(actions, 'ndim', 'n/a')}. "
            f"Expected a 2-D np.ndarray, a tuple, or a 1-D np.ndarray."
        )

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

        Raises:
            RuntimeError: If the environment reset does not return a valid state.
        """

        env = self._env
        ship_ids = self._ship_ids

        # Cache ship environments once
        ship_envs = {sid: self._get_ship_env(sid) for sid in ship_ids}

        # Warn if any ship environment could not be resolved
        missing = [sid for sid, senv in ship_envs.items() if senv is None]
        if missing:
            warnings.warn(
                f"Could not resolve sub-environments for ship(s): {missing}. "
                f"Their data will be empty.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Initialise per-ship accumulators
        accumulators: dict[str, dict] = {
            sid: self._empty_accumulator(sid, episode)
            for sid in ship_ids
        }

        # Reset environment
        state, _ = env.reset()

        if self._profile is not None:
            _rollout_sim = getattr(env, "phys_sim", None) or getattr(env, "_phys_sim", None)
            if _rollout_sim is not None:
                SimulatorApplier.apply(self._profile, _rollout_sim)
            elif hasattr(env, "sub_environments") and env.sub_environments:
                sub_envs = env.sub_environments
                iterable = sub_envs.values() if isinstance(sub_envs, dict) else sub_envs

                for sub in iterable:
                    _sub_sim = getattr(sub, "phys_sim", None) or getattr(sub, "_phys_sim", None)
                    if _sub_sim is not None:
                        SimulatorApplier.apply(self._profile, _sub_sim)

        done = np.zeros(len(ship_ids), dtype=bool)
        steps = 0

        # Capture initial state for each ship
        for sid, ship_env in ship_envs.items():
            if ship_env is not None:
                ShipDataExtractor.record_initial(ship_env, accumulators[sid])

        # Rolling buffers for collision diagnostic (last 100 steps)
        _value_buffer: deque = deque(maxlen=100)
        _dist_buffer: deque = deque(maxlen=100)

        # Resolve the primary ship env once for distance queries
        _primary_ship_env = next(iter(ship_envs.values()), None)

        info = {}
        with self._render_mode(ship_envs, live_animation):
            while not np.all(done):
                action, _, value = self._agent.predict(state=state, deterministic=True)
                if value:
                    _value_buffer.append(float(np.asarray(value).reshape(-1)[0]))

                # Apply optional per-action profile transform (e.g. fixed thrust).
                if self._profile is not None:
                    action = self._transform_actions(self._profile, action)

                state, reward, terminated, truncated, info = env.step(action)

                # Current minimum distance to any collision entity (post-step)
                _cur_dist = float("inf")
                if (
                    _primary_ship_env is not None
                    and getattr(_primary_ship_env, "collision_entities", None)
                    and getattr(_primary_ship_env, "ship_pos", None) is not None
                ):
                    _cur_dist = min(
                        e.distance_to_point(_primary_ship_env.ship_pos)
                        for e in _primary_ship_env.collision_entities
                    )
                _dist_buffer.append(_cur_dist)

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

        # Collision value diagnostic: print pre-collision value + distance history.
        _terminal_info = info[0] if isinstance(info, (list, tuple)) else info
        if isinstance(_terminal_info, dict):
            _col_info = _terminal_info.get("collision_info", {}) or {}
            if _col_info.get("collision"):
                _vbuf = list(_value_buffer)
                _dbuf = list(_dist_buffer)
                _col_type = _col_info.get("collision_type", "unknown")
                _n = len(_vbuf)
                print(
                    f"\n[VALUE_DIAG] Episode {episode} — collision! type={_col_type}, "
                    f"steps_in_buffer={_n}"
                )
                if _vbuf:
                    # Print step | value | dist side-by-side for readability
                    _offset = max(0, 100 - _n)
                    header = f"  {'step':>5}  {'value':>9}  {'min_dist':>10}"
                    print(header)
                    for i, (v, d) in enumerate(zip(_vbuf, _dbuf)):
                        _d_str = f"{d:10.1f}" if d < float("inf") else "       inf"
                        print(f"  {_offset + i:>5}  {v:>9.3f}  {_d_str}")
                    print(
                        f"  --- value: min={min(_vbuf):.3f}  max={max(_vbuf):.3f}  "
                        f"final={_vbuf[-1]:.3f}  mean_last10={float(np.mean(_vbuf[-10:])):.3f}"
                    )
                    _finite_dists = [d for d in _dbuf if d < float("inf")]
                    if _finite_dists:
                        print(
                            f"  --- dist:  min={min(_finite_dists):.1f}m  "
                            f"final={_dbuf[-1]:.1f}m"
                        )

        # Attach terminal info per ship (env-reported collision / COLREG data).
        for idx, sid in enumerate(ship_ids):
            ship_info = info[idx] if isinstance(info, (list, tuple)) else info
            if not isinstance(ship_info, dict):
                continue
            ci = ship_info.get("collision_info", {}) or {}
            acc = accumulators[sid]
            acc["collision"] = bool(ci.get("collision", False))
            acc["collision_type"] = ci.get("collision_type")
            acc["collision_side"] = ci.get("collision_side")
            acc["colreg_violations"] = int(ship_info.get("colreg_violations", 0) or 0)
            clf_info = ship_info.get("classification", {}) or {}
            encounters = clf_info.get("encounters", []) or []
            # Copy each dict so downstream mutation can't affect env state.
            acc["scenario_encounters"] = [dict(enc) for enc in encounters]

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
            "thrust_actions": [],
            "thrust_changes": [],
            "heading_errors": [],
            "velocities": [],
            "ukc_measurements": [],
            "reward": 0.0,
            "collision": False,
            "collision_type": None,
            "collision_side": None,
            "colreg_violations": 0,
            "scenario_encounters": [],
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
            thrust_actions=np.array(acc["thrust_actions"]),
            thrust_changes=np.array(acc["thrust_changes"]),
            heading_errors=np.array(acc["heading_errors"]),
            velocities=np.array(acc["velocities"]),
            ukc_measurements=np.array(acc["ukc_measurements"]),
            reward=acc.get("reward", 0.0),
            collision=acc.get("collision", False),
            collision_type=acc.get("collision_type"),
            collision_side=acc.get("collision_side"),
            colreg_violations=acc.get("colreg_violations", 0),
            scenario_encounters=acc.get("scenario_encounters", []),
        )
