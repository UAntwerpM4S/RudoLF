from __future__ import annotations

import logging
import warnings
import numpy as np

from typing import Optional
from collections import deque
from contextlib import contextmanager
from fw.configuration.config_profile import VesselProfile
from fw.postprocessing.output_helpers import compute_heading_error
from fw.configuration.profile_applier import SimulatorApplier, ActionAdapter
from fw.postprocessing.ship_episode_data import RunData, ShipData, ShipEpisodeData

logger = logging.getLogger(__name__)


class ShipDataExtractor:
    """
    Appends a ship environment's state and action data to a per-ship accumulator.

    Only attributes present on the environment are recorded, so it works with
    different environment implementations.
    """

    @staticmethod
    def record_initial(ship_env, acc: dict) -> None:
        """
        Record the initial position, cross-track error, velocity and heading error.
        Rudder and thrust start at the env's actuator state after reset
        (performed_action; 0.0 if it has none yet), their changes at 0.0.
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
            start = ship_env.performed_action
            acc["rudder_actions"].append(0.0 if start is None else start[0])
            acc["rudder_changes"].append(0.0)
            acc["thrust_actions"].append(0.0 if start is None else start[1])
            acc["thrust_changes"].append(0.0)

    @staticmethod
    def record_step(ship_env, acc: dict) -> None:
        """
        Record one step: position, cross-track and heading error, velocities,
        rudder/thrust actions and their absolute changes, and UKC.

        The first UKC value is recorded twice so the array aligns with the others,
        which include the initial state.
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
    Runs evaluation episodes with a trained agent and records per-step data
    (positions, errors, actions, rewards) per ship, for single- and multi-ship
    environments.
    """

    def __init__(self, agent, env, profile: VesselProfile, ship_ids: Optional[list[str]] = None, airsim_bridge=None):
        """
        Args:
            agent: Trained agent with predict(state, deterministic=True).
            env: Single-ship env, or multi-ship env exposing sub_environments.
            profile: Vessel profile applied to the simulator and actions.
            ship_ids: Ship IDs to track (default: sub_environments keys, else ["ship_0"]).
            airsim_bridge: Optional connected AirSim bridge; the ship pose is
                mirrored to AirSim each step.

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
        Run evaluation episodes and collect per-ship data.

        Args:
            num_episodes: Number of episodes.
            live_animation: Render in "human" mode during the run.

        Returns:
            RunData with one ShipEpisodeData per ship per episode.

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
        Apply ActionAdapter.transform to each action, keeping the container type.

        Accepts an (N, action_dim) array, a tuple of N actions, or a single
        (action_dim,) array.

        Raises:
            TypeError: For any other input type.
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
        Context manager: if enabled, switch the ship envs to "human" render mode
        and restore their original modes on exit (None envs are skipped).
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
                if sid in render_modes:
                    ship_env.render_mode = render_modes[sid]

    def _run_episode(self, episode: int, live_animation: bool = False) -> dict[str, ShipEpisodeData]:
        """
        Run one episode and return ship ID to ShipEpisodeData.

        On a collision, the last 100 critic values and minimum distances are
        logged ([VALUE_DIAG]).
        """

        env = self._env
        ship_ids = self._ship_ids

        ship_envs = {sid: self._get_ship_env(sid) for sid in ship_ids}

        missing = [sid for sid, senv in ship_envs.items() if senv is None]
        if missing:
            warnings.warn(
                f"Could not resolve sub-environments for ship(s): {missing}. "
                f"Their data will be empty.",
                RuntimeWarning,
                stacklevel=2,
            )

        accumulators: dict[str, dict] = {
            sid: self._empty_accumulator(sid, episode)
            for sid in ship_ids
        }

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

        for sid, ship_env in ship_envs.items():
            if ship_env is not None:
                ShipDataExtractor.record_initial(ship_env, accumulators[sid])

        # Rolling buffers for collision diagnostic (last 100 steps)
        _value_buffer: deque = deque(maxlen=100)
        _dist_buffer: deque = deque(maxlen=100)

        # Primary ship env, used for the distance diagnostic
        _primary_ship_env = next(iter(ship_envs.values()), None)

        info = {}
        with self._render_mode(ship_envs, live_animation):
            while not np.all(done):
                action, _, value = self._agent.predict(state=state, deterministic=True)
                if value is not None:
                    value_arr = np.asarray(value)
                    if value_arr.size > 0:
                        _value_buffer.append(float(value_arr.reshape(-1)[0]))

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

        # Collision diagnostic: value and distance history before the collision.
        _terminal_info = info[0] if isinstance(info, (list, tuple)) else info
        if isinstance(_terminal_info, dict):
            _col_info = _terminal_info.get("collision_info", {}) or {}
            if _col_info.get("collision"):
                _vbuf = list(_value_buffer)
                _dbuf = list(_dist_buffer)
                _col_type = _col_info.get("collision_type", "unknown")
                _n = len(_vbuf)
                logger.info(
                    "[VALUE_DIAG] Episode %s \u2014 collision! type=%s, steps_in_buffer=%d",
                    episode, _col_type, _n
                )
                if _vbuf:
                    _offset = max(0, 100 - _n)
                    header = f"  {'step':>5}  {'value':>9}  {'min_dist':>10}"
                    _rows = [header]
                    for i, (v, d) in enumerate(zip(_vbuf, _dbuf)):
                        _d_str = f"{d:10.1f}" if d < float("inf") else "       inf"
                        _rows.append(f"  {_offset + i:>5}  {v:>9.3f}  {_d_str}")
                    logger.info("[VALUE_DIAG]\n%s", "\n".join(_rows))
                    logger.info(
                        "[VALUE_DIAG]   --- value: min=%.3f  max=%.3f  final=%.3f  mean_last10=%.3f",
                        min(_vbuf), max(_vbuf), _vbuf[-1], float(np.mean(_vbuf[-10:]))
                    )
                    _finite_dists = [d for d in _dbuf if d < float("inf")]
                    if _finite_dists:
                        logger.info(
                            "[VALUE_DIAG]   --- dist:  min=%.1fm  final=%.1fm",
                            min(_finite_dists), _dbuf[-1]
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

        logger.info("Episode %d finished after %d steps", episode + 1, steps)

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
        Record one step per ship and add the reward to its running sum.

        reward may be a scalar (shared), a sequence (by ship index) or a dict
        keyed by ship ID.
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
        Return the ship's sub-environment (multi-ship) or the env itself (single-ship).
        """

        if hasattr(self._env, "sub_environments") and self._env.sub_environments:
            return self._env.sub_environments.get(ship_id)
        return self._env  # single-ship: env IS the ship env

    @staticmethod
    def _empty_accumulator(ship_id: str, episode: int) -> dict:
        """
        Return an empty per-ship accumulator.
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
        Convert an accumulator to ShipEpisodeData (an empty path becomes shape (0, 2)).
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
