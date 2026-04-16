from __future__ import annotations

import numpy as np

from typing import Optional
from fw.output_helpers import compute_heading_error
from fw.ship_episode_data import RunData, ShipData, ShipEpisodeData


class ShipDataExtractor:
    """
    Responsible ONLY for extracting data from env → accumulator.
    """

    @staticmethod
    def record_initial(ship_env, acc):
        if hasattr(ship_env, "ship_pos"):
            acc["path"].append(np.copy(ship_env.ship_pos))
        if hasattr(ship_env, "cross_error"):
            acc["cross_errors"].append(ship_env.cross_error)
        if hasattr(ship_env, "velocities"):
            acc["velocities"].append(ship_env.velocities)
        if hasattr(ship_env, "phys_sim"):
            acc["heading_errors"].append(
                compute_heading_error(ship_env)
            )

    @staticmethod
    def record_step(ship_env, acc):
        if hasattr(ship_env, "ship_pos"):
            acc["path"].append(np.copy(ship_env.ship_pos))

        if hasattr(ship_env, "cross_error") and hasattr(ship_env, "checkpoint_index"):
            if ship_env.checkpoint_index >= 2:
                acc["cross_errors"].append(ship_env.cross_error)

        if hasattr(ship_env, "heading_error"):
            acc["heading_errors"].append(ship_env.heading_error)

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
            acc["ukc_measurements"].append(ship_env.ukc)

# ---------------------------------------------------------------------------
# Episode collector
# ---------------------------------------------------------------------------

class EpisodeCollector:
    """
    Runs evaluation episodes and returns structured per-ship data.

    Replaces the inline episode loop in AgentManager.visualize_trained_agent().
    Works for both single-ship (one entry in ship_data) and multi-ship setups.

    Parameters
    ----------
    agent       : trained agent (SinglePPOAgent or MultiPPOAgent)
    env         : the environment to evaluate in
    ship_ids    : list of ship identifiers; defaults to ["ship_0"] for
                  single-ship setups, or the keys of env.sub_environments
    """

    def __init__(self, agent, env, ship_ids: Optional[list[str]] = None):
        self._agent = agent
        self._env = env

        if ship_ids is not None:
            self._ship_ids = ship_ids
        elif hasattr(env, "sub_environments") and self._env.sub_environments:  # <-- add truthiness check
            self._ship_ids = list(env.sub_environments.keys())
        else:
            self._ship_ids = ["ship_0"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, num_episodes: int = 5) -> RunData:
        """
        Run `num_episodes` evaluation episodes.

        Returns
        -------
        ship_data : dict mapping each ship_id to a list of ShipEpisodeData,
                    one entry per episode.
        """
        ship_data: ShipData = {sid: [] for sid in self._ship_ids}

        for episode in range(num_episodes):
            episode_data = self._run_episode(episode)
            for sid, data in episode_data.items():
                ship_data[sid].append(data)

        return RunData(
            ships=ship_data,
            num_episodes=num_episodes,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_episode(self, episode: int) -> dict[str, ShipEpisodeData]:
        """Run a single episode and return per-ship data."""
        env = self._env

        # Initialise per-ship accumulators
        accumulators: dict[str, dict] = {
            sid: self._empty_accumulator(sid, episode)
            for sid in self._ship_ids
        }

        state, _ = env.reset()
        done = np.zeros(len(self._ship_ids), dtype=bool)
        steps = 0

        # Capture initial state for each ship
        for sid in self._ship_ids:
            ship_env = self._get_ship_env(sid)
            if ship_env is not None:
                ShipDataExtractor.record_initial(ship_env, accumulators[sid])

        while not np.all(done):
            action, _, _ = self._agent.predict(state=state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            env_done = np.logical_or(terminated, truncated)
            done = np.logical_or(done, env_done)

            steps += 1
            self._record_step(accumulators, reward, steps)

        # reward is accumulated inline during _record_step; nothing to finalize here

        print(f"Episode {episode + 1} finished after {steps} steps")

        return {
            sid: self._accumulator_to_episode_data(acc)
            for sid, acc in accumulators.items()
        }

    def _record_step(self, accumulators: dict, reward: float, steps: int) -> None:
        """Append one timestep of observations to each ship's accumulator."""
        for sid in self._ship_ids:
            ship_env = self._get_ship_env(sid)
            acc = accumulators[sid]

            if ship_env is None:
                continue

            ShipDataExtractor.record_step(ship_env, acc)

            # Running mean reward (simple incremental average)
            if isinstance(reward, dict):
                r = reward.get(sid, 0.0)
            elif isinstance(reward, (list, np.ndarray)):
                idx = self._ship_ids.index(sid)
                r = reward[idx]
            else:
                r = reward  # shared scalar

            acc["reward"] = acc.get("reward", 0.0) + r

    def _get_ship_env(self, ship_id: str):
        """Return the sub-environment for a ship, or the env itself for single-ship."""
        if hasattr(self._env, "sub_environments") and self._env.sub_environments:
            return self._env.sub_environments.get(ship_id)
        return self._env  # single-ship: env IS the ship env

    @staticmethod
    def _empty_accumulator(ship_id: str, episode: int) -> dict:
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
