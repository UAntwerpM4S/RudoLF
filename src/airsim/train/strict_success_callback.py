"""
Strict success early stopping callback for Stable Baselines3.

The callback stops training once the agent achieves N consecutive episodes
where all checkpoints are passed, the episode never violates the strict
cross-track bound, and the average episode reward is near its normalized
maximum.
"""

from __future__ import annotations

import logging
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class StrictSuccessStoppingCallback(BaseCallback):
    """
    Stop training after repeated strict-success episodes.

    Args:
        required_consecutive_episodes: Number of strict-success episodes
            required in a row before stopping.
        min_average_episode_reward: Threshold for the average episode reward.
            With a normalized reward in [-1, 1], values close to 1.0 enforce
            near-perfect episodes.
        strict_cte_limit_m: Maximum allowed absolute CTE during successful
            episodes.
        verbose: SB3 callback verbosity.
    """

    def __init__(
        self,
        required_consecutive_episodes: int = 5,
        min_average_episode_reward: float = 0.95,
        strict_cte_limit_m: float = 10.0,
        verbose: int = 0,
    ) -> None:
        """
        Initialize strict stopping thresholds and episode trackers.
        """

        super().__init__(verbose=verbose)
        self.required_consecutive_episodes = max(int(required_consecutive_episodes), 1)
        self.min_average_episode_reward = float(min_average_episode_reward)
        self.strict_cte_limit_m = float(strict_cte_limit_m)

        self._consecutive_successes = 0
        self._episode_reward_sum: dict[int, float] = {}
        self._episode_steps: dict[int, int] = {}
        self._episode_max_abs_cte: dict[int, float] = {}
        self._episode_cte_violated: dict[int, bool] = {}
        self._episode_all_checkpoints_passed: dict[int, bool] = {}

    def _on_training_start(self) -> None:
        """
        Reset per-env episode trackers when SB3 training begins.
        """

        n_envs = int(getattr(self.training_env, "num_envs", 1))
        for env_index in range(n_envs):
            self._reset_episode_trackers(env_index)

    def _reset_episode_trackers(self, env_index: int) -> None:
        """
        Clear one vectorized environment's in-progress episode metrics.
        """

        self._episode_reward_sum[env_index] = 0.0
        self._episode_steps[env_index] = 0
        self._episode_max_abs_cte[env_index] = 0.0
        self._episode_cte_violated[env_index] = False
        self._episode_all_checkpoints_passed[env_index] = False

    def _on_step(self) -> bool:
        """
        Update episode metrics from SB3 rollout data and stop if qualified.
        """

        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if len(dones) == 0:
            return True

        for env_index, done in enumerate(dones):
            info = infos[env_index] if env_index < len(infos) else {}
            reward = float(rewards[env_index]) if env_index < len(rewards) else 0.0

            self._episode_reward_sum[env_index] = (
                self._episode_reward_sum.get(env_index, 0.0) + reward
            )
            self._episode_steps[env_index] = self._episode_steps.get(env_index, 0) + 1

            cte = abs(float(info.get("cross_track_error", 0.0)))
            self._episode_max_abs_cte[env_index] = max(
                self._episode_max_abs_cte.get(env_index, 0.0),
                cte,
            )
            if cte > self.strict_cte_limit_m or bool(info.get("strict_cte_violation", False)):
                self._episode_cte_violated[env_index] = True

            route_total_checkpoints = int(info.get("route_total_checkpoints", 1))
            checkpoint_index = int(info.get("checkpoint_index", 0))
            if checkpoint_index >= route_total_checkpoints or bool(
                info.get("all_checkpoints_passed", False)
            ):
                self._episode_all_checkpoints_passed[env_index] = True

            if not bool(done):
                continue

            monitor_episode = info.get("episode")
            if isinstance(monitor_episode, dict):
                episode_reward = float(
                    monitor_episode.get("r", self._episode_reward_sum.get(env_index, 0.0))
                )
                episode_steps = int(
                    monitor_episode.get("l", self._episode_steps.get(env_index, 1))
                )
            else:
                episode_reward = float(self._episode_reward_sum.get(env_index, 0.0))
                episode_steps = int(self._episode_steps.get(env_index, 1))

            average_episode_reward = episode_reward / max(episode_steps, 1)
            strict_success = bool(
                self._episode_all_checkpoints_passed.get(env_index, False)
                and not self._episode_cte_violated.get(env_index, False)
                and average_episode_reward >= self.min_average_episode_reward
            )

            if strict_success:
                self._consecutive_successes += 1
            else:
                self._consecutive_successes = 0

            if self.verbose > 0:
                logger.info(
                    "[StrictSuccessStoppingCallback] env=%s strict_success=%s "
                    "cte_max=%.3fm avg_reward=%.4f consecutive=%d/%d",
                    env_index, strict_success,
                    self._episode_max_abs_cte.get(env_index, 0.0),
                    average_episode_reward, self._consecutive_successes,
                    self.required_consecutive_episodes
                )

            self._reset_episode_trackers(env_index)

            if self._consecutive_successes >= self.required_consecutive_episodes:
                if self.verbose > 0:
                    logger.info(
                        "[StrictSuccessStoppingCallback] Stopping training after %d "
                        "consecutive strict-success episodes.", self._consecutive_successes
                    )
                return False

        return True
