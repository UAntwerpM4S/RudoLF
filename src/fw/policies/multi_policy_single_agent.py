"""
Single agent with N switchable policies (one per scenario).

The environment appends a one-hot vector to the observation selecting the
acting policy, e.g. for N=4: [1, 0, 0, 0] is valid, [0, 0, 0, 0] and
[1, 1, 0, 0] are not. During training, trajectories are split at scenario
switches and each policy trains only on its own segments, with per-segment
GAE bootstrapping.

Usage:
    agent = MultiPolicySingleAgent(
        environment=my_env,        # env that appends one-hot to obs
        num_policies=4,            # number of switchable policies
        policy_labels=["path-following", "head-on", "overtaking", "crossing-right"],
        ...                        # standard PPO hyperparameters
    )
    agent.learn(stop_condition=...)
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import zipfile
import tempfile
import logging

from collections import deque
from fw.configuration.stop_condition import StopCondition
from fw.policies.ppo_policy import PPOPolicy
from fw.policies.base_ppo_agent import BasePPOAgent
from typing import Optional, Tuple, Union, List, Dict
from fw.common.tools.logging_utils import dump_metrics_to_csv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from fw.configuration.profile_applier import SimulatorApplier
from fw.common.tools.ppo_utils import reseed_env_rng

logger = logging.getLogger(__name__)


class CollapseWatchdog:
    """
    Per-policy collapse detector with weights-only rollback.

    Tracks a rolling mean of the per-iteration reward per policy and snapshots
    the network state_dict at every new peak. It triggers when
    smoothed / peak < ratio_threshold (typically negative: -1.75 means the
    current mean is at -175 % of the peak) and returns the snapshot. The caller
    loads it and should also reset the optimizer, otherwise Adam's moment
    buffers push the weights back along the failure path (rollback ping-pong).

    Safeguards: warmup_iterations (no rollback before), min_gap_between_rollbacks
    (cooldown), max_rollbacks (cap per policy), min_peak_for_trigger (the ratio
    test is ill-defined for non-positive peaks).
    """

    def __init__(
        self,
        num_policies: int,
        policy_labels: List[str],
        window_size: int = 50,
        ratio_threshold: float = -1.0,
        warmup_iterations: int = 200,
        min_gap_between_rollbacks: int = 5,
        max_rollbacks: int = 100,
        min_peak_for_trigger: float = 0.1,
    ):
        """
        Args:
            num_policies: Number of policies to monitor.
            policy_labels: Policy names (logging).
            window_size: Rolling window for the smoothed reward.
            ratio_threshold: Trigger when smoothed / peak falls below this.
            warmup_iterations: Iterations before rollbacks are allowed.
            min_gap_between_rollbacks: Minimum iterations between rollbacks per policy.
            max_rollbacks: Maximum rollbacks per policy.
            min_peak_for_trigger: Peak smoothed reward required before checking.
        """

        self.num_policies = num_policies
        self.policy_labels = policy_labels
        self.window_size = int(window_size)
        self.ratio_threshold = float(ratio_threshold)
        self.warmup_iterations = int(warmup_iterations)
        self.min_gap_between_rollbacks = int(min_gap_between_rollbacks)
        self.max_rollbacks = int(max_rollbacks)
        self.min_peak_for_trigger = float(min_peak_for_trigger)

        self._reward_history: Dict[int, deque] = {
            pid: deque(maxlen=self.window_size) for pid in range(num_policies)
        }
        self._best_smoothed: Dict[int, float] = {
            pid: float("-inf") for pid in range(num_policies)
        }
        self._best_state_dict: Dict[int, Optional[dict]] = {
            pid: None for pid in range(num_policies)
        }
        self._last_rollback_iter: Dict[int, int] = {
            pid: -10 ** 9 for pid in range(num_policies)
        }
        self._rollback_count: Dict[int, int] = {
            pid: 0 for pid in range(num_policies)
        }

    def observe(
        self,
        pid: int,
        iteration: int,
        reward: float,
        network: nn.Module,
    ) -> Optional[dict]:
        """
        Record this iteration's reward for pid, snapshot network at a new peak,
        and return the best state_dict if a collapse is detected (else None).
        A NaN reward (no segments collected) is ignored.
        """

        if reward is None or np.isnan(reward):
            return None

        self._reward_history[pid].append(float(reward))

        # Need a full window first
        if len(self._reward_history[pid]) < self.window_size:
            return None

        smoothed = float(np.mean(self._reward_history[pid]))

        # New peak: snapshot the network
        if smoothed > self._best_smoothed[pid]:
            self._best_smoothed[pid] = smoothed
            self._best_state_dict[pid] = {
                k: v.detach().clone() for k, v in network.state_dict().items()
            }

        if iteration < self.warmup_iterations:
            return None

        if self._rollback_count[pid] >= self.max_rollbacks:
            return None

        # Peak too small or negative: ratio test meaningless
        peak = self._best_smoothed[pid]
        if peak <= self.min_peak_for_trigger or self._best_state_dict[pid] is None:
            return None

        # Cooldown
        if iteration - self._last_rollback_iter[pid] < self.min_gap_between_rollbacks:
            return None

        # E.g. ratio_threshold = -1.75, peak 0.7: smoothed < -1.225 triggers.
        if smoothed / peak < self.ratio_threshold:
            return self._best_state_dict[pid]

        return None

    def record_rollback(self, pid: int, iteration: int) -> None:
        """
        Record a rollback for pid and clear its reward window, so collapsed
        iterations do not drag down the new mean. The best snapshot is kept.
        """

        self._rollback_count[pid] += 1
        self._last_rollback_iter[pid] = iteration
        self._reward_history[pid].clear()


class MultiPolicySingleAgent(BasePPOAgent):
    """
    Single-agent PPO with N scenario-conditioned policies.

    The agent strips the one-hot suffix (length num_policies) from each
    observation and routes the base observation to the selected policy.
    Trajectories are split into segments per active policy; at a scenario
    switch the outgoing policy's value bootstraps GAE (mini-episode boundary).
    Each policy trains only on its own segments.
    """

    MINIMUM_ITERATIONS_FOR_BEST_SAVE = 20

    def __init__(
        self,
        *args,
        num_policies: int,
        policy_labels: Optional[List[str]] = None,
        steps_per_policy_update: int = 4096,
        watchdog_enabled: bool = True,
        watchdog_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Args:
            *args, **kwargs: Forwarded to BasePPOAgent.
            num_policies: Number of policies; equals the env's one-hot length.
            policy_labels: Policy names; default ["policy_0", ...].
            steps_per_policy_update: Segment steps a slot accumulates (over one or
                more iterations) before its PPO update runs on the buffered
                segments. No off-policy bias: a slot's networks do not change
                between its updates.
            watchdog_enabled: Use a CollapseWatchdog (per-policy rollback).
            watchdog_kwargs: CollapseWatchdog overrides (ignored if disabled).

        Raises:
            ValueError: If num_policies < 2 or the observation space cannot hold
                both base features and the one-hot.
            NotImplementedError: If the action space is not a Box.
        """

        super().__init__(*args, **kwargs)

        if num_policies < 2:
            raise ValueError(
                f"num_policies must be >= 2, got {num_policies}.  "
                f"Use SinglePPOAgent for a single policy."
            )

        self.num_policies = num_policies
        self.policy_labels = policy_labels or [f"policy_{i}" for i in range(num_policies)]
        self.frozen_pids: set = set()

        if len(self.policy_labels) != num_policies:
            raise ValueError(
                f"Length of policy_labels ({len(self.policy_labels)}) "
                f"must match num_policies ({num_policies})."
            )

        full_obs_dim = self.env.observation_space.shape[0]
        self.policy_obs_dim = full_obs_dim - num_policies

        if self.policy_obs_dim <= 0:
            raise ValueError(
                f"Observation space dim ({full_obs_dim}) must be larger than "
                f"num_policies ({num_policies}).  The last {num_policies} dims "
                f"are reserved for the one-hot scenario encoding."
            )

        action_space = self.env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            # The policy is a tanh-squashed Gaussian (continuous outputs in [-1, 1]).
            raise NotImplementedError(
                f"MultiPolicySingleAgent supports only Box action spaces, got {action_space}."
            )
        self.output_dim = action_space.shape[0]

        self.action_space = action_space

        # N independent policies and optimizers
        self.policies: List[PPOPolicy] = [
            PPOPolicy(self.policy_obs_dim, self.output_dim, str(self.device))
            for _ in range(num_policies)
        ]

        self.optimizers: List[optim.Adam] = [
            optim.Adam(policy.network.parameters(), lr=self.learning_rate)
            for policy in self.policies
        ]

        # Vectorized environment, created lazily
        self._vec_env = None

        # Reward normalization, shared across policies
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4

        # Per-policy pending segments until steps_per_policy_update is reached
        self.steps_per_policy_update = int(steps_per_policy_update)
        self._pending_segments: Dict[int, List[dict]] = {
            k: [] for k in range(num_policies)
        }
        self._pending_steps: Dict[int, int] = {
            k: 0 for k in range(num_policies)
        }

        if watchdog_enabled:
            wd_kwargs = dict(watchdog_kwargs or {})
            self.watchdog: Optional[CollapseWatchdog] = CollapseWatchdog(
                num_policies=num_policies,
                policy_labels=self.policy_labels,
                **wd_kwargs,
            )
        else:
            self.watchdog = None

    # ------------------------------------------------------------------
    # One-hot extraction & validation
    # ------------------------------------------------------------------
    def _extract_scenario(self, obs: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Strip the one-hot suffix and return (base_obs, scenario_id).

        Raises:
            ValueError: If the suffix is not exactly one-hot.
        """

        one_hot = obs[-self.num_policies:]
        base_obs = obs[:-self.num_policies]

        if not np.all((one_hot == 0.0) | (one_hot == 1.0)):
            raise ValueError(
                f"Expected binary (0/1) one-hot encoding in the last "
                f"{self.num_policies} observation dims, got: {one_hot}.  "
                f"Ensure the environment outputs an exact one-hot vector."
            )

        active_count = int(np.sum(one_hot))
        if active_count != 1:
            raise ValueError(
                f"One-hot suffix must have exactly 1 active bit, "
                f"got {active_count} in: {one_hot}."
            )

        scenario_id = int(np.argmax(one_hot))
        return base_obs, scenario_id

    # ------------------------------------------------------------------
    # Prediction interface (required by BasePPOAgent)
    # ------------------------------------------------------------------
    def predict(
        self,
        state: Union[np.ndarray, Tuple[np.ndarray, ...]],
        deterministic: bool = False,
    ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], np.ndarray, np.ndarray]:
        """
        Predict for one observation (with one-hot suffix) using the selected policy.

        Returns:
            (action, log_prob, value)
        """

        base_obs, scenario_id = self._extract_scenario(state)
        actions, log_probs, values = self.policies[scenario_id].predict(
            base_obs, deterministic=deterministic,
        )
        return actions[0], log_probs.item(), values.item()

    def _predict(self, states, deterministic=False):
        """
        Batched prediction for (num_envs, obs_dim) states: envs sharing a scenario
        go through their policy in one call (at most N forward passes per step).

        Returns:
            actions (num_envs, action_dim), raw_actions (num_envs, action_dim),
            log_probs (num_envs,), values (num_envs,).
        """

        states_arr = np.asarray(states)
        num_envs = states_arr.shape[0]

        base_obs_all = states_arr[:, :-self.num_policies]
        one_hot = states_arr[:, -self.num_policies:]

        # Vectorized _extract_scenario validation
        if not np.all((one_hot == 0.0) | (one_hot == 1.0)):
            raise ValueError(
                f"Expected binary (0/1) one-hot encoding in the last "
                f"{self.num_policies} observation dims of every env."
            )
        active_counts = one_hot.sum(axis=1)
        if not np.all(active_counts == 1):
            raise ValueError(
                f"One-hot suffix must have exactly 1 active bit per env; "
                f"got per-env active counts {active_counts.tolist()}."
            )

        scenario_ids = np.argmax(one_hot, axis=1)

        actions = np.empty((num_envs, self.output_dim), dtype=np.float32)
        raw_actions = np.empty((num_envs, self.output_dim), dtype=np.float32)
        log_probs = np.empty(num_envs, dtype=np.float32)
        values = np.empty(num_envs, dtype=np.float32)

        for pid in np.unique(scenario_ids):
            mask = scenario_ids == pid
            a, raw, lp, v = self.policies[int(pid)].predict_with_raw(
                base_obs_all[mask], deterministic=deterministic,
            )
            actions[mask] = a
            raw_actions[mask] = raw
            log_probs[mask] = lp
            values[mask] = v

        return actions, raw_actions, log_probs, values

    def _get_bootstrap_value(self, next_state, env_id: Optional[int] = None):
        """
        Bootstrap value of next_state from the selected policy's critic (env_id unused).
        """

        base_obs, scenario_id = self._extract_scenario(next_state)
        with torch.no_grad():
            state_t = torch.FloatTensor(base_obs).unsqueeze(0).to(self.device)
            _, _, value = self.policies[scenario_id].network(state_t)
            return value.squeeze(0).item()

    # ------------------------------------------------------------------
    # Vectorized environments (as in SinglePPOAgent)
    # ------------------------------------------------------------------
    def make_env(self, worker_seed=None):
        """
        Return a factory for a deep-copied, monitored environment. It captures
        only the env (not the agent), so SubprocVecEnv does not pickle CUDA
        tensors.

        Args:
            worker_seed: SeedSequence or int for the worker's env RNG; None
                spawns a new child of the agent's worker SeedSequence.
        """

        env = self.env
        profile = self._profile
        if worker_seed is None:
            worker_seed = self._spawn_worker_seeds(1)[0]

        def _init():
            """
            Deep-copy the env, apply the profile's simulator settings, reseed, wrap in Monitor.
            """

            env_copy = copy.deepcopy(env)
            if profile is not None:
                _sim = getattr(env_copy, "phys_sim", None) or getattr(env_copy, "_phys_sim", None)
                if _sim is not None:
                    SimulatorApplier.apply(profile, _sim)
            reseed_env_rng(env_copy, worker_seed)
            return Monitor(env_copy)

        return _init

    def _reset_envs(self, num_envs):
        """
        (Re)create the vectorized envs if needed, reset them; returns (num_envs, obs_dim).
        """

        if self._vec_env is None or getattr(self._vec_env, "num_envs", None) != num_envs:
            env_fns = [self.make_env(worker_seed) for worker_seed in self._spawn_worker_seeds(num_envs)]
            self._vec_env = SubprocVecEnv(env_fns) if num_envs > 1 else DummyVecEnv(env_fns)

        if hasattr(self._vec_env, "env_method"):
            self._vec_env.env_method("randomize")

        return self._vec_env.reset()

    def _step_envs(self, actions):
        """
        Step all envs with (num_envs, action_dim) actions.

        Returns:
            next_states, rewards, terminations, truncations, final_states (real
            last observations of finished envs, for bootstrapping).
        """

        next_states, next_rewards, next_terminations, infos = self._vec_env.step(actions)

        if infos and all(
            all(k in env_info for k in ("rewards", "terminations", "truncations"))
            for env_info in infos
        ):
            rewards = np.array([env_info["rewards"] for env_info in infos], dtype=np.float32)
            terminations = np.array([env_info["terminations"] for env_info in infos], dtype=bool)
            truncations = np.array([env_info["truncations"] for env_info in infos], dtype=bool)
        else:
            rewards = next_rewards

            # SB3's done = terminated OR truncated. Split off truncation via
            # "TimeLimit.truncated": a truncated step must be bootstrapped.
            if infos:
                truncations = np.array(
                    [bool(env_info.get("TimeLimit.truncated", False)) for env_info in infos],
                    dtype=bool,
                )
            else:
                truncations = np.zeros(actions.shape[0], dtype=bool)
            terminations = np.asarray(next_terminations, dtype=bool) & ~truncations

        # SB3 auto-resets finished envs; their real last observation is in
        # info["terminal_observation"].
        final_states = np.array(next_states, copy=True)
        if infos:
            dones = np.asarray(next_terminations, dtype=bool)
            for env_id, env_info in enumerate(infos):
                if dones[env_id] and "terminal_observation" in env_info:
                    final_states[env_id] = env_info["terminal_observation"]

        return next_states, rewards, terminations, truncations, final_states

    # ------------------------------------------------------------------
    # Per-segment GAE
    # ------------------------------------------------------------------

    def _split_trajectory_and_compute_gae(
        self, episode: dict
    ) -> Dict[int, List[dict]]:
        """
        Split a trajectory into per-policy segments and compute GAE per segment;
        at a switch the outgoing policy's value of the boundary state bootstraps.

        Args:
            episode: Trajectory from run_parallel_episodes() (states include
                the one-hot suffix).

        Returns:
            {policy_id: [segment]}, each segment with states (one-hot stripped),
            actions, raw_actions, rewards, values, log_probs, advantages, returns.

        Raises:
            KeyError: If the episode has no "raw_actions" (not produced by
                `run_parallel_episodes`).
        """

        if "raw_actions" not in episode:
            raise KeyError(
                "PPO update needs 'raw_actions' (pre-tanh rollout actions) in every "
                "episode; collect episodes with run_parallel_episodes()."
            )

        T = len(episode["rewards"])

        # Scenario per timestep
        scenario_ids = np.array([
            int(np.argmax(episode["states"][t, -self.num_policies:]))
            for t in range(T)
        ])

        # Contiguous segments: (start, end_exclusive, policy_id)
        segments: List[Tuple[int, int, int]] = []
        seg_start = 0
        for t in range(1, T):
            if scenario_ids[t] != scenario_ids[seg_start]:
                segments.append((seg_start, t, int(scenario_ids[seg_start])))
                seg_start = t
        segments.append((seg_start, T, int(scenario_ids[seg_start])))

        policy_segments: Dict[int, List[dict]] = {
            k: [] for k in range(self.num_policies)
        }

        for i, (start, end, policy_id) in enumerate(segments):
            seg_states = episode["states"][start:end, :-self.num_policies]
            seg_actions = episode["actions"][start:end]
            seg_raw_actions = episode["raw_actions"][start:end]
            seg_rewards = episode["rewards"][start:end]
            seg_values = episode["values"][start:end]
            seg_log_probs = episode["log_probs"][start:end]
            seg_terminations = episode["terminations"][start:end]

            # Bootstrap value at the end of the segment
            is_last_segment = (i == len(segments) - 1)
            next_value = 0.0

            if is_last_segment:
                last_terminated = bool(episode["terminations"][end - 1])
                if not last_terminated:
                    # Truncated: bootstrap from the state after the segment
                    next_state = episode["next_states"][end - 1]
                    base_next = next_state[:-self.num_policies]
                    with torch.no_grad():
                        st = torch.FloatTensor(base_next).unsqueeze(0).to(self.device)
                        _, _, v = self.policies[policy_id].network(st)
                        next_value = v.squeeze(0).item()
            else:
                # Scenario switch: outgoing policy's value of the next segment's first state
                boundary_state = episode["states"][end]
                base_boundary = boundary_state[:-self.num_policies]
                with torch.no_grad():
                    st = torch.FloatTensor(base_boundary).unsqueeze(0).to(self.device)
                    _, _, v = self.policies[policy_id].network(st)
                    next_value = v.squeeze(0).item()

            advantages = self.compute_gae(
                seg_rewards,
                seg_values,
                seg_terminations,
                next_value,
                self.gamma,
                self.gae_lambda,
            )

            returns = advantages + seg_values

            policy_segments[policy_id].append({
                "states": seg_states,
                "actions": seg_actions,
                "raw_actions": seg_raw_actions,
                "rewards": seg_rewards,
                "values": seg_values,
                "log_probs": seg_log_probs,
                "advantages": advantages,
                "returns": returns,
            })

        return policy_segments

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train_policy(self, policy_id: int, segments: List[dict]) -> Tuple[float, float, float, float]:
        """
        PPO update of one policy on its segments (as SinglePPOAgent.train()).

        Returns:
            Mean total loss, policy loss, value loss and entropy bonus over all
            minibatches; all NaN if the slot is frozen or has no usable data.
        """

        if policy_id in self.frozen_pids:
            return float("nan"), float("nan"), float("nan"), float("nan")

        if not segments:
            return float("nan"), float("nan"), float("nan"), float("nan")

        policy = self.policies[policy_id]
        optimizer = self.optimizers[policy_id]
        policy.network.train()

        all_states = torch.as_tensor(
            np.concatenate([s["states"] for s in segments]),
            dtype=torch.float32,
        ).to(self.device)

        all_raw_actions = torch.as_tensor(
            np.concatenate([s["raw_actions"] for s in segments]),
            dtype=torch.float32,
        ).to(self.device)

        all_advantages = torch.as_tensor(
            np.concatenate([s["advantages"] for s in segments]),
            dtype=torch.float32,
        ).to(self.device)

        all_old_log_probs = torch.as_tensor(
            np.concatenate([s["log_probs"] for s in segments]),
            dtype=torch.float32,
        ).to(self.device)

        all_returns = torch.as_tensor(
            np.concatenate([s["returns"] for s in segments]),
            dtype=torch.float32,
        ).to(self.device)

        if all_advantages.numel() <= 1:
            return float("nan"), float("nan"), float("nan"), float("nan")

        # One NaN/Inf would permanently corrupt the slot's weights.
        for name, tensor in (
            ("states", all_states),
            ("advantages", all_advantages),
            ("returns", all_returns),
        ):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.warning(
                    "[Alert] NaN/Inf detected in %s of policy '%s' before PPO update - skipping batch.",
                    name, self.policy_labels[policy_id],
                )
                return float("nan"), float("nan"), float("nan"), float("nan")

        # Normalize advantages (skipped if near-constant).
        if all_advantages.std() > 1e-6:
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        batch_size = all_states.shape[0]
        loss_per_epoch = []
        policy_loss_per_mb = []
        value_loss_per_mb = []
        entropy_bonus_per_mb = []

        for _ in range(self.num_epochs):
            indices = torch.randperm(batch_size)

            for mb_start in range(0, batch_size, self.batch_size):
                mb_end = mb_start + self.batch_size
                mb_idx = indices[mb_start:mb_end]

                mb_states = all_states[mb_idx]
                mb_raw_actions = all_raw_actions[mb_idx]
                mb_old_log_probs = all_old_log_probs[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_advantages = all_advantages[mb_idx]

                action_mean, action_std, state_values = policy.network(mb_states)
                dist = torch.distributions.Normal(action_mean, action_std)

                # Same evaluation point as in the rollout (PPOPolicy.raw_log_prob).
                log_probs = policy.raw_log_prob(dist, mb_raw_actions)

                # PPO clipped objective
                log_ratios = torch.clamp(log_probs - mb_old_log_probs, max=PPOPolicy.MAX_LOG_RATIO)
                ratios = torch.exp(log_ratios)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(state_values.squeeze(-1), mb_returns)
                entropy_loss = dist.entropy().sum(dim=-1).mean()

                entropy_bonus = self.entropy_coef * entropy_loss
                loss = policy_loss + (self.value_loss_coef * value_loss) - entropy_bonus

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        "[Alert] NaN/Inf loss for policy '%s' (policy=%.4f, value=%.4f) - skipping update step.",
                        self.policy_labels[policy_id], policy_loss.item(), value_loss.item(),
                    )
                    continue

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(policy.network.parameters(), self.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    # A non-finite norm would turn every weight of the slot into NaN.
                    logger.warning(
                        "[Alert] Non-finite gradient norm (%s) for policy '%s' - skipping update step.",
                        float(grad_norm), self.policy_labels[policy_id],
                    )
                    optimizer.zero_grad()
                    continue
                optimizer.step()

                loss_per_epoch.append(loss.item())
                policy_loss_per_mb.append(policy_loss.item())
                value_loss_per_mb.append(value_loss.item())
                entropy_bonus_per_mb.append(entropy_bonus.item())

        if not loss_per_epoch:
            return float("nan"), float("nan"), float("nan"), float("nan")

        return (
            float(np.mean(loss_per_epoch)),
            float(np.mean(policy_loss_per_mb)),
            float(np.mean(value_loss_per_mb)),
            float(np.mean(entropy_bonus_per_mb)),
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
    ) -> None:
        """
        Training loop: collect, split by scenario, per-segment GAE, train each policy.
        """

        logger.info(
            "Starting multi-policy training with %d policies, %d parallel env(s) on '%s'",
            self.num_policies, self.num_envs, self.device
        )

        last_save_index = 0
        stopping_condition = stop_condition or StopCondition()

        # Per-policy tracking
        best_rewards: Dict[int, float] = {k: float("-inf") for k in range(self.num_policies)}
        training_metrics: Dict[str, list] = {
            label: [] for label in self.policy_labels
        }

        best_policy_files = [
            f"Best_{self.env.training_name}_{label}_policy"
            for label in self.policy_labels
        ]

        try:
            for iteration in range(stopping_condition.max_time_steps):
                self.set_policy_eval()

                # 1. Collect (routing via _predict)
                batch_data = self.run_parallel_episodes(self.num_envs)

                # 2. Split trajectories, per-segment GAE
                accumulated_segments: Dict[int, List[dict]] = {
                    k: [] for k in range(self.num_policies)
                }

                for episode in batch_data:
                    policy_segments = self._split_trajectory_and_compute_gae(episode)
                    for pid, segs in policy_segments.items():
                        accumulated_segments[pid].extend(segs)

                # 3. Train each policy
                iter_rewards = {}
                iter_losses = {}
                iter_policy_losses = {}
                iter_value_losses = {}
                iter_entropy_bonuses = {}

                for pid in range(self.num_policies):
                    segs = accumulated_segments[pid]

                    self._pending_segments[pid].extend(segs)
                    self._pending_steps[pid] += sum(len(s["rewards"]) for s in segs)

                    if self._pending_steps[pid] >= self.steps_per_policy_update:
                        update_segs = self._pending_segments[pid]
                        loss, policy_loss, value_loss, entropy_bonus = self._train_policy(
                            pid, update_segs,
                        )
                        flushed_steps = self._pending_steps[pid]
                        self._pending_segments[pid] = []
                        self._pending_steps[pid] = 0
                    else:
                        loss = policy_loss = value_loss = entropy_bonus = float("nan")
                        flushed_steps = 0

                    iter_losses[pid] = loss
                    iter_policy_losses[pid] = policy_loss
                    iter_value_losses[pid] = value_loss
                    iter_entropy_bonuses[pid] = entropy_bonus

                    # Reward of this iteration's segments only (not the pending buffer).
                    if segs:
                        all_seg_rewards = np.concatenate([s["rewards"] for s in segs])
                        iter_rewards[pid] = float(np.mean(all_seg_rewards))
                    else:
                        iter_rewards[pid] = float("nan")

                    training_metrics[self.policy_labels[pid]].append({
                        "iteration": iteration,
                        "loss": loss,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "entropy_bonus": entropy_bonus,
                        "reward": iter_rewards[pid],
                        "n_steps": sum(len(s["rewards"]) for s in segs),
                        "pending_steps": self._pending_steps[pid],
                        "flushed_steps": flushed_steps,
                    })

                    # Save the best policy per scenario, only on iterations with an update.
                    if (
                        flushed_steps > 0
                        and iteration >= self.MINIMUM_ITERATIONS_FOR_BEST_SAVE
                        and not np.isnan(iter_rewards[pid])
                        and iter_rewards[pid] > best_rewards[pid]
                    ):
                        best_rewards[pid] = iter_rewards[pid]
                        self._save_single_policy(pid, best_policy_files[pid])

                    # Collapse watchdog (not for frozen slots).
                    rollback_triggered = False
                    if self.watchdog is not None and pid not in self.frozen_pids:
                        rollback_state = self.watchdog.observe(
                            pid=pid,
                            iteration=iteration,
                            reward=iter_rewards[pid],
                            network=self.policies[pid].network,
                        )
                        if rollback_state is not None:
                            self.policies[pid].network.load_state_dict(rollback_state)
                            # Fresh optimizer: Adam's moments from the collapse
                            # would push the weights straight back (ping-pong).
                            self.optimizers[pid] = optim.Adam(
                                self.policies[pid].network.parameters(),
                                lr=self.learning_rate,
                            )
                            # Pending segments are off-policy for the rolled-back net.
                            self._pending_segments[pid] = []
                            self._pending_steps[pid] = 0
                            self.watchdog.record_rollback(pid, iteration)
                            rollback_triggered = True
                            logger.warning(
                                "[watchdog] iter %d: collapse detected for '%s' "
                                "(peak smoothed=%.3f, rollback %d/%d). "
                                "Rolled back to best rolling-mean snapshot.",
                                iteration, self.policy_labels[pid],
                                self.watchdog._best_smoothed[pid],
                                self.watchdog._rollback_count[pid],
                                self.watchdog.max_rollbacks,
                            )

                    if training_metrics[self.policy_labels[pid]]:
                        training_metrics[self.policy_labels[pid]][-1][
                            "rollback_triggered"
                        ] = rollback_triggered

                # 4. Logging
                valid_rewards = [r for r in iter_rewards.values() if not np.isnan(r)]
                valid_losses = [l for l in iter_losses.values() if not np.isnan(l)]
                agg_reward = float(np.mean(valid_rewards)) if valid_rewards else 0.0
                agg_loss = float(np.mean(valid_losses)) if valid_losses else float("nan")

                if True: # iteration % self.eval_frequency == 0
                    parts = []
                    for pid in range(self.num_policies):
                        label = self.policy_labels[pid]
                        r = iter_rewards[pid]
                        pl = iter_policy_losses[pid]
                        vl = iter_value_losses[pid]
                        eb = iter_entropy_bonuses[pid]
                        n = sum(len(s["rewards"]) for s in accumulated_segments[pid])
                        pending = self._pending_steps[pid]
                        upd = "" if np.isnan(pl) else " [upd]"
                        parts.append(
                            f"{label}: r={r:.3f}, loss: p={pl:.3f}, v={vl:.3f}, e={eb:.3f}, "
                            f"steps={n}, pending={pending}{upd}"
                        )
                    logger.info("[Iter %d] %s", iteration, " | ".join(parts))

                # Periodic checkpoint (all policies in one ZIP)
                if iteration > 0 and iteration % self.eval_frequency == 0:
                    self.save_policy(f"multi_policy_checkpoint_{iteration}")

                # 5. Stopping condition
                should_stop, reason = stopping_condition.should_stop(
                    current_step=iteration,
                    mean_reward=agg_reward,
                    mean_loss=agg_loss,
                )

                if should_stop:
                    logger.info("Stopping: %s", reason)
                    last_save_index = iteration
                    break

            # Final checkpoint and metrics, only after a normal end.
            self.save_policy(
                f"multi_policy_checkpoint_"
                f"{last_save_index if last_save_index > 0 else stopping_condition.max_time_steps}"
            )
            self._dump_training_metrics(training_metrics)
        finally:
            # Always shut down the workers, also on an exception or Ctrl-C.
            if self._vec_env is not None:
                try:
                    self._vec_env.close()
                except Exception:
                    logger.exception("Closing the VecEnv failed.")
                self._vec_env = None

        logger.info("Multi-policy training completed!")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_policy(self, policy_file_name: str) -> None:
        """
        Save all N policies into one ZIP (policy_0.pth ... policy_{N-1}.pth,
        serialized PPOPolicy). policy_file_name is given without extension.
        """

        if not policy_file_name.endswith(".zip"):
            policy_file_name += ".zip"

        full_path = (
            os.path.join(self.model_dir, policy_file_name)
            if self.model_dir
            else policy_file_name
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                for i, policy in enumerate(self.policies):
                    agent_path = os.path.join(tmp_dir, f"policy_{i}.pth")
                    torch.save(policy.serialize_policy(), agent_path)

                with zipfile.ZipFile(full_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for i in range(self.num_policies):
                        fname = f"policy_{i}.pth"
                        zf.write(os.path.join(tmp_dir, fname), arcname=fname)

                #print(f"[save] Saved multi-policy archive to {full_path}")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to save multi-policy archive to '{full_path}'."
                ) from e

    def load_policy(self, policy_file_name: str) -> None:
        """
        Load N policies from a ZIP archive (".zip" added if missing) and rebuild
        the optimizers.

        Raises:
            FileNotFoundError: If the archive is missing.
            RuntimeError: If loading fails or required files are absent.
        """

        if not policy_file_name.endswith(".zip"):
            policy_file_name += ".zip"

        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        if not os.path.exists(policy_file_name):
            raise FileNotFoundError(f"Policy archive not found: {policy_file_name}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                with zipfile.ZipFile(policy_file_name, "r") as zf:
                    expected = {f"policy_{i}.pth" for i in range(self.num_policies)}
                    missing = expected - set(zf.namelist())
                    if missing:
                        raise RuntimeError(f"Archive missing files: {missing}")
                    zf.extractall(tmp_dir)

                for i in range(self.num_policies):
                    agent_path = os.path.join(tmp_dir, f"policy_{i}.pth")
                    saved = torch.load(agent_path, map_location=self.device, weights_only=False)
                    self.policies[i] = PPOPolicy.build_policy_from_state(saved, self.device)

                self.optimizers = [
                    optim.Adam(p.network.parameters(), lr=self.learning_rate)
                    for p in self.policies
                ]

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load multi-policy archive: {e}"
                ) from e

    def load_individual_policy(
        self,
        policy_id: int,
        policy_file_name: str,
        frozen: bool = False,
    ) -> None:
        """
        Load a pre-trained SinglePPOAgent-style .zip policy into slot policy_id
        (0..N-1); frozen slots are excluded from PPO updates.

        Raises:
            IndexError: If policy_id is out of range.
        """

        if not 0 <= policy_id < self.num_policies:
            raise IndexError(
                f"policy_id {policy_id} out of range [0, {self.num_policies})"
            )

        loaded = PPOPolicy.load(policy_file_name, str(self.device))
        self.policies[policy_id] = loaded
        self.optimizers[policy_id] = optim.Adam(
            loaded.network.parameters(), lr=self.learning_rate,
        )

        if frozen:
            self.frozen_pids.add(policy_id)
        else:
            self.frozen_pids.discard(policy_id)

    def load_pretrained_policies(
        self,
        directory: str,
        specs: Dict[int, dict],
    ) -> None:
        """
        Load pre-trained policies from directory.

        Args:
            directory: Directory with the .zip policy files.
            specs: {policy_id: {"file": name without .zip, "frozen": bool (default False)}}.

        Slots whose file is missing are left unchanged (a warning is logged).
        """

        for pid, spec in specs.items():
            file_stem = spec["file"]
            frozen = bool(spec.get("frozen", False))
            full_path = os.path.join(directory, file_stem)
            zip_path = full_path if full_path.endswith(".zip") else full_path + ".zip"

            if not os.path.exists(zip_path):
                logger.warning("[pretrained] No file at %s, slot %d left unchanged.", zip_path, pid)
                continue

            self.load_individual_policy(pid, full_path, frozen=frozen)
            label = self.policy_labels[pid]
            state = "frozen" if frozen else "trainable"
            logger.info("[pretrained] Loaded '%s' into slot %d (%s) [%s]", file_stem, pid, label, state)

    def _save_single_policy(self, policy_id: int, policy_file_name: str) -> None:
        """
        Save one slot as a standalone SinglePPOAgent-format file (name without extension).
        """

        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        self.policies[policy_id].save(policy_file_name)
        #print(f"[save] Saved {self.policy_labels[policy_id]} policy to {policy_file_name}.zip")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def set_policy_eval(self):
        """
        Set all policy networks to evaluation mode.
        """

        for policy in self.policies:
            policy.network.eval()

    def _dump_training_metrics(self, metrics: Dict[str, list]) -> None:
        """
        Flatten per-policy metrics and write them to CSV.
        """

        flat = []
        for label, records in metrics.items():
            for record in records:
                flat.append({"policy": label, **record})

        if flat:
            dump_metrics_to_csv(self.model_dir, "multi_policy_training_metrics.csv", flat)
