"""
Single-agent with N switchable policies (one per scenario).

The environment must append a one-hot encoding to the observation
indicating which policy should act at each timestep.
For example (N=4):
- Good usage: [1, 0, 0, 0]
- Bad usage: [0, 0, 0, 0]
- Bad usage: [1, 1, 0, 0]
During training, trajectories are split at scenario-switch boundaries and each policy
receives only the segments that belong to it, with correct per-segment
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

from collections import deque
from typing import Optional, Tuple, Union, List, Dict
from fw.stop_condition import StopCondition
from fw.policies.ppo_policy import PPOPolicy
from fw.policies.base_ppo_agent import BasePPOAgent
from fw.common.tools.logging_utils import dump_metrics_to_csv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


class CollapseWatchdog:
    """
    Tracks a rolling mean of per-iteration reward for each policy.  Whenever
    the rolling mean reaches a new peak, the policy's network ``state_dict``
    is snapshotted in memory (deep-cloned tensors).  After a warmup period,
    the current rolling mean is compared against the peak rolling mean.  The
    watchdog triggers when ``smoothed / peak < ratio_threshold``, where
    ``ratio_threshold`` is typically negative (e.g. ``-1.75`` means "current
    is at -175% of peak").

    On trigger, the watchdog returns the snapshotted ``state_dict``; the
    caller is responsible for applying it via ``network.load_state_dict``
    and (recommended) for re-initializing the corresponding optimizer so
    that Adam's ``m``/``v`` moment buffers to avoid pushing the rolled-back
    weights back along the failure trajectory. Without an optimizer reset,
    weights-only rollback can produce a rollback ping-pong on policies with
    high reward variance.

    Safeguards:
      * ``warmup_iterations``: no rollback before this many iterations have
        elapsed.
      * ``min_gap_between_rollbacks``: enforce a cooldown for each policy.
      * ``max_rollbacks``: cap the total rollbacks per policy.
      * ``min_peak_for_trigger``: require the peak rolling mean to be above
        this floor before the ratio test is meaningful (the formulation is
        ill-defined for non-positive peaks).
    """

    def __init__(
        self,
        num_policies: int,
        policy_labels: List[str],
        window_size: int = 6,
        ratio_threshold: float = -1.0,
        warmup_iterations: int = 50,
        min_gap_between_rollbacks: int = 5,
        max_rollbacks: int = 100,
        min_peak_for_trigger: float = 0.1,
    ):
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
        Record this iteration's reward for ``pid`` and, if appropriate,
        either snapshot a new best or return a rollback ``state_dict``.

        Args:
            pid (int): Policy index.
            iteration (int): Current training iteration.
            reward (float): Mean reward for ``pid`` this iteration (NaN if
                the policy collected no segments — will be ignored).
            network (nn.Module): The policy's live network; snapshotted
                (cloned) when a new peak rolling mean is observed.

        Returns:
            Optional[dict]: A cloned ``state_dict`` to load if a collapse
                is detected; ``None`` otherwise.
        """
        if reward is None or np.isnan(reward):
            return None

        self._reward_history[pid].append(float(reward))

        # Need a full window before the rolling mean is meaningful
        if len(self._reward_history[pid]) < self.window_size:
            return None

        smoothed = float(np.mean(self._reward_history[pid]))

        # New best — snapshot the current network state
        if smoothed > self._best_smoothed[pid]:
            self._best_smoothed[pid] = smoothed
            self._best_state_dict[pid] = {
                k: v.detach().clone() for k, v in network.state_dict().items()
            }

        # Watchdog disabled until warmup elapses
        if iteration < self.warmup_iterations:
            return None

        # Out of rollback budget
        if self._rollback_count[pid] >= self.max_rollbacks:
            return None

        # No usable peak yet (peak too small / negative — ratio test
        # would be meaningless or trigger-happy)
        peak = self._best_smoothed[pid]
        if peak <= self.min_peak_for_trigger or self._best_state_dict[pid] is None:
            return None

        # Cooldown after a previous rollback for this policy
        if iteration - self._last_rollback_iter[pid] < self.min_gap_between_rollbacks:
            return None

        # Trigger condition: current rolling mean is far below the peak.
        # E.g. ratio_threshold = -1.75 means "smoothed must be below -175%
        # of peak" (so for peak=0.7, smoothed < -1.225 triggers).
        if smoothed / peak < self.ratio_threshold:
            return self._best_state_dict[pid]

        return None

    def record_rollback(self, pid: int, iteration: int) -> None:
        """
        Mark a rollback as having occurred for ``pid`` at ``iteration``.

        Clears the rolling reward history so that the post-rollback
        rolling mean is not dragged down by the collapsed iterations
        still sitting in the window.  The best snapshot itself is
        retained: subsequent improvements may still surpass it.
        """
        self._rollback_count[pid] += 1
        self._last_rollback_iter[pid] = iteration
        self._reward_history[pid].clear()


class MultiPolicySingleAgent(BasePPOAgent):
    """
    Single-agent PPO with N scenario-conditioned policies.

    At each timestep the environment appends a one-hot vector of length
    ``num_policies`` to the observation.  The agent strips this suffix,
    determines which policy should act, and forwards only the base
    observation to that policy's actor-critic network.

    During training, each collected trajectory is split into contiguous
    segments where the same policy was active.  At segment boundaries
    (scenario switches), the outgoing policy's value estimate is used for
    GAE bootstrapping — treating the switch as a mini-episode boundary.
    Each policy is then trained independently on only its own segments.

    This design is fully generalizable to any number of policies N.
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
        Initialize the multi-policy single agent.

        Args:
            *args: Positional arguments forwarded to ``BasePPOAgent``.
            num_policies (int): Number of switchable policies.  Must match
                the length of the one-hot suffix appended by the environment.
            policy_labels (list[str], optional): Human-readable names for
                each policy.  Defaults to ``["policy_0", ...]``.
            steps_per_policy_update (int): Minimum number of segment steps a
                slot must have accumulated (across one or more iterations)
                before its PPO update fires. Iterations where a slot is
                below threshold append the new segments to a per-slot
                pending buffer and skip the optimizer step. When the
                threshold is reached, the buffered segments are flushed
                through one PPO update and the buffer is cleared.
                Because a slot's policy/value networks do not change
                between updates, accumulating across iterations does not
                introduce off-policy bias for that slot.
            watchdog_enabled (bool): If True, instantiate a
                ``CollapseWatchdog`` that monitors per-policy rolling-mean
                reward and rolls back to the best snapshot when a collapse
                is detected.  Per-policy and weights-only.
            watchdog_kwargs (Optional[dict]): Override defaults for the
                ``CollapseWatchdog`` (``window_size``, ``ratio_threshold``,
                ``warmup_iterations``, ``min_gap_between_rollbacks``,
                ``max_rollbacks``, ``min_peak_for_trigger``).  Ignored if
                ``watchdog_enabled`` is False.
            **kwargs: Keyword arguments forwarded to ``BasePPOAgent``.

        Raises:
            ValueError: If ``num_policies < 2`` or the observation space is
                too small to contain both base features and the one-hot.
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

        # Observation / action dimensions
        full_obs_dim = self.env.observation_space.shape[0]
        self.policy_obs_dim = full_obs_dim - num_policies

        if self.policy_obs_dim <= 0:
            raise ValueError(
                f"Observation space dim ({full_obs_dim}) must be larger than "
                f"num_policies ({num_policies}).  The last {num_policies} dims "
                f"are reserved for the one-hot scenario encoding."
            )

        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            self.output_dim = action_space.n
        else:
            self.output_dim = action_space.shape[0]

        self.action_space = action_space

        # Create N independent policies and optimizers
        self.policies: List[PPOPolicy] = [
            PPOPolicy(self.policy_obs_dim, self.output_dim, str(self.device))
            for _ in range(num_policies)
        ]

        self.optimizers: List[optim.Adam] = [
            optim.Adam(policy.network.parameters(), lr=self.learning_rate)
            for policy in self.policies
        ]

        # Vectorized environment handle (lazy-init like SinglePPOAgent)
        self._vec_env = None

        # Reward normalization (shared across policies — same environment)
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4

        # Per-policy update gating: accumulate segments across iterations
        # until the slot's pending step count crosses
        # ``steps_per_policy_update``, then flush through one PPO update.
        self.steps_per_policy_update = int(steps_per_policy_update)
        self._pending_segments: Dict[int, List[dict]] = {
            k: [] for k in range(num_policies)
        }
        self._pending_steps: Dict[int, int] = {
            k: 0 for k in range(num_policies)
        }

        # Collapse watchdog (per-policy, weights-only rollback)
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
        Strip the one-hot suffix and return ``(base_obs, scenario_id)``.

        Args:
            obs (np.ndarray): Full observation including one-hot suffix.

        Returns:
            Tuple[np.ndarray, int]: Base observation and active scenario index.

        Raises:
            ValueError: If the one-hot suffix contains values other than 0/1
                or does not have exactly one active bit.
        """

        one_hot = obs[-self.num_policies:]
        base_obs = obs[:-self.num_policies]

        # Validate binary encoding
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
        Predict an action for a single observation by routing to the
        appropriate policy based on the one-hot scenario suffix.

        Args:
            state (np.ndarray): Single observation (with one-hot suffix).
            deterministic (bool): Whether to use deterministic actions.

        Returns:
            Tuple[np.ndarray, float, float]:
                - action, log_prob, value estimate.
        """

        base_obs, scenario_id = self._extract_scenario(state)
        actions, log_probs, values = self.policies[scenario_id].predict(
            base_obs, deterministic=deterministic,
        )
        return actions[0], log_probs.item(), values.item()

    def _predict(self, states, deterministic=False):
        """
        Batched prediction across parallel environments.

        Each environment may be in a different scenario, so predictions
        are routed per-environment.

        Args:
            states (np.ndarray): Shape ``(num_envs, obs_dim)``.
            deterministic (bool): Whether to use deterministic actions.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                actions, log_probs, values (all shape ``(num_envs, ...)``).
        """

        num_envs = len(states)
        all_actions = []
        all_log_probs = []
        all_values = []

        for i in range(num_envs):
            base_obs, scenario_id = self._extract_scenario(states[i])
            actions, log_probs, values = self.policies[scenario_id].predict(
                base_obs, deterministic=deterministic,
            )
            all_actions.append(actions[0])
            all_log_probs.append(log_probs.item())
            all_values.append(values.item())

        return (
            np.stack(all_actions),
            np.array(all_log_probs, dtype=np.float32),
            np.array(all_values, dtype=np.float32),
        )

    def _get_bootstrap_value(self, next_state, env_id: Optional[int] = None):
        """
        Estimate value for bootstrapping using the correct policy's critic.

        Args:
            next_state (np.ndarray): State to evaluate.
            env_id (Optional[int]): Unused (single-env agent).

        Returns:
            float: Estimated value.
        """

        base_obs, scenario_id = self._extract_scenario(next_state)
        with torch.no_grad():
            state_t = torch.FloatTensor(base_obs).unsqueeze(0).to(self.device)
            _, _, value = self.policies[scenario_id].network(state_t)
            return value.squeeze(0).item()

    # ------------------------------------------------------------------
    # Vectorized environment management (mirrors SinglePPOAgent)
    # ------------------------------------------------------------------
    def make_env(self):
        """Create a callable factory for deep-copied, monitored environments.

        The closure captures only ``self.env`` (not the full agent) to
        avoid pickling CUDA policy tensors when ``SubprocVecEnv`` sends
        the factory to child processes.
        """

        env = self.env  # capture only the lightweight env, not the agent

        def _init():
            return Monitor(copy.deepcopy(env))

        return _init

    def _reset_envs(self, num_envs):
        """
        Initialize or reset vectorized environments.

        Args:
            num_envs (int): Number of parallel environments.

        Returns:
            np.ndarray: Initial observations ``(num_envs, obs_dim)``.
        """

        if self._vec_env is None or getattr(self._vec_env, "num_envs", None) != num_envs:
            env_fns = [self.make_env() for _ in range(num_envs)]
            self._vec_env = SubprocVecEnv(env_fns) if num_envs > 1 else DummyVecEnv(env_fns)

        if hasattr(self._vec_env, "env_method"):
            self._vec_env.env_method("randomize")

        return self._vec_env.reset()

    def _step_envs(self, actions):
        """
        Step all environments with given actions.

        Args:
            actions (np.ndarray): Shape ``(num_envs, action_dim)``.

        Returns:
            Tuple: next_states, rewards, terminations, truncations.
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
            terminations = next_terminations
            truncations = np.zeros(actions.shape[0], dtype=bool)

        return next_states, rewards, terminations, truncations

    # ------------------------------------------------------------------
    # Per-segment GAE: the core of Option 3
    # ------------------------------------------------------------------

    def _split_trajectory_and_compute_gae(
        self, episode: dict
    ) -> Dict[int, List[dict]]:
        """
        Split a collected trajectory into per-policy segments and compute
        GAE independently for each segment with proper bootstrapping at
        scenario-switch boundaries.

        At each switch point, the outgoing policy's value estimate of the
        boundary state is used as the bootstrap value, treating the switch
        as a mini-episode boundary.

        Args:
            episode (dict): Raw trajectory from ``run_parallel_episodes()``,
                containing keys: states, actions, rewards, values, log_probs,
                terminations, truncations, next_states, advantages, returns,
                env_id.  ``states`` includes the one-hot suffix.

        Returns:
            Dict[int, List[dict]]: Mapping from ``policy_id`` to a list of
                segment dictionaries.  Each segment has keys: states (one-hot
                stripped), actions, rewards, values, log_probs, advantages,
                returns.
        """

        T = len(episode["rewards"])

        # --- Extract scenario ID per timestep from the stored states ---
        scenario_ids = np.array([
            int(np.argmax(episode["states"][t, -self.num_policies:]))
            for t in range(T)
        ])

        # --- Identify contiguous segments ---
        # Each segment is (start_idx, end_idx_exclusive, policy_id)
        segments: List[Tuple[int, int, int]] = []
        seg_start = 0
        for t in range(1, T):
            if scenario_ids[t] != scenario_ids[seg_start]:
                segments.append((seg_start, t, int(scenario_ids[seg_start])))
                seg_start = t
        segments.append((seg_start, T, int(scenario_ids[seg_start])))

        # --- Compute per-segment GAE ---
        policy_segments: Dict[int, List[dict]] = {
            k: [] for k in range(self.num_policies)
        }

        for i, (start, end, policy_id) in enumerate(segments):
            # Slice segment data (strip one-hot from states)
            seg_states = episode["states"][start:end, :-self.num_policies]
            seg_actions = episode["actions"][start:end]
            seg_rewards = episode["rewards"][start:end]
            seg_values = episode["values"][start:end]
            seg_log_probs = episode["log_probs"][start:end]
            seg_terminations = episode["terminations"][start:end]

            # ----------------------------------------------------------
            # Determine bootstrap value at end of segment
            # ----------------------------------------------------------
            is_last_segment = (i == len(segments) - 1)

            if is_last_segment:
                # Last segment: bootstrap depends on true episode termination
                last_terminated = bool(episode["terminations"][end - 1])
                if last_terminated:
                    next_value = 0.0
                else:
                    # Truncation or safeguard — bootstrap with this policy's
                    # value estimate of the state after the segment
                    next_state = episode["next_states"][end - 1]
                    base_next = next_state[:-self.num_policies]
                    with torch.no_grad():
                        st = torch.FloatTensor(base_next).unsqueeze(0).to(self.device)
                        _, _, v = self.policies[policy_id].network(st)
                        next_value = v.squeeze(0).item()
            else:
                # Mid-episode scenario switch — bootstrap with the outgoing
                # policy's value estimate of the next state (first state of
                # the next segment).  This treats the switch as a soft
                # episode boundary.
                boundary_state = episode["states"][end]
                base_boundary = boundary_state[:-self.num_policies]
                with torch.no_grad():
                    st = torch.FloatTensor(base_boundary).unsqueeze(0).to(self.device)
                    _, _, v = self.policies[policy_id].network(st)
                    next_value = v.squeeze(0).item()

            # Compute GAE for this segment
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
        Run PPO optimization for a single policy on its collected segments.

        Mirrors ``SinglePPOAgent.train()`` but operates on the given
        policy/optimizer pair.

        Args:
            policy_id (int): Index of the policy to train.
            segments (list[dict]): Segment dicts with keys: states, actions,
                advantages, log_probs, returns.

        Returns:
            Tuple[float, float, float, float]: Mean total loss, mean policy
                loss, mean value loss, and mean entropy bonus (entropy coef *
                mean entropy) across all minibatches/epochs.  All NaN if no
                usable data.
        """
        if policy_id in self.frozen_pids:
            return float("nan"), float("nan"), float("nan"), float("nan")

        if not segments:
            return float("nan"), float("nan"), float("nan"), float("nan")

        policy = self.policies[policy_id]
        optimizer = self.optimizers[policy_id]
        policy.network.train()

        # Concatenate segments into batched tensors
        all_states = torch.as_tensor(
            np.concatenate([s["states"] for s in segments]),
            dtype=torch.float32,
        ).to(self.device)

        all_actions = torch.as_tensor(
            np.concatenate([s["actions"] for s in segments]),
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

        # Normalize advantages
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
                mb_actions = all_actions[mb_idx]
                mb_old_log_probs = all_old_log_probs[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_advantages = all_advantages[mb_idx]

                # Forward pass through this policy's network
                action_mean, action_std, state_values = policy.network(mb_states)
                dist = torch.distributions.Normal(action_mean, action_std)

                # Inverse-tanh for log-prob correction
                if isinstance(self.action_space, gym.spaces.Box):
                    raw_actions = torch.atanh(mb_actions.clamp(-0.999, 0.999))
                else:
                    raw_actions = mb_actions
                log_probs = policy.calc_log_probs(dist, raw_actions, mb_actions)

                # PPO clipped objective
                ratios = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value + entropy losses
                value_loss = nn.functional.mse_loss(state_values.squeeze(-1), mb_returns)
                entropy_loss = dist.entropy().sum(dim=-1).mean()

                entropy_bonus = self.entropy_coef * entropy_loss
                loss = policy_loss + (self.value_loss_coef * value_loss) - entropy_bonus

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.network.parameters(), self.max_grad_norm)
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
        Full training loop: collect trajectories, split by scenario,
        compute per-segment GAE, and train each policy.

        Args:
            stop_condition (Optional[StopCondition]): Stopping criterion.
        """

        print(
            f"Starting multi-policy training with {self.num_policies} policies, "
            f"{self.num_envs} parallel env(s) on '{self.device}'"
        )

        last_save_index = 0
        stopping_condition = stop_condition or StopCondition()

        # Per-policy tracking
        best_rewards: Dict[int, float] = {k: float("-inf") for k in range(self.num_policies)}
        training_metrics: Dict[str, list] = {
            label: [] for label in self.policy_labels
        }

        best_policy_files = [
            f"Best_{self.env.type_name}_{label}_policy"
            for label in self.policy_labels
        ]

        for iteration in range(stopping_condition.max_time_steps):
            self.set_policy_eval()

            # --- 1. Collect trajectories (base class handles routing via _predict) ---
            batch_data = self.run_parallel_episodes(self.num_envs)

            # --- 2. Split every trajectory & recompute per-segment GAE ---
            accumulated_segments: Dict[int, List[dict]] = {
                k: [] for k in range(self.num_policies)
            }

            for episode in batch_data:
                policy_segments = self._split_trajectory_and_compute_gae(episode)
                for pid, segs in policy_segments.items():
                    accumulated_segments[pid].extend(segs)

            # --- 3. Train each policy on its segments ---
            iter_rewards = {}
            iter_losses = {}
            iter_policy_losses = {}
            iter_value_losses = {}
            iter_entropy_bonuses = {}

            for pid in range(self.num_policies):
                segs = accumulated_segments[pid]

                # Push this iteration's segments into the per-slot pending
                # buffer; the update only fires once the buffer crosses
                # ``steps_per_policy_update``.
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

                # Reward reflects only this iteration's freshly collected
                # segments (not the pending buffer), so the per-iteration
                # reward curve isn't smeared by stale data.
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

                # Save best policy per scenario (only on iterations where
                # an update actually fired, so we don't snapshot a stale
                # policy against a noisy single-iteration reward).
                if (
                    flushed_steps > 0
                    and iteration >= self.MINIMUM_ITERATIONS_FOR_BEST_SAVE
                    and not np.isnan(iter_rewards[pid])
                    and iter_rewards[pid] > best_rewards[pid]
                ):
                    best_rewards[pid] = iter_rewards[pid]
                    self._save_single_policy(pid, best_policy_files[pid])

                # Collapse watchdog: track rolling-mean reward, snapshot
                # the network at new peaks, and roll back on collapse.
                # Skipped for frozen slots (which don't update anyway).
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
                        # Reset this policy's Adam optimizer state. The
                        # exponential moving averages of past gradients
                        # (Adam's ``m`` and ``v`` buffers) were accumulated
                        # during the collapse phase and, if reused after
                        # rolling weights back, would push the network
                        # right along the failure trajectory — producing
                        # a rapid second collapse and a rollback ping-pong.
                        # A fresh optimizer ensures the next gradient steps
                        # are computed cleanly from the rolled-back weights.
                        self.optimizers[pid] = optim.Adam(
                            self.policies[pid].network.parameters(),
                            lr=self.learning_rate,
                        )
                        # Discard pending segments — they were collected
                        # under the now-discarded post-collapse weights and
                        # would be off-policy w.r.t. the rolled-back net.
                        self._pending_segments[pid] = []
                        self._pending_steps[pid] = 0
                        self.watchdog.record_rollback(pid, iteration)
                        rollback_triggered = True
                        print(
                            f"[watchdog] iter {iteration}: collapse detected for "
                            f"'{self.policy_labels[pid]}' "
                            f"(peak smoothed={self.watchdog._best_smoothed[pid]:.3f}, "
                            f"rollback {self.watchdog._rollback_count[pid]}/"
                            f"{self.watchdog.max_rollbacks}). "
                            f"Rolled back to best rolling-mean snapshot."
                        )

                # Annotate the most recent metrics record with watchdog state
                if training_metrics[self.policy_labels[pid]]:
                    training_metrics[self.policy_labels[pid]][-1][
                        "rollback_triggered"
                    ] = rollback_triggered

            # --- 4. Logging ---
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
                print(f"[Iter {iteration}] " + " | ".join(parts))

            # Periodic checkpoint (all policies in one ZIP)
            if iteration > 0 and iteration % self.eval_frequency == 0:
                self.save_policy(f"multi_policy_checkpoint_{iteration}")

            # --- 5. Stopping condition ---
            should_stop, reason = stopping_condition.should_stop(
                current_step=iteration,
                mean_reward=agg_reward,
                mean_loss=agg_loss,
            )

            if should_stop:
                print(f"Stopping: {reason}")
                last_save_index = iteration
                break

        # Final checkpoint
        self.save_policy(
            f"multi_policy_checkpoint_"
            f"{last_save_index if last_save_index > 0 else stopping_condition.max_time_steps}"
        )

        # Dump per-policy training metrics
        self._dump_training_metrics(training_metrics)

        if self._vec_env is not None:
            self._vec_env.close()
            self._vec_env = None

        print("Multi-policy training completed!")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_policy(self, policy_file_name: str) -> None:
        """
        Save all N policies into a single ZIP archive.

        Archive layout::

            policy_0.pth   (serialized PPOPolicy)
            policy_1.pth
            ...
            policy_{N-1}.pth

        Args:
            policy_file_name (str): File name (without extension).
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
        Load N policies from a ZIP archive.

        Args:
            policy_file_name (str): Path to the ZIP (extension added if missing).

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

                # Rebuild optimizers for the loaded networks
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
        Load a single pre-trained policy (e.g. from a ``SinglePPOAgent``
        checkpoint) into one of the N policy slots.

        This is useful for initializing the multi-policy agent with
        policies that were pre-trained in specialized environments.

        Args:
            policy_id (int): Which slot to load into (0..N-1).
            policy_file_name (str): Path to a ``SinglePPOAgent``-style
                ``.zip`` policy file.
            frozen (bool): If True, the slot is excluded from PPO updates
                (kept fixed during training).

        Raises:
            IndexError: If ``policy_id`` is out of range.
            RuntimeError: If loading fails.
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
        Load multiple pre-trained policies from a single directory.

        Args:
            directory (str): Directory containing the pre-trained ``.zip``
                policy files.
            specs (dict[int, dict]): Mapping ``policy_id -> {"file": str,
                "frozen": bool}``.  ``file`` is the file name (without
                ``.zip``) inside ``directory``.  ``frozen`` defaults to
                False.

        Slots whose file is missing are left untouched and a warning is
        printed.
        """

        for pid, spec in specs.items():
            file_stem = spec["file"]
            frozen = bool(spec.get("frozen", False))
            full_path = os.path.join(directory, file_stem)
            zip_path = full_path if full_path.endswith(".zip") else full_path + ".zip"

            if not os.path.exists(zip_path):
                print(f"[pretrained] No file at {zip_path}, slot {pid} left unchanged.")
                continue

            self.load_individual_policy(pid, full_path, frozen=frozen)
            label = self.policy_labels[pid]
            state = "frozen" if frozen else "trainable"
            print(f"[pretrained] Loaded '{file_stem}' into slot {pid} ({label}) [{state}]")

    def _save_single_policy(self, policy_id: int, policy_file_name: str) -> None:
        """
        Save one policy slot as a standalone file (same format as
        ``SinglePPOAgent``).

        Args:
            policy_id (int): Which policy to save.
            policy_file_name (str): File name (without extension).
        """

        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        self.policies[policy_id].save(policy_file_name)
        #print(f"[save] Saved {self.policy_labels[policy_id]} policy to {policy_file_name}.zip")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def set_policy_eval(self):
        """Set all policy networks to evaluation mode."""
        for policy in self.policies:
            policy.network.eval()

    def _dump_training_metrics(self, metrics: Dict[str, list]) -> None:
        """Flatten per-policy metrics and write to CSV."""

        flat = []
        for label, records in metrics.items():
            for record in records:
                flat.append({"policy": label, **record})

        if flat:
            dump_metrics_to_csv(self.model_dir, "multi_policy_training_metrics.csv", flat)
