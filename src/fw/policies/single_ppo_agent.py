import os
import copy
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

from pathlib import Path
from typing import Optional, Tuple, Union
from fw.policies.ppo_policy import PPOPolicy
from fw.policies.base_ppo_agent import BasePPOAgent
from stable_baselines3.common.monitor import Monitor
from fw.configuration.stop_condition import StopCondition
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from fw.policies.ppo_policy import (apply_policy_yaw_rate_obs_scale, require_policy_dims,
                                    start_fresh_policy, yaw_rate_obs_scale_changes)
from fw.common.airsim.tools.airsim_training_session import AirSimTrainingSession
from fw.common.tools.training_iteration_logger import TrainingIterationLogger
# from fw.common.tools.ppo_utils import normalize_policy_zip_path
from fw.common.tools.logging_utils import dump_metrics_to_csv
from fw.configuration.profile_applier import SimulatorApplier
from fw.configuration.config_profile import VesselProfile
from fw.common.tools.ppo_utils import reseed_env_rng

logger = logging.getLogger(__name__)


class SinglePPOAgent(BasePPOAgent):
    """
    PPO agent for a single environment, run as vectorized workers.

    Handles inference, PPO updates, the training loop and checkpointing; rollout
    and GAE logic come from `BasePPOAgent`. Only continuous (Box) action spaces
    are supported.
    """

    MINIMUM_ITERATIONS_FOR_BEST_SAVE = 20
    MINIMUM_TRAINABLE_TRAJECTORY_LENGTH = 2
    MAX_CONSECUTIVE_UNTRAINABLE_AIRSIM_BATCHES = 25

    def __init__(self, profile: Optional[VesselProfile] = None, *args, **kwargs):
        """
        Initialize the PPO agent.

        Args:
            profile: Profile used for training or visualization. When None,
                profile-dependent initialization and action adaptation are skipped.
            *args: Positional arguments forwarded to `BasePPOAgent`.
            **kwargs: Keyword arguments forwarded to `BasePPOAgent`.

        Input and output dimensions are inferred from the environment.

        Raises:
            NotImplementedError: If the action space is not a Box.
        """

        super().__init__(profile, *args, **kwargs)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # The policy is a tanh-squashed Gaussian, so only Box action spaces work.
        if not isinstance(self.action_space, gym.spaces.Box):
            raise NotImplementedError(
                f"SinglePPOAgent supports only Box action spaces, got {self.action_space}."
            )
        self.output_dim = self.action_space.shape[0]

        self.input_dim = self.observation_space.shape[0]
        self.policy = PPOPolicy(self.input_dim, self.output_dim, str(self.device))
        start_fresh_policy(self.policy, self.env)
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=self.learning_rate)
        self._vec_env = None

    def _is_airsim_env(self) -> bool:
        """
        Return whether the wrapped environment is the AirSim trainer env.
        """

        return getattr(self.env, "type_name", "") == "AirSimEnv"

    def predict(
            self,
            state: Union[np.ndarray, Tuple[np.ndarray, ...]],
            deterministic: bool = False
    ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], np.ndarray, np.ndarray]:
        """
        Predict an action for a single state.

        Args:
            state (Union[np.ndarray, Tuple[np.ndarray, ...]]): Input observation.
            deterministic (bool): Whether to use deterministic policy output.

        Returns:
            Tuple (action, log-probability, state value).
        """

        actions, log_probs, values = self.policy.predict(state, deterministic)
        return actions[0], log_probs.item(), values.item()


    def train(self, all_episodes_data) -> float:
        """
        Perform PPO optimization on collected trajectories: normalized advantages,
        mini-batch updates over num_epochs with clipping, value loss and entropy
        bonus. Steps with a non-finite loss or gradient norm are skipped.

        Args:
            all_episodes_data (list[dict]): Trajectories with states, raw_actions
                (pre-tanh), advantages, log_probs and returns (np.ndarray each).

        Returns:
            float: Mean loss over all updates, or NaN if there is no valid data.

        Raises:
            KeyError: If an episode has no "raw_actions" (not produced by
                run_parallel_episodes).
        """

        if not all_episodes_data:
            return float('nan')

        if any("raw_actions" not in ep for ep in all_episodes_data):
            raise KeyError(
                "PPO update needs 'raw_actions' (pre-tanh rollout actions) in every "
                "episode; collect episodes with run_parallel_episodes()."
            )

        self.policy.network.train()

        all_states = torch.as_tensor(np.concatenate([ep['states'] for ep in all_episodes_data]),
                                     dtype=torch.float32).to(self.device)
        all_raw_actions = torch.as_tensor(np.concatenate([ep['raw_actions'] for ep in all_episodes_data]),
                                          dtype=torch.float32).to(self.device)
        all_advantages = torch.as_tensor(np.concatenate([ep['advantages'] for ep in all_episodes_data]),
                                         dtype=torch.float32).to(self.device)
        all_old_log_probs = torch.as_tensor(np.concatenate([ep['log_probs'] for ep in all_episodes_data]),
                                            dtype=torch.float32).to(self.device)
        all_returns = torch.as_tensor(np.concatenate([ep['returns'] for ep in all_episodes_data]),
                                      dtype=torch.float32).to(self.device)

        if all_advantages.numel() <= 1:
            logger.warning("[Warning] Very short trajectory detected. Length: %d", all_advantages.numel())
            return float('nan')

        for name, tensor in (
            ("states", all_states),
            ("advantages", all_advantages),
            ("returns", all_returns),
        ):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.warning("[Alert] NaN/Inf detected in %s before PPO update - skipping batch.", name)
                return float('nan')

        # Normalizing a near-constant tensor produces NaN/inf gradients.
        if all_advantages.numel() > 1 and all_advantages.std() > 1e-6:
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        batch_size = all_states.shape[0]
        loss_per_epoch = []

        for _ in range(self.num_epochs):
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_states = all_states[mb_idx]
                mb_raw_actions = all_raw_actions[mb_idx]
                mb_old_log_probs = all_old_log_probs[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_advantages = all_advantages[mb_idx]

                action_mean, action_std, state_values = self.policy.network(mb_states)
                dist = torch.distributions.Normal(action_mean, action_std)

                # Same evaluation point as in the rollout (PPOPolicy.raw_log_prob).
                log_probs = self.policy.raw_log_prob(dist, mb_raw_actions)

                log_ratios = torch.clamp(log_probs - mb_old_log_probs, max=PPOPolicy.MAX_LOG_RATIO)
                ratios = torch.exp(log_ratios)

                surrogate_1 = ratios * mb_advantages
                surrogate_2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages

                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                value_loss = torch.nn.functional.mse_loss(state_values.squeeze(-1), mb_returns)
                entropy_loss = dist.entropy().sum(dim=-1).mean()    # Encourage exploration

                loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy_loss)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("[Alert] NaN/Inf loss detected (policy=%.4f, "
                                   "value=%.4f) - skipping update step.", policy_loss.item(), value_loss.item())
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.network.parameters(), self.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    # A non-finite norm would turn every weight into NaN.
                    logger.warning("[Alert] Non-finite gradient norm (%s) - skipping update step.", float(grad_norm))
                    self.optimizer.zero_grad()
                    continue
                if float(grad_norm) > 1000.0:
                    logger.warning("[Alert] Exploding gradient: norm=%.2f "
                                   "(clip=%.2f). Clipping applied.", float(grad_norm), self.max_grad_norm)
                self.optimizer.step()

                loss_per_epoch.append(loss.item())

        return float(np.mean(loss_per_epoch)) if loss_per_epoch else float('nan')

    @classmethod
    def filter_trainable_episodes(cls, batch_data):
        """
        Split off trajectories too short for a PPO update (fewer than
        MINIMUM_TRAINABLE_TRAJECTORY_LENGTH transitions), which give NaN losses.

        Returns:
            Tuple (trainable episodes, number skipped).
        """

        valid_episodes = []
        skipped_count = 0

        for episode in batch_data:
            rewards = np.asarray(episode.get("rewards", ()), dtype=np.float32)
            advantages = np.asarray(episode.get("advantages", ()), dtype=np.float32)
            trajectory_length = int(min(rewards.size, advantages.size))
            if trajectory_length < cls.MINIMUM_TRAINABLE_TRAJECTORY_LENGTH:
                skipped_count += 1
                continue
            valid_episodes.append(episode)

        return valid_episodes, skipped_count


    def make_env(self, worker_seed=None):
        """
        Create a callable environment factory for vectorized execution.

        Args:
            worker_seed: Seed (np.random.SeedSequence or int) for this worker's
                env RNG. None spawns a new child seed (see BaseAgent 'seed').

        Returns:
            Callable[[], gym.Env]: Creates a Monitor-wrapped deep copy of the
            environment, with the profile pose applied and its own RNG seed.
        """

        profile = self._profile  # bound now, not at call time
        if worker_seed is None:
            worker_seed = self._spawn_worker_seeds(1)[0]

        def _init():
            """
            Create one monitored deep copy of the environment.
            """

            env_copy = copy.deepcopy(self.env)

            # Apply the profile pose to this worker's simulator (not only to self.env).
            if profile is not None:
                _sim = getattr(env_copy, "phys_sim", None) or getattr(env_copy, "_phys_sim", None)
                if _sim is not None:
                    SimulatorApplier.apply(profile, _sim)

            # Deep copies share the RNG state; reseed so workers sample
            # different start poses / routes (reproducibly).
            reseed_env_rng(env_copy, worker_seed)

            return Monitor(env_copy)

        return _init


    def _reset_envs(self, num_envs):
        """
        Initialize or reset vectorized environments.

        Args:
            num_envs (int): Number of parallel environments.

        Returns:
            np.ndarray: Initial observations of shape (num_envs, obs_dim).

        The VecEnv (SubprocVecEnv for several envs, else DummyVecEnv) is reused
        across rollouts, as in SB3's on-policy algorithm, and only recreated
        when num_envs changes.
        """

        if self._vec_env is None or getattr(self._vec_env, "num_envs", None) != num_envs:
            env_fns = [self.make_env(worker_seed) for worker_seed in self._spawn_worker_seeds(num_envs)]
            self._vec_env = SubprocVecEnv(env_fns) if num_envs > 1 else DummyVecEnv(env_fns)

        # Optional environment randomization
        if hasattr(self._vec_env, "env_method"):
            self._vec_env.env_method("randomize")
            if self._is_airsim_env():
                self._vec_env.env_method("mark_explicit_reset_context", True)

        return self._vec_env.reset()


    def _step_envs(self, actions):
        """
        Step all environments with the given actions.

        Args:
            actions (np.ndarray): Actions of shape (num_envs, action_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - next_states: Next observations (auto-reset ones for finished envs).
                - rewards: Rewards per environment.
                - terminations: Episode termination flags.
                - truncations: Episode truncation flags.
                - final_states: Real last observations (used to bootstrap truncations).

        Rewards and flags come from the per-env info dicts when all provide
        them; otherwise from the SB3 outputs, with truncation recovered from
        "TimeLimit.truncated" (False if absent).
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

            # SB3 'done' = terminated OR truncated. Split off truncation, which
            # must be bootstrapped rather than treated as terminal.
            if infos:
                truncations = np.array(
                    [bool(env_info.get("TimeLimit.truncated", False)) for env_info in infos],
                    dtype=bool,
                )
            else:
                truncations = np.zeros(actions.shape[0], dtype=bool)
            terminations = np.asarray(next_terminations, dtype=bool) & ~truncations

        # SB3 auto-resets finished envs, so next_states[i] is then the next
        # episode's first observation; the real last one is in
        # info["terminal_observation"].
        final_states = np.array(next_states, copy=True)
        if infos:
            dones = np.asarray(next_terminations, dtype=bool)
            for env_id, env_info in enumerate(infos):
                if dones[env_id] and "terminal_observation" in env_info:
                    final_states[env_id] = env_info["terminal_observation"]

        return next_states, rewards, terminations, truncations, final_states


    def _predict(self, states, deterministic=False):
        """
        Compute batched actions, raw actions, log-probabilities, and value estimates.

        Args:
            states (np.ndarray): Observations of shape (num_envs, obs_dim).
            deterministic (bool): Whether to use deterministic policy output.

        Returns:
            Tuple (actions, raw_actions, log_probs, values).
        """

        return self.policy.predict_with_raw(states, deterministic=deterministic)


    def _get_bootstrap_value(self, next_state, env_id: Optional[int] = None):
        """
        Estimate value for a given state for bootstrapping.

        Args:
            next_state (np.ndarray): State to evaluate.
            env_id (Optional[int]): Environment index (unused in this implementation).

        Returns:
            float: Estimated state value.
        """

        with torch.no_grad():
            last_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, _, next_values = self.policy.network(last_state)
            return next_values.squeeze(0).item()


    def _drain_env_run_events(self) -> list[dict]:
        """
        Collect queued AirSim run events from vectorized envs.
        """

        if getattr(self, "_vec_env", None) is not None and hasattr(self._vec_env, "env_method"):
            drained = self._vec_env.env_method("drain_run_events")
            events = []
            for env_slot, env_events in enumerate(drained or []):
                for event in env_events or []:
                    normalized = dict(event)
                    normalized.setdefault("env_slot", int(env_slot))
                    events.append(normalized)
            return events

        if hasattr(self.env, "drain_run_events"):
            return list(self.env.drain_run_events())

        return []


    def _get_training_export_snapshot(self):
        """
        Return the first available AirSim export snapshot from the vec env.
        """

        if getattr(self, "_vec_env", None) is not None and hasattr(self._vec_env, "env_method"):
            snapshots = self._vec_env.env_method("get_training_export_snapshot")
            for snapshot in snapshots or []:
                if snapshot:
                    return snapshot

        if hasattr(self.env, "get_training_export_snapshot"):
            return self.env.get_training_export_snapshot()

        return None


    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
    ):
        """
        Execute the full PPO training loop.

        Args:
            stop_condition (Optional[StopCondition]): Stopping criteria; a
                default StopCondition if None.

        Per iteration: collect rollouts, run a PPO update, log metrics, save
        the best policy and periodic checkpoints, and check the stop condition.
        For AirSim, too-short rollouts are skipped and training aborts after
        MAX_CONSECUTIVE_UNTRAINABLE_AIRSIM_BATCHES empty batches. On Ctrl-C an
        interrupt checkpoint is saved and KeyboardInterrupt is re-raised after
        cleanup. The VecEnv is always closed.

        Raises:
            RuntimeError: On repeated untrainable AirSim batches, or when the
                final checkpoint/metrics cannot be saved.
        """

        logger.info("Starting training with %d parallel environments on a '%s' device", self.num_envs, self.device)

        project_src_dir = Path(__file__).resolve().parents[2]
        is_airsim_env = self._is_airsim_env()

        training_session = (
            AirSimTrainingSession(
                env_name=self.env.training_name,
                agent_type="PPO",
                device=str(self.device),
                src_dir=project_src_dir,
                model_dir=self.model_dir,
            )
            if is_airsim_env
            else None
        )
        iteration_logger = (
            training_session.iteration_logger
            if training_session is not None
            else TrainingIterationLogger(
                env_name=self.env.training_name,
                agent_type="PPO",
                device=str(self.device),
                src_dir=project_src_dir,
            )
        )
        logger.info("Iteration log: %s", iteration_logger.path)
        if training_session is not None:
            logger.info("Run log: %s", training_session.run_log_path)
            logger.info("Metadata: %s", training_session.metadata_path)

        last_save_index = 0
        training_metrics = []
        best_mean_reward = float('-inf')
        best_policy_file = f"Best_{self.env.training_name}_PPO_policy"
        stopping_condition = stop_condition if stop_condition is not None else StopCondition()
        learn_started = time.perf_counter()

        iteration = 0
        last_completed_iteration = -1
        skipped_short_rollouts = 0
        consecutive_untrainable_airsim_batches = 0
        interrupted = False
        stop_reason = ""
        cached_route_snapshot = None
        keyboard_interrupt = None
        completed_successfully = False

        try:
            while iteration < stopping_condition.max_time_steps:
                self.set_policy_eval()

                episode_data = []
                total_reward = 0
                total_advantage = 0

                batch_data = self.run_parallel_episodes(self.num_envs)

                if training_session is not None:
                    training_session.log_run_events(self._drain_env_run_events())
                    if cached_route_snapshot is None:
                        cached_route_snapshot = self._get_training_export_snapshot()
                        training_session.set_route_snapshot(cached_route_snapshot)

                if is_airsim_env:
                    batch_data, skipped_count = self.filter_trainable_episodes(batch_data)
                else:
                    skipped_count = 0

                if skipped_count > 0:
                    skipped_short_rollouts += skipped_count
                    logger.warning(
                        "[Warning] Skipping %d too-short trajectory batch(es) before PPO update.",
                        skipped_count,
                    )

                if not batch_data:
                    if is_airsim_env:
                        consecutive_untrainable_airsim_batches += 1
                        if (
                            consecutive_untrainable_airsim_batches == 1
                            or consecutive_untrainable_airsim_batches % 5 == 0
                        ):
                            logger.warning(
                                "[Warning] AirSim produced no trainable rollout batch. "
                                "Consecutive empty batches: %d/%d.",
                                consecutive_untrainable_airsim_batches,
                                self.MAX_CONSECUTIVE_UNTRAINABLE_AIRSIM_BATCHES,
                            )
                        if (
                            consecutive_untrainable_airsim_batches
                            >= self.MAX_CONSECUTIVE_UNTRAINABLE_AIRSIM_BATCHES
                        ):
                            raise RuntimeError(
                                "AirSim training aborted after repeated untrainable rollout "
                                "batches. Check reset stability, startup pose capture, or "
                                "episode termination conditions before retrying."
                            )
                    continue
                consecutive_untrainable_airsim_batches = 0

                for episode_result in batch_data:
                    episode_data.append(episode_result)
                    total_reward += np.mean(episode_result['rewards'])
                    total_advantage += np.mean(episode_result['advantages'])

                loss = self.train(episode_data)

                mean_reward = total_reward / len(batch_data) if batch_data else 0
                mean_advantage = total_advantage / len(batch_data) if batch_data else 0

                training_metrics.append({
                    'loss': loss,
                    'advantage': mean_advantage,
                    'reward': mean_reward,
                })
                iteration_logger.log_iteration(
                    iteration=iteration,
                    mean_reward=mean_reward,
                    mean_advantage=mean_advantage,
                    loss=loss,
                    num_episode_batches=len(batch_data),
                    elapsed_seconds=time.perf_counter() - learn_started,
                )

                if iteration >= self.MINIMUM_ITERATIONS_FOR_BEST_SAVE and mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    self.save_policy(best_policy_file)

                should_stop, reason = stopping_condition.should_stop(
                    current_step=iteration,
                    mean_reward=mean_reward,
                    mean_loss=loss
                )

                if should_stop:
                    stop_reason = str(reason)
                    logger.info("Stopping: %s", reason)
                    last_save_index = iteration
                    last_completed_iteration = iteration
                    completed_successfully = True
                    break

                if iteration % self.eval_frequency == 0:
                    logger.info("Iteration %d, Average Reward: %.3f, Loss: %.3f", iteration, mean_reward, loss)

                    if iteration > 0:
                        self.save_policy(self.output_file_name(f"ppo_checkpoint_{iteration}"))

                last_completed_iteration = iteration
                iteration += 1

            if not interrupted:
                completed_successfully = True
        except KeyboardInterrupt as exc:
            interrupted = True
            stop_reason = "Interrupted by user"
            keyboard_interrupt = exc
            logger.warning("Training interrupted by user. Saving logs and plots...")

            try:
                _interrupt_checkpoint = self.output_file_name(f"ppo_interrupt_{max(last_completed_iteration, 0)}")
                self.save_policy(_interrupt_checkpoint)
                logger.info("Interrupt checkpoint saved: %s", _interrupt_checkpoint)
            except Exception as _save_err:
                logger.warning("Could not save interrupt checkpoint: %s", _save_err)
        finally:
            # Each cleanup step is guarded so a failure can neither mask the
            # original exception nor prevent closing the VecEnv.
            final_save_errors = []

            def _safe(step_name, fn, *fn_args, critical=False, **fn_kwargs):
                """
                Run one cleanup step and log a failure. Failures of `critical`
                steps are re-raised after the VecEnv is closed.
                """

                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as step_exc:
                    logger.exception("Training cleanup step '%s' failed; continuing.", step_name)
                    if critical:
                        final_save_errors.append(step_exc)
                    return None

            try:
                if training_session is not None:
                    run_events = _safe("drain env run events", self._drain_env_run_events)
                    _safe("log run events", training_session.log_run_events, run_events or [])
                    latest_snapshot = _safe("route snapshot", self._get_training_export_snapshot)
                    if latest_snapshot is not None:
                        cached_route_snapshot = latest_snapshot
                    _safe("set route snapshot", training_session.set_route_snapshot, cached_route_snapshot)

                if completed_successfully:
                    final_checkpoint_iteration = (
                        last_save_index
                        if last_save_index > 0
                        else last_completed_iteration
                    )
                    if final_checkpoint_iteration < 0:
                        final_checkpoint_iteration = 0
                    _safe("final checkpoint", self.save_policy, self.output_file_name(f'ppo_checkpoint_{final_checkpoint_iteration}'), critical=True)
                    _safe("iteration log", iteration_logger.save, critical=True)
                    _safe("training metrics CSV", dump_metrics_to_csv, self.output_dir, "training_metrics.csv", training_metrics, critical=True)

                if training_session is not None:
                    _safe(
                        "finalize training session",
                        training_session.finalize,
                        training_metrics=training_metrics,
                        interrupted=interrupted,
                        skipped_short_rollouts=skipped_short_rollouts,
                        last_completed_iteration=last_completed_iteration,
                        best_mean_reward=best_mean_reward,
                        stop_rule=str(stopping_condition) if not stop_reason else stop_reason,
                    )
            finally:
                if getattr(self, "_vec_env", None) is not None:
                    _safe("close VecEnv", self._vec_env.close)
                    self._vec_env = None

            # Critical steps only run after a successful run.
            if final_save_errors:
                raise final_save_errors[0]

        if skipped_short_rollouts > 0:
            logger.info(
                "[Training summary] Skipped %d too-short rollout batch(es).",
                skipped_short_rollouts,
            )

        if training_session is not None:
            logger.info("Route plot: %s", training_session.route_plot_path)
            logger.info("Training metrics plot: %s", training_session.training_metrics_plot_path)

        if interrupted:
            logger.warning("Training interrupted; logs saved.")
            if keyboard_interrupt is not None:
                raise keyboard_interrupt
        else:
            logger.info("Training completed!")


    def load_policy(self, policy_file_name: str) -> None:
        """
        Load a saved policy from disk.

        Args:
            policy_file_name (str): Name or path of the policy file.

        The env takes over the yaw-rate observation scale the policy was trained
        with (apply_policy_yaw_rate_obs_scale).

        Raises:
            RuntimeError: If loading fails.
            ValueError: If the policy's observation/action size differs from the env's.
        """

        try:
            if self.model_dir:
                policy_file_name = os.path.join(self.model_dir, policy_file_name)

            policy = PPOPolicy.load(policy_file_name, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the policy '{policy_file_name}'.") from e

        require_policy_dims(policy, self.input_dim, self.output_dim, policy_file_name)

        if self._vec_env is not None and yaw_rate_obs_scale_changes(policy, self.env):
            # Worker copies were made with the old scale: rebuild them on the next rollout.
            try:
                self._vec_env.close()
            except Exception as e:
                raise RuntimeError(f"Failed to close the worker copies before loading '{policy_file_name}'.") from e
            self._vec_env = None
        apply_policy_yaw_rate_obs_scale(policy, self.env, policy_file_name)
        self.policy = policy
        # load() returns a new network: rebind the optimizer to its parameters.
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=self.learning_rate)

        # Disabled: restore Adam momentum from a companion optimizer file.
        # opt_path = normalize_policy_zip_path(policy_file_name).replace(".zip", "_optimizer.pth")
        # if os.path.isfile(opt_path):
        #     try:
        #         self.optimizer.load_state_dict(
        #             torch.load(opt_path, map_location=self.device, weights_only=True)
        #         )
        #     except Exception as opt_exc:  # shape mismatch on arch change, not fatal
        #         import logging as _logging
        #         _logging.getLogger(__name__).warning(
        #             "Could not restore optimizer state from '%s': %s", opt_path, opt_exc
        #         )


    def save_policy(self, policy_file_name: str) -> None:
        """
        Save the current policy to disk.

        Args:
            policy_file_name (str): Name or path for the saved policy.

        Raises:
            RuntimeError: If saving fails.
        """

        try:
            if self.model_dir:
                policy_file_name = os.path.join(self.model_dir, policy_file_name)

            self.policy.save(policy_file_name)

            # Disabled: save the optimizer state in a file next to the zip.
            # if self.optimizer is not None:
            #     opt_path = normalize_policy_zip_path(policy_file_name).replace(".zip", "_optimizer.pth")
            #     torch.save(self.optimizer.state_dict(), opt_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save the policy '{policy_file_name}'.") from e


    def set_policy_eval(self):
        """
        Set the policy network to evaluation mode.
        """

        self.policy.network.eval()
