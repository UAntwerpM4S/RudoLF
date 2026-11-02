import numpy as np
import logging

from typing import Optional
from abc import abstractmethod
from fw.policies.base_agent import BaseAgent
from fw.environments.base_env import BaseEnv
from fw.configuration.profile_applier import ActionAdapter
from fw.configuration.config_profile import VesselProfile

logger = logging.getLogger(__name__)


class BasePPOAgent(BaseAgent):
    """
    Abstract base class for PPO agents: trajectory collection, reward
    normalization and GAE.

    Subclasses implement environment interaction (`_reset_envs`, `_step_envs`)
    and model inference (`_predict`, `_get_bootstrap_value`).
    """

    @staticmethod
    def compute_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
        gamma: float,
        lambda_: float,
    ) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards (np.ndarray): Array of rewards for the batch.
            values (np.ndarray): Array of value estimates for each state.
            dones (np.ndarray): Termination flags (zero the bootstrap).
            next_value (float): Value estimate for the state after the batch.
            gamma (float): Discount factor for future rewards.
            lambda_ (float): GAE bias-variance trade-off parameter.

        Returns:
            np.ndarray: Advantages (float32).
        """

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + gamma * lambda_ * next_non_terminal * last_gae
            advantages[t] = last_gae

        return advantages


    @staticmethod
    def find_first_true(bool_array):
        """
        Return the index of the first True value in a boolean array.

        Args:
            bool_array (np.ndarray): A NumPy array of boolean values.

        Returns:
            int: The index of the first True value, or -1 if not found.
        """

        true_indices = np.where(bool_array)[0]
        return true_indices[0] if true_indices.size > 0 else -1


    def __init__(
        self,
        profile: Optional[VesselProfile] = None,
        environment: BaseEnv = None,
        num_envs: int = 1,
        eval_frequency: int = 1000,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        num_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
        model_dir: str = "models",
        max_nbr_iterations: int = 200000,
        normalize: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the PPO agent with environment and training hyperparameters.

        Args:
            profile: Profile that will be used for training or visualization.
            environment (BaseEnv): Training environment instance.
            num_envs (int): Number of parallel environments.
            eval_frequency (int): Training iterations between evaluations.
            learning_rate (float): Optimizer learning rate.
            clip_range (float): PPO clipping range.
            value_loss_coef (float): Weight for value function loss.
            max_grad_norm (float): Maximum gradient norm for clipping.
            gamma (float): Discount factor.
            gae_lambda (float): GAE smoothing parameter.
            entropy_coef (float): Entropy regularization coefficient.
            num_epochs (int): Number of optimization epochs per update.
            batch_size (int): Mini-batch size.
            device (str): Device for computation ("cpu" or "cuda").
            model_dir (str): Directory for saving models.
            max_nbr_iterations (int): Maximum steps per rollout (safeguard
                against endless episodes).
            normalize (bool): Whether to normalize rewards.
            seed (int, optional): Base seed for the worker env RNGs (see BaseAgent).
        """

        super().__init__(
            environment=environment,
            num_envs=num_envs,
            eval_frequency=eval_frequency,
            learning_rate=learning_rate,
            clip_range=clip_range,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            model_dir=model_dir,
            seed=seed,
        )

        self._profile = profile
        self.max_nbr_iterations = max_nbr_iterations
        self.normalize = normalize


    # -------------------------
    # Abstract interface
    # -------------------------
    @abstractmethod
    def _reset_envs(self, num_envs):
        """
        Reset all environments.

        Args:
            num_envs (int): Number of environments to reset.

        Returns:
            np.ndarray: Initial observations of shape (num_envs, obs_dim).
        """

        pass


    @abstractmethod
    def _step_envs(self, actions):
        """
        Perform a step in all environments.

        Args:
            actions (np.ndarray): Actions of shape (num_envs, action_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - next_states: Next observations (auto-reset ones for finished envs).
                - rewards: Rewards per environment.
                - terminations: Episode termination flags.
                - truncations: Episode truncation flags.
                - final_states: Real last observations (used to bootstrap truncations).
        """

        pass


    @abstractmethod
    def _predict(self, states, deterministic=False):
        """
        Compute actions, log-probabilities, and value estimates.

        Args:
            states (np.ndarray): Observations of shape (num_envs, obs_dim).
            deterministic (bool): Whether to use deterministic policy.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - actions: Sampled actions.
                - log_probs: Log-probabilities of actions.
                - values: Value estimates.
        """

        pass


    @abstractmethod
    def _get_bootstrap_value(self, states, env_id: Optional[int] = None):
        """
        Estimate the bootstrap value after truncation or the iteration limit.

        Args:
            states (np.ndarray): State(s) to evaluate.
            env_id (Optional[int]): Environment index (for multi-env setups).

        Returns:
            float: Estimated value of the given state.
        """

        pass


    @staticmethod
    def _repack_actions(original_actions, transformed_actions: list):
        """
        Repack per-env actions into the container type returned by `_predict`
        (ndarray for SinglePPOAgent, tuple for MultiPPOAgent), as `_step_envs`
        expects.

        Args:
            original_actions: The actions container returned by `_predict`,
                used only to determine the target type.
            transformed_actions (list): Per-environment actions after neutral
                masking and profile transformation.

        Returns:
            Actions repacked in the original container type.

        Raises:
            TypeError: If `original_actions` is not an `np.ndarray`, `tuple`,
                or `list`.
        """

        if isinstance(original_actions, np.ndarray):
            return np.stack(transformed_actions)
        if isinstance(original_actions, tuple):
            return tuple(transformed_actions)
        if isinstance(original_actions, list):
            return transformed_actions
        raise TypeError(
            f"Unsupported action container type: {type(original_actions).__name__}. "
            f"Expected np.ndarray, tuple, or list."
        )


    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Normalize rewards with running mean/variance (parallel Welford update).

        Args:
            rewards: Raw rewards array of shape (batch_size,).

        Returns:
            Rewards standardized with the updated running statistics.
        """

        batch_mean = float(np.mean(rewards))
        batch_var = np.var(rewards)
        batch_count = len(rewards)

        delta = batch_mean - self.reward_mean
        tot_count = self.reward_count + batch_count

        new_mean = self.reward_mean + delta * batch_count / tot_count
        m_a = self.reward_var * self.reward_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / tot_count
        new_var = m2 / tot_count

        self.reward_mean = new_mean
        self.reward_var = new_var
        self.reward_count = tot_count

        return (rewards - new_mean) / (np.sqrt(new_var) + 1e-8)


    def run_parallel_episodes(self, num_envs: int) -> list:
        """
        Collect one trajectory per environment in parallel.

        Environments are stepped synchronously until each is terminated or
        truncated, or `max_nbr_iterations` is reached. Finished environments get
        the env's neutral action so the joint simulator is not perturbed.

        Args:
            num_envs (int): Number of environments to simulate.

        Returns:
            list[dict]: One trajectory per environment (empty ones skipped) with
                states, actions, rewards, terminations, truncations, values,
                log_probs, next_states, advantages, returns (arrays) and env_id.
                Trajectories are trimmed to the first terminal/truncated step;
                non-terminated ones are bootstrapped.

        Raises:
            ValueError: If `num_envs` <= 0.
        """

        if num_envs <= 0:
            raise ValueError("num_envs must be greater than 0.")

        states = self._reset_envs(num_envs)
        envs_done = np.zeros(num_envs, dtype=bool)

        collected_data = [
            {
                "states": [],
                "actions": [],
                "rewards": [],
                "terminations": [],
                "truncations": [],
                "values": [],
                "log_probs": [],
                "next_states": [],
            }
            for _ in range(num_envs)
        ]

        # Last valid observation per env, frozen when the env finishes, so the
        # auto-reset state of a finished env does not leak into the shared
        # agent state that running envs depend on.
        frozen_states = list(states) if hasattr(states, "__len__") else [states]

        safeguard = 0
        while not np.all(envs_done) and safeguard < self.max_nbr_iterations:

            # Done envs use their frozen state (stable pose for
            # MultiSimEnv.update_agent_states).
            step_states = type(states)(
                frozen_states[env_id] if envs_done[env_id] else states[env_id]
                for env_id in range(num_envs)
            ) if isinstance(states, tuple) else np.stack([
                frozen_states[env_id] if envs_done[env_id] else states[env_id]
                for env_id in range(num_envs)
            ])

            actions, log_probs, values = self._predict(step_states, deterministic=False)

            # Done envs get the neutral action, without profile transform, so
            # they do not perturb the joint physics of running envs.
            effective_actions = []
            for env_id, action in enumerate(actions):
                if envs_done[env_id]:
                    effective_actions.append(self.env.get_neutral_action(env_id, action))
                else:
                    transformed = ActionAdapter.transform(self._profile, action) if self._profile is not None else action
                    effective_actions.append(transformed)

            effective_actions = self._repack_actions(actions, effective_actions)
            next_states, rewards, terminations, truncations, final_states = self._step_envs(effective_actions)

            for env_id in range(num_envs):
                if envs_done[env_id]:
                    continue

                collected_data[env_id]["states"].append(states[env_id])
                collected_data[env_id]["actions"].append(actions[env_id])
                collected_data[env_id]["rewards"].append(rewards[env_id])
                collected_data[env_id]["terminations"].append(terminations[env_id])
                collected_data[env_id]["truncations"].append(truncations[env_id])
                collected_data[env_id]["values"].append(values[env_id])
                collected_data[env_id]["log_probs"].append(log_probs[env_id])
                # Real last observation (not auto-reset), for bootstrapping.
                collected_data[env_id]["next_states"].append(final_states[env_id])

            env_done = np.logical_or(terminations, truncations)

            # Freeze the last valid observation of envs that just finished.
            for env_id in range(num_envs):
                if env_done[env_id] and not envs_done[env_id]:
                    frozen_states[env_id] = states[env_id]

            envs_done = np.logical_or(envs_done, env_done)

            states = next_states
            safeguard += 1

        if safeguard >= self.max_nbr_iterations:
            logger.warning("[Warning] Safeguard triggered at %d steps (max_nbr_iterations=%d).", safeguard, self.max_nbr_iterations)

        # Trim, bootstrap and compute advantages per environment.
        episode_data = []
        for env_id in range(num_envs):
            trajectory = collected_data[env_id]

            if len(trajectory["states"]) == 0:
                continue

            # Episode end: first termination or truncation.
            done_mask = np.logical_or(trajectory["terminations"], trajectory["truncations"])
            first_done = self.find_first_true(done_mask)

            if first_done == -1:
                first_done = len(done_mask) - 1  # safeguard stop: use everything

            last_step_terminated = trajectory["terminations"][first_done]

            next_value = 0.0
            if not last_step_terminated:
                # Bootstrap on truncation or safeguard stop.
                next_value = self._get_bootstrap_value(trajectory["next_states"][first_done], env_id)

            episode = {}

            for key in trajectory:
                relevant_steps = trajectory[key][:first_done + 1]

                if key in ["states", "actions", "next_states"]:
                    episode[key] = np.stack(relevant_steps)
                elif key in ["terminations", "truncations"]:
                    episode[key] = np.asarray(relevant_steps, dtype=bool)
                else:
                    episode[key] = np.asarray(relevant_steps, dtype=np.float32)

            if self.normalize:
                episode["rewards"] = self.normalize_rewards(episode["rewards"])

            # Only true terminations zero the bootstrap.
            episode["advantages"] = self.compute_gae(
                episode["rewards"],
                episode["values"],
                episode["terminations"],  # only real terminals
                next_value,
                self.gamma,
                self.gae_lambda,
            )

            episode["returns"] = episode["advantages"] + episode["values"]
            episode["env_id"] = env_id
            episode_data.append(episode)

        return episode_data
