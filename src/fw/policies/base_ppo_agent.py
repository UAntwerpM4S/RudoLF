from typing import Optional

import numpy as np

from abc import abstractmethod
from fw.policies.base_agent import BaseAgent
from fw.environments.base_env import BaseEnv


class BasePPOAgent(BaseAgent):
    """
    Abstract base class for reinforcement learning agents.

    This class provides a unified interface and shared hyperparameters
    for reinforcement learning algorithms. Subclasses must implement
    the core training and inference logic.
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

        GAE helps in reducing the variance of the policy gradient estimates by using both bootstrapping and Monte Carlo estimates.

        Args:
            rewards (np.ndarray): Array of rewards for the batch.
            values (np.ndarray): Array of value estimates for each state.
            dones (np.ndarray): Array of done flags indicating if the episode ended.
            next_value (float): Value estimate for the state after the batch.
            gamma (float): Discount factor for future rewards.
            lambda_ (float): GAE parameter that controls bias-variance tradeoff (default: 0.95).

        Returns:
            np.ndarray: Computed advantages as a NumPy array.
        """

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        # Reverse iteration for GAE calculation
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
        Find the index of the first True value in a boolean array.
        If no True value exists, return -1.

        Args:
            bool_array (np.ndarray): A NumPy array of boolean values.

        Returns:
            int: The index of the first True value, or -1 if not found.
        """

        # Find the index of the first True value
        true_indices = np.where(bool_array)[0]

        # Return the first index if it exists, otherwise return -1
        return true_indices[0] if true_indices.size > 0 else -1


    def __init__(
        self,
        environment: BaseEnv,
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
    ):
        """
        Initialize the base agent with environment and hyperparameters.

        Args:
            environment (BaseEnv): The training environment.
            num_envs (int): The number of training environments (for parallel training).
            eval_frequency (int): Number of training iterations between evaluations.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 3e-4.
            clip_range (float, optional): PPO clipping range. Default is 0.2.
            value_loss_coef (float, optional): Coefficient for value loss. Default is 0.5.
            max_grad_norm (float, optional): Maximum gradient norm for gradient clipping. Default is 0.5.
            gamma (float, optional): Discount factor for future rewards. Default is 0.99.
            gae_lambda (float, optional): Lambda for Generalized Advantage Estimation. Default is 0.95.
            entropy_coef (float, optional): Coefficient for entropy bonus. Default is 0.01.
            num_epochs (int, optional): Number of training epochs per update. Default is 10.
            batch_size (int, optional): Mini-batch size. Default is 64.
            device (str, optional): Device for training, either 'cpu' or 'cuda'. Default is 'cpu'.
            model_dir (str, optional): Folder in which the generated agents are saved.
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
        )

        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        self.max_nbr_iterations = max_nbr_iterations
        self.normalize = normalize


    # -------------------------
    # Abstract interface
    # -------------------------
    @abstractmethod
    def _reset_envs(self, num_envs):
        """
        Returns:
            np.ndarray of shape (num_envs, obs_dim)
        """

        pass

    @abstractmethod
    def _step_envs(self, actions):
        """
        Args:
            actions: np.ndarray shape (num_envs, action_dim)

        Returns:
            next_states, rewards, terminations, truncations
        """

        pass

    @abstractmethod
    def _predict(self, states, deterministic=False):
        """
        Returns:
            actions, log_probs, values
        """

        pass

    @abstractmethod
    def _get_bootstrap_value(self, states, env_id: Optional[int] = None):
        """
        Returns value estimate for final state
        """

        pass


    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running statistics (Welford's algorithm).

        Args:
            rewards: Raw rewards array of shape (batch_size,).

        Returns:
            Normalized rewards with zero mean and unit variance.
        """

        # Welford's algorithm for running stats
        batch_mean = np.mean(rewards)
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

        return (rewards - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)


    # -------------------------
    # Core PPO rollout
    # -------------------------
    def run_parallel_episodes(self, num_envs) -> list:
        """
        Collect one episode per environment in parallel.

        All environments are reset and stepped synchronously until each has
        reached a terminal or truncated state, or until the maximum number of
        iterations is reached. Trajectories are stored separately per environment.

        Args:
            num_envs (int): Number of environments to simulate (must match `self.num_envs`).

        Returns:
            list[dict]: List of per-environment trajectory dictionaries. Each dict contains:
                - states (np.ndarray): Sequence of observations.
                - actions (np.ndarray): Sequence of actions.
                - rewards (np.ndarray): Sequence of rewards.
                - dones (np.ndarray): Episode termination flags.
                - values (np.ndarray): Value function estimates.
                - log_probs (np.ndarray): Log-probabilities of actions.
                - advantages (np.ndarray): Generalized Advantage Estimates (GAE).
                - returns (np.ndarray): Discounted returns (advantages + values).

        Notes:
            - Episodes may have different lengths across environments.
            - Bootstrapping is applied if an episode does not terminate within
              `max_nbr_iterations`.
            - Reward normalization (if enabled) is applied per trajectory.
        """

        if num_envs <= 0:
            raise ValueError("num_envs must be greater than 0.")

        # Reset all environments
        states = self._reset_envs(num_envs)
        envs_done = np.zeros(num_envs, dtype=bool)

        # Storage for episode data
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

        safeguard = 0
        while not np.all(envs_done) and safeguard < self.max_nbr_iterations:
            # Select actions and get value predictions
            actions, log_probs, values = self._predict(states, deterministic=False)

            next_states, rewards, terminations, truncations = self._step_envs(actions)

            # Store per-agent transitions
            for env_id in range(num_envs):
                if envs_done[env_id]:
                    continue  # Skip already completed environments

                collected_data[env_id]["states"].append(states[env_id])
                collected_data[env_id]["actions"].append(actions[env_id])
                collected_data[env_id]["rewards"].append(rewards[env_id])
                collected_data[env_id]["terminations"].append(terminations[env_id])
                collected_data[env_id]["truncations"].append(truncations[env_id])
                collected_data[env_id]["values"].append(values[env_id])
                collected_data[env_id]["log_probs"].append(log_probs[env_id])
                collected_data[env_id]["next_states"].append(next_states[env_id])

            # Update done flags at environment level
            env_done = np.logical_or(terminations, truncations)
            envs_done = np.logical_or(envs_done, env_done)

            # Update states for the next step
            states = next_states
            safeguard += 1

        if safeguard >= 135000:
            print(f"Safeguard surpassed the 135000 boundary: {safeguard}.")

        if safeguard >= self.max_nbr_iterations:
            print(f"Safeguard triggered due to reaching max_iterations ({self.max_nbr_iterations}).")

        # self._vec_env.close()

        # Process episode data with proper trimming
        episode_data = []
        for env_id in range(num_envs):
            trajectory = collected_data[env_id]

            if len(trajectory["states"]) == 0:
                continue

            # ----------------------------------------------------------
            # Determine episode end (termination OR truncation)
            # ----------------------------------------------------------

            # First we'll trim the episode to the correct length, i.e. the moment where 'done == True' was reached.
            # After that, the episode is padded with irrelevant data until all environments have reached their 'done' state.

            # Find the first True in dones for this environment
            done_mask = np.logical_or(trajectory["terminations"], trajectory["truncations"])
            first_done = self.find_first_true(done_mask)

            # Handle cases where no True is found (episode did not terminate)
            if first_done == -1:
                first_done = len(done_mask) - 1  # Use the entire episode

            # ----------------------------------------------------------
            # Determine bootstrap condition
            # ----------------------------------------------------------
            last_step_terminated = trajectory["terminations"][first_done]

            next_value = 0.0
            if not last_step_terminated:
                # We bootstrap for:
                # - truncation
                # - safeguard stop
                next_value = self._get_bootstrap_value(trajectory["next_states"][first_done])

            # ----------------------------------------------------------
            # Trim trajectory to valid region
            # ----------------------------------------------------------
            episode = {}

            for key in trajectory:
                relevant_steps = trajectory[key][:first_done + 1]

                if key in ["states", "actions", "next_states"]:
                    episode[key] = np.stack(relevant_steps)
                elif key in ["terminations", "truncations"]:
                    episode[key] = np.asarray(relevant_steps, dtype=bool)
                else:
                    episode[key] = np.asarray(relevant_steps, dtype=np.float32)

            # ----------------------------------------------------------
            # Optional reward normalization
            # ----------------------------------------------------------
            if self.normalize:
                episode["rewards"] = self.normalize_rewards(episode["rewards"])

            # ----------------------------------------------------------
            # Compute GAE (ONLY true terminations zero bootstrap)
            # ----------------------------------------------------------

            # Compute advantages and returns
            episode["advantages"] = self.compute_gae(
                episode["rewards"],
                episode["values"],
                episode["terminations"],  # only real terminals
                next_value,
                self.gamma,
                self.gae_lambda,
            )

            episode["returns"] = episode["advantages"] + episode["values"]
            episode_data.append(episode)

        return episode_data
