import numpy as np

from typing import Optional
from abc import abstractmethod
from fw.policies.base_agent import BaseAgent
from fw.environments.base_env import BaseEnv


class BasePPOAgent(BaseAgent):
    """
    Abstract base class for Proximal Policy Optimization (PPO) agents.

    This class extends `BaseAgent` with PPO-specific functionality, including
    trajectory collection, reward normalization, and Generalized Advantage
    Estimation (GAE). It defines a standardized interface for interacting
    with environments and computing policy/value updates.

    Subclasses are responsible for implementing environment interaction
    (`_reset_envs`, `_step_envs`) and model inference (`_predict`,
    `_get_bootstrap_value`).
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
        Return the index of the first True value in a boolean array.

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
        Initialize the PPO agent with environment and training hyperparameters.

        Args:
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
            max_nbr_iterations (int): Maximum steps per rollout.
            normalize (bool): Whether to normalize rewards.

        Attributes:
            max_nbr_iterations: Safeguard value to avoid infinite stepping through environments.
            normalize: Flag that indicates whether reward values should be normalized.
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
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - next_states: Next observations.
                - rewards: Rewards per environment.
                - terminations: Episode termination flags.
                - truncations: Episode truncation flags.
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
        Estimate value for bootstrapping.

        Used when an episode ends due to truncation or iteration limits.

        Args:
            states (np.ndarray): State(s) to evaluate.
            env_id (Optional[int]): Environment index (for multi-env setups).

        Returns:
            float: Estimated value of the given state.
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

        old_mean = self.reward_mean
        old_var = self.reward_var
        self.reward_mean = new_mean
        self.reward_var = new_var
        self.reward_count = tot_count

        return (rewards - old_mean) / (np.sqrt(old_var) + 1e-8)


    def run_parallel_episodes(self, num_envs: int) -> list:
        """
        Collect one trajectory per environment in parallel.

        Environments are stepped synchronously until each one reaches a terminal
        or truncated state, or until `max_nbr_iterations` is reached.

        Actions for agents that have already finished are replaced with zeros
        so the joint simulator is not perturbed after an agent's episode ends.

        Args:
            num_envs (int): Number of environments to simulate.

        Returns:
            list[dict]: List of trajectory dictionaries (one per environment), each containing:
                - states (np.ndarray): Observations.
                - actions (np.ndarray): Actions taken.
                - rewards (np.ndarray): Rewards received.
                - terminations (np.ndarray): True termination flags.
                - truncations (np.ndarray): Truncation flags.
                - values (np.ndarray): Value estimates.
                - log_probs (np.ndarray): Action log-probabilities.
                - advantages (np.ndarray): GAE advantages.
                - returns (np.ndarray): Discounted returns.

        Notes:
            - Trajectories are trimmed to the first terminal/truncated step.
            - Bootstrapping is applied if termination is not reached.
            - Reward normalization is applied per trajectory if enabled.
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

            # Policy predicts actions for all agents from live states
            actions, log_probs, values = self._predict(states, deterministic=False)

            # Build effective_actions: zero out done agents before stepping the sim.
            # This must happen BEFORE _step_envs so the simulator receives a
            # stable neutral input for ships that have already finished.
            effective_actions = list(actions)

            for env_id in range(num_envs):
                if envs_done[env_id]:
                    # Replace with zero action (neutral helm) — sim receives a
                    # stopped ship rather than a randomly-acting one
                    effective_actions[env_id] = np.zeros_like(actions[env_id])

            # Convert back to the same container type _step_envs expects.
            # SinglePPOAgent uses np.ndarray; MultiPPOAgent uses tuple.
            if isinstance(actions, np.ndarray):
                effective_actions = np.stack(effective_actions)
            else:
                effective_actions = tuple(effective_actions)

            next_states, rewards, terminations, truncations = self._step_envs(effective_actions)

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

        if safeguard >= self.max_nbr_iterations:
            print(f"[Warning] Safeguard triggered at {safeguard} steps (max_nbr_iterations={self.max_nbr_iterations}).")

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
                next_value = self._get_bootstrap_value(trajectory["next_states"][first_done], env_id)

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
            episode["env_id"] = env_id
            episode_data.append(episode)

        return episode_data
