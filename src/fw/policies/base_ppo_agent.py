import numpy as np

from typing import Optional
from abc import abstractmethod
from fw.policies.base_agent import BaseAgent
from fw.environments.base_env import BaseEnv
from fw.profile_applier import ActionAdapter
from fw.config_profile import VesselProfile


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


    @staticmethod
    def _repack_actions(original_actions, transformed_actions: list):
        """
        Repack a list of (transformed) actions into the same container type
        as the original actions produced by `_predict`.

        `SinglePPOAgent._predict` returns an `np.ndarray`; `MultiPPOAgent`
        returns a `tuple`.  `_step_envs` in each subclass expects the same
        type back, so the list built during the rollout loop must be converted.

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

        return (rewards - new_mean) / (np.sqrt(new_var) + 1e-8)


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

        # Snapshot of each environment's last valid observation.  Once an
        # environment finishes its episode, its entry here is frozen so that
        # subsequent calls to _predict and _step_envs receive a stable,
        # meaningful observation rather than an auto-reset initial state from
        # a ghost episode.  This prevents finished environments from
        # contaminating the shared agent-state that still-running environments
        # rely on for their observations and rewards.
        frozen_states = list(states) if hasattr(states, "__len__") else [states]

        safeguard = 0
        while not np.all(envs_done) and safeguard < self.max_nbr_iterations:

            # Build the observation array for this step: use the frozen last
            # valid state for done environments so _predict does not waste a
            # forward pass on a ghost auto-reset observation, and so that
            # MultiSimEnv.update_agent_states sees a stable last-known pose.
            step_states = type(states)(
                frozen_states[env_id] if envs_done[env_id] else states[env_id]
                for env_id in range(num_envs)
            ) if isinstance(states, tuple) else np.stack([
                frozen_states[env_id] if envs_done[env_id] else states[env_id]
                for env_id in range(num_envs)
            ])

            # Policy predicts actions for all agents from live (or frozen) states
            actions, log_probs, values = self._predict(step_states, deterministic=False)

            # Replace actions for done environments with a neutral input BEFORE
            # stepping so the simulator receives a stable helm-zero signal for
            # ships that have already finished their episode.
            # Profile transforms are NOT applied to neutral actions — the neutral
            # signal must reach the simulator unmodified so it does not perturb
            # the joint physics state for still-running environments.
            effective_actions = []
            for env_id, action in enumerate(actions):
                if envs_done[env_id]:
                    effective_actions.append(self.env.get_neutral_action(env_id, action))
                else:
                    transformed = ActionAdapter.transform(self._profile, action) if self._profile is not None else action
                    effective_actions.append(transformed)

            # Restore the container type that _step_envs expects, i.e.
            # np.ndarray for SinglePPOAgent, tuple for MultiPPOAgent.
            effective_actions = self._repack_actions(actions, effective_actions)
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

            # Freeze the last valid observation for environments that just
            # finished, before envs_done is updated.
            for env_id in range(num_envs):
                if env_done[env_id] and not envs_done[env_id]:
                    frozen_states[env_id] = states[env_id]

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
