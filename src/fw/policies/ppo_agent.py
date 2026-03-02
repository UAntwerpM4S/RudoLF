import os
import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

from typing import Optional, Tuple
from fw.stop_condition import StopCondition
from fw.policies.base_agent import BaseAgent
from fw.policies.ppo_policy import PPOPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent for reinforcement learning.

    This class handles the training, evaluation, and policy management for PPO-based reinforcement learning agents.
    """

    MINIMUM_ITERATIONS_FOR_BEST_SAVE = 20

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


    def __init__(
        self,
        *args,
        max_nbr_iterations: int = 200000,
        normalize: bool = False,
        **kwargs
    ):
        """
        Initialize PPOAgent with environment and training hyperparameters.

        This constructor passes core arguments to the BaseAgent and adds two specific options
        for training behavior: maximum number of iterations and reward normalization.

        Args:
            *args: Positional arguments passed to the BaseAgent initializer.
            max_nbr_iterations (int, optional): The maximum number of environment steps before terminating training.
                Used as a safeguard to prevent infinite loops. Defaults to 200000.
            normalize (bool, optional): Whether to normalize rewards using running statistics.
                Helps stabilize learning when reward scales vary significantly. Defaults to False.
            **kwargs: Keyword arguments passed to the BaseAgent initializer.
        """
        super().__init__(*args, **kwargs)

        # Handle discrete and continuous action spaces
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.output_dim = self.action_space.n
        else:
            self.output_dim = self.action_space.shape[0]

        if len(self.observation_space.shape) > 1:
            self.input_dim = self.observation_space.shape[-1]
        else:
            self.input_dim = self.observation_space.shape[0]
        self.policy = PPOPolicy(self.input_dim, self.output_dim, self.device)
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=self.learning_rate)
        self.max_nbr_iterations = max_nbr_iterations
        self.normalize = normalize
        self._vec_env = None

        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4  # avoid division by zero


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


    def predict(self, state: np.ndarray, deterministic: bool = False) -> Tuple:
        """
        Predict an action for a given state using the policy.

        Args:
            state (np.ndarray): Current state as a NumPy array.
            deterministic (boolean): Whether the action should be deterministic or not.

        Returns:
            Tuple: A tuple containing the predicted action and log probability.
        """
        actions, log_probs, values = self.policy.predict(state, deterministic)
        return actions, log_probs, values


    def train(self, all_episodes_data) -> float:
        """
        Train the PPO policy on a batch of data.

        This method trains the policy network on the provided batch of episode data by optimizing the PPO objective.

        Args:
            all_episodes_data (list): List of dictionaries, each containing episode data:
                - states: numpy array of states
                - actions: numpy array of actions
                - advantages: numpy array of advantages
                - old_log_probs: numpy array of old log probabilities
                - returns: numpy array of returns

        Returns:
            float: Average loss value after training.
        """
        if not all_episodes_data:
            return float('nan')

        self.policy.network.train()

        # Concatenate all episode data into single arrays for batch processing
        all_states = torch.as_tensor(np.concatenate([ep['states'] for ep in all_episodes_data]),
                                     dtype=torch.float32).to(self.device)
        all_actions = torch.as_tensor(np.concatenate([ep['actions'] for ep in all_episodes_data]),
                                      dtype=torch.float32).to(self.device)
        all_advantages = torch.as_tensor(np.concatenate([ep['advantages'] for ep in all_episodes_data]),
                                         dtype=torch.float32).to(self.device)
        all_old_log_probs = torch.as_tensor(np.concatenate([ep['log_probs'] for ep in all_episodes_data]),
                                            dtype=torch.float32).to(self.device)
        all_returns = torch.as_tensor(np.concatenate([ep['returns'] for ep in all_episodes_data]),
                                      dtype=torch.float32).to(self.device)

        if all_advantages.numel() <= 1:
            print(f"[Warning] Very short trajectory detected. Length: {all_advantages.numel()}")
            return float('nan')

        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Train over multiple epochs
        batch_size = all_states.shape[0]
        loss_per_epoch = []

        for _ in range(self.num_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                # Slicing existing tensors is faster than re-creating them
                mb_states = all_states[mb_idx]
                mb_actions = all_actions[mb_idx]
                mb_old_log_probs = all_old_log_probs[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_advantages = all_advantages[mb_idx]

                # Get current policy distribution
                action_mean, action_std, state_values = self.policy.network(mb_states)
                dist = torch.distributions.Normal(action_mean, action_std)

                # Use the raw actions logic safely
                if isinstance(self.action_space, gym.spaces.Box):
                    # Numerical safety for atanh
                    raw_actions = torch.atanh(mb_actions.clamp(-0.999, 0.999))
                else:
                    raw_actions = mb_actions
                log_probs = self.policy.calc_log_probs(dist, raw_actions, mb_actions)

                # PPO Objective
                ratios = torch.exp(log_probs - mb_old_log_probs)

                surrogate_1 = ratios * mb_advantages
                surrogate_2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages

                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Value Loss: MSE between predicted value and GAE-based returns
                value_loss = torch.nn.functional.mse_loss(state_values.squeeze(-1), mb_returns)
                entropy_loss = dist.entropy().sum(dim=-1).mean()    # Encourage exploration

                # Total loss
                loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy_loss)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                loss_per_epoch.append(loss.item())

        # Return mean loss over all updates
        return np.mean(loss_per_epoch)


    def make_env(self):
        """
        Utility function for multiprocessing env creation.
        """

        def _init():
            return Monitor(copy.deepcopy(self.env))

        return _init


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


    def run_parallel_episodes(self, num_envs) -> list:
        """
        Run multiple episodes in parallel using Gym vectorized environments.
        Collect trajectories per (env_id, agent_id) and compute GAE.

        Args:
            num_envs (int): Number of parallel environments to run.

        Returns:
            list: A list of dictionaries containing episode data for each environment.
        """

        if num_envs <= 0:
            raise ValueError("num_envs must be greater than 0.")

        # Create vectorized environments once and reuse
        # follows the approach of https://stable-baselines3.readthedocs.io/en/v0.11.1/_modules/stable_baselines3/common/on_policy_algorithm.html
        # where the vector is passed for each rollout instead of recreating it for all environments
        if self._vec_env is None or getattr(self._vec_env, "num_envs", None) != num_envs:
            # Recreate vec_env only if it does not exist or if the requested number of envs changed
            env_fns = [self.make_env() for _ in range(num_envs)]
            self._vec_env = SubprocVecEnv(env_fns) if num_envs > 1 else DummyVecEnv(env_fns)

        # ------------------------------------------------------------------
        # Optional environment randomization
        # ------------------------------------------------------------------
        if hasattr(self._vec_env, "env_method"):
            self._vec_env.env_method("randomize")

        # Reset all environments
        states = self._vec_env.reset()

        if states.ndim == 3:
            # Multi-agent case: (num_envs, num_agents, obs_dim)
            num_envs, num_agents, obs_dim = states.shape
        elif states.ndim == 2:
            # Single-agent case: (num_envs, obs_dim)
            num_envs, obs_dim = states.shape
            states = states[:, None, :]
            num_agents = 1
        else:
            raise ValueError(
                f"Unexpected states shape {states.shape}. "
                "Expected 2D (num_envs, obs_dim) or 3D (num_envs, num_agents, obs_dim)."
            )

        # Track which environments are finished
        envs_done = np.zeros(num_envs, dtype=bool)

        # Storage for episode data
        collected_data = [
            [
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
                for _ in range(num_agents)
            ]
            for _ in range(num_envs)
        ]

        safeguard = 0
        while not np.all(envs_done) and safeguard < self.max_nbr_iterations:
            # Inference: No gradients needed here
            actions, log_probs, values = self.predict(states, deterministic=False)

            if len(self.observation_space.shape) == 1 and actions.ndim == 3 and actions.shape[1] == 1:
                actions = actions[:, 0, :]

            # Step all environments
            next_states, next_rewards, next_terminations, infos = self._vec_env.step(actions)

            if infos and all(
                    all(k in env_info for k in ("rewards", "terminations", "truncations"))
                    for env_info in infos
            ):
                rewards = np.stack([env_info["rewards"] for env_info in infos]).astype(np.float32)
                terminations = np.stack([env_info["terminations"] for env_info in infos]).astype(bool)
                truncations = np.stack([env_info["truncations"] for env_info in infos]).astype(bool)
            else:
                rewards = next_rewards
                terminations = next_terminations
                truncations = np.zeros((num_envs, num_agents), dtype=bool)

            if len(self.observation_space.shape) == 1:
                if actions.ndim == 2:  # (num_envs, action_dim)
                    actions = actions[:, None, :]  # → (num_envs, 1, action_dim)

                if log_probs.ndim == 1:  # (num_envs,)
                    log_probs = log_probs[:, None]  # → (num_envs, 1)

                if values.ndim == 1:  # (num_envs,)
                    values = values[:, None]  # → (num_envs, 1)

                if next_states.ndim == 2:
                    next_states = next_states[:, None, :]
                    rewards = next_rewards[:, None]
                    terminations = next_terminations[:, None]

            # Store per-agent transitions
            for env_id in range(num_envs):
                if envs_done[env_id]:
                    continue  # Skip already completed environments

                for agent_id in range(num_agents):
                    collected_data[env_id][agent_id]["states"].append(states[env_id, agent_id])
                    collected_data[env_id][agent_id]["actions"].append(actions[env_id, agent_id])
                    collected_data[env_id][agent_id]["rewards"].append(rewards[env_id, agent_id])
                    collected_data[env_id][agent_id]["terminations"].append(terminations[env_id, agent_id])
                    collected_data[env_id][agent_id]["truncations"].append(truncations[env_id, agent_id])
                    collected_data[env_id][agent_id]["values"].append(values[env_id, agent_id])
                    collected_data[env_id][agent_id]["log_probs"].append(log_probs[env_id, agent_id])
                    collected_data[env_id][agent_id]["next_states"].append(next_states[env_id, agent_id])

            # Update done flags at environment level
            env_done = np.all(terminations | truncations, axis=1)
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
            for agent_id in range(num_agents):
                trajectory = collected_data[env_id][agent_id]

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
                    with torch.no_grad():
                        last_state = torch.FloatTensor(trajectory["next_states"][first_done]).unsqueeze(0).to(self.device)
                        _, _, next_values = self.policy.network(last_state)
                        next_value = next_values.squeeze(0).item()

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


    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
    ):
        """
        Train the policy using multiple parallel environments.

        This method runs the training loop for the specified number of timesteps using parallel environments.
        It collects experience from the environments, updates the policy, logs training metrics, and saves
        agent checkpoints at regular intervals.

        Args:
            stop_condition (StopCondition): Condition that defines when training should stop.
        """
        print(f"Starting training with {self.num_envs} parallel environments on a '{self.device}' device")

        last_save_index = 0
        training_metrics = []
        best_mean_reward = float('-inf')
        best_policy_file = f"Best_{self.env.type_name}_PPO_policy"
        stopping_condition = stop_condition if stop_condition is not None else StopCondition()

        for iteration in range(stopping_condition.max_time_steps):
            self.policy.network.eval()

            # Prepare batch for training
            episode_data = []
            total_reward = 0
            total_advantage = 0

            # Collect batch data from multiple parallel environments
            batch_data = self.run_parallel_episodes(self.num_envs)

            for episode_result in batch_data:
                episode_data.append(episode_result)
                total_reward += np.mean(episode_result['rewards'])
                total_advantage += np.mean(episode_result['advantages'])

            # Update policy using all collected episodes
            loss = self.train(episode_data)

            # Log results
            mean_reward = total_reward / len(batch_data) if batch_data else 0
            mean_advantage = total_advantage / len(batch_data) if batch_data else 0

            training_metrics.append({'loss': loss, 'advantage': mean_advantage, 'reward': mean_reward})

            # Save best policy so far
            if iteration >= self.MINIMUM_ITERATIONS_FOR_BEST_SAVE and mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                self.save_policy(best_policy_file)

            should_stop, reason = stopping_condition.should_stop(
                current_step=iteration,
                mean_reward=mean_reward,
                mean_loss=loss
            )

            # Check if the condition is met for early stopping
            if should_stop:
                print(f"Stopping: {reason}")
                last_save_index = iteration
                break

            # Log evaluation results periodically
            if iteration % self.eval_frequency == 0:
                print(f"Iteration {iteration}, Average Reward: {mean_reward:.3f}, Loss: {loss:.3f}")

                # Save checkpoint every 'self.eval_frequency' iterations
                if iteration > 0 and iteration % self.eval_frequency == 0:
                    self.save_policy(f"ppo_checkpoint_{iteration}")

        # Save final checkpoint
        self.save_policy(f'ppo_checkpoint_{last_save_index if last_save_index > 0 else stopping_condition.max_time_steps}')

        # Dump training metrics to CSV
        self.dump_metrics_to_csv("training_metrics.csv", training_metrics)

        if getattr(self, "_vec_env", None) is not None:
            self._vec_env.close()
            self._vec_env = None

        print("Training completed!")


    def dump_metrics_to_csv(self, csv_filename, training_metrics):
        """
        Save training metrics to a CSV file.

        This method writes the training metrics (such as loss, advantage, reward)
        to a CSV file with headers.

        Args:
            csv_filename (str): The name of the CSV file to write.
            training_metrics (list): A list of dictionaries containing the training metrics.

        """
        if not training_metrics:
            print("Warning: No training metrics to write.")
            return

        filepath = os.path.join(self.model_dir, csv_filename)

        with open(filepath, mode="w", newline="") as file:
            # Get all possible keys (assuming all dicts have the same structure)
            fieldnames = training_metrics[0].keys()

            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_metrics)


    def load_policy(self, policy_file_name: str) -> None:
        """
        Load a trained agent's policy.

        Args:
            policy_file_name (str): Name of the policy file to load.

        Raises:
            RuntimeError: If no agent is created or the policy fails to load.
        """
        try:
            if self.model_dir:
                policy_file_name = os.path.join(self.model_dir, policy_file_name)

            self.policy = self.policy.load(policy_file_name, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the policy '{policy_file_name}'.") from e


    def save_policy(self, policy_file_name: str) -> None:
        """
        Save the trained agent's policy.

        Args:
            policy_file_name (str): Name of the policy file to save.

        Raises:
            RuntimeError: If no agent is created or the policy fails to save.
        """
        try:
            if self.model_dir:
                policy_file_name = os.path.join(self.model_dir, policy_file_name)

            self.policy.save(policy_file_name)
        except Exception as e:
            raise RuntimeError(f"Failed to save the policy '{policy_file_name}'.") from e


    def set_policy_eval(self):
        """
        Set policy in evaluation mode.
        """
        self.policy.network.eval()
