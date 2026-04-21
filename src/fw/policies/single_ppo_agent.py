import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

from typing import Optional, Tuple, Union
from fw.stop_condition import StopCondition
from fw.policies.ppo_policy import PPOPolicy
from fw.policies.base_ppo_agent import BasePPOAgent
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from fw.common.tools.logging_utils import dump_metrics_to_csv


class SinglePPOAgent(BasePPOAgent):
    """
    PPO agent operating on a single (vectorized) environment instance.

    This class implements policy optimization, environment interaction,
    and training logic for Proximal Policy Optimization (PPO). It supports
    both discrete and continuous action spaces and leverages vectorized
    environments for parallel rollout collection.

    Responsibilities include:
        - Policy inference and optimization
        - Parallel trajectory collection
        - PPO loss computation and updates
        - Model checkpointing and evaluation

    Inherits rollout and GAE logic from `BasePPOAgent`.
    """

    MINIMUM_ITERATIONS_FOR_BEST_SAVE = 20

    def __init__(self, *args, **kwargs):
        """
        Initialize the PPO agent.

        Args:
            *args: Positional arguments forwarded to `BasePPOAgent`.
            **kwargs: Keyword arguments forwarded to `BasePPOAgent`.

        Notes:
            - Automatically infers input and output dimensions from the environment.
            - Initializes the PPO policy network and optimizer.
            - Sets up running statistics for optional reward normalization.
        """

        super().__init__(*args, **kwargs)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Handle discrete and continuous action spaces
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.output_dim = self.action_space.n
        else:
            self.output_dim = self.action_space.shape[0]

        self.input_dim = self.observation_space.shape[0]
        self.policy = PPOPolicy(self.input_dim, self.output_dim, str(self.device))
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=self.learning_rate)
        self._vec_env = None

        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4  # avoid division by zero

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
            Tuple:
                - action: Selected action.
                - log_prob: Log-probability of the action.
                - value: Estimated state value.
        """

        actions, log_probs, values = self.policy.predict(state, deterministic)
        return actions[0], log_probs.item(), values.item()


    def train(self, all_episodes_data) -> float:
        """
        Perform PPO policy optimization on collected trajectories.

        Args:
            all_episodes_data (list[dict]): List of trajectory dictionaries, each containing:
                - states (np.ndarray)
                - actions (np.ndarray)
                - advantages (np.ndarray)
                - log_probs (np.ndarray)
                - returns (np.ndarray)

        Returns:
            float: Mean training loss across all updates. Returns NaN if no valid data.

        Notes:
            - Advantages are normalized before optimization.
            - Uses mini-batch SGD over multiple epochs.
            - Applies PPO clipping, value loss, and entropy regularization.
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

                mb_states = all_states[mb_idx]
                mb_actions = all_actions[mb_idx]
                mb_old_log_probs = all_old_log_probs[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_advantages = all_advantages[mb_idx]

                # Get current policy distribution
                action_mean, action_std, state_values = self.policy.network(mb_states)
                dist = torch.distributions.Normal(action_mean, action_std)

                # Adjust actions if action space is continuous
                if isinstance(self.action_space, gym.spaces.Box):
                    raw_actions = torch.atanh(mb_actions.clamp(-0.999, 0.999))
                else:
                    raw_actions = mb_actions
                log_probs = self.policy.calc_log_probs(dist, raw_actions, mb_actions)

                # PPO clipped loss
                ratios = torch.exp(log_probs - mb_old_log_probs)

                surrogate_1 = ratios * mb_advantages
                surrogate_2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages

                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Value loss
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
        Create a callable environment factory for vectorized execution.

        Returns:
            Callable[[], gym.Env]: Function that creates a monitored copy of the base environment.

        Notes:
            - Each environment is deep-copied to ensure independence.
            - Wrapped with `Monitor` for logging episode statistics.
        """

        def _init():
            """
            Create one monitored deep-copied environment instance.

            Returns:
                A monitored Gym environment copied from the agent base
                environment.
            """
            return Monitor(copy.deepcopy(self.env))

        return _init


    def _reset_envs(self, num_envs):
        """
        Initialize or reset vectorized environments.

        Args:
            num_envs (int): Number of parallel environments.

        Returns:
            np.ndarray: Initial observations of shape (num_envs, obs_dim).

        Notes:
            - Reuses existing vectorized environment if possible.
            - Uses `SubprocVecEnv` for multiple environments, `DummyVecEnv` otherwise.
            - Optionally randomizes environments if supported.
        """

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
        return self._vec_env.reset()


    def _step_envs(self, actions):
        """
        Step all environments with the given actions.

        Args:
            actions (np.ndarray): Actions of shape (num_envs, action_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - next_states: Next observations.
                - rewards: Rewards per environment.
                - terminations: Episode termination flags.
                - truncations: Episode truncation flags.

        Notes:
            - Supports both standard Gymnasium outputs and custom info dictionaries.
            - Falls back to default values if truncation data is unavailable.
        """

        # Step all environments
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


    def _predict(self, states, deterministic=False):
        """
        Compute batched actions, log-probabilities, and value estimates.

        Args:
            states (np.ndarray): Observations of shape (num_envs, obs_dim).
            deterministic (bool): Whether to use deterministic policy output.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - actions
                - log_probs
                - values
        """

        return self.policy.predict(states, deterministic=deterministic)


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


    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
    ):
        """
        Execute the full PPO training loop.

        Args:
            stop_condition (Optional[StopCondition]): Stopping criteria for training.
                If not provided, a default condition is used.

        Workflow:
            1. Collect trajectories using parallel environments.
            2. Compute advantages and returns.
            3. Update policy using PPO optimization.
            4. Log metrics and save checkpoints.
            5. Apply early stopping if conditions are met.

        Notes:
            - Tracks best-performing policy based on mean reward.
            - Periodically saves checkpoints.
            - Cleans up vectorized environments after training.
        """

        print(f"Starting training with {self.num_envs} parallel environments on a '{self.device}' device")

        last_save_index = 0
        training_metrics = []
        best_mean_reward = float('-inf')
        best_policy_file = f"Best_{self.env.type_name}_PPO_policy"
        stopping_condition = stop_condition if stop_condition is not None else StopCondition()

        for iteration in range(stopping_condition.max_time_steps):
            self.set_policy_eval()

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
            print(f"Iteration {iteration}, Average Reward: {mean_reward:.3f}, Loss: {loss:.3f}")

            # Save the best policy so far
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
                if iteration > 0:
                    self.save_policy(f"ppo_checkpoint_{iteration}")

        # Save final checkpoint
        self.save_policy(f'ppo_checkpoint_{last_save_index if last_save_index > 0 else stopping_condition.max_time_steps}')

        # Dump training metrics to CSV
        dump_metrics_to_csv(self.model_dir, "training_metrics.csv", training_metrics)

        if getattr(self, "_vec_env", None) is not None:
            self._vec_env.close()
            self._vec_env = None

        print("Training completed!")


    def load_policy(self, policy_file_name: str) -> None:
        """
        Load a saved policy from disk.

        Args:
            policy_file_name (str): Name or path of the policy file.

        Raises:
            RuntimeError: If loading fails.
        """

        try:
            if self.model_dir:
                policy_file_name = os.path.join(self.model_dir, policy_file_name)

            self.policy = self.policy.load(policy_file_name, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the policy '{policy_file_name}'.") from e


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
            #print(f"[save] Saved policy to {policy_file_name}.zip")
        except Exception as e:
            raise RuntimeError(f"Failed to save the policy '{policy_file_name}'.") from e


    def set_policy_eval(self):
        """
        Set the policy network to evaluation mode.

        Disables training-specific behavior such as dropout and
        batch normalization updates.
        """

        self.policy.network.eval()
