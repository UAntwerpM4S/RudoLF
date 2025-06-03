import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

from typing import Optional
from gymnasium.vector import SyncVectorEnv
from fw.stop_condition import StopCondition
from fw.policies.base_model import BaseModel
from fw.policies.ppo_policy import PPOPolicy


class PPOModel(BaseModel):
    """
    Proximal Policy Optimization (PPO) Model for reinforcement learning.

    This class handles the training, evaluation, and policy management for PPO-based reinforcement learning agents.
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
            advantages[t] = last_gae = delta + gamma * lambda_ * next_non_terminal * last_gae

        return advantages


    def __init__(
        self,
        environment: gym.Env,
        eval_frequency: int,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        num_epochs: int = 10,
        normalize: bool = False,
        max_nbr_iterations: int = 125000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize PPOModel with environment and hyperparameters.

        This initializes the PPO model with the specified environment and learning parameters.

        Args:
            environment (gym.Env): Gym-like environment for training.
            eval_frequency (int): Frequency of evaluation during training.
            learning_rate (float, optional): Learning rate for the optimizer (default: 3e-4).
            clip_range (float, optional): Clipping range for PPO loss (default: 0.2).
            value_loss_coef (float, optional): Coefficient for value loss in total loss (default: 0.5).
            max_grad_norm (float, optional): Maximum gradient norm for clipping (default: 0.5).
            gamma (float, optional): Discount factor for rewards (default: 0.99).
            gae_lambda (float, optional): GAE parameter (default: 0.95).
            entropy_coef (float, optional): Coefficient for entropy bonus (default: 0.01).
            num_epochs (int, optional): Number of epochs for training on collected data (default: 10).
            normalize (bool, optional): Whether to normalize the rewards between [-1,1] (default: False).
            max_nbr_iterations (int, optional): Maximum number of steps that is allowed before aborting (default: 125000).
            batch_size (int, optional): Size of the (mini) batches (default: 64).
            device (str, optional): Device for computation ('cpu' or 'cuda') (default: 'cpu').
        """
        super().__init__(environment, eval_frequency, learning_rate, clip_range, value_loss_coef, max_grad_norm, gamma,
                         gae_lambda, entropy_coef, num_epochs, normalize, max_nbr_iterations, batch_size, device)

        # Handle discrete and continuous action spaces
        if isinstance(environment.action_space, gym.spaces.Discrete):
            self.output_dim = environment.action_space.n
        else:
            self.output_dim = environment.action_space.shape[0]

        self.input_dim = environment.observation_space.shape[0]
        self.policy = PPOPolicy(self.input_dim, self.output_dim, device)
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=self.learning_rate)

        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4  # avoid division by zero


    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        # Welford's algorithm for running stats
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)

        delta = batch_mean - self.reward_mean
        tot_count = self.reward_count + batch_count

        new_mean = self.reward_mean + delta * batch_count / tot_count
        m_a = self.reward_var * self.reward_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / tot_count
        new_var = M2 / tot_count

        self.reward_mean = new_mean
        self.reward_var = new_var
        self.reward_count = tot_count

        return (rewards - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)


    def predict(self, state: np.ndarray) -> tuple:
        """
        Predict an action for a given state using the policy.

        Args:
            state (np.ndarray): Current state as a NumPy array.

        Returns:
            tuple: A tuple containing the predicted action and log probability.
        """
        actions, log_probs, values = self.policy.predict(state)
        return actions[0], log_probs.item(), values.item()


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
        self.policy.network.train()

        # Concatenate all episode data into single arrays for batch processing
        all_states = np.concatenate([ep['states'] for ep in all_episodes_data])
        all_actions = np.concatenate([ep['actions'] for ep in all_episodes_data])
        all_advantages = np.concatenate([ep['advantages'] for ep in all_episodes_data])
        all_old_log_probs = np.concatenate([ep['log_probs'] for ep in all_episodes_data])
        all_returns = np.concatenate([ep['returns'] for ep in all_episodes_data])

        # Convert data to tensors
        states = torch.as_tensor(all_states, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(all_actions, dtype=torch.float32).to(self.device)
        advantages = torch.as_tensor(all_advantages, dtype=torch.float32).to(self.device)
        old_log_probs = torch.as_tensor(all_old_log_probs, dtype=torch.float32).to(self.device)
        returns = torch.as_tensor(all_returns, dtype=torch.float32).to(self.device)

        if advantages.numel() <= 1:
            print(f"[Warning] Very short trajectory detected. Length: {advantages.numel()}")
            return float('nan')

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train over multiple epochs
        batch_size = states.shape[0]
        minibatch_size = self.batch_size

        loss_per_epoch = []

        for _ in range(self.num_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Get current policy distribution
                action_mean, action_std, state_values = self.policy.network(mb_states)
                dist = torch.distributions.Normal(action_mean, action_std)

                # Adjust actions if action space is continuous
                if isinstance(self.env.action_space, gym.spaces.Box):
                    raw_actions = torch.atanh(mb_actions.clamp(-0.99, 0.99))
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
                entropy = dist.entropy().sum(dim=-1).mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                loss_per_epoch.append(loss.detach().cpu().item())

        # Return mean loss over all updates
        return np.mean(loss_per_epoch)


    def create_env(self):
        """
        Create a copy of the environment for parallel runs.

        Returns:
            gym.Env: A new instance of the environment.
        """
        env = copy.deepcopy(self.get_env())

        # Optionally randomize the environment's initial state
        if hasattr(env, "randomize"):
            env.randomize()

        return env


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

        Args:
            num_envs (int): Number of parallel environments to run.

        Returns:
            list: A list of dictionaries containing episode data for each environment.
        """
        if num_envs <= 0:
            raise ValueError("num_envs must be greater than 0.")

        # Create vectorized environments
        env_fns = [lambda: self.create_env() for _ in range(num_envs)]  # Clone environment
        vec_env = SyncVectorEnv(env_fns)  # Sync execution across environments

        # Reset all environments
        states, _ = vec_env.reset()
        envs_done = np.zeros(num_envs, dtype=bool)

        # Storage for episode data
        collected_data = {k: [] for k in ["states", "actions", "rewards", "dones", "values", "log_probs"]}
        safeguard = 0

        while not np.all(envs_done) and safeguard < self.max_nbr_iterations:
            # Select actions and get value predictions
            actions, log_probs, values = self.policy.predict(states)

            # Step all environments
            next_states, rewards, terminations, truncations, _ = vec_env.step(actions)
            envs_done = np.logical_or(envs_done, np.logical_or(terminations, truncations))

            # Store episode data
            collected_data["states"].append(states)
            collected_data["actions"].append(actions)
            collected_data["rewards"].append(rewards)
            collected_data["dones"].append(np.logical_or(terminations, truncations))
            collected_data["values"].append(values)
            collected_data["log_probs"].append(log_probs)

            # Update states for the next step
            states = next_states
            safeguard += 1

        if safeguard >= 135000:
            print(f"Safeguard surpassed the 135000 boundary: {safeguard}.")

        if safeguard == self.max_nbr_iterations:
            print(f"Safeguard triggered due to reaching max_iterations ({self.max_nbr_iterations}).")

        vec_env.close()

        # Compute next values for the last state in each environment
        with torch.no_grad():
            _, _, next_values = self.policy.network(torch.FloatTensor(states).to(self.device))
            next_values = next_values.cpu().numpy()

        # Process episode data
        episode_data = []
        for i in range(num_envs):
            # First we'll trim the episode to the correct length, i.e. the moment where 'done == True' was reached.
            # After that, the episode is padded with irrelevant data until all environments have reached their 'done' state.

            # Find the first True in dones for this environment
            episode_dones = np.concatenate([np.array([step[i]]) if np.isscalar(step[i]) else step[i] for step in collected_data["dones"]])
            first_true = self.find_first_true(episode_dones)

            # Handle cases where no True is found (episode did not terminate)
            if first_true == -1:
                first_true = len(episode_dones) - 1  # Use the entire episode

            episode = {}
            for key in collected_data:
                # Extract only the relevant portion of the data for this environment
                if key in ["states", "actions"]:
                    episode[key] = np.stack([step[i] for step in collected_data[key][:first_true + 1]])
                else:
                    episode[key] = np.concatenate([np.array([step[i]]) if np.isscalar(step[i]) else step[i] for step in collected_data[key][:first_true + 1]])

            if self.normalize:
                episode["rewards"] = self.normalize_rewards(episode["rewards"])

            # Compute advantages and returns
            episode["advantages"] = self.compute_gae(
                episode["rewards"],
                episode["values"],
                episode["dones"],
                next_values[i],
                self.gamma,
                self.gae_lambda,
            )

            episode["returns"] = episode["advantages"] + episode["values"]
            episode_data.append(episode)

        return episode_data


    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
        num_envs: int = 1,  # Number of parallel environments
        callback: Optional[callable] = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Train the policy using multiple parallel environments.

        This method runs the training loop for the specified number of timesteps using parallel environments.
        It collects experience from the environments, updates the policy, logs training metrics, and saves
        model checkpoints at regular intervals.

        Args:
            stop_condition (StopCondition): Condition that defines when training should stop.
            num_envs (int): Number of parallel environments. Default is 1.
            callback (callable, optional): The callback handler for custom actions during training.
            log_interval (int): The interval at which to log training information. Default is 1.
            tb_log_name (str): The name of the tensorboard log file. Default is "OnPolicyAlgorithm".
            reset_num_timesteps (bool): Whether to reset the number of timesteps after each training run.
                Default is True.
            progress_bar (bool): Whether to display a progress bar. Default is False.

        """
        print(f"Starting training with {num_envs} parallel environments on a '{self.device}' device")

        iteration = 0
        last_save_index = 0
        training_metrics = []
        stopping_condition = stop_condition if stop_condition is not None else StopCondition()

        while True:
            self.policy.network.eval()

            # Prepare batch for training
            episode_data = []
            total_reward = 0
            total_advantage = 0

            # Collect batch data from multiple parallel environments
            batch_data = self.run_parallel_episodes(num_envs)

            for episode_result in batch_data:
                episode_data.append(episode_result)
                total_reward += np.mean(episode_result['rewards'])
                total_advantage += np.mean(episode_result['advantages'])

            # Update policy using all collected episodes
            loss = self.train(episode_data)

            # Log results
            mean_reward = total_reward / num_envs
            mean_advantage = total_advantage / num_envs

            training_metrics.append({'loss': loss, 'advantage': mean_advantage, 'reward': mean_reward})

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
                    torch.save({
                        'iteration': iteration,
                        'model_state_dict': self.policy.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, f'ppo_checkpoint_{iteration}.pt')

            iteration += 1

        # Save final checkpoint
        torch.save({
            'iteration': last_save_index if last_save_index > 0 else stopping_condition.max_time_steps,
            'model_state_dict': self.policy.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'ppo_checkpoint_{last_save_index if last_save_index > 0 else stopping_condition.max_time_steps}.pt')

        # Dump training metrics to CSV
        self.dump_metrics_to_csv("training_metrics.csv", training_metrics)

        print("Training completed!")


    @staticmethod
    def dump_metrics_to_csv(csv_filename, training_metrics):
        """
        Save training metrics to a CSV file.

        This method writes the training metrics (such as loss, advantage, reward)
        to a CSV file with headers.

        Args:
            csv_filename (str): The name of the CSV file to write.
            training_metrics (list): A list of dictionaries containing the training metrics.

        """
        with open(csv_filename, mode="w", newline="") as file:
            # Get all possible keys (assuming all dicts have the same structure)
            fieldnames = training_metrics[0].keys()

            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_metrics)
