# lstm_model.py
import os
import time
import copy
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from fw.policies.memory_buffer import MemoryBuffer
from fw.policies.lstm_policy import LstmPolicy, get_device
from fw.stop_condition import StopCondition
from fw.policies.base_model import BaseModel

# --- Hyperparameters (tweakable) ---
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.95
EPISODES = 200
MAX_STEPS_PER_EPISODE = 65000
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 2
FC_HIDDEN_SIZE = 256
DROPOUT = 0.1
LOG_INTERVAL = 10
SAVE_INTERVAL = 5
BATCH_SIZE = 16
UPDATE_FREQUENCY = 10
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
PPO_EPS = 0.2
PPO_EPOCHS = 4
MINI_BATCH = 4


class LstmModel(BaseModel):
    """
    Recurrent PPO-like training harness using episodic buffer and LSTM policy.
    """

    def __init__(self, *args, max_nbr_iterations: int = 200000, normalize: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        # Only continuous actions supported here
        if isinstance(self.action_space, gym.spaces.Discrete):
            raise NotImplementedError("This recurrent implementation supports continuous actions only.")
        else:
            self.output_dim = int(self.action_space.shape[0])

        self.input_dim = int(self.observation_space.shape[0])
        self.device = get_device("auto")
        self.policy = LstmPolicy(self.input_dim, self.output_dim, device=self.device.type)
        # Memory holds episodes of variable length
        self.memory = MemoryBuffer(capacity=5000)
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=EPISODES, eta_min=1e-6)
        self.normalize = normalize
        self.max_nbr_iterations = max_nbr_iterations

        # Running reward stats
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4

    @staticmethod
    def compute_gae(
            rewards: torch.Tensor,
            values: torch.Tensor,
            masks: torch.Tensor,
            gamma: float = 0.99,
            tau: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Tensor of rewards (batch_size, seq_len)
            values: Tensor of value estimates (batch_size, seq_len)
            masks: Tensor of episode masks (batch_size, seq_len)
            gamma: Discount factor
            tau: GAE parameter

        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        batch_size, seq_len = rewards.shape

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + gamma * next_value * masks[:, t] - values[:, t]
            gae = delta + gamma * tau * masks[:, t] * gae
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]

        return returns, advantages

    def predict(self, state: np.ndarray, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, deterministic: bool = False):
        actions, log_probs, values, new_hidden = self.policy.predict(state, hidden, deterministic)
        # Return in format used previously: action vector (1d), scalar log_prob, and new_hidden
        if isinstance(actions, np.ndarray) and actions.ndim > 1 and actions.shape[0] == 1:
            action_out = actions[0]
        else:
            action_out = actions
        # log_probs may be array or scalar
        if isinstance(log_probs, np.ndarray) and log_probs.size > 0:
            lp = float(np.asarray(log_probs).flat[0])
        else:
            lp = float(log_probs)
        return action_out, lp, new_hidden

    def train(self) -> float:
        """
        Single training pass over samples from the episodic buffer.
        """
        self.policy.network.train()
        # Sample batch from memory
        batch = self.memory.sample(BATCH_SIZE)

        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_rewards = batch['rewards'].to(self.device)
        masks = batch['masks'].to(self.device)

        batch_size, seq_len = old_rewards.shape

        # Compute returns and advantages using GAE
        with torch.no_grad():
            # Get new value estimates - ensure proper dimensions
            hidden = self.policy.network.init_hidden(batch_size, self.device)

            # Forward pass through the network
            action_mean, action_std, new_values, _ = self.policy.network(observations, hidden)

            # Remove the last dimension if present and ensure correct shape
            if new_values.dim() == 3:
                new_values = new_values.squeeze(-1)  # (batch_size, seq_len, 1) -> (batch_size, seq_len)

            # Ensure new_values has the same shape as old_rewards
            if new_values.shape != old_rewards.shape:
                print(f"Shape mismatch: new_values {new_values.shape}, old_rewards {old_rewards.shape}")
                # Use the minimum sequence length
                min_seq_len = min(new_values.shape[1], old_rewards.shape[1])
                new_values_trunc = new_values[:, :min_seq_len]
                old_rewards_trunc = old_rewards[:, :min_seq_len]
                masks_trunc = masks[:, :min_seq_len]
            else:
                new_values_trunc = new_values
                old_rewards_trunc = old_rewards
                masks_trunc = masks

            returns, advantages = self.compute_gae(old_rewards_trunc, new_values_trunc, masks_trunc, GAMMA, TAU)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy update
        actor_loss_total = 0
        critic_loss_total = 0
        entropy_loss_total = 0

        # Reset hidden state for forward pass
        hidden = self.policy.network.init_hidden(batch_size, self.device)

        # Forward pass through agent
        action_mean, action_std, pred_values, _ = self.policy.network(observations, hidden)

        # Remove the last dimension if present and ensure correct shape
        if pred_values.dim() == 3:
            pred_values = pred_values.squeeze(-1)  # (batch_size, seq_len, 1) -> (batch_size, seq_len)

        # Ensure pred_values has correct shape
        if pred_values.shape != returns.shape:
            print(f"Shape mismatch in policy forward: pred_values {pred_values.shape}, returns {returns.shape}")
            # Truncate to match sequence length
            min_seq_len = min(pred_values.shape[1], returns.shape[1])
            pred_values_trunc = pred_values[:, :min_seq_len]
            returns_trunc = returns[:, :min_seq_len]
            advantages_trunc = advantages[:, :min_seq_len]
            actions_trunc = actions[:, :min_seq_len]
            action_mean_trunc = action_mean[:, :min_seq_len]
            action_std_trunc = action_std[:, :min_seq_len]
        else:
            pred_values_trunc = pred_values
            returns_trunc = returns
            advantages_trunc = advantages
            actions_trunc = actions
            action_mean_trunc = action_mean
            action_std_trunc = action_std

        # Create action distribution
        action_dist = torch.distributions.Normal(action_mean_trunc, action_std_trunc)

        # Compute log probabilities of taken actions
        log_probs = action_dist.log_prob(actions_trunc).sum(dim=-1)

        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages_trunc.detach()).mean()

        # Critic loss (value function)
        critic_loss = F.mse_loss(pred_values_trunc, returns_trunc.detach())

        # Entropy loss for exploration
        entropy = action_dist.entropy().sum(dim=-1).mean()
        entropy_loss = -ENTROPY_COEF * entropy

        # Total loss
        total_loss = actor_loss + VALUE_LOSS_COEF * critic_loss + entropy_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(), MAX_GRAD_NORM)

        self.optimizer.step()

        # Log losses for debugging
        actor_loss_total += actor_loss.item()
        critic_loss_total += critic_loss.item()
        entropy_loss_total += entropy_loss.item()

        return total_loss.item()

    def make_env(self):
        def _init():
            return Monitor(self.create_env())

        return _init

    def create_env(self) -> gym.Env:
        env = copy.deepcopy(self.env)
        if hasattr(env, "randomize"):
            env.randomize()
        return env

    @staticmethod
    def find_first_true(bool_array: np.ndarray) -> int:
        true_indices = np.where(bool_array)[0]
        return int(true_indices[0]) if true_indices.size > 0 else -1

    def learn(self, stop_condition: Optional[StopCondition] = None) -> None:
        # Tracking metrics
        episode_lengths = deque(maxlen=100)
        success_rate = deque(maxlen=100)

        # Create save directory
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        print("Starting training...")

        for episode in range(EPISODES):
            episode_start_time = time.time()

            # Reset environment and agent
            obs, _ = self.env.reset()
            hidden = self.policy.network.init_hidden(1, self.device)

            # Episode data collection
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_log_probs = []

            total_reward = 0
            step_count = 0
            reward = 0

            for step in range(MAX_STEPS_PER_EPISODE):
                action_np, log_prob, value, new_hidden = self.policy.predict(obs, hidden)

                # Store episode data - ensure proper dimensions
                episode_obs.append(obs.flatten())
                episode_actions.append(action_np.flatten())
                episode_values.append(float(value))
                episode_log_probs.append(float(log_prob))

                # Take environment step
                next_obs, reward, done, truncated, _ = self.env.step(
                    action_np[0] if hasattr(action_np, '__len__') and len(action_np) > 0 else action_np)

                episode_rewards.append(float(reward))
                total_reward += reward
                step_count += 1

                # Update for next iteration
                obs = next_obs
                hidden = new_hidden

                if done:
                    break

            # Calculate episode statistics
            episode_length = step_count
            is_success = self.env._has_reached_target() if hasattr(self.env, '_has_reached_target') else (reward > 0)

            # Convert to numpy arrays with proper shapes
            episode_data = {
                'observations': np.array(episode_obs),
                'actions': np.array(episode_actions),
                'rewards': np.array(episode_rewards),
                'values': np.array(episode_values),
                'log_probs': np.array(episode_log_probs)
            }
            self.memory.push(episode_data)

            # Update tracking metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            success_rate.append(1.0 if is_success else 0.0)

            # Perform learning update
            if (episode + 1) % UPDATE_FREQUENCY == 0 and len(self.memory) >= BATCH_SIZE:
                self.train()

            # Logging and saving
            if (episode + 1) % LOG_INTERVAL == 0:
                avg_reward = np.mean(episode_rewards)
                avg_length = np.mean(episode_lengths)
                avg_success = np.mean(success_rate)
                episode_time = time.time() - episode_start_time

                print(
                    f'Episode {episode + 1}/{EPISODES} | '
                    f'Avg Reward: {avg_reward:.2f} | '
                    f'Avg Length: {avg_length:.1f} | '
                    f'Success Rate: {avg_success:.3f} | '
                    f'Episode Time: {episode_time:.2f}s | '
                    f'LR: {self.scheduler.get_last_lr()[0]:.6f}'
                )

            if (episode + 1) % SAVE_INTERVAL == 0:
                pass
                # save_checkpoint(agent, optimizer, episode, save_dir)

            # Update learning rate
            self.scheduler.step()

        # Final save
        # save_checkpoint(agent, optimizer, EPISODES - 1, save_dir, final=True)
        print("Training completed!")

    def dump_metrics_to_csv(self, csv_filename: str, training_metrics: list) -> None:
        if not training_metrics:
            print("Warning: No training metrics to write.")
            return
        filepath = os.path.join(self.model_dir, csv_filename)
        import csv

        with open(filepath, mode="w", newline="") as file:
            fieldnames = training_metrics[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_metrics)

    def load_policy(self, policy_file_name: str) -> None:
        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)
        self.policy = self.policy.load(policy_file_name, self.device.type)
        print(f"Loaded policy {policy_file_name}")

    def save_policy(self, policy_file_name: str) -> None:
        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)
        self.policy.save(policy_file_name)
        print(f"Saved policy {policy_file_name}")

    def set_policy_eval(self) -> None:
        self.policy.network.eval()
        print("Policy set to evaluation mode.")
