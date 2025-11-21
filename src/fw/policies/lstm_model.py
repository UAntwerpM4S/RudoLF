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
from fw.policies.ppo_model import PPOModel
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


class LstmModel(PPOModel):
    """
    Recurrent PPO-like training harness using episodic buffer and LSTM policy.
    """

    def __init__(self, *args, lstm_hidden_size: int = 128, **kwargs):
        # Force single environment for Phase 1
        if 'num_envs' in kwargs and kwargs['num_envs'] != 1:
            print("Warning: Phase1LSTMPPOModel only supports num_envs=1. Forcing to 1.")
        kwargs['num_envs'] = 1

        super().__init__(*args, **kwargs)

        self.lstm_hidden_size = lstm_hidden_size

        # Replace with LSTM policy
        self.policy = LstmPolicy(
            self.input_dim,
            self.output_dim,
            self.device.type,
            lstm_hidden_size
        )
        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=self.learning_rate)

    def run_parallel_episodes(self, num_envs: int):
        """Phase 1: Only support single environment"""
        if num_envs != 1:
            raise ValueError("Phase1LSTMPPOModel only supports single environment. Use num_envs=1")

        # Single episode collection using LSTM
        env = self.create_env()
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Handle gymnasium tuple return

        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []

        for step in range(self.max_nbr_iterations):
            # Collect current state
            states.append(state)

            # Get action from policy (will handle LSTM internally)
            # For now, we'll use single-step prediction to test
            action, log_prob, value = self.policy.predict(state)

            # Step environment
            next_state, reward, done, truncated, info = env.step(action[0])
            done = done or truncated

            # Store data
            actions.append(action[0])
            rewards.append(reward)
            dones.append(done)
            values.append(value[0])
            log_probs.append(log_prob[0])

            if done:
                break

            state = next_state

        env.close()

        # Convert to numpy arrays
        episode_data = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values),
            'log_probs': np.array(log_probs)
        }

        # Compute advantages and returns (using your existing method)
        next_value = 0.0  # Terminal state
        episode_data['advantages'] = self.compute_gae(
            episode_data['rewards'],
            episode_data['values'],
            episode_data['dones'],
            next_value,
            self.gamma,
            self.gae_lambda
        )
        episode_data['returns'] = episode_data['advantages'] + episode_data['values']

        return [episode_data]  # Return as list for compatibility

    def train(self, all_episodes_data) -> float:
        """LSTM-compatible training that uses the original PPO logic"""
        if not all_episodes_data:
            return float('nan')

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

                # 🛠️ FIX: Handle LSTM output - ignore hidden state during training
                # For Phase 1, we treat each state independently (like original PPO)
                action_mean, action_std, state_values, _ = self.policy.network(
                    mb_states,
                    None  # No hidden state for independent transitions
                )

                # The rest is exactly the same as original PPO
                dist = torch.distributions.Normal(action_mean, action_std)

                # Adjust actions if action space is continuous
                if isinstance(self.action_space, gym.spaces.Box):
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
                torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                loss_per_epoch.append(loss.detach().cpu().item())

        # Return mean loss over all updates
        return float(np.mean(loss_per_epoch))

    # def train(self, all_episodes_data) -> float:
    #     """Proper LSTM training that maintains temporal dependencies"""
    #     if not all_episodes_data:
    #         return float('nan')
    #
    #     self.policy.network.train()
    #
    #     loss_per_epoch = []
    #
    #     # Train over multiple epochs
    #     for _ in range(self.num_epochs):
    #         # Shuffle episodes for each epoch
    #         episode_indices = torch.randperm(len(all_episodes_data))
    #
    #         for ep_idx in episode_indices:
    #             episode = all_episodes_data[ep_idx]
    #
    #             # Process each episode as a sequence
    #             states = torch.as_tensor(episode['states'], dtype=torch.float32).to(self.device)
    #             actions = torch.as_tensor(episode['actions'], dtype=torch.float32).to(self.device)
    #             advantages = torch.as_tensor(episode['advantages'], dtype=torch.float32).to(self.device)
    #             old_log_probs = torch.as_tensor(episode['log_probs'], dtype=torch.float32).to(self.device)
    #             returns = torch.as_tensor(episode['returns'], dtype=torch.float32).to(self.device)
    #
    #             if len(states) <= 1:
    #                 continue
    #
    #             # Normalize advantages for this episode
    #             ep_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #
    #             # Process the entire episode sequence through LSTM
    #             # This maintains proper temporal dependencies
    #             action_mean, action_std, state_values, _ = self.policy.network(
    #                 states.unsqueeze(1),  # Add sequence dimension: (seq_len, 1, features)
    #                 None  # Let LSTM start with initial hidden state
    #             )
    #
    #             # Calculate losses (same as original PPO)
    #             dist = torch.distributions.Normal(action_mean, action_std)
    #
    #             if isinstance(self.action_space, gym.spaces.Box):
    #                 raw_actions = torch.atanh(actions.clamp(-0.99, 0.99))
    #             else:
    #                 raw_actions = actions
    #
    #             log_probs = self.policy.calc_log_probs(dist, raw_actions, actions)
    #
    #             # PPO clipped loss
    #             ratios = torch.exp(log_probs - old_log_probs)
    #             surrogate_1 = ratios * ep_advantages
    #             surrogate_2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * ep_advantages
    #             policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
    #
    #             # Value loss
    #             value_loss = torch.nn.functional.mse_loss(state_values.squeeze(-1), returns)
    #             entropy = dist.entropy().sum(dim=-1).mean()
    #
    #             # Total loss
    #             loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
    #
    #             # Optimize
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(), self.max_grad_norm)
    #             self.optimizer.step()
    #
    #             loss_per_epoch.append(loss.detach().cpu().item())
    #
    #     return float(np.mean(loss_per_epoch)) if loss_per_epoch else float('nan')
