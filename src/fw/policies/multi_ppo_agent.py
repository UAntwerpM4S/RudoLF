import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

from typing import Optional, Tuple
from fw.stop_condition import StopCondition
from fw.policies.base_agent import BaseAgent
from fw.policies.ppo_policy import PPOPolicy


class MultiPPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent for reinforcement learning.

    This class handles the training, evaluation, and policy management for PPO-based reinforcement learning agents.
    """

    MINIMUM_ITERATIONS_FOR_BEST_SAVE = 20

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

        self._policies = []
        self.optimizers = []

        for env in self.env.environments:
            input_dim = env.observation_space.shape[0]

            if isinstance(env.action_space, gym.spaces.Discrete):
                output_dim = env.action_space.n
            else:
                output_dim = env.action_space.shape[0]

            policy = PPOPolicy(input_dim, output_dim, self.device)
            self._policies.append(policy)

            self.optimizers.append(
                optim.Adam(policy.network.parameters(), lr=self.learning_rate)
            )
        self.num_policies = len(self._policies)
        self.max_nbr_iterations = max_nbr_iterations
        self.normalize = normalize
        self._vec_env = None


    @property
    def policies(self):
        """Policy.
        """

        return self._policies


    def predict(self, state: np.ndarray, deterministic: bool = False, policy_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict an action for a given state using the policy.

        Args:
            state (np.ndarray): Current state as a NumPy array.
            policy_id (int): Policy index
            deterministic (boolean): Whether the action should be deterministic or not.
            policy_id (int, optional): The id of the policy to get a prediction from.

        Returns:
            Tuple: A tuple containing the predicted action and log probability.
        """

        actions, log_probs, values = self.policies[policy_id].predict(state, deterministic)
        return actions, log_probs, values


    def train(self, all_episodes_data, policy_id) -> float:
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
            policy_id: The index of the policy that will be trained on.

        Returns:
            float: Average loss value after training.
        """

        if not all_episodes_data:
            return float('nan')

        policy = self.policies[policy_id]
        optimizer = self.optimizers[policy_id]
        policy.network.train()

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
                action_mean, action_std, state_values = policy.network(mb_states)
                dist = torch.distributions.Normal(action_mean, action_std)

                # Adjust actions if action space is continuous
                if isinstance(self.action_space, gym.spaces.Box):
                    raw_actions = torch.atanh(mb_actions.clamp(-0.999, 0.999))
                else:
                    raw_actions = mb_actions
                log_probs = policy.calc_log_probs(dist, raw_actions, mb_actions)

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
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.network.parameters(), self.max_grad_norm)
                optimizer.step()

                loss_per_epoch.append(loss.item())

        # Return mean loss over all updates
        return np.mean(loss_per_epoch)


    def run_parallel_episodes(self) -> list:
        """
        Run multiple episodes in parallel using Gym vectorized environments.

        Returns:
            list: A list of dictionaries containing episode data for each environment.
        """

        num_envs = len(self.env.environments)

        states = []
        dones = np.zeros(num_envs, dtype=bool)

        # Reset all envs first
        for env in self.env.environments:
            state, _ = env.reset()
            states.append(state)

        # Per-env storage
        buffers = [
            {k: [] for k in ["states", "actions", "rewards", "dones", "values", "log_probs"]}
            for _ in range(num_envs)
        ]

        safeguard = 0

        while not np.all(dones) and safeguard < self.max_nbr_iterations:

            actions = []
            log_probs = []
            values = []

            for i in range(num_envs):
                if dones[i]:
                    actions.append(None)
                    log_probs.append(None)
                    values.append(None)
                    continue

                action, log_prob, value = self.policies[i].predict(states[i], deterministic=False)

                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                value = value.squeeze(0)

                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

            next_states = [None] * num_envs

            for i in range(num_envs):
                if dones[i]:
                    next_states[i] = states[i]
                    continue

                next_state, reward, terminated, truncated, _ = self.env.environments[i].step(actions[i])
                done = terminated or truncated

                # store transition
                buffers[i]["states"].append(states[i])
                buffers[i]["actions"].append(actions[i])
                buffers[i]["rewards"].append(reward)
                buffers[i]["dones"].append(done)
                buffers[i]["values"].append(values[i])
                buffers[i]["log_probs"].append(log_probs[i])

                next_states[i] = next_state
                dones[i] = done

            states = next_states
            safeguard += 1

        episode_data = []

        for i in range(num_envs):
            data = buffers[i]
            next_value = 0.0

            # convert to arrays
            for key in data:
                data[key] = np.array(data[key])

            if self.normalize:
                data["rewards"] = self.normalize_rewards(data["rewards"])

            # compute next value
            last_state = states[i]
            if last_state is None:
                next_value = 0.0
            else:
                with torch.no_grad():
                    _, _, next_value = self.policies[i].network(
                        torch.FloatTensor(last_state).to(self.device)
                    )
                    next_value = next_value.cpu().numpy()

            data["advantages"] = self.compute_gae(
                data["rewards"],
                data["values"],
                data["dones"],
                next_value,
                self.gamma,
                self.gae_lambda,
            )

            data["returns"] = data["advantages"] + data["values"]

            episode_data.append(data)

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
            self.set_policy_eval()

            # Collect batch data from multiple parallel environments
            batch_data = self.run_parallel_episodes()

            for policy_id in range(self.num_policies):
                episode = batch_data[policy_id]
                mean_reward = np.mean(episode['rewards'])
                mean_advantage = np.mean(episode['advantages'])

                loss = self.train([episode], policy_id)

                training_metrics.append({'loss': loss, 'advantage': mean_advantage, 'reward': mean_reward})

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
                    if iteration > 0 and iteration % self.eval_frequency == 0:
                        self.save_policy(f"ppo_checkpoint_{iteration}")

        # Save final checkpoint
        self.save_policy(f'ppo_checkpoint_{last_save_index if last_save_index > 0 else stopping_condition.max_time_steps}')

        # Dump training metrics to CSV
        self.dump_metrics_to_csv(self.model_dir, "training_metrics.csv", training_metrics)

        if getattr(self, "_vec_env", None) is not None:
            self._vec_env.close()
            self._vec_env = None

        print("Training completed!")


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

            self.policies[0] = self.policies[0].load(policy_file_name, self.device)
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

            self.policies[0].save(policy_file_name)
        except Exception as e:
            raise RuntimeError(f"Failed to save the policy '{policy_file_name}'.") from e


    def set_policy_eval(self):
        """
        Set policy in evaluation mode.
        """

        for policy in self.policies:
            policy.network.eval()
