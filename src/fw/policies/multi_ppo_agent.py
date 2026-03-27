import csv
import os
import tempfile
import zipfile

import numpy as np

from typing import Optional, Tuple

import torch
import gymnasium as gym
from torch import nn

from fw.policies.ppo_agent import PPOAgent
from fw.stop_condition import StopCondition
from fw.policies.base_agent import BaseAgent
from fw.environments.multi_py_sim_env import MultiPySimEnv


class MultiPPOAgent(BaseAgent):
    """
    Independent Proximal Policy Optimization (IPPO) wrapper for multi-agent settings.
    """

    def __init__(self, agents: list[BaseAgent], env: MultiPySimEnv):
        """
        This class manages a collection of independent PPO agents interacting within a
        shared multi-agent environment (`MultiPySimEnv`). Each agent maintains its own
        policy, value function, and optimizer, and is trained using trajectories
        collected from its respective observation and action subspaces.

        The environment is assumed to expose:
            - A concatenated global observation vector
            - A concatenated global action vector
            - Per-agent rewards and termination signals

        Observations and actions are partitioned using predefined slices corresponding
        to each sub-environment.

        Attributes:
            agents (list[BaseAgent]): List of independent PPO agents.
            env: Multi-environment encapsulating n individual environments.

        Notes:
            - This implementation corresponds to IPPO (Independent PPO), not MAPPO.
            - No parameter sharing or centralized critic is used.
            - Each agent is optimized independently using its own collected trajectories.
        """

        super().__init__(environment=env)

        self.agents = agents

        # Copy hyperparameters from the first agent (assuming all agents share same config)
        first_agent = agents[0]
        self.learning_rate = first_agent.learning_rate
        self.gamma = first_agent.gamma
        self.gae_lambda = first_agent.gae_lambda
        self.num_epochs = first_agent.num_epochs
        self.batch_size = first_agent.batch_size
        self.clip_range = first_agent.clip_range
        self.value_loss_coef = first_agent.value_loss_coef
        self.entropy_coef = first_agent.entropy_coef
        self.max_grad_norm = first_agent.max_grad_norm
        self.num_envs = first_agent.num_envs
        self.eval_frequency = first_agent.eval_frequency
        self.model_dir = first_agent.model_dir
        self.device = first_agent.device

        self.observation_slices: list[slice] = []
        self.action_slices: list[slice] = []

        # --- Build observation slices ---
        start = 0
        for sub_env in env.environments:
            dim = sub_env.observation_space.shape[0]
            end = start + dim
            self.observation_slices.append(slice(start, end))
            start = end

        # --- Build action slices ---
        start = 0
        for sub_env in env.environments:
            dim = sub_env.action_space.shape[0]
            end = start + dim
            self.action_slices.append(slice(start, end))
            start = end

    # ==========================================================
    # PREDICT
    # ==========================================================
    def predict(self, state: np.ndarray, deterministic: bool = False):
        """
        Compute actions, log-probabilities, and value estimates for all agents.

        The global state is split into per-agent observations using predefined slices.
        Each agent independently produces its action, log-probability, and value estimate.

        Args:
            state (np.ndarray): Concatenated global observation vector of shape (obs_dim,).
            deterministic (bool): If True, agents act deterministically (e.g., mean action).
                Otherwise, actions are sampled from the policy distribution.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - actions: Concatenated action vector of shape (act_dim,).
                - log_probs: Per-agent log-probabilities of shape (num_agents,).
                - values: Per-agent value estimates of shape (num_agents,).

        Notes:
            - Assumes each agent operates on a disjoint slice of the global observation.
            - Actions are flattened and concatenated in the same order as agents.
        """

        actions = []
        log_probs = []
        values = []

        for agent, obs_slice in zip(self.agents, self.observation_slices):
            sub_state = state[obs_slice]

            action, log_prob, value = agent.predict(
                state=sub_state,
                deterministic=deterministic
            )

            actions.append(np.asarray(action).reshape(-1))
            log_probs.append(float(np.asarray(log_prob)))
            values.append(float(np.asarray(value)))

        return (
            np.concatenate(actions, axis=0),
            np.array(log_probs, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )

    # ==========================================================
    # DATA COLLECTION
    # ==========================================================
    def run_parallel_episodes(self, num_envs: int) -> list:
        """
        Collect trajectories for each agent by interacting with the environment.

        This method runs a single rollout where all agents act in the shared environment
        until each agent reaches a terminal condition (terminated or truncated).
        Trajectories are stored independently per agent.

        Args:
            num_envs (int): Number of parallel environments (currently unused; kept for API compatibility).

        Returns:
            list[list[dict]]: A list of length `num_agents`, where each element is a list
            of episode dictionaries for that agent. Each episode dictionary contains:

                - "states" (np.ndarray): Shape (T, obs_dim_i)
                - "actions" (np.ndarray): Shape (T, act_dim_i)
                - "rewards" (np.ndarray): Shape (T,)
                - "values" (np.ndarray): Shape (T,)
                - "log_probs" (np.ndarray): Shape (T,)
                - "advantages" (np.ndarray): Shape (T,)
                - "returns" (np.ndarray): Shape (T,)

        Notes:
            - Per-agent termination is handled independently.
            - Final transitions (including terminal states) are preserved.
            - Generalized Advantage Estimation (GAE) is computed per agent.
            - Bootstrapping uses the value function unless the trajectory ends in a terminal state.
            - A safeguard limit prevents infinite loops in non-terminating environments.
        """

        states, _ = self.env.reset()

        # Per-agent buffers
        collected_data = [
            {
                "states": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "values": [],
                "log_probs": [],
                "next_states": [],
            }
            for _ in self.agents
        ]

        # Track per-agent done
        agent_dones = np.zeros(len(self.agents), dtype=bool)

        safeguard = 0

        while not np.all(agent_dones) and safeguard < 150_000:
            actions, log_probs, values = self.predict(states)

            next_states, rewards, terminated, truncated, _ = self.env.step(actions)

            # IMPORTANT: reconstruct per-agent dones
            # Since env returns global flags, we infer per-agent:
            # assume env internally stepped all, so we track manually
            step_dones = np.logical_or(terminated, truncated)

            for i, (obs_slice, act_slice) in enumerate(zip(self.observation_slices, self.action_slices)):
                if agent_dones[i]:
                    continue  # stop collecting for finished agents

                collected_data[i]["states"].append(states[obs_slice])
                collected_data[i]["actions"].append(actions[act_slice])
                collected_data[i]["rewards"].append(rewards[i])
                collected_data[i]["dones"].append(step_dones[i])
                collected_data[i]["values"].append(values[i])
                collected_data[i]["log_probs"].append(log_probs[i])
                collected_data[i]["next_states"].append(next_states[obs_slice])

            # Update per-agent done flags
            agent_dones = np.logical_or(agent_dones, step_dones)

            states = next_states
            safeguard += 1

        if safeguard >= 150_000:
            print("Safeguard triggered (max steps reached)")

        # ======================================================
        # BUILD EPISODES PER AGENT
        # ======================================================
        episode_data = []

        episodes_by_agent = [[] for _ in range(len(self.agents))]

        for i, agent_data in enumerate(collected_data):
            if not agent_data["states"]:
                continue

            rewards = np.array(agent_data["rewards"], dtype=np.float32)
            values = np.array(agent_data["values"], dtype=np.float32)
            dones = np.array(agent_data["dones"], dtype=np.float32)
            next_value = 0.0

            # Bootstrap
            if dones[-1]:
                next_value = 0.0
            else:
                with torch.no_grad():
                    last_state = torch.FloatTensor(agent_data["next_states"][-1]).unsqueeze(0).to(self.device)
                    # Fix: Ensure last_state is on correct device
                    last_state = last_state.to(self.agents[i].device)
                    _, _, v = self.agents[i].policy.network(last_state)
                    next_value = v.item()

            advantages = PPOAgent.compute_gae(
                rewards,
                values,
                dones,
                next_value,
                self.gamma,
                self.gae_lambda,
            )

            episode = {
                "states": np.array(agent_data["states"], dtype=np.float32),
                "actions": np.array(agent_data["actions"], dtype=np.float32),
                "rewards": rewards,
                "values": values,
                "log_probs": np.array(agent_data["log_probs"], dtype=np.float32),
                "advantages": advantages,
                "returns": advantages + values,
            }

            episodes_by_agent[i].append(episode)

        return episodes_by_agent

    # ==========================================================
    # TRAIN PER AGENT
    # ==========================================================
    def train(self, episode_data, index) -> float:
        """
        Perform PPO updates for a single agent using collected trajectories.

        This method applies multiple epochs of minibatch stochastic gradient descent
        using the PPO clipped surrogate objective.

        Args:
            episode_data (list[dict]): List of episode dictionaries for the agent.
                Each dictionary must contain:
                    - "states", "actions", "log_probs", "advantages", "returns"
            index (int): Index of the agent to train.

        Returns:
            float: Mean training loss across all minibatches and epochs. Returns NaN
            if insufficient data is available.

        Notes:
            - Advantages are normalized before training.
            - Supports continuous action spaces using a Gaussian policy.
            - Uses clipped policy objective, value loss, and entropy regularization.
            - Gradient clipping is applied to stabilize training.
        """

        if not episode_data:
            return float('nan')

        agent = self.agents[index]
        agent.policy.network.train()

        # Flatten batch
        all_states = torch.as_tensor(np.concatenate([ep['states'] for ep in episode_data]), dtype=torch.float32).to(self.device)
        all_actions = torch.as_tensor(np.concatenate([ep['actions'] for ep in episode_data]), dtype=torch.float32).to(self.device)
        all_advantages = torch.as_tensor(np.concatenate([ep['advantages'] for ep in episode_data]), dtype=torch.float32).to(self.device)
        all_old_log_probs = torch.as_tensor(np.concatenate([ep['log_probs'] for ep in episode_data]), dtype=torch.float32).to(self.device)
        all_returns = torch.as_tensor(np.concatenate([ep['returns'] for ep in episode_data]), dtype=torch.float32).to(self.device)

        if all_advantages.numel() <= 1:
            return float('nan')

        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        batch_size = all_states.shape[0]
        losses = []

        for _ in range(self.num_epochs):
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, self.batch_size):
                mb_idx = indices[start:start + self.batch_size]

                mb_states = all_states[mb_idx]
                mb_actions = all_actions[mb_idx]
                mb_old_log_probs = all_old_log_probs[mb_idx]
                mb_returns = all_returns[mb_idx]
                mb_advantages = all_advantages[mb_idx]

                action_mean, action_std, state_values = agent.policy.network(mb_states)
                dist = torch.distributions.Normal(action_mean, action_std)

                if isinstance(agent.action_space, gym.spaces.Box):
                    raw_actions = torch.atanh(mb_actions.clamp(-0.999, 0.999))
                else:
                    raw_actions = mb_actions

                log_probs = agent.policy.calc_log_probs(dist, raw_actions, mb_actions)

                ratios = torch.exp(log_probs - mb_old_log_probs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(state_values.squeeze(-1), mb_returns)
                entropy_loss = dist.entropy().sum(dim=-1).mean()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.policy.network.parameters(), self.max_grad_norm)
                agent.optimizer.step()

                losses.append(loss.item())

        return float(np.mean(losses))

    # ==========================================================
    # LEARNING LOOP
    # ==========================================================
    def learn(self, stop_condition: Optional[StopCondition] = None):
        """
        Run the main training loop for all agents.

        This method repeatedly:
            1. Collects trajectories from the environment
            2. Trains each agent independently
            3. Evaluates stopping conditions

        Args:
            stop_condition (Optional[StopCondition]): Stopping criterion controlling
                training duration. If None, a default condition is used.

        Notes:
            - Each agent is trained only if it has collected data.
            - Logging occurs at intervals defined by `eval_frequency`.
            - Early stopping is triggered if the provided condition is met.
        """

        stopping_condition = stop_condition if stop_condition else StopCondition()

        print(f"Training {len(self.agents)} independent PPO agents")

        for iteration in range(stopping_condition.max_time_steps):

            episodes_by_agent = self.run_parallel_episodes(self.num_envs)

            for i in range(len(self.agents)):
                if episodes_by_agent[i]:  # Only train if we have data
                    loss = self.train(episodes_by_agent[i], i)

                    # Calculate mean reward for this agent across all episodes
                    mean_reward = np.mean([ep['rewards'].mean() for ep in episodes_by_agent[i]])

                    if iteration % self.eval_frequency == 0:
                        print(f"Iter {iteration} | Agent {i} | Reward: {mean_reward:.3f} | Loss: {loss:.3f}")

                    should_stop, reason = stopping_condition.should_stop(
                        current_step=iteration,
                        mean_reward=mean_reward,
                        mean_loss=loss
                    )

                    if should_stop:
                        print(f"Stopping early: {reason}")
                        return

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
        Load a multi-agent policy from a ZIP archive.

        The archive must contain one policy file per agent.

        Args:
            policy_file_name (str): Path to the policy archive.

        Raises:
            FileNotFoundError: If the archive does not exist.
            RuntimeError: If loading fails.
        """

        if not policy_file_name.endswith(".zip"):
            policy_file_name += ".zip"

        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        if not os.path.exists(policy_file_name):
            raise FileNotFoundError(f"Policy archive not found: {policy_file_name}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # 1. Extract the main archive
                with zipfile.ZipFile(policy_file_name, 'r') as main_zip:
                    main_zip.extractall(tmp_dir)

                # 2. Load each agent
                for i, agent in enumerate(self.agents):
                    # We pass the path WITHOUT .zip because PPOPolicy.load
                    # likely adds it internally, just like .save did.
                    agent_base_path = os.path.join(tmp_dir, f"agent_{i}")

                    # If your load() method DOES NOT append .zip automatically,
                    # change agent_base_path to f"agent_{i}.zip"
                    agent.policy = agent.policy.load(agent_base_path, self.device)

            except Exception as e:
                raise RuntimeError(f"Failed to load multi-agent policy: {e}") from e


    def save_policy(self, policy_file_name: str) -> None:
        """
        Save all agent policies into a single ZIP archive.

        Args:
            policy_file_name (str): Output file name.

        Raises:
            RuntimeError: If saving fails.
            FileNotFoundError: If an expected agent policy file is missing.
        """

        # Ensure the parent file has the .zip extension
        if not policy_file_name.endswith(".zip"):
            policy_file_name += ".zip"

        if self.model_dir:
            full_path = os.path.join(self.model_dir, policy_file_name)
        else:
            full_path = policy_file_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # 1. Have each agent save its individual ZIP into the temp directory
                for i, agent in enumerate(self.agents):
                    agent_base_path = os.path.join(tmp_dir, f"agent_{i}")
                    agent.policy.save(agent_base_path)

                # 2. Bundle the resulting .zip files into the final parent ZIP
                with zipfile.ZipFile(full_path, 'w') as main_zip:
                    for i in range(len(self.agents)):
                        inner_zip_name = f"agent_{i}.zip"
                        inner_zip_path = os.path.join(tmp_dir, inner_zip_name)

                        if os.path.exists(inner_zip_path):
                            main_zip.write(inner_zip_path, arcname=inner_zip_name)
                        else:
                            raise FileNotFoundError(f"Expected inner policy {inner_zip_name} was not created.")

            except Exception as e:
                raise RuntimeError(f"Failed to save multi-agent policy to '{full_path}'.") from e


    def set_policy_eval(self):
        """
        Set all agent policy networks to evaluation mode.

        This disables training-specific behaviors such as dropout and ensures
        deterministic inference where applicable.
        """

        for agent in self.agents:
            agent.policy.network.eval()
