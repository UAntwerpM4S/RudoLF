import os
import torch
import zipfile
import tempfile
import numpy as np

from typing import Optional, Tuple, Union
from fw.stop_condition import StopCondition
from fw.policies.ppo_policy import PPOPolicy
from fw.policies.base_agent import BaseAgent
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.common.tools.logging_utils import dump_multi_agent_metrics
from fw.environments.multi_sim_env import MultiSimEnv


class MultiPPOAgent(BaseAgent):
    """
    Multi-agent Proximal Policy Optimization (PPO) trainer operating over a
    tuple-structured multi-environment.

    This class coordinates multiple independent `SinglePPOAgent` instances,
    each controlling one sub-environment within a `MultiSimEnv`. Experience
    is collected in parallel across all environments, after which each agent is
    trained independently on its own trajectory.

    The interaction model is *synchronous but factorized*:
    - Environments are stepped together using tuple-based actions.
    - Each agent receives only its corresponding observation.
    - No parameter sharing or centralized critic is used.

    Attributes:
        agents (list[SinglePPOAgent]): List of independent PPO agents, one per environment.
        num_envs (int): Number of parallel environments/agents.
        max_nbr_iterations (int): Maximum steps per rollout to prevent infinite loops.
        normalize (bool): Whether to normalize rewards per episode.
    """

    MINIMUM_ITERATIONS_FOR_BEST_SAVE = 20

    def __init__(
        self,
        *args,
        agents: list[SinglePPOAgent],
        max_nbr_iterations: int = 200000,
        normalize: bool = False,
        **kwargs
    ):
        """
        Initialize the multi-agent PPO trainer.

        Args:
            *args: Positional arguments forwarded to `BaseAgent`.
            agents (list[SinglePPOAgent]): List of initialized PPO agents. The
                length must match the number of sub-environments in `self.env`.
            max_nbr_iterations (int, optional): Maximum number of environment
                steps per rollout before forced termination. Defaults to 200000.
            normalize (bool, optional): If True, apply reward normalization per
                episode before advantage computation. Defaults to False.
            **kwargs: Keyword arguments forwarded to `BaseAgent`.

        Raises:
            ValueError: If the number of agents does not match the number of environments.
        """

        super().__init__(*args, **kwargs)

        self.agents = agents
        self.num_envs = len(self.env.sub_environments)

        if not isinstance(self.env, MultiSimEnv):
            raise TypeError("MultiPPOAgent requires MultiSimEnv")

        if len(agents) != self.num_envs:
            raise ValueError(
                f"Number of agents ({len(agents)}) must match "
                f"number of sub-environments ({self.num_envs})"
            )

        self.max_nbr_iterations = max_nbr_iterations
        self.normalize = normalize

    def predict(
            self,
            state: Union[np.ndarray, Tuple[np.ndarray, ...]],
            deterministic: bool = False
    ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], np.ndarray, np.ndarray]:
        """
        Compute actions, log-probabilities, and value estimates for all agents.

        Each agent receives its corresponding observation and produces an action
        independently. The outputs are aggregated into tuple/array structures.

        Args:
            state (tuple[np.ndarray, ...]): Tuple of observations, one per environment.
            deterministic (bool, optional): If True, use deterministic policy actions.
                Otherwise, sample from the action distribution.

        Returns:
            tuple:
                - actions (tuple): Tuple of actions, one per agent/environment.
                - log_probs (np.ndarray): Array of shape (num_envs,) containing
                  log-probabilities of the sampled actions.
                - values (np.ndarray): Array of shape (num_envs,) containing
                  state-value estimates.

        Notes:
            The structure of `state` must match `env.observation_space`.
        """

        actions = []
        log_probs = []
        values = []

        for i, agent in enumerate(self.agents):
            action, log_prob, value = agent.predict(
                state[i],
                deterministic=deterministic
            )

            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

        return (
            tuple(actions),
            np.stack(log_probs),
            np.stack(values),
        )


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

        # Reset all environments
        states, _ = self.env.reset()
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
            actions, log_probs, values = self.predict(states, deterministic=False)

            # Step all environments
            next_states, next_rewards, next_terminations, truncations, infos = self.env.step(actions)

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
                with torch.no_grad():
                    last_state = torch.FloatTensor(trajectory["next_states"][first_done]).unsqueeze(0).to(self.device)
                    _, _, next_values = self.agents[env_id].policy.network(last_state)
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
        Train all agents using parallel environment rollouts.

        The training loop alternates between:
        1. Collecting trajectories from all environments.
        2. Updating each agent independently using its own trajectory.

        Training continues until the provided stopping condition is met or the
        maximum number of iterations is reached.

        Args:
            stop_condition (Optional[StopCondition]): Stopping criterion based on
                iteration count, reward, or loss. If None, a default condition is used.

        Side Effects:
            - Updates agent policies in-place.
            - Saves best-performing and periodic checkpoint policies to disk.
            - Logs training metrics to CSV.

        Notes:
            - Policy evaluation mode is enabled during rollout collection.
            - Best model saving is delayed until a minimum number of iterations
              (`MINIMUM_ITERATIONS_FOR_BEST_SAVE`) is reached.
            - Early stopping is evaluated per agent using its own metrics.
        """

        print(f"Starting training with {self.num_envs} parallel environments on a '{self.device}' device")

        last_save_index = 0
        best_rewards = [float('-inf') for _ in range(len(self.agents))]
        training_metrics = {f"agent{i}": [] for i in range(len(self.agents))}
        stopping_condition = stop_condition if stop_condition is not None else StopCondition()

        best_policy_files = [
            f"Best_{self.env.type_name}_PPO_agent{i}_policy"
            for i in range(len(self.agents))
        ]

        for iteration in range(stopping_condition.max_time_steps):
            self.set_policy_eval()

            # Collect batch data from multiple parallel environments
            batch_data = self.run_parallel_episodes(self.num_envs)

            # Validate batch_data length matches number of agents
            if len(batch_data) != len(self.agents):
                raise RuntimeError(
                    f"Batch data size ({len(batch_data)}) does not match "
                    f"number of agents ({len(self.agents)})"
                )

            mean_rewards = []
            mean_losses = []

            # Train each agent
            for agent_id, (agent, episode) in enumerate(zip(self.agents, batch_data)):
                mean_reward = np.mean(episode['rewards'])
                mean_advantage = np.mean(episode['advantages'])

                # Train agent on its episode
                loss = agent.train([episode])

                # Store metrics
                mean_rewards.append(mean_reward)
                mean_losses.append(loss)

                training_metrics[f"agent{agent_id}"].append({
                    'iteration': iteration,
                    'loss': loss,
                    'advantage': mean_advantage,
                    'reward': mean_reward
                })

                # Save the best policy per agent
                if iteration >= self.MINIMUM_ITERATIONS_FOR_BEST_SAVE and mean_reward > best_rewards[agent_id]:
                    best_rewards[agent_id] = mean_reward
                    agent.save_policy(best_policy_files[agent_id])

                # Log evaluation results periodically
                if iteration % self.eval_frequency == 0:
                    print(f"Iteration {iteration}, Average Reward: {mean_reward:.3f}, Loss: {loss:.3f}")

            # Save checkpoint every 'self.eval_frequency' iterations
            if iteration > 0 and iteration % self.eval_frequency == 0:
                self.save_policy(f"ppo_checkpoint_{iteration}")

            # Aggregate metrics for stopping condition
            mean_reward_all_agents = np.mean(mean_rewards) if mean_rewards else 0.0
            mean_loss_all_agents = np.mean(mean_losses) if mean_losses else 0.0

            # Check if the condition is met for early stopping
            should_stop, reason = stopping_condition.should_stop(
                current_step=iteration,
                mean_reward=mean_reward_all_agents,
                mean_loss=mean_loss_all_agents
            )

            if should_stop:
                print(f"Stopping: {reason}")
                last_save_index = iteration
                break

        # Save final checkpoint
        self.save_policy(f'ppo_checkpoint_{last_save_index if last_save_index > 0 else stopping_condition.max_time_steps}')

        # Dump training metrics to CSV
        dump_multi_agent_metrics(self.model_dir, training_metrics)

        print("Training completed!")


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

        # Ensure the parent file has the .zip extension
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
                    # Validate archive contents
                    file_list = main_zip.namelist()
                    expected_files = {f"agent_{i}.pth" for i in range(len(self.agents))}
                    if not expected_files.issubset(set(file_list)):
                        missing = expected_files - set(file_list)
                        raise RuntimeError(f"Archive missing required files: {missing}")

                    main_zip.extractall(tmp_dir)

                # 2. Load each agent
                for i, agent in enumerate(self.agents):
                    agent_path = os.path.join(tmp_dir, f"agent_{i}.pth")

                    if not os.path.exists(agent_path):
                        raise FileNotFoundError(f"Missing policy file: agent_{i}.pth")

                    saved_variables = torch.load(agent_path, map_location=self.device, weights_only=False)

                    if agent.policy is None:
                        raise RuntimeError(f"Agent {i} has no initialized policy.")

                    agent.policy = PPOPolicy.build_policy_from_state(saved_variables, self.device)

                # Verify all agents loaded
                if any(agent.policy is None for agent in self.agents):
                    raise RuntimeError("Some agents failed to load policies")

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

        full_path = os.path.join(self.model_dir, policy_file_name) if self.model_dir else policy_file_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # 1. Save each agent as a .pth file (state_dict only)
                for i, agent in enumerate(self.agents):
                    agent_path = os.path.join(tmp_dir, f"agent_{i}.pth")

                    if agent.policy is None:
                        raise RuntimeError(f"Agent {i} has no policy to save.")

                    torch.save(
                        agent.policy.serialize_policy(),
                        agent_path,
                    )

                # 2. Zip all .pth files into one archive
                with zipfile.ZipFile(full_path, 'w', compression=zipfile.ZIP_DEFLATED) as main_zip:
                    for i in range(len(self.agents)):
                        file_name = f"agent_{i}.pth"
                        file_path = os.path.join(tmp_dir, file_name)

                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"Expected policy file {file_name} not found.")

                        main_zip.write(file_path, arcname=file_name)

            except Exception as e:
                raise RuntimeError(f"Failed to save multi-agent policy to '{full_path}'.") from e


    def set_policy_eval(self):
        """
        Set all agent policies to evaluation mode.

        This disables training-specific behavior such as dropout and ensures
        deterministic forward passes where applicable.
        """

        for agent in self.agents:
            agent.policy.network.eval()
