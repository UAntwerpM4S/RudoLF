import os
import torch
import zipfile
import tempfile
import numpy as np

from typing import Optional, Tuple, Union
from fw.config_profile import VesselProfile
from fw.stop_condition import StopCondition
from fw.policies.ppo_policy import PPOPolicy
from fw.policies.base_ppo_agent import BasePPOAgent
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.common.tools.logging_utils import dump_multi_agent_metrics
from fw.environments.multi_sim_env import MultiSimEnv


class MultiPPOAgent(BasePPOAgent):
    """
    Multi-agent PPO trainer for tuple-based environments.

    This class coordinates multiple independent `SinglePPOAgent` instances,
    each controlling a sub-environment within a `MultiSimEnv`. Experience is
    collected synchronously across all environments, while policy updates are
    performed independently per agent.

    The training setup is decentralized:
        - Each agent observes only its own state.
        - Policies are not shared between agents.
        - No centralized critic is used.

    Attributes:
        agents (list[SinglePPOAgent]): List of PPO agents (one per environment).
        num_envs (int): Number of sub-environments.
        max_nbr_iterations (int): Maximum rollout length.
        normalize (bool): Whether to normalize rewards per trajectory.
    """

    MINIMUM_ITERATIONS_FOR_BEST_SAVE = 20

    def __init__(
        self,
        profile: VesselProfile,
        *args,
        agents: list[SinglePPOAgent],
        **kwargs
    ):
        """
        Initialize the multi-agent PPO trainer.

        Args:
            profile: Profile that will be used for training or visualization.
            *args: Positional arguments forwarded to `BasePPOAgent`.
            agents (list[SinglePPOAgent]): List of initialized agents, one per sub-environment.
            **kwargs: Keyword arguments forwarded to `BasePPOAgent`.

        Raises:
            TypeError: If the environment is not a `MultiSimEnv`.
            ValueError: If the number of agents does not match the number of sub-environments.

        Notes:
            - Each agent operates independently with its own policy and optimizer.
            - The number of agents must match `env.sub_environments`.
        """

        super().__init__(profile, *args, **kwargs)

        self.agents = agents
        self.num_envs = len(self.env.sub_environments)

        if not isinstance(self.env, MultiSimEnv):
            raise TypeError("MultiPPOAgent requires MultiSimEnv")

        if len(agents) != self.num_envs:
            raise ValueError(
                f"Number of agents ({len(agents)}) must match "
                f"number of sub-environments ({self.num_envs})"
            )


    def predict(
            self,
            state: Union[np.ndarray, Tuple[np.ndarray, ...]],
            deterministic: bool = False
    ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], np.ndarray, np.ndarray]:
        """
        Compute actions and value estimates for all agents.

        Args:
            state (Union[np.ndarray, Tuple[np.ndarray, ...]]): Tuple of observations,
                one per agent/environment.
            deterministic (bool): Whether to use deterministic policy outputs.

        Returns:
            Tuple:
                - actions (tuple): Actions per agent.
                - log_probs (np.ndarray): Log-probabilities of actions (num_envs,).
                - values (np.ndarray): Value estimates (num_envs,).

        Notes:
            - Each agent processes only its corresponding observation.
            - Output ordering matches agent/environment indexing.
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


    def _reset_envs(self, num_envs):
        """
        Reset the multi-environment.

        Args:
            num_envs (int): Number of environments (unused, inferred from env).

        Returns:
            np.ndarray: Tuple or array of initial observations.
        """

        for sub_env in self.env.sub_environments.values():
            if hasattr(sub_env, "randomize"):
                sub_env.randomize()

        states, _ = self.env.reset()
        return states


    def _step_envs(self, actions):
        """
        Step all sub-environments with joint actions.

        Args:
            actions (tuple): Tuple of actions, one per agent.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - next_states: Next observations.
                - rewards: Rewards per agent.
                - terminations: Termination flags.
                - truncations: Truncation flags.

        Notes:
            - Supports custom info dictionaries for richer transition data.
            - Falls back to standard Gymnasium outputs if unavailable.
        """

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

        return next_states, rewards, terminations, truncations


    def _predict(self, states, deterministic=False):
        """
        Internal wrapper for batched prediction.

        Args:
            states (np.ndarray or tuple): Observations per agent.
            deterministic (bool): Whether to use deterministic actions.

        Returns:
            Same as `predict`.
        """

        return self.predict(states, deterministic=deterministic)


    def _get_bootstrap_value(self, next_state, env_id: Optional[int] = None):
        """
        Estimate value for bootstrapping for a specific agent.

        Args:
            next_state (np.ndarray): State to evaluate.
            env_id (Optional[int]): Index of the agent/environment.

        Returns:
            float: Estimated value.

        Raises:
            TypeError: If `env_id` is None (subscripting a list with None raises TypeError).
        """

        if env_id is None:
            raise TypeError(
                "_get_bootstrap_value() requires an integer env_id; got None. "
                "Pass the index of the agent whose value function should be queried."
            )

        with torch.no_grad():
            last_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, _, next_values = self.agents[env_id].policy.network(last_state)
            return next_values.squeeze(0).item()


    def learn(
            self,
            stop_condition: Optional[StopCondition] = None,
    ) -> None:
        """
        Train all agents using parallel rollouts.

        Args:
            stop_condition (Optional[StopCondition]): Stopping criterion. If None,
                a default condition is used.

        Workflow:
            1. Collect trajectories across all environments.
            2. Train each agent independently on its trajectory.
            3. Track metrics and save best-performing policies.
            4. Apply early stopping if conditions are met.

        Side Effects:
            - Updates all agent policies in-place.
            - Saves checkpoints and best-performing models.
            - Logs per-agent training metrics to disk.

        Notes:
            - Training is synchronous across environments.
            - Episodes are matched to agents by env_id tag, not list position.
            - A missing trajectory for an agent causes a warning + skip, not a crash.
            - Metrics are recorded as NaN for skipped agents so the CSV stays aligned.
            - Early stopping uses aggregated metrics across agents.
            - Best policies are tracked per agent.
        """

        print(f"Starting training with {self.num_envs} parallel environments on a '{self.device}' device")

        last_save_index = 0
        best_rewards = [float('-inf')] * len(self.agents)
        training_metrics = {f"agent{i}": [] for i in range(len(self.agents))}
        stopping_condition = stop_condition if stop_condition is not None else StopCondition()

        best_policy_files = [
            f"Best_{env.type_name}_{i+1}_PPO_policy" if self.num_envs > 1
            else f"Best_{env.type_name}_PPO_policy"
            for i, env in enumerate(self.env.sub_environments.values())
        ]

        for iteration in range(stopping_condition.max_time_steps):
            self.set_policy_eval()

            # Collect batch data from multiple parallel environments
            batch_data = self.run_parallel_episodes(self.num_envs)

            # Build a dict keyed by env_id for safe lookup.
            # Each episode dict now carries episode["env_id"].
            episode_by_env: dict[int, dict] = {
                ep["env_id"]: ep
                for ep in batch_data
                if "env_id" in ep
            }

            # Fallback: if env_id tags are absent (e.g. running old base code),
            # reconstruct by position — same behaviour as before.
            if not episode_by_env:
                episode_by_env = {i: ep for i, ep in enumerate(batch_data)}

            mean_rewards = []
            mean_losses = []

            # Train each agent
            for agent_id, agent in enumerate(self.agents):
                episode = episode_by_env.get(agent_id)

                if episode is None:
                    print(f"[Warning] Iteration {iteration}: no trajectory for agent {agent_id} — skipping update.")
                    mean_rewards.append(float('nan'))
                    mean_losses.append(float('nan'))
                    training_metrics[f"agent{agent_id}"].append({
                        'iteration': iteration,
                        'loss': float('nan'),
                        'advantage': float('nan'),
                        'reward': float('nan'),
                    })
                    continue

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
                    'reward': mean_reward,
                })

                # Save the best policy per agent
                if iteration >= self.MINIMUM_ITERATIONS_FOR_BEST_SAVE and mean_reward > best_rewards[agent_id]:
                    best_rewards[agent_id] = mean_reward
                    agent.save_policy(best_policy_files[agent_id])

                # Log evaluation results periodically
                if iteration % self.eval_frequency == 0:
                    print(f"Iteration {iteration} | agent {agent_id} | average reward: {mean_reward:.3f}, loss: {loss:.3f}")

            # Periodic checkpoint (all agents in one zip)
            if iteration > 0 and iteration % self.eval_frequency == 0:
                self.save_policy(f"ppo_checkpoint_{iteration}")

            # Aggregate metrics for stopping condition — ignore NaN entries
            valid_rewards = [r for r in mean_rewards if not np.isnan(r)]
            valid_losses = [l for l in mean_losses if not np.isnan(l)]

            mean_reward_all_agents = np.mean(valid_rewards) if valid_rewards else 0.0
            mean_loss_all_agents = np.mean(valid_losses) if valid_losses else float('nan')

            # Check if the condition is met for early stopping
            should_stop, reason = stopping_condition.should_stop(
                current_step=iteration,
                mean_reward=mean_reward_all_agents,
                mean_loss=mean_loss_all_agents,
            )

            if should_stop:
                print(f"Stopping: {reason}")
                last_save_index = iteration
                break

        # Save final checkpoint
        self.save_policy(f'ppo_checkpoint_{last_save_index if last_save_index > 0 else stopping_condition.max_time_steps}')

        # Dump training metrics to CSV
        dump_multi_agent_metrics(self.model_dir, training_metrics)

        for sub_env in self.env.sub_environments.values():
            sub_env.close()

        print("Training completed!")


    def load_policy(self, policy_file_name: str) -> None:
        """
        Load a multi-agent policy from a ZIP archive.

        Args:
            policy_file_name (str): Path to the ZIP archive containing agent policies.

        Raises:
            FileNotFoundError: If the archive does not exist.
            RuntimeError: If loading fails or required files are missing.

        Notes:
            - Archive must contain files named `policy_0.pth`, `policy_1.pth`, etc.
            - Each file represents a serialized policy for one agent.
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
                    expected_files = {f"policy_{i}.pth" for i in range(len(self.agents))}
                    if not expected_files.issubset(set(file_list)):
                        missing = expected_files - set(file_list)
                        raise RuntimeError(f"Archive missing required files: {missing}")

                    main_zip.extractall(tmp_dir)

                # 2. Load each agent
                for i, agent in enumerate(self.agents):
                    agent_path = os.path.join(tmp_dir, f"policy_{i}.pth")

                    if not os.path.exists(agent_path):
                        raise FileNotFoundError(f"Missing policy file: policy_{i}.pth")

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
            policy_file_name (str): Output file name ('.zip' will be appended if missing).

        Raises:
            RuntimeError: If saving fails.
            FileNotFoundError: If any expected policy file is missing.

        Notes:
            - Each agent is saved as a separate `.pth` file inside the archive.
            - Policies are serialized using `policy.serialize_policy()`.
        """

        # Ensure the parent file has the .zip extension
        if not policy_file_name.endswith(".zip"):
            policy_file_name += ".zip"

        full_path = os.path.join(self.model_dir, policy_file_name) if self.model_dir else policy_file_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # 1. Save each agent as a .pth file (state_dict only)
                for i, agent in enumerate(self.agents):
                    agent_path = os.path.join(tmp_dir, f"policy_{i}.pth")

                    if agent.policy is None:
                        raise RuntimeError(f"Agent {i} has no policy to save.")

                    torch.save(
                        agent.policy.serialize_policy(),
                        agent_path,
                    )

                # 2. Zip all .pth files into one archive
                with zipfile.ZipFile(full_path, 'w', compression=zipfile.ZIP_DEFLATED) as main_zip:
                    for i in range(len(self.agents)):
                        file_name = f"policy_{i}.pth"
                        file_path = os.path.join(tmp_dir, file_name)

                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"Expected policy file {file_name} not found.")

                        main_zip.write(file_path, arcname=file_name)

            except Exception as e:
                raise RuntimeError(f"Failed to save multi-agent policy to '{full_path}'.") from e


    def set_policy_eval(self):
        """
        Set all agent policies to evaluation mode.

        Disables training-specific behavior such as dropout and ensures
        consistent inference across all agents.
        """

        for agent in self.agents:
            agent.policy.network.eval()
