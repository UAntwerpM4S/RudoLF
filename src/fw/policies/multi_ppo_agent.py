import os
import tempfile
import zipfile

import numpy as np

from typing import Optional, Tuple
from fw.stop_condition import StopCondition
from fw.policies.base_agent import BaseAgent
from fw.environments.multi_py_sim_env import MultiPySimEnv


class MultiPPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent for reinforcement learning.

    This class handles the training, evaluation, and policy management for PPO-based reinforcement learning agents.
    """

    def __init__(self, agents: list[BaseAgent], env: MultiPySimEnv):
        super().__init__(environment=env)

        self.agents = agents
        self.observation_slices: list[slice] = []

        start = 0
        for sub_env in env.environments:
            dim = sub_env.observation_space.shape[0]
            end = start + dim

            self.observation_slices.append(slice(start, end))
            start = end


    def predict(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict an action for a given state using the policy.

        Args:
            state (np.ndarray): Current state as a NumPy array.
            deterministic (boolean): Whether the action should be deterministic or not.

        Returns:
            Tuple: A tuple containing the predicted action and log probability.
        """

        actions = []
        log_probs = []
        values = []

        for agent, obs_slice in zip(self.agents, self.observation_slices):
            sub_state = state[obs_slice]
            action, log_prob, value = agent.predict(state=sub_state,deterministic=deterministic)

            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

        return (
            np.concatenate(actions, axis=0),
            np.concatenate(log_probs, axis=0),
            np.concatenate(values, axis=0),
        )


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

        for agent in self.agents:
            agent.learn(stop_condition=stop_condition)


    def load_policy(self, policy_file_name: str) -> None:
        """
        Load a trained agent's policy.

        Args:
            policy_file_name (str): Name of the policy file to load.

        Raises:
            RuntimeError: If no agent is created or the policy fails to load.
            ValueError: If the loaded policy dimensions do not match the environment.
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
        Save the trained agent's policy.

        Args:
            policy_file_name (str): Name of the policy file to save.

        Raises:
            RuntimeError: If no agent is created or the policy fails to save.
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
        Set policy in evaluation mode.
        """
        for agent in self.agents:
            agent.policy.network.eval()
