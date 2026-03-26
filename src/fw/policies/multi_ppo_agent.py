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
    Multi-agent PPO wrapper that coordinates multiple independent PPO agents.

    Each agent operates on a slice of the global observation space and produces
    a corresponding slice of the global action space.
    """

    def __init__(self, agents: list[BaseAgent], env: MultiPySimEnv):
        """
        Initialize the multi-agent PPO wrapper.

        Args:
            agents (list[BaseAgent]): Individual PPO agents.
            env (MultiPySimEnv): Combined multi-environment.
        """

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
        Predict actions, log probabilities, and values for all agents.

        Args:
            state (np.ndarray): Flattened global observation.
            deterministic (bool): Whether to use deterministic actions.

        Returns:
            Tuple:
                - np.ndarray: Concatenated actions
                - np.ndarray: Concatenated log probabilities
                - np.ndarray: Concatenated value estimates
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
        Train all underlying agents independently.

        Args:
            stop_condition (Optional[StopCondition]): Condition to stop training.
        """

        for agent in self.agents:
            agent.learn(stop_condition=stop_condition)


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
        Set all agent policies to evaluation mode.
        """

        for agent in self.agents:
            agent.policy.network.eval()
