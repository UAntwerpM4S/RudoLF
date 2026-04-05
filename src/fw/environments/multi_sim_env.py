import numpy as np

from typing import Optional
from gymnasium import spaces
from fw.environments.base_env import BaseEnv


class MultiSimEnv(BaseEnv):
    """
    Wrapper environment that composes multiple independent `BaseEnv` instances
    into a single vectorized environment with structured (tuple-based) spaces.

    Each sub-environment is stepped independently, while observations and actions
    are grouped using `gymnasium.spaces.Tuple`. This preserves structural
    information and ensures compatibility with Gymnasium's space validation.

    The environment behaves as a synchronous vectorized environment:
    all sub-environments are stepped together, and results are returned as
    tuples (observations) or NumPy arrays (rewards, termination flags).

    Attributes:
        _sub_environments (list[BaseEnv]): List of wrapped sub-environments.
        num_envs (int): Number of sub-environments.
        observation_space (spaces.Tuple): Tuple of sub-environment observation spaces.
        action_space (spaces.Tuple): Tuple of sub-environment action spaces.
    """

    def __init__(self, sub_environments: list[BaseEnv], render_mode: Optional[str] = None):
        """
        Initialize the multi-environment wrapper.

        Args:
            sub_environments (list[BaseEnv]): Non-empty list of environments to wrap.
                Each environment must define compatible Gymnasium observation
                and action spaces.
            render_mode (Optional[str]): Rendering mode passed to `BaseEnv`.

        Raises:
            ValueError: If `environments` is empty.
            TypeError: If any sub-environment does not define valid Gymnasium
                observation or action spaces.
        """

        super().__init__(render_mode=render_mode)

        self._sub_environments = {}
        for i, env in enumerate(sub_environments):
            name = f"env_{i}"
            env.env_name = name
            self._sub_environments[name] = env
        self.num_envs = len(sub_environments)

        if self.num_envs == 0:
            raise ValueError("MultiSimEnv requires at least one environment")

        # --- Structured observation space ---
        self.observation_space = spaces.Tuple(
            tuple(env.observation_space for env in sub_environments)
        )

        # --- Structured action space ---
        self.action_space = spaces.Tuple(
            tuple(env.action_space for env in sub_environments)
        )


    def reset(self, seed: int = None, options=None):
        """
        Reset all sub-environments.

        Each sub-environment is reset independently. If a seed is provided,
        it is deterministically offset per environment (`seed + i`) to ensure
        reproducibility while avoiding identical trajectories.

        Args:
            seed (int, optional): Base random seed.
            options (dict, optional): Additional environment-specific reset options.

        Returns:
            tuple:
                - observations (tuple[np.ndarray, ...]): Tuple containing the
                  initial observation of each sub-environment.
                - infos (list[dict]): List of info dictionaries returned by each
                  sub-environment reset.

        Notes:
            The returned observation structure matches `observation_space`.
        """

        super().reset(seed=seed, options=options)

        observations = []
        infos = []

        for i, env in enumerate(self.sub_environments.values()):
            env_seed = None if seed is None else seed + i
            obs, info = env.reset(seed=env_seed, options=options)

            observations.append(obs)
            infos.append(info)

        return tuple(observations), infos


    def step(self, action: np.ndarray):
        """
        Step all sub-environments with the provided actions.

        The input action must be a tuple (or list) containing one action per
        sub-environment, matching the structure of `action_space`.

        Args:
            action (Sequence): Tuple or list of actions, where `action[i]` is
                applied to the i-th sub-environment.

        Returns:
            tuple:
                - observations (tuple[np.ndarray, ...]): Tuple of next observations.
                - rewards (np.ndarray): Array of shape (num_envs,) containing rewards.
                - terminations (np.ndarray): Boolean array indicating per-env termination.
                - truncations (np.ndarray): Boolean array indicating per-env truncation.
                - infos (list[dict]): List of info dictionaries from each sub-environment.

        Raises:
            AssertionError: If the action is not a sequence of length `num_envs`.
            ValueError: If the action does not conform to `action_space`.

        Notes:
            - Termination and truncation signals are returned per environment
              (vectorized), not aggregated.
            - No automatic resetting is performed for terminated environments.
        """

        assert isinstance(action, (tuple, list)), "Action must be tuple or list"
        assert len(action) == self.num_envs, "Action count mismatch"

        # Optional strict validation (recommended in debug mode)
        if not self.action_space.contains(action):
            raise ValueError("Action does not match action_space")

        observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []

        for i, env in enumerate(self.sub_environments.values()):
            obs, reward, terminated, truncated, info = env.step(action[i])

            observations.append(obs)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos.append(info)

        return (
            tuple(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminations, dtype=bool),
            np.asarray(truncations, dtype=bool),
            infos,
        )
