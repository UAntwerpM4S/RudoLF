import numpy as np

from typing import Optional
from gymnasium import spaces
from fw.environments.base_env import BaseEnv


class MultiPySimEnv(BaseEnv):
    """
    Composite environment that combines multiple independent environments
    into a single flattened observation and action space.

    Each sub-environment operates independently, while observations and actions
    are concatenated into a single vector interface.
    """

    def __init__(self, environments: list[BaseEnv], render_mode: Optional[str] = None):
        """
        Initialize the multi-environment wrapper.

        Args:
            environments (list[BaseEnv]): List of environments to combine.
            render_mode (Optional[str]): Rendering mode passed to BaseEnv.

        Raises:
            TypeError: If any observation or action space is not a Box space.
        """

        super().__init__(render_mode=render_mode)
        self.environments = environments

        self.obs_slices = []
        self.act_slices = []

        # --- Build observation space ---
        obs_lows, obs_highs = [], []
        start = 0

        for env in self.environments:
            space = env.observation_space
            if not isinstance(space, spaces.Box):
                raise TypeError("All observation spaces must be Box spaces.")

            flat_dim = space.shape[0]
            end = start + flat_dim

            self.obs_slices.append((start, end))
            start = end

            obs_lows.append(space.low.flatten())
            obs_highs.append(space.high.flatten())

        self.observation_space = spaces.Box(
            low=np.concatenate(obs_lows),
            high=np.concatenate(obs_highs),
            dtype=np.float32
        )

        # --- Build action space ---
        act_lows, act_highs = [], []
        start = 0

        for env in self.environments:
            space = env.action_space
            if not isinstance(space, spaces.Box):
                raise TypeError("All action spaces must be Box spaces.")

            flat_dim = space.shape[0]
            end = start + flat_dim

            self.act_slices.append((start, end))
            start = end

            act_lows.append(space.low.flatten())
            act_highs.append(space.high.flatten())

        self.action_space = spaces.Box(
            low=np.concatenate(act_lows),
            high=np.concatenate(act_highs),
            dtype=np.float32
        )

    def reset(self, seed: int = None, options=None):
        """
        Reset all sub-environments and return a concatenated observation.

        Args:
            seed (int, optional): Base seed for reproducibility. Each sub-env
                receives an incremented seed.
            options (dict, optional): Additional reset options.

        Returns:
            Tuple[np.ndarray, dict]: Concatenated observations and empty info dict.
        """

        super().reset(seed=seed, options=options)

        observations = []
        for i, env in enumerate(self.environments):
            env_seed = None if seed is None else seed + i
            obs, _ = env.reset(seed=env_seed, options=options)
            observations.append(obs)

        return np.concatenate(observations), {}

    def step(self, action: np.ndarray):
        """
        Step all sub-environments using sliced actions.

        Args:
            action (np.ndarray): Flattened action vector for all environments.

        Returns:
            Tuple:
                - np.ndarray: Concatenated observations
                - np.ndarray: Rewards per sub-environment
                - bool: True if any environment is terminated
                - bool: True if any environment is truncated
                - list: Info dicts from each sub-environment
        """

        observations, rewards = [], []
        terminateds, truncateds, infos = [], [], []

        for i, env in enumerate(self.environments):
            start, end = self.act_slices[i]
            sub_action = action[start:end]

            obs, reward, terminated, truncated, info = env.step(sub_action)

            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        return (
            np.concatenate(observations),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            infos
        )
