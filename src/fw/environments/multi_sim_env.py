import os
import numpy as np

from typing import Optional
from gymnasium import spaces
from fw.environments.base_env import BaseEnv


class MultiSimEnv(BaseEnv):
    """
    Synchronous multi-ship environment composed of several BaseEnv instances.

    Observations and actions use `gymnasium.spaces.Tuple`; rewards and flags
    are returned as NumPy arrays. Sub-environments are named
    "<type_name>_<index>".

    Attributes:
        _sub_environments (dict[str, BaseEnv]): Wrapped sub-environments by name.
        num_envs (int): Number of sub-environments.
        observation_space (spaces.Tuple): Tuple of sub-environment observation spaces.
        action_space (spaces.Tuple): Tuple of sub-environment action spaces.
    """

    def __init__(self, sub_environments: list[BaseEnv], render_mode: Optional[str] = None):
        """
        Initialize the multi-environment wrapper.

        Args:
            sub_environments (list[BaseEnv]): Non-empty list of environments to wrap.
            render_mode (Optional[str]): Rendering mode passed to `BaseEnv`.

        Raises:
            ValueError: If `sub_environments` is empty.
        """

        if not sub_environments:
            raise ValueError("MultiSimEnv requires at least one environment")

        super().__init__(render_mode=render_mode)

        for i, env in enumerate(sub_environments):
            name = f"{env.type_name}_{i}"
            env.env_name = name
            self._sub_environments[name] = env

        self.num_envs = len(sub_environments)
        self.agent_states = {}

        self.observation_space = spaces.Tuple(
            tuple(env.observation_space for env in sub_environments)
        )

        self.action_space = spaces.Tuple(
            tuple(env.action_space for env in sub_environments)
        )


    def _get_ordered_envs(self):
        """
        Return the sub-environments in insertion order.
        """

        return list(self.sub_environments.values())


    def get_sub_env(self, env_id: int):
        """
        Return the sub-environment at index `env_id`.
        """

        return self._get_ordered_envs()[env_id]


    def get_neutral_action(self, env_id: int, action_template):
        """
        Return the sub-environment's `neutral_action()`, or zeros like `action_template`.
        """

        sub_env = self.get_sub_env(env_id)

        if hasattr(sub_env, "neutral_action"):
            return sub_env.neutral_action()

        return np.zeros_like(action_template)


    def update_agent_states(self):
        """
        Refresh `agent_states` (env name to `get_agent_state()`) for all sub-environments.
        """

        self.agent_states.update({
            env.env_name: env.get_agent_state()
            for env in self.sub_environments.values()
        })



    def reset(self, seed: int = None, options=None):
        """
        Reset all sub-environments.

        A given seed is offset per environment (seed + i), so runs are
        reproducible without identical sub-environments.

        Args:
            seed (int, optional): Base random seed.
            options (dict, optional): Additional environment-specific reset options.

        Returns:
            tuple: (observations as a tuple matching `observation_space`,
                list of info dicts).
        """

        observations = []
        infos = []

        for i, env in enumerate(self.sub_environments.values()):
            env_seed = None if seed is None else seed + i
            obs, info = env.reset(seed=env_seed, options=options)

            observations.append(obs)
            infos.append(info)

        self.update_agent_states()

        return tuple(observations), infos


    def step(self, action: np.ndarray):
        """
        Step all sub-environments with the provided actions.

        Agent states are updated after all sub-environments have stepped, so
        every ship acts on the previous step's world state (no ordering bias).

        Args:
            action (tuple or list): One action per sub-environment.

        Returns:
            tuple:
                - observations (tuple[np.ndarray, ...]): Next observations.
                - rewards (np.ndarray): Per-env rewards, shape (num_envs,).
                - terminations (np.ndarray): Per-env termination flags.
                - truncations (np.ndarray): Per-env truncation flags.
                - infos (list[dict]): Per-env info dicts.

        Raises:
            TypeError: If `action` is not a tuple or list.
            ValueError: On a wrong action count (or, with DEBUG_ENV set, an
                action outside `action_space`).
        """

        if not isinstance(action, (tuple, list)):
            raise TypeError(f"Action must be tuple or list, got {type(action).__name__}")
        if len(action) != self.num_envs:
            raise ValueError(
                f"Action count mismatch: got {len(action)}, expected {self.num_envs}"
            )

        # spaces.Tuple.contains() rejects valid float32 arrays on dtype details,
        # so it only runs with DEBUG_ENV=1.
        if os.environ.get("DEBUG_ENV"):
            if not self.action_space.contains(action):
                raise ValueError(
                    f"Action does not match action_space.\n"
                    f"action_space: {self.action_space}\n"
                    f"received:     {action}"
                )

        observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []

        for i, env in enumerate(self.sub_environments.values()):
            env.set_other_agents_states(self.agent_states)
            obs, reward, terminated, truncated, info = env.step(action[i])

            observations.append(obs)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos.append(info)

        self.update_agent_states()

        return (
            tuple(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminations, dtype=bool),
            np.asarray(truncations, dtype=bool),
            infos,
        )
