import os
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
        _sub_environments (dict[BaseEnv]): Dict of wrapped sub-environments.
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

        # --- Structured observation space ---
        self.observation_space = spaces.Tuple(
            tuple(env.observation_space for env in sub_environments)
        )

        # --- Structured action space ---
        self.action_space = spaces.Tuple(
            tuple(env.action_space for env in sub_environments)
        )


    def _get_ordered_envs(self):
        """
        Returns a stable ordered list of sub-environments.
        Cached to ensure consistent indexing.
        """

        return list(self.sub_environments.values())


    def get_sub_env(self, env_id: int):
        """
        Resolve sub-environment by index.
        """

        return self._get_ordered_envs()[env_id]


    def get_neutral_action(self, env_id: int, action_template):
        """
        Return neutral action for a specific sub-environment.
        """

        sub_env = self.get_sub_env(env_id)

        if hasattr(sub_env, "neutral_action"):
            return sub_env.neutral_action()

        return np.zeros_like(action_template)


    def update_agent_states(self):
        """
        Updates the stored states of all agents from their sub-environments.

        Iterates over all sub-environments, retrieves each agent's current
        state using `get_agent_state()`, and updates `self.agent_states`
        with a mapping from environment names to agent states.

        This method overwrites existing entries in `self.agent_states`
        for agents present in `self.sub_environments`.

        Returns:
            None
        """

        self.agent_states.update({
            env.env_name: env.get_agent_state()
            for env in self.sub_environments.values()
        })



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

        Agent states are updated AFTER all sub-environments have stepped.
        This implements synchronous (simultaneous) execution: every ship acts
        on the world state from the end of the previous timestep. This is
        intentional — it avoids ordering bias between agents and matches the
        standard multi-agent RL assumption.

        Args:
            action (tuple or list): One action per sub-environment.

        Returns:
            tuple:
                - observations (tuple[np.ndarray, ...]): Next observations.
                - rewards (np.ndarray): Per-env rewards, shape (num_envs,).
                - terminations (np.ndarray): Per-env termination flags.
                - truncations (np.ndarray): Per-env truncation flags.
                - infos (list[dict]): Per-env info dicts.
        """

        if not isinstance(action, (tuple, list)):
            raise TypeError(f"Action must be tuple or list, got {type(action).__name__}")
        if len(action) != self.num_envs:
            raise ValueError(
                f"Action count mismatch: got {len(action)}, expected {self.num_envs}"
            )

        # spaces.Tuple.contains() is sensitive to exact dtypes and rejects valid
        # float32 arrays in practice. Use DEBUG_ENV=1 to enable during development.
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

        # States updated after all ships have stepped (synchronous execution).
        self.update_agent_states()

        return (
            tuple(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminations, dtype=bool),
            np.asarray(truncations, dtype=bool),
            infos,
        )
