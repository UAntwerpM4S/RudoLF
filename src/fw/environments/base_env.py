import numpy as np
import gymnasium as gym

from typing import Optional


class BaseEnv(gym.Env):
    """
    Common Gymnasium base for all framework environments: shared settings, identity and hooks.
    """

    def __init__(self, render_mode=None):
        """
        Args:
            render_mode (str, optional): Rendering mode.
        """

        super().__init__()

        self._initial_speed = 0.0
        self._initial_thrust = -1.0
        self._render_mode = render_mode
        self._smoothing_enabled = False
        self._rudder_angle_scaling_enabled = False
        self._env_name = type(self).__name__
        self._sub_environments = {}
        self._training_name = None


    @property
    def env_name(self):
        """
        str: Name of the environment instance.
        """

        return self._env_name


    @env_name.setter
    def env_name(self, value):
        """
        Set the environment name.
        """

        self._env_name = value


    @property
    def initial_speed(self):
        """
        float: Initial vessel speed at reset.
        """

        return self._initial_speed


    @initial_speed.setter
    def initial_speed(self, value):
        """
        Set the initial vessel speed.
        """

        self._initial_speed = value


    @property
    def initial_thrust(self):
        """
        float: Thrust the actuators start an episode with, on the scale of the
        policy action ([-1, 1]; -1 = min_thrust, the default).
        """

        return self._initial_thrust


    @initial_thrust.setter
    def initial_thrust(self, value):
        """
        Set the start thrust (action scale), applied on the next reset().

        Raises:
            ValueError: If value is outside [-1, 1] (or NaN).
        """

        # "not (a <= x <= b)" also rejects NaN
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"initial_thrust must be in [-1, 1] (action scale), got {value}.")
        self._initial_thrust = float(value)


    @property
    def render_mode(self):
        """
        str | None: Active render mode.
        """

        return self._render_mode


    @render_mode.setter
    def render_mode(self, value):
        """
        Set the render mode.
        """

        self._render_mode = value


    @property
    def enable_smoothing(self):
        """
        bool: Whether action smoothing is enabled.
        """

        return self._smoothing_enabled


    @enable_smoothing.setter
    def enable_smoothing(self, value):
        """
        Enable or disable action smoothing.
        """

        self._smoothing_enabled = value


    @property
    def enable_rudder_angle_scaling(self):
        """
        bool: Rescale the rudder action for sim-to-sim transfer.

        When True, the simulator scales the rudder by
        framework_max_rudder_angle / target_sim_max_rudder_angle (both read at
        runtime) to compensate for differing maximum rudder angles. When False,
        the raw policy output is forwarded.
        """

        return self._rudder_angle_scaling_enabled


    @enable_rudder_angle_scaling.setter
    def enable_rudder_angle_scaling(self, value):
        """
        Enable or disable rudder angle scaling.
        """

        self._rudder_angle_scaling_enabled = value


    @property
    def sub_environments(self):
        """
        Sub-environments (multi-environments only).
        """

        return self._sub_environments


    def neutral_action(self):
        """
        Return a neutral (no-op) action, e.g. zero thrust action and centred rudder.

        Used to keep stepping environments whose episode has already ended.

        Returns:
            np.ndarray: Action matching the action space.

        Raises:
            NotImplementedError: If the subclass does not implement it.
        """

        raise NotImplementedError


    def get_neutral_action(self, env_id: int, action_template):
        """
        Return neutral_action(), or zeros_like(action_template) if not implemented.

        Args:
            env_id (int): Sub-environment index (ignored for single envs).
            action_template: Reference array for the fallback's shape and dtype.

        Returns:
            np.ndarray: Neutral action.
        """

        try:
            return self.neutral_action()
        except NotImplementedError:
            return np.zeros_like(action_template)


    def get_agent_state(self):
        """
        Return this agent's current state (format defined by the subclass).

        Raises:
            NotImplementedError: If the subclass does not implement it.
        """

        raise NotImplementedError


    def set_other_agents_states(self, agent_states):
        """
        Store the states of all other agents.

        Args:
            agent_states (dict): Agent identifier to state.

        Raises:
            NotImplementedError: If the subclass does not implement it.
        """

        raise NotImplementedError


    @property
    def type_name(self):
        """
        str: Class name of the gym implementation (e.g. "PySimEnv").

        Several logical environments can share one class, so do not use this as
        an identity for policy files, metrics keys or config lookups; use
        training_name.
        """

        return type(self).__name__


    @property
    def training_name(self):
        """
        str: Logical identity of this environment.

        The registered wrapper name (e.g. "FhSimEnv", "PySimColavEnv") set by the
        BaseEnvironment wrapper; unique even when wrappers share a gym class. Use
        it for policy file names, metrics keys and similar. Falls back to
        type_name when not set (env built directly, e.g. in tests).
        """

        return self._training_name if self._training_name is not None else self.type_name


    @training_name.setter
    def training_name(self, value):
        """
        Set the logical training identity (registered wrapper name).
        """

        self._training_name = value


    def randomize(self, randomization_scale: Optional[float] = None):
        """
        Apply domain randomization (no-op by default; override in subclasses).

        Args:
            randomization_scale (Optional[float]): Randomization strength.
        """

        pass


    def env_specific_reset(self):
        """
        Hook for subclass-specific reset logic (no-op by default).
        """

        pass
