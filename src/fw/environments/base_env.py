import numpy as np
import gymnasium as gym

from typing import Optional


class BaseEnv(gym.Env):
    """
    Base class for all custom environments, extending Gymnasium's Env class.

    This class provides a common interface and base functionality for derived environments,
    including standardized initialization, environment-type identification, and stubs for
    environment-specific behaviors.
    """

    def __init__(self, render_mode=None):
        """
        Initialize the base environment.

        This constructor sets up the base Gym environment. It is intended to be extended
        by subclasses that implement specific environment logic.

        Args:
            render_mode (str, optional): The rendering mode. Defaults to None.
        """

        super().__init__()

        self._initial_speed = 0.0
        self._render_mode = render_mode
        self._smoothing_enabled = False
        self._rudder_angle_scaling_enabled = False
        self._env_name = type(self).__name__
        self._sub_environments = {}
        self._training_name = None


    @property
    def env_name(self):
        """
        Name of the environment instance.

        Returns:
            str: Environment name.
        """

        return self._env_name


    @env_name.setter
    def env_name(self, value):
        """
        Set the environment name.

        Args:
            value (str): Environment name.
        """

        self._env_name = value


    @property
    def initial_speed(self):
        """
        Initial vessel speed at environment reset.

        Returns:
            float: Initial speed.
        """

        return self._initial_speed


    @initial_speed.setter
    def initial_speed(self, value):
        """
        Set the initial vessel speed.

        Args:
            value (float): Initial speed.
        """

        self._initial_speed = value


    @property
    def render_mode(self):
        """
        Rendering mode used by the environment.

        Returns:
            str | None: Active render mode.
        """

        return self._render_mode


    @render_mode.setter
    def render_mode(self, value):
        """
        Set the rendering mode.

        Args:
            value (str | None): Render mode identifier.
        """

        self._render_mode = value


    @property
    def enable_smoothing(self):
        """
        Whether action smoothing is enabled.

        Returns:
            bool: True if smoothing is enabled, otherwise False.
        """

        return self._smoothing_enabled


    @enable_smoothing.setter
    def enable_smoothing(self, value):
        """
        Enable or disable action smoothing.

        Args:
            value (bool): Smoothing enabled state.
        """

        self._smoothing_enabled = value


    @property
    def enable_rudder_angle_scaling(self):
        """
        Enable rudder angle scaling for sim-to-sim transfer.

        When *True*, the rudder action is rescaled at every step to compensate
        for differing maximum rudder angles between the original (training)
        simulator and the current (target) simulator.  The scale is computed
        internally by the simulator as::

            rudder_scale = framework_max_rudder_angle / target_sim_max_rudder_angle

        Both values are read directly from the simulator at runtime, so no
        manual angle configuration is required.  *False* means the raw policy
        output is forwarded to the target simulator without rescaling.
        """

        return self._rudder_angle_scaling_enabled


    @enable_rudder_angle_scaling.setter
    def enable_rudder_angle_scaling(self, value):
        """
        Enable rudder angle scaling for sim-to-sim transfer.

        Args:
            value (bool): True to enable scaling, False to disable.
        """

        self._rudder_angle_scaling_enabled = value


    @property
    def sub_environments(self):
        """
        Returns the sub-environments. This property should only be used by multi-environment.
        """

        return self._sub_environments


    def neutral_action(self):
        """
        Return a neutral (no-op) action for the environment.

        A neutral action represents a control input that does not actively
        influence the system (e.g., zero thrust, centered steering). This is
        required for algorithms that continue stepping the simulator after an
        episode has terminated in one or more environments.

        Subclasses must implement this method and return an action that is
        compatible with the environment's action space.

        Returns:
            np.ndarray: Neutral action with the correct shape and dtype for
            the environment.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """

        raise NotImplementedError


    def get_neutral_action(self, env_id: int, action_template):
        """
        Return a neutral action for a given env_id.

        Delegates to self.neutral_action() so subclass-defined neutral
        actions are respected. Falls back to zeros_like(action_template)
        only if neutral_action() is not implemented.

        Args:
            env_id (int): Sub-environment index (ignored for single envs).
            action_template: An action array used for shape/dtype reference
                if the fallback path is taken.

        Returns:
            np.ndarray: Neutral action compatible with this environment's
                action space.
        """

        try:
            return self.neutral_action()
        except NotImplementedError:
            return np.zeros_like(action_template)


    def get_agent_state(self):
        """
        Retrieves the current state of this agent.

        This method should be implemented by subclasses to return a
        representation of the agent's internal state at the current time step.

        Returns:
            Any: The current state of the agent. The exact format and structure
            depend on the specific implementation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """

        raise NotImplementedError


    def set_other_agents_states(self, agent_states):
        """
        Updates the stored states of other agents in the environment.

        This method should be implemented by subclasses to store or process
        the states of all agents except the current one.

        Args:
            agent_states (dict): A mapping of agent identifiers (e.g., names or IDs)
                to their corresponding states.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """

        raise NotImplementedError


    @property
    def type_name(self):
        """
        Return the type name of the environment.

        This reflects the underlying gym implementation *class* only (e.g.
        "PySimEnv"). Multiple logically distinct environments (e.g. FhSimEnv
        and PySimEnv) can share the same implementation class, so this must
        NOT be used as an identity for policy files, metrics keys, or config
        lookups. Use `training_name` for that instead.

        Returns:
            str: The class name of the current environment instance.
        """

        return type(self).__name__


    @property
    def training_name(self):
        """
        Logical training identity of this environment.

        This is the registered wrapper name (e.g. "FhSimEnv",
        "PySimColavEnv") assigned by the `BaseEnvironment` factory wrapper
        that constructed this gym env, and is guaranteed unique per logical
        environment even when several wrappers share the same underlying
        gym implementation class. Use this (not `type_name`) for policy file
        naming, metrics dict keys, and any other place that needs to identify
        *which training run/environment* this is.

        Falls back to `type_name` when never explicitly set, e.g. when a gym
        env is constructed directly (tests, scripts) without going through a
        `BaseEnvironment` wrapper.

        Returns:
            str: The logical training identity for this environment.
        """

        return self._training_name if self._training_name is not None else self.type_name


    @training_name.setter
    def training_name(self, value):
        """
        Set the logical training identity for this environment.

        Args:
            value (str): The registered wrapper name to assign.
        """

        self._training_name = value


    def randomize(self, randomization_scale: Optional[float] = None):
        """
        Apply domain randomization to the environment.

        This method should be overridden by subclasses to apply random perturbations
        or modifications to the environment's parameters to improve generalization.

        Args:
            randomization_scale (Optional[Any]): Scale or configuration for the randomization.
                The exact structure depends on the specific environment's needs.
        """

        pass


    def env_specific_reset(self):
        """
        Perform any environment-specific reset logic.

        This method should be overridden to implement additional reset behavior
        that is specific to a subclass (e.g., setting custom internal state).
        """

        pass
