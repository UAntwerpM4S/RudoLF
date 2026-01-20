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
        self._render_mode = render_mode
        self._smoothing_enabled = True


    @property
    def render_mode(self):
        return self._render_mode


    @render_mode.setter
    def render_mode(self, value):
        self._render_mode = value


    @property
    def enable_smoothing(self):
        return self._smoothing_enabled


    @enable_smoothing.setter
    def enable_smoothing(self, value):
        self._smoothing_enabled = value


    @property
    def type_name(self):
        """
        Return the type name of the environment.

        Returns:
            str: The class name of the current environment instance.
        """
        return type(self).__name__


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
