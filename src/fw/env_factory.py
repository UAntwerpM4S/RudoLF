import types

from fw.simulators.py_sim_env import PySimEnv
from fw.simulators.fh_sim_env import FhSimEnv
from fw.simulators.lunar_lander import LunarLander


class EnvironmentRegistryError(Exception):
    """ General environment registry error """


class _Factory:
    """Environment factory to create and store unique environment instances.

    This class keeps track of all registered environment classes and ensures
    that only one instance of each environment is created.
    """

    _environment_classes = {}
    _environment_instances = {}


    @classmethod
    def register(cls, environment_class):
        """Registers a new environment class for later instantiation.

        This method is automatically called when an environment subclass is defined.

        Args:
            environment_class (type): The environment class to register.

        Raises:
            EnvironmentRegistryError: If a class with the same environment name
            is already registered.
        """
        temp_instance = environment_class("", [])  # Create a temporary instance to extract the environment name
        environment_name = type(temp_instance._env).__name__
        assert environment_name, "Environment name must not be empty."

        try:
            existing_class = cls._environment_classes[environment_name]
        except KeyError:  # good, no existing class, no clash.
            cls._environment_classes[environment_name] = environment_class
            print(f"Successfully registered new environment class {environment_name}.")
        else:
            registered_names = ', '.join([f"'{name}'" for name in cls._environment_classes.keys()])
            print(f"Environment name '{environment_name}' not unique. Already registered: {registered_names}")

            raise EnvironmentRegistryError(f"An environment class called '{environment_name}' is already defined. "
                                           f"Class names must be unique.")


    @classmethod
    def get(cls, environment_name, configuration=None):
        """Retrieves an environment instance by name.

        If the environment instance does not exist yet, it will be created.

        Args:
            environment_name (str): The name of the environment to retrieve.
            configuration (dict, optional): Configuration to use for instantiation.
                Required if the environment is created for the first time.

        Returns:
            BaseEnvironment: The requested environment instance.

        Raises:
            EnvironmentRegistryError: If the environment name is not registered,
                or if a conflicting configuration is provided.
        """
        try:
            instance = cls._environment_instances[environment_name]  # Existing instance is candidate, if available.
        except KeyError:  # New instance! Exciting. Find the appropriate class.
            try:
                environment_class = cls._environment_classes[environment_name]
            except KeyError:
                available_list = ', '.join([f"'{name}'" for name in cls._environment_classes.keys()]
                                          ) if cls._environment_classes else 'none.'
                raise EnvironmentRegistryError(f"No environment '{environment_name}' is registered. "
                                               f"Available: {available_list}")

            # Instantiate and register instance for later use
            if configuration is None:
                raise EnvironmentRegistryError("A configuration must be provided when instantiating an environment "
                                               "for the first time")

            instance = environment_class(environment_name, configuration)
            cls._environment_instances[environment_name] = instance

        else:
            # Just in case: if about to return an existing instance, check configurations are compatible
            if configuration is not None and configuration != instance.configuration:
                raise EnvironmentRegistryError(f"A configuration was now provided for '{environment_name}' "
                                               f"which clashes with the configuration used with the environment "
                                               f"previously. Try again without providing a configuration.")

        assert len(cls._environment_instances) <= len(cls._environment_classes), "More environment instances than classes!"
        return instance


class BaseEnvironment:
    """Base class for an environment.

    Environment classes should inherit from this base class. Environments
    are automatically registered upon subclassing unless their name starts with '_'.
    """

    def __init__(self, name, configuration):
        """Initializes a BaseEnvironment.

        Args:
            name (str): The name of the environment.
            configuration (dict): Parameters used to configure the environment.
        """

        super().__init__()
        self._configuration = configuration
        self._name = name
        self._env = None  # Will hold the actual simulation environment (e.g., PySimEnv or FhSimEnv)


    def __init_subclass__(cls):
        """Automatically registers environment subclasses.

        This method is triggered when a new subclass is defined.
        """
        environment_name = cls.__name__

        # if the name doesn't start with an '_', register the class in the environment factory,
        # thus making it available for instantiation
        if not environment_name.startswith('_'):
            _Factory.register(cls)


    def __copy__(self):
        """Returns a shallow copy of the environment.

        Returns:
            BaseEnvironment: A shallow copy of the environment (self).
        """
        return self


    def __deepcopy__(self, _):
        """Returns a deep copy of the environment.

        Returns:
            BaseEnvironment: A deep copy of the environment (self).
        """
        return self


    @property
    def gym_env(self):
        """Returns the underlying Gym environment.

        Returns:
            gym.Env: The Gym-compatible environment object.

        Raises:
            SystemError: If the environment has not been created yet.
        """
        if self._env:
            return self._env
        else:
            raise SystemError('No environment created yet!')


    @property
    def configuration(self):
        """Returns a read-only view of the environment's configuration.

        Returns:
            MappingProxyType: An immutable dictionary view of the configuration.
        """
        return types.MappingProxyType(self._configuration)  # maybe overkill, but saves confusion


    def __repr__(self):
        """Returns the environment's name.

        Returns:
            str: The name of the environment.
        """
        return self._name


class PySimEnvironment(BaseEnvironment):
    """
    Class for Python Simulator environments
    """

    def __init__(self, name, configuration):
        super().__init__(name, configuration)

        """
        Ship_pos, target_pos and obstacles should be generated by the framework and send to env
        """
        ship_pos = [2098, -58]
        target_pos = [11530.0, 13000.0]

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self._env = DummyVecEnv([lambda: Monitor(PySimEnv(render_mode='human', max_steps=2000, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        self._env = PySimEnv(max_steps=8000, ship_pos=ship_pos, target_pos=target_pos)


class FhSimEnvironment(BaseEnvironment):
    """
    Class for the FH Sim environment
    """

    def __init__(self, name, configuration):
        super().__init__(name, configuration)

        """
        Ship_pos, target_pos and obstacles should be generated by the framework and send to env
        """
        ship_pos = [2245, -47]
        target_pos = [11530.0, 13000.0]

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self._env = DummyVecEnv([lambda: Monitor(FhSimEnv(render_mode='human', max_steps=1200, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        self._env = FhSimEnv(time_step=0.4, max_steps=65000, ship_pos=ship_pos, target_pos=target_pos)


class LunarEnvironment(BaseEnvironment):
    """
    Class for the lunar environment
    """

    def __init__(self, name, configuration):
        super().__init__(name, configuration)

        self._env = LunarLander(continuous=True)


# Our external API.
get = _Factory.get

py_sim_env_name = "PySimEnv"
fh_sim_env_name = "FhSimEnv"
lunar_env_name = "LunarLander"

def initialize_all_environments():
    get(py_sim_env_name, configuration=[])
    get(fh_sim_env_name, configuration=[])
    get(lunar_env_name, configuration=[])
