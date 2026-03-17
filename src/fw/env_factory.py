import types
from collections.abc import Mapping

from airsim.config.airsim_config import (
    DEFAULT_AIRSIM_SAVE_SLOT,
    DEFAULT_AIRSIM_SIMULATION_SPEED,
)
from fw.config import get_config_parameters
from fw.environments.py_sim_env import PySimEnv
from fw.environments.fh_sim_env import FhSimEnv
from airsim.environments.airsim_env import AirSimEnv
from fw.environments.lunar_lander import LunarLander
from fw.environments.py_sim_polar_env import PySimPolarEnv


class EnvironmentRegistryError(Exception):
    """ General environment registry error """


def _normalize_configuration(configuration):
    """Normalize placeholder configuration values to stable runtime objects.

    A 'configuration' is a dict of key-value pairs that callers pass when
    requesting an environment, e.g. {"training_mode": True, "max_steps": 500}.
    Inside the factory and environment constructors, configuration values are
    read by key (cfg.get("training_mode", True)), so the object MUST behave
    like a dict — supporting membership tests, equality comparisons and .get().

    'Normalizing' means converting whatever the caller passed into a plain dict
    so all downstream code can rely on a single, consistent type.  Three input
    cases are handled:

      - None          → returned as-is; signals "no configuration provided yet"
                        (the factory will refuse to create a new instance without one).
      - Mapping       → copied into a plain dict; this covers OrderedDict,
                        MappingProxyType, custom mappings, etc.
      - [] or ()      → converted to {}; historically callers used an empty list
                        as a shorthand for "no configuration", but an empty list
                        does not support .get() or key-based equality checks.
                        Converting it to an empty dict makes it safe to use
                        everywhere a dict is expected, without changing the
                        semantic meaning ("no extra settings").
      - anything else → returned unchanged (unknown type; let the caller decide).
    """
    if configuration is None:
        return None
    if isinstance(configuration, Mapping):
        return dict(configuration)
    if isinstance(configuration, (list, tuple)) and len(configuration) == 0:
        # Empty list/tuple → empty dict. See docstring above for rationale.
        return {}
    return configuration


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
        try:
            temp_instance = environment_class("", [])  # Create a temporary instance to extract the environment name
        except Exception as e:
            print(f"Skipping registration of {environment_class.__name__}: {e}")
            return
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
        normalized_configuration = _normalize_configuration(configuration)

        try:
            environment_class = cls._environment_classes[environment_name]
        except KeyError:
            available_list = ', '.join([f"'{name}'" for name in cls._environment_classes.keys()]
                                      ) if cls._environment_classes else 'none.'
            raise EnvironmentRegistryError(f"No environment '{environment_name}' is registered. "
                                           f"Available: {available_list}")

        if not getattr(environment_class, "_cache_instance", True):
            if normalized_configuration is None:
                raise EnvironmentRegistryError("A configuration must be provided when instantiating an environment "
                                               "for the first time")
            return environment_class(environment_name, normalized_configuration)

        try:
            instance = cls._environment_instances[environment_name]  # Existing instance is candidate, if available.
        except KeyError:  # New instance! Exciting.
            if normalized_configuration is None:
                raise EnvironmentRegistryError("A configuration must be provided when instantiating an environment "
                                               "for the first time")

            instance = environment_class(environment_name, normalized_configuration)
            cls._environment_instances[environment_name] = instance

        else:
            # Just in case: if about to return an existing instance, check configurations are compatible
            if normalized_configuration is not None and normalized_configuration != instance.configuration:
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

    _cache_instance = True

    def __init__(self, name, configuration):
        """Initializes a BaseEnvironment.

        Args:
            name (str): The name of the environment.
            configuration (dict): Parameters used to configure the environment.
        """

        super().__init__()
        normalized_configuration = _normalize_configuration(configuration)
        self._configuration = {} if normalized_configuration is None else normalized_configuration
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
            object: An immutable mapping view when the configuration is
                mapping-like, otherwise the normalized configuration value.
        """
        if isinstance(self._configuration, Mapping):
            return types.MappingProxyType(self._configuration)  # maybe overkill, but saves confusion
        return self._configuration


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
        ship_pos = [2245, -47]
        target_pos = [-1559, 11990]
        config = get_config_parameters()

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self._env = DummyVecEnv([lambda: Monitor(PySimEnv(render_mode='human', max_steps=2000, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        self._env = PySimEnv(time_step=2.0, max_steps=14000, start_pos=ship_pos, target_pos=target_pos,
                             dynamics_model=config["dynamics_model"], ship_model=config["ship_model"],
                             wind=config["wind"], current=config["current"],
                             numerical_damping=config["numerical_damping"])


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
        target_pos = [-1559, 11990]
        config = get_config_parameters()

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self._env = DummyVecEnv([lambda: Monitor(FhSimEnv(render_mode='human', max_steps=1200, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        self._env = FhSimEnv(time_step=2.0, max_steps=135000, start_pos=ship_pos, target_pos=target_pos,
                             ship_model=config["ship_model"], scenario=config["scenario"])


class AirSimEnvironment(BaseEnvironment):
    """
    Class for the AirSim (UE5) environment.
    Requires UE5 with CosysAirSim to be running.
    """

    _cache_instance = False

    def __init__(self, name, configuration):
        super().__init__(name, configuration)
        
        # Factory calls __init__("", []) to extract the type name during class registration.
        # We return a dummy object to avoid connecting to UE5 unnecessarily.
        # The dummy's class name must be "AirSimEnv" so the factory registers the correct key.
        if name == "":
            self._env = type("AirSimEnv", (), {})()
            return

        ship_pos = [2245, -47]
        target_pos = [-1559, 11990]

        # Allow callers to control training vs. inference mode via configuration dict.
        # training_mode=True  → disables UE5 rendering + uses fast frame-stepping.
        # training_mode=False → restores UE5 rendering + uses real-time stepping.
        cfg = configuration if isinstance(configuration, dict) else {}
        training_mode = cfg.get("training_mode", True)
        use_pause = cfg.get("use_pause_stepping", training_mode)
        max_steps = int(cfg.get("max_steps", 14000))
        ip = cfg.get("ip", "")
        port = int(cfg.get("port", 41451))
        use_airsim = bool(cfg.get("use_airsim", True))
        timeout = float(cfg.get("timeout", 10.0))
        max_retries = int(cfg.get("max_retries", 3))
        health_check_interval = float(cfg.get("health_check_interval", 30.0))
        csv_log_path = str(cfg.get("csv_log_path", ""))
        save_slot = str(cfg.get("save_slot", DEFAULT_AIRSIM_SAVE_SLOT))
        simulation_speed = float(cfg.get("simulation_speed", DEFAULT_AIRSIM_SIMULATION_SPEED))
        frames_per_step = int(cfg.get("frames_per_step", 10))
        time_step = float(cfg.get("time_step", 0.1))
        physics_update_timeout = float(cfg.get("physics_update_timeout", 1.0))
        state_poll_interval = float(cfg.get("state_poll_interval", 0.005))
        latency_warning_threshold = cfg.get("latency_warning_threshold", None)

        try:
            self._env = AirSimEnv(
                ship_pos=ship_pos,
                target_pos=target_pos,
                max_steps=max_steps,
                ip=ip,
                port=port,
                use_airsim=use_airsim,
                timeout=timeout,
                max_retries=max_retries,
                health_check_interval=health_check_interval,
                csv_log_path=csv_log_path,
                save_slot=save_slot,
                simulation_speed=simulation_speed,
                training_mode=training_mode,
                use_pause_stepping=use_pause,
                frames_per_step=frames_per_step,
                time_step=time_step,
                physics_update_timeout=physics_update_timeout,
                state_poll_interval=state_poll_interval,
                latency_warning_threshold=(
                    float(latency_warning_threshold)
                    if latency_warning_threshold is not None
                    else None
                ),
            )
        except (ImportError, ConnectionError) as exc:
            # Graceful fallback: if AirSim is unavailable (missing package or no UE5 connection),
            print(
                "[env_factory] AirSim environment unavailable (reason: %s)." % exc
            )
            config = get_config_parameters()
            env_types = config.get("env_type", [])
            if isinstance(env_types, str):
                env_types = [env_types]

            # Find fallback environments in the configuration
            other_envs = []
            for e in env_types:
                # Use loose checking since resolve_environment_name might not be cleanly available if refactored
                e_lower = str(e).strip().lower()
                if e_lower not in ("airsimenv", "airsim"):
                    other_envs.append(e)

            if not other_envs:
                print("[env_factory] Only AirSimEnv was selected in config.py. Exiting.")
                import sys
                sys.exit(1)
            else:
                fallback_name = other_envs[0]
                print(f"[env_factory] Falling back to {fallback_name}.")
                fallback_name_lower = str(fallback_name).strip().lower()
                
                if fallback_name_lower in ("fhsimenv", "fhsim"):
                    self._env = FhSimEnvironment(name, configuration)._env
                elif fallback_name_lower in ("lunarlander", "lunar"):
                    self._env = LunarEnvironment(name, configuration)._env
                elif fallback_name_lower in ("pysimpolarenv", "pysimpolar", "polar"):
                    self._env = PySimPolarEnvironment(name, configuration)._env
                else:
                    self._env = PySimEnvironment(name, configuration)._env


class LunarEnvironment(BaseEnvironment):
    """
    Class for the lunar environment
    """

    def __init__(self, name, configuration):
        super().__init__(name, configuration)

        self._env = LunarLander(continuous=True)


class PySimPolarEnvironment(BaseEnvironment):
    """
    Class for Python Simulator environments
    """

    def __init__(self, name, configuration):
        super().__init__(name, configuration)

        """
        Ship_pos, target_pos and obstacles should be generated by the framework and send to env
        """
        ship_pos = [2098, -58]
        target_pos = [-1559, 11990]

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self._env = DummyVecEnv([lambda: Monitor(PySimEnv(render_mode='human', max_steps=2000, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        self._env = PySimPolarEnv(ship_pos=ship_pos, target_pos=target_pos, max_steps=2000, verbose=False)


# Our external API.
get = _Factory.get

py_sim_env_name = "PySimEnv"
fh_sim_env_name = "FhSimEnv"
airsim_env_name = "AirSimEnv"
lunar_env_name = "LunarLander"
py_sim_polar_env_name = "PySimPolarEnv"

_ENVIRONMENT_NAME_ALIASES = {
    py_sim_env_name.lower(): py_sim_env_name,
    "pysim": py_sim_env_name,
    fh_sim_env_name.lower(): fh_sim_env_name,
    "fhsim": fh_sim_env_name,
    airsim_env_name.lower(): airsim_env_name,
    "airsim": airsim_env_name,
    lunar_env_name.lower(): lunar_env_name,
    "lunar": lunar_env_name,
    py_sim_polar_env_name.lower(): py_sim_polar_env_name,
    "pysimpolar": py_sim_polar_env_name,
    "polar": py_sim_polar_env_name,
}


def resolve_environment_name(env_type):
    """Resolve a configured environment selector to a registered name.

    Args:
        env_type: Configured environment selector, alias, or canonical name.

    Returns:
        The canonical registered environment name used by the framework.

    Raises:
        EnvironmentRegistryError: If the selector does not match a known
            environment name or alias.
    """
    if env_type is None:
        return py_sim_env_name

    normalized = str(env_type).strip()
    if not normalized:
        return py_sim_env_name

    try:
        return _ENVIRONMENT_NAME_ALIASES[normalized.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_Factory._environment_classes.keys()))
        raise EnvironmentRegistryError(
            f"Unknown env_type '{env_type}'. Available environments: {available}"
        ) from exc


def resolve_environment_names(env_type):
    """Resolve one or more configured environment selectors to canonical names.

    Accepts either a single environment selector string or a list/tuple of
    selectors, and returns a deduplicated list of canonical environment names
    in the order they were specified.

    Args:
        env_type: A single environment selector string, or a list/tuple of them.

    Returns:
        A list of canonical registered environment names.

    Raises:
        EnvironmentRegistryError: If any selector does not match a known
            environment name or alias.
    """
    if isinstance(env_type, (list, tuple)):
        seen = []
        for name in env_type:
            resolved = resolve_environment_name(name)
            if resolved not in seen:
                seen.append(resolved)
        return seen
    return [resolve_environment_name(env_type)]


def build_default_environment(environment_name):
    """Instantiate the configured default environment with safe defaults.

    Args:
        environment_name: Canonical environment name returned by
            ``resolve_environment_name``.

    Returns:
        A registered environment instance ready for training or
        visualization.
    """
    if environment_name == airsim_env_name:
        return get(environment_name, configuration={"training_mode": True})
    return get(environment_name, configuration={})

def initialize_all_environments():
    get(py_sim_env_name, configuration={})
    get(fh_sim_env_name, configuration={})
    get(lunar_env_name, configuration={})
    get(py_sim_polar_env_name, configuration={})