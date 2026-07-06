import sys
import types

from typing import Any
from collections.abc import Mapping
from fw.environments.py_sim_env import PySimEnv
# from fw.environments.fh_sim_env import FhSimEnv
from airsim.environments.airsim_env import AirSimEnv
from fw.environments.lunar_lander import LunarLander
from fw.environments.multi_sim_env import MultiSimEnv
from fw.environments.py_sim_polar_env import PySimPolarEnv
from fw.environments.py_sim_colav_env import PySimColavEnv
from fw.common.tools.config_utils import normalize_configuration
from fw.environments.action_repeat_wrapper import ActionRepeatWrapper

from airsim.config.airsim_config import (
    DEFAULT_AIRSIM_SAVE_SLOT,
    DEFAULT_AIRSIM_SIMULATION_SPEED,
)
from fw.common.default_positions import (
    DEFAULT_FRAMEWORK_SHIP_POS,
    DEFAULT_FRAMEWORK_TARGET_POS,
    DEFAULT_PYSIM_POLAR_SHIP_POS,
)
from fw.environments.fh_sim_sync_env import FhSimSyncEnv


class EnvironmentRegistryError(Exception):
    """ General environment registry error """


class _Factory:
    """
    Environment factory to create and store unique environment instances.

    This class keeps track of all registered environment classes and ensures
    that only one instance of each environment is created.
    """

    _environment_classes = {}
    _environment_instances = {}


    @classmethod
    def register(cls, environment_class):
        """
        Registers a new environment class for later instantiation.

        This method is automatically called when an environment subclass is defined.
        The class must declare an `ENVIRONMENT_NAME` class attribute that provides
        the canonical registration key; no temporary instance is constructed.

        Args:
            environment_class (type): The environment class to register.

        Raises:
            EnvironmentRegistryError: If `ENVIRONMENT_NAME` is missing/empty, or
                if a class with the same name is already registered.
        """

        environment_name = getattr(environment_class, "ENVIRONMENT_NAME", None)
        if not environment_name:
            raise EnvironmentRegistryError(
                f"Environment class '{environment_class.__name__}' must define a non-empty "
                f"'ENVIRONMENT_NAME' class attribute."
            )

        try:
            cls._environment_classes[environment_name]
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
        """
        Retrieves an environment instance by name.

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

        normalized_configuration = normalize_configuration(configuration)

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
    """
    Base class for an environment.

    Environment classes should inherit from this base class. Environments
    are automatically registered upon subclassing unless their name starts with '_'.
    """

    _cache_instance = True

    def __init__(self, name, configuration):
        """
        Initializes a BaseEnvironment.

        Args:
            name (str): The name of the environment.
            configuration (dict): Parameters used to configure the environment.
        """

        super().__init__()
        normalized_configuration = normalize_configuration(configuration)
        self._configuration = {} if normalized_configuration is None else normalized_configuration
        self._name = name
        self._env = None  # Will hold the actual simulation environment (e.g., PySimEnv or FhSimEnv)


    def __init_subclass__(cls):
        """
        Automatically registers environment subclasses.

        This method is triggered when a new subclass is defined.
        """

        environment_name = cls.__name__

        # if the name doesn't start with an '_', register the class in the environment factory,
        # thus making it available for instantiation
        if not environment_name.startswith('_'):
            _Factory.register(cls)


    def __copy__(self):
        """
        Returns a shallow copy of the environment.

        Returns:
            BaseEnvironment: A shallow copy of the environment (self).
        """

        return self


    def __deepcopy__(self, _):
        """
        Returns a deep copy of the environment.

        Returns:
            BaseEnvironment: A deep copy of the environment (self).
        """

        return self


    @property
    def name(self):
        """
        str: Name of the environment.

        Returns the name assigned to the environment when the instance was
        created. The name can be used for identification, logging, and other
        environment management tasks.
        """

        return self._name


    @property
    def gym_env(self):
        """
        Returns the underlying Gym environment.

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
    def configuration(self) -> Mapping[str, Any]:
        """
        Returns a read-only view of the environment's configuration.

        Returns:
            object: An immutable mapping view when the configuration is
                mapping-like, otherwise the normalized configuration value.
        """

        if isinstance(self._configuration, Mapping):
            return types.MappingProxyType(self._configuration)  # maybe overkill, but saves confusion
        return self._configuration


    def __repr__(self):
        """
        Returns the environment's name.

        Returns:
            str: The name of the environment.
        """

        return self._name


class PySimEnvironment(BaseEnvironment):
    """
    Wrapper for the Python Simulator environment.
    """

    ENVIRONMENT_NAME = "PySimEnv"

    def __init__(self, name, configuration):
        """
        Create the default PySim environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_pysim_config`).
                Expected keys: wind, current, ship_model,
                dynamics_model, reward_name, reward_weights.
        """

        super().__init__(name, configuration)

        # Legacy callers (and `build_default_environment` without a framework
        # config) may instantiate this environment with an empty configuration
        # purely as a registry placeholder.  Skip the underlying PySimEnv
        # construction in that case rather than failing on missing keys.
        if not self.configuration:
            return

        # Ship_pos and target_pos should eventually be generated by the framework.
        ship_pos = list(DEFAULT_FRAMEWORK_SHIP_POS)
        target_pos = list(DEFAULT_FRAMEWORK_TARGET_POS)

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self._env = DummyVecEnv([lambda: Monitor(PySimEnv(render_mode='human', max_steps=2000, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        self._env = PySimEnv(time_step=2.0, max_steps=14000, start_pos=ship_pos, target_pos=target_pos,
                             dynamics_model=self.configuration["dynamics_model"],
                             ship_model=self.configuration["ship_model"],
                             wind=self.configuration["wind"],
                             current=self.configuration["current"],
                             ua_surrogate_config=self.configuration.get("ua_surrogate_config", {}),
                             reward_name=self.configuration.get("reward_name", "default"),
                             reward_weights=self.configuration.get("reward_weights", None))


class MultiPySimEnvironment(BaseEnvironment):
    """
    Wrapper for the multi-agent Python Simulator environment.
    """

    ENVIRONMENT_NAME = "MultiSimEnv"

    def __init__(self, name, configuration):
        """
        Create the default multi-PySim environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_pysim_config`).
                Expected keys: wind, current, ship_model,
                dynamics_model, reward_name, reward_weights.
        """

        super().__init__(name, configuration)

        # Legacy callers (and `build_default_environment` without a framework
        # config) may instantiate this environment with an empty configuration
        # purely as a registry placeholder.  Skip the underlying PySimEnv
        # construction in that case rather than failing on missing keys.
        if not self.configuration:
            return

        # Ship_pos and target_pos should eventually be generated by the framework.
        ship_pos = [2245, -47]
        target_pos = [-1559, 11990]

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self._env = DummyVecEnv([lambda: Monitor(PySimEnv(render_mode='human', max_steps=2000, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        single_pysim_env1 = PySimEnv(time_step=2.0, max_steps=14000, start_pos=ship_pos, target_pos=target_pos,
                             dynamics_model=self.configuration["dynamics_model"],
                             ship_model=self.configuration["ship_model"],
                             wind=self.configuration["wind"],
                             current=self.configuration["current"],
                             reward_name=self.configuration.get("reward_name", "default"),
                             reward_weights=self.configuration.get("reward_weights", None))

        single_pysim_env2 = PySimEnv(time_step=2.0, max_steps=14000, start_pos=ship_pos, target_pos=target_pos,
                             dynamics_model=self.configuration["dynamics_model"],
                             ship_model=self.configuration["ship_model"],
                             wind=self.configuration["wind"],
                             current=self.configuration["current"],
                             reward_name=self.configuration.get("reward_name", "default"),
                             reward_weights=self.configuration.get("reward_weights", None))

        self._env = MultiSimEnv(sub_environments=[single_pysim_env1, single_pysim_env2])


class PySimColavEnvironment(BaseEnvironment):
    """
    Class for a single PySimColav environment (one COLREGs scenario type).
    """

    ENVIRONMENT_NAME = "PySimColavEnv"

    def __init__(self, name, configuration):
        """
        Create the default PySimColav environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_colav_config`).
                Expected keys: wind, current, ship_model, dynamics_model,
                enable_obstacles.
        """
        super().__init__(name, configuration)

        # Legacy callers (and `build_default_environment` without a framework
        # config) may instantiate this environment with an empty configuration
        # purely as a registry placeholder.  Skip the underlying PySimColavEnv
        # construction in that case rather than failing on missing keys.
        if not self.configuration:
            return

        ship_pos = list(DEFAULT_FRAMEWORK_SHIP_POS)
        target_pos = list(DEFAULT_FRAMEWORK_TARGET_POS)

        self._env = PySimColavEnv(
            time_step=1.0,
            max_steps=1500,
            ship_pos=ship_pos,
            target_pos=target_pos,
            dynamics_model=self.configuration["dynamics_model"],
            ship_model=self.configuration["ship_model"],
            wind=self.configuration["wind"],
            current=self.configuration["current"],
            enable_obstacles=self.configuration.get("enable_obstacles", True),
        )


class MultiPolicyColavEnvironment(BaseEnvironment):
    """
    Single PySimColavEnv with scenario one-hot encoding enabled, for use
    with MultiPolicySingleAgent.  The environment appends a one-hot vector
    to the observation indicating the active scenario type, allowing the
    agent to route each timestep to the correct policy.
    """

    ENVIRONMENT_NAME = "MultiPolicyColavEnv"

    def __init__(self, name, configuration):
        """
        Create the MultiPolicyColav environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see
                `build_multi_policy_colav_config`). Expected keys: wind,
                current, ship_model, dynamics_model, enable_obstacles.
                eval/eval_scenario/eval_obstacle_count/
                include_scenario_one_hot/minimal_path_render remain
                hardcoded below pending a future rework.
        """
        super().__init__(name, configuration)

        # Legacy callers (and `build_default_environment` without a framework
        # config) may instantiate this environment with an empty configuration
        # purely as a registry placeholder.  Skip the underlying PySimColavEnv
        # construction in that case rather than failing on missing keys.
        if not self.configuration:
            return

        ship_pos = list(DEFAULT_FRAMEWORK_SHIP_POS)
        target_pos = list(DEFAULT_FRAMEWORK_TARGET_POS)

        self._env = PySimColavEnv(
            time_step=1.0,
            max_steps=1500,
            ship_pos=ship_pos,
            target_pos=target_pos,
            dynamics_model=self.configuration["dynamics_model"],
            ship_model=self.configuration["ship_model"],
            wind=self.configuration["wind"],
            current=self.configuration["current"],
            enable_obstacles=self.configuration.get("enable_obstacles", True),
            evaluate=False,
            eval_scenario="crossing-right",
            eval_obstacle_count=1,
            include_scenario_one_hot=True,
            minimal_path_render=True
        )


class FhSimEnvironment(BaseEnvironment):
    """
    Wrapper for the FH Simulator environment.
    """

    ENVIRONMENT_NAME = "FhSimEnv"

    def __init__(self, name, configuration):
        """
        Create the default FH simulator environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_fhsim_config`).
                Expected keys: ship_model, scenario.
        """

        super().__init__(name, configuration)

        # Registry placeholder instantiation with no config - nothing to build.
        if not self.configuration:
            return

        # Ship_pos and target_pos should eventually be generated by the framework.
        ship_pos = list(DEFAULT_FRAMEWORK_SHIP_POS)
        target_pos = list(DEFAULT_FRAMEWORK_TARGET_POS)

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self._env = DummyVecEnv([lambda: Monitor(FhSimEnv(render_mode='human', max_steps=1200, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        self._env = PySimEnv(time_step=2.0, max_steps=135000, start_pos=ship_pos, target_pos=target_pos,
                             ship_model=self.configuration["ship_model"],
                             scenario=self.configuration["scenario"],
                             dynamics_model=self.configuration["dynamics_model"])




class FhSimSyncEnvironment(BaseEnvironment):
    """
    Realtime sibling of FhSimEnvironment.

    Connects to a running simSync instance over UDP instead of initializing
    a local FH math model. Same framework-facing shape as FhSimEnvironment:
    the framework gets the same env interface, the same observation/reward
    pipeline, the same agent loop — only the underlying simulator changes.
    """

    ENVIRONMENT_NAME = "FhSimSyncEnv"

    def __init__(self, name, configuration):
        """
        Create the realtime simSync environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_fhsimsync_config`).
                Required keys: ship_model.
                Optional keys (all have sensible defaults if absent):
                    remote_ip, remote_port: simSync endpoint to connect to.
                    local_ip, local_port:   local UDP endpoint to bind.
                    connect_timeout:        seconds to wait for the first state event.
                    step_timeout_per_tick:  seconds to wait per sim tick during step().
        """

        super().__init__(name, configuration)

        # Registry placeholder instantiation with no config - nothing to build.
        if not self.configuration:
            return

        # ship_pos and target_pos remain framework-side concerns (they feed
        # observation construction and reward), even though the *actual*
        # vessel state lives in the remote simSync.
        ship_pos = list(DEFAULT_FRAMEWORK_SHIP_POS)
        target_pos = list(DEFAULT_FRAMEWORK_TARGET_POS)

        self._env = FhSimSyncEnv(
            time_step=2.0,
            max_steps=135000,
            start_pos=ship_pos,
            target_pos=target_pos,
            ship_model=self.configuration["ship_model"],
            remote_ip=self.configuration.get("remote_ip", "127.0.0.1"),
            remote_port=self.configuration.get("remote_port", 5000),
            local_ip=self.configuration.get("local_ip", "127.0.0.1"),
            local_port=self.configuration.get("local_port", 5001),
            connect_timeout=self.configuration.get("connect_timeout", 10.0),
            step_timeout_per_tick=self.configuration.get("step_timeout_per_tick", 1.0),
        )

class AirSimEnvironment(BaseEnvironment):
    """
    Wrapper for the AirSim (UE5) environment.

    Requires UE5 with CosysAirSim to be running.
    """

    ENVIRONMENT_NAME = "AirSimEnv"
    _cache_instance = False

    def __init__(self, name, configuration):
        """
        Create the AirSim environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped AirSim configuration dict
                (see `build_airsim_config`). All AirSim-specific keys with
                their defaults live in that builder function.

        Raises:
            ImportError: If the airsim package is not installed.
            ConnectionError: If UE5 / CosysAirSim is not reachable.
        """

        super().__init__(name, configuration)

        ship_pos = list(DEFAULT_FRAMEWORK_SHIP_POS)
        target_pos = list(DEFAULT_FRAMEWORK_TARGET_POS)

        # Allow callers to control training vs. inference mode via configuration dict.
        # training_mode=True  -> disables UE5 rendering + uses fast frame-stepping.
        # training_mode=False -> restores UE5 rendering + uses real-time stepping.
        cfg = configuration if isinstance(configuration, dict) else {}
        training_mode = cfg.get("training_mode", True)
        no_rendering_mode = bool(cfg.get("no_rendering_mode", False))
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
        load_world_on_start = cfg.get("load_world_on_start", None)
        capture_live_start_pose = cfg.get("capture_live_start_pose", None)
        simulation_speed = float(cfg.get("simulation_speed", DEFAULT_AIRSIM_SIMULATION_SPEED))
        draw_calibration_reference = bool(cfg.get("draw_calibration_reference", True))
        episode_log_path = str(cfg.get("episode_log_path", ""))
        normalize_reward = bool(cfg.get("normalize_reward", True))
        reward_normalization_scale = float(cfg.get("reward_normalization_scale", 5.0))
        reset_mode = str(cfg.get("reset_mode", "rewind_pose"))
        progress_log_interval_steps = int(cfg.get("progress_log_interval_steps", 0))
        log_pause_watchdog_warnings = bool(cfg.get("log_pause_watchdog_warnings", False))
        use_sim_time_stepping = bool(cfg.get("use_sim_time_stepping", False))
        sim_time_per_action = float(cfg.get("sim_time_per_action", cfg.get("time_step", 0.1)))
        frames_per_step = int(cfg.get("frames_per_step", 8))
        time_step = float(cfg.get("time_step", 0.1))
        physics_update_timeout = float(cfg.get("physics_update_timeout", 1.0))
        state_poll_interval = float(cfg.get("state_poll_interval", 0.005))
        latency_warning_threshold = cfg.get("latency_warning_threshold", None)
        curriculum_enabled = bool(cfg.get("curriculum_enabled", False))
        curriculum_window_size = int(cfg.get("curriculum_window_size", 8))
        curriculum_target_checkpoint = cfg.get("curriculum_target_checkpoint", None)
        curriculum_frontier_increment = int(cfg.get("curriculum_frontier_increment", 4))
        curriculum_window_increment = int(cfg.get("curriculum_window_increment", 6))
        curriculum_success_patience = int(cfg.get("curriculum_success_patience", 2))
        curriculum_randomization_scale = float(cfg.get("curriculum_randomization_scale", 0.0))
        guidance_assist_enabled = bool(cfg.get("guidance_assist_enabled", False))
        guidance_residual_steering_scale = float(cfg.get("guidance_residual_steering_scale", 0.45))
        guidance_residual_throttle_scale = float(cfg.get("guidance_residual_throttle_scale", 0.55))
        control_interface = str(cfg.get("control_interface", "car_physics"))
        observation_profile = str(cfg.get("observation_profile", "ppo_route_relative_car"))

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
            load_world_on_start=load_world_on_start,
            capture_live_start_pose=capture_live_start_pose,
            simulation_speed=simulation_speed,
            draw_calibration_reference=draw_calibration_reference,
            episode_log_path=episode_log_path,
            normalize_reward=normalize_reward,
            reward_normalization_scale=reward_normalization_scale,
            reset_mode=reset_mode,
            progress_log_interval_steps=progress_log_interval_steps,
            log_pause_watchdog_warnings=log_pause_watchdog_warnings,
            use_sim_time_stepping=use_sim_time_stepping,
            sim_time_per_action=sim_time_per_action,
            no_rendering_mode=no_rendering_mode,
            training_mode=training_mode,
            use_pause_stepping=use_pause,
            frames_per_step=frames_per_step,
            time_step=time_step,
            physics_update_timeout=physics_update_timeout,
            state_poll_interval=state_poll_interval,
            curriculum_enabled=curriculum_enabled,
            curriculum_window_size=curriculum_window_size,
            curriculum_target_checkpoint=(
                int(curriculum_target_checkpoint)
                if curriculum_target_checkpoint is not None
                else None
            ),
            curriculum_frontier_increment=curriculum_frontier_increment,
            curriculum_window_increment=curriculum_window_increment,
            curriculum_success_patience=curriculum_success_patience,
            curriculum_randomization_scale=curriculum_randomization_scale,
            guidance_assist_enabled=guidance_assist_enabled,
            guidance_residual_steering_scale=guidance_residual_steering_scale,
            guidance_residual_throttle_scale=guidance_residual_throttle_scale,
            control_interface=control_interface,
            observation_profile=observation_profile,
            latency_warning_threshold=(
                float(latency_warning_threshold)
                if latency_warning_threshold is not None
                else None
            ),
        )

        # Action repeat wrapper can be set dynamically or statically (e.g., repeat=4)
        repeat_steps = int(cfg.get("action_repeat", 1))
        if repeat_steps > 1:
            self._env = ActionRepeatWrapper(self._env, repeat=repeat_steps)


class LunarEnvironment(BaseEnvironment):
    """
    Wrapper for the LunarLander environment.
    """

    ENVIRONMENT_NAME = "LunarLander"

    def __init__(self, name, configuration):
        """
        Create the default continuous LunarLander wrapper.

        Args:
            name: Registered environment name.
            configuration: Optional runtime configuration forwarded by the
                environment factory.
        """

        super().__init__(name, configuration)

        self._env = LunarLander(continuous=True)


class PySimPolarEnvironment(BaseEnvironment):
    """
    Wrapper for the polar-coordinate Python Simulator environment.
    """

    ENVIRONMENT_NAME = "PySimPolarEnv"

    def __init__(self, name, configuration):
        """
        Create the default PySim polar environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Optional runtime configuration forwarded by the
                environment factory.
        """

        super().__init__(name, configuration)

        # Ship_pos and target_pos should eventually be generated by the framework.
        ship_pos = list(DEFAULT_PYSIM_POLAR_SHIP_POS)
        target_pos = list(DEFAULT_FRAMEWORK_TARGET_POS)

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self._env = DummyVecEnv([lambda: Monitor(PySimEnv(render_mode='human', max_steps=2000, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        self._env = PySimPolarEnv(ship_pos=ship_pos, target_pos=target_pos, max_steps=2000, verbose=False)



# ---------------------------------------------------------------------------
# Environment-scoped configuration builders
#
# Each builder extracts only the keys that a specific environment class needs
# from the full framework_config dict.  This prevents environments from
# receiving (and accidentally depending on) unrelated configuration like
# reward_configs, action, viz_mode, or AirSim settings.
#
# reward_name / reward_weights are passed through when present so that the
# evaluate_reward workflow can override them per-config without disturbing the
# base framework settings.
# ---------------------------------------------------------------------------

def build_pysim_config(framework_config: dict, extra: dict | None = None) -> dict:
    """
    Extract PySim-relevant keys from *framework_config*.

    Args:
        framework_config: Full resolved framework configuration dict.
        extra: Optional overrides merged on top (e.g. reward_name/weights
            injected by the evaluate_reward workflow).

    Returns:
        A dict containing only the keys consumed by PySimEnvironment and
        MultiPySimEnvironment.
    """

    cfg = {
        "wind":                 framework_config["wind"],
        "current":              framework_config["current"],
        "ship_model":           framework_config["ship_model"],
        "dynamics_model":       framework_config["dynamics_model"],
        "ua_surrogate_config":  framework_config["ua_surrogate_config"],
        "reward_weights":       framework_config.get("reward_weights", None),
        "reward_name":          framework_config.get("reward_name", "default"),
    }
    if extra:
        cfg.update(extra)
    return cfg


def build_colav_config(framework_config: dict, extra: dict | None = None) -> dict:
    """
    Extract PySimColav-relevant keys from *framework_config*.

    Args:
        framework_config: Full resolved framework configuration dict.
        extra: Optional overrides merged on top.

    Returns:
        A dict containing only the keys consumed by PySimColavEnvironment.
    """

    cfg = {
        "wind":                framework_config["wind"],
        "current":             framework_config["current"],
        "ship_model":          framework_config["ship_model"],
        "dynamics_model":      framework_config["dynamics_model"],
        "enable_obstacles":    framework_config.get("enable_obstacles", True),
    }
    if extra:
        cfg.update(extra)
    return cfg


def build_multi_policy_colav_config(framework_config: dict, extra: dict | None = None) -> dict:
    """
    Extract MultiPolicyColav-relevant keys from *framework_config*.

    Currently identical to `build_colav_config`; the eval/scenario-related
    fields (eval, eval_scenario, eval_obstacle_count, include_scenario_one_hot,
    minimal_path_render) remain hardcoded in MultiPolicyColavEnvironment
    pending a future rework, so they are not sourced from framework_config yet.

    Args:
        framework_config: Full resolved framework configuration dict.
        extra: Optional overrides merged on top.

    Returns:
        A dict containing only the keys consumed by MultiPolicyColavEnvironment.
    """

    return build_colav_config(framework_config, extra)


def build_fhsim_config(framework_config: dict) -> dict:
    """
    Extract FhSim-relevant keys from *framework_config*.

    Args:
        framework_config: Full resolved framework configuration dict.

    Returns:
        A dict containing only the keys consumed by FhSimEnvironment.
    """

    return {
        "ship_model": framework_config["ship_model"],
        "scenario":   framework_config["scenario"],
        "dynamics_model": framework_config["dynamics_model"],
    }


def build_fhsimsync_config(framework_config: dict) -> dict:
    """
    Extract FhSimSync-relevant keys from *framework_config*.

    Args:
        framework_config: Full resolved framework configuration dict.

    Returns:
        A dict containing the required keys consumed by FhSimSyncEnvironment
        (ship_model, scenario) plus any optional UDP connection overrides
        (remote_ip, remote_port, local_ip, local_port, connect_timeout,
        step_timeout_per_tick) when present in the framework config.
    """

    cfg = {
        "ship_model": framework_config["ship_model"],
    }
    # Forward optional UDP / timing overrides when explicitly set in config.
    for optional_key in (
        "remote_ip", "remote_port",
        "local_ip", "local_port",
        "connect_timeout", "step_timeout_per_tick",
    ):
        if optional_key in framework_config:
            cfg[optional_key] = framework_config[optional_key]
    return cfg


def build_airsim_config(framework_config: dict, training_mode: bool = True) -> dict:
    """
    Extract AirSim-relevant keys from *framework_config*.

    AirSim configuration lives under the `airsim` sub-key (populated by
    `get_airsim_config()`).  Any key not present falls back to the
    AirSimEnvironment constructor defaults.

    Args:
        framework_config: Full resolved framework configuration dict.
        training_mode: When True (default) forces training-mode regardless of
            what the config specifies.  Pass False for visualize/inference runs.

    Returns:
        A dict containing only the keys consumed by AirSimEnvironment, with
        `training_mode` always explicitly set.
    """

    # Fall back to the flat framework_config if no nested "airsim" sub-key
    # exists, preserving backward compatibility with the current config layout
    # where AirSim settings live at the top level of CONFIG_PARAMETERS.
    airsim_cfg = framework_config.get("airsim") or framework_config
    cfg = dict(airsim_cfg)
    cfg["training_mode"] = training_mode
    return cfg


def _pick_airsim_fallback(configuration) -> str:
    """
    Return the canonical name of the first non-AirSim environment in config.

    Args:
        configuration: The raw configuration passed to the failed AirSim call.
            Used only to retrieve `env_type`; may be `None`.

    Returns:
        Canonical environment name to use as a fallback.

    Side-effects:
        Calls `sys.exit(1)` if AirSimEnv is the *only* configured environment
        and no fallback is available.
    """

    # Prefer the configuration that was passed in so we avoid a redundant
    # config reload.  Fall back to a fresh load only when nothing was provided.
    if configuration and isinstance(configuration, dict):
        config = configuration
    else:
        from fw.config import get_config_parameters  # local import — last resort only
        config = get_config_parameters()

    env_types = config.get("env_type", [])
    if isinstance(env_types, str):
        env_types = [env_types]

    other_envs = [
        e for e in env_types
        if str(e).strip().lower() not in ("airsimenv", "airsim")
    ]

    if not other_envs:
        print("[env_factory] Only AirSimEnv was selected in config.py. Exiting.")
        sys.exit(1)

    fallback_raw = other_envs[0]
    fallback_name = resolve_environment_name(fallback_raw)
    print(f"[env_factory] Falling back to {fallback_name}.")
    return fallback_name


def get(environment_name: str, configuration=None):
    """
    Retrieve (or create) an environment instance by name.

    This is the single public entry-point for obtaining environments.  For
    every environment other than AirSim it delegates directly to
    `_Factory.get`.  For AirSim it additionally intercepts
    `ImportError` and `ConnectionError` and transparently re-routes the
    call to the configured fallback environment so that callers always
    receive an instance whose Python type exactly matches the environment
    that is actually running.

    Args:
        environment_name (str): Canonical or alias environment name.
        configuration (dict, optional): Configuration forwarded to the
            environment on first instantiation.

    Returns:
        BaseEnvironment: A fully constructed environment instance.

    Raises:
        EnvironmentRegistryError: Propagated from `_Factory.get` for any
            non-AirSim environment, or for AirSim when the resolved fallback
            is also unavailable.
    """

    if environment_name != airsim_env_name:
        return _Factory.get(environment_name, configuration)

    try:
        return _Factory.get(environment_name, configuration)
    except (ImportError, ConnectionError) as exc:
        print(f"[env_factory] AirSim environment unavailable (reason: {exc}).")
        # `configuration` here is an AirSim-scoped dict (from build_airsim_config),
        # *not* the full framework config.  Passing it to build_default_environment
        # for a PySim-family fallback would cause a KeyError on keys such as
        # "wind", and "current" that the AirSim config does not carry.
        # Load the full framework config instead so the fallback
        # environment receives the keys it actually needs.
        from fw.config import get_config_parameters as _get_fw_config  # local to avoid circular import
        fallback_framework_config = _get_fw_config()
        fallback_name = _pick_airsim_fallback(fallback_framework_config)
        return build_default_environment(fallback_name, framework_config=fallback_framework_config)


py_sim_env_name = "PySimEnv"
fh_sim_env_name = "FhSimEnv"
sync_sim_env_name = "FhSimSyncEnv"
airsim_env_name = "AirSimEnv"
lunar_env_name = "LunarLander"
py_sim_polar_env_name = "PySimPolarEnv"
multi_sim_env_name = "MultiSimEnv"
py_sim_colav_env_name = "PySimColavEnv"
multi_policy_colav_env_name = "MultiPolicyColavEnv"

_ENVIRONMENT_NAME_ALIASES = {
    py_sim_env_name.lower(): py_sim_env_name,
    "pysim": py_sim_env_name,
    fh_sim_env_name.lower(): fh_sim_env_name,
    "fhsim": fh_sim_env_name,
    sync_sim_env_name.lower(): sync_sim_env_name,
    "fhsimsync": sync_sim_env_name,
    airsim_env_name.lower(): airsim_env_name,
    "airsim": airsim_env_name,
    lunar_env_name.lower(): lunar_env_name,
    "lunar": lunar_env_name,
    py_sim_polar_env_name.lower(): py_sim_polar_env_name,
    "pysimpolar": py_sim_polar_env_name,
    "polar": py_sim_polar_env_name,
    multi_sim_env_name.lower(): multi_sim_env_name,
    "multipysim": multi_sim_env_name,
    py_sim_colav_env_name.lower(): py_sim_colav_env_name,
    "pysimcolav": py_sim_colav_env_name,
    "colav": py_sim_colav_env_name,
    multi_policy_colav_env_name.lower(): multi_policy_colav_env_name,
    "multipolicycolav": multi_policy_colav_env_name,
    "mpcolav": multi_policy_colav_env_name,
}


def resolve_environment_name(env_type):
    """
    Resolve a configured environment selector to a registered name.

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
    """
    Resolve one or more configured environment selectors to canonical names.

    Accepts either a single environment selector string or a list/tuple of
    selectors, and returns the canonical environment names in the exact order
    they were specified.

    Args:
        env_type: A single environment selector string, or a list/tuple of them.

    Returns:
        A list of canonical registered environment names.

    Raises:
        EnvironmentRegistryError: If any selector does not match a known
            environment name or alias.
    """

    if isinstance(env_type, (list, tuple)):
        return [resolve_environment_name(name) for name in env_type]
    return [resolve_environment_name(env_type)]


def build_default_environment(environment_name, framework_config=None):
    """
    Instantiate the configured default environment.

    Extracts a scoped configuration dict for the requested environment from
    *framework_config* so that each environment receives only the keys it
    actually needs rather than the entire framework config.

    Args:
        environment_name: Canonical environment name returned by
            `resolve_environment_name`.
        framework_config: The already-resolved framework configuration dict
            (from `get_config_parameters()`).  Required for all non-Lunar,
            non-Polar environments; pass `None` only when those environments
            are requested without a live config (rare / test scenarios).

    Returns:
        A registered environment instance ready for training or visualization.
    """

    if environment_name == airsim_env_name:
        cfg = build_airsim_config(framework_config, training_mode=True) if framework_config else {"training_mode": True}
        return get(environment_name, configuration=cfg)

    if environment_name in (py_sim_env_name, multi_sim_env_name):
        cfg = build_pysim_config(framework_config) if framework_config else {}
        return get(environment_name, configuration=cfg)

    if environment_name == fh_sim_env_name:
        cfg = build_fhsim_config(framework_config) if framework_config else {}
        return get(environment_name, configuration=cfg)

    if environment_name == sync_sim_env_name:
        cfg = build_fhsimsync_config(framework_config) if framework_config else {}
        return get(environment_name, configuration=cfg)

    if environment_name == py_sim_colav_env_name:
        cfg = build_colav_config(framework_config) if framework_config else {}
        return get(environment_name, configuration=cfg)

    if environment_name == multi_policy_colav_env_name:
        cfg = build_multi_policy_colav_config(framework_config) if framework_config else {}
        return get(environment_name, configuration=cfg)

    # LunarLander and PySimPolarEnv need no framework-level config keys.
    return get(environment_name, configuration={})


def initialize_all_environments():
    """
    Trigger registration of all supported environment classes.

    Because registration is now driven purely by the `ENVIRONMENT_NAME`
    class attribute (set at class-definition time), this function no longer
    needs to construct any instances.  Subclassing `BaseEnvironment`
    automatically calls `_Factory.register` via `__init_subclass__`, so
    by the time this function runs all classes defined in this module are
    already registered.

    The function is kept as an explicit call-site in `main.py` to make the
    registration step visible and to allow future pre-flight checks (e.g.
    verifying that expected environment names are present in the registry).
    """

    expected = {
        py_sim_env_name, fh_sim_env_name, sync_sim_env_name, lunar_env_name,
        py_sim_polar_env_name, multi_sim_env_name,
        py_sim_colav_env_name, multi_policy_colav_env_name,
    }
    missing = expected - set(_Factory._environment_classes)
    if missing:
        raise EnvironmentRegistryError(
            f"Expected environments not registered: {', '.join(sorted(missing))}. "
            f"Ensure all environment classes are imported before calling "
            f"initialize_all_environments()."
        )
