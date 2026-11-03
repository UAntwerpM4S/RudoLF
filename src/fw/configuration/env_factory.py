import sys
import types
import logging
import numpy as np

from typing import Any
from collections.abc import Mapping
from fw.dynamics.dynamics_model import Model
from fw.environments.py_sim_env import PySimEnv
# from fw.environments.fh_sim_env import FhSimEnv
from airsim.environments.airsim_env import AirSimEnv
from fw.environments.lunar_lander import LunarLander
from fw.environments.multi_sim_env import MultiSimEnv
from fw.environments.py_sim_colav_env import PySimColavEnv
from fw.environments.action_repeat_wrapper import ActionRepeatWrapper
from fw.common.tools.config_utils import normalize_configuration, get_validated_bool
from fw.simulators.sync_sim_bridge import SimSyncBridge
from fw.simulators.fh_sim_bridge import FhSimBridge

from airsim.config.airsim_config import (
    DEFAULT_AIRSIM_SAVE_SLOT,
    DEFAULT_AIRSIM_SIMULATION_SPEED,
)
from fw.common.default_positions import (
    DEFAULT_FRAMEWORK_SHIP_POS,
    DEFAULT_FRAMEWORK_TARGET_POS,
)

logger = logging.getLogger(__name__)


class EnvironmentRegistryError(Exception):
    """ General environment registry error """


class _Factory:
    """
    Registry of environment classes; caches one instance per environment
    (unless the class sets `_cache_instance = False`).
    """

    _environment_classes = {}
    _environment_instances = {}


    @classmethod
    def register(cls, environment_class):
        """
        Register an environment class under its `ENVIRONMENT_NAME` attribute.

        Called automatically when a BaseEnvironment subclass is defined.

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
        except KeyError:  # no clash
            cls._environment_classes[environment_name] = environment_class
            logger.debug("Successfully registered new environment class %s.", environment_name)
        else:
            registered_names = ', '.join([f"'{name}'" for name in cls._environment_classes.keys()])
            logger.error("Environment name '%s' not unique. Already registered: %s", environment_name, registered_names)

            raise EnvironmentRegistryError(f"An environment class called '{environment_name}' is already defined. "
                                           f"Class names must be unique.")


    @classmethod
    def get(cls, environment_name, configuration=None):
        """
        Return the environment instance for a name, creating it if needed.

        Args:
            environment_name (str): The name of the environment to retrieve.
            configuration (dict, optional): Required when the instance is created;
                must match the cached instance's configuration otherwise.

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
            instance = cls._environment_instances[environment_name]
        except KeyError:
            if normalized_configuration is None:
                raise EnvironmentRegistryError("A configuration must be provided when instantiating an environment "
                                               "for the first time")

            instance = environment_class(environment_name, normalized_configuration)
            cls._environment_instances[environment_name] = instance

        else:
            # Returning a cached instance: refuse a conflicting configuration.
            if normalized_configuration is not None and normalized_configuration != instance.configuration:
                raise EnvironmentRegistryError(f"A configuration was now provided for '{environment_name}' "
                                               f"which clashes with the configuration used with the environment "
                                               f"previously. Try again without providing a configuration.")

        assert len(cls._environment_instances) <= len(cls._environment_classes), "More environment instances than classes!"
        return instance


class BaseEnvironment:
    """
    Base class for environment wrappers.

    Subclasses are registered automatically unless their class name starts with '_'.
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
        self._env = None  # the wrapped gym environment


    def __init_subclass__(cls):
        """
        Register each subclass whose class name does not start with '_'.
        """

        environment_name = cls.__name__

        if not environment_name.startswith('_'):
            _Factory.register(cls)


    def __copy__(self):
        """
        Return self: environment wrappers are shared, never copied.
        """

        return self


    def __deepcopy__(self, _):
        """
        Return self: environment wrappers are shared, never copied.
        """

        return self


    @property
    def gym_env(self):
        """
        The wrapped gym environment.

        Raises:
            SystemError: If the environment has not been created yet.
        """

        if self._env:
            return self._env
        else:
            raise SystemError('No environment created yet!')


    @property
    def env(self):
        """
        The wrapped gym env. Assigning it stamps the wrapper's registered name
        onto the env as `training_name`, a stable key for policies/metrics/configs
        even when several wrappers share the same gym class.
        """

        return self._env


    @env.setter
    def env(self, value):
        """
        Set the wrapped env and stamp its `training_name`.
        """

        self._env = value

        if value is not None:
            try:
                value.training_name = self._name
            except AttributeError:
                # Not a BaseEnv (e.g. a VecEnv wrapper): nothing to stamp.
                pass


    @property
    def configuration(self) -> Mapping[str, Any]:
        """
        Read-only view of the configuration (MappingProxyType for mappings,
        otherwise the normalized value as is).
        """

        if isinstance(self._configuration, Mapping):
            return types.MappingProxyType(self._configuration)
        return self._configuration


    def __repr__(self):
        """
        Return the environment's name.
        """

        return self._name


class PySimEnvironment(BaseEnvironment):
    """
    Wrapper for the Python Simulator environment.
    """

    ENVIRONMENT_NAME = "PySimEnv"

    def __init__(self, name, configuration):
        """
        Create the PySim environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_pysim_config`).
        """

        super().__init__(name, configuration)

        # Empty configuration: registry placeholder only, nothing to build.
        if not self.configuration:
            return

        # TODO: start/target positions should eventually be generated by the framework.
        ship_pos = np.array(self.configuration["start_pos"], dtype=np.float32)
        target_pos = np.array(self.configuration["target_pos"], dtype=np.float32)

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self.env = DummyVecEnv([lambda: Monitor(PySimEnv(render_mode='human', max_steps=2000, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        self.env = PySimEnv(time_step=2.0, max_steps=14000, start_pos=ship_pos, target_pos=target_pos,
                            overall_contour=self.configuration["overall_contour"],
                            depth_contour=self.configuration["depth_contour"],
                            trajectory=self.configuration["trajectory"],
                            dynamics_model=self.configuration["dynamics_model"],
                            ship_model=self.configuration["ship_model"],
                            enable_bank_collision_check=self.configuration["enable_bank_collision_check"],
                            enable_ukc_check=self.configuration["enable_ukc_check"],
                            wind=self.configuration["wind"],
                            current=self.configuration["current"],
                            ua_surrogate_config=self.configuration.get("ua_surrogate_config", {}),
                            ugent_surrogate_config=self.configuration.get("ugent_surrogate_config", {}),
                            ugent_b01_surrogate_config=self.configuration.get("ugent_b01_surrogate_config", {}),
                            reward_name=self.configuration.get("reward_name", "default"),
                            reward_weights=self.configuration.get("reward_weights", None))


class MultiPySimEnvironment(BaseEnvironment):
    """
    Wrapper for the multi-agent Python Simulator environment.
    """

    ENVIRONMENT_NAME = "MultiSimEnv"

    def __init__(self, name, configuration):
        """
        Create the multi-agent wrapper (two PySimEnv sub-environments).

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_pysim_config`).
        """

        super().__init__(name, configuration)

        # Empty configuration: registry placeholder only, nothing to build.
        if not self.configuration:
            return

        # TODO: start/target positions should eventually be generated by the framework.
        ship_pos = np.array(self.configuration["start_pos"], dtype=np.float32)
        target_pos = np.array(self.configuration["target_pos"], dtype=np.float32)

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self.env = DummyVecEnv([lambda: Monitor(PySimEnv(render_mode='human', max_steps=2000, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        single_pysim_env1 = PySimEnv(time_step=2.0, max_steps=14000, start_pos=ship_pos, target_pos=target_pos,
                             overall_contour=self.configuration["overall_contour"],
                             depth_contour=self.configuration["depth_contour"],
                             trajectory=self.configuration["trajectory"],
                             dynamics_model=self.configuration["dynamics_model"],
                             ship_model=self.configuration["ship_model"],
                             enable_bank_collision_check=self.configuration["enable_bank_collision_check"],
                             enable_ukc_check=self.configuration["enable_ukc_check"],
                             wind=self.configuration["wind"],
                             current=self.configuration["current"],
                             ua_surrogate_config=self.configuration.get("ua_surrogate_config", {}),
                             ugent_surrogate_config=self.configuration.get("ugent_surrogate_config", {}),
                             ugent_b01_surrogate_config=self.configuration.get("ugent_b01_surrogate_config", {}),
                             reward_name=self.configuration.get("reward_name", "default"),
                             reward_weights=self.configuration.get("reward_weights", None))

        single_pysim_env2 = PySimEnv(time_step=2.0, max_steps=14000, start_pos=ship_pos, target_pos=target_pos,
                             overall_contour=self.configuration["overall_contour"],
                             depth_contour=self.configuration["depth_contour"],
                             trajectory=self.configuration["trajectory"],
                             dynamics_model=self.configuration["dynamics_model"],
                             ship_model=self.configuration["ship_model"],
                             enable_bank_collision_check=self.configuration["enable_bank_collision_check"],
                             enable_ukc_check=self.configuration["enable_ukc_check"],
                             wind=self.configuration["wind"],
                             current=self.configuration["current"],
                             ua_surrogate_config=self.configuration.get("ua_surrogate_config", {}),
                             ugent_surrogate_config=self.configuration.get("ugent_surrogate_config", {}),
                             ugent_b01_surrogate_config=self.configuration.get("ugent_b01_surrogate_config", {}),
                             reward_name=self.configuration.get("reward_name", "default"),
                             reward_weights=self.configuration.get("reward_weights", None))

        self.env = MultiSimEnv(sub_environments=[single_pysim_env1, single_pysim_env2])


class PySimColavEnvironment(BaseEnvironment):
    """
    Class for a single PySimColav environment (one COLREGs scenario type).
    """

    ENVIRONMENT_NAME = "PySimColavEnv"

    def __init__(self, name, configuration):
        """
        Create the PySimColav environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_colav_config`).
        """

        super().__init__(name, configuration)

        # Empty configuration: registry placeholder only, nothing to build.
        if not self.configuration:
            return

        ship_pos = np.array(self.configuration["start_pos"], dtype=np.float32)
        target_pos = np.array(self.configuration["target_pos"], dtype=np.float32)

        self.env = PySimColavEnv(
            time_step=1.0,
            max_steps=1500,
            ship_pos=ship_pos,
            target_pos=target_pos,
            overall_contour=self.configuration["overall_contour"],
            depth_contour=self.configuration["depth_contour"],
            trajectory=self.configuration["trajectory"],
            dynamics_model=self.configuration["dynamics_model"],
            ship_model=self.configuration["ship_model"],
            wind=self.configuration["wind"],
            current=self.configuration["current"],
            ugent_surrogate_config=self.configuration.get("ugent_surrogate_config", {}),
            ugent_b01_surrogate_config=self.configuration.get("ugent_b01_surrogate_config", {}),
            enable_obstacles=self.configuration.get("enable_obstacles", True),
        )


class MultiPolicyColavEnvironment(BaseEnvironment):
    """
    PySimColavEnv for MultiPolicySingleAgent: the observation gets a one-hot
    scenario vector so the agent can route each step to the matching policy.
    """

    ENVIRONMENT_NAME = "MultiPolicyColavEnv"

    def __init__(self, name, configuration):
        """
        Create the MultiPolicyColav environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see
                `build_multi_policy_colav_config`). evaluate,
                include_scenario_one_hot and minimal_path_render are hardcoded
                pending the planned eval rework.
        """

        super().__init__(name, configuration)

        # Empty configuration: registry placeholder only, nothing to build.
        if not self.configuration:
            return

        ship_pos = np.array(self.configuration["start_pos"], dtype=np.float32)
        target_pos = np.array(self.configuration["target_pos"], dtype=np.float32)

        self.env = PySimColavEnv(
            time_step=1.0,
            max_steps=1500,
            ship_pos=ship_pos,
            target_pos=target_pos,
            overall_contour=self.configuration["overall_contour"],
            depth_contour=self.configuration["depth_contour"],
            trajectory=self.configuration["trajectory"],
            dynamics_model=self.configuration["dynamics_model"],
            ship_model=self.configuration["ship_model"],
            wind=self.configuration["wind"],
            current=self.configuration["current"],
            ugent_surrogate_config=self.configuration.get("ugent_surrogate_config", {}),
            ugent_b01_surrogate_config=self.configuration.get("ugent_b01_surrogate_config", {}),
            enable_obstacles=self.configuration.get("enable_obstacles", True),
            evaluate=False,
            # eval_scenario/eval_obstacle_count are not PySimColavEnv parameters yet.
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
        Create a PySimEnv driven by the FH simulator (FhSimBridge).

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_fhsim_config`).
        """

        super().__init__(name, configuration)

        # Empty configuration: registry placeholder only, nothing to build.
        if not self.configuration:
            return

        # TODO: start/target positions should eventually be generated by the framework.
        ship_pos = np.array(self.configuration["start_pos"], dtype=np.float32)
        target_pos = np.array(self.configuration["target_pos"], dtype=np.float32)

        # Wrap the environment in a Monitor and a DummyVecEnv
        # self.env = DummyVecEnv([lambda: Monitor(FhSimEnv(render_mode='human', max_steps=1200, verbose=True,
        #                                                   ship_pos=ship_pos, target_pos=target_pos))])

        dynamics_model = self.configuration["dynamics_model"]
        if dynamics_model != Model.FH_TABULAR:
            logger.warning(
                "Using FhSimEnv with a %s dynamics model is not recommended, "
                "use %s instead.", dynamics_model, Model.FH_TABULAR
            )

        fh_bridge = FhSimBridge(
            ship_model=self.configuration["ship_model"],
            scenario=self.configuration["scenario"],
        )

        self.env = PySimEnv(time_step=2.0, max_steps=135000, start_pos=ship_pos, target_pos=target_pos,
                            overall_contour=self.configuration["overall_contour"],
                            depth_contour=self.configuration["depth_contour"],
                            trajectory=self.configuration["trajectory"],
                            ship_model=self.configuration["ship_model"],
                            enable_bank_collision_check=self.configuration["enable_bank_collision_check"],
                            enable_ukc_check=self.configuration["enable_ukc_check"],
                            scenario=self.configuration["scenario"],
                            dynamics_model=dynamics_model,
                            wind=self.configuration["wind"],
                            current=self.configuration["current"],
                            ua_surrogate_config=self.configuration.get("ua_surrogate_config", {}),
                            ugent_surrogate_config=self.configuration.get("ugent_surrogate_config", {}),
                            ugent_b01_surrogate_config=self.configuration.get("ugent_b01_surrogate_config", {}),
                            reward_name=self.configuration.get("reward_name", "default"),
                            reward_weights=self.configuration.get("reward_weights", None),
                            fh_bridge=fh_bridge)


class FhSimSyncEnvironment(BaseEnvironment):
    """
    Realtime variant of FhSimEnvironment: drives a running simSync instance
    over UDP (SimSyncBridge) instead of a local FH math model. The env
    interface, observations and rewards are the same.
    """

    ENVIRONMENT_NAME = "FhSimSyncEnv"

    def __init__(self, name, configuration):
        """
        Create the realtime simSync environment wrapper.

        Args:
            name: Registered environment name.
            configuration: Scoped configuration dict (see `build_fhsimsync_config`).
                Optional keys (with defaults):
                    remote_ip, remote_port: simSync endpoint to connect to.
                    local_ip, local_port:   local UDP endpoint to bind.
                    connect_timeout:        seconds to wait for the first state event.
                    step_timeout_per_tick:  seconds to wait per sim tick during step().
        """

        super().__init__(name, configuration)

        # Empty configuration: registry placeholder only, nothing to build.
        if not self.configuration:
            return

        # Start/target stay framework-side (observations, reward); the vessel
        # state itself lives in simSync.
        ship_pos = np.array(self.configuration["start_pos"], dtype=np.float32)
        target_pos = np.array(self.configuration["target_pos"], dtype=np.float32)

        fh_bridge = SimSyncBridge(
            ship_model=self.configuration["ship_model"],
            remote_ip=self.configuration.get("remote_ip", "127.0.0.1"),
            remote_port=self.configuration.get("remote_port", 5000),
            local_ip=self.configuration.get("local_ip", "127.0.0.1"),
            local_port=self.configuration.get("local_port", 5001),
            connect_timeout=self.configuration.get("connect_timeout", 10.0),
            step_timeout_per_tick=self.configuration.get("step_timeout_per_tick", 1.0),
        )

        self.env = PySimEnv(
            time_step=2.0,
            max_steps=135000,
            start_pos=ship_pos,
            target_pos=target_pos,
            scenario=self.configuration["scenario"],
            dynamics_model=self.configuration["dynamics_model"],
            overall_contour=self.configuration["overall_contour"],
            depth_contour=self.configuration["depth_contour"],
            trajectory=self.configuration["trajectory"],
            ship_model=self.configuration["ship_model"],
            enable_bank_collision_check=self.configuration["enable_bank_collision_check"],
            enable_ukc_check=self.configuration["enable_ukc_check"],
            wind=self.configuration["wind"],
            current=self.configuration["current"],
            ua_surrogate_config=self.configuration.get("ua_surrogate_config", {}),
            ugent_surrogate_config=self.configuration.get("ugent_surrogate_config", {}),
            ugent_b01_surrogate_config=self.configuration.get("ugent_b01_surrogate_config", {}),
            reward_name=self.configuration.get("reward_name", "default"),
            reward_weights=self.configuration.get("reward_weights", None),
            fh_bridge=fh_bridge,
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
            configuration: Scoped AirSim configuration dict (see
                `build_airsim_config`); missing keys use the defaults below.

        Raises:
            ImportError: If the airsim package is not installed.
            ConnectionError: If UE5 / CosysAirSim is not reachable.
        """

        super().__init__(name, configuration)

        ship_pos = list(DEFAULT_FRAMEWORK_SHIP_POS)
        target_pos = list(DEFAULT_FRAMEWORK_TARGET_POS)

        # training_mode=True: UE5 rendering off, fast frame-stepping.
        # training_mode=False: rendering on, real-time stepping.
        cfg = configuration if isinstance(configuration, dict) else {}
        training_mode = get_validated_bool(cfg, "training_mode", True)
        no_rendering_mode = get_validated_bool(cfg, "no_rendering_mode", False)
        use_pause = get_validated_bool(cfg, "use_pause_stepping", training_mode)
        max_steps = int(cfg.get("max_steps", 14000))
        ip = cfg.get("ip", "")
        port = int(cfg.get("port", 41451))
        use_airsim = get_validated_bool(cfg, "use_airsim", True)
        timeout = float(cfg.get("timeout", 10.0))
        max_retries = int(cfg.get("max_retries", 3))
        health_check_interval = float(cfg.get("health_check_interval", 30.0))
        csv_log_path = str(cfg.get("csv_log_path", ""))
        save_slot = str(cfg.get("save_slot", DEFAULT_AIRSIM_SAVE_SLOT))
        load_world_on_start = cfg.get("load_world_on_start", None)
        capture_live_start_pose = cfg.get("capture_live_start_pose", None)
        simulation_speed = float(cfg.get("simulation_speed", DEFAULT_AIRSIM_SIMULATION_SPEED))
        draw_calibration_reference = get_validated_bool(cfg, "draw_calibration_reference", True)
        episode_log_path = str(cfg.get("episode_log_path", ""))
        normalize_reward = get_validated_bool(cfg, "normalize_reward", True)
        reward_normalization_scale = float(cfg.get("reward_normalization_scale", 5.0))
        reset_mode = str(cfg.get("reset_mode", "rewind_pose"))
        progress_log_interval_steps = int(cfg.get("progress_log_interval_steps", 0))
        log_pause_watchdog_warnings = get_validated_bool(cfg, "log_pause_watchdog_warnings", False)
        use_sim_time_stepping = get_validated_bool(cfg, "use_sim_time_stepping", False)
        sim_time_per_action = float(cfg.get("sim_time_per_action", cfg.get("time_step", 0.1)))
        frames_per_step = int(cfg.get("frames_per_step", 8))
        time_step = float(cfg.get("time_step", 0.1))
        physics_update_timeout = float(cfg.get("physics_update_timeout", 1.0))
        state_poll_interval = float(cfg.get("state_poll_interval", 0.005))
        latency_warning_threshold = cfg.get("latency_warning_threshold", None)
        curriculum_enabled = get_validated_bool(cfg, "curriculum_enabled", False)
        curriculum_window_size = int(cfg.get("curriculum_window_size", 8))
        curriculum_target_checkpoint = cfg.get("curriculum_target_checkpoint", None)
        curriculum_frontier_increment = int(cfg.get("curriculum_frontier_increment", 4))
        curriculum_window_increment = int(cfg.get("curriculum_window_increment", 6))
        curriculum_success_patience = int(cfg.get("curriculum_success_patience", 2))
        curriculum_randomization_scale = float(cfg.get("curriculum_randomization_scale", 0.0))
        guidance_assist_enabled = get_validated_bool(cfg, "guidance_assist_enabled", False)
        guidance_residual_steering_scale = float(cfg.get("guidance_residual_steering_scale", 0.45))
        guidance_residual_throttle_scale = float(cfg.get("guidance_residual_throttle_scale", 0.55))
        control_interface = str(cfg.get("control_interface", "car_physics"))
        observation_profile = str(cfg.get("observation_profile", "ppo_route_relative_car"))

        self.env = AirSimEnv(
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

        # Optional action repeat (config key 'action_repeat').
        repeat_steps = int(cfg.get("action_repeat", 1))
        if repeat_steps > 1:
            self.env = ActionRepeatWrapper(self.env, repeat=repeat_steps)


class LunarEnvironment(BaseEnvironment):
    """
    Wrapper for the LunarLander environment.
    """

    ENVIRONMENT_NAME = "LunarLander"

    def __init__(self, name, configuration):
        """
        Create the continuous LunarLander wrapper.

        Args:
            name: Registered environment name.
            configuration: Ignored apart from being stored.
        """

        super().__init__(name, configuration)

        self.env = LunarLander(continuous=True)


# ---------------------------------------------------------------------------
# Environment-scoped configuration builders
#
# Each builder passes only the keys an environment needs, so environments do
# not depend on unrelated settings. reward_name/reward_weights are passed
# through so the evaluate_reward workflow can override them per config.
# ---------------------------------------------------------------------------

def build_pysim_config(framework_config: dict, extra: dict | None = None) -> dict:
    """
    Extract PySim-relevant keys from *framework_config*.

    Args:
        framework_config: Full resolved framework configuration dict.
        extra: Optional overrides merged on top (e.g. reward_name/weights
            from the evaluate_reward workflow).

    Returns:
        The keys used by PySimEnvironment and MultiPySimEnvironment.
    """

    cfg = {
        "wind":                         framework_config["wind"],
        "current":                      framework_config["current"],
        "ship_model":                   framework_config["ship_model"],
        "dynamics_model":               framework_config["dynamics_model"],
        "start_pos":                    framework_config["start_pos"],
        "target_pos":                   framework_config["target_pos"],
        "overall_contour":              framework_config["overall_contour"],
        "depth_contour":                framework_config["depth_contour"],
        "trajectory":                   framework_config["trajectory"],
        "enable_bank_collision_check":  get_validated_bool(framework_config, "enable_bank_collision_check", True),
        "enable_ukc_check":             get_validated_bool(framework_config, "enable_ukc_check", True),
        "ua_surrogate_config":          framework_config["ua_surrogate_config"],
        "ugent_surrogate_config":       framework_config.get("ugent_surrogate_config", {}),
        "ugent_b01_surrogate_config":    framework_config.get("ugent_b01_surrogate_config", {}),
        "reward_weights":               framework_config.get("reward_weights", None),
        "reward_name":                  framework_config.get("reward_name", "default"),
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
        The keys used by PySimColavEnvironment.
    """

    cfg = {
        "wind":                framework_config["wind"],
        "current":             framework_config["current"],
        "ship_model":          framework_config["ship_model"],
        "dynamics_model":      framework_config["dynamics_model"],
        "start_pos":           framework_config["start_pos"],
        "target_pos":          framework_config["target_pos"],
        "overall_contour":     framework_config["overall_contour"],
        "depth_contour":       framework_config["depth_contour"],
        "trajectory":          framework_config["trajectory"],
        "enable_obstacles":    framework_config.get("enable_obstacles", True),
        "ugent_surrogate_config": framework_config.get("ugent_surrogate_config", {}),
        "ugent_b01_surrogate_config": framework_config.get("ugent_b01_surrogate_config", {}),
    }
    if extra:
        cfg.update(extra)
    return cfg


def build_multi_policy_colav_config(framework_config: dict, extra: dict | None = None) -> dict:
    """
    Extract MultiPolicyColav-relevant keys from *framework_config*.

    Currently identical to `build_colav_config` (the eval-related fields are
    still hardcoded in MultiPolicyColavEnvironment).

    Args:
        framework_config: Full resolved framework configuration dict.
        extra: Optional overrides merged on top.

    Returns:
        The keys used by MultiPolicyColavEnvironment.
    """

    return build_colav_config(framework_config, extra)


def build_fhsim_config(framework_config: dict) -> dict:
    """
    Extract FhSim-relevant keys from *framework_config*.

    Args:
        framework_config: Full resolved framework configuration dict.

    Returns:
        The keys used by FhSimEnvironment.
    """

    return {
        "ship_model":                   framework_config["ship_model"],
        "scenario":                     framework_config["scenario"],
        "dynamics_model":               framework_config["dynamics_model"],
        "start_pos":                    framework_config["start_pos"],
        "target_pos":                   framework_config["target_pos"],
        "overall_contour":              framework_config["overall_contour"],
        "depth_contour":                framework_config["depth_contour"],
        "trajectory":                   framework_config["trajectory"],
        "enable_bank_collision_check":  get_validated_bool(framework_config, "enable_bank_collision_check", True),
        "enable_ukc_check":             get_validated_bool(framework_config, "enable_ukc_check", True),
        "wind":                         framework_config["wind"],
        "current":                      framework_config["current"],
        "ua_surrogate_config":          framework_config["ua_surrogate_config"],
        "ugent_surrogate_config":       framework_config.get("ugent_surrogate_config", {}),
        "ugent_b01_surrogate_config":    framework_config.get("ugent_b01_surrogate_config", {}),
        "reward_weights":               framework_config.get("reward_weights", None),
        "reward_name":                  framework_config.get("reward_name", "default"),
    }


def build_fhsimsync_config(framework_config: dict) -> dict:
    """
    Extract FhSimSync-relevant keys from *framework_config*.

    Args:
        framework_config: Full resolved framework configuration dict.

    Returns:
        The keys used by FhSimSyncEnvironment (as build_fhsim_config), plus the
        optional UDP/timing overrides (remote_ip, remote_port, local_ip,
        local_port, connect_timeout, step_timeout_per_tick) when present.
    """

    cfg = {
        "ship_model":                   framework_config["ship_model"],
        "scenario":                     framework_config["scenario"],
        "dynamics_model":               framework_config["dynamics_model"],
        "start_pos":                    framework_config["start_pos"],
        "target_pos":                   framework_config["target_pos"],
        "overall_contour":              framework_config["overall_contour"],
        "depth_contour":                framework_config["depth_contour"],
        "trajectory":                   framework_config["trajectory"],
        "wind":                         framework_config["wind"],
        "current":                      framework_config["current"],
        "enable_bank_collision_check":  get_validated_bool(framework_config, "enable_bank_collision_check", True),
        "enable_ukc_check":             get_validated_bool(framework_config, "enable_ukc_check", True),
        "ua_surrogate_config":          framework_config["ua_surrogate_config"],
        "ugent_surrogate_config":       framework_config.get("ugent_surrogate_config", {}),
        "ugent_b01_surrogate_config":    framework_config.get("ugent_b01_surrogate_config", {}),
        "reward_weights":               framework_config.get("reward_weights", None),
        "reward_name":                  framework_config.get("reward_name", "default"),
    }
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

    Uses the `airsim` sub-dict when present, else the flat framework config.
    Missing keys fall back to the AirSimEnvironment defaults.

    Args:
        framework_config: Full resolved framework configuration dict.
        training_mode: Value written to 'training_mode', overriding the config.
            Pass False for visualization/inference runs.

    Returns:
        A copy of the AirSim settings with 'training_mode' set.
    """

    # Current configs keep AirSim settings at the top level of CONFIG_PARAMETERS.
    airsim_cfg = framework_config.get("airsim") or framework_config
    cfg = dict(airsim_cfg)
    cfg["training_mode"] = training_mode
    return cfg


def _pick_airsim_fallback(configuration) -> str:
    """
    Return the canonical name of the first non-AirSim environment in `env_type`.

    Args:
        configuration: Framework config dict to read `env_type` from; when
            None/empty, the config is loaded via get_config_parameters().

    Returns:
        Canonical environment name to use as a fallback.

    Side-effects:
        Calls `sys.exit(1)` if AirSimEnv is the only configured environment.
    """

    if configuration and isinstance(configuration, dict):
        config = configuration
    else:
        from fw.configuration.config import get_config_parameters
        config = get_config_parameters()

    env_types = config.get("env_type", [])
    if isinstance(env_types, str):
        env_types = [env_types]

    other_envs = [
        e for e in env_types
        if str(e).strip().lower() not in ("airsimenv", "airsim")
    ]

    if not other_envs:
        logger.error("Only AirSimEnv was selected in config.py. Exiting.")
        sys.exit(1)

    fallback_raw = other_envs[0]
    fallback_name = resolve_environment_name(fallback_raw)
    logger.warning("Falling back to %s.", fallback_name)
    return fallback_name


def get(environment_name: str, configuration=None):
    """
    Retrieve (or create) an environment instance by name.

    Public entry point; delegates to `_Factory.get`. If AirSim is unavailable
    (ImportError/ConnectionError), the first other configured environment is
    built instead.

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
        logger.warning("AirSim environment unavailable (reason: %s).", exc)
        # `configuration` is AirSim-scoped and lacks keys like "wind"/"current",
        # so load the full framework config for the fallback environment.
        from fw.configuration.config import get_config_parameters as _get_fw_config  # avoids circular import
        fallback_framework_config = _get_fw_config()
        fallback_name = _pick_airsim_fallback(fallback_framework_config)
        return build_default_environment(fallback_name, framework_config=fallback_framework_config)


py_sim_env_name = "PySimEnv"
fh_sim_env_name = "FhSimEnv"
sync_sim_env_name = "FhSimSyncEnv"
airsim_env_name = "AirSimEnv"
lunar_env_name = "LunarLander"
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
    Resolve a configured environment selector (alias or name, case-insensitive).

    Args:
        env_type: Environment selector; None or empty gives PySimEnv.

    Returns:
        The canonical registered environment name.

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

    Accepts a single selector or a list/tuple; order is preserved.

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

    Each environment receives a scoped configuration built from
    `framework_config` by the matching builder.

    Args:
        environment_name: Canonical name from `resolve_environment_name`.
        framework_config: Resolved framework configuration dict. Needed for
            all environments except LunarLander; without it an empty config is
            passed (registry placeholder, e.g. in tests).

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

    # LunarLander needs no framework-level config keys.
    return get(environment_name, configuration={})


def initialize_all_environments():
    """
    Check that the expected environment classes are registered.

    Registration itself happens at class definition (`__init_subclass__`);
    this explicit call in main.py makes the step visible and fails fast.

    Raises:
        EnvironmentRegistryError: If an expected environment is missing
            (AirSimEnv is not required).
    """

    expected = {
        py_sim_env_name, fh_sim_env_name, sync_sim_env_name, lunar_env_name,
        multi_sim_env_name, py_sim_colav_env_name, multi_policy_colav_env_name,
    }
    missing = expected - set(_Factory._environment_classes)
    if missing:
        raise EnvironmentRegistryError(
            f"Expected environments not registered: {', '.join(sorted(missing))}. "
            f"Ensure all environment classes are imported before calling "
            f"initialize_all_environments()."
        )
