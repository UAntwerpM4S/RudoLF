from copy import deepcopy
from enum import Enum, auto
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.configuration.profile_applier import apply_profile
from fw.configuration.config_profile import ProfileRegistry
from fw.common.default_positions import (
    DEFAULT_FRAMEWORK_SHIP_POS,
    DEFAULT_FRAMEWORK_TARGET_POS,
)


class VisualizationMode(Enum):
    """
    Visualization modes for rendering simulation results.

    Attributes:
        SIMPLE: Vessel trajectory only.
        COMPLEX: Trajectory with diagnostics (cross-track error, rudder input,
            UKC, and performance metrics).
    """

    SIMPLE = auto()
    COMPLEX = auto()


class PolicyStrategy(Enum):
    """
    Defines strategies for handling policies during environment switching.

    Attributes:
        RESET_POLICY: Start with a fresh policy.
        REUSE_CURRENT_POLICY: Continue training with the current policy.
        REUSE_OTHER_POLICY: Load and use a policy trained on another environment.
    """

    RESET_POLICY = auto()
    REUSE_CURRENT_POLICY = auto()
    REUSE_OTHER_POLICY = auto()


PPO_POLICY_NAME = "PPO"
PPO2_POLICY_NAME = "PPO2"
MULTI_PPO_POLICY_NAME = "MPPO"
MULTI_POLICY_SINGLE_NAME = "MPSA"  # Multi-Policy Single Agent

AIRSIM_SESSION_TELEMETRY_FILENAME_TEMPLATE = "{session_type}_session.csv"
AIRSIM_BRIDGE_LOG_FILENAME = "airsim_telemetry.csv"
AIRSIM_METADATA_FILENAME = "metadata.json"
AIRSIM_INFERENCE_TRAJECTORY_PLOT_FILENAME = "inference_trajectory.png"
AIRSIM_COMPARISON_TRAJECTORY_PLOT_FILENAME = "trajectory_2d_vs_3d.png"


"""
Agent hyperparameters, keyed by policy name. AirSimEnv overrides are applied on top
by get_hyperparameters(). To be moved to a config file eventually.
"""

DEFAULT_HYPERPARAMETERS = {
    PPO_POLICY_NAME: {
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'clip_range': 0.05,
        'eval_frequency': 100,
        'num_envs': 1,
        'num_epochs': 10,
        'normalize': False,
        'max_nbr_iterations': 200000,
        'batch_size': 64,
        'device': "cpu",
        'rollout_device': "cpu",  # policy forward passes in rollouts; the update runs on 'device'
        'model_dir': "models",
        'seed': None,       # base seed of the worker env RNGs; None = random, logged
    },
    PPO2_POLICY_NAME: {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'clip_range': 0.2,
        'eval_frequency': 10000,
        'num_envs': 1,
        'total_time_steps': 200000,
        'log_interval': 10,
        'n_steps': 2048,
        'batch_size': 64,
        'num_epochs': 10,
        'reward_threshold': 100000.8,
        'device': "cpu",
        'model_dir': "models",
        'log_dir': "logs",
        'tensorboard_log': "logs/tensorboard",
        'verbose': True,
        'seed': None,       # base seed of the worker env RNGs; None = random, logged
    },
    MULTI_PPO_POLICY_NAME: {
        'learning_rate': 1e-4,
        'gamma': 0.95,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'clip_range': 0.05,
        'eval_frequency': 100,
        'num_envs': 1,
        'num_epochs': 10,
        'normalize': False,
        'max_nbr_iterations': 200000,
        'batch_size': 64,
        'device': "cpu",
        'rollout_device': "cpu",  # policy forward passes in rollouts; the update runs on 'device'
        'model_dir': "models",
        'seed': None,       # base seed of the worker env RNGs; None = random, logged
    },
    MULTI_POLICY_SINGLE_NAME: {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.025,
        'value_loss_coef': 0.1,
        'max_grad_norm': 0.5,
        'clip_range': 0.2,
        'eval_frequency': 10,
        'num_envs': 12,
        'num_epochs': 8,
        'normalize': False,
        'max_nbr_iterations': 5000,
        'batch_size': 2048,
        'device': "cpu",
        'rollout_device': "cpu",  # policy forward passes in rollouts; the update runs on 'device'
        'model_dir': "models",
        'steps_per_policy_update': 8192,
        'seed': None,       # base seed of the worker env RNGs; None = random, logged
    },
}

AIRSIM_HYPERPARAMETER_OVERRIDES = {
    PPO_POLICY_NAME: {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_range': 0.2,
        'device': 'cuda',  # switch to 'cpu' if CUDA is unavailable
    },
}

CONFIG_PARAMETERS = {
    # One or a list of: "PySimEnv", "FhSimEnv", "MultiSimEnv",
    # "MultiPolicyColavEnv", "AirSimEnv", "LunarLander".
    'env_type': ["PySimEnv"],
    'profile': "default_profile",       # see ProfileRegistry, e.g.
                                        # "myzako_pysim_to_fhsim_transfer_profile"
    'action': "train",  # Valid action values: "train", "visualize", "evaluate_reward"
    'log_level': "INFO",  # Valid: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    # Wind and current, both or neither: the observation gets 4 wind/current
    # features (real for FH_TABULAR and the UAntwerp *_full presets, zeros
    # otherwise); FH then loads its wind/current exercise with a random wind
    # field per episode. The *_full presets require them.
    'wind': False,
    'current': False,
    'enable_ukc_check': True,
    'enable_bank_collision_check': True,
    'visualize_best_trained': False,
    'viz_mode': VisualizationMode.COMPLEX,
    'start_pos': DEFAULT_FRAMEWORK_SHIP_POS,    # (2245.0, -47.0)
    'target_pos': DEFAULT_FRAMEWORK_TARGET_POS, # (-1559.0, 11990.0)
    'overall_contour': "env_Sche_no_scale.csv",
    'depth_contour': "env_Sche_250cm_no_scale.csv",
    'trajectory': "trajectory_points_no_scale.csv",
    'scenario': "DDShip_scenario1_nowindnocurrent",
    # UAntwerp surrogate (Model.UANTWERP). "preset" (fw/surrogate/presets.py) is
    # {tripoli,myzako}_{clean,confined,full}: clean = open water, confined = UKC
    # lookup, full = UKC + wind + current lookup (.csv.gz). Optional overrides:
    # "checkpoint", "lookup_csv". The preset's ship must equal 'ship_model' and
    # the env time step must be a multiple of 0.1 s. "env_smoothing_tau" [s]:
    # moving average of the looked-up UKC (and wind/current with "lookup";
    # with "random" also the time constant of the averaged river line, which
    # then defaults to 10 s); None = off (e.g. 10.0 for RL). "wind_current"
    # (*_full presets only, no effect otherwise): "random" = random wind and
    # current per episode (current along the averaged river line of the
    # lookup), "lookup" = from the lookup CSV at every step.
    'ua_surrogate_config': {"preset": "myzako_confined", "env_smoothing_tau": None, "wind_current": "random"},
    'ugent_surrogate_config': {"ukc_percent": 363},
    # UGent B01 model-scale surrogate (Model.UGENT_B01_SURROGATE, ShipModel.B01).
    # "ukc_percent" required (tested 379, 153, 35; interpolated in between, clamped
    # outside). Optional: "n_max_rpm", "min_bank_clearance", "max_bank_clearance";
    # see fw.dynamics.ugent_b01_surrogate.
    'ugent_b01_surrogate_config': {"ukc_percent": 379},
    'ship_model': ShipModel.MYZAKO_37,
    'dynamics_model': Model.FOSSEN,
    'live_animation': False,

    # Evaluate reward configurations
    'reward_configs': [
        {
            "type": "PySimEnv",
            "reward_name": "default",
            "reward_weights": {
                "forward": -1.25,
                "heading_alignment": -0.5,
                "cross_track": -0.05,
                "rudder_change": -0.4,
                "rudder_angle": -0.05,
                "thrust_change": 0.0,
            },
        },
        {
            "type": "PySimEnv",
            "reward_name": "default",
            "reward_weights": {
                "forward": -1.25,
                "heading_alignment": -0.5,
                "cross_track": -0.05,
                "rudder_change": -0.4,
                "rudder_angle": -0.05,
                "thrust_change": 0.0,
            },
        },
    ],
}


def get_hyperparameters(agent_name, env_name=None):
    """
    Return a copy of DEFAULT_HYPERPARAMETERS[agent_name]; for "AirSimEnv" the
    AIRSIM_HYPERPARAMETER_OVERRIDES are applied on top.

    Raises:
        ValueError: For an unknown agent name.
    """

    try:
        params = DEFAULT_HYPERPARAMETERS[agent_name].copy()
        if env_name == "AirSimEnv":
            params.update(AIRSIM_HYPERPARAMETER_OVERRIDES.get(agent_name, {}))
        return params
    except KeyError:
        raise ValueError(f"Unknown agent type: {agent_name}")


def load_config_parameters() -> dict:
    """
    Return a deep copy of CONFIG_PARAMETERS without applying a profile.
    """

    return deepcopy(CONFIG_PARAMETERS)


def get_config_parameters() -> dict:
    """
    Return a copy of CONFIG_PARAMETERS with the configured profile validated and
    applied. This is the only place the profile is applied; callers must not
    re-apply it.
    """

    framework_config = load_config_parameters()
    profile_name = framework_config.get("profile", "default_profile")
    profile = ProfileRegistry.get(profile_name)
    profile.validate()
    apply_profile(profile, config=framework_config)

    return framework_config
