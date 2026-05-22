from enum import Enum, auto
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.profile_applier import apply_profile
from fw.config_profile import ProfileRegistry
from airsim.config.airsim_config import get_airsim_config  # re-export


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

AIRSIM_SESSION_TELEMETRY_FILENAME_TEMPLATE = "{session_type}_session.csv"
AIRSIM_BRIDGE_LOG_FILENAME = "airsim_telemetry.csv"
AIRSIM_METADATA_FILENAME = "metadata.json"
AIRSIM_INFERENCE_TRAJECTORY_PLOT_FILENAME = "inference_trajectory.png"
AIRSIM_COMPARISON_TRAJECTORY_PLOT_FILENAME = "trajectory_2d_vs_3d.png"


"""
Following (hyper)parameters are for fine-tuning the agent. The key of the dictionary
that they are linked to, is the name of the environment that they're associated with.
In the long term this should be moved to a config-file which will then be read.
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
        'model_dir': "models"
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
        'verbose': True
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
        'model_dir': "models"
    },
}

AIRSIM_HYPERPARAMETER_OVERRIDES = {
    PPO_POLICY_NAME: {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_range': 0.2,
        'device': 'cuda', # AirSim training benefits from GPU acceleration, can be switched to 'cpu' if CUDA is not available or not applicable.
    },
}

CONFIG_PARAMETERS = {
    # Valid env_type values: "PySimEnv", "FhSimEnv", "AirSimEnv", "PySimPolarEnv", "LunarLander"
    # Can also be a list of environment names, e.g. ["PySimEnv", "FhSimEnv"],
    # to train or visualize across multiple environments simultaneously.
    'env_type': ["PySimEnv"],
    'profile': "default_profile",       # Possible options: "default_profile",
                                        # "myzako_pysim_to_fhsim_transfer_profile",
                                        # "tripoli_pysim_to_fhsim_transfer_profile",
                                        # "ugent_surrogate_training_profile", ...
    'action': "train",  # Valid action values: "train", "visualize", "evaluate_reward"
    'wind': False,
    'current': False,
    'numerical_damping': False,
    'visualize_best_trained': False,
    'viz_mode': VisualizationMode.COMPLEX,  # Either "simple" or "complex"
    'scenario': "DDShip_scenario1_nowindnocurrent",
    'ship_model': ShipModel.MYZAKO_37,
    'dynamics_model': Model.FOSSEN,
    'live_animation': False,
    # AirSim settings are now in airsim.config.airsim_config
    # Use get_airsim_config() to access them.

    # Evaluate reward configurations
    'reward_configs': [
        {
            "type": "PySimEnv",
            "reward_name": "default",
            "reward_weights": {
                "forward": -0.25,
                "heading_alignment": -0.5,
                "cross_track": -0.05,
                "rudder_change": -0.4,
                "rudder_angle": -0.05,
                "thrust_change": -0.1,
            },
        },
        {
            "type": "PySimEnv",
            "reward_name": "default",
            "reward_weights": {
                "forward": -0.25,
                "heading_alignment": -0.5,
                "cross_track": -0.05,
                "rudder_change": -0.4,
                "rudder_angle": -0.05,
                "thrust_change": -0.2,
            },
        },
        {
            "type": "PySimEnv",
            "reward_name": "default",
            "reward_weights": {
                "forward": -0.25,
                "heading_alignment": -0.5,
                "cross_track": -0.05,
                "rudder_change": -0.4,
                "rudder_angle": -0.05,
                "thrust_change": -0.4,
            },
        },
        {
            "type": "PySimEnv",
            "reward_name": "default",
            "reward_weights": {
                "forward": -0.25,
                "heading_alignment": -0.5,
                "cross_track": -0.05,
                "rudder_change": -0.4,
                "rudder_angle": -0.05,
                "thrust_change": -0.6,
            },
        },
    ],
}


def get_hyperparameters(agent_name, env_name=None):
    """
    Get validated hyperparameters.

    AirSim uses the base PPO tuning set. Other environments get the legacy
    PPO settings through the override map so their training behavior stays
    separate from the AirSim tuning.
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
    Return a raw copy of CONFIG_PARAMETERS without applying any profile.

    Use this when you need the plain configuration data and will handle
    profile application yourself (e.g. in get_config_parameters below, or
    in tests that want to control the profile step explicitly).
    """

    return CONFIG_PARAMETERS.copy()


def get_config_parameters() -> dict:
    """
    Return configuration parameters with the configured profile applied.

    Resolves the profile name from the config, validates it, applies it to
    the config dict, and returns the result. This is the single authoritative
    place where profile application onto the config happens — callers should
    not re-apply the profile to the config dict after calling this function.

    Returns:
        dict: A copy of CONFIG_PARAMETERS with the profile's overrides applied.
    """

    framework_config = load_config_parameters()
    profile_name = framework_config.get("profile", "default_profile")
    profile = ProfileRegistry.get(profile_name)
    profile.validate()
    apply_profile(profile, config=framework_config)

    return framework_config
