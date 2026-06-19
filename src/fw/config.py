from enum import Enum, auto
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
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
MULTI_POLICY_SINGLE_NAME = "MPSA"  # Multi-Policy Single Agent

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
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'clip_range': 0.2,
        'eval_frequency': 100,
        'num_envs': 1,
        'num_epochs': 8,
        'normalize': False,
        'max_nbr_iterations': 5000,
        'batch_size': 512,
        'device': "cpu",
        'model_dir': "models",
        'fixed_horizon': True,
        'steps_per_update': 8192,
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
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.03,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'clip_range': 0.2,
        'eval_frequency': 100,
        'num_envs': 3,
        'num_epochs': 4,
        'normalize': False,
        'max_nbr_iterations': 200000,
        'batch_size': 4096,
        'device': "cpu",
        'model_dir': "models"
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
        'model_dir': "models",
        'fixed_horizon': False,
        'steps_per_update': 8192,
        'steps_per_policy_update': 8192,
        # Per-pid log_std clamp. Path-following (pid 0) is floored just
        # below the natural commitment point (log_std ≈ -1.42 for
        # entropy_coef=0.025, action_dim=2). The floor is inactive until
        # the policy naturally collapses past that point, then holds
        # σ at ≈ 0.22 indefinitely so under-sampled path segments retain
        # enough exploration noise to be learned even after the rest of
        # the path has committed. Other pids are unfloored (default
        # (-20, 2)).
        'log_std_bounds': {
            0: (-1.5, 2.0),
        },
    },
}

CONFIG_PARAMETERS = {
    # Valid env_type values: "PySimEnv", "FhSimEnv", "AirSimEnv", "PySimPolarEnv", "LunarLander",
    #   "PySimColav" (single colav env),
    #   "MultiPolicyColavEnv" / "mpcolav" (single env with multi-policy switching)
    #   "pysimcolavplanner" (single env with a planner-based policy)
    # Can also be a list of environment names, e.g. ["PySimEnv", "FhSimEnv"],
    # to train or visualize across multiple environments simultaneously.
    'env_type': ["pysimcolavplanner"],
    'action': "visualize",  # Valid action values: "train", "visualize"
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
}


def get_hyperparameters(agent_name):
    """Get validated hyperparameters"""
    try:
        return DEFAULT_HYPERPARAMETERS[agent_name].copy()
    except KeyError:
        raise ValueError(f"Unknown agent type: {agent_name}")


def get_config_parameters():
    """Get  the configuration parameters"""
    return CONFIG_PARAMETERS.copy()
