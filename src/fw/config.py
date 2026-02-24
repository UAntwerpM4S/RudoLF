from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model

PPO_POLICY_NAME = "PPO"
PPO2_POLICY_NAME = "PPO2"


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
    }
}

CONFIG_PARAMETERS = {
    'wind': False,
    'current': False,
    'numerical_damping': False,
    'visualize_best_trained': False,
    'scenario': "DDShip_scenario1_nowindnocurrent",
    'ship_model': ShipModel.MYZAKO_DEFAULT,
    'dynamics_model': Model.FOSSEN,
    'live_animation': False,
    'use_airsim': False,
    'airsim_ip': '',
    'airsim_port': 41451,
    'airsim_kp': 5.0,
    'airsim_ki': 0.5,
    'airsim_kd': 1.0, # Derivative term to dampen oscillation
    'airsim_ff': 0.5,
    'airsim_lat_gain': 0.5, # Cross-Track Error gain to correct position drift
    'airsim_speed_scale': 1.05, # 5.6 m/s / 5.0 m/s = 1.12 => 5.06 m/s cruise 
    'airsim_log_telemetry': True,
    'airsim_throttle_kp': 1.0, # Reduced from 2.0 to prevent oscillation
    'airsim_throttle_ki': 0.2, # Increased from 0.1 to help hold speed against drag
    'airsim_dist_gain': 0.1,
    'airsim_dist_ki': 0.01,
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
