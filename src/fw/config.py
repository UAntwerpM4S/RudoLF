PPO_POLICY_NAME = "PPO"
PPO2_POLICY_NAME = "PPO2"


"""
Following (hyper)parameters are for fine-tuning the model. The key of the dictionary
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
    'live_animation': False,
    'visualize_best_trained': False
}


def get_hyperparameters(model_name):
    """Get validated hyperparameters"""
    try:
        return DEFAULT_HYPERPARAMETERS[model_name].copy()
    except KeyError:
        raise ValueError(f"Unknown model type: {model_name}")


def get_config_parameters():
    """Get  the configuration parameters"""
    return CONFIG_PARAMETERS.copy()
