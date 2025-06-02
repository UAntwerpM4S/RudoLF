PPO_POLICY_NAME = "PPO"


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
        'max_nbr_iterations': 175000,
        'batch_size': 64,
        'device': "cpu"
    }
}

CONFIG_PARAMETERS = {
    'live_animation': False
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
