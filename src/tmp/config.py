"""
Configuration file for maritime RL training.
Contains hyperparameters and training settings for different algorithms.
"""

import torch

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

ENV_CONFIG = {
    'time_step': 0.1,
    'max_steps': 1500,
    'verbose': False,
    'render_mode': None,  # Set to 'human' for visualization during training
}

# Environmental conditions for different training scenarios
SCENARIOS = {
    'basic': {
        'wind': False,
        'current': False,
        'description': 'Basic training without environmental disturbances'
    },
    'wind_only': {
        'wind': False,
        'current': False,
        'description': 'Training with wind effects only'
    },
    'current_only': {
        'wind': False,
        'current': False,
        'description': 'Training with current effects only'
    },
    'full_environment': {
        'wind': False,
        'current': False,
        'description': 'Full training with wind and current effects'
    }
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    'total_timesteps': 500000,
    'eval_freq': 10000,
    'save_freq': 50000,
    'n_eval_episodes': 10,
    'reward_threshold': 10.8,  # Stop training when this reward is reached
    'n_envs': 4,  # Number of parallel environments
    'seed': 42,
}

# =============================================================================
# ALGORITHM CONFIGURATIONS
# =============================================================================

# Common policy network architecture
POLICY_KWARGS = {
    'net_arch': [256, 256, 128],
    'activation_fn': torch.nn.ReLU,
}

# PPO (Proximal Policy Optimization) - Good for continuous control
PPO_CONFIG = {
    'policy': "MlpPolicy",
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'clip_range_vf': None,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'policy_kwargs': POLICY_KWARGS,
    'tensorboard_log': "logs/tensorboard/",
}

# SAC (Soft Actor-Critic) - Good for complex continuous control
SAC_CONFIG = {
    'policy': "MlpPolicy",
    'learning_rate': 3e-4,
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'ent_coef': 'auto',
    'policy_kwargs': POLICY_KWARGS,
    'tensorboard_log': "logs/tensorboard/",
}

# TD3 (Twin Delayed Deep Deterministic Policy Gradient) - Alternative to SAC
TD3_CONFIG = {
    'policy': "MlpPolicy",
    'learning_rate': 3e-4,
    'buffer_size': 1000000,
    'learning_starts': 100,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': (1, "step"),
    'gradient_steps': 1,
    'policy_delay': 2,
    'target_policy_noise': 0.2,
    'target_noise_clip': 0.5,
    'policy_kwargs': POLICY_KWARGS,
    'tensorboard_log': "logs/tensorboard/",
}

# =============================================================================
# HYPERPARAMETER TUNING RANGES (for Optuna)
# =============================================================================

HYPERPARAMETER_RANGES = {
    'PPO': {
        'learning_rate': (1e-5, 1e-3),
        'n_steps': [1024, 2048, 4096],
        'batch_size': [32, 64, 128, 256],
        'n_epochs': [5, 10, 20],
        'gamma': (0.95, 0.999),
        'gae_lambda': (0.9, 0.99),
        'clip_range': (0.1, 0.3),
        'ent_coef': (1e-4, 1e-1),
        'vf_coef': (0.1, 1.0),
    },
    'SAC': {
        'learning_rate': (1e-5, 1e-3),
        'batch_size': [64, 128, 256, 512],
        'tau': (0.001, 0.02),
        'gamma': (0.95, 0.999),
        'ent_coef': ['auto', 0.01, 0.1, 0.5],
    },
    'TD3': {
        'learning_rate': (1e-5, 1e-3),
        'batch_size': [64, 128, 256, 512],
        'tau': (0.001, 0.02),
        'gamma': (0.95, 0.999),
        'policy_delay': [1, 2, 3],
        'target_policy_noise': (0.1, 0.3),
        'target_noise_clip': (0.3, 0.7),
    }
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVAL_CONFIG = {
    'n_eval_episodes': 10,
    'deterministic': True,
    'render': False,
    'success_threshold': 10.5,  # Reward threshold for considering episode successful
}

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

LOGGING_CONFIG = {
    'tensorboard_log': "logs/tensorboard/",
    'log_interval': 10,
    'eval_log_path': "logs/eval",
    'monitor_path': "logs/monitor",
    'save_path': "models/",
    'progress_bar': True,
}

# =============================================================================
# SPECIFIC TRAINING RECIPES
# =============================================================================

TRAINING_RECIPES = {
    'quick_test': {
        'algorithm': 'PPO',
        'scenario': 'basic',
        'total_timesteps': 50000,
        'n_envs': 2,
        'description': 'Quick test run for debugging'
    },
    'robust_training': {
        'algorithm': 'SAC',
        'scenario': 'full_environment',
        'total_timesteps': 1000000,
        'n_envs': 8,
        'description': 'Robust training with all environmental effects'
    },
    'baseline': {
        'algorithm': 'PPO',
        'scenario': 'basic',
        'total_timesteps': 600000,
        'n_envs': 4,
        'description': 'Standard baseline training'
    },
    'advanced': {
        'algorithm': 'SAC',
        'scenario': 'full_environment',
        'total_timesteps': 500000,
        'n_envs': 6,
        'description': 'Advanced training with environmental challenges'
    }
}
