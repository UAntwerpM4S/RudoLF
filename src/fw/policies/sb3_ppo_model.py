import gymnasium as gym

from typing import Optional
from fw.stop_condition import StopCondition
from fw.policies.base_model import BaseModel


class SB3PPOModel(BaseModel):
    """
    Proximal Policy Optimization (PPO) Model for reinforcement learning.

    This class handles the training, evaluation, and policy management for PPO-based reinforcement learning agents.
    """

    def __init__(
        self,
        environment: gym.Env,
        eval_frequency: int,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        num_epochs: int = 10,
        normalize: bool = False,
        max_nbr_iterations: int = 125000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize PPOModel with environment and hyperparameters.

        This initializes the PPO model with the specified environment and learning parameters.

        Args:
            environment (gym.Env): Gym-like environment for training.
            eval_frequency (int): Frequency of evaluation during training.
            learning_rate (float, optional): Learning rate for the optimizer (default: 3e-4).
            clip_range (float, optional): Clipping range for PPO loss (default: 0.2).
            value_loss_coef (float, optional): Coefficient for value loss in total loss (default: 0.5).
            max_grad_norm (float, optional): Maximum gradient norm for clipping (default: 0.5).
            gamma (float, optional): Discount factor for rewards (default: 0.99).
            gae_lambda (float, optional): GAE parameter (default: 0.95).
            entropy_coef (float, optional): Coefficient for entropy bonus (default: 0.01).
            num_epochs (int, optional): Number of epochs for training on collected data (default: 10).
            normalize (bool, optional): Whether to normalize the rewards between [-1,1] (default: False).
            max_nbr_iterations (int, optional): Maximum number of steps that is allowed before aborting (default: 125000).
            batch_size (int, optional): Size of the (mini) batches (default: 64).
            device (str, optional): Device for computation ('cpu' or 'cuda') (default: 'cpu').
        """
        super().__init__(environment, eval_frequency, learning_rate, clip_range, value_loss_coef, max_grad_norm, gamma,
                         gae_lambda, entropy_coef, num_epochs, normalize, max_nbr_iterations, batch_size, device)


    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
        num_envs: int = 1,  # Number of parallel environments
        callback: Optional[callable] = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Train the policy using multiple parallel environments.

        This method runs the training loop for the specified number of timesteps using parallel environments.
        It collects experience from the environments, updates the policy, logs training metrics, and saves
        model checkpoints at regular intervals.

        Args:
            stop_condition (StopCondition): Condition that defines when training should stop.
            num_envs (int): Number of parallel environments. Default is 1.
            callback (callable, optional): The callback handler for custom actions during training.
            log_interval (int): The interval at which to log training information. Default is 1.
            tb_log_name (str): The name of the tensorboard log file. Default is "OnPolicyAlgorithm".
            reset_num_timesteps (bool): Whether to reset the number of timesteps after each training run.
                Default is True.
            progress_bar (bool): Whether to display a progress bar. Default is False.

        """
        pass
