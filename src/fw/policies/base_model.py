import torch
import numpy as np
import gymnasium as gym

from typing import Optional
from fw.stop_condition import StopCondition


class BaseModel:
    """
    Base Model class for reinforcement learning.

    This class handles the training, evaluation, and policy management for reinforcement learning agents.
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
        max_nbr_iterations: int = 175000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize BaseModel with environment and hyperparameters.

        This initializes the base model with the specified environment and learning parameters.

        Args:
            environment (gym.Env): Gym-like environment for training.
            eval_frequency (int): Frequency of evaluation during training.
            learning_rate (float, optional): Learning rate for the optimizer (default: 3e-4).
            clip_range (float, optional): Clipping range for the loss (default: 0.2).
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
        self.env = environment
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.normalize = normalize
        self.max_nbr_iterations = max_nbr_iterations
        self.batch_size = batch_size

        # Training parameters
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate


    def predict(self, state: np.ndarray) -> tuple:
        """
        Predict an action for a given state using the policy.

        Args:
            state (np.ndarray): Current state as a NumPy array.

        Returns:
            tuple: A tuple containing the predicted action and log probability.
        """
        raise NotImplementedError


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
        raise NotImplementedError


    def get_env(self) -> gym.Env:
        """
        Get the training environment.

        This method returns the current training environment, which is used
        during policy training.

        Returns:
            gym.Env: The training environment instance.

        """
        return self.env
