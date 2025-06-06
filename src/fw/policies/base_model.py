import torch
import numpy as np
import gymnasium as gym

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from fw.stop_condition import StopCondition


class BaseModel(ABC):
    """
    Abstract base class for reinforcement learning models.

    This class provides a unified interface and shared hyperparameters
    for reinforcement learning algorithms. Subclasses must implement
    the core training and inference logic.
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
        Initialize the base model with environment and hyperparameters.

        Args:
            environment (gym.Env): The training environment.
            eval_frequency (int): Number of training iterations between evaluations.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 3e-4.
            clip_range (float, optional): PPO clipping range. Default is 0.2.
            value_loss_coef (float, optional): Coefficient for value loss. Default is 0.5.
            max_grad_norm (float, optional): Maximum gradient norm for gradient clipping. Default is 0.5.
            gamma (float, optional): Discount factor for future rewards. Default is 0.99.
            gae_lambda (float, optional): Lambda for Generalized Advantage Estimation. Default is 0.95.
            entropy_coef (float, optional): Coefficient for entropy bonus. Default is 0.01.
            num_epochs (int, optional): Number of training epochs per update. Default is 10.
            normalize (bool, optional): If True, normalize rewards to [-1, 1]. Default is False.
            max_nbr_iterations (int, optional): Maximum number of training iterations before stopping. Default is 175000.
            batch_size (int, optional): Mini-batch size. Default is 64.
            device (str, optional): Device for training, either 'cpu' or 'cuda'. Default is 'cpu'.
        """
        self.env = environment
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space

        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.normalize = normalize
        self.max_nbr_iterations = max_nbr_iterations
        self.batch_size = batch_size
        self.device = torch.device(device)


    @abstractmethod
    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Predict an action for a given state.

        Args:
            state (np.ndarray): Current state observation.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the selected action and its log probability.
        """
        pass


    @abstractmethod
    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
        num_envs: int = 1,
        callback: Optional[Callable[..., Any]] = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Train the model using collected experiences.

        Args:
            stop_condition (StopCondition, optional): A stopping criterion to terminate training early.
            num_envs (int, optional): Number of parallel environments. Default is 1.
            callback (callable, optional): Optional callback function called during training.
            log_interval (int, optional): How often to log training statistics. Default is 1.
            tb_log_name (str, optional): Name used for logging (e.g., for TensorBoard). Default is "OnPolicyAlgorithm".
            reset_num_timesteps (bool, optional): Whether to reset timestep count at start. Default is True.
            progress_bar (bool, optional): Show a progress bar during training. Default is False.
        """
        pass


    def save_policy(self, policy_file_name: str):
        """
        Optional: Save the model to a file.

        Args:
            policy_file_name (str): Name of the policy file to save.
        """
        raise NotImplementedError


    def load_policy(self, policy_file_name: str):
        """
        Optional: Load the model from a file.

        Args:
            policy_file_name (str): Name of the policy file to load.
        """
        raise NotImplementedError


    def get_env(self) -> gym.Env:
        """
        Get the training environment.

        Returns:
            gym.Env: The training environment instance.
        """
        return self.env
