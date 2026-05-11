import os
import torch
import numpy as np
import gymnasium as gym

from typing import Optional, Tuple
from abc import ABC, abstractmethod
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
        num_envs: int = 1,
        eval_frequency: int = 1000,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        num_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
        model_dir: str = "models",
    ):
        """
        Initialize the base model with environment and hyperparameters.

        Args:
            environment (gym.Env): The training environment.
            num_envs (int): The number of training environments (for parallel training).
            eval_frequency (int): Number of training iterations between evaluations.
            learning_rate (float, optional): Learning rate for the optimizer. Default is 3e-4.
            clip_range (float, optional): PPO clipping range. Default is 0.2.
            value_loss_coef (float, optional): Coefficient for value loss. Default is 0.5.
            max_grad_norm (float, optional): Maximum gradient norm for gradient clipping. Default is 0.5.
            gamma (float, optional): Discount factor for future rewards. Default is 0.99.
            gae_lambda (float, optional): Lambda for Generalized Advantage Estimation. Default is 0.95.
            entropy_coef (float, optional): Coefficient for entropy bonus. Default is 0.01.
            num_epochs (int, optional): Number of training epochs per update. Default is 10.
            batch_size (int, optional): Mini-batch size. Default is 64.
            device (str, optional): Device for training, either 'cpu' or 'cuda'. Default is 'cpu'.
            model_dir (str, optional): Folder in which the generated models are saved.
        """
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")

        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self.env = environment
        self.num_envs = num_envs
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
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model_dir = model_dir

        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)


    @abstractmethod
    def predict(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Predict an action for a given state.

        Args:
            state (np.ndarray): Current state observation.
            deterministic (boolean): Whether the action should be deterministic or not.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the selected action and its log probability.
        """
        pass


    @abstractmethod
    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
    ):
        """
        Train the model using collected experiences.

        Args:
            stop_condition (StopCondition, optional): A stopping criterion to terminate training early.
        """
        pass


    @abstractmethod
    def save_policy(self, policy_file_name: str):
        """
        Save the model policy to a file.

        Args:
            policy_file_name (str): Name of the policy file to save.
        """
        pass


    @abstractmethod
    def load_policy(self, policy_file_name: str):
        """
        Load the model policy from a file.

        Args:
            policy_file_name (str): Name of the policy file to load.
        """
        pass


    @abstractmethod
    def set_policy_eval(self):
        """
        Set policy in eval mode.
        """
        pass


    def get_env(self) -> gym.Env:
        """
        Get the training environment.

        Returns:
            gym.Env: The training environment instance.
        """
        return self.env
