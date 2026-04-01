import os
import csv
import torch
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from fw.stop_condition import StopCondition
from fw.environments.base_env import BaseEnv
from fw.common.tools.path_utils import resolve_path_relative_to_src


class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.

    This class provides a unified interface and shared hyperparameters
    for reinforcement learning algorithms. Subclasses must implement
    the core training and inference logic.
    """

    @staticmethod
    def compute_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
        gamma: float,
        lambda_: float,
    ) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE helps in reducing the variance of the policy gradient estimates by using both bootstrapping and Monte Carlo estimates.

        Args:
            rewards (np.ndarray): Array of rewards for the batch.
            values (np.ndarray): Array of value estimates for each state.
            dones (np.ndarray): Array of done flags indicating if the episode ended.
            next_value (float): Value estimate for the state after the batch.
            gamma (float): Discount factor for future rewards.
            lambda_ (float): GAE parameter that controls bias-variance tradeoff (default: 0.95).

        Returns:
            np.ndarray: Computed advantages as a NumPy array.
        """

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        # Reverse iteration for GAE calculation
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + gamma * lambda_ * next_non_terminal * last_gae
            advantages[t] = last_gae

        return advantages


    @staticmethod
    def find_first_true(bool_array):
        """
        Find the index of the first True value in a boolean array.
        If no True value exists, return -1.

        Args:
            bool_array (np.ndarray): A NumPy array of boolean values.

        Returns:
            int: The index of the first True value, or -1 if not found.
        """

        # Find the index of the first True value
        true_indices = np.where(bool_array)[0]

        # Return the first index if it exists, otherwise return -1
        return true_indices[0] if true_indices.size > 0 else -1


    @staticmethod
    def dump_metrics_to_csv(model_dir, csv_filename, training_metrics):
        """
        Save training metrics to a CSV file.

        This method writes the training metrics (such as loss, advantage, reward)
        to a CSV file with headers.

        Args:
            model_dir (): The folder under which all models are stored.
            csv_filename (str): The name of the CSV file to write.
            training_metrics (list): A list of dictionaries containing the training metrics.

        """

        if not training_metrics:
            print("Warning: No training metrics to write.")
            return

        filepath = os.path.join(model_dir, csv_filename)

        with open(filepath, mode="w", newline="") as file:
            # Get all possible keys (assuming all dicts have the same structure)
            fieldnames = training_metrics[0].keys()

            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_metrics)


    def __init__(
        self,
        environment: BaseEnv,
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
        Initialize the base agent with environment and hyperparameters.

        Args:
            environment (BaseEnv): The training environment.
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
            model_dir (str, optional): Folder in which the generated agents are saved.
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

        # Normalize relative model paths so they always land under src/models.
        # This prevents cases where training from inside src/ creates src/src/models.
        if self.model_dir:
            project_src_dir = Path(__file__).resolve().parents[2]
            model_dir_path = resolve_path_relative_to_src(self.model_dir, project_src_dir)
            self.model_dir = str(model_dir_path)
            os.makedirs(self.model_dir, exist_ok=True)

        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4  # avoid division by zero


    def normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running statistics (Welford's algorithm).

        Args:
            rewards: Raw rewards array of shape (batch_size,).

        Returns:
            Normalized rewards with zero mean and unit variance.
        """

        # Welford's algorithm for running stats
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)

        delta = batch_mean - self.reward_mean
        tot_count = self.reward_count + batch_count

        new_mean = self.reward_mean + delta * batch_count / tot_count
        m_a = self.reward_var * self.reward_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / tot_count
        new_var = m2 / tot_count

        self.reward_mean = new_mean
        self.reward_var = new_var
        self.reward_count = tot_count

        return (rewards - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)


    @abstractmethod
    def predict(
            self,
            state: Union[np.ndarray, Tuple[np.ndarray, ...]],
            deterministic: bool = False
    ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], np.ndarray, np.ndarray]:
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
        Train the agent using collected experiences.

        Args:
            stop_condition (StopCondition, optional): A stopping criterion to terminate training early.
        """
        pass


    @abstractmethod
    def save_policy(self, policy_file_name: str):
        """
        Save the agent policy to a file.

        Args:
            policy_file_name (str): Name of the policy file to save.
        """
        pass


    @abstractmethod
    def load_policy(self, policy_file_name: str):
        """
        Load the agent policy from a file.

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


    def get_env(self) -> BaseEnv:
        """
        Get the training environment.

        Returns:
            BaseEnv: The training environment instance.
        """
        return self.env
