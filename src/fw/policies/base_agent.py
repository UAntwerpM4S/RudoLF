import os
import logging
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from fw.environments.base_env import BaseEnv
from fw.policies.ppo_policy import get_device
from fw.common.tools.ppo_utils import resolve_base_seed
from fw.configuration.stop_condition import StopCondition
from fw.common.tools.path_utils import resolve_path_relative_to_src

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for RL agents: shared PPO hyperparameters, device,
    model directory and worker seeding. Subclasses implement training and inference.
    """

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
        seed: Optional[int] = None,
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
            device (str, optional): Device for training. One of 'cpu', 'cuda',
                or 'auto' ('auto' resolves to 'cuda' if available, else 'cpu').
                Default is 'cpu'.
            model_dir (str, optional): Folder in which the generated agents are saved.
            seed (int, optional): Base seed for the worker environment RNGs
                ('seed' hyperparameter); each worker gets an independent child
                seed. None draws a random base seed, which is logged.

        Raises:
            ValueError: If `device` is not 'cpu', 'cuda' or 'auto', if
                `num_envs` is not positive, or if `seed` is not None or a
                non-negative integer.
        """

        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")

        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self.env = environment
        self.num_envs = num_envs

        # PPO hyperparameters
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

        self.device = get_device(device)
        self.model_dir = model_dir

        # Resolve relative model paths against src/ (independent of the cwd).
        if self.model_dir:
            project_src_dir = Path(__file__).resolve().parents[2]
            model_dir_path = resolve_path_relative_to_src(self.model_dir, project_src_dir)
            self.model_dir = str(model_dir_path)
            os.makedirs(self.model_dir, exist_ok=True)

        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4  # avoid division by zero

        # Worker seeding: SeedSequence.spawn() is deterministic in call order,
        # so recreated workers get new but reproducible seeds.
        self.seed = resolve_base_seed(seed)
        self._worker_seed_seq = np.random.SeedSequence(self.seed)
        # Only env RNG streams (start poses, routes, scenarios) are seeded, not
        # torch policy sampling, so training is not bit-reproducible.
        logger.info(
            "%s: base seed of the env RNG streams = %d%s.",
            type(self).__name__, self.seed,
            "" if seed is not None else " (randomly drawn; set the 'seed' hyperparameter to get the same env streams again)",
        )


    def _spawn_worker_seeds(self, n: int) -> list[np.random.SeedSequence]:
        """
        Child seeds for `n` new worker environments.

        Args:
            n: Number of workers.

        Returns:
            list[np.random.SeedSequence]: One independent child seed per worker.
        """

        return self._worker_seed_seq.spawn(n)


    @abstractmethod
    def predict(
            self,
            state: Union[np.ndarray, Tuple[np.ndarray, ...]],
            deterministic: bool = False
    ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], np.ndarray, np.ndarray]:
        """
        Predict an action for a given state.

        Args:
            state: Current observation (array or tuple of arrays).
            deterministic (bool): Whether to act deterministically.

        Returns:
            Tuple of (action, log probability, value).
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
