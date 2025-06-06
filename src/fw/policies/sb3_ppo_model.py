import os
import copy
import torch
import numpy as np
import gymnasium as gym

from typing import Optional, Callable
from fw.stop_condition import StopCondition
from fw.policies.base_model import BaseModel
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


class SB3PPOModel(BaseModel):
    """
    SB3PPOModel integrates the Stable-Baselines3 implementation of Proximal Policy Optimization (PPO)
    into a custom training framework based on the BaseModel interface.

    It wraps an SB3 PPO agent and provides standardized methods for environment setup, policy prediction,
    and training with callbacks, logging, and evaluation support. This class is fully compliant with the
    BaseModel API, enabling interchangeable use with other reinforcement learning models in the framework.

    Attributes:
        ppo (stable_baselines3.PPO): The underlying SB3 PPO model.
        env (gym.Env): The main training environment (vectorized).
        eval_env (gym.Env): The evaluation environment.
        max_nbr_iterations (int): Maximum number of training steps.
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
        model_dir: str = "models",
        log_dir: str = "logs",
        results_dir: str = "results",
        tensorboard_log: Optional[str] = "logs/tensorboard",
        verbose: bool = True,
    ):
        """
        Initialize the SB3PPOModel with a specified environment and PPO hyperparameters.

        Args:
            environment (gym.Env): The base environment used for training.
            eval_frequency (int): Frequency (in steps) to evaluate and log model performance.
            learning_rate (float): Learning rate for the policy optimizer.
            clip_range (float): PPO clipping range for policy updates.
            value_loss_coef (float): Coefficient for the value loss term.
            max_grad_norm (float): Maximum allowed norm for gradient clipping.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda parameter for Generalized Advantage Estimation.
            entropy_coef (float): Coefficient for the policy entropy bonus.
            num_epochs (int): Number of optimization epochs per update.
            normalize (bool): Whether to normalize rewards during training.
            max_nbr_iterations (int): Total number of training steps.
            batch_size (int): Minibatch size for training.
            device (str): Device used for computation ('cpu' or 'cuda').
        """
        super().__init__(environment, eval_frequency, learning_rate, clip_range, value_loss_coef, max_grad_norm, gamma,
                         gae_lambda, entropy_coef, num_epochs, normalize, max_nbr_iterations, batch_size, device)

        self.log_dir = log_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.n_envs = 1

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.eval_env = DummyVecEnv([lambda: Monitor(self.create_env(), "logs/env_0")])

        policy_kwargs = dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        )

        self.ppo = PPO(
            "MlpPolicy",
            self.eval_env,
            learning_rate=self.learning_rate,
            n_steps=2048,  # could be exposed as a hyperparameter if needed
            batch_size=self.batch_size,
            n_epochs=self.num_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            clip_range_vf=None,
            ent_coef=self.entropy_coef,
            vf_coef=self.value_loss_coef,
            max_grad_norm=self.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=int(self.verbose),
            device=self.device,
            tensorboard_log=self.tensorboard_log
        )


    def create_env(self) -> gym.Env:
        """
        Create a fresh copy of the base environment.

        Returns:
            gym.Env: A deep copy of the original environment. If the environment supports randomization,
                     a random initial configuration is applied.
        """
        env = copy.deepcopy(self.get_env())
        if hasattr(env, "randomize"):
            env.randomize()
        return env


    def predict(self, state: np.ndarray) -> tuple:
        """
        Predict an action for a given observation using the current policy.

        Args:
            state (np.ndarray): Current observation from the environment.

        Returns:
            tuple: A tuple (action, None), where action is the predicted action. Log-probabilities are not returned.
        """
        return self.ppo.predict(state, deterministic=True)


    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
        num_envs: int = 1,
        callback: Optional[Callable] = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Train the PPO agent using Stable-Baselines3 with evaluation callbacks and logging.

        Args:
            stop_condition (StopCondition, optional): Optional custom stop condition (not currently used).
            num_envs (int): Number of parallel environments (currently fixed at 1).
            callback (callable, optional): Optional additional training callback.
            log_interval (int): Frequency (in episodes) of logging to the console.
            tb_log_name (str): TensorBoard logging directory name.
            reset_num_timesteps (bool): Whether to reset the timestep counter between runs.
            progress_bar (bool): Whether to display a progress bar during training.

        Notes:
            The model is saved both during training (if performance improves) and after training is completed.
        """
        if self.verbose:
            print(f"Starting training for {self.max_nbr_iterations} timesteps...")

        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=100000.8,  # Could also be parameterized
            verbose=1
        )

        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.model_dir, "best_ppo"),
            log_path=os.path.join(self.log_dir, "eval"),
            eval_freq=10000, # self.eval_frequency,
            deterministic=True,
            render=False,
            callback_on_new_best=stop_callback,
            verbose=1
        )

        self.ppo.learn(
            total_timesteps=120000, # self.max_nbr_iterations,
            callback=eval_callback,
            log_interval=10,
            progress_bar=True
        )

        final_model_path = os.path.join(self.model_dir, f"final_ppo_{self.max_nbr_iterations}")
        self.save_policy(final_model_path)

        print(f"Training completed! Final model saved to {final_model_path}")


    def load_policy(self, policy_file_name: str) -> None:
        """
        Load a trained model's policy.

        Args:
            policy_file_name (str): Name of the policy file to load.

        Raises:
            RuntimeError: If no model is created or the policy fails to load.
        """
        self.ppo = PPO.load(policy_file_name)


    def save_policy(self, policy_file_name: str) -> None:
        """
        Save the trained model's policy.

        Args:
            policy_file_name (str): Name of the policy file to save.

        Raises:
            RuntimeError: If no model is created or the policy fails to save.
        """
        self.ppo.save(policy_file_name)
