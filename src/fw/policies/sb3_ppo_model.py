import os
import copy
import torch
import numpy as np
import gymnasium as gym

from typing import Optional
from fw.stop_condition import StopCondition
from fw.policies.base_model import BaseModel
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


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

        self.n_envs = 1
        self.verbose = True
        self.model = None
        self.env = environment
        self.eval_env = None

        # Create directories for saving models and logs
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Set random seed for reproducibility
        set_random_seed(42)

        self.setup_environments()

        policy_kwargs = dict(
            net_arch=[256, 256, 128],  # Neural network architecture
            activation_fn=torch.nn.ReLU
        )

        self.ppo = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                device="cpu",
                tensorboard_log="logs/tensorboard/")


    def create_env(self):
        """
        Create a copy of the environment for parallel runs.

        Returns:
            gym.Env: A new instance of the environment.
        """
        env = copy.deepcopy(self.get_env())

        # Optionally randomize the environment's initial state
        if hasattr(env, "randomize"):
            env.randomize()

        return env


    def make_env(self, rank, seed=0):   # render_mode=None, wind=False, current=False):
        """
        Utility function for multiprocessing env creation.
        """

        def _init():
            env = self.create_env()
            env.seed(seed + rank)
            env = Monitor(env, f"logs/env_{rank}")
            return env

        set_random_seed(seed)
        return _init


    def setup_environments(self, wind=False, current=False):
        """
        Setup training and evaluation environments.
        """
        print(f"Setting up {self.n_envs} parallel environments...")

        # Create evaluation environment
        self.eval_env = Monitor(
            self.create_env(),
            "logs/eval_env"
        )

        # Create vectorized training environment
        # if self.n_envs > 1:
        #     self.env = SubprocVecEnv([
        #         self.make_env(i, wind=wind, current=current)
        #         for i in range(self.n_envs)
        #     ])
        # else:
        self.env = DummyVecEnv([
            lambda: Monitor(self.create_env(), "logs/env_0")
        ])

        print("Environments setup complete!")


    def predict(self, state: np.ndarray) -> tuple:
        """
        Predict an action for a given state using the policy.

        Args:
            state (np.ndarray): Current state as a NumPy array.

        Returns:
            tuple: A tuple containing the predicted action and log probability.
        """
        return self.ppo.predict(state, deterministic=True)
        # return actions, None, states


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
        print(f"Starting training for {self.max_nbr_iterations} timesteps...")

        # Create callback for evaluation and early stopping
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=100000.8,  # Stop when average reward reaches 0.8
            verbose=1
        )

        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=f"models/best_ppo",
            log_path="logs/eval",
            eval_freq=10000, # self.eval_frequency,
            deterministic=True,
            render=False,
            callback_on_new_best=stop_callback,
            verbose=1
        )

        # Train the model
        self.ppo.learn(
            total_timesteps=600000, # self.max_nbr_iterations,
            callback=eval_callback,
            log_interval=10,
            progress_bar=True
        )

        # Save the final model
        model_path = f"models/final_ppo_{self.max_nbr_iterations}"
        self.ppo.save(model_path)
        print(f"Training completed! Final model saved to {model_path}")
