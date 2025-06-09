import os
import copy
import torch
import numpy as np
import gymnasium as gym

from typing import Optional, Tuple
from fw.stop_condition import StopCondition
from fw.policies.base_model import BaseModel
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


class PPO2Model(BaseModel):
    """
    PPO2Model integrates the Stable-Baselines3 implementation of Proximal Policy Optimization (PPO)
    into a custom training framework based on the BaseModel interface.

    It wraps an SB3 PPO agent and provides standardized methods for environment setup, policy prediction,
    and training with callbacks, logging, and evaluation support. This class is fully compliant with the
    BaseModel API, enabling interchangeable use with other reinforcement learning models in the framework.

    Attributes:
        ppo (stable_baselines3.PPO): The underlying SB3 PPO model.
        env (gym.Env): The input environment.
        train_env (gym.Env): The training environment (can be vectorized).
    """

    def __init__(
        self,
        *args,
        total_time_steps = 200000,
        log_interval = 10,
        n_steps: int = 2048,
        reward_threshold: float = 100000.8,
        log_dir: str = "logs",
        tensorboard_log: Optional[str] = "logs/tensorboard",
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize the PPO2Model with environment and Stable-Baselines3-specific settings.

        This constructor wraps the SB3 PPO model and adds extended configuration options
        such as total training steps, logging intervals, and TensorBoard support.

        Args:
            *args: Positional arguments forwarded to the BaseModel constructor.
            total_time_steps (int, optional): Total number of environment steps for training.
                Defaults to 200000.
            log_interval (int, optional): Number of updates between logging to stdout.
                Defaults to 10.
            n_steps (int, optional): Number of steps to run for each environment per policy update.
                Corresponds to `n_steps` in SB3 PPO. Defaults to 2048.
            reward_threshold (float, optional): Reward threshold to stop training early if exceeded.
                Used with `StopTrainingOnRewardThreshold` callback. Defaults to 100000.8.
            log_dir (str, optional): Directory where evaluation logs and intermediate results are stored.
                Defaults to "logs".
            tensorboard_log (Optional[str], optional): Path to log TensorBoard summaries, or None to disable.
                Defaults to "logs/tensorboard".
            verbose (bool, optional): Verbosity flag; if True, enables console output during training.
                Defaults to True.
            **kwargs: Additional keyword arguments passed to the BaseModel constructor.
        """
        super().__init__(*args, **kwargs)

        self.total_time_steps = total_time_steps
        self.log_interval = log_interval
        self.n_steps = n_steps
        self.reward_threshold = reward_threshold
        self.log_dir = log_dir
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose

        self.train_env = DummyVecEnv([lambda: Monitor(self.create_env(), "logs/env_0")])

        policy_kwargs = dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        )

        self.ppo = PPO(
            policy="MlpPolicy",
            env=self.train_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.num_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
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
        env = copy.deepcopy(self.env)
        if hasattr(env, "randomize"):
            env.randomize()
        return env


    def predict(self, state: np.ndarray) -> Tuple:
        """
        Predict an action for a given observation using the current policy.

        Args:
            state (np.ndarray): Current observation from the environment.

        Returns:
            Tuple: A tuple (action, None), where action is the predicted action. Log-probabilities are not returned.
        """
        action, _ = self.ppo.predict(state, deterministic=True)
        return action, 0.0


    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
    ):
        """
        Train the PPO agent using Stable-Baselines3 with evaluation callbacks and logging.

        Args:
            stop_condition (StopCondition, optional): Optional custom stop condition (not currently used).

        Notes:
            The model is saved both during training (if performance improves) and after training is completed.
        """
        print(f"Starting training for {self.total_time_steps} time steps...")

        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=self.reward_threshold,
            verbose=int(self.verbose),
        )

        eval_callback = EvalCallback(
            self.train_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=self.eval_frequency,
            deterministic=True,
            callback_on_new_best=stop_callback,
            verbose=int(self.verbose),
        )

        self.ppo.learn(
            total_timesteps=self.total_time_steps,
            callback=eval_callback,
            log_interval=self.log_interval,
            progress_bar=True,
        )

        final_model_path = f"final_ppo_{self.total_time_steps}"
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
        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        self.ppo = PPO.load(policy_file_name, device=self.device)


    def save_policy(self, policy_file_name: str) -> None:
        """
        Save the trained model's policy.

        Args:
            policy_file_name (str): Name of the policy file to save.

        Raises:
            RuntimeError: If no model is created or the policy fails to save.
        """
        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        self.ppo.save(policy_file_name)


    def set_policy_eval(self):
        """
        Set policy in evaluation mode.
        """
        self.ppo.policy.eval()
