import os
import copy
import torch
import logging
import numpy as np
import gymnasium as gym

from pathlib import Path
from typing import Optional, Tuple
from fw.policies.base_agent import BaseAgent
from fw.configuration.stop_condition import StopCondition
from fw.common.tools.ppo_utils import reseed_env_rng
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parents[2]


class PPO2Agent(BaseAgent):
    """
    PPO2Agent integrates the Stable-Baselines3 implementation of Proximal Policy Optimization (PPO)
    into a custom training framework based on the BaseAgent interface.

    It wraps an SB3 PPO agent and provides standardized methods for environment setup, policy prediction,
    and training with callbacks, logging, and evaluation support. This class is fully compliant with the
    BaseAgent API, enabling interchangeable use with other reinforcement learning agents in the framework.

    Attributes:
        env (gym.Env): The input environment.
        ppo (stable_baselines3.PPO): The underlying SB3 PPO agent.
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
        Initialize the PPO2Agent with environment and Stable-Baselines3-specific settings.

        This constructor wraps the SB3 PPO agent and adds extended configuration options
        such as total training steps, logging intervals, and TensorBoard support.

        Args:
            *args: Positional arguments forwarded to the BaseAgent constructor.
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
            **kwargs: Additional keyword arguments passed to the BaseAgent constructor.
        """

        super().__init__(*args, **kwargs)

        self.total_time_steps = total_time_steps
        self.log_interval = log_interval
        self.n_steps = n_steps
        self.reward_threshold = reward_threshold
        self.log_dir = str(_SRC_DIR / log_dir)
        # None disables TensorBoard logging (as documented).
        self.tensorboard_log = str(_SRC_DIR / tensorboard_log) if tensorboard_log else None
        self.verbose = verbose

        if self.num_envs > 1:
            train_env = SubprocVecEnv([self.make_env(i) for i in range(self.num_envs)])
        else:
            logging_dir = str(_SRC_DIR / "logs/env_0")
            train_env = DummyVecEnv([lambda: Monitor(self.create_env(), logging_dir)])

        policy_kwargs = dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU
        )

        self.ppo = PPO(
            policy="MlpPolicy",
            env=train_env,
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


    def make_env(self, rank):
        """
        Utility function for multiprocessing env creation.
        """

        def _init():
            """
            Create one monitored environment instance for a worker process.

            Returns:
                A monitored Gym environment instance for the given worker rank.
            """

            logging_dir = str(_SRC_DIR / f"logs/env_{rank}")
            return Monitor(self.create_env(), logging_dir)

        return _init


    def create_env(self) -> gym.Env:
        """
        Create a fresh copy of the base environment.

        Returns:
            gym.Env: A deep copy of the original environment. If the environment supports randomization,
                     a random initial configuration is applied.
        """

        env = copy.deepcopy(self.env)
        # Each copy gets its own RNG; otherwise every worker would draw the
        # same randomization as the template environment.
        reseed_env_rng(env)
        if hasattr(env, "randomize"):
            env.randomize()
        return env


    def predict(self, state: np.ndarray, deterministic: bool = False) -> Tuple:
        """
        Predict an action for a given observation using the current policy.

        Args:
            state (np.ndarray): Current observation from the environment.
            deterministic (boolean): Whether the action should be deterministic or not.

        Returns:
            Tuple[np.ndarray, None, None]: (action, log_prob, value). SB3 does not
            expose per-step log-probabilities or value estimates at inference time;
            both are None. All callers must unpack exactly three values.
        """

        action, _ = self.ppo.predict(state, deterministic=deterministic)
        # Return (action, log_prob, value). SB3 does not expose per-step
        # log-probabilities or value estimates at inference time; both are
        # returned as None so callers that unpack three values (e.g.
        # agent_manager_simple, episode_collector) work without modification.
        return action, None, None


    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
    ):
        """
        Train the PPO agent using Stable-Baselines3 with evaluation callbacks and logging.

        Args:
            stop_condition (StopCondition, optional): Optional custom stop condition (not currently used).

        Notes:
            The agent is saved both during training (if performance improves) and after training is completed.
        """

        logger.info("Starting training for %d time steps...", self.total_time_steps)

        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=self.reward_threshold,
            verbose=int(self.verbose),
        )

        logging_dir = str(_SRC_DIR / "logs/eval_env")
        eval_env = Monitor(self.create_env(), logging_dir)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=self.eval_frequency,
            deterministic=True,
            callback_on_new_best=stop_callback,
            verbose=int(self.verbose),
        )

        try:
            self.ppo.learn(
                total_timesteps=self.total_time_steps,
                callback=eval_callback,
                log_interval=self.log_interval,
                progress_bar=True,
            )
        finally:
            eval_env.close()

        final_model_path = f"final_ppo_{self.total_time_steps}"
        self.save_policy(final_model_path)

        logger.info("Training completed! Final agent saved to %s", final_model_path)


    def load_policy(self, policy_file_name: str) -> None:
        """
        Load a trained agent's policy.

        Args:
            policy_file_name (str): Name of the policy file to load.

        Raises:
            RuntimeError: If no agent is created or the policy fails to load.
            ValueError: If the loaded policy dimensions do not match the environment.
        """

        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        loaded_ppo = PPO.load(policy_file_name, device=self.device)
        
        if loaded_ppo.observation_space.shape != self.env.observation_space.shape:
            raise ValueError(f"Observation space dimension mismatch: loaded policy expects {loaded_ppo.observation_space.shape}, but environment provides {self.env.observation_space.shape}.")
            
        if loaded_ppo.action_space.shape != self.env.action_space.shape:
            raise ValueError(f"Action space dimension mismatch: loaded policy expects {loaded_ppo.action_space.shape}, but environment requires {self.env.action_space.shape}.")
            
        # Attach the current training env so that learn() can continue from
        # the loaded weights (PPO.load without env leaves model.env = None).
        # Re-loading WITH the env lets SB3 size the model for this env's
        # num_envs (set_env would assert when it differs from the saved one).
        current_env = self.ppo.get_env() if self.ppo is not None else None
        if current_env is not None:
            try:
                loaded_ppo = PPO.load(policy_file_name, env=current_env, device=self.device)
            except ValueError as exc:
                # Same shapes (checked above) but other space details, e.g. Box
                # bounds: keep the env-less model, as before this change, so
                # predict/visualize still work; continued training then needs
                # matching spaces.
                logger.warning(
                    "Loaded '%s' without attaching the training env (%s); "
                    "predict works, continued training needs matching spaces.",
                    policy_file_name, exc,
                )

        self.ppo = loaded_ppo


    def save_policy(self, policy_file_name: str) -> None:
        """
        Save the trained agent's policy.

        Args:
            policy_file_name (str): Name of the policy file to save.

        Raises:
            RuntimeError: If no agent is created or the policy fails to save.
        """

        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        self.ppo.save(policy_file_name)


    def set_policy_eval(self):
        """
        Set policy in evaluation mode.
        """

        self.ppo.policy.eval()


    def close(self) -> None:
        """
        Close the vectorized training environment (worker processes and,
        for simulator-backed envs, their connections).
        """

        if self.ppo is not None:
            vec_env = self.ppo.get_env()
            if vec_env is not None:
                vec_env.close()
