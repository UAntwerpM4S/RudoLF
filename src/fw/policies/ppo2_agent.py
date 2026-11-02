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
    BaseAgent wrapper around the Stable-Baselines3 PPO implementation.

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
        Create the vectorized training env and the SB3 PPO model.

        Args:
            *args: Forwarded to BaseAgent.
            total_time_steps (int, optional): Total training environment steps.
            log_interval (int, optional): Updates between stdout logs.
            n_steps (int, optional): Steps per environment per update (SB3 `n_steps`).
            reward_threshold (float, optional): Eval reward that stops training early.
            log_dir (str, optional): Evaluation log directory (relative to src).
            tensorboard_log (Optional[str], optional): TensorBoard directory
                (relative to src), or None to disable.
            verbose (bool, optional): Enable SB3 console output.
            **kwargs: Forwarded to BaseAgent.
        """

        super().__init__(*args, **kwargs)

        self.total_time_steps = total_time_steps
        self.log_interval = log_interval
        self.n_steps = n_steps
        self.reward_threshold = reward_threshold
        self.log_dir = str(_SRC_DIR / log_dir)
        self.tensorboard_log = str(_SRC_DIR / tensorboard_log) if tensorboard_log else None
        self.verbose = verbose

        # One child seed per training worker (see BaseAgent 'seed').
        worker_seeds = self._spawn_worker_seeds(self.num_envs)
        if self.num_envs > 1:
            train_env = SubprocVecEnv([self.make_env(i, worker_seeds[i]) for i in range(self.num_envs)])
        else:
            logging_dir = str(_SRC_DIR / "logs/env_0")
            train_env = DummyVecEnv([lambda: Monitor(self.create_env(worker_seeds[0]), logging_dir)])

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


    def make_env(self, rank, worker_seed=None):
        """
        Return a factory creating one monitored env for a SubprocVecEnv worker.

        Args:
            rank: Worker index (used for the Monitor log directory).
            worker_seed: Seed for this worker's env RNG; None spawns a new
                child of the agent's worker SeedSequence.
        """

        if worker_seed is None:
            worker_seed = self._spawn_worker_seeds(1)[0]

        def _init():
            """
            Create the monitored environment for this worker.
            """

            logging_dir = str(_SRC_DIR / f"logs/env_{rank}")
            return Monitor(self.create_env(worker_seed), logging_dir)

        return _init


    def create_env(self, worker_seed=None) -> gym.Env:
        """
        Create a deep copy of the base environment with its own seeded RNG.

        Args:
            worker_seed: Seed for the copy's env RNG; None spawns a new
                child of the agent's worker SeedSequence (see BaseAgent 'seed').

        Returns:
            gym.Env: The copy, randomized if it supports `randomize()`.
        """

        if worker_seed is None:
            worker_seed = self._spawn_worker_seeds(1)[0]
        env = copy.deepcopy(self.env)
        reseed_env_rng(env, worker_seed)
        if hasattr(env, "randomize"):
            env.randomize()
        return env


    def predict(self, state: np.ndarray, deterministic: bool = False) -> Tuple:
        """
        Predict an action with the current policy.

        Args:
            state (np.ndarray): Current observation.
            deterministic (bool): Use the deterministic action.

        Returns:
            Tuple[np.ndarray, None, None]: (action, None, None); SB3 does not
            expose log-probabilities or values at inference time.
        """

        action, _ = self.ppo.predict(state, deterministic=deterministic)
        return action, None, None


    def learn(
        self,
        stop_condition: Optional[StopCondition] = None,
    ):
        """
        Train with SB3, evaluating every `eval_frequency` steps.

        The best model is saved by the eval callback and the final model as
        "final_ppo_<total_time_steps>".

        Args:
            stop_condition (StopCondition, optional): Not used.
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
        Load a saved SB3 policy (relative to model_dir).

        Args:
            policy_file_name (str): Name of the policy file to load.

        Raises:
            ValueError: If the observation or action space shapes do not match.
        """

        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        loaded_ppo = PPO.load(policy_file_name, device=self.device)
        
        if loaded_ppo.observation_space.shape != self.env.observation_space.shape:
            raise ValueError(f"Observation space dimension mismatch: loaded policy expects {loaded_ppo.observation_space.shape}, but environment provides {self.env.observation_space.shape}.")
            
        if loaded_ppo.action_space.shape != self.env.action_space.shape:
            raise ValueError(f"Action space dimension mismatch: loaded policy expects {loaded_ppo.action_space.shape}, but environment requires {self.env.action_space.shape}.")
            
        # Re-load with the current training env so learn() can continue (SB3
        # then sizes the model for this num_envs; set_env would assert instead).
        current_env = self.ppo.get_env() if self.ppo is not None else None
        if current_env is not None:
            try:
                loaded_ppo = PPO.load(policy_file_name, env=current_env, device=self.device)
            except ValueError as exc:
                # Other space details differ (e.g. Box bounds): keep the env-less
                # model, which still predicts.
                logger.warning(
                    "Loaded '%s' without attaching the training env (%s); "
                    "predict works, continued training needs matching spaces.",
                    policy_file_name, exc,
                )

        self.ppo = loaded_ppo


    def save_policy(self, policy_file_name: str) -> None:
        """
        Save the SB3 policy (relative to model_dir).

        Args:
            policy_file_name (str): Name of the policy file to save.
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
        Close the vectorized training env (worker processes and simulator connections).
        """

        if self.ppo is not None:
            vec_env = self.ppo.get_env()
            if vec_env is not None:
                vec_env.close()
