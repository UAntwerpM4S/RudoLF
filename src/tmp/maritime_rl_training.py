import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import torch
import warnings
warnings.filterwarnings("ignore")

# Import your environment
from py_sim_env import PySimEnv

class MaritimeRLTrainer:
    """
    A comprehensive trainer for the maritime environment using various RL algorithms.
    """
    
    def __init__(self, algorithm='PPO', n_envs=4, verbose=1):
        """
        Initialize the trainer.
        
        Args:
            algorithm (str): RL algorithm to use ('PPO', 'SAC', 'TD3')
            n_envs (int): Number of parallel environments
            verbose (int): Verbosity level
        """
        self.algorithm = algorithm
        self.n_envs = n_envs
        self.verbose = verbose
        self.model = None
        self.env = None
        self.eval_env = None
        
        # Create directories for saving models and logs
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Set random seed for reproducibility
        set_random_seed(42)
        
    def create_env(self, render_mode=None, wind=False, current=False):
        """Create a single environment instance."""
        return PySimEnv(
            render_mode=render_mode,
            time_step=0.1,
            max_steps=1500,
            verbose=False,
            wind=wind,
            current=current
        )
    
    def make_env(self, rank, seed=0, render_mode=None, wind=False, current=False):
        """
        Utility function for multiprocessing env creation.
        """
        def _init():
            env = self.create_env(render_mode=render_mode, wind=wind, current=current)
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
        
        # Create vectorized training environment
        if self.n_envs > 1:
            self.env = SubprocVecEnv([
                self.make_env(i, wind=wind, current=current) 
                for i in range(self.n_envs)
            ])
        else:
            self.env = DummyVecEnv([
                lambda: Monitor(self.create_env(wind=wind, current=current), "logs/env_0")
            ])
        
        # Create evaluation environment
        self.eval_env = Monitor(
            self.create_env(render_mode=None, wind=wind, current=current),
            "logs/eval_env"
        )
        
        print("Environments setup complete!")
    
    def create_model(self, total_timesteps=500000):
        """
        Create the RL model based on the specified algorithm.
        """
        print(f"Creating {self.algorithm} model...")
        
        # Common policy kwargs
        policy_kwargs = dict(
            net_arch=[256, 256, 128],  # Neural network architecture
            activation_fn=torch.nn.ReLU
        )
        
        if self.algorithm == 'PPO':
            self.model = PPO(
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
                tensorboard_log="logs/tensorboard/"
            )
        
        elif self.algorithm == 'SAC':
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                buffer_size=1000000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                tensorboard_log="logs/tensorboard/"
            )
        
        elif self.algorithm == 'TD3':
            self.model = TD3(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                buffer_size=1000000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                tensorboard_log="logs/tensorboard/"
            )
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        print(f"{self.algorithm} model created successfully!")
    
    def train(self, total_timesteps=500000, eval_freq=10000, save_freq=50000):
        """
        Train the RL agent.
        
        Args:
            total_timesteps (int): Total number of training timesteps
            eval_freq (int): Frequency of evaluation
            save_freq (int): Frequency of model saving
        """
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Create callback for evaluation and early stopping
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=100000.8,  # Stop when average reward reaches 0.8
            verbose=1
        )
        
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"models/best_{self.algorithm.lower()}",
            log_path="logs/eval",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            callback_on_new_best=stop_callback,
            verbose=1
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=10,
            progress_bar=True
        )
        
        # Save the final model
        model_path = f"models/final_{self.algorithm.lower()}_{total_timesteps}"
        self.model.save(model_path)
        print(f"Training completed! Final model saved to {model_path}")
    
    def evaluate(self, n_eval_episodes=10, render=False):
        """
        Evaluate the trained model.
        
        Args:
            n_eval_episodes (int): Number of episodes to evaluate
            render (bool): Whether to render during evaluation
        """
        if self.model is None:
            print("No model to evaluate! Train a model first.")
            return
        
        print(f"Evaluating model for {n_eval_episodes} episodes...")
        
        # Create evaluation environment with rendering if requested
        eval_env = self.create_env(
            render_mode='human' if render else None,
            wind=False,  # You can change these based on your evaluation needs
            current=False
        )
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if render:
                    eval_env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check if episode was successful (positive final reward)
            if episode_reward > 0.5:
                success_count += 1
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}, Length = {episode_length}")
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        success_rate = success_count / n_eval_episodes
        
        print(f"\n--- Evaluation Results ---")
        print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"Mean Episode Length: {mean_length:.1f}")
        print(f"Success Rate: {success_rate:.1%}")
        
        eval_env.close()
        return mean_reward, std_reward, success_rate
    
    def load_model(self, model_path):
        """Load a pre-trained model."""
        print(f"Loading model from {model_path}...")
        
        if self.algorithm == 'PPO':
            self.model = PPO.load(model_path, env=self.env)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(model_path, env=self.env)
        elif self.algorithm == 'TD3':
            self.model = TD3.load(model_path, env=self.env)
        
        print("Model loaded successfully!")
    
    def plot_training_curves(self):
        """Plot training curves from tensorboard logs."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            import glob
            
            log_dirs = glob.glob("logs/tensorboard/PPO_*")
            if not log_dirs:
                print("No tensorboard logs found!")
                return
            
            # This is a basic implementation - you might want to use tensorboard directly
            print("Training curves can be viewed using: tensorboard --logdir logs/tensorboard/")
            
        except ImportError:
            print("Please install tensorboard to view training curves: pip install tensorboard")


def main():
    """
    Main training script with different configuration options.
    """
    print("=== Maritime RL Training ===")
    
    # Configuration options
    configs = [
        {
            'name': 'Basic Training',
            'algorithm': 'PPO',
            'wind': False,
            'current': False,
            'timesteps': 300000
        },
        {
            'name': 'Advanced Training with Environmental Effects',
            'algorithm': 'SAC',
            'wind': True,
            'current': True,
            'timesteps': 500000
        }
    ]
    
    # Choose configuration
    print("Available training configurations:")
    for i, config in enumerate(configs):
        print(f"{i+1}. {config['name']} ({config['algorithm']})")
    
    choice = input("\nSelect configuration (1-2) or press Enter for basic: ").strip()
    
    if choice == '2':
        selected_config = configs[1]
    else:
        selected_config = configs[0]
    
    print(f"\nSelected: {selected_config['name']}")
    
    # Initialize trainer
    trainer = MaritimeRLTrainer(
        algorithm=selected_config['algorithm'],
        n_envs=4,  # Use 4 parallel environments
        verbose=1
    )
    
    # Setup environments
    trainer.setup_environments(
        wind=selected_config['wind'],
        current=selected_config['current']
    )
    
    # Create and train model
    trainer.create_model()
    trainer.train(total_timesteps=selected_config['timesteps'])
    
    # Evaluate the trained model
    print("\n=== Evaluation ===")
    trainer.evaluate(n_eval_episodes=5, render=False)
    
    # Ask if user wants to see a rendered evaluation
    show_render = input("\nWould you like to see a rendered evaluation? (y/n): ").strip().lower()
    if show_render == 'y':
        trainer.evaluate(n_eval_episodes=1, render=True)
    
    print("\n=== Training Complete ===")
    print("Models saved in 'models/' directory")
    print("Logs saved in 'logs/' directory")
    print("View training progress with: tensorboard --logdir logs/tensorboard/")


if __name__ == "__main__":
    main()
