"""
Hyperparameter tuning script for maritime RL using Optuna.
This script automatically finds the best hyperparameters for your specific environment.
"""

import optuna
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings("ignore")

from py_sim_env import PySimEnv
from config import HYPERPARAMETER_RANGES, ENV_CONFIG, SCENARIOS

class HyperparameterTuner:
    """
    Automated hyperparameter tuning for maritime RL using Optuna.
    """
    
    def __init__(self, algorithm='PPO', scenario='basic', n_trials=50, 
                 training_timesteps=100000, eval_episodes=5):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            algorithm (str): RL algorithm to tune ('PPO', 'SAC', 'TD3')
            scenario (str): Environment scenario to use
            n_trials (int): Number of optimization trials
            training_timesteps (int): Timesteps per trial
            eval_episodes (int): Episodes for evaluation per trial
        """
        self.algorithm = algorithm
        self.scenario = scenario
        self.n_trials = n_trials
        self.training_timesteps = training_timesteps
        self.eval_episodes = eval_episodes
        
        # Get scenario configuration
        self.env_config = {**ENV_CONFIG, **SCENARIOS[scenario]}
        
        print(f"Tuning {algorithm} on scenario: {self.env_config['description']}")
    
    def create_env(self):
        """Create environment for training and evaluation."""
        return PySimEnv(
            render_mode=None,
            time_step=self.env_config['time_step'],
            max_steps=self.env_config['max_steps'],
            verbose=False,
            wind=self.env_config['wind'],
            current=self.env_config['current']
        )
    
    def sample_hyperparameters(self, trial):
        """Sample hyperparameters for the given algorithm."""
        ranges = HYPERPARAMETER_RANGES[self.algorithm]
        
        if self.algorithm == 'PPO':
            return {
                'learning_rate': trial.suggest_float('learning_rate', *ranges['learning_rate'], log=True),
                'n_steps': trial.suggest_categorical('n_steps', ranges['n_steps']),
                'batch_size': trial.suggest_categorical('batch_size', ranges['batch_size']),
                'n_epochs': trial.suggest_categorical('n_epochs', ranges['n_epochs']),
                'gamma': trial.suggest_float('gamma', *ranges['gamma']),
                'gae_lambda': trial.suggest_float('gae_lambda', *ranges['gae_lambda']),
                'clip_range': trial.suggest_float('clip_range', *ranges['clip_range']),
                'ent_coef': trial.suggest_float('ent_coef', *ranges['ent_coef'], log=True),
                'vf_coef': trial.suggest_float('vf_coef', *ranges['vf_coef']),
            }
        
        elif self.algorithm == 'SAC':
            ent_coef = trial.suggest_categorical('ent_coef', ranges['ent_coef'])
            if ent_coef == 'auto':
                ent_coef = 'auto'
            
            return {
                'learning_rate': trial.suggest_float('learning_rate', *ranges['learning_rate'], log=True),
                'batch_size': trial.suggest_categorical('batch_size', ranges['batch_size']),
                'tau': trial.suggest_float('tau', *ranges['tau']),
                'gamma': trial.suggest_float('gamma', *ranges['gamma']),
                'ent_coef': ent_coef,
            }
        
        elif self.algorithm == 'TD3':
            return {
                'learning_rate': trial.suggest_float('learning_rate', *ranges['learning_rate'], log=True),
                'batch_size': trial.suggest_categorical('batch_size', ranges['batch_size']),
                'tau': trial.suggest_float('tau', *ranges['tau']),
                'gamma': trial.suggest_float('gamma', *ranges['gamma']),
                'policy_delay': trial.suggest_categorical('policy_delay', ranges['policy_delay']),
                'target_policy_noise': trial.suggest_float('target_policy_noise', *ranges['target_policy_noise']),
                'target_noise_clip': trial.suggest_float('target_noise_clip', *ranges['target_noise_clip']),
            }
    
    def objective(self, trial):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Mean reward from evaluation
        """
        try:
            # Sample hyperparameters
            hyperparams = self.sample_hyperparameters(trial)
            
            # Create environment
            env = DummyVecEnv([lambda: Monitor(self.create_env())])
            eval_env = self.create_env()
            
            # Common configuration
            policy_kwargs = {
                'net_arch': [256, 256, 128],
                'activation_fn': torch.nn.ReLU
            }
            
            # Create model with sampled hyperparameters
            if self.algorithm == 'PPO':
                model = PPO(
                    "MlpPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    **hyperparams
                )
            elif self.algorithm == 'SAC':
                model = SAC(
                    "MlpPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    buffer_size=50000,  # Smaller buffer for faster training
                    learning_starts=100,
                    **hyperparams
                )
            elif self.algorithm == 'TD3':
                model = TD3(
                    "MlpPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    buffer_size=50000,  # Smaller buffer for faster training
                    learning_starts=100,
                    **hyperparams
                )
            
            # Train the model
            model.learn(total_timesteps=self.training_timesteps)
            
            # Evaluate the model
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=self.eval_episodes, deterministic=True
            )
            
            # Clean up
            env.close()
            eval_env.close()
            del model
            
            return mean_reward
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return -1000  # Return very low reward for failed trials
    
    def tune(self, study_name=None):
        """
        Run hyperparameter tuning.
        
        Args:
            study_name (str): Name for the Optuna study
            
        Returns:
            optuna.Study: The completed study object
        """
        if study_name is None:
            study_name = f"{self.algorithm}_{self.scenario}_tuning"
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        print(f"Starting hyperparameter tuning with {self.n_trials} trials...")
        print(f"Algorithm: {self.algorithm}")
        print(f"Scenario: {self.scenario}")
        print(f"Training timesteps per trial: {self.training_timesteps}")
        print("-" * 50)
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Print results
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING RESULTS")
        print("="*50)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best reward: {study.best_value:.3f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save study results
        import joblib
        joblib.dump(study, f"results/{study_name}.pkl")
        print(f"\nStudy saved to results/{study_name}.pkl")
        
        return study
    
    def plot_optimization_history(self, study):
        """Plot optimization history."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create optimization history plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(study.trials))),
                y=[trial.value for trial in study.trials if trial.value is not None],
                mode='lines+markers',
                name='Trial Rewards'
            ))
            
            # Add best value line
            best_values = []
            current_best = float('-inf')
            for trial in study.trials:
                if trial.value is not None and trial.value > current_best:
                    current_best = trial.value
                best_values.append(current_best if current_best != float('-inf') else None)
            
            fig.add_trace(go.Scatter(
                x=list(range(len(study.trials))),
                y=best_values,
                mode='lines',
                name='Best So Far',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'{self.algorithm} Hyperparameter Optimization History',
                xaxis_title='Trial',
                yaxis_title='Mean Reward',
                hovermode='x'
            )
            
            fig.write_html(f"results/{self.algorithm}_{self.scenario}_optimization_history.html")
            print(f"Optimization history saved to results/{self.algorithm}_{self.scenario}_optimization_history.html")
            
        except ImportError:
            print("Install plotly for visualization: pip install plotly")
    
    def train_best_model(self, study, training_timesteps=500000):
        """
        Train a model with the best hyperparameters found.
        
        Args:
            study: Optuna study object
            training_timesteps: Total timesteps for final training
        """
        print(f"\nTraining final model with best hyperparameters for {training_timesteps} timesteps...")
        
        # Get best hyperparameters
        best_params = study.best_params
        
        # Create environment
        env = DummyVecEnv([lambda: Monitor(self.create_env())])
        
        # Common configuration
        policy_kwargs = {
            'net_arch': [256, 256, 128],
            'activation_fn': torch.nn.ReLU
        }
        
        # Create model with best hyperparameters
        if self.algorithm == 'PPO':
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/",
                **best_params
            )
        elif self.algorithm == 'SAC':
            model = SAC(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/",
                buffer_size=1000000,
                learning_starts=100,
                **best_params
            )
        elif self.algorithm == 'TD3':
            model = TD3(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/tensorboard/",
                buffer_size=1000000,
                learning_starts=100,
                **best_params
            )
        
        # Train the model
        model.learn(total_timesteps=training_timesteps)
        
        # Save the model
        model_path = f"models/best_tuned_{self.algorithm.lower()}_{self.scenario}"
        model.save(model_path)
        print(f"Best model saved to {model_path}")
        
        # Evaluate the final model
        eval_env = self.create_env()
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        print(f"Final model performance: {mean_reward:.3f} ± {std_reward:.3f}")
        
        env.close()
        eval_env.close()
        
        return model


def run_comprehensive_tuning():
    """
    Run comprehensive hyperparameter tuning for different algorithms and scenarios.
    """
    import os
    os.makedirs("results", exist_ok=True)
    
    # Define tuning configurations
    tuning_configs = [
        {'algorithm': 'PPO', 'scenario': 'basic', 'n_trials': 30},
        {'algorithm': 'SAC', 'scenario': 'basic', 'n_trials': 30},
        {'algorithm': 'PPO', 'scenario': 'full_environment', 'n_trials': 50},
        {'algorithm': 'SAC', 'scenario': 'full_environment', 'n_trials': 50},
    ]
    
    results = {}
    
    for config in tuning_configs:
        print(f"\n{'='*60}")
        print(f"TUNING: {config['algorithm']} on {config['scenario']}")
        print('='*60)
        
        tuner = HyperparameterTuner(
            algorithm=config['algorithm'],
            scenario=config['scenario'],
            n_trials=config['n_trials'],
            training_timesteps=100000,  # Shorter for tuning
            eval_episodes=5
        )
        
        study = tuner.tune()
        results[f"{config['algorithm']}_{config['scenario']}"] = {
            'study': study,
            'best_reward': study.best_value,
            'best_params': study.best_params
        }
        
        # Plot results
        tuner.plot_optimization_history(study)
    
    # Summary of all results
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL TUNING RESULTS")
    print('='*60)
    
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Best Reward: {result['best_reward']:.3f}")
        print(f"  Best Params: {result['best_params']}")
        print()
    
    # Find overall best configuration
    best_config = max(results.items(), key=lambda x: x[1]['best_reward'])
    print(f"Overall Best Configuration: {best_config[0]}")
    print(f"Best Reward: {best_config[1]['best_reward']:.3f}")
    
    return results


def quick_tune(algorithm='PPO', scenario='basic', n_trials=20):
    """
    Quick hyperparameter tuning for testing.
    
    Args:
        algorithm (str): Algorithm to tune
        scenario (str): Scenario to use
        n_trials (int): Number of trials
    """
    tuner = HyperparameterTuner(
        algorithm=algorithm,
        scenario=scenario,
        n_trials=n_trials,
        training_timesteps=50000,  # Quick training
        eval_episodes=3
    )
    
    study = tuner.tune()
    tuner.plot_optimization_history(study)
    
    # Train best model
    print(f"\nTraining best model...")
    best_model = tuner.train_best_model(study, training_timesteps=200000)
    
    return study, best_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Maritime RL Hyperparameter Tuning")
    parser.add_argument("--algorithm", choices=['PPO', 'SAC', 'TD3'], default='PPO',
                        help="RL algorithm to tune")
    parser.add_argument("--scenario", choices=['basic', 'wind_only', 'current_only', 'full_environment'], 
                        default='basic', help="Environment scenario")
    parser.add_argument("--trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--comprehensive", action='store_true', 
                        help="Run comprehensive tuning for all algorithms and scenarios")
    parser.add_argument("--quick", action='store_true', help="Run quick tuning for testing")
    
    args = parser.parse_args()
    
    if args.comprehensive:
        run_comprehensive_tuning()
    elif args.quick:
        quick_tune(args.algorithm, args.scenario, n_trials=10)
    else:
        tuner = HyperparameterTuner(
            algorithm=args.algorithm,
            scenario=args.scenario,
            n_trials=args.trials
        )
        study = tuner.tune()
        tuner.plot_optimization_history(study)
        
        # Ask user if they want to train the best model
        train_best = input("\nTrain best model? (y/n): ").strip().lower() == 'y'
        if train_best:
            tuner.train_best_model(study)
