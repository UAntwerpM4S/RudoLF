#!/usr/bin/env python3
"""
Simple training script for maritime RL.
This is the easiest way to get started with training your agent.
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maritime_rl_training import MaritimeRLTrainer
from config import TRAINING_RECIPES

def main():
    """Simple interface for training maritime RL agents."""
    
    print("🚢 Maritime RL Training 🚢")
    print("=" * 40)
    
    # Show available training recipes
    print("\nAvailable Training Recipes:")
    recipes = list(TRAINING_RECIPES.keys())
    for i, (name, config) in enumerate(TRAINING_RECIPES.items(), 1):
        print(f"{i}. {name.replace('_', ' ').title()}")
        print(f"   Algorithm: {config['algorithm']}")
        print(f"   Scenario: {config['scenario']}")
        print(f"   Description: {config['description']}")
        print()
    
    # Let user choose
    while True:
        try:
            choice = input(f"Choose a recipe (1-{len(recipes)}) or 'custom' for custom training: ").strip()
            
            if choice.lower() == 'custom':
                return custom_training()
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(recipes):
                recipe_name = recipes[choice_idx]
                return train_with_recipe(recipe_name)
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number or 'custom'.")

def train_with_recipe(recipe_name):
    """Train using a predefined recipe."""
    config = TRAINING_RECIPES[recipe_name]
    
    print(f"\n🚀 Starting {recipe_name.replace('_', ' ').title()} Training")
    print("-" * 50)
    print(f"Algorithm: {config['algorithm']}")
    print(f"Scenario: {config['scenario']}")
    print(f"Timesteps: {config['total_timesteps']:,}")
    print(f"Parallel Environments: {config['n_envs']}")
    print(f"Description: {config['description']}")
    
    # Confirm before starting
    confirm = input("\nProceed with training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Initialize trainer
    trainer = MaritimeRLTrainer(
        algorithm=config['algorithm'],
        n_envs=config['n_envs'],
        verbose=1
    )
    
    # Get scenario settings
    from config import SCENARIOS
    scenario_config = SCENARIOS[config['scenario']]
    
    # Setup environments
    trainer.setup_environments(
        wind=scenario_config['wind'],
        current=scenario_config['current']
    )
    
    # Create and train model
    trainer.create_model()
    
    # Start training with timestamp
    start_time = datetime.now()
    print(f"\n⏰ Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer.train(total_timesteps=config['total_timesteps'])
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"⏰ Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total training time: {duration}")
    
    # Evaluate the trained model
    print("\n📈 Evaluating trained model...")
    trainer.evaluate(n_eval_episodes=10, render=False)
    
    # Ask if user wants to see a demo
    demo = input("\n🎬 Would you like to see a visual demo? (y/n): ").strip().lower()
    if demo == 'y':
        print("Running visual demonstration...")
        trainer.evaluate(n_eval_episodes=1, render=True)
    
    print("\n✅ Training complete!")
    print(f"📁 Models saved in: models/")
    print(f"📊 Logs saved in: logs/")
    print(f"🔍 View training progress: tensorboard --logdir logs/tensorboard/")

def custom_training():
    """Custom training with user-specified parameters."""
    print("\n🛠️  Custom Training Setup")
    print("-" * 30)
    
    # Choose algorithm
    algorithms = ['PPO', 'SAC', 'TD3']
    print("Available algorithms:")
    for i, alg in enumerate(algorithms, 1):
        print(f"{i}. {alg}")
    
    while True:
        try:
            alg_choice = int(input("Choose algorithm (1-3): ")) - 1
            if 0 <= alg_choice < len(algorithms):
                algorithm = algorithms[alg_choice]
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")
    
    # Choose scenario
    from config import SCENARIOS
    scenarios = list(SCENARIOS.keys())
    print(f"\nAvailable scenarios:")
    for i, (name, config) in enumerate(SCENARIOS.items(), 1):
        print(f"{i}. {name.replace('_', ' ').title()}: {config['description']}")
    
    while True:
        try:
            scenario_choice = int(input("Choose scenario (1-4): ")) - 1
            if 0 <= scenario_choice < len(scenarios):
                scenario = scenarios[scenario_choice]
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")
    
    # Choose training duration
    print(f"\nTraining duration presets:")
    durations = {
        1: ("Quick test", 50000),
        2: ("Short training", 200000),
        3: ("Standard training", 500000),
        4: ("Long training", 1000000),
        5: ("Custom", None)
    }
    
    for key, (name, steps) in durations.items():
        if steps:
            print(f"{key}. {name} ({steps:,} timesteps)")
        else:
            print(f"{key}. {name}")
    
    while True:
        try:
            duration_choice = int(input("Choose training duration (1-5): "))
            if duration_choice in durations:
                if duration_choice == 5:
                    timesteps = int(input("Enter custom timesteps: "))
                else:
                    timesteps = durations[duration_choice][1]
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")
    
    # Choose number of environments
    n_envs = 4
    try:
        custom_envs = input(f"Number of parallel environments (default {n_envs}): ").strip()
        if custom_envs:
            n_envs = int(custom_envs)
    except ValueError:
        print(f"Using default: {n_envs}")
    
    # Summary
    print(f"\n📋 Training Configuration Summary:")
    print(f"Algorithm: {algorithm}")
    print(f"Scenario: {scenario.replace('_', ' ').title()}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Parallel Environments: {n_envs}")
    
    confirm = input("\nStart training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    # Start training
    trainer = MaritimeRLTrainer(algorithm=algorithm, n_envs=n_envs, verbose=1)
    
    scenario_config = SCENARIOS[scenario]
    trainer.setup_environments(
        wind=scenario_config['wind'],
        current=scenario_config['current']
    )
    
    trainer.create_model()
    
    start_time = datetime.now()
    print(f"\n⏰ Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer.train(total_timesteps=timesteps)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"⏰ Training completed in: {duration}")
    
    trainer.evaluate(n_eval_episodes=5, render=False)
    print("\n✅ Custom training complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Please check your environment setup and try again.")
