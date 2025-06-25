#!/usr/bin/env python3
"""
Visualization script for trained maritime RL models.
Based on the visualize_trained_model function to create comprehensive visualizations.
"""

import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO, SAC, TD3
from py_sim_env import PySimEnv

class ModelVisualizer:
    """
    Visualizer for trained maritime RL models.
    """
    
    def __init__(self, model_path, algorithm='PPO'):
        """
        Initialize the visualizer.
        
        Args:
            model_path (str): Path to the trained model
            algorithm (str): Algorithm used ('PPO', 'SAC', 'TD3')
        """
        self.model_path = model_path
        self.algorithm = algorithm
        self.model = None
        self.env = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        print(f"Loading {self.algorithm} model from {self.model_path}...")
        
        try:
            if self.algorithm == 'PPO':
                self.model = PPO.load(self.model_path)
            elif self.algorithm == 'SAC':
                self.model = SAC.load(self.model_path)
            elif self.algorithm == 'TD3':
                self.model = TD3.load(self.model_path)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def create_environment(self, wind=False, current=False, render_mode=None):
        """Create environment for visualization."""
        return PySimEnv(
            render_mode=render_mode,
            time_step=0.1,
            max_steps=1500,
            verbose=False,
            wind=wind,
            current=current
        )
    
    def compute_heading_error(self, env):
        """Compute heading error relative to desired path."""
        if env.current_checkpoint < len(env.checkpoints):
            # Calculate desired heading to next checkpoint
            target_pos = env.checkpoints[env.current_checkpoint]['pos']
            direction = target_pos - env.ship_pos
            desired_heading = np.arctan2(direction[1], direction[0])
            
            # Calculate heading error
            heading_error = (desired_heading - env.state[2] + np.pi) % (2 * np.pi) - np.pi
            return heading_error
        return 0.0
    
    def compute_metrics(self, all_cross_errors, all_heading_errors):
        """
        Compute navigation metrics.
        
        Args:
            all_cross_errors: List of cross-track errors for each episode
            all_heading_errors: List of heading errors for each episode
            
        Returns:
            tuple: (mTEI, MTE, mHEI) metrics
        """
        # Mean Track Error Integral (mTEI)
        tei_values = []
        for errors in all_cross_errors:
            if len(errors) > 0:
                tei = np.trapz(np.abs(errors))  # Integral of absolute cross-track error
                tei_values.append(tei)
        mTEI = np.mean(tei_values) if tei_values else 0.0
        
        # Maximum Track Error (MTE)
        all_errors_flat = np.concatenate(all_cross_errors) if all_cross_errors else np.array([])
        MTE = np.max(np.abs(all_errors_flat)) if len(all_errors_flat) > 0 else 0.0
        
        # Mean Heading Error Integral (mHEI)
        hei_values = []
        for errors in all_heading_errors:
            if len(errors) > 0:
                hei = np.trapz(np.abs(errors))  # Integral of absolute heading error
                hei_values.append(hei)
        mHEI = np.mean(hei_values) if hei_values else 0.0
        
        return mTEI, MTE, mHEI
    
    def create_hashed_area(self, obstacles, overall):
        """Create hatched area for visualization."""
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
            
            # Create polygons
            obstacle_poly = Polygon(obstacles)
            overall_poly = Polygon(overall)
            
            # Calculate difference
            difference = overall_poly.difference(obstacle_poly)
            
            # Handle MultiPolygon or Polygon result
            if hasattr(difference, 'geoms'):
                # MultiPolygon
                coords_list = []
                for geom in difference.geoms:
                    if hasattr(geom, 'exterior'):
                        coords_list.append(list(geom.exterior.coords))
            else:
                # Single Polygon
                if hasattr(difference, 'exterior'):
                    coords_list = [list(difference.exterior.coords)]
                else:
                    coords_list = []
            
            return coords_list
            
        except Exception as e:
            print(f"Warning: Could not create hashed area: {e}")
            return []
    
    def visualize_trained_model(self, num_episodes=1, live_animation=False, wind=False, current=False):
        """
        Run and visualize a trained model in the environment over multiple episodes.
        
        Args:
            num_episodes (int): Number of episodes to visualize
            live_animation (bool): Enable live animation of the agent
            wind (bool): Enable wind effects
            current (bool): Enable current effects
        """
        if not self.model:
            raise RuntimeError("No model has been loaded!")
        
        print(f"\nRunning {num_episodes} episodes...")
        print(f"Environment: Wind={wind}, Current={current}")
        
        # Create environment
        env = self.create_environment(
            wind=wind, 
            current=current, 
            render_mode='human' if live_animation else None
        )
        
        # Storage for results
        all_paths = []
        all_heading_errors = []
        total_rewards = []
        all_cross_errors = []
        all_rudder_actions = []
        all_thrust_actions = []
        start_pos = None
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            start_pos = copy.deepcopy(env.ship_pos)
            
            # Storage for this episode
            cross_errors = []
            heading_errors = []
            rudder_actions = []
            thrust_actions = []
            path_positions = [env.ship_pos.copy()]
            cross_errors.append(getattr(env, 'cross_error', 0.0))
            
            # Calculate initial heading error
            initial_heading_error = self.compute_heading_error(env)
            heading_errors.append(initial_heading_error)
            
            while not done:
                # Get action from model
                action, _ = self.model.predict(state, deterministic=True)
                
                # Take step in environment
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store data
                path_positions.append(env.ship_pos.copy())
                if env.current_checkpoint >= 2:
                    cross_errors.append(getattr(env, 'cross_error', 0.0))
                
                heading_error = self.compute_heading_error(env)
                heading_errors.append(heading_error)
                
                # Store actions
                if hasattr(env, 'current_action') and len(env.current_action) >= 2:
                    rudder_actions.append(env.current_action[0])
                    thrust_actions.append(env.current_action[1])
                else:
                    rudder_actions.append(action[0] if len(action) > 0 else 0.0)
                    thrust_actions.append(action[1] if len(action) > 1 else 0.0)
                
                episode_reward += reward
                steps += 1
                
                # Render if live animation
                if live_animation:
                    env.render()
                
                if done:
                    print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
                    break
            
            # Store episode data
            all_paths.append(np.array(path_positions))
            all_cross_errors.append(np.array(cross_errors))
            all_heading_errors.append(np.array(heading_errors))
            all_rudder_actions.append(np.array(rudder_actions))
            all_thrust_actions.append(np.array(thrust_actions))
            total_rewards.append(episode_reward)
        
        # Calculate metrics
        mTEI, MTE, mHEI = self.compute_metrics(all_cross_errors, all_heading_errors)
        
        # Print summary statistics
        avg_reward = np.mean(total_rewards)
        print(f"\n=== Evaluation Results ===")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Best episode reward: {max(total_rewards):.2f}")
        print(f"Worst episode reward: {min(total_rewards):.2f}")
        print(f"Mean Track Error Integral (mTEI): {mTEI:.2f}")
        print(f"Maximum Track Error (MTE): {MTE:.2f}")
        print(f"Mean Heading Error Integral (mHEI): {mHEI:.2f} rad")
        
        # Create visualization
        self.create_visualization(env, all_paths, all_cross_errors, all_heading_errors, 
                                all_rudder_actions, all_thrust_actions, total_rewards, 
                                start_pos, num_episodes)
        
        env.close()
        return avg_reward, mTEI, MTE, mHEI
    
    def create_visualization(self, env, all_paths, all_cross_errors, all_heading_errors, 
                           all_rudder_actions, all_thrust_actions, total_rewards, 
                           start_pos, num_episodes):
        """Create comprehensive visualization plots."""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Add hatched pattern to first subplot
            difference_coords = self.create_hashed_area(env.obstacles, env.overall)
            for coords in difference_coords:
                if coords:  # Check if coords is not empty
                    ax1.add_patch(patches.Polygon(
                        coords,
                        facecolor='lightblue',
                        edgecolor='gray',
                        hatch='///',
                        alpha=0.3,
                        label='Shallow water area' if coords == difference_coords[0] else ""
                    ))
            
            # Plot checkpoints
            checkpoint_positions = [checkpoint['pos'] for checkpoint in env.checkpoints]
            checkpoint_positions = np.array(checkpoint_positions)
            for i, checkpoint in enumerate(env.checkpoints):
                circle = plt.Circle((checkpoint['pos'][0], checkpoint['pos'][1]),
                                  radius=15,
                                  color='gray',
                                  alpha=0.5,
                                  fill=True)
                ax1.add_patch(circle)
                # Add checkpoint numbers
                ax1.text(checkpoint['pos'][0], checkpoint['pos'][1], str(i), 
                        ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Draw the path between checkpoints
            ax1.plot(checkpoint_positions[:, 0], checkpoint_positions[:, 1],
                    'g--', alpha=0.7, linewidth=2, label='Planned Path')
            
            # Plot obstacles
            polygon_patch = patches.Polygon(env.obstacles, closed=True,
                                          edgecolor='red', facecolor='none',
                                          linewidth=2, label='Navigation Channel')
            ax1.add_patch(polygon_patch)
            
            western_scheldt = patches.Polygon(env.overall, closed=True,
                                            edgecolor='brown', facecolor='none',
                                            linewidth=2, label='Western Scheldt')
            ax1.add_patch(western_scheldt)
            
            # Plot paths for each episode
            colors = plt.cm.rainbow(np.linspace(0, 1, max(num_episodes, 1)))
            for i, path in enumerate(all_paths):
                ax1.plot(path[:, 0], path[:, 1], '-',
                        color=colors[i] if num_episodes > 1 else 'blue', 
                        alpha=0.7, linewidth=2,
                        label=f'Episode {i + 1} (R: {total_rewards[i]:.1f})')
            
            # Plot start and target positions
            if start_pos is not None:
                ax1.scatter([start_pos[0]], [start_pos[1]], c='green', s=150, 
                          marker='o', label='Start', zorder=5)
            ax1.scatter([env.target_pos[0]], [env.target_pos[1]], c='red', s=150, 
                      marker='*', label='Target', zorder=5)
            
            ax1.set_title(f'Ship Trajectory Visualization\n{self.algorithm} Model Performance')
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # Plot 2: Cross-track Error
            self.plot_averaged_data(ax2, all_cross_errors, colors, num_episodes, 
                                  'Cross-Track Error Over Time', 'Timestep', 'Cross-Track Error (m)')
            
            # Plot 3: Rudder Actions
            self.plot_averaged_data(ax3, all_rudder_actions, colors, num_episodes,
                                  'Rudder Actions Over Time', 'Timestep', 'Rudder Action')
            
            # Plot 4: Thrust Actions
            self.plot_averaged_data(ax4, all_thrust_actions, colors, num_episodes,
                                  'Thrust Actions Over Time', 'Timestep', 'Thrust Action')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'trajectory_visualization_{self.algorithm}_{timestamp}.png'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.show()
            
            print(f"\n📊 Visualization saved as '{filename}'")
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
        finally:
            plt.close('all')
    
    def plot_averaged_data(self, ax, all_data, colors, num_episodes, title, xlabel, ylabel):
        """Plot individual and averaged data."""
        if not all_data:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return
        
        # Find maximum length
        max_length = max(len(data) for data in all_data) if all_data else 0
        
        if max_length == 0:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return
        
        # Plot individual episodes
        for i, data in enumerate(all_data):
            timesteps = np.arange(len(data))
            ax.plot(timesteps, data, '-',
                   color=colors[i] if num_episodes > 1 else 'blue', 
                   alpha=0.3,
                   label=f'Episode {i + 1}' if i == 0 else "")
        
        # Calculate and plot average
        data_sum = np.zeros(max_length)
        data_count = np.zeros(max_length)
        
        for data in all_data:
            for i, value in enumerate(data):
                data_sum[i] += value
                data_count[i] += 1
        
        avg_data = np.divide(data_sum, data_count, out=np.zeros_like(data_sum), where=data_count != 0)
        timesteps = np.arange(len(avg_data))
        ax.plot(timesteps, avg_data, '-',
               color='black', alpha=0.8, linewidth=2,
               label='Average')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)


def main():
    """Main visualization function."""
    print("🚢 Maritime RL Model Visualization 🚢")
    print("=" * 50)
    
    # List available models
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found!")
        print("Please train a model first using the training scripts.")
        return
    
    # Find model files
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.zip'):
            model_files.append(file)
    
    if not model_files:
        print(f"❌ No model files found in '{models_dir}'!")
        print("Please train a model first using the training scripts.")
        return
    
    print(f"Available models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    
    # Let user choose model
    while True:
        try:
            choice = input(f"\nChoose a model (1-{len(model_files)}): ").strip()
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(model_files):
                selected_model = model_files[model_idx]
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")
    
    # Determine algorithm from filename
    model_path = os.path.join(models_dir, selected_model)
    algorithm = 'PPO'  # Default
    if 'ppo' in selected_model.lower():
        algorithm = 'PPO'
    elif 'sac' in selected_model.lower():
        algorithm = 'SAC'
    elif 'td3' in selected_model.lower():
        algorithm = 'TD3'
    
    print(f"Detected algorithm: {algorithm}")
    
    # Choose visualization options
    print(f"\nVisualization options:")
    print("1. Single episode, no animation")
    print("2. Single episode with live animation")
    print("3. Multiple episodes comparison")
    
    while True:
        try:
            vis_choice = input("Choose option (1-3): ").strip()
            if vis_choice in ['1', '2', '3']:
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter 1, 2, or 3.")
    
    # Choose environment conditions
    print(f"\nEnvironment conditions:")
    print("1. Basic (no wind/current)")
    print("2. Wind only")
    print("3. Current only") 
    print("4. Full environment (wind + current)")
    
    while True:
        try:
            env_choice = input("Choose conditions (1-4): ").strip()
            if env_choice in ['1', '2', '3', '4']:
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter 1, 2, 3, or 4.")
    
    # Set environment parameters
    wind_conditions = {'1': False, '2': True, '3': False, '4': True}
    current_conditions = {'1': False, '2': False, '3': True, '4': True}
    
    wind = wind_conditions[env_choice]
    current = current_conditions[env_choice]
    
    # Set visualization parameters
    if vis_choice == '1':
        num_episodes = 1
        live_animation = False
    elif vis_choice == '2':
        num_episodes = 1
        live_animation = True
    else:  # vis_choice == '3'
        while True:
            try:
                num_episodes = int(input("Number of episodes to compare (2-10): "))
                if 2 <= num_episodes <= 10:
                    break
                else:
                    print("Please enter a number between 2 and 10.")
            except ValueError:
                print("Please enter a valid number.")
        live_animation = False
    
    # Run visualization
    try:
        print(f"\n🚀 Starting visualization...")
        print(f"Model: {selected_model}")
        print(f"Algorithm: {algorithm}")
        print(f"Episodes: {num_episodes}")
        print(f"Live animation: {live_animation}")
        print(f"Wind: {wind}, Current: {current}")
        
        visualizer = ModelVisualizer(model_path, algorithm)
        avg_reward, mTEI, MTE, mHEI = visualizer.visualize_trained_model(
            num_episodes=num_episodes,
            live_animation=live_animation,
            wind=wind,
            current=current
        )
        
        print(f"\n✅ Visualization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
