import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from gymnasium import Env
from enum import auto, Enum
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME
from fw.stop_condition import StopCondition
from fw.policies.base_model import BaseModel
from fw.policies.ppo2_model import PPO2Model
from fw.policies.ppo_model import PPOModel


class PolicyStrategy(Enum):
    """Defines strategies for handling policies during environment switching.

    Attributes:
        RESET_POLICY: Start with a fresh policy.
        REUSE_CURRENT_POLICY: Continue training with the current policy.
        REUSE_OTHER_POLICY: Load and use a policy trained on another environment.
    """
    RESET_POLICY = auto()
    REUSE_CURRENT_POLICY = auto()
    REUSE_OTHER_POLICY = auto()


class Agent:
    """Agent class for managing models and environments during training.

    This class includes methods for training, saving/loading policies, and
    visualizing paths taken by the agent during evaluation.
    """

    def __init__(self, model_type: str, hyperparameters: dict):
        """Initialize the Agent.

        Args:
            model_type (str): Type of model to use (e.g., 'PPO').
            hyperparameters (dict): Hyperparameters for configuring the model.
        """
        self._model = None
        self._model_type = model_type
        self._hyperparameters = hyperparameters


    @property
    def model_type(self) -> str:
        """
        Get the type of model being used by the agent.

        Returns:
            str: The model type.
        """
        return self._model_type


    @property
    def model(self) -> BaseModel:
        """
        Get the model that is used by the agent.

        Returns:
            BaseModel: The model.
        """
        return self._model


    def set_environment(self, env: Env, policy_strategy: PolicyStrategy = PolicyStrategy.RESET_POLICY,
                        other_env_name: str = None) -> None:
        """
        Assign an environment to the model and configure the policy strategy.

        Args:
            env (Env): The environment to set.
            policy_strategy (PolicyStrategy, optional): Strategy for handling policy (reset, reuse current, or reuse other). Default is PolicyStrategy.RESET_POLICY.
            other_env_name (str, optional): Name of the environment whose policy should be reused (if applicable).

        Raises:
            RuntimeError: If an unknown policy type is specified.
        """
        policy_file_name = None

        if policy_strategy == PolicyStrategy.REUSE_CURRENT_POLICY and self._model:
            policy_file_name = f'{self.get_env().type_name}_{self._model_type}_policy'
            self._model.save_policy(policy_file_name)
        elif policy_strategy == PolicyStrategy.REUSE_OTHER_POLICY and other_env_name:
            policy_file_name = f'{other_env_name}_{self._model_type}_policy'

        if PPO_POLICY_NAME == self._model_type:
            self._model = PPOModel(environment=env, **self._hyperparameters)
        elif PPO2_POLICY_NAME == self._model_type:
            self._model = PPO2Model(environment=env, **self._hyperparameters)
        else:
            raise RuntimeError(f"Unknown {self._model_type} policy type.")

        if policy_file_name:
            full_policy_file_name = ".".join([policy_file_name, "zip"])
            if self.model.model_dir:
                full_policy_file_name = os.path.join(self.model.model_dir, full_policy_file_name)

            if Path(full_policy_file_name).exists():
                print(f"Reusing {policy_file_name}.")
                self._model.load_policy(policy_file_name)
            else:
                raise RuntimeError(f"The policy file {policy_file_name} does not exist.")


    def train(self, stop_condition: StopCondition) -> None:
        """
        Train the agent in the assigned environment.

        Args:
            stop_condition (StopCondition): The condition that determines when training should stop.

        Raises:
            RuntimeError: If no model is created.
        """
        if not self._model:
            raise RuntimeError("No model has been created yet!")

        # Start training
        self._model.learn(
            stop_condition=stop_condition,
        )

        # Cleanup the environment
        if hasattr(self.get_env(), "fh_sim"):
            self.get_env().fh_sim.dispose()


    @staticmethod
    def compute_heading_error(env):
        """
        Compute heading error relative to desired path direction
        """
        # Get current checkpoint index and position
        curr_checkpoint = env.checkpoints[env.current_checkpoint]
        next_checkpoint = env.checkpoints[min(env.current_checkpoint + 1, len(env.checkpoints) - 1)]

        # Calculate desired heading (angle of path segment)
        desired_heading = np.arctan2(
            next_checkpoint['pos'][1] - curr_checkpoint['pos'][1],
            next_checkpoint['pos'][0] - curr_checkpoint['pos'][0]
        )

        # Calculate heading error (difference between current and desired heading)
        heading_error = (env.state[2] - desired_heading + np.pi) % (2 * np.pi) - np.pi
        return heading_error


    @staticmethod
    def compute_metrics(cross_errors, heading_errors):
        """
        Compute Mean Track Error Integral (mTEI), Maximum Track Error (MTE),
        and Mean Heading Error Integral (mHEI)

        mTEI = (1/(N1-N0+1)) * sum(|e(k)|)
        MTE = max{|e(N0)|, |e(N0+1)|, ..., |e(N1)|}
        mHEI = (1/(N1-N0+1)) * sum(|e_psi(k)|)
        """
        all_mTEIs = []
        all_MTEs = []
        all_mHEIs = []

        for errors, heading_errs in zip(cross_errors, heading_errors):
            # Calculate mTEI
            n = len(errors)
            mTEI = np.sum(np.abs(errors)) / n
            all_mTEIs.append(mTEI)

            # Calculate MTE
            MTE = np.max(np.abs(errors))
            all_MTEs.append(MTE)

            # Calculate mHEI
            mHEI = np.sum(np.abs(heading_errs)) / n
            all_mHEIs.append(mHEI)

        return np.mean(all_mTEIs), np.mean(all_MTEs), np.mean(all_mHEIs)


    # Plot paths in the first subplot
    # Create hashed pattern for the area between obstacles and Western Scheldt
    @staticmethod
    def create_hashed_area(obstacles, overall):
        obstacle_poly = Polygon(obstacles)
        overall_poly = Polygon(overall)
        difference = overall_poly.difference(obstacle_poly)

        if isinstance(difference, MultiPolygon):
            coords = []
            for poly in difference.geoms:
                coords.append(np.array(poly.exterior.coords))
        else:
            coords = [np.array(difference.exterior.coords)]

        return coords


    def visualize_trained_model(self, num_episodes: int=5, live_animation=False) -> None:
        """
        Run and visualize a trained model in the environment over multiple episodes.

        Args:
            num_episodes (int, optional): Number of episodes to visualize (default is 5).
            live_animation (bool, optional): Enable live animation of the agent in the environment.

        Raises:
            RuntimeError: If no model is created.
        """
        # Load checkpoint
        # self._model.policy = PPOPolicy.load("PySimEnv_PPO_policy", device="cpu")

        if not self._model:
            raise RuntimeError("No model has been created yet!")

        print(f"\nRunning {num_episodes} episodes...")

        self.model.set_policy_eval()
        env = self.get_env()

        if hasattr(env, "ship_pos"):
            # Store results for each episode
            all_paths = []
            all_heading_errors = []
            total_rewards = []
            all_cross_errors = []  # Store cross-track errors for each episode
            all_rudder_actions = []
            all_thrust_actions = []
            start_pos = None

            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                steps = 0

                start_pos = copy.deepcopy(env.ship_pos)

                # Store heading errors for this episode
                cross_errors = []
                heading_errors = []
                rudder_actions = []
                thrust_actions = []
                path_positions = [env.ship_pos.copy()]  # Store initial position
                cross_errors.append(env.cross_error)  # Store initial cross-track error

                # Calculate initial heading error relative to desired path
                initial_heading_error = self.compute_heading_error(env)
                heading_errors.append(initial_heading_error)

                while not done:
                    # Select action
                    action, _ = self._model.predict(state)

                    # Take step in environment
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # Store current position and errors
                    path_positions.append(env.ship_pos.copy())
                    if env.current_checkpoint >= 2:
                        cross_errors.append(env.cross_error)
                    heading_error = self.compute_heading_error(env)
                    heading_errors.append(heading_error)
                    rudder_actions.append(env.current_action[0])
                    thrust_actions.append(env.current_action[1])

                    episode_reward += reward if steps == 0 else (reward - episode_reward) / steps
                    steps += 1

                    # Uncomment for animated evaluation
                    if live_animation:
                        env.render()

                    if done:
                        print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
                        break

                all_paths.append(np.array(path_positions))
                all_cross_errors.append(np.array(cross_errors))
                all_heading_errors.append(np.array(heading_errors))
                all_rudder_actions.append(np.array(rudder_actions))
                all_thrust_actions.append(np.array(thrust_actions))
                total_rewards.append(episode_reward)

            # Calculate mTEI, MTE, and mHEI
            mTEI, MTE, mHEI = self.compute_metrics(all_cross_errors, all_heading_errors)

            # Print summary statistics
            avg_reward = np.mean(total_rewards)
            print(f"\nEvaluation completed!")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Best episode reward: {max(total_rewards):.2f}")
            print(f"Worst episode reward: {min(total_rewards):.2f}")
            print(f"Mean Track Error Integral (mTEI): {mTEI:.2f}")
            print(f"Maximum Track Error (MTE): {MTE:.2f}")
            print(f"Mean Heading Error Integral (mHEI): {mHEI:.2f} rad")

            try:
                # Create a figure with two subplots
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))

                # Add hatched pattern to first subplot
                difference_coords = self.create_hashed_area(env.obstacles, env.overall)
                for coords in difference_coords:
                    ax1.add_patch(patches.Polygon(
                        coords,
                        facecolor='none',
                        edgecolor='gray',
                        hatch='///',
                        alpha=0.3,
                        label='Low water level area' if coords is difference_coords[0] else ""
                    ))

                # Plot checkpoints
                checkpoint_positions = [checkpoint['pos'] for checkpoint in env.checkpoints]
                checkpoint_positions = np.array(checkpoint_positions)
                for checkpoint in env.checkpoints:
                    circle = plt.Circle((checkpoint['pos'][0], checkpoint['pos'][1]),
                                        radius=10,
                                        color='gray',
                                        alpha=0.5,
                                        fill=True)
                    ax1.add_patch(circle)

                # Draw the path between checkpoints
                ax1.plot(checkpoint_positions[:, 0], checkpoint_positions[:, 1],
                         'g--', alpha=0.5, label='Ideal Path')

                # Plot obstacles
                polygon_patch = patches.Polygon(env.obstacles, closed=True,
                                                edgecolor='r', facecolor='none',
                                                lw=2, label='Waterway')
                ax1.add_patch(polygon_patch)

                western_scheldt = patches.Polygon(env.overall, closed=True,
                                                  edgecolor='brown', facecolor='none',
                                                  lw=2, label='Western Scheldt')
                ax1.add_patch(western_scheldt)

                # Plot paths for each episode with different colors
                colors = plt.cm.rainbow(np.linspace(0, 1, num_episodes))
                for i, path in enumerate(all_paths):
                    ax1.plot(path[:, 0], path[:, 1], '-',
                             color=colors[i], alpha=0.4,
                             label=f'Episode {i + 1} (reward: {total_rewards[i]:.1f})')

                # Plot start and target positions
                ax1.scatter([start_pos[0]], [start_pos[1]], c='blue', s=100, label='Start')
                ax1.scatter([env.target_pos[0]], [env.target_pos[1]], c='red', s=100, label='Target')

                ax1.set_title('Trajectory with Heading Control')
                ax1.set_xlabel('X Position')
                ax1.set_ylabel('Y Position')
                ax1.legend(loc='upper left')
                ax1.grid(True)
                ax1.axis('equal')

                # Compute average errors and actions across episodes
                # First, find the maximum length among all episodes
                max_error_length = max(len(errors) for errors in all_cross_errors)
                max_rudder_length = max(len(actions) for actions in all_rudder_actions)
                max_thrust_length = max(len(actions) for actions in all_thrust_actions)

                # Initialize arrays to store the sum and count for averaging
                error_sum = np.zeros(max_error_length)
                error_count = np.zeros(max_error_length)
                rudder_sum = np.zeros(max_rudder_length)
                rudder_count = np.zeros(max_rudder_length)
                thrust_sum = np.zeros(max_thrust_length)
                thrust_count = np.zeros(max_thrust_length)

                # Sum up all values and count occurrences at each timestep
                for errors in all_cross_errors:
                    for i, error in enumerate(errors):
                        error_sum[i] += error
                        error_count[i] += 1

                for actions in all_rudder_actions:
                    for i, action in enumerate(actions):
                        rudder_sum[i] += action
                        rudder_count[i] += 1

                for actions in all_thrust_actions:
                    for i, action in enumerate(actions):
                        thrust_sum[i] += action
                        thrust_count[i] += 1

                # Calculate averages (avoid division by zero)
                avg_errors = np.divide(error_sum, error_count, out=np.zeros_like(error_sum), where=error_count != 0)
                avg_rudder = np.divide(rudder_sum, rudder_count, out=np.zeros_like(rudder_sum), where=rudder_count != 0)
                avg_thrust = np.divide(thrust_sum, thrust_count, out=np.zeros_like(thrust_sum), where=thrust_count != 0)

                # Plot individual episode errors with lower alpha
                for i, errors in enumerate(all_cross_errors):
                    timesteps = np.arange(len(errors))
                    ax2.plot(timesteps, errors, '-',
                             color=colors[i], alpha=0.2,
                             label=f'Episode {i + 1}' if i == 0 else "")

                # Plot average errors with higher alpha and thicker line
                timesteps = np.arange(len(avg_errors))
                ax2.plot(timesteps, avg_errors, '-',
                         color='black', alpha=0.2,
                         label='Average Cross-track Error')

                ax2.set_title(f'Cross-Track Error Over Time (Individual and Average Policy)')
                ax2.set_xlabel('Timestep')
                ax2.set_ylabel('Cross-Track Error')
                ax2.legend(loc='upper right')
                ax2.grid(True)

                # Plot individual episode rudder actions with lower alpha
                for i, actions in enumerate(all_rudder_actions):
                    timesteps = np.arange(len(actions))
                    ax3.plot(timesteps, actions, '-',
                             color=colors[i], alpha=0.2,
                             label=f'Episode {i + 1}' if i == 0 else "")

                # Plot average rudder actions with higher alpha and thicker line
                timesteps = np.arange(len(avg_rudder))
                ax3.plot(timesteps, avg_rudder, '-',
                         color='black', alpha=0.2,
                         label='Average Rudder Action')

                ax3.set_title(f'Rudder Actions Over Time (Individual and Average Policy)')
                ax3.set_xlabel('Timestep')
                ax3.set_ylabel('Rudder Action')
                ax3.legend(loc='upper right')
                ax3.grid(True)

                # Plot individual episode thrust actions with lower alpha
                for i, actions in enumerate(all_thrust_actions):
                    timesteps = np.arange(len(actions))
                    ax4.plot(timesteps, actions, '-',
                             color=colors[i], alpha=0.2,
                             label=f'Episode {i + 1}' if i == 0 else "")

                # Plot average thrust actions with higher alpha and thicker line
                timesteps = np.arange(len(avg_thrust))
                ax4.plot(timesteps, avg_thrust, '-',
                         color='black', alpha=0.2,
                         label='Average Thrust Action')

                ax4.set_title(f'Thrust Actions Over Time (Individual and Average Policy)')
                ax4.set_xlabel('Timestep')
                ax4.set_ylabel('Thrust Action')
                ax4.legend(loc='upper right')
                ax4.grid(True)

                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(f'trajectory.png', bbox_inches='tight', dpi=300)
                plt.show()

                print(f"\nPath and error visualization saved as 'trajectory.png'")
            finally:
                plt.close('all')

                # Cleanup the environment
                if hasattr(env, "fh_sim"):
                    env.fh_sim.dispose()
        else:
            # Reset the environment
            obs, _ = env.reset()

            # Render the environment before starting the loop
            env.render()

            # Run the agent in the environment
            done = False
            while not done:
                # The agent chooses an action based on the current observation
                prediction = self._model.predict(obs)
                if len(prediction) == 3:
                    action, _, _ = prediction
                else:
                    action, _ = prediction

                # Take a step in the environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Render the environment each step to visualize the agent's action
                env.render()


    def get_env(self):
        """
        Retrieve the environment associated with the agent's model.

        Returns:
            The environment object.
        """
        return self._model.get_env()
