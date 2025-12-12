import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from gymnasium import Env
from enum import auto, Enum
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

    def render(self, env, start_pos: np.ndarray, all_paths: list, total_rewards: list) -> None:
        """Render and save a visualization of paths taken by the agent.

        Args:
            env: Environment object containing boundaries, obstacles, and checkpoints.
            start_pos (np.ndarray): Starting position of the agent.
            all_paths (list): List of paths taken over multiple episodes.
            total_rewards (list): List of total rewards corresponding to each episode.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Draw environment boundaries
            ax.plot(
                [env.MIN_GRID_POS, env.MAX_GRID_POS, env.MAX_GRID_POS, env.MIN_GRID_POS, env.MIN_GRID_POS],
                [env.MIN_GRID_POS, env.MIN_GRID_POS, env.MAX_GRID_POS, env.MAX_GRID_POS, env.MIN_GRID_POS],
                'k-', label='Boundary'
            )

            # Plot all checkpoints as semi-transparent circles
            checkpoint_positions = np.array([checkpoint['pos'] for checkpoint in env.checkpoints])
            for checkpoint in env.checkpoints:
                circle = plt.Circle(
                    (checkpoint['pos'][0], checkpoint['pos'][1]),
                    radius=10,  # Radius can be tuned to match environment scale
                    color='gray',
                    alpha=0.5,
                    fill=True
                )
                ax.add_patch(circle)

            # Draw ideal straight-line connections between checkpoints
            ax.plot(
                checkpoint_positions[:, 0], checkpoint_positions[:, 1],
                'g--', alpha=0.5, label='Ideal Path'
            )

            # Plot environment obstacles (e.g., waterway)
            ax.add_patch(
                patches.Polygon(
                    env.obstacles,
                    closed=True,
                    edgecolor='r',
                    facecolor='none',
                    lw=2,
                    label='Waterway'
                )
            )

            # Plot additional environment features (e.g., general area)
            ax.add_patch(
                patches.Polygon(
                    env.overall,
                    closed=True,
                    edgecolor='brown',
                    facecolor='none',
                    lw=2,
                    label='Western Scheldt'
                )
            )

            # Plot all agent paths from different episodes
            colors = plt.cm.rainbow(np.linspace(0, 1, len(all_paths)))
            for i, path in enumerate(all_paths):
                ax.plot(
                    path[:, 0], path[:, 1],
                    '-', color=colors[i], alpha=0.7,
                    label=f'Episode {i + 1} (reward: {total_rewards[i]:.1f})'
                )

            # Mark start and target positions
            ax.scatter([start_pos[0]], [start_pos[1]], c='blue', s=100, label='Start')
            ax.scatter([env.target_pos[0]], [env.target_pos[1]], c='red', s=100, label='Target')

            # Final plot adjustments
            ax.set_title('Paths Across Multiple Episodes')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)

            # Set aspect ratio to be equal
            ax.set_aspect('equal', adjustable='box')

            trajectory_file_name = (
                f'{self._model.model_dir}/trajectory.png'
                if self._model.model_dir
                else 'trajectory.png'
            )

            # Save the figure
            plt.savefig(trajectory_file_name, bbox_inches='tight', dpi=300)
            print("\nPath visualization saved as 'trajectory.png'.")

            plt.show()
        finally:
            plt.close('all')


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
            full_policy_file_name = f"{policy_file_name}.zip"
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
            total_rewards = []
            start_pos = None

            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                steps = 0

                start_pos = np.copy(env.ship_pos)

                # Store positions for this episode
                path_positions = [np.copy(env.ship_pos)]

                while not done:
                    # Select action
                    action, _ = self._model.predict(state=state, deterministic=True)

                    # Take step in environment
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    if not done:
                        # Store current position
                        path_positions.append(np.copy(env.ship_pos))

                        # Calculate the average episode reward
                        episode_reward += reward if steps == 0 else (reward - episode_reward) / steps
                        steps += 1

                        # Uncomment for animated evaluation
                        if live_animation:
                            env.render()
                    else:
                        print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
                        break

                all_paths.append(np.array(path_positions))
                total_rewards.append(episode_reward)

            # Print summary statistics
            avg_reward = np.mean(total_rewards)
            print(f"\nEvaluation completed! Average reward: {avg_reward:.2f}")
            print(f"Best episode reward: {max(total_rewards):.2f}")
            print(f"Worst episode reward: {min(total_rewards):.2f}")

            self.render(env, start_pos, all_paths, total_rewards)

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
                prediction = self._model.predict(state=obs, deterministic=True)
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
