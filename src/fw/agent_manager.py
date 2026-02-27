import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from gymnasium import Env
from enum import auto, Enum
from matplotlib.colors import to_rgb
from contextlib import contextmanager
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME
from fw.simulators.airsim_bridge import AirSimBridge
from fw.stop_condition import StopCondition
from fw.policies.base_agent import BaseAgent
from fw.policies.ppo2_agent import PPO2Agent
from fw.policies.ppo_agent import PPOAgent


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


def normalized_time_axis(length: int) -> np.ndarray:
    """Return a normalized time axis in [0, 1] for a trajectory of given length."""
    if length <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 100.0, length)


def shade_color(base_color, n, lo=0.45, hi=1.0):
    """
    Generate n shaded variants of a base color by scaling lightness.
    """
    rgb = np.array(to_rgb(base_color))
    factors = np.linspace(hi, lo, n)

    return [tuple(np.clip(rgb * f, 0, 1)) for f in factors]


def resample_to_common_time(signal: np.ndarray, common_time: np.ndarray) -> np.ndarray:
    """
    Resample a 1D signal defined over normalized time [0,1] onto a common grid.
    """
    if len(signal) == 1:
        return np.full_like(common_time, signal[0])

    original_time = np.linspace(0.0, 1.0, len(signal))
    return np.interp(common_time, original_time, signal)


class AgentManager:
    """AgentManager class for managing agents and environments during training.

    This class includes methods for training, saving/loading policies, and
    visualizing paths taken by the agent during evaluation.
    """

    ENV_COLORMAPS = [
        "#ff7f0e",  # orange
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#17becf",  # cyan
    ]

    def __init__(self, agent_type: str, hyperparameters: dict):
        """Initialize the Agent.

        Args:
            agent_type (str): Type of agent to use (e.g., 'PPO').
            hyperparameters (dict): Hyperparameters for configuring the agent.
        """
        self._agent = None
        self._agent_type = agent_type
        self.prev_smoothing_setting = None
        self._hyperparameters = hyperparameters

        self._fig = None
        self._axes = None
        self._overlay_active = False


    def begin_overlay_plot(self):
        """Create a persistent matplotlib figure for overlaying multiple environments."""
        if self._fig is None:
            self._fig, self._axes = plt.subplots(4, 1, figsize=(12, 16))
            self._overlay_active = True

    def end_overlay_plot(self):
        """Finalize and close the persistent matplotlib figure."""
        if self._fig is not None:
            plt.tight_layout()
            plt.show()
            plt.close(self._fig)

            self._fig = None
            self._axes = None
            self._overlay_active = False


    @property
    def agent_type(self) -> str:
        """
        Get the type of agent being used by the agent manager.

        Returns:
            str: The agent type.
        """
        return self._agent_type


    @property
    def agent(self) -> BaseAgent:
        """
        Get the agent that is used by the agent manager.

        Returns:
            BaseAgent: The agent.
        """
        return self._agent


    def set_environment(self, env: Env, policy_strategy: PolicyStrategy = PolicyStrategy.RESET_POLICY,
                        other_env_name: str = None) -> None:
        """
        Assign an environment to the agent and configure the policy strategy.

        Args:
            env (Env): The environment to set.
            policy_strategy (PolicyStrategy, optional): Strategy for handling policy (reset, reuse current, or reuse other). Default is PolicyStrategy.RESET_POLICY.
            other_env_name (str, optional): Name of the environment whose policy should be reused (if applicable).

        Raises:
            RuntimeError: If an unknown policy type is specified.
        """
        policy_file_name = None

        if policy_strategy == PolicyStrategy.REUSE_CURRENT_POLICY and self._agent:
            policy_file_name = f'{self.get_env().type_name}_{self._agent_type}_policy'
            self._agent.save_policy(policy_file_name)
        elif policy_strategy == PolicyStrategy.REUSE_OTHER_POLICY and other_env_name:
            policy_file_name = f'{other_env_name}_{self._agent_type}_policy'

        if PPO_POLICY_NAME == self._agent_type:
            self._agent = PPOAgent(environment=env, **self._hyperparameters)
        elif PPO2_POLICY_NAME == self._agent_type:
            self._agent = PPO2Agent(environment=env, **self._hyperparameters)
        else:
            raise RuntimeError(f"Unknown {self._agent_type} policy type.")

        if policy_file_name:
            full_policy_file_name = f"{policy_file_name}.zip"
            if self.agent.model_dir:
                full_policy_file_name = os.path.join(self.agent.model_dir, full_policy_file_name)

            if Path(full_policy_file_name).exists():
                print(f"Reusing {policy_file_name}.")
                self._agent.load_policy(policy_file_name)
            else:
                raise RuntimeError(f"The policy file {policy_file_name} does not exist.")


    def train(self, stop_condition: StopCondition) -> None:
        """
        Train the agent in the assigned environment.

        Args:
            stop_condition (StopCondition): The condition that determines when training should stop.

        Raises:
            RuntimeError: If no agent is created.
        """
        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        # Start training
        self._agent.learn(
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
        curr_checkpoint = env.checkpoints[env.checkpoint_index]
        next_checkpoint = env.checkpoints[min(env.checkpoint_index + 1, len(env.checkpoints) - 1)]

        # Calculate desired heading (angle of path segment)
        desired_heading = np.arctan2(
            next_checkpoint['pos'][1] - curr_checkpoint['pos'][1],
            next_checkpoint['pos'][0] - curr_checkpoint['pos'][0]
        )

        # Calculate heading error (difference between current and desired heading)
        heading_error = (env.phys_sim.state.heading - desired_heading + np.pi) % (2 * np.pi) - np.pi
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


    @contextmanager
    def smoothing(self, env):
        if hasattr(env, "enable_smoothing"):
            try:
                self.prev_smoothing_setting = env.enable_smoothing
                env.enable_smoothing = False
                yield
            finally:
                env.enable_smoothing = self.prev_smoothing_setting
                self.prev_smoothing_setting = None

    def visualize_trained_agent(
            self,
            num_episodes: int = 5,
            live_animation: bool = False,
            env_color_index: int = 0,
            airsim_bridge: AirSimBridge = None,
    ) -> None:

        """
        Run and visualize a trained agent in the environment over multiple episodes.

        Args:
            num_episodes (int, optional): Number of episodes to visualize (default is 5).
            live_animation (bool, optional): Enable live animation of the agent in the environment.
            env_color_index (int, optional): Index of the environment color.
            airsim_bridge (AirSimBridge, optional): AirSimBridge to create a connection with the Cosys AirSim.

        Raises:
            RuntimeError: If no agent is created.
        """
        # Load checkpoint
        # self._agent.policy = PPOPolicy.load("PySimEnv_PPO_policy", device_str="cpu")

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        print(f"\nRunning {num_episodes} episodes...")

        self.agent.set_policy_eval()
        env = self.get_env()

        with self.smoothing(env):
            if hasattr(env, "ship_pos"):
                # Store results for each episode
                all_paths = []
                all_heading_errors = []
                total_rewards = []
                all_cross_errors = []  # Store cross-track errors for each episode
                all_rudder_actions = []
                all_ukc_measurements = []
                start_pos = None

                for episode in range(num_episodes):
                    state, _ = env.reset()
                    episode_reward = 0
                    done = False
                    steps = 0

                    start_pos = np.copy(env.ship_pos)

                    # Store heading errors for this episode
                    cross_errors = []
                    heading_errors = []
                    rudder_actions = []
                    ukc_measurements = []
                    path_positions = [np.copy(env.ship_pos)]  # Store initial position
                    cross_errors.append(env.cross_error)  # Store initial cross-track error

                    # Calculate initial heading error relative to desired path
                    initial_heading_error = self.compute_heading_error(env)
                    heading_errors.append(initial_heading_error)

                    while not done:
                        # Select action
                        action, _ = self._agent.predict(state=state, deterministic=True)

                        # Take step in environment
                        state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        if not done:
                            # Store current position and errors
                            path_positions.append(np.copy(env.ship_pos))
                            if env.checkpoint_index >= 2:
                                cross_errors.append(env.cross_error)
                            heading_error = self.compute_heading_error(env)
                            heading_errors.append(heading_error)
                            rudder_actions.append(env.performed_action[0])
                            ukc_measurements.append(env.ukc)

                            episode_reward += reward if steps == 0 else (reward - episode_reward) / steps
                            steps += 1

                            # Uncomment for animated evaluation
                            if live_animation:
                                env.render()
                        else:
                            print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
                            break

                    all_paths.append(np.array(path_positions))
                    all_cross_errors.append(np.array(cross_errors))
                    all_heading_errors.append(np.array(heading_errors))
                    all_rudder_actions.append(np.array(rudder_actions))
                    all_ukc_measurements.append(np.array(ukc_measurements))
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
                    if self._overlay_active and self._fig is not None:
                        ax1, ax2, ax3, ax4 = self._axes
                    else:
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

                    # Select colormap for this environment
                    env_cmap = self.ENV_COLORMAPS[env_color_index % len(self.ENV_COLORMAPS)]

                    # Plot paths for each episode with different colors
                    colors = shade_color(env_cmap, num_episodes)

                    line_style = ["-", "--", "-.", ":"][env_color_index % 4]

                    for i, path in enumerate(all_paths):
                        ax1.plot(path[:, 0], path[:, 1], '-',
                                 color=colors[i], alpha=0.4,
                                 label=f'{env.type_name} – Ep {i + 1} (R={total_rewards[i]:.1f})',
                                 linestyle=line_style)

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
                    max_ukc_length = max(len(actions) for actions in all_ukc_measurements)

                    # Initialize arrays to store the sum and count for averaging
                    error_sum = np.zeros(max_error_length)
                    error_count = np.zeros(max_error_length)
                    rudder_resampled = []
                    ukc_sum = np.zeros(max_ukc_length)
                    ukc_count = np.zeros(max_ukc_length)

                    COMMON_TIME = np.linspace(0.0, 1.0, 200)

                    # Sum up all values and count occurrences at each timestep
                    for errors in all_cross_errors:
                        for i, error in enumerate(errors):
                            error_sum[i] += error
                            error_count[i] += 1

                    for actions in all_rudder_actions:
                        rudder_resampled.append(
                            resample_to_common_time(actions, COMMON_TIME)
                        )

                    for ukcs in all_ukc_measurements:
                        for i, ukc in enumerate(ukcs):
                            ukc_sum[i] += ukc
                            ukc_count[i] += 1

                    # Calculate averages (avoid division by zero)
                    avg_errors = np.divide(error_sum, error_count, out=np.zeros_like(error_sum), where=error_count != 0)
                    avg_ukcs = np.divide(ukc_sum, ukc_count, out=np.zeros_like(ukc_sum), where=ukc_count != 0)

                    # Plot individual episode errors with lower alpha
                    for i, errors in enumerate(all_cross_errors):
                        timesteps = normalized_time_axis(len(errors))
                        ax2.plot(timesteps, errors, '-',
                                 color=colors[i], alpha=0.2,
                                 label=f'Episode {i + 1}' if i == 0 else "", linestyle=line_style)

                    # Plot average errors with higher alpha and thicker line
                    timesteps = normalized_time_axis(len(avg_errors))
                    ax2.plot(timesteps, avg_errors, '-',
                             color='black', alpha=0.2,
                             label='Average Cross-track Error', linestyle=line_style)

                    ax2.set_title(f'Cross-Track Error Over Time (Individual and Average Policy)')
                    ax2.set_xlabel('Episode progress [%]')
                    ax2.set_ylabel('Cross-Track Error')
                    ax2.legend(loc='upper right')
                    ax2.grid(True)
                    ax2.set_ylim(0, 45)
                    ax2.set_autoscale_on(False)
                    ax2.axhline(45, color='k', lw=0.8, ls='--', alpha=0.4)
                    ax2.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)

                    # Plot individual episode rudder actions with lower alpha
                    for i, actions in enumerate(all_rudder_actions):
                        resampled = resample_to_common_time(actions, COMMON_TIME)
                        ax3.plot(COMMON_TIME, resampled, '-',
                                 color=colors[i], alpha=0.2,
                                 label=f'Episode {i + 1}' if i == 0 else "", linestyle=line_style)

                    # Plot average rudder actions with higher alpha and thicker line
                    # timesteps = np.arange(len(avg_rudder))
                    # ax3.plot(timesteps, avg_rudder, '-', color='black', alpha=0.6, lw=2, label='Average Rudder')

                    ax3.set_title(f'Rudder Actions Over Time (Individual and Average Policy)')
                    ax3.set_xlabel('Episode progress [%]')
                    ax3.set_ylabel('Rudder Action')
                    ax3.legend(loc='upper right')
                    ax3.grid(True)
                    ax3.set_xlim(0.0, 1.0)

                    # Plot individual episode ukcs with lower alpha
                    for i, ukcs in enumerate(all_ukc_measurements):
                        timesteps = normalized_time_axis(len(ukcs))
                        ax4.plot(timesteps, ukcs, '-',
                                 color=colors[i], alpha=0.2,
                                 label=f'Episode {i + 1}' if i == 0 else "", linestyle=line_style)

                    # Plot average ukcs with higher alpha and thicker line
                    timesteps = normalized_time_axis(len(avg_ukcs))
                    ax4.plot(timesteps, avg_ukcs, '-',
                             color='black', alpha=0.2,
                             label='Average UKC', linestyle=line_style)

                    ax4.set_title(f'UKC Over Time (Individual and Average Policy)')
                    ax4.set_xlabel('Episode progress [%]')
                    ax4.set_ylabel('UKC')
                    ax4.legend(loc='upper right')
                    ax4.grid(True)
                    ax4.set_ylim(0, 45)
                    ax4.set_autoscale_on(False)
                    ax4.axhline(45, color='k', lw=0.8, ls='--', alpha=0.4)
                    ax4.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)

                    # Adjust layout and save
                    plt.tight_layout()

                    trajectory_file_name = (
                        f'{self.agent.model_dir}/trajectory.png'
                        if self.agent.model_dir
                        else 'trajectory.png'
                    )

                    plt.savefig(trajectory_file_name, bbox_inches='tight', dpi=300)

                    print(f"\nPath and error visualization saved as 'trajectory.png'")
                finally:
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
                    prediction = self._agent.predict(state=obs, deterministic=True)
                    action = prediction[0]
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # Render the environment each step to visualize the agent's action
                    env.render()

                if self._fig is not None and self._overlay_active:
                    plt.close(self._fig)

                    self._fig = None
                    self._axes = None
                    self._overlay_active = False


    def get_env(self):
        """
        Retrieve the environment associated with the agent.

        Returns:
            The environment object.
        """
        return self._agent.get_env()
