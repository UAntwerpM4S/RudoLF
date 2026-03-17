import os
from collections import defaultdict

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
from fw.stop_condition import StopCondition
from fw.policies.base_model import BaseModel
from fw.policies.ppo2_model import PPO2Model
from fw.policies.ppo_model import PPOModel
from fw.simulators.airsim_bridge import AirSimBridge



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


class ColavAgent:
    """Agent class for managing models and environments during training.

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

    def __init__(self, model_type: str, hyperparameters: dict):
        """Initialize the Agent.

        Args:
            model_type (str): Type of model to use (e.g., 'PPO').
            hyperparameters (dict): Hyperparameters for configuring the model.
        """
        self._model = None
        self._model_type = model_type
        self.prev_smoothing_setting = None
        self._hyperparameters = hyperparameters

        self._fig = None
        self._axes = None
        self._overlay_active = False
        self._collision_episode_records = []
        self._warned_reset_seed_not_supported = False


    def begin_overlay_plot(self):
        """Create a persistent matplotlib figure for overlaying multiple environments."""
        if self._fig is None:
            self._fig, self._axes = plt.subplots(4, 1, figsize=(12, 16))
            self._overlay_active = True

    def end_overlay_plot(self):
        """Finalize and close the persistent matplotlib figure."""
        if self._fig is not None:
            plt.tight_layout()
            self._fig.tight_layout()
            self._fig.subplots_adjust(hspace=0.6)
            plt.show()
            plt.close(self._fig)

            self._fig = None
            self._axes = None
            self._overlay_active = False


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

    def visualize_trained_model(
            self,
            num_episodes: int = 1,
            live_animation: bool = False,
            env_color_index: int = 0,
            airsim_bridge: AirSimBridge = None,
            replay_collision_episodes: bool = True,
            max_replays: int = 5,
    ) -> None:

        """
        Run and visualize a trained model in the environment over multiple episodes.

        Args:
            num_episodes (int, optional): Number of episodes to visualize (default is 5).
            live_animation (bool, optional): Enable live animation of the agent in the environment.
            env_color_index (int, optional): Index of the environment color.

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
        env.eval = True

        if replay_collision_episodes:
            self._collision_episode_records = []

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
                collisions_total = 0
                collisions_by_type = defaultdict(int)
                collision_points = []
                all_rewards = []
                all_collision_sides = defaultdict(int)
                all_collision_cte = []


                for episode in range(num_episodes):
                    print(f"Starting episode {episode + 1}")
                    episode_had_collision = False
                    #state, _ = env.reset()
                    # seeded reset for reproducibility
                    episode_seed = int(np.random.randint(0, 2**31 - 1))
                    state, _ = env.reset(episode_seed)
                    episode_reward = 0
                    done = False
                    steps = 0
                    episode_actions = []
                    first_collision_step = None
                    first_collision_type = None

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
                        action, _ = self._model.predict(state=state, deterministic=True)

                        if replay_collision_episodes:
                            episode_actions.append(np.array(action, copy=True).tolist())

                        # Take step in environment
                        state, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        collision_flag = bool(info.get("collision", False))
                        if not terminated:
                            # make sure that large negative rewards upon termination do not destroy the scale
                            # in the plot of all rewards over an episode
                            all_rewards.append(reward)

                        if collision_flag and not episode_had_collision:
                            episode_had_collision = True
                            collisions_total += 1
                            ctype = info.get("collision_type", "unknown") if info else "unknown"
                            collisions_by_type[ctype] += 1
                            collision_points.append(np.copy(env.ship_pos))
                            cside = info.get("collision_side","unknown") if info else "unknown"
                            all_collision_sides[cside] += 1
                            all_collision_cte.append(info.get("collision_cte", "unknown") if info else "unknown")

                            if replay_collision_episodes:
                                first_collision_step = len(episode_actions)
                                first_collision_type = ctype


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
                            print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f} and with collision: {episode_had_collision}")
                            break

                    all_paths.append(np.array(path_positions))
                    all_cross_errors.append(np.array(cross_errors))
                    all_heading_errors.append(np.array(heading_errors))
                    all_rudder_actions.append(np.array(rudder_actions))
                    all_ukc_measurements.append(np.array(ukc_measurements))
                    total_rewards.append(episode_reward)

                    if replay_collision_episodes and episode_had_collision:
                        self._collision_episode_records.append({
                            "episode_index": episode,
                            "seed": episode_seed,
                            "collision_step": first_collision_step,
                            "collision_type": first_collision_type or "unknown",
                            "actions": episode_actions,
                        })

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
                print(f"Collisions: {collisions_total}/{num_episodes} "
                      f"({100.0 * collisions_total / max(1, num_episodes):.1f}%)")
                if all_collision_cte and all_collision_sides:
                    print(f"Max CTE at point of collision: {np.max(all_collision_cte)}")
                    print(f"Min CTE at point of collision: {np.min(all_collision_cte)}")
                    print(f"Average CTE at point of collision: {np.mean(all_collision_cte)}")
                    print(f"Collision sides:", dict(all_collision_sides))

                if collisions_by_type:
                    print("Collision types:", dict(collisions_by_type))

                if replay_collision_episodes:
                    print(f"Stored {len(self._collision_episode_records)} collision episode(s) for replay.")

                if replay_collision_episodes and self._collision_episode_records:
                    self.replay_collision_episodes(max_replays=max_replays)

                try:
                    # Create a figure with two subplots
                    if self._overlay_active and self._fig is not None:
                        ax1, ax2, ax3, ax5 = self._axes
                    else:
                        fig, (ax1, ax2, ax3, ax5) = plt.subplots(4, 1, figsize=(12, 20))

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

                    # Collision points
                    if len(collision_points) > 0:
                        cp = np.array(collision_points)
                        ax1.scatter(cp[:, 0], cp[:, 1], marker='x', s=80, c='red', label='Collision')

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
                    max_rudder_length = max(len(actions) for actions in all_rudder_actions)
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
                    avg_rudder = np.mean(rudder_resampled, axis=0)
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
                    # for i, ukcs in enumerate(all_ukc_measurements):
                    #     timesteps = normalized_time_axis(len(ukcs))
                    #     ax4.plot(timesteps, ukcs, '-',
                    #              color=colors[i], alpha=0.2,
                    #              label=f'Episode {i + 1}' if i == 0 else "", linestyle=line_style)

                    # Plot average ukcs with higher alpha and thicker line
                    # timesteps = normalized_time_axis(len(avg_ukcs))
                    # ax4.plot(timesteps, avg_ukcs, '-',
                    #          color='black', alpha=0.2,
                    #          label='Average UKC', linestyle=line_style)
                    #
                    # ax4.set_title(f'UKC Over Time (Individual and Average Policy)')
                    # ax4.set_xlabel('Episode progress [%]')
                    # ax4.set_ylabel('UKC')
                    # ax4.legend(loc='upper right')
                    # ax4.grid(True)
                    # ax4.set_ylim(0, 45)
                    # ax4.set_autoscale_on(False)
                    # ax4.axhline(45, color='k', lw=0.8, ls='--', alpha=0.4)
                    # ax4.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)

                    # Plot rewards obtained throughout the episode
                    t_rewards = normalized_time_axis(len(all_rewards))
                    ax5.plot(t_rewards, all_rewards, '-',
                             color='tab:blue', alpha=0.8,
                             label='Reward', linestyle=line_style)

                    ax5.set_title('Rewards over episode')
                    ax5.set_xlabel('Episode progress [%]')
                    ax5.set_ylabel('Reward')
                    ax5.legend(loc='upper right')
                    ax5.grid(True)



                    # Adjust layout and save
                    plt.tight_layout()

                    trajectory_file_name = (
                        f'{self.model.model_dir}/trajectory.png'
                        if self.model.model_dir
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
                    prediction = self._model.predict(state=obs, deterministic=True)
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

    def replay_collision_episodes(self, max_replays=None):
        """
        Replay previously stored collision episodes using recorded actions.

        Args:
            max_replays (int | None): Max number of stored collision episodes to replay.
        """
        if not self._collision_episode_records:
            print("No stored collision episodes to replay.")
            return

        if not self._model:
            raise RuntimeError("No model has been created yet!")

        env = self.get_env()
        self.model.set_policy_eval()

        records = self._collision_episode_records
        if max_replays is not None:
            records = records[:max_replays]

        print(f"\nReplaying {len(records)} stored collision episode(s)...")

        with self.smoothing(env):
            try:
                for rec in records:
                    print(
                        f"\nReplay episode {rec['episode_index'] + 1} "
                        f"(collision_type={rec.get('collision_type', 'unknown')}, "
                        f"collision_step={rec.get('collision_step', 'unknown')})"
                    )

                    # Seeded reset (directly, since your env supports it)
                    obs, _ = env.reset(seed=rec["seed"])

                    # Always render during replay
                    env.render()

                    done = False
                    collision_seen = False

                    for step_idx, action in enumerate(rec["actions"], start=1):
                        action_np = np.array(action)

                        obs, reward, terminated, truncated, info = env.step(action_np)
                        done = terminated or truncated

                        env.render()

                        if bool(info.get("collision", False)):
                            collision_seen = True
                            print(f"  Collision reached during replay at step {step_idx}.")

                        if done:
                            break

                    if not collision_seen:
                        print(
                            "  Replay finished without detecting a collision "
                            "(possible non-determinism despite seeded reset)."
                        )
            finally:
                if hasattr(env, "fh_sim"):
                    env.fh_sim.dispose()


    def get_env(self):
        """
        Retrieve the environment associated with the agent's model.

        Returns:
            The environment object.
        """
        return self._model.get_env()
