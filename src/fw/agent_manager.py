import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from enum import auto, Enum
from matplotlib.colors import to_rgb
from contextlib import contextmanager
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME
from airsim.communication.airsim_bridge import AirSimBridge
from fw.environments.multi_sim_env import MultiSimEnv
from fw.environments.py_sim_colav_env import PySimColavEnv
from fw.environments.base_env import BaseEnv

from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.policies.multi_ppo_agent import MultiPPOAgent
from fw.policies.ppo2_agent import PPO2Agent
from fw.policies.base_agent import BaseAgent
from fw.stop_condition import StopCondition


class PolicyStrategy(Enum):
    """
    Defines strategies for handling policies during environment switching.

    Attributes:
        RESET_POLICY: Start with a fresh policy.
        REUSE_CURRENT_POLICY: Continue training with the current policy.
        REUSE_OTHER_POLICY: Load and use a policy trained on another environment.
    """

    RESET_POLICY = auto()
    REUSE_CURRENT_POLICY = auto()
    REUSE_OTHER_POLICY = auto()


def normalized_time_axis(length: int) -> np.ndarray:
    """
    Return a normalized time axis in [0, 1] for a trajectory of given length.
    """

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

    signal = np.asarray(signal)

    if signal.shape[0] == 1:
        return np.repeat(signal, len(common_time), axis=0)

    original_time = np.linspace(0.0, 1.0, signal.shape[0])

    # --- 1D case (fast path) ---
    if signal.ndim == 1:
        return np.interp(common_time, original_time, signal)

    # --- ND case ---
    # reshape to (T, -1), interpolate each column, then reshape back
    T = signal.shape[0]
    rest_shape = signal.shape[1:]

    flat_signal = signal.reshape(T, -1)  # (T, D)

    resampled = np.stack([
        np.interp(common_time, original_time, flat_signal[:, i])
        for i in range(flat_signal.shape[1])
    ], axis=1)

    return resampled.reshape(len(common_time), *rest_shape)


class AgentManager:
    """
    AgentManager class for managing agents and environments during training.

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
        """
        Initialize the Agent.

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
        """
        Create a persistent matplotlib figure for overlaying multiple environments.
        """

        if self._fig is None:
            self._fig, self._axes = plt.subplots(4, 1, figsize=(12, 16))
            self._overlay_active = True

    def end_overlay_plot(self):
        """
        Finalize and close the persistent matplotlib figure.
        """

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


    def _build_single_agent(self, env):
        """
        Create a `SinglePPOAgent` from the given environment.

        If the environment exposes a `sub_environments` attribute, this method
        ensures that exactly one sub-environment is used. Otherwise, the environment
        itself is used directly.

        Args:
            env: Environment instance, optionally containing `sub_environments`.

        Returns:
            SinglePPOAgent: Initialized PPO agent for a single environment.

        Raises:
            RuntimeError: If more than one sub-environment is provided.
        """

        sub_envs = getattr(env, "sub_environments", None)

        if sub_envs is None or len(sub_envs) < 1:
            target_env = env
        elif len(sub_envs) == 1:
            target_env = next(iter(sub_envs.values()))
        else:
            raise RuntimeError("SinglePPOAgent requires exactly one sub-environment.")

        return SinglePPOAgent(environment=target_env, **self._hyperparameters)


    def _build_multi_agent(self, env):
        """
        Create a `MultiPPOAgent` composed of multiple `SinglePPOAgent` instances.

        Each sub-environment in the provided `MultiSimEnv` is assigned its own
        `SinglePPOAgent`. These agents are then combined into a `MultiPPOAgent`
        for coordinated training.

        Args:
            env (MultiSimEnv): Multi-environment containing sub-environments.

        Returns:
            MultiPPOAgent: Initialized multi-agent PPO trainer.

        Raises:
            TypeError: If `env` is not an instance of `MultiSimEnv`.
            ValueError: If no sub-environments are defined.
        """

        if not isinstance(env, MultiSimEnv):
            raise TypeError("MultiPPOAgent requires a MultiSimEnv instance.")

        if not env.sub_environments:
            raise ValueError("MultiSimEnv must contain at least one sub-environment.")

        agents = [
            SinglePPOAgent(environment=sub_env, **self._hyperparameters)
            for sub_env in env.sub_environments.values()
        ]

        return MultiPPOAgent(environment=env, agents=agents, **self._hyperparameters)


    def set_environment(self, env: BaseEnv, policy_strategy: PolicyStrategy = PolicyStrategy.RESET_POLICY,
                        other_env_name: str = None) -> None:
        """
        Assign an environment to the agent and configure the policy strategy.

        Args:
            env (BaseEnv): The environment to set.
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
            self._agent = self._build_single_agent(env)
        elif PPO2_POLICY_NAME == self._agent_type:
            self._agent = PPO2Agent(environment=env, **self._hyperparameters)
        elif MULTI_PPO_POLICY_NAME == self._agent_type:
            self._agent = self._build_multi_agent(env)
        else:
            raise RuntimeError(f"Unknown {self._agent_type} policy type.")

        if policy_file_name:
            full_policy_file_name = f"{policy_file_name}.zip"
            if self.agent.model_dir:
                full_policy_file_name = os.path.join(self.agent.model_dir, full_policy_file_name)

            if Path(full_policy_file_name).exists():
                print(f"Reusing {policy_file_name}.")
                try:
                    self._agent.load_policy(policy_file_name)
                except ValueError as e:
                    print(f"[Fallback] Incompatible model: {e}")
                    print(f"[Fallback] Reverting to RESET_POLICY for {self._agent_type}.")
            else:
                print(f"[Fallback] The policy file {policy_file_name} does not exist.")
                print(f"[Fallback] Reverting to RESET_POLICY for {self._agent_type}.")


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
    def compute_metrics(cross_errors, rudder_angles, rudder_angle_changes, heading_errors, velocities):
        """
        Compute Mean Track Error Integral (mTEI), Maximum Track Error (MTE),
        and Mean Heading Error Integral (mHEI)

        mTEI = (1/(N1-N0+1)) * sum(|e(k)|)
        MTE = max{|e(N0)|, |e(N0+1)|, ..., |e(N1)|}
        mHEI = (1/(N1-N0+1)) * sum(|e_psi(k)|)
        """

        all_mTEIs = []
        all_MTEs = []
        all_mRAs = []
        all_MRAs = []
        all_mRACs = []
        all_mHEs = []
        all_mus = []
        all_Mus = []

        for ct_errors, rudder_angle, rudder_angle_change, heading_errors, velocity in zip(cross_errors, rudder_angles, rudder_angle_changes, heading_errors, velocities):
            # Calculate mTEI
            n = len(ct_errors)
            mTEI = np.sum(np.abs(ct_errors)) / n
            all_mTEIs.append(mTEI)

            # Calculate MTE
            MTE = np.max(np.abs(ct_errors))
            all_MTEs.append(MTE)

            # Calculate mRA
            mRA = np.sum(np.abs(rudder_angle)) / n
            all_mRAs.append(mRA)

            # Calculate MRA
            MRA = np.max(np.abs(rudder_angle))
            all_MRAs.append(MRA)

            # Calculate mRAC
            mRAC = np.sum(np.abs(rudder_angle_change)) / n
            all_mRACs.append(mRAC)

            # Calculate mHE
            mHE = np.sum(np.abs(heading_errors)) / n
            all_mHEs.append(mHE)

            # Calculate mu
            mu = np.sum(np.abs(velocity)) / n
            all_mus.append(mu)

            # Calculate Mu
            Mu = np.max(np.abs(velocity))
            all_Mus.append(Mu)

        return np.mean(all_mTEIs), np.mean(all_MTEs), np.mean(all_mRAs), np.mean(all_MRAs), np.mean(all_mRACs), np.mean(all_mHEs), np.mean(all_mus), np.mean(all_Mus)


    # Plot paths in the first subplot
    # Create hashed pattern for the area between obstacles and Western Scheldt
    @staticmethod
    def create_hashed_area(obstacles, overall):
        """
        Create hashed area.

        Args:
            obstacles: Description.
            overall: Description.
        """

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
        """
		Smoothing.

        Args:
            env: Description.
        """

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
            num_episodes: int = 1,
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
        # self._agent.policy = PPOPolicy.load("PySimEnv_PPO_policy", device="cpu")

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        print(f"\nRunning {num_episodes} episodes...")

        self.agent.set_policy_eval()
        env = self.get_env()
        
        # T-011 / Visualisation logic: prepare Fossen->AirSim coord transform if bridge active
        coord_transform = None
        if airsim_bridge and airsim_bridge.is_connected:
            from airsim.communication.coord_transform import CoordTransform
            coord_transform = CoordTransform()

        with self.smoothing(env):
            # ── Multi-colav dispatch ──
            # When the environment is a MultiSimEnv containing PySimColavEnv
            # sub-environments, use the dedicated multi-colav visualizer that
            # collects per-scenario collision avoidance metrics.
            if isinstance(env, MultiSimEnv):
                sub_envs = list(env.sub_environments.values())
                if sub_envs and isinstance(sub_envs[0], PySimColavEnv):
                    self._visualize_multi_colav(
                        num_episodes=num_episodes,
                        live_animation=live_animation,
                    )
                    return

            if hasattr(env, "ship_pos"):
                # Store results for each episode
                all_paths = []
                all_heading_errors = []
                total_rewards = []
                all_cross_errors = []  # Store cross-track errors for each episode
                all_rudder_angles = []
                all_rudder_angle_changes = []
                all_velocities = []
                all_rudder_actions = []
                all_ukc_measurements = []
                start_pos = None
                collision_count = 0
                collision_type_count = defaultdict(int)
                collision_side_count = defaultdict(int)
                colreg_violation_count = 0

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
                    velocities = []
                    previous_rudder_action = 0.0
                    rudder_changes = []
                    ukc_measurements = []
                    path_positions = [np.copy(env.ship_pos)]  # Store initial position
                    cross_errors.append(env.cross_error)  # Store initial cross-track error
                    velocities.append(env.velocities)  # Store initial cross-track error

                    # Calculate initial heading error relative to desired path
                    initial_heading_error = self.compute_heading_error(env)
                    heading_errors.append(initial_heading_error)

                    while not done:
                        # Select action
                        action, _, _ = self._agent.predict(state=state, deterministic=True)

                        # Take step in environment
                        state, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                        if not done:
                            # Store current position and errors
                            path_positions.append(np.copy(env.ship_pos))
                            if env.checkpoint_index >= 2:
                                cross_errors.append(env.cross_error)
                            heading_errors.append(env.heading_error)
                            velocities.append(env.velocities)
                            rudder_actions.append(env.performed_action)
                            rudder_changes.append(abs(env.performed_action - previous_rudder_action))
                            previous_rudder_action = env.performed_action
                            ukc_measurements.append(env.ukc)

                            episode_reward += reward if steps == 0 else (reward - episode_reward) / steps
                            steps += 1

                            # Uncomment for animated evaluation
                            if live_animation:
                                env.render()
                            
                            # Bridge position to UE5 if connected
                            if coord_transform and hasattr(env, 'ship_pos') and hasattr(env, 'phys_sim'):
                                ax, ay, _ = coord_transform.fossen_to_airsim(
                                    float(env.ship_pos[0]), 
                                    float(env.ship_pos[1])
                                )
                                ayaw = coord_transform.fossen_heading_to_airsim_yaw(
                                    float(env.phys_sim.state.heading)
                                )
                                airsim_bridge.set_pose(ax, ay, 0.0, ayaw)
                                
                        else:
                            collision_info = info.get('collision_info', {})
                            if collision_info and collision_info.get('collision'):
                                collision_count += 1
                                ctype = collision_info.get('collision_type')
                                collision_type_count[ctype] += 1
                                cside = collision_info.get('collision_side')
                                collision_side_count[cside] += 1
                            colreg_violations = info.get('colreg_violations', 0)
                            if colreg_violations:
                                colreg_violation_count += colreg_violations
                            print(f"Episode {episode + 1} finished after {steps} steps with reward {episode_reward:.2f}")
                            break

                    all_paths.append(np.array(path_positions))
                    all_cross_errors.append(np.array(cross_errors))
                    all_rudder_angles.append(np.array(rudder_actions))
                    all_rudder_angle_changes.append(np.array(rudder_changes))
                    all_heading_errors.append(np.array(heading_errors))
                    all_velocities.append(np.array(velocities))
                    all_rudder_actions.append(np.array(rudder_actions))
                    all_ukc_measurements.append(np.array(ukc_measurements))
                    total_rewards.append(episode_reward)

                # Calculate mTEI, MTE, and mHEI
                mTEI, MTE, mRA, MRA, mRAC, mHE, mu, Mu = self.compute_metrics(all_cross_errors, all_rudder_angles, all_rudder_angle_changes, all_heading_errors, all_velocities)

                # Print summary statistics
                avg_reward = np.mean(total_rewards)
                print(f"\nEvaluation completed!")
                print(f"Average reward: {avg_reward:.2f}")
                print(f"Best episode reward: {max(total_rewards):.2f}")
                print(f"Worst episode reward: {min(total_rewards):.2f}")
                print(f"Mean Surge Velocity (mu): {mu:.2f}")
                print(f"Maximum Surge Velocity (Mu): {Mu:.2f}")
                print(f"Mean Track Error Integral (mTEI): {mTEI:.2f}")
                print(f"Maximum Track Error (MTE): {MTE:.2f}")
                print(f"Mean Rudder Angle (mRA): {mRA:.3f}")
                print(f"Maximum Rudder Angle (MRA): {MRA:.3f}")
                print(f"Mean Rudder Angle Change (mRAC): {mRAC:.4f}")
                print(f"Mean LOS Guidance Heading Error (mHE): {mHE:.3f} rad")
                print(f"Collision rate (CR): {collision_count / num_episodes * 100:.2f}%")
                print(f"Collision per type: {dict(collision_type_count)}")
                print(f"Collision per side: {dict(collision_side_count)}")
                print(f"COLREG violation count: {colreg_violation_count}")

                # plot min distances kept during episodes if available
                if hasattr(env, "plot_min_distance_PDF"):
                    env.plot_min_distance_PDF(f'{self.agent.model_dir}/min_distance_PDF.png')

                try:
                    # AirSimEnv has no obstacles/overall for matplotlib — skip trajectory plot.
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
                    
                    # Bridge position to UE5 if connected
                    if coord_transform and hasattr(env, 'ship_pos') and hasattr(env, 'phys_sim'):
                        ax, ay, _ = coord_transform.fossen_to_airsim(
                            float(env.ship_pos[0]), 
                            float(env.ship_pos[1])
                        )
                        ayaw = coord_transform.fossen_heading_to_airsim_yaw(
                            float(env.phys_sim.state.heading)
                        )
                        airsim_bridge.set_pose(ax, ay, 0.0, ayaw)

                if self._fig is not None and self._overlay_active:
                    plt.close(self._fig)

                    self._fig = None
                    self._axes = None
                    self._overlay_active = False


    def _visualize_multi_colav(
            self,
            num_episodes: int,
            live_animation: bool,
    ) -> None:
        """
        Visualize a trained MultiPPOAgent across multiple PySimColavEnv sub-environments.

        Runs episodes through the MultiSimEnv, collecting per-scenario collision
        avoidance metrics (collision rate, collision types, collision sides,
        COLREG violations) and path-following metrics, then prints an aggregated
        summary.

        Args:
            num_episodes: Number of episodes to evaluate.
            live_animation: Whether to render each sub-environment live.
        """

        multi_env = self.get_env()
        sub_envs = list(multi_env.sub_environments.values())
        num_sub = len(sub_envs)

        # Identify scenario labels
        scenario_labels = []
        for se in sub_envs:
            label = getattr(se, 'eval_scenario', se.type_name)
            scenario_labels.append(label)

        # Per-scenario accumulators
        per_scenario = {
            label: {
                'collision_count': 0,
                'collision_type_count': defaultdict(int),
                'collision_side_count': defaultdict(int),
                'colreg_violation_count': 0,
                'total_rewards': [],
                'all_cross_errors': [],
                'all_heading_errors': [],
                'all_rudder_angles': [],
                'all_rudder_angle_changes': [],
                'all_velocities': [],
                'all_paths': [],
            }
            for label in scenario_labels
        }

        for episode in range(num_episodes):
            state, _ = multi_env.reset()
            done_flags = [False] * num_sub
            episode_rewards = [0.0] * num_sub
            steps = [0] * num_sub

            # Per-sub-env episode tracking
            ep_paths = [[] for _ in range(num_sub)]
            ep_cross_errors = [[] for _ in range(num_sub)]
            ep_heading_errors = [[] for _ in range(num_sub)]
            ep_rudder_actions = [[] for _ in range(num_sub)]
            ep_rudder_changes = [[] for _ in range(num_sub)]
            ep_velocities = [[] for _ in range(num_sub)]
            prev_rudder = [0.0] * num_sub

            # Record initial state
            for i, se in enumerate(sub_envs):
                ep_paths[i].append(np.copy(se.ship_pos))
                if hasattr(se, 'cross_error'):
                    ep_cross_errors[i].append(se.cross_error)
                if hasattr(se, 'velocities'):
                    ep_velocities[i].append(se.velocities)
                ep_heading_errors[i].append(self.compute_heading_error(se))

            while not all(done_flags):
                # Build actions: real predictions for active sub-envs, neutral for done ones
                actions = []
                predictions = self._agent.predict(state=state, deterministic=True)
                pred_actions = predictions[0]

                for i in range(num_sub):
                    if done_flags[i]:
                        neutral = multi_env.get_neutral_action(i, pred_actions[i])
                        actions.append(neutral)
                    else:
                        actions.append(pred_actions[i])

                state, rewards, terminations, truncations, infos = multi_env.step(tuple(actions))

                for i in range(num_sub):
                    if done_flags[i]:
                        continue

                    se = sub_envs[i]
                    label = scenario_labels[i]
                    done_i = bool(terminations[i]) or bool(truncations[i])

                    if done_i:
                        done_flags[i] = True
                        # Collect terminal info
                        info = infos[i]
                        collision_info = info.get('collision_info', {})
                        if collision_info and collision_info.get('collision'):
                            per_scenario[label]['collision_count'] += 1
                            ctype = collision_info.get('collision_type')
                            per_scenario[label]['collision_type_count'][ctype] += 1
                            cside = collision_info.get('collision_side')
                            per_scenario[label]['collision_side_count'][cside] += 1
                        colreg_violations = info.get('colreg_violations', 0)
                        per_scenario[label]['colreg_violation_count'] += colreg_violations
                        episode_rewards[i] += float(rewards[i])
                        per_scenario[label]['total_rewards'].append(episode_rewards[i])
                        print(f"  [{label}] Episode {episode + 1} finished after "
                              f"{steps[i]} steps with reward {episode_rewards[i]:.2f}")
                    else:
                        # Accumulate step data
                        ep_paths[i].append(np.copy(se.ship_pos))
                        if hasattr(se, 'cross_error') and hasattr(se, 'checkpoint_index') and se.checkpoint_index >= 2:
                            ep_cross_errors[i].append(se.cross_error)
                        if hasattr(se, 'heading_error'):
                            ep_heading_errors[i].append(se.heading_error)
                        if hasattr(se, 'velocities'):
                            ep_velocities[i].append(se.velocities)
                        if hasattr(se, 'performed_action'):
                            ep_rudder_actions[i].append(se.performed_action)
                            ep_rudder_changes[i].append(abs(se.performed_action - prev_rudder[i]))
                            prev_rudder[i] = se.performed_action

                        episode_rewards[i] += float(rewards[i]) if steps[i] == 0 else (
                            (float(rewards[i]) - episode_rewards[i]) / steps[i]
                        )
                        steps[i] += 1

                        if live_animation:
                            se.render()

            # Store per-scenario episode data
            for i, label in enumerate(scenario_labels):
                per_scenario[label]['all_paths'].append(np.array(ep_paths[i]))
                per_scenario[label]['all_cross_errors'].append(np.array(ep_cross_errors[i]))
                per_scenario[label]['all_heading_errors'].append(np.array(ep_heading_errors[i]))
                per_scenario[label]['all_rudder_angles'].append(np.array(ep_rudder_actions[i]))
                per_scenario[label]['all_rudder_angle_changes'].append(np.array(ep_rudder_changes[i]))
                per_scenario[label]['all_velocities'].append(np.array(ep_velocities[i]))

        # ── Print summary ──
        print("\n" + "=" * 60)
        print("MULTI-COLAV EVALUATION SUMMARY")
        print("=" * 60)

        total_collisions = 0
        total_colreg_violations = 0
        all_type_counts = defaultdict(int)
        all_side_counts = defaultdict(int)

        for label in scenario_labels:
            sc = per_scenario[label]
            total_collisions += sc['collision_count']
            total_colreg_violations += sc['colreg_violation_count']
            for k, v in sc['collision_type_count'].items():
                all_type_counts[k] += v
            for k, v in sc['collision_side_count'].items():
                all_side_counts[k] += v

            avg_reward = np.mean(sc['total_rewards']) if sc['total_rewards'] else 0.0

            print(f"\n── Scenario: {label} ──")
            print(f"  Average reward:       {avg_reward:.2f}")
            print(f"  Collision rate (CR):   {sc['collision_count'] / num_episodes * 100:.2f}%")
            print(f"  Collision per type:    {dict(sc['collision_type_count'])}")
            print(f"  Collision per side:    {dict(sc['collision_side_count'])}")
            print(f"  COLREG violations:     {sc['colreg_violation_count']}")

            # Compute path-following metrics if data is available
            if sc['all_cross_errors'] and any(len(ce) > 0 for ce in sc['all_cross_errors']):
                valid_ce = [ce for ce in sc['all_cross_errors'] if len(ce) > 0]
                valid_ra = [ra for ra in sc['all_rudder_angles'] if len(ra) > 0]
                valid_rac = [rac for rac in sc['all_rudder_angle_changes'] if len(rac) > 0]
                valid_he = [he for he in sc['all_heading_errors'] if len(he) > 0]
                valid_vel = [v for v in sc['all_velocities'] if len(v) > 0]

                if valid_ce and valid_ra and valid_rac and valid_he and valid_vel:
                    mTEI, MTE, mRA, MRA, mRAC, mHE, mu, Mu = self.compute_metrics(
                        valid_ce, valid_ra, valid_rac, valid_he, valid_vel
                    )
                    print(f"  Mean Track Error (mTEI): {mTEI:.2f}")
                    print(f"  Max Track Error (MTE):   {MTE:.2f}")
                    print(f"  Mean Rudder Angle (mRA): {mRA:.3f}")
                    print(f"  Max Rudder Angle (MRA):  {MRA:.3f}")
                    print(f"  Mean Rudder Change (mRAC): {mRAC:.4f}")
                    print(f"  Mean Heading Error (mHE):  {mHE:.3f} rad")
                    print(f"  Mean Surge Velocity (mu):  {mu:.2f}")
                    print(f"  Max Surge Velocity (Mu):   {Mu:.2f}")

        total_episodes = num_episodes * num_sub
        print(f"\n── Aggregated (all scenarios) ──")
        print(f"  Total collision rate:   {total_collisions / total_episodes * 100:.2f}%")
        print(f"  Total collision count:  {total_collisions}")
        print(f"  Collision per type:     {dict(all_type_counts)}")
        print(f"  Collision per side:     {dict(all_side_counts)}")
        print(f"  Total COLREG violations: {total_colreg_violations}")
        print("=" * 60)

        # Plot min-distance PDFs for each sub-env if available
        for i, se in enumerate(sub_envs):
            if hasattr(se, 'plot_min_distance_PDF'):
                label = scenario_labels[i]
                save_path = (
                    f'{self.agent.model_dir}/min_distance_PDF_{label}.png'
                    if self.agent.model_dir
                    else f'min_distance_PDF_{label}.png'
                )
                se.plot_min_distance_PDF(save_path)


    def get_env(self):
        """
        Retrieve the environment associated with the agent.

        Returns:
            The environment object.
        """

        return self._agent.get_env()
