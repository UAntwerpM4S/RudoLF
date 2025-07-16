import matplotlib

try:
    matplotlib.use('TkAgg')  # Try using the interactive backend
except ImportError:
    print("Warning: 'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use('Agg')  # Use non-interactive backend as fallback

import os
import numpy as np
import tkinter as tk
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from functools import lru_cache
from shapely.geometry import Polygon
from typing import Tuple, Optional, Dict
from fw.simulators.base_env import BaseEnv
from fw.simulators.tools import create_checkpoints_from_simple_path, check_collision_ship


def calculate_perpendicular_lines(checkpoints: list, line_length: float = 100.0) -> list:
    """
    Calculate perpendicular lines at each checkpoint using smoothed tangent direction.

    Args:
        checkpoints: List of dicts containing checkpoint positions and radii
        line_length: Length of perpendicular lines

    Returns:
        list: Tuples of (start_point, end_point) for each perpendicular line
    """

    def smooth_tangent(check_points: list, index: int) -> np.ndarray:
        """Calculate the tangent at checkpoint `i` by averaging vectors to neighbors."""
        if index == 0:  # Start of the path
            tangent = check_points[1]['pos'] - check_points[0]['pos']
        elif index == len(check_points) - 1:  # End of the path
            tangent = check_points[-1]['pos'] - check_points[-2]['pos']
        else:   # Middle of the path
            tangent = (
                check_points[index + 1]['pos'] - check_points[index]['pos'] +
                check_points[index]['pos'] - check_points[index - 1]['pos']
            )   # Average direction
        norm = np.linalg.norm(tangent)
        return tangent / norm if norm != 0 else tangent


    lines = []  # to store start and end points of perpendicular lines
    for i, checkpoint in enumerate(checkpoints):
        # Get the perpendicular direction using the smoothed tangent at the current checkpoint
        smoothed_tangent = smooth_tangent(checkpoints, i)
        perpendicular = np.array([-smoothed_tangent[1], smoothed_tangent[0]])

        # Calculate the start and end points of the perpendicular line at the checkpoint
        midpoint = checkpoint['pos']
        offset = perpendicular * (line_length / 2)
        lines.append((midpoint + offset, midpoint - offset))

    return lines


class PySimEnv(BaseEnv):
    """Custom Python Simulator environment for ship navigation with improved physics."""

    # Physical limits
    MIN_SURGE_VELOCITY = 0.0
    MIN_SWAY_VELOCITY = -2.0
    MIN_YAW_RATE = -0.5
    MAX_SURGE_VELOCITY = 5.0
    MAX_SWAY_VELOCITY = 2.0
    MAX_YAW_RATE = 0.5
    YAW_RATE_DAMPING = 0.1
    MAX_RUDDER_RATE = 0.06  # Maximum change in rudder angle per time step
    MAX_THRUST_RATE = 0.05  # Maximum change in thrust per time step
    CHECKPOINTS_DISTANCE = 350
    MIN_GRID_POS = -11700
    MAX_GRID_POS = 14500

    # Reward parameters
    SHAPING_WINDOW = 5
    DECAY_SCALE = 1.0
    REWARD_DISTANCE_SCALE = 2.0
    REWARD_DIRECTION_SCALE = 1.0
    PENALTY_DISTANCE_SCALE = 1.0
    CROSS_TRACK_ERROR_PENALTY_SCALE = 50.0
    MAX_STEPS_PENALTY = -15
    COLLISION_PENALTY = -10
    SUCCESS_REWARD = 50.0
    CHECKPOINT_AREA_SIZE = 5.0
    TARGET_AREA_SIZE = 7.0

    # Control parameters
    RUDDER_DEAD_ZONE = 0.5
    THRUST_DEAD_ZONE = 0.5
    DERIVATIVE_FILTER_ALPHA = 0.2
    MAX_RUDDER_RATE_CHANGE = 0.1
    MAX_THRUST_RATE_CHANGE = 0.15
    ANTI_WINDUP_THRESHOLD = 0.95

    # Rendering constants
    MAX_FIG_WIDTH = 1200
    MAX_FIG_HEIGHT = 900
    DPI = 100

    def __init__(self,
                 render_mode: Optional[str] = None,
                 time_step: float = 0.1,
                 max_steps: int = 1500,
                 verbose: Optional[bool] = None,
                 ship_pos: Optional[np.ndarray] = None,
                 target_pos: Optional[np.ndarray] = None,
                 wind: bool = False,
                 current: bool = False):
        """Initialize the ship navigation environment.

        Args:
            render_mode: Either None or 'human' for visualization
            time_step: Simulation time step in seconds
            max_steps: Maximum steps per episode
            verbose: Whether to print debug information
            ship_pos: Optional initial ship position
            target_pos: Optional target position
            wind: Whether to enable wind effects
            current: Whether to enable current effects
        """
        super().__init__(render_mode)

        # Environment parameters
        self.time_step = time_step
        self.max_steps = max_steps
        self.verbose = verbose
        self.wind = wind
        self.current = current

        # Initialize state
        self._initialize_state(ship_pos)
        self._initialize_control_parameters()
        self._load_environment_data()

        # Gym spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = self._initialize_observation_space()

        self.reward_weights = {
            'distance': 0.3,
            'heading': 0.3,
            'cross_track': 0.2,
            'rudder': 0.2
        }

        # Rendering
        self.initialize_plots = True


    def _initialize_state(self, ship_pos: Optional[np.ndarray]) -> None:
        """Initialize the ship's state variables."""
        self.initial_ship_pos = np.array(ship_pos, dtype=np.float32) if ship_pos else np.array([5.0, 5.0], dtype=np.float32)
        self.ship_pos = np.copy(self.initial_ship_pos)
        self.previous_ship_pos = np.zeros(2, dtype=np.float32)
        self.previous_heading = 0.0
        self.ship_angle = 0.0
        self.randomization_scale = 1.0
        self.max_dist = np.sqrt(2) * self.MAX_GRID_POS
        self.state = np.array([*self.ship_pos, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.current_action = np.zeros(2, dtype=np.float32)


    def _initialize_control_parameters(self) -> None:
        """Initialize PID controller and related parameters."""
        # PID gains
        self.rudder_kp = 0.2   # Proportional gain for rudder
        self.rudder_ki = 0.1   # Integral gain for rudder
        self.rudder_kd = 0.15  # Helps dampen oscillations
        self.thrust_kp = 0.2   # Proportional gain for thrust
        self.thrust_ki = 0.02  # Integral gain for thrust
        self.thrust_kd = 0.05  # Less derivative for thrust

        # Error terms
        self.rudder_error_sum = 0.0
        self.thrust_error_sum = 0.0
        self.previous_rudder_error = 0.0
        self.previous_thrust_error = 0.0
        self.previous_rudder_target = 0.0
        self.previous_thrust_target = 0.0
        self.filtered_rudder_derivative = 0.0
        self.filtered_thrust_derivative = 0.0

        # Environmental effects
        self.radians_current = np.radians(180)
        self.current_direction = np.array([np.cos(self.radians_current), np.sin(self.radians_current)], dtype=np.float32)
        self.current_strength = 0.35
        self.radians_wind = np.radians(90)
        self.wind_direction = np.array([np.cos(self.radians_wind), np.sin(self.radians_wind)], dtype=np.float32)
        self.wind_strength = 0.35


    def _load_environment_data(self) -> None:
        """Load obstacles, paths and initialize checkpoints."""
        base_path = os.path.dirname(os.path.abspath(__file__))

        try:
            self.obstacles = np.loadtxt(
                os.path.join(base_path, 'env_Sche_250cm_no_scale.csv'),
                delimiter=',', skiprows=1
            )
            self.polygon_shape = Polygon(self.obstacles)

            self.overall = np.loadtxt(
                os.path.join(base_path, 'env_Sche_no_scale.csv'),
                delimiter=',', skiprows=1
            )

            path = np.loadtxt(
                os.path.join(base_path, 'trajectory_points_no_scale.csv'),
                delimiter=',', skiprows=1
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required environment file: {e}")

        # Optional: fail if obstacle/overall/path is not 2D with 2 columns
        if self.obstacles.ndim != 2 or self.obstacles.shape[1] != 2:
            raise ValueError("Obstacles data must be a 2D array with shape (N, 2)")
        if self.overall.ndim != 2 or self.overall.shape[1] != 2:
            raise ValueError("Overall map data must be a 2D array with shape (N, 2)")
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("Path data must be a 2D array with shape (N, 2)")

        path = self._reduce_path(path, self.initial_ship_pos)
        path = create_checkpoints_from_simple_path(path, self.CHECKPOINTS_DISTANCE)
        path = np.insert(path, 0, self.ship_pos, axis=0)  # Safely insert initial ship position

        checkpoints = [{'pos': np.array(point, dtype=np.float32), 'radius': self.CHECKPOINT_AREA_SIZE, 'reward': (i / len(path)) * 10} for i, point in enumerate(path)]
        checkpoints[-1]['radius'] = self.TARGET_AREA_SIZE
        checkpoints[-1]['reward'] = self.SUCCESS_REWARD
        lines = calculate_perpendicular_lines(checkpoints, line_length=50)

        self.checkpoints = [{**checkpoint, 'perpendicular_line': line} for checkpoint, line in zip(checkpoints, lines)]
        self.target_pos = self.checkpoints[-1]['pos']
        self.checkpoint_index = 1
        self.step_count = 0
        self.stuck_steps = 0
        self.cross_error = 0.0
        self.desired_heading = 0.0

        # Hydrodynamic coefficients
        self.xu = -0.02  # Surge damping
        self.yv = -0.4   # Sway damping
        self.yv_r = -0.09  # Default sway-to-yaw coupling coefficient
        self.nr = -0.26   # Yaw damping
        self.l = 50.0    # Ship length

        # Dynamic coefficients
        self.k_t = 0.05  # Thrust coefficient
        self.k_r = 0.039   # Rudder coefficient
        self.k_v = 0.03 # Sway coefficient


    def _reduce_path(self, path: np.ndarray, start_pos: np.ndarray) -> np.ndarray:
        """Reduces the path to start from the closest point to the given start position.

        This method searches the provided path for the point that is closest to `start_pos`
        and returns a sub-path starting from that point to the end. The first point of the
        reduced path is replaced by `start_pos` itself. If reduction is not performed, the
        original path is returned as-is.

        Args:
            path: Original path as array of coordinates
            start_pos: New start position to align path with

        Returns:
            np.ndarray: Reduced path starting at closest point to start_pos

        Note:
            This stub implementation does not perform any reduction and returns the path
            unchanged. Override this method to apply actual reduction logic.
        """
        return path  # Implementation note: Currently returns original path


    def _apply_pi_controller(self, target_action: np.ndarray) -> np.ndarray:
        """Apply advanced PID controller with dead zone and derivative filtering.

        Args:
            target_action: Array of [target_rudder, target_thrust] values in [-1, 1]

        Returns:
            np.ndarray: Smoothed action array [rudder, thrust] with rate limiting
        """
        target_rudder, target_thrust = target_action[0], abs(target_action[1])

        # Dead zone for small changes
        if abs(target_rudder - self.previous_rudder_target) < self.RUDDER_DEAD_ZONE:
            target_rudder = self.previous_rudder_target

        if abs(target_thrust - self.previous_thrust_target) < self.THRUST_DEAD_ZONE:
            target_thrust = self.previous_thrust_target

        # Calculate errors
        rudder_error = target_rudder - self.previous_rudder_target
        thrust_error = target_thrust - self.previous_thrust_target

        # Calculate and filter derivatives
        rudder_derivative = (rudder_error - self.previous_rudder_error) / self.time_step
        thrust_derivative = (thrust_error - self.previous_thrust_error) / self.time_step

        self.filtered_rudder_derivative = (
                self.DERIVATIVE_FILTER_ALPHA * rudder_derivative +
                (1 - self.DERIVATIVE_FILTER_ALPHA) * self.filtered_rudder_derivative
        )
        self.filtered_thrust_derivative = (
                self.DERIVATIVE_FILTER_ALPHA * thrust_derivative +
                (1 - self.DERIVATIVE_FILTER_ALPHA) * self.filtered_thrust_derivative
        )

        # Update integral terms with anti-windup
        if abs(self.previous_rudder_target) < self.ANTI_WINDUP_THRESHOLD:
            self.rudder_error_sum = np.clip(
                self.rudder_error_sum + rudder_error * self.time_step,
                -0.5, 0.5
            )

        if abs(self.previous_thrust_target) < self.ANTI_WINDUP_THRESHOLD:
            self.thrust_error_sum = np.clip(
                self.thrust_error_sum + thrust_error * self.time_step,
                -0.5, 0.5
            )

        # Calculate PID outputs
        rudder_output = (
                self.rudder_kp * rudder_error +
                self.rudder_ki * self.rudder_error_sum +
                self.rudder_kd * self.filtered_rudder_derivative
        )

        thrust_output = (
                self.thrust_kp * thrust_error +
                self.thrust_ki * self.thrust_error_sum +
                self.thrust_kd * self.filtered_thrust_derivative
        )

        # Apply rate limiting (prevents sudden jumps)
        rudder_output = np.clip(rudder_output, -self.MAX_RUDDER_RATE_CHANGE, self.MAX_RUDDER_RATE_CHANGE)
        thrust_output = np.clip(thrust_output, -self.MAX_THRUST_RATE_CHANGE, self.MAX_THRUST_RATE_CHANGE)

        new_rudder = np.clip(self.previous_rudder_target + rudder_output, -1.0, 1.0)
        new_thrust = np.clip(self.previous_thrust_target + thrust_output, -1.0, 1.0)

        # Update state
        self.previous_rudder_error = rudder_error
        self.previous_thrust_error = thrust_error
        self.previous_rudder_target = new_rudder
        self.previous_thrust_target = new_thrust

        return np.array([new_rudder, new_thrust], dtype=np.float32)


    def _initialize_observation_space(self) -> gym.spaces.Box:
        """Initialize and return the observation space for the environment.

        Observation space includes:
        - Normalized ship position, heading, velocities
        - Distances to current and next checkpoints
        - Cross-track and heading errors
        - Current control actions
        - Optional wind/current parameters if enabled

        Returns:
            gym.spaces.Box: The observation space definition
        """
        # Base observations
        base_low=np.array([
            self.MIN_GRID_POS,          # Ship position x
            self.MIN_GRID_POS,          # Ship position y
            -np.pi,                     # Ship heading
            self.MIN_SURGE_VELOCITY,    # Surge velocity
            self.MIN_SWAY_VELOCITY,     # Sway velocity
            self.MIN_YAW_RATE,          # Yaw rate
            0.0,                        # Distance to current checkpoint
            0.0,                        # Distance to checkpoint+1
            0.0,                        # Distance to checkpoint+2
            0.0,                        # Cross-track error
            -np.pi,                     # Heading error
            -1.0,                       # Rudder angle
            -1.0,                       # Thrust
        ], dtype=np.float32)

        base_high=np.array([
            self.MAX_GRID_POS,          # Ship position x
            self.MAX_GRID_POS,          # Ship position y
            np.pi,                      # Ship heading
            self.MAX_SURGE_VELOCITY,    # Surge velocity
            self.MAX_SWAY_VELOCITY,     # Sway velocity
            self.MAX_YAW_RATE,          # Yaw rate
            self.MAX_GRID_POS,          # Distance to current checkpoint
            self.MAX_GRID_POS,          # Distance to checkpoint+1
            self.MAX_GRID_POS,          # Distance to checkpoint+2
            self.MAX_GRID_POS,          # Cross-track error
            np.pi,                      # Heading error
            1.0,                        # Rudder angle
            1.0,                        # Thrust
        ], dtype=np.float32)

        # Add wind/current if enabled
        if self.wind:
            wind_low = np.array([-1.0, -1.0], dtype=np.float32)
            wind_high = np.array([1.0, 1.0], dtype=np.float32)
            base_low = np.hstack([base_low, wind_low])
            base_high = np.hstack([base_high, wind_high])

        if self.current:
            current_low = np.array([-1.0, -1.0], dtype=np.float32)
            current_high = np.array([1.0, 1.0], dtype=np.float32)
            base_low = np.hstack([base_low, current_low])
            base_high = np.hstack([base_high, current_high])

        return gym.spaces.Box(low=base_low, high=base_high, dtype=np.float32)


    def _initialize_rendering(self):
        """Set up the rendering elements for visualization."""
        self.initialize_plots = False

        # self.fig, self.ax = plt.subplots(figsize=(18,15))
        # Create a temporary Tkinter root window to get screen dimensions
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()  # Close the temporary Tkinter window
        except tk.TclError:
            screen_width, screen_height = self.MAX_FIG_WIDTH, self.MAX_FIG_HEIGHT   # Fallback

        # Define figure dimensions (ensure it fits within screen)
        fig_width = min(self.MAX_FIG_WIDTH, screen_width)
        fig_height = min(self.MAX_FIG_HEIGHT, screen_height)
        dpi = self.DPI

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)

        # Get figure manager
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend()

        # Set window position based on the backend
        try:
            if backend in {"TkAgg"} and hasattr(manager, "window"):
                manager.window.wm_geometry(f"+0+0")  # Move to top-left
            elif backend in {"QtAgg", "Qt5Agg"} and hasattr(manager, "window"):
                manager.window.setGeometry(0, 0, fig_width, fig_height)
            elif backend == "GTK3Agg" and hasattr(manager, "window"):
                manager.window.move(0, 0)
            else:
                print(f"Backend {backend} does not support direct window positioning.")
        except Exception as e:
            print(f"Error setting window position: {e}")

        self.ship_plotC, = plt.plot([], [], 'bo', markersize=10, label='ShipC')
        self.target_plot, = plt.plot([], [], 'ro', markersize=10, label='Target')
        self.heading_line, = plt.plot([], [], color='black', linewidth=2, label='Heading')

        self.ax.set_xlim(self.MIN_GRID_POS, self.MAX_GRID_POS)
        self.ax.set_ylim(self.MIN_GRID_POS, self.MAX_GRID_POS)

        # Update plot title and legend
        self.ax.set_title('Ship Navigation in a Path Following Environment')
        self.ax.legend()


    @staticmethod
    def _normalize(val, min_val, max_val):
        return 2 * (val - min_val) / (max_val - min_val) - 1


    def randomize(self, randomization_scale: Optional[float] = None):
        """Randomize the ship's initial position within specified bounds.

        Args:
            randomization_scale: Maximum absolute value for position
                randomization. If None, uses the class's default scale.

        Returns:
            None

        Raises:
            ValueError: If randomization_scale is not positive
        """
        if randomization_scale is not None:
            if randomization_scale <= 0:
                raise ValueError("randomization_scale must be positive")

            self.randomization_scale = randomization_scale

        perturbation = np.random.uniform(
            low=-self.randomization_scale,
            high=self.randomization_scale,
            size=self.initial_ship_pos.shape
        )
        self.initial_ship_pos = np.clip(self.initial_ship_pos + perturbation, self.MIN_GRID_POS, self.MAX_GRID_POS)


    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to its initial state.

        Args:
            seed: Optional seed for random number generation
            kwargs: Additional arguments

        Returns:
            tuple: (observation, info) where:
                observation: Initial observation
                info: Additional information dictionary
        """
        super().reset(seed=seed)

        self.env_specific_reset()

        self.ship_pos = np.copy(self.initial_ship_pos)
        self.previous_ship_pos = np.zeros(2, dtype=np.float32)
        self.checkpoint_index = 1

        direction_vector = self.checkpoints[self.checkpoint_index]['pos'] - self.ship_pos
        self.ship_angle = np.arctan2(direction_vector[1], direction_vector[0])  # Angle in radians
        # Set the initial state
        self.state = np.array([*self.ship_pos, self.ship_angle, 0.0, 0.0, 0.0], dtype=np.float32)

        # Reset control parameters
        self.rudder_error_sum = 0.0
        self.thrust_error_sum = 0.0
        self.previous_rudder_error = 0.0
        self.previous_thrust_error = 0.0
        self.previous_rudder_target = 0.0
        self.previous_thrust_target = 0.0
        self.filtered_rudder_derivative = 0.0
        self.filtered_thrust_derivative = 0.0

        self.step_count = 0
        self.stuck_steps = 0

        return self._get_obs(), {}


    def _get_obs(self) -> np.ndarray:
        """Construct and return the normalized observation vector.

        Returns:
            np.ndarray: Normalized observation array containing:
            - Position (normalized to [-1,1] in grid)
            - Heading (normalized to [-1,1] in radians)
            - Velocities (normalized to [-1,1] relative to max)
            - Distances to current and next checkpoints (normalized)
            - Cross-track and heading errors (normalized)
            - Current control actions
            - Optional wind/current observations
        """
        # Normalize positions
        norm_pos = self._normalize(self.ship_pos, self.MIN_GRID_POS, self.MAX_GRID_POS)

        # Normalize velocities
        norm_velocities = np.array([
            np.clip(self.state[3] / self.MAX_SURGE_VELOCITY, -1, 1),
            np.clip(self.state[4] / (self.MAX_SWAY_VELOCITY/2), -1, 1),
            np.clip(self.state[5] / self.MAX_YAW_RATE, -1, 1)
        ], dtype=np.float32)

        checkpoint_idx = self.checkpoint_index if self.checkpoint_index < len(self.checkpoints) else len(self.checkpoints) - 1

        # Checkpoint distances
        current_checkpoint_pos = self.checkpoints[checkpoint_idx]['pos']
        distance_to_checkpoint = np.linalg.norm(self.ship_pos - current_checkpoint_pos)
        norm_distance = distance_to_checkpoint / self.max_dist

        # Next checkpoint distances
        norm_next_distances = np.zeros(2, dtype=np.float32)
        for i in range(1, 3):
            idx = checkpoint_idx + i
            if idx < len(self.checkpoints):
                next_pos = self.checkpoints[idx]['pos']
                norm_next_distances[i-1] = np.linalg.norm(self.ship_pos - next_pos) / self.max_dist

        # Cross-track error
        prev_checkpoint_pos = self.checkpoints[checkpoint_idx - 1]['pos']
        cross_track_error = self._distance_from_point_to_line(
            self.ship_pos, prev_checkpoint_pos, current_checkpoint_pos)
        norm_cross_error = cross_track_error / (self.CHECKPOINTS_DISTANCE / 2)

        # Heading error
        direction_to_checkpoint = current_checkpoint_pos - self.ship_pos
        desired_heading = np.arctan2(direction_to_checkpoint[1], direction_to_checkpoint[0])
        heading_error = (desired_heading - self.state[2] + np.pi) % (2 * np.pi) - np.pi

        # Build observation
        obs = np.hstack([
            norm_pos,                               # Normalized position
            [self.state[2] / np.pi],                # Normalized heading
            norm_velocities,                        # Normalized velocities
            [norm_distance],                        # Distance to current checkpoint
            norm_next_distances,                    # Distances to next checkpoints
            [norm_cross_error],                     # Normalized cross-track error
            [heading_error / np.pi],                # Normalized heading error
            self.current_action,                    # Current action
        ])

        # Add environmental observations if enabled
        if self.wind:
            obs = np.hstack([obs, self.wind_direction * self.wind_strength])

        if self.current:
            obs = np.hstack([obs, self.current_direction * self.current_strength])

        return obs.astype(np.float32)


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment timestep.

        Args:
            action: Array-like with [rudder, thrust] commands in [-1, 1]

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must be shape (2,), got {action.shape}")

        # Update dynamics and get reward
        smoothened_action = self._smoothen_action(action)
        self._update_ship_dynamics(smoothened_action)
        reward, terminated = self._calculate_reward()

        # Step limit check
        self.step_count += 1
        if not terminated and self.step_count >= self.max_steps:
            reward = self.MAX_STEPS_PENALTY
            terminated = True

        # Collision check
        if not terminated and not check_collision_ship(self.ship_pos, self.polygon_shape):
            reward = self.COLLISION_PENALTY
            terminated = True

        return self._get_obs(), reward, terminated, False, {}


    def _smoothen_action(self, action: np.ndarray):
        """Smoothen the action to prevent erratic behaviour of the ship.

        Args:
            action: Array of [rudder, thrust] commands
        """
        # Apply rate limiting to action changes
        target_rudder, target_thrust = action[0], abs(action[1])
        current_rudder, current_thrust = self.current_action

        # Calculate and limit rudder change
        rudder_change = target_rudder - current_rudder
        if abs(rudder_change) > self.MAX_RUDDER_RATE:
            rudder_change = np.sign(rudder_change) * self.MAX_RUDDER_RATE

        # Calculate and limit thrust change
        thrust_change = target_thrust - current_thrust
        if abs(thrust_change) > self.MAX_THRUST_RATE:
            thrust_change = np.sign(thrust_change) * self.MAX_THRUST_RATE

        # Apply gradual changes
        gradual_rudder = current_rudder + rudder_change
        gradual_thrust = current_thrust + thrust_change

        # Smooth actions
        final_rudder = gradual_rudder if abs(target_rudder - current_rudder) > 0.2 else current_rudder

        # Apply PID controller and update current action
        # smoothed_action = self._apply_pi_controller(smoothed_action)

        self.previous_rudder_target = self.current_action[0]
        self.previous_thrust_target = self.current_action[1]
        self.current_action = np.array([final_rudder, gradual_thrust], dtype=np.float32)

        return self.current_action


    def _update_ship_dynamics(self, action: np.ndarray) -> None:
        """Update ship state based on actions and environmental effects.

        Args:
            action: Array of [rudder, thrust] commands
        """
        # Convert to physical values
        delta_r = np.radians(action[0] * 60)
        t = action[1] * 60

        # Current state
        x, y, psi, u, v, r = self.state

        # Environmental effects in ship coordinates
        wind_effect = np.zeros(2, dtype=np.float32)
        if self.wind:
            relative_wind_angle = self.radians_wind - psi
            wind_effect = np.array([
                self.wind_strength * np.cos(relative_wind_angle),
                self.wind_strength * np.sin(relative_wind_angle)
            ], dtype=np.float32)

        current_effect = np.zeros(2, dtype=np.float32)
        if self.current:
            relative_current_angle = self.radians_current - psi
            current_effect = np.array([
                self.current_strength * np.cos(relative_current_angle),
                self.current_strength * np.sin(relative_current_angle)
            ], dtype=np.float32)

        # Precompute reusable values
        sin_delta_r = np.sin(delta_r)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Update dynamics (simplified 3DOF model)
        du = self.k_t * t + self.xu * u + wind_effect[0] + current_effect[0]
        dv = self.k_v * sin_delta_r + self.yv * v + wind_effect[1] + current_effect[1]
        dr = self.k_r * delta_r + self.nr * r + self.yv_r * v + v * u / self.l - self.YAW_RATE_DAMPING * r

        # Update state with limits
        new_u = np.clip(u + du * self.time_step, self.MIN_SURGE_VELOCITY, self.MAX_SURGE_VELOCITY)
        new_v = np.clip(v + dv * self.time_step, self.MIN_SWAY_VELOCITY, self.MAX_SWAY_VELOCITY)
        new_r = np.clip(r + dr * self.time_step, self.MIN_YAW_RATE, self.MAX_YAW_RATE)  # Update yaw rate with limits

        # Update position and heading
        dx = new_u * cos_psi - new_v * sin_psi
        dy = new_u * sin_psi + new_v * cos_psi
        dpsi = new_r

        # Store previous state and update
        self.previous_ship_pos = np.copy(self.ship_pos)
        self.previous_heading = self.state[2]

        new_x = np.clip(self.state[0] + dx * self.time_step, self.MIN_GRID_POS, self.MAX_GRID_POS)
        new_y = np.clip(self.state[1] + dy * self.time_step, self.MIN_GRID_POS, self.MAX_GRID_POS)
        new_heading = self.state[2] + dpsi * self.time_step # % (2 * np.pi)

        # Update state
        self.state = np.array([new_x, new_y, new_heading, new_u, new_v, new_r], dtype=np.float32)

        # Update ship position
        self.ship_pos = self.state[:2]


    @staticmethod
    @lru_cache(maxsize=1024)
    def _distance_from_point_to_line_cached(point: tuple, line_seg_start: tuple, line_seg_end: tuple) -> float:
        """Calculate perpendicular distance from point to line segment.

        Args:
            point: Point coordinates [x,y]
            line_seg_start: Line segment start point [x,y]
            line_seg_end: Line segment end point [x,y]

        Returns:
            float: Perpendicular distance from point to line
        """
        point = np.array(point)
        line_seg_start = np.array(line_seg_start)
        line_seg_end = np.array(line_seg_end)

        line_vec = line_seg_end - line_seg_start
        point_vec = point - line_seg_start

        # Line magnitude squared (to avoid division by zero)
        line_mag_squared = np.dot(line_vec, line_vec)
        if line_mag_squared == 0:
            # If the two points defining the line are identical, return the distance to this point
            return np.linalg.norm(point - line_seg_start)

        # Projection of the point onto the line
        projection_scalar = np.dot(point_vec, line_vec) / line_mag_squared
        projected_point = line_seg_start + projection_scalar * line_vec

        # Distance from the point to the projected point on the infinite line
        return np.linalg.norm(point - projected_point)


    def _distance_from_point_to_line(self, point: np.ndarray,
                                     line_start: np.ndarray,
                                     line_end: np.ndarray) -> float:
        """Calculate distance with caching wrapper."""
        return self._distance_from_point_to_line_cached(tuple(point), tuple(line_start), tuple(line_end))


    @staticmethod
    def _calculate_heading_error(target_heading: float,
                                 current_heading: float,
                                 dead_zone_deg: float = 5.0) -> float:
        """Calculate heading error with dead zone handling.

        Args:
            target_heading: Desired heading in radians
            current_heading: Current heading in radians
            dead_zone_deg: Angular dead zone in degrees

        Returns:
            float: Heading error in radians, 0 if within dead zone
        """
        error = (target_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
        dead_zone = np.radians(dead_zone_deg)

        return 0.0 if abs(error) < dead_zone else error


    def _calculate_reward(self) -> Tuple[float, bool]:
        """Calculate reward and termination conditions.

        Returns:
            tuple: (reward, done) where:
                reward: Calculated reward value
                done: True if episode should terminate
        """
        prev_checkpoint_pos = self.checkpoints[self.checkpoint_index - 1]['pos']
        current_checkpoint = self.checkpoints[self.checkpoint_index]
        current_checkpoint_pos = current_checkpoint['pos']

        # === Path Following Reward ===
        path_vec = current_checkpoint_pos - prev_checkpoint_pos
        path_length = np.linalg.norm(path_vec)
        path_unit = path_vec / path_length

        rel_prev = self.previous_ship_pos - prev_checkpoint_pos
        rel_now = self.ship_pos - prev_checkpoint_pos

        proj_prev = np.dot(rel_prev, path_unit)
        proj_now = np.dot(rel_now, path_unit)

        # Normalized forward progress
        progress_ratio = (proj_now - proj_prev) / path_length
        forward_reward = self.REWARD_DISTANCE_SCALE * np.tanh(progress_ratio)

        # Velocity alignment with path
        velocity = self.ship_pos - self.previous_ship_pos
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > 1e-3:
            velocity_unit = velocity / velocity_norm
            path_alignment_reward = self.REWARD_DIRECTION_SCALE * np.dot(velocity_unit, path_unit)
        else:
            path_alignment_reward = 0.0

        # Perpendicular distance penalty
        perp_vector = rel_now - proj_now * path_unit
        perp_dist = np.linalg.norm(perp_vector)
        path_deviation_penalty = -self.PENALTY_DISTANCE_SCALE * np.tanh(perp_dist)

        path_following_reward = forward_reward + path_alignment_reward + path_deviation_penalty

        # === Heading Alignment ===
        direction_vec = current_checkpoint_pos - self.ship_pos
        self.desired_heading = np.arctan2(direction_vec[1], direction_vec[0])
        heading_error = self._calculate_heading_error(self.desired_heading, self.state[2])
        heading_alignment_reward = np.exp(-abs(heading_error))

        # === Cross-Track Penalty ===
        cross_track_error = self._distance_from_point_to_line(self.ship_pos, prev_checkpoint_pos, current_checkpoint_pos)
        cross_track_penalty = -0.5 * np.tanh(cross_track_error / self.CROSS_TRACK_ERROR_PENALTY_SCALE)
        self.cross_error = cross_track_error

        # === Action Penalties ===
        rudder_penalty = -0.2 * abs(self.current_action[0])
        # rudder_change_penalty = -0.5 * abs(self.current_action[0] - self.previous_rudder_target)
        # thrust_change_penalty = -0.1 * abs(self.current_action[1] - self.previous_thrust_target)

        # === Combined Reward ===
        reward = (
                self.reward_weights['distance'] * path_following_reward +
                self.reward_weights['heading'] * heading_alignment_reward +
                self.reward_weights['cross_track'] * cross_track_penalty +
                self.reward_weights['rudder'] * rudder_penalty  # + rudder_change_penalty + thrust_change_penalty
        )

        # === Stuck Penalty ===
        movement = np.linalg.norm(self.ship_pos - self.previous_ship_pos)
        if movement < 0.07:
            self.stuck_steps += 1
            if self.stuck_steps > 40:
                reward -= 0.6
        else:
            self.stuck_steps = 0

        # === Termination and Reward Shaping ===

        # Determine if checkpoint (waypoint or target) was reached
        done = False
        reward += self._is_checkpoint_reached_or_passed(current_checkpoint)

        # Early termination conditions
        if (
                cross_track_error > 2.0 * self.CHECKPOINTS_DISTANCE or
                abs(self.state[2] - self.previous_heading) > np.pi / 2.0
           ):
            reward -= 1.0
            done = True

        done |= self.checkpoint_index >= len(self.checkpoints)

        return reward, done


    def _is_checkpoint_reached_or_passed(self, current_checkpoint):
        """Check if current checkpoint is reached or passed"""
        reward = np.float32(0.0)
        checkpoint_reached_or_passed = False
        checkpoint_distance = np.linalg.norm(self.ship_pos - current_checkpoint['pos'])

        if (checkpoint_distance <= current_checkpoint['radius']):
            if self.checkpoint_index == len(self.checkpoints) - 1:
                print(f"Target REACHED at distance {checkpoint_distance}")
            # else:
            #     print(f"Checkpoint {self.checkpoint_index} reached")    # REACHED at distance {checkpoint_distance}")
            reward = current_checkpoint['reward']
            checkpoint_reached_or_passed = True
        elif (self._is_near_perpendicular_line(current_checkpoint)):
            if self.checkpoint_index == len(self.checkpoints) - 1:
                print(f"Target passed at distance {checkpoint_distance}")
            # else:
            #     print(f"Checkpoint {self.checkpoint_index} passed")     # at distance {checkpoint_distance}")
            checkpoint_reached_or_passed = True
        elif self.checkpoint_index >= max(0, len(self.checkpoints) - self.SHAPING_WINDOW):
            target_area_size = self.checkpoints[-1]['radius']
            decay = self.DECAY_SCALE * (checkpoint_distance - target_area_size) / max(target_area_size, 1e-6)
            reward = self.checkpoints[-1]['reward'] * np.exp(-decay)

        if checkpoint_reached_or_passed:
            self.checkpoint_index += 1
            self.step_count = 0

        return reward


    def _is_near_perpendicular_line(self, checkpoint):
        """Check if ship is close enough to checkpoint's perpendicular line"""
        line_start, line_end = checkpoint['perpendicular_line']
        return self._distance_from_point_to_line(self.ship_pos, line_start, line_end) <= 2.0


    def _draw_fixed_landmarks(self):
        # Draw the path as straight lines
        if self.checkpoints:  # Check if there are checkpoints to draw
            # Start with the ship position and end with the target position
            path_points = [self.ship_pos] + [checkpoint['pos'] for checkpoint in self.checkpoints] + [self.target_pos]

            # Loop through the path points and draw straight lines between consecutive points
            for i in range(len(path_points) - 1):
                start_point = path_points[i]
                end_point = path_points[i + 1]
                self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                             'g-', label='Path' if i == 0 else "")  # Green lines for the path

            for i, checkpoint in enumerate(self.checkpoints):
                check = patches.Circle((checkpoint['pos'][0], checkpoint['pos'][1]),
                                       10, color='black', alpha=0.3)
                self.ax.add_patch(check)
                start_point = checkpoint['perpendicular_line'][0]
                end_point = checkpoint['perpendicular_line'][1]
                self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                             'g-', label='Path' if i == 0 else "")  # Green lines for the path

        # Draw the target location (if necessary)
        self.target_plot.set_data(self.target_pos[0:1], self.target_pos[1:2])

        # Add the obstacles as polygons (uncomment if needed)
        # for x, y in self.obstacles:
        #     rect = patches.Circle((x, y), 100, color='black', alpha=0.3)
        #     self.ax.add_patch(rect)

        # Add the polygon to the plot
        polygon_patch = patches.Polygon(self.obstacles, closed=True, edgecolor='r', facecolor='none',
                                        lw=2, label='Waterway')
        western_scheldt = patches.Polygon(self.overall, closed=True, edgecolor='brown', facecolor='none',
                                          lw=2, label='Western Scheldt')
        self.ax.add_patch(polygon_patch)
        self.ax.add_patch(western_scheldt)


    def _on_draw(self, event):
        """
        Updates the cached background after a canvas redraw.

        This ensures blitting stays in sync after zoom or pan interactions.

        Args:
            event (matplotlib.backend_bases.DrawEvent): The matplotlib draw event.
        """
        if event.canvas is self.fig.canvas:
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)


    def render(self):
        """Render the environment and visualize the ship's movement."""
        if self.render_mode != 'human':
            return

        if self.initialize_plots and not hasattr(self, "ship_plotC"):
            self._initialize_rendering()
            self._draw_fixed_landmarks()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig.canvas.mpl_connect('draw_event', self._on_draw)
            plt.show(block=False)

        # Restore background before redraw
        if hasattr(self, "background"):
            self.fig.canvas.restore_region(self.background)

        # Update ship's position
        heading_line_length = 30
        self.ship_plotC.set_data(self.ship_pos[0:1], self.ship_pos[1:2])
        heading_x = self.ship_pos[0] + np.cos(self.state[2]) * heading_line_length
        heading_y = self.ship_pos[1] + np.sin(self.state[2]) * heading_line_length
        self.heading_line.set_data([self.ship_pos[0], heading_x], [self.ship_pos[1], heading_y])

        if hasattr(self.fig.canvas, "blit"):
            self.ax.draw_artist(self.ship_plotC)
            self.ax.draw_artist(self.heading_line)
            self.fig.canvas.blit(self.ax.bbox)
        else:
            self.fig.canvas.draw()

        self.fig.canvas.flush_events()


    def close(self):
        """Close the environment."""
        if self.render_mode == 'human':
            plt.close()
