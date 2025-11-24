import matplotlib

try:
    matplotlib.use('TkAgg')  # Try using the interactive backend
except ImportError:
    print("Warning: 'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use('Agg')  # Use non-interactive backend as fallback

import os
import math
import torch
import numpy as np
import tkinter as tk
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from shapely.geometry import Point
from shapely.geometry import Polygon
from typing import Tuple, Optional, Dict
from fw.simulators.base_env import BaseEnv


class PySimPolarEnv(BaseEnv):
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
    SHAPING_WINDOW = 3
    DECAY_SCALE = 1.0
    REWARD_DISTANCE_SCALE = 2.0
    REWARD_DIRECTION_SCALE = 1.0
    PENALTY_DISTANCE_SCALE = 1.0
    CROSS_TRACK_ERROR_PENALTY_SCALE = 50.0
    EARLY_TERMINATION_PENALTY = -15
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

    # Performance optimization parameters
    NUM_RAYS = 180  # Reduced from 360 for performance
    MAX_DISTANCE = 50  # Reduced from 100 for performance
    BIN_SIZE = 1.0
    RENDER_EVERY_N_STEPS = 10  # Render every 10 steps during training

    def __init__(self, ship_pos: np.ndarray, target_pos: np.ndarray, render_mode: Optional[str] = None,
                 time_step: float = 0.1, max_steps: int = 1500,
                 verbose: Optional[bool] = None, wind: bool = False, current: bool = False):
        """Initialize the ship navigation environment.

        Args:
            ship_pos: Optional initial ship position
            target_pos: Optional target position
            render_mode: Either None or 'human' for visualization
            time_step: Simulation time step in seconds
            max_steps: Maximum steps per episode
            verbose: Whether to print debug information
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
        self.step_count = 0

        # Add a check to see if the ship has reached the target
        self.target_reached = False

        # Rendering
        self.initialize_plots = True

        # Convert inputs to numpy arrays
        self.ship_pos = np.asarray(ship_pos, dtype=np.float32)
        self.target_pos = np.asarray(target_pos, dtype=np.float32)

        # Store the previous distance to the target to calculate reward shaping
        self.previous_distance_to_target = np.linalg.norm(self.target_pos - self.ship_pos)

        # Hydrodynamic coefficients
        self.xu = -0.02  # Surge damping
        self.yv = -0.4  # Sway damping
        self.yv_r = -0.09  # Default sway-to-yaw coupling coefficient
        self.nr = -0.26  # Yaw damping
        self.l = 50.0  # Ship length

        # Dynamic coefficients
        self.k_t = 0.05  # Thrust coefficient
        self.k_r = 0.039  # Rudder coefficient
        self.k_v = 0.03  # Sway coefficient

        """Initialize the ship's state variables."""
        self.initial_ship_pos = np.array(ship_pos, dtype=np.float32)
        self.ship_pos = np.copy(self.initial_ship_pos)
        self.previous_ship_pos = self.ship_pos.copy()
        self.previous_heading = 0.0
        self.ship_angle = 0.0
        self.randomization_scale = 1.0
        self.max_dist = np.sqrt(2) * self.MAX_GRID_POS
        self.state = np.array([*self.ship_pos, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.current_action = np.zeros(2, dtype=np.float32)
        self.target_pos = target_pos

        """Initialize PID controller and related parameters."""
        # PID gains
        self.rudder_kp = 0.2  # Proportional gain for rudder
        self.rudder_ki = 0.1  # Integral gain for rudder
        self.rudder_kd = 0.15  # Helps dampen oscillations
        self.thrust_kp = 0.2  # Proportional gain for thrust
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
        self.radians_current = np.pi
        self.current_direction = np.array([np.cos(self.radians_current), np.sin(self.radians_current)],
                                          dtype=np.float32)
        self.current_strength = 0.35
        self.radians_wind = np.pi / 2
        self.wind_direction = np.array([np.cos(self.radians_wind), np.sin(self.radians_wind)], dtype=np.float32)
        self.wind_strength = 0.35

        # Performance optimization attributes (all pickleable)
        self._obs_cache = {}  # Simple dict is pickleable
        self._last_ship_pos = None
        self._last_ship_angle = None
        self._cached_observation = None
        self._prepared_polygon = None  # Will be created on first use

        # Gym spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = self._initialize_observation_space()

        self.observed_surroundings = None

        self._load_environment_data()

    def _load_environment_data(self) -> None:
        base_path = os.path.dirname(os.path.abspath(__file__))

        def load_csv(name: str) -> np.ndarray:
            try:
                data = np.loadtxt(os.path.join(base_path, name), delimiter=',', skiprows=1)
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing required environment file: {name}")
            except Exception as e:
                raise RuntimeError(f"Failed to load {name}: {e}")
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError(f"{name} must be a 2D array with shape (N, 2)")
            return data

        # Load and validate environment components
        self.obstacles = load_csv('env_Sche_250cm_no_scale.csv')
        self.polygon_shape = Polygon(self.obstacles)
        self.overall = load_csv('env_Sche_no_scale.csv')

        # Don't create prepared_polygon here - create on first use to avoid pickling issues

    def __getstate__(self):
        """Control what gets pickled - exclude non-pickleable objects."""
        state = self.__dict__.copy()
        # Remove non-pickleable objects
        if '_prepared_polygon' in state:
            del state['_prepared_polygon']
        if 'fig' in state:
            del state['fig']
        if 'ax' in state:
            del state['ax']
        if '_polar_fig' in state:
            del state['_polar_fig']
        if '_polar_ax' in state:
            del state['_polar_ax']
        if '_polar_mesh' in state:
            del state['_polar_mesh']
        # Clear matplotlib-related objects
        state['initialize_plots'] = True
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # Reinitialize non-pickleable objects
        self._prepared_polygon = None
        self.initialize_plots = True

    @property
    def prepared_polygon(self):
        """Lazy initialization of prepared polygon to avoid pickling issues."""
        if self._prepared_polygon is None:
            from shapely.prepared import prep
            self._prepared_polygon = prep(self.polygon_shape)
        return self._prepared_polygon

    def _initialize_rendering(self):
        """Set up the rendering elements for visualization."""
        self.initialize_plots = False

        # Create a temporary Tkinter root window to get screen dimensions
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()  # Close the temporary Tkinter window
        except tk.TclError:
            screen_width, screen_height = self.MAX_FIG_WIDTH, self.MAX_FIG_HEIGHT  # Fallback

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
            assert manager is not None
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
        n_bins = int(self.MAX_DISTANCE / self.BIN_SIZE)
        total_polar_elements = self.NUM_RAYS * n_bins

        # Base observations
        base_low = np.array([0.0] * total_polar_elements + [
            self.MIN_SURGE_VELOCITY,  # Surge velocity
            self.MIN_SWAY_VELOCITY,  # Sway velocity
            self.MIN_YAW_RATE,  # Yaw rate
            0.0,  # Distance to current checkpoint
            -np.pi,  # Ship heading
            -1.0,  # Rudder angle
            -1.0,  # Thrust
        ], dtype=np.float32)

        base_high = np.array([1.0] * total_polar_elements + [
            self.MAX_SURGE_VELOCITY,  # Surge velocity
            self.MAX_SWAY_VELOCITY,  # Sway velocity
            self.MAX_YAW_RATE,  # Yaw rate
            self.MAX_GRID_POS,  # Distance to current checkpoint
            np.pi,  # Ship heading
            1.0,  # Rudder angle
            1.0,  # Thrust
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

    def _ship_in_open_water(self, ship_position, polygon: Polygon) -> bool:
        """
        Check if the ship makes contact with any edge of the polygon.
        """
        x, y = ship_position
        min_x, min_y, max_x, max_y = polygon.bounds

        # Combine bounding box checks
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

        contains = polygon.contains(Point(ship_position))

        return contains and self._has_enough_keel_clearance()

    @staticmethod
    def _has_enough_keel_clearance():
        return True  # Assume always enough clearance for simplicity

    @staticmethod
    def _compute_polar_coordinates_fast(ship_x: float, ship_y: float, ship_angle: float,
                                        num_rays: int, max_distance: int, bin_size: float) -> tuple:
        """Fast computation for polar coordinates without numba dependency."""
        n_bins = int(max_distance / bin_size)
        x_vals = np.zeros((num_rays, n_bins), dtype=np.float64)
        y_vals = np.zeros((num_rays, n_bins), dtype=np.float64)

        angle_step = 2 * np.pi / num_rays

        for i in range(num_rays):
            angle = -np.pi + (i * angle_step)
            total_angle = ship_angle + angle
            cos_angle = np.cos(total_angle)
            sin_angle = np.sin(total_angle)

            for j in range(n_bins):
                dist = (j * bin_size) + (bin_size / 2.0)
                x_vals[i, j] = ship_x + dist * cos_angle
                y_vals[i, j] = ship_y + dist * sin_angle

        return x_vals, y_vals

    def get_ship_surroundings_polar_coords(self, polygon: Polygon, ship_pos: Tuple[float, float], ship_angle: float,
                                           num_rays: int = None, max_distance: int = None) -> np.ndarray:
        """Optimized polar coordinates observation generation."""

        # Use optimized defaults if not specified
        if num_rays is None:
            num_rays = self.NUM_RAYS
        if max_distance is None:
            max_distance = self.MAX_DISTANCE

        # Check if we can reuse cached observation
        if (self._last_ship_pos is not None and self._last_ship_angle is not None and
                self._cached_observation is not None):
            pos_diff = np.linalg.norm(np.array(ship_pos) - self._last_ship_pos)
            angle_diff = abs(ship_angle - self._last_ship_angle)

            # Reuse cache if ship hasn't moved much (reduces computation by ~80%)
            if pos_diff < 2.0 and angle_diff < 0.1:  # 2 meters and 0.1 radians threshold
                return self._cached_observation

        # Compute all coordinates at once using optimized function
        x_vals, y_vals = self._compute_polar_coordinates_fast(
            ship_pos[0], ship_pos[1], ship_angle,
            num_rays, max_distance, self.BIN_SIZE
        )

        # Vectorized collision checking
        n_bins = int(max_distance / self.BIN_SIZE)
        mask = np.ones((num_rays, n_bins), dtype=np.float32)

        # Batch collision checks with bounds checking first
        for i in range(num_rays):
            for j in range(n_bins):
                x, y = x_vals[i, j], y_vals[i, j]

                # Fast bounds check first
                if not (self.MIN_GRID_POS <= x <= self.MAX_GRID_POS and
                        self.MIN_GRID_POS <= y <= self.MAX_GRID_POS):
                    mask[i, j] = 0.0
                    continue

                # Use cached result if available
                cache_key = (int(x / 10), int(y / 10))  # 10m resolution caching
                if cache_key in self._obs_cache:
                    mask[i, j] = self._obs_cache[cache_key]
                else:
                    contains = self.prepared_polygon.contains(Point(x, y))
                    self._obs_cache[cache_key] = float(contains)
                    mask[i, j] = float(contains)

        # Limit cache size
        if len(self._obs_cache) > 5000:
            # Remove oldest entries (Python 3.7+ preserves insertion order)
            keys_to_remove = list(self._obs_cache.keys())[:1000]
            for key in keys_to_remove:
                del self._obs_cache[key]

        # Cache the result
        self._last_ship_pos = np.array(ship_pos)
        self._last_ship_angle = ship_angle
        self._cached_observation = mask.flatten()

        return self._cached_observation

    @staticmethod
    def create_rotated_grid_from_polygon(polygon: Polygon, origin: Tuple[float, float],
                                         angle_degrees: float, grid_size: float = 1.0) -> torch.Tensor:
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)

        # Get polygon coordinates and shift origin to (0, 0)
        coords = np.array(polygon.exterior.coords)
        shifted_coords = coords - np.array(origin)

        # Create rotation matrix
        cos_angle = math.cos(-angle_rad)  # Negative for counter-clockwise rotation
        sin_angle = math.sin(-angle_rad)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])

        # Apply rotation to coordinates
        rotated_coords = shifted_coords @ rotation_matrix.T

        # Create rotated polygon
        from shapely.geometry import Polygon as ShapelyPolygon
        rotated_polygon = ShapelyPolygon(rotated_coords)

        # Find bounding box of rotated polygon
        minx, miny, maxx, maxy = rotated_polygon.bounds

        # Expand bounds to ensure full coverage
        minx = math.floor(minx / grid_size) * grid_size
        miny = math.floor(miny / grid_size) * grid_size
        maxx = math.ceil(maxx / grid_size) * grid_size
        maxy = math.ceil(maxy / grid_size) * grid_size

        # Calculate grid dimensions
        grid_width = int((maxx - minx) / grid_size)
        grid_height = int((maxy - miny) / grid_size)

        # Create coordinate grids
        x_coords = np.linspace(minx + grid_size / 2, maxx - grid_size / 2, grid_width)
        y_coords = np.linspace(miny + grid_size / 2, maxy - grid_size / 2, grid_height)

        # Create meshgrid
        x, y = np.meshgrid(x_coords, y_coords)

        # Flatten coordinates for polygon containment check
        points = np.column_stack([x.ravel(), y.ravel()])

        # Check which points are inside the polygon
        from shapely.geometry import Point as ShapelyPoint
        inside = np.array([rotated_polygon.contains(ShapelyPoint(point)) for point in points])

        # Reshape to grid format
        grid = inside.reshape(grid_height, grid_width)

        # Convert to PyTorch tensor
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        return grid_tensor

    def _update_ship_dynamics(self, action: np.ndarray) -> None:
        """Update ship state based on actions and environmental effects.

        Args:
            action: Array of [rudder, thrust] commands
        """
        self.previous_rudder_target = self.current_action[0]
        self.previous_thrust_target = self.current_action[1]

        self.current_action = action

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
        new_heading = self.state[2] + dpsi * self.time_step  # % (2 * np.pi)

        # Update state
        self.state = np.array([new_x, new_y, new_heading, new_u, new_v, new_r], dtype=np.float32)

        # Update ship position
        self.ship_pos = self.state[:2]

    def _compute_environment_transformation(self, action: np.ndarray) -> np.ndarray:
        """Compute transformation matrix to keep ship at origin while transforming environment.

        This method mimics _update_ship_dynamics but instead of updating the ship state,
        it returns a 3x3 transformation matrix that can be used to transform the environment
        such that the ship appears to remain at coordinates (0,0) with rotation 0.

        Args:
            action: Array of [rudder, thrust] commands

        Returns:
            np.ndarray: 3x3 transformation matrix for translating and rotating the environment
                       Matrix format: [[cos(θ), -sin(θ), tx],
                                     [sin(θ),  cos(θ), ty],
                                     [0,       0,      1 ]]
        """
        # Store current state to compute transformation without modifying it
        current_state = np.copy(self.state)
        current_ship_pos = np.copy(self.ship_pos)
        current_prev_pos = np.copy(self.previous_ship_pos)
        current_prev_heading = self.previous_heading

        # Temporarily update dynamics to get new position/heading
        self._update_ship_dynamics(action)

        # Get the new position and heading that would have been applied
        new_x, new_y, new_heading = self.state[0], self.state[1], self.state[2]

        # Restore the original state (keep ship at origin conceptually)
        self.state = current_state
        self.ship_pos = current_ship_pos
        self.previous_ship_pos = current_prev_pos
        self.previous_heading = current_prev_heading

        # Compute the inverse transformation to move environment instead of ship
        # If ship would move from (x0,y0,θ0) to (x1,y1,θ1),
        # we need to transform environment by (-x1,-y1,-θ1) relative to (-x0,-y0,-θ0)

        # Calculate the delta transformation
        dx = new_x - current_state[0]  # Change in x position
        dy = new_y - current_state[1]  # Change in y position
        dtheta = new_heading - current_state[2]  # Change in heading

        # Create inverse transformation matrix
        # Environment moves opposite to ship movement
        cos_theta = np.cos(-dtheta)
        sin_theta = np.sin(-dtheta)

        # Translation is negated (environment moves opposite to ship)
        tx = -dx
        ty = -dy

        # Create 3x3 homogeneous transformation matrix
        # This matrix will rotate the environment by -dtheta and translate by (-dx, -dy)
        transformation_matrix = np.array([
            [cos_theta, -sin_theta, tx],
            [sin_theta, cos_theta, ty],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        return transformation_matrix

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to its initial state."""
        self.ship_pos = np.copy(self.initial_ship_pos)
        self.state = np.array([*self.ship_pos, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        self.current_action = np.zeros(2, dtype=np.float32)
        self.previous_distance_to_target = np.linalg.norm(self.target_pos - self.ship_pos)
        self.target_reached = False

        # Clear caches on reset
        self._obs_cache.clear()
        self._last_ship_pos = None
        self._last_ship_angle = None
        self._cached_observation = None

        return self._get_obs(), {}

    def _check_collision(self) -> bool:
        """Check if the ship has collided with any obstacles."""
        return not self._ship_in_open_water(self.ship_pos, self.polygon_shape)

    def _has_reached_target(self) -> bool:
        """Check if the ship has reached the target area."""
        return np.linalg.norm(self.ship_pos - self.target_pos) < self.TARGET_AREA_SIZE

    def _calculate_reward(self) -> Tuple[float, bool]:
        """Calculate the reward and termination status for the current step."""
        terminated = False
        reward = 0.0

        # Collision Check
        if self._check_collision():
            reward = self.EARLY_TERMINATION_PENALTY - 25.0  # Extra penalty for collision
            terminated = True
            if self.verbose:
                print("Collision detected!")
            return reward, terminated

        # Target Reached Check
        if self._has_reached_target():
            reward = self.SUCCESS_REWARD
            terminated = True
            self.target_reached = True
            if self.verbose:
                print("Target reached!")
            return reward, terminated

        # Goal-seeking reward (shaping)
        distance_to_target = np.linalg.norm(self.target_pos - self.state[:2])
        reward += (self.previous_distance_to_target - distance_to_target) * self.REWARD_DISTANCE_SCALE
        self.previous_distance_to_target = distance_to_target

        # Direction reward (pointing towards target)
        angle_to_target = np.arctan2(self.target_pos[1] - self.state[1], self.target_pos[0] - self.state[0])
        heading_error = abs(angle_to_target - self.state[2])
        reward += np.cos(heading_error) * self.REWARD_DIRECTION_SCALE

        # Penalize for distance to target to encourage progress
        reward -= (distance_to_target / self.max_dist) * self.PENALTY_DISTANCE_SCALE

        # enalize zig-zagging (excessive rudder changes)
        rudder_change = abs(self.current_action[0] - self.previous_rudder_target)
        reward -= rudder_change * 0.1

        # Penalize for being too close to obstacles
        # Penalize if any of the forward-facing rays detect a close obstacle
        if self.observed_surroundings is not None:
            forward_rays = self.observed_surroundings.reshape(self.NUM_RAYS, -1)
            # considering rays from -45 to 45 degrees as forward
            forward_indices = list(range(self.NUM_RAYS - 45, self.NUM_RAYS)) + list(range(0, 45))
            forward_view = forward_rays[forward_indices]
            if np.any(forward_view[:, 0] == 0.0):  # checks the first bin for obstacles
                reward -= 2.0

        # Time penalty to encourage efficiency
        reward -= 0.01

        # Check for termination by max steps
        if self.step_count >= self.max_steps:
            terminated = True
            if self.verbose:
                print("Max steps reached.")

        return reward, terminated

    def step(self, action_in: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment timestep.
        Args:
            action_in: Array-like with [rudder, thrust] commands in [-1, 1]

        Returns:
            tuple: (observation, reward, terminated)
        """
        action = np.array([action_in[0], action_in[1]])
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must be shape (2,), got {action.shape}")

        self._update_ship_dynamics(action)
        self.step_count += 1

        # Only render every N steps for performance
        # if self.step_count % self.RENDER_EVERY_N_STEPS == 0:
        #     self.render()

        # Get observation before calculating reward, as reward might use it
        obs = self._get_obs()
        reward, terminated = self._calculate_reward()

        return obs, reward, terminated, False, {}

    def _get_obs(self) -> np.ndarray:
        self.observed_surroundings = self.get_ship_surroundings_polar_coords(
            self.polygon_shape,
            (self.state[0], self.state[1]),
            self.state[2],
            num_rays=self.NUM_RAYS,
            max_distance=self.MAX_DISTANCE
        )
        # Normalize velocities
        norm_velocities = np.array([
            np.clip(self.state[3] / self.MAX_SURGE_VELOCITY, -1, 1),
            np.clip(self.state[4] / (self.MAX_SWAY_VELOCITY / 2), -1, 1),
            np.clip(self.state[5] / self.MAX_YAW_RATE, -1, 1)
        ], dtype=np.float32)
        distance_to_target = np.linalg.norm(self.target_pos - self.state[:2])
        angle_to_target = np.arctan2(self.target_pos[1] - self.state[1], self.target_pos[0] - self.state[0])
        angle_to_target = angle_to_target - self.state[2]
        obs = np.hstack((
            self.observed_surroundings,
            norm_velocities,
            distance_to_target,
            angle_to_target,
            self.current_action
        )).flatten()

        return obs

    def visualize_polar_observation(
            self,
            ship_pos: Optional[np.ndarray] = None,
            ship_angle: Optional[float] = None,
            num_rays: int = 36,
            num_bins: int = 5,
            max_distance: float = 100.0,
            persistent: bool = True,
    ) -> None:
        """
        Visualize the ship's polar surroundings as a circular (radar-like) grid.

        If `persistent=True`, it reuses a single figure window for live updates.
        The radar automatically rotates so the ship's heading (bow) points to 0°.
        """
        # Use current observation if available
        if self.observed_surroundings is None:
            return

        # Infer resolution from returned observation
        inferred_rays = self.NUM_RAYS
        inferred_bins = len(self.observed_surroundings) // inferred_rays
        values = self.observed_surroundings.reshape(inferred_rays, inferred_bins)

        # --- Downsample to desired visual resolution ---
        ray_step = inferred_rays // num_rays
        bin_step = inferred_bins // num_bins
        values_reduced = np.zeros((num_bins, num_rays))
        for i in range(num_bins):
            for j in range(num_rays):
                ray_slice = slice(j * ray_step, (j + 1) * ray_step)
                bin_slice = slice(i * bin_step, (i + 1) * bin_step)
                values_reduced[i, j] = values[ray_slice, bin_slice].max()

        # --- Prepare polar grid ---
        theta = np.linspace(0, 2 * np.pi, num_rays + 1)
        r = np.linspace(0, max_distance, num_bins + 1)
        t, r = np.meshgrid(theta, r)

        # --- Initialize or update persistent figure ---
        if persistent:
            if not hasattr(self, "_polar_fig") or not hasattr(self, "_polar_ax"):
                # Create new figure once
                self._polar_fig, self._polar_ax = plt.subplots(
                    subplot_kw={"projection": "polar"}, figsize=(6, 6)
                )
                self._polar_mesh = self._polar_ax.pcolormesh(
                    t, r, values_reduced, cmap=plt.cm.binary, shading="auto"
                )
                self._polar_ax.set_ylim(0, max_distance)
                self._polar_ax.set_title("Polar Observation Grid", va="bottom")
                self._polar_ax.set_xticks(np.linspace(0, 2 * np.pi, num_rays, endpoint=False))
                self._polar_ax.set_yticks(np.linspace(0, max_distance, num_bins + 1))
                self._polar_ax.grid(True, linestyle="--", alpha=0.5)

                # Align ship's heading (bow = 0°)
                self._polar_ax.set_theta_zero_location("N")  # 0° at top
                self._polar_ax.set_theta_direction(-1)  # clockwise rotation
                plt.ion()  # Interactive mode on
                plt.show(block=False)
            else:
                # --- Safely update existing plot ---
                try:
                    self._polar_fig.canvas.mpl_disconnect(getattr(self, "_polar_draw_cid", None))
                except Exception:
                    pass

                # Ensure heading is finite
                # Ensure heading is a valid float
                try:
                    ship_angle = float(ship_angle) if ship_angle is not None else self.state[2]
                except (TypeError, ValueError):
                    ship_angle = 0.0

                if not np.isfinite(ship_angle):
                    ship_angle = 0.0

                # Update mesh data
                self._polar_mesh.set_array(values_reduced.ravel())

                # Safely update heading alignment
                with plt.rc_context():
                    self._polar_ax.set_theta_offset(ship_angle % (2 * np.pi))

                # Redraw figure safely
                try:
                    self._polar_fig.canvas.draw_idle()
                    self._polar_fig.canvas.flush_events()
                except Exception:
                    # Fallback if backend is busy
                    plt.pause(0.001)
        else:
            # One-shot visualization (non-persistent)
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
            ax.pcolormesh(t, r, values_reduced, cmap=plt.cm.binary, shading="auto")
            ax.set_ylim(0, max_distance)
            ax.set_title("Polar Observation Grid", va="bottom")
            ax.set_xticks(np.linspace(0, 2 * np.pi, num_rays, endpoint=False))
            ax.set_yticks(np.linspace(0, max_distance, num_bins + 1))
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_theta_offset(ship_angle if ship_angle is not None else self.state[2])
            plt.show(block=False)
            plt.close(fig)


    def _draw_fixed_landmarks(self):
        # Draw the path as straight lines
        # if self.checkpoints:  # Check if there are checkpoints to draw
        #     # Start with the ship position and end with the target position
        #     path_points = [self.ship_pos] + [checkpoint['pos'] for checkpoint in self.checkpoints] + [self.target_pos]
        #
        #     # Loop through the path points and draw straight lines between consecutive points
        #     for i in range(len(path_points) - 1):
        #         start_point = path_points[i]
        #         end_point = path_points[i + 1]
        #         self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
        #                      'g-', label='Path' if i == 0 else "")  # Green lines for the path
        #
        #     for i, checkpoint in enumerate(self.checkpoints):
        #         check = patches.Circle((checkpoint['pos'][0], checkpoint['pos'][1]),
        #                                10, color='black', alpha=0.3)
        #         self.ax.add_patch(check)
        #         start_point = checkpoint['perpendicular_line'][0]
        #         end_point = checkpoint['perpendicular_line'][1]
        #         self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
        #                      'g-', label='Path' if i == 0 else "")  # Green lines for the path

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
        # Skip rendering if not in human mode or not every N steps
        if self.render_mode != 'human' or self.step_count % self.RENDER_EVERY_N_STEPS != 0:
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
        # Clear caches
        self._obs_cache.clear()
        self._cached_observation = None
        self._last_ship_pos = None
        self._last_ship_angle = None
        self._prepared_polygon = None  # Allow recreation if needed

        # Close plots
        if hasattr(self, '_polar_fig'):
            plt.close(self._polar_fig)
        if hasattr(self, 'fig'):
            plt.close(self.fig)
