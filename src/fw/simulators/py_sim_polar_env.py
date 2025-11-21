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
import gymnasium as gym
import matplotlib.pyplot as plt

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

    def __init__(self, ship_pos: np.ndarray, target_pos: np.ndarray, time_step: float = 0.1, max_steps: int = 1500,
                 verbose: Optional[bool] = None, wind: bool = False, current: bool = False):
        """Initialize the ship navigation environment.
        Args:
            time_step: Simulation time step in seconds
            max_steps: Maximum steps per episode
            verbose: Whether to print debug information
            ship_pos: Optional initial ship position
            target_pos: Optional target position
            wind: Whether to enable wind effects
            current: Whether to enable current effects
        """
        # Environment parameters
        super().__init__()

        self.time_step = time_step
        self.max_steps = max_steps
        self.verbose = verbose
        self.wind = wind
        self.current = current
        self.step_count = 0

        # Add a check to see if the ship has reached the target
        self.target_reached = False

        # Convert inputs to numpy arrays
        self.ship_pos = np.asarray(ship_pos, dtype=np.float32)
        self.target_pos = np.asarray(target_pos, dtype=np.float32)

        # Store the previous distance to the target to calculate reward shaping
        self.previous_distance_to_target = np.linalg.norm(self.target_pos - self.ship_pos)

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
        self.radians_current = np.pi
        self.current_direction = np.array([np.cos(self.radians_current), np.sin(self.radians_current)], dtype=np.float32)
        self.current_strength = 0.35
        self.radians_wind = np.pi / 2
        self.wind_direction = np.array([np.cos(self.radians_wind), np.sin(self.radians_wind)], dtype=np.float32)
        self.wind_strength = 0.35

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
        base_low=np.array([0.0] * 72000 + [
                self.MIN_SURGE_VELOCITY,    # Surge velocity
                self.MIN_SWAY_VELOCITY,     # Sway velocity
                self.MIN_YAW_RATE,          # Yaw rate
                0.0,                        # Distance to current checkpoint
                -np.pi,                     # Ship heading
                -1.0,                       # Rudder angle
                -1.0,                       # Thrust
            ], dtype=np.float32)

        base_high=np.array([1.0] * 72000 + [
                self.MAX_SURGE_VELOCITY,    # Surge velocity
                self.MAX_SWAY_VELOCITY,     # Sway velocity
                self.MAX_YAW_RATE,          # Yaw rate
                self.MAX_GRID_POS,          # Distance to current checkpoint
                np.pi,                      # Ship heading
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


    def get_ship_surroundings_polar_coords(self, polygon: Polygon, ship_pos: Tuple[float, float], ship_angle: float,
                                          num_rays: int = 360, max_distance: int = 100) -> np.ndarray:

        distance_bin_size = 1.0
        half_bin = distance_bin_size / 2.0

        n_bins = max_distance / distance_bin_size
        ship_x, ship_y = ship_pos
        ship_angle = ship_angle
        # Generate rays
        angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)

        ''' OLD SLOW LOGIC
        result = np.array([], dtype=np.float32)
        for angle in angles:
            distances = np.full(math.ceil(n_bins), 1.0, dtype=np.float32)
            for j, obstacle_value in enumerate(distances):
                # Currently just checks the middle range of the bin for collision
                # TODO: check for collision in the whole bin
                x_val = ship_x + (j*distance_bin_size + half_bin) * np.cos(ship_angle + angle)
                y_val = ship_y + (j*distance_bin_size + half_bin) * np.sin(ship_angle + angle)

                if not self._ship_in_open_water((x_val, y_val), polygon):
                    distances[j] = 0.0
            result = np.append(result, distances)
        '''
        result = []
        bin_distances = np.arange(n_bins) * distance_bin_size + half_bin
        for angle in angles:
        # Vectorized x,y coordinates for all distances along this ray
            x_vals = ship_x + bin_distances * np.cos(ship_angle + angle)
            y_vals = ship_y + bin_distances * np.sin(ship_angle + angle)

        # Vectorize open-water check if possible
        # Otherwise, fall back to list comprehension
            mask = np.array([self._ship_in_open_water((x, y), polygon) for x, y in zip(x_vals, y_vals)])
        # 1.0 where open water, 0.0 where obstacle
            distances = mask.astype(np.float32)
            result.append(distances)
        return np.concatenate(result)
            

    def create_rotated_grid_from_polygon(self, polygon: Polygon, origin: Tuple[float, float], 
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
        x_coords = np.linspace(minx + grid_size/2, maxx - grid_size/2, grid_width)
        y_coords = np.linspace(miny + grid_size/2, maxy - grid_size/2, grid_height)
        
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
        new_heading = self.state[2] + dpsi * self.time_step # % (2 * np.pi)

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
            [sin_theta,  cos_theta, ty],
            [0.0,        0.0,       1.0]
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
            reward = self.EARLY_TERMINATION_PENALTY - 25.0 # Extra penalty for collision
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
            forward_rays = self.observed_surroundings.reshape(360, -1)
            # considering rays from -45 to 45 degrees as forward
            forward_view = np.concatenate((forward_rays[-45:], forward_rays[:45]))
            if np.any(forward_view[:, 0] == 0.0): # checks the first bin for obstacles
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
        
        # Get observation before calculating reward, as reward might use it
        obs = self._get_obs()
        reward, terminated = self._calculate_reward()
        
        return obs, reward, terminated, False, {}
        
    def _get_obs(self) -> np.ndarray:
        self.observed_surroundings = self.get_ship_surroundings_polar_coords(
            self.polygon_shape, 
            (self.state[0], self.state[1]), 
            self.state[2],
            num_rays=360,
            max_distance=200
        )
        # Normalize velocities
        norm_velocities = np.array([
            np.clip(self.state[3] / self.MAX_SURGE_VELOCITY, -1, 1),
            np.clip(self.state[4] / (self.MAX_SWAY_VELOCITY/2), -1, 1),
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
        # Infer resolution from returned observation
        inferred_rays = 360
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
                    ship_angle = float(ship_angle)
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
            ax.set_theta_offset(ship_angle)
            plt.show(block=False)
            plt.close(fig)
