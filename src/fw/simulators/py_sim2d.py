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
import torch
import math

from functools import lru_cache
from shapely.geometry import Point
from shapely.geometry import Polygon
from typing import Tuple, Optional, Dict
from fw.simulators.base_env import BaseEnv
from fw.simulators.tools import create_checkpoints_from_simple_path



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
        """Loads static environment data including obstacles, map outlines, and trajectory checkpoints.

        This method performs the following steps:
        1. Loads CSV files containing obstacle, overall map, and trajectory point coordinates.
        2. Validates that each dataset has the expected 2D shape with two columns.
        3. Reduces and transforms the trajectory path into evenly spaced checkpoints.
        4. Calculates perpendicular guidance lines for each checkpoint.
        5. Initializes internal state such as checkpoints, target position, and navigation counters.

        Raises:
            FileNotFoundError: If any required CSV file is missing.
            ValueError: If any loaded file does not have the expected shape.
            RuntimeError: If any other error occurs while reading the files.
        """
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
        path = load_csv('trajectory_points_no_scale.csv')

        # Preprocess the trajectory path
        path = create_checkpoints_from_simple_path(path, self.CHECKPOINTS_DISTANCE)
        path = np.insert(path, 0, self.ship_pos, axis=0)  # Start with current ship position

        '''
        # Generate checkpoint data
        checkpoints = [
            {
                'pos': np.array(point, dtype=np.float32),
                'radius': self.CHECKPOINT_AREA_SIZE,
                'reward': (i / len(path)) * 10
            }
            for i, point in enumerate(path)
        ]

        # Set target properties
        checkpoints[-1]['radius'] = self.TARGET_AREA_SIZE
        checkpoints[-1]['reward'] = self.SUCCESS_REWARD

        # Add perpendicular lines to checkpoints
        lines = calculate_perpendicular_lines(checkpoints, line_length=50)
        self.checkpoints = [{**checkpoint, 'perpendicular_line': line} for checkpoint, line in zip(checkpoints, lines)]

        # Initialize environment state
        self.target_pos = self.checkpoints[-1]['pos']
        self.checkpoint_index = 1
        self.step_count = 0
        self.stuck_steps = 0
        self.cross_error = 0.0
        self.desired_heading = 0.0
        '''
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

    def create_rotated_grid_from_polygon(self, polygon: Polygon, origin: Tuple[float, float], 
                                       angle_degrees: float, grid_size: float = 1.0) -> torch.Tensor:
        """
        Create a 2D grid of squares containing the polygon, shifted and rotated.
        
        Args:
            polygon: Shapely Polygon object to be gridded
            origin: (x, y) coordinate to be shifted to (0, 0)
            angle_degrees: Rotation angle in degrees
            grid_size: Size of each grid square (default 1.0 for 1x1 squares)
        
        Returns:
            torch.Tensor: 2D grid where 1 indicates polygon coverage, 0 indicates empty space
        """
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
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Flatten coordinates for polygon containment check
        points = np.column_stack([X.ravel(), Y.ravel()])
        
        # Check which points are inside the polygon
        from shapely.geometry import Point as ShapelyPoint
        inside = np.array([rotated_polygon.contains(ShapelyPoint(point)) for point in points])
        
        # Reshape to grid format
        grid = inside.reshape(grid_height, grid_width)
        
        # Convert to PyTorch tensor
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        
        return grid_tensor