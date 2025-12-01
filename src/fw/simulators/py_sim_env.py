import os
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gymnasium as gym
import tkinter as tk
import numpy as np

# Try interactive backend; fallback behavior unchanged
try:
    matplotlib.use("TkAgg")
except ImportError:
    print("Warning: 'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use("Agg")

from functools import lru_cache
from shapely.geometry import Point, Polygon
from typing import Dict, List, Optional, Tuple
from fw.simulators.tools import create_checkpoints_from_simple_path
from fw.simulators.base_env import BaseEnv


def calculate_perpendicular_lines(checkpoints: List[dict], line_length: float = 100.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculate perpendicular lines at each checkpoint using a smoothed tangent direction.

    Args:
        checkpoints: List of dicts containing checkpoint 'pos' (array-like) and possibly 'radius'
        line_length: Length of perpendicular line centered at checkpoint position

    Returns:
        list of tuples: (start_point, end_point) for each perpendicular line (as numpy arrays)
    """

    def smooth_tangent(check_points: List[dict], index: int) -> np.ndarray:
        """Compute a smoothed tangent at a checkpoint by averaging neighbor vectors."""
        # Convert inputs into numpy arrays for arithmetic (the code expects these)
        if index == 0:
            tangent = check_points[1]['pos'] - check_points[0]['pos']
        elif index == len(check_points) - 1:
            tangent = check_points[-1]['pos'] - check_points[-2]['pos']
        else:
            # Average forward and backward differences to smooth
            tangent = 0.5 * (
                check_points[index + 1]['pos'] - check_points[index]['pos'] +
                check_points[index]['pos'] - check_points[index - 1]['pos']
            )   # Average direction

        norm = np.linalg.norm(tangent)
        return (tangent / norm if norm != 0 else tangent).astype(np.float32)

    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, cp in enumerate(checkpoints):
        pos = np.asarray(cp["pos"], dtype=np.float32)
        smth_tangent = smooth_tangent(checkpoints, i)
        perp = np.array([-smth_tangent[1], smth_tangent[0]], dtype=np.float32)
        offset = perp * (line_length / 2.0)
        lines.append((pos + offset, pos - offset))
    return lines


class PySimEnv(BaseEnv):
    """
    Cleaned version of PySimEnv with improved structure and documentation.
    Functionality preserved.
    """

    # -----------------------
    # Physical limits / config
    # -----------------------
    MIN_SURGE_VELOCITY = 0.0
    MIN_SWAY_VELOCITY = -2.0
    MIN_YAW_RATE = -0.5
    MAX_SURGE_VELOCITY = 5.0
    MAX_SWAY_VELOCITY = 2.0
    MAX_YAW_RATE = 0.5
    YAW_RATE_DAMPING = 0.1

    # Rate limits applied in smoothing
    MAX_RUDDER_RATE = 0.06
    MAX_THRUST_RATE = 0.05

    # Grid limits
    CHECKPOINTS_DISTANCE = 350
    MIN_GRID_POS = -11700
    MAX_GRID_POS = 14500

    # Reward / shaping
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

    # -----------------------
    # Initialization
    # -----------------------
    def __init__(
        self,
        render_mode: Optional[str] = None,
        time_step: float = 0.1,
        max_steps: int = 1500,
        verbose: Optional[bool] = None,
        ship_pos: Optional[np.ndarray] = None,
        target_pos: Optional[np.ndarray] = None,
        wind: bool = False,
        current: bool = False,
    ) -> None:
        """
        Initialize the environment.

        Args:
            render_mode: 'human' for rendering, otherwise None.
            time_step: simulation time step (s).
            max_steps: maximum steps per episode (for truncation).
            verbose: whether to print informational logs.
            ship_pos: optional initial ship position (2-array).
            target_pos: optional target position (unused if checkpoints available).
            wind: enable wind effects.
            current: enable current effects.
        """
        super().__init__(render_mode)

        # Core parameters
        self.time_step = float(time_step)
        self.max_steps = int(max_steps)
        self.verbose = bool(verbose) if verbose is not None else False
        self.wind = bool(wind)
        self.current = bool(current)

        # State and control placeholders
        self._initialize_state(ship_pos)
        self._initialize_control_parameters()

        # Load environment static data and build checkpoints/polygons
        self._load_environment_data(target_pos)

        # Gym spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._initialize_observation_space()

        # Reward component weights (kept from original)
        self.reward_weights = {
            "forward": 0.5,
            "alignment": 0.1,
            "deviation": 0.4,
            "heading": 0.3,
            "cross_track": 0.1,
            "rudder": 0.05,
            "terminal": 0.2,
        }

        # Rendering flag
        self.initialize_plots = True

    # -----------------------
    # Initialization helpers
    # -----------------------
    def _initialize_state(self, ship_pos: Optional[np.ndarray]) -> None:
        """Initialize state vectors and convenience values."""
        if ship_pos is not None:
            init = np.array(ship_pos, dtype=np.float32)
        else:
            init = np.array([5.0, 5.0], dtype=np.float32)

        self.initial_ship_pos = init
        self.ship_pos = np.copy(self.initial_ship_pos)
        self.previous_ship_pos = np.zeros(2, dtype=np.float32)
        self.previous_heading = 0.0
        self.randomization_scale = 1.0
        self.max_dist = np.sqrt(2) * float(self.MAX_GRID_POS)

        # state vector: [x, y, psi, u, v, r]
        self.state = np.array([self.ship_pos[0], self.ship_pos[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.current_action = np.zeros(2, dtype=np.float32)

    def _initialize_control_parameters(self) -> None:
        """Initialize PID control parameters and environmental effect defaults."""
        # PID gains (kept from original)
        self.rudder_kp = 0.2
        self.rudder_ki = 0.1
        self.rudder_kd = 0.15
        self.thrust_kp = 0.2
        self.thrust_ki = 0.02
        self.thrust_kd = 0.05

        # Integral and derivative state
        self.rudder_error_sum = 0.0
        self.thrust_error_sum = 0.0
        self.previous_rudder_error = 0.0
        self.previous_thrust_error = 0.0
        self.previous_rudder_target = 0.0
        self.previous_thrust_target = 0.0
        self.filtered_rudder_derivative = 0.0
        self.filtered_thrust_derivative = 0.0

        # Environmental defaults (kept values)
        self.radians_current = np.radians(180.0)
        self.current_direction = np.array([np.cos(self.radians_current), np.sin(self.radians_current)], dtype=np.float32)
        self.current_strength = 0.35

        self.radians_wind = np.radians(90.0)
        self.wind_direction = np.array([np.cos(self.radians_wind), np.sin(self.radians_wind)], dtype=np.float32)
        self.wind_strength = 0.35

    # -----------------------
    # Environment data loaders
    # -----------------------
    def _load_environment_data(self, target_pos: np.ndarray) -> None:
        """
        Load static environment CSVs, build checkpoints, polygon shapes, and initialize
        path/target-related state.

        Exceptions:
            - FileNotFoundError if CSVs missing
            - RuntimeError for other IO issues
            - ValueError if CSV shapes are unexpected
        """
        base_path = os.path.dirname(os.path.abspath(__file__))

        def load_csv_strict(name: str) -> np.ndarray:
            csv_path = os.path.join(base_path, name)
            try:
                data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing required environment file: {csv_path}")
            except Exception as exc:
                raise RuntimeError(f"Failed to load {csv_path}: {exc}")
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError(f"{csv_path} must be a 2D array with shape (N, 2)")
            return data

        # Load obstacles and overall map shapes
        self.obstacles = load_csv_strict("env_Sche_250cm_no_scale.csv")
        self.polygon_shape = Polygon(self.obstacles)
        self.overall = load_csv_strict("env_Sche_no_scale.csv")

        # Load trajectory
        path = load_csv_strict("trajectory_points_no_scale.csv")

        # Preprocess path (stub currently returns same path)
        path = self._reduce_path(path, self.initial_ship_pos)

        # Create checkpoints spaced along path
        path = create_checkpoints_from_simple_path(path, self.CHECKPOINTS_DISTANCE)

        # Ensure start is current ship position
        path = np.insert(path, 0, self.ship_pos, axis=0)

        # Build checkpoint dicts
        checkpoints = [
            {
                "pos": np.array(point, dtype=np.float32),
                "radius": float(self.CHECKPOINT_AREA_SIZE),
                "reward": (i / max(1, len(path))) * 10.0,
            }
            for i, point in enumerate(path)
        ]

        # Last checkpoint is the target
        if checkpoints:
            checkpoints[-1]["radius"] = float(self.TARGET_AREA_SIZE)
            checkpoints[-1]["reward"] = float(self.SUCCESS_REWARD)

        # Perpendicular lines for guidance
        lines = calculate_perpendicular_lines(checkpoints, line_length=50.0)
        self.checkpoints = [{**cp, "perpendicular_line": line} for cp, line in zip(checkpoints, lines)]

        # Navigation and counters
        self.target_pos = self.checkpoints[-1]["pos"] if self.checkpoints else np.array(target_pos, dtype=np.float32)
        self.checkpoint_index = 1
        self.step_count = 0
        self.stuck_steps = 0
        self.cross_error = 0.0
        self.desired_heading = 0.0

        # Hydrodynamic & dynamic coefficients (kept original values)
        self.xu = -0.02
        self.yv = -0.4
        self.yv_r = -0.09
        self.nr = -0.26
        self.l = 50.0
        self.k_t = 0.05
        self.k_r = 0.039
        self.k_v = 0.03

    # -----------------------
    # Utility & math helpers
    # -----------------------
    def sample_point_in_river(self, nbr_samples: int = 1, max_tries: int = 20000) -> np.ndarray:
        """
        Sample N random points INSIDE the river polygon using rejection sampling.

        Returns:
            array of shape (N, 2)
        """
        minx, miny, maxx, maxy = self.polygon_shape.bounds

        samples = []
        tries = 0

        rng = np.random.default_rng()  # truly random

        while len(samples) < nbr_samples and tries < max_tries:
            tries += 1
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)

            if self._ship_in_open_water(np.array([x, y])):
                samples.append((x, y))

        if len(samples) < nbr_samples:
            print( f"Warning: Only generated {len(samples)} points out of requested {nbr_samples} " 
                   f"(polygon may be thin or max_tries reached)." )

        return np.array(samples, dtype=np.float32)

    def _has_enough_keel_clearance(self, depth_threshold: float = 0.5) -> bool:
        """
        Placeholder to check keel clearance; returns True by default to preserve
        existing behavior. Override or expand if real depth data available.
        """
        return True

    def _ship_in_open_water(self, ship_position: np.ndarray) -> bool:
        """
        Check if the ship is inside the polygon bounds and has keel clearance.

        Args:
            ship_position: (x, y) coordinates

        Returns:
            bool: True if ship is inside polygon and keel clearance check passes
        """
        x, y = float(ship_position[0]), float(ship_position[1])
        min_x, min_y, max_x, max_y = self.polygon_shape.bounds
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

        contains = self.polygon_shape.contains(Point(ship_position))
        deep_enough = self._has_enough_keel_clearance()
        return bool(contains and deep_enough)

    def _reduce_path(self, path: np.ndarray, start_pos: np.ndarray) -> np.ndarray:
        """
        Reduce path to start at the point closest to start_pos.
        Currently, a pass-through returning the original path (preserves behavior).
        """
        # Implementation intentionally left as originally was: returns input path.
        return path

    @staticmethod
    def _normalize(val: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Normalize value(s) to the range [-1, 1]. Works with scalars or arrays."""
        return 2.0 * (val - min_val) / (max_val - min_val) - 1.0

    @staticmethod
    @lru_cache(maxsize=1024)
    def _distance_from_point_to_line_cached(point_tuple: tuple, line_seg_start_tuple: tuple, line_seg_end_tuple: tuple) -> float:
        """
        Calculate perpendicular distance from point to infinite line defined by two points.
        This cached wrapper expects tuples to maximize cache hits.
        """
        point = np.array(point_tuple, dtype=np.float32)
        line_start = np.array(line_seg_start_tuple, dtype=np.float32)
        line_end = np.array(line_seg_end_tuple, dtype=np.float32)

        line_vec = line_end - line_start
        point_vec = point - line_start
        line_mag_squared = np.dot(line_vec, line_vec)

        if line_mag_squared == 0.0:
            return float(np.linalg.norm(point - line_start))

        projection_scalar = np.dot(point_vec, line_vec) / line_mag_squared
        projected_point = line_start + projection_scalar * line_vec
        return float(np.linalg.norm(point - projected_point))

    def _distance_from_point_to_line(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Wrapper that converts numpy args to tuple and calls cached function."""
        return self._distance_from_point_to_line_cached(tuple(point), tuple(line_start), tuple(line_end))

    @staticmethod
    def _calculate_heading_error(target_heading: float, current_heading: float, dead_zone_deg: float = 5.0) -> float:
        """
        Calculate the smallest angular error between target and current heading, with a dead zone.

        Args:
            target_heading: desired heading in radians
            current_heading: current heading in radians
            dead_zone_deg: threshold in degrees within which error is considered zero

        Returns:
            float: heading error in radians (0 if within dead zone)
        """
        error = (target_heading - current_heading + np.pi) % (2.0 * np.pi) - np.pi
        dead_zone = np.radians(dead_zone_deg)
        return 0.0 if abs(error) < dead_zone else float(error)

    # -----------------------
    # Action smoothing / PI controller
    # -----------------------
    def _apply_pi_controller(self, target_action: np.ndarray) -> np.ndarray:
        """
        Apply an internal PI/PID-like update to produce smoothed controls.

        Args:
            target_action: array-like [target_rudder, target_thrust] with values in [-1,1]

        Returns:
            np.ndarray: Updated [rudder, thrust] after PID integration and rate limiting.
        """
        target_rudder = float(target_action[0])
        target_thrust = float(abs(target_action[1]))  # thrust considered non-negative in the controller

        # Dead-zone handling keeps the previous target if change is small
        if abs(target_rudder - self.previous_rudder_target) < self.RUDDER_DEAD_ZONE:
            target_rudder = self.previous_rudder_target
        if abs(target_thrust - self.previous_thrust_target) < self.THRUST_DEAD_ZONE:
            target_thrust = self.previous_thrust_target

        # Errors relative to previous target
        rudder_error = target_rudder - self.previous_rudder_target
        thrust_error = target_thrust - self.previous_thrust_target

        # Derivative estimates (filtered)
        rudder_derivative = (rudder_error - self.previous_rudder_error) / max(self.time_step, 1e-9)
        thrust_derivative = (thrust_error - self.previous_thrust_error) / max(self.time_step, 1e-9)

        self.filtered_rudder_derivative = (
            self.DERIVATIVE_FILTER_ALPHA * rudder_derivative + (1.0 - self.DERIVATIVE_FILTER_ALPHA) * self.filtered_rudder_derivative
        )
        self.filtered_thrust_derivative = (
            self.DERIVATIVE_FILTER_ALPHA * thrust_derivative + (1.0 - self.DERIVATIVE_FILTER_ALPHA) * self.filtered_thrust_derivative
        )

        # Integrator anti-windup
        if abs(self.previous_rudder_target) < self.ANTI_WINDUP_THRESHOLD:
            self.rudder_error_sum = np.clip(self.rudder_error_sum + rudder_error * self.time_step, -0.5, 0.5)
        if abs(self.previous_thrust_target) < self.ANTI_WINDUP_THRESHOLD:
            self.thrust_error_sum = np.clip(self.thrust_error_sum + thrust_error * self.time_step, -0.5, 0.5)

        # PID outputs (proportional + integral + derivative)
        rudder_output = (
            self.rudder_kp * rudder_error + self.rudder_ki * self.rudder_error_sum + self.rudder_kd * self.filtered_rudder_derivative
        )
        thrust_output = (
            self.thrust_kp * thrust_error + self.thrust_ki * self.thrust_error_sum + self.thrust_kd * self.filtered_thrust_derivative
        )

        # Rate limiting on PID outputs (prevents sudden jumps)
        rudder_output = np.clip(rudder_output, -self.MAX_RUDDER_RATE_CHANGE, self.MAX_RUDDER_RATE_CHANGE)
        thrust_output = np.clip(thrust_output, -self.MAX_THRUST_RATE_CHANGE, self.MAX_THRUST_RATE_CHANGE)

        new_rudder = float(np.clip(self.previous_rudder_target + rudder_output, -1.0, 1.0))
        new_thrust = float(np.clip(self.previous_thrust_target + thrust_output, -1.0, 1.0))

        # Update stored PID state
        self.previous_rudder_error = rudder_error
        self.previous_thrust_error = thrust_error
        self.previous_rudder_target = new_rudder
        self.previous_thrust_target = new_thrust

        return np.array([new_rudder, new_thrust], dtype=np.float32)

    def _smoothen_action(self, action: np.ndarray) -> np.ndarray:
        """
        Smoothen the provided action to avoid abrupt control changes. This method
        combines simple rate limiting with a small heuristic for rudder changes.
        """
        action = np.asarray(action, dtype=np.float32)
        target_rudder, target_thrust = float(action[0]), float(abs(action[1]))
        current_rudder, current_thrust = float(self.current_action[0]), float(self.current_action[1])

        # Rudder rate limit
        rudder_change = target_rudder - current_rudder
        if abs(rudder_change) > self.MAX_RUDDER_RATE:
            rudder_change = np.sign(rudder_change) * self.MAX_RUDDER_RATE

        # Thrust rate limit
        thrust_change = target_thrust - current_thrust
        if abs(thrust_change) > self.MAX_THRUST_RATE:
            thrust_change = np.sign(thrust_change) * self.MAX_THRUST_RATE

        gradual_rudder = current_rudder + rudder_change
        gradual_thrust = current_thrust + thrust_change

        # Heuristic: if desired rudder change is tiny, keep current rudder to avoid jitter
        final_rudder = gradual_rudder if abs(target_rudder - current_rudder) > 0.2 else current_rudder

        return np.array([final_rudder, gradual_thrust], dtype=np.float32)

    # -----------------------
    # Observation & reset
    # -----------------------
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """
        Construct the observation space box using configured ranges.
        Observation structure (normalized values):
          [pos_x, pos_y, heading, surge_vel, sway_vel, yaw_rate,
           dist_ckpt, dist_ckpt+1, dist_ckpt+2, cross_track, heading_error,
           rudder, thrust, (wind?), (current?)]
        """
        base_low = np.array(
            [
                self.MIN_GRID_POS,  # x
                self.MIN_GRID_POS,  # y
                -np.pi,  # heading
                self.MIN_SURGE_VELOCITY,
                self.MIN_SWAY_VELOCITY,
                self.MIN_YAW_RATE,
                0.0,  # dist current ckpt
                0.0,  # dist next+1
                0.0,  # dist next+2
                0.0,  # cross track
                -np.pi,  # heading error
                -1.0,  # rudder
                -1.0,  # thrust
            ],
            dtype=np.float32,
        )

        base_high = np.array(
            [
                self.MAX_GRID_POS,
                self.MAX_GRID_POS,
                np.pi,
                self.MAX_SURGE_VELOCITY,
                self.MAX_SWAY_VELOCITY,
                self.MAX_YAW_RATE,
                self.MAX_GRID_POS,
                self.MAX_GRID_POS,
                self.MAX_GRID_POS,
                self.MAX_GRID_POS,
                np.pi,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        if self.wind:
            base_low = np.hstack([base_low, np.array([-1.0, -1.0], dtype=np.float32)])
            base_high = np.hstack([base_high, np.array([1.0, 1.0], dtype=np.float32)])

        if self.current:
            base_low = np.hstack([base_low, np.array([-1.0, -1.0], dtype=np.float32)])
            base_high = np.hstack([base_high, np.array([1.0, 1.0], dtype=np.float32)])

        return gym.spaces.Box(low=base_low, high=base_high, dtype=np.float32)

    def randomize(self, randomization_scale: Optional[float] = None) -> None:
        """
        Randomize initial ship position within bounds.

        Args:
            randomization_scale: max absolute perturbation; must be positive if provided.
        """
        if randomization_scale is not None:
            if randomization_scale <= 0:
                raise ValueError("randomization_scale must be positive")
            self.randomization_scale = float(randomization_scale)

        perturbation = np.random.uniform(low=-self.randomization_scale, high=self.randomization_scale, size=self.initial_ship_pos.shape)
        self.initial_ship_pos = np.clip(self.initial_ship_pos + perturbation, self.MIN_GRID_POS, self.MAX_GRID_POS)

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state and return initial observation and info dict.
        """
        super().reset(seed=seed)

        # Hook for environment-specific reset operations (kept for compatibility)
        self.env_specific_reset()

        self.ship_pos = np.copy(self.initial_ship_pos)
        self.previous_ship_pos = np.zeros(2, dtype=np.float32)
        self.previous_heading = 0.0
        self.checkpoint_index = 1

        # Set heading toward first checkpoint
        direction_vector = self.checkpoints[self.checkpoint_index]["pos"] - self.ship_pos
        ship_angle = float(np.arctan2(direction_vector[1], direction_vector[0])) if np.linalg.norm(direction_vector) > 0 else 0.0

        self.state = np.array([self.ship_pos[0], self.ship_pos[1], ship_angle, 0.0, 0.0, 0.0], dtype=np.float32)
        self.current_action = np.zeros(2, dtype=np.float32)

        # Reset PID integrators and filters
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
        self.cross_error = 0.0
        self.desired_heading = 0.0

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        Build and return the observation vector (all values normalized / clipped).
        """
        # Normalized position to [-1, 1]
        norm_pos = self._normalize(self.ship_pos.astype(np.float32), self.MIN_GRID_POS, self.MAX_GRID_POS)

        # Normalized heading
        norm_heading = np.array([self.state[2] / np.pi], dtype=np.float32)

        # Normalized velocities
        norm_velocities = np.array(
            [
                np.clip(self.state[3] / self.MAX_SURGE_VELOCITY, -1.0, 1.0),
                np.clip(self.state[4] / (self.MAX_SWAY_VELOCITY / 2.0), -1.0, 1.0),
                np.clip(self.state[5] / self.MAX_YAW_RATE, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

        # Determine checkpoint indices safely
        checkpoint_idx = self.checkpoint_index if self.checkpoint_index < len(self.checkpoints) else len(self.checkpoints) - 1

        current_checkpoint_pos = self.checkpoints[checkpoint_idx]["pos"]
        distance_to_checkpoint = np.linalg.norm(self.ship_pos - current_checkpoint_pos)
        norm_distance = float(distance_to_checkpoint / max(1.0, self.max_dist))

        # Distances to next two checkpoints
        norm_next_distances = np.zeros(2, dtype=np.float32)
        for i in range(1, 3):
            idx = checkpoint_idx + i
            if idx < len(self.checkpoints):
                next_pos = self.checkpoints[idx]["pos"]
                norm_next_distances[i - 1] = float(np.linalg.norm(self.ship_pos - next_pos) / max(1.0, self.max_dist))

        # Cross-track error
        prev_checkpoint_pos = self.checkpoints[max(0, checkpoint_idx - 1)]["pos"]
        cross_track_error = self._distance_from_point_to_line(self.ship_pos, prev_checkpoint_pos, current_checkpoint_pos)
        norm_cross_error = float(cross_track_error / max(1.0, self.CHECKPOINTS_DISTANCE / 2.0))

        # Heading error toward current checkpoint
        direction_to_checkpoint = current_checkpoint_pos - self.ship_pos
        desired_heading = float(np.arctan2(direction_to_checkpoint[1], direction_to_checkpoint[0])) if np.linalg.norm(direction_to_checkpoint) > 0 else 0.0
        heading_error = (desired_heading - self.state[2] + np.pi) % (2.0 * np.pi) - np.pi

        obs = np.hstack(
            [
                norm_pos,
                norm_heading,
                norm_velocities,
                [norm_distance],
                norm_next_distances,
                [norm_cross_error],
                [heading_error / np.pi],
                self.current_action,
            ]
        )

        # Append environmental inputs if enabled
        if self.wind:
            obs = np.hstack([obs, self.wind_direction * self.wind_strength])
        if self.current:
            obs = np.hstack([obs, self.current_direction * self.current_strength])

        return obs.astype(np.float32)

    # -----------------------
    # Dynamics update & step
    # -----------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply the action, step dynamics, compute reward and termination.

        Returns:
            obs, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must be shape (2,), got {action.shape} with values {action}")

        # Smooth action and update dynamics
        smoothened_action = self._smoothen_action(action)
        self._update_ship_dynamics(smoothened_action)
        self.step_count += 1

        # Reward and termination check
        reward, terminated = self._calculate_reward()

        # Return observation and info (truncated False as original)
        return self._get_obs(), float(reward), bool(terminated), False, {}

    def _update_ship_dynamics(self, action: np.ndarray) -> None:
        """
        Update ship state using a simplified 3DOF model with wind/current effects.

        Args:
            action: array-like [rudder, thrust] with values in [-1, 1]
        """
        # Keep previous targets, update current action
        self.previous_rudder_target = float(self.current_action[0])
        self.previous_thrust_target = float(self.current_action[1])

        self.current_action = np.array(action, dtype=np.float32)

        # Convert control inputs to physical deltas
        delta_r = np.radians(self.current_action[0] * 60.0)  # rudder angle in radians
        t = float(self.current_action[1] * 60.0)  # scaled thrust value

        # Unpack current dynamic state
        x, y, psi, u, v, r = [float(val) for val in self.state]

        # Environmental effects relative to ship heading
        wind_effect = np.zeros(2, dtype=np.float32)
        if self.wind:
            relative_wind_angle = self.radians_wind - psi
            wind_effect = np.array([self.wind_strength * np.cos(relative_wind_angle), self.wind_strength * np.sin(relative_wind_angle)], dtype=np.float32)

        current_effect = np.zeros(2, dtype=np.float32)
        if self.current:
            relative_current_angle = self.radians_current - psi
            current_effect = np.array([self.current_strength * np.cos(relative_current_angle), self.current_strength * np.sin(relative_current_angle)], dtype=np.float32)

        # Precomputed trigonometric values
        sin_delta_r = np.sin(delta_r)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Simplified dynamics (keeps behavior identical to original)
        du = self.k_t * t + self.xu * u + wind_effect[0] + current_effect[0]
        dv = self.k_v * sin_delta_r + self.yv * v + wind_effect[1] + current_effect[1]
        dr = self.k_r * delta_r + self.nr * r + self.yv_r * v + (v * u) / max(self.l, 1e-9) - self.YAW_RATE_DAMPING * r

        # Integrate with limits
        new_u = np.clip(u + du * self.time_step, self.MIN_SURGE_VELOCITY, self.MAX_SURGE_VELOCITY)
        new_v = np.clip(v + dv * self.time_step, self.MIN_SWAY_VELOCITY, self.MAX_SWAY_VELOCITY)
        new_r = np.clip(r + dr * self.time_step, self.MIN_YAW_RATE, self.MAX_YAW_RATE)

        # Position integration in world coordinates
        dx = new_u * cos_psi - new_v * sin_psi
        dy = new_u * sin_psi + new_v * cos_psi
        dpsi = new_r

        # Save previous values and update
        self.previous_ship_pos = np.copy(self.ship_pos)
        self.previous_heading = float(self.state[2])

        new_x = np.clip(self.state[0] + dx * self.time_step, self.MIN_GRID_POS, self.MAX_GRID_POS)
        new_y = np.clip(self.state[1] + dy * self.time_step, self.MIN_GRID_POS, self.MAX_GRID_POS)
        new_heading = self.state[2] + dpsi * self.time_step

        self.state = np.array([new_x, new_y, new_heading, new_u, new_v, new_r], dtype=np.float32)
        self.ship_pos = self.state[:2]

    # -----------------------
    # Reward calculation
    # -----------------------
    def _calculate_reward(self) -> Tuple[float, bool]:
        """
        Calculate the reward and determine termination.

        Returns:
            (reward, done)
        """
        # Defensive index checking for checkpoints
        prev_checkpoint_pos = self.checkpoints[max(0, self.checkpoint_index - 1)]["pos"]
        current_checkpoint = self.checkpoints[self.checkpoint_index]
        current_checkpoint_pos = current_checkpoint["pos"]

        # Path vector and unit vector
        path_vec = current_checkpoint_pos - prev_checkpoint_pos
        path_length = float(np.linalg.norm(path_vec))
        if path_length == 0.0:
            path_unit = np.zeros_like(path_vec)
        else:
            path_unit = path_vec / path_length

        # Relative projections along the path (progress)
        rel_prev = self.previous_ship_pos - prev_checkpoint_pos
        rel_now = self.ship_pos - prev_checkpoint_pos
        proj_prev = float(np.dot(rel_prev, path_unit))
        proj_now = float(np.dot(rel_now, path_unit))

        progress_ratio = np.clip((proj_now - proj_prev) / max(path_length, 1e-9), -2.0, 2.0)
        forward_reward = self.REWARD_DISTANCE_SCALE * np.tanh(progress_ratio)

        # Velocity alignment
        velocity = self.ship_pos - self.previous_ship_pos
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > 1e-3:
            velocity_unit = velocity / velocity_norm
            path_alignment_reward = self.REWARD_DIRECTION_SCALE * float(np.dot(velocity_unit, path_unit))
        else:
            path_alignment_reward = 0.0

        # Perpendicular deviation penalty
        perp_vector = rel_now - proj_now * path_unit
        perp_dist = float(np.linalg.norm(perp_vector))
        path_deviation_penalty = -self.PENALTY_DISTANCE_SCALE * float(np.tanh(perp_dist))

        # Heading alignment towards current checkpoint
        direction_vec = current_checkpoint_pos - self.ship_pos
        self.desired_heading = float(np.arctan2(direction_vec[1], direction_vec[0])) if np.linalg.norm(direction_vec) > 0 else 0.0
        heading_error = self._calculate_heading_error(self.desired_heading, self.state[2])
        heading_alignment_reward = float(np.exp(-abs(heading_error)))

        # Cross-track penalty
        cross_track_error = self._distance_from_point_to_line(self.ship_pos, prev_checkpoint_pos, current_checkpoint_pos)
        cross_track_penalty = -0.5 * float(np.tanh(cross_track_error / self.CROSS_TRACK_ERROR_PENALTY_SCALE))
        self.cross_error = float(cross_track_error)

        # Action penalty (rudder magnitude)
        rudder_penalty = -0.2 * abs(float(self.current_action[0]))

        # Combine weighted rewards and penalties
        reward = (
            self.reward_weights["forward"] * forward_reward
            + self.reward_weights["alignment"] * path_alignment_reward
            + self.reward_weights["deviation"] * path_deviation_penalty
            + self.reward_weights["heading"] * heading_alignment_reward
            + self.reward_weights["cross_track"] * cross_track_penalty
            + self.reward_weights["rudder"] * rudder_penalty
        )

        # Stuck penalty
        movement = float(np.linalg.norm(self.ship_pos - self.previous_ship_pos))
        if movement < 0.07:
            self.stuck_steps += 1
            if self.stuck_steps > 40:
                reward -= 0.6
        else:
            self.stuck_steps = 0

        # Checkpoint handling (may increment checkpoint_index)
        reward += self._is_checkpoint_reached_or_passed(current_checkpoint)

        done = False
        heading_change = abs(self._calculate_heading_error(self.previous_heading, self.state[2]))

        # Early termination conditions (preserve original checks)
        if (
            cross_track_error > 2.0 * self.CHECKPOINTS_DISTANCE
            or not self._ship_in_open_water(self.ship_pos)
            or self.step_count >= self.max_steps
            or heading_change > np.pi / 2.0
        ):
            reward = float(self.EARLY_TERMINATION_PENALTY)
            done = True

        # Normal completion (reached or passed all checkpoints)
        if self.checkpoint_index >= len(self.checkpoints):
            done = True

        return float(reward), bool(done)

    def _is_checkpoint_reached_or_passed(self, current_checkpoint: dict) -> float:
        """
        Check if the current checkpoint is reached or passed. If so, advance the checkpoint index
        and return the checkpoint reward or shaping reward toward the final target.
        """
        checkpoint_distance = float(np.linalg.norm(self.ship_pos - current_checkpoint["pos"]))
        reward = 0.0

        # Reached checkpoint circle
        if checkpoint_distance <= float(current_checkpoint["radius"]):
            if self.checkpoint_index == len(self.checkpoints) - 1: # and self.verbose:
                print(f"Target REACHED at distance {checkpoint_distance:.2f}")
            reward = float(current_checkpoint["reward"])
            self.checkpoint_index += 1
            self.step_count = 0
            return reward

        # Passed perpendicular guidance line
        if self._is_near_perpendicular_line(current_checkpoint):
            # if self.checkpoint_index == len(self.checkpoints) - 1: # and self.verbose:
            #     print(f"Target passed at distance {checkpoint_distance:.2f}")
            self.checkpoint_index += 1
            self.step_count = 0
            return 0.0

        # Terminal shaping when near the end of the path
        if self.checkpoint_index >= max(0, len(self.checkpoints) - self.SHAPING_WINDOW):
            target_radius = float(self.checkpoints[-1]["radius"])
            target_reward = float(self.checkpoints[-1]["reward"])
            # sharper decay parameter preserved from cleaned suggestion
            decay_scale = 0.3
            decay = (checkpoint_distance - target_radius) / (max(target_radius, 1e-6) * decay_scale)
            shaping = target_reward * float(np.exp(-decay))
            reward += self.reward_weights["terminal"] * shaping

        return float(reward)

    def _is_near_perpendicular_line(self, checkpoint: dict) -> bool:
        """Return True if ship is close enough to the checkpoint's perpendicular line."""
        line_start, line_end = checkpoint["perpendicular_line"]
        return self._distance_from_point_to_line(self.ship_pos, line_start, line_end) <= 2.0

    # -----------------------
    # Rendering
    # -----------------------
    def _initialize_rendering(self) -> None:
        """Set up Matplotlib figure and axes for rendering."""
        self.initialize_plots = False

        # Create temporary Tk root to determine screen size when possible
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except tk.TclError:
            screen_width, screen_height = self.MAX_FIG_WIDTH, self.MAX_FIG_HEIGHT

        fig_width = min(self.MAX_FIG_WIDTH, screen_width)
        fig_height = min(self.MAX_FIG_HEIGHT, screen_height)
        dpi = self.DPI

        self.fig, self.ax = plt.subplots(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)

        # Try to set window geometry when backend supports it; ignore errors
        try:
            manager = plt.get_current_fig_manager()
            backend = plt.get_backend()
            if manager is not None:
                if backend in {"TkAgg"} and hasattr(manager, "window"):
                    manager.window.wm_geometry(f"+0+0")
                elif backend in {"QtAgg", "Qt5Agg"} and hasattr(manager, "window"):
                    manager.window.setGeometry(0, 0, fig_width, fig_height)
                elif backend == "GTK3Agg" and hasattr(manager, "window"):
                    manager.window.move(0, 0)
        except (RuntimeError, AttributeError, TypeError, ValueError):
            # Keep behavior identical; do not raise on rendering setup failure
            pass

        # Plot handles
        self.ship_plot, = plt.plot([], [], "bo", markersize=10, label="Ship")
        self.target_plot, = plt.plot([], [], "ro", markersize=10, label="Target")
        self.heading_line, = plt.plot([], [], color="black", linewidth=2, label="Heading")

        self.ax.set_xlim(self.MIN_GRID_POS, self.MAX_GRID_POS)
        self.ax.set_ylim(self.MIN_GRID_POS, self.MAX_GRID_POS)
        self.ax.set_title("Ship Navigation in a Path Following Environment")
        self.ax.legend()

    def _draw_fixed_landmarks(self) -> None:
        """Draw static features: path, checkpoints, obstacles, polygons."""
        if not getattr(self, "ax", None) or not self.checkpoints:
            return

        # Path: start with ship_pos followed by checkpoints and target
        path_points = [self.ship_pos] + [cp["pos"] for cp in self.checkpoints] + [self.target_pos]

        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            # label only the first segment to avoid duplicate legend entries
            self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], "g-", label="Path" if i == 0 else "")

        # Draw checkpoint patches and perpendicular lines
        for i, checkpoint in enumerate(self.checkpoints):
            circle = patches.Circle((checkpoint["pos"][0], checkpoint["pos"][1]), 10.0, color="black", alpha=0.3)
            self.ax.add_patch(circle)
            start_point, end_point = checkpoint["perpendicular_line"]
            self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], "g-", label="PerpLine" if i == 0 else "")

        # Draw polygons for environment outlines
        polygon_patch = patches.Polygon(self.obstacles, closed=True, edgecolor="r", facecolor="none", lw=2, label="Waterway")
        western_scheldt = patches.Polygon(self.overall, closed=True, edgecolor="brown", facecolor="none", lw=2, label="Western Scheldt")
        self.ax.add_patch(polygon_patch)
        self.ax.add_patch(western_scheldt)

        # Update target plot
        self.target_plot.set_data(self.target_pos[0:1], self.target_pos[1:2])

    def _on_draw(self, event) -> None:
        """Matplotlib draw event handler to update cached background for blitting."""
        if event.canvas is self.fig.canvas:
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def render(self) -> None:
        """Render environment if render_mode == 'human'."""
        if self.render_mode != "human":
            return

        # Initialize plotting objects once
        if self.initialize_plots and not hasattr(self, "ship_plot"):
            self._initialize_rendering()
            self._draw_fixed_landmarks()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig.canvas.mpl_connect("draw_event", self._on_draw)
            plt.show(block=False)

        canvas = self.fig.canvas

        # --- Helper: safe execution without aborting the simulation ---
        def _safe(callable_obj):
            try:
                callable_obj()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass

        # Restore background for blit optimization when available
        if hasattr(self, "background"):
            _safe(lambda: canvas.restore_region(self.background))

        # Update ship and heading graphics
        heading_line_length = 30.0
        self.ship_plot.set_data(self.ship_pos[0:1], self.ship_pos[1:2])
        heading_x = self.ship_pos[0] + np.cos(self.state[2]) * heading_line_length
        heading_y = self.ship_pos[1] + np.sin(self.state[2]) * heading_line_length
        self.heading_line.set_data([self.ship_pos[0], heading_x], [self.ship_pos[1], heading_y])

        # Try blitting for efficient updates; otherwise draw everything
        try:
            if hasattr(self.fig.canvas, "blit"):
                self.ax.draw_artist(self.ship_plot)
                self.ax.draw_artist(self.heading_line)
                self.fig.canvas.blit(self.ax.bbox)
            else:
                self.fig.canvas.draw()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            # Fallback to safe full redraw
            _safe(self.fig.canvas.draw)

        # Flush pending GUI events without breaking the simulation
        _safe(canvas.flush_events)

    def close(self) -> None:
        """Close rendering windows when environment is done."""
        if self.render_mode == "human" and hasattr(self, "fig"):
            try:
                plt.close(self.fig)
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass
