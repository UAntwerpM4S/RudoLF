import os
import math
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gymnasium as gym
import tkinter as tk
import numpy as np

# First try interactive backend
try:
    matplotlib.use('TkAgg')  # Try using the interactive backend
except ImportError:
    print("Warning: 'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use('Agg')  # Use non-interactive backend as fallback

from functools import lru_cache
from shapely.geometry import Point, Polygon
from typing import Dict, List, Optional, Tuple
from fw.environments.tools import create_checkpoints_from_simple_path
from fw.ships.ship_factory import create_ship, ShipModel
from fw.simulators.ship_simulator import ShipSimulator
from fw.ships.vessel_state import VesselState
from fw.environments.base_env import BaseEnv
from fw.dynamics.ugent_surrogate import UGentSurrogate
from fw.dynamics.dynamics_model import Model
from fw.dynamics.fossen import Fossen
from fw.dynamics.nomoto import Nomoto


# -----------------------
# Module-level constants
# -----------------------
# Physical limits / config

# Grid limits
MAX_CHECKPOINTS_DISTANCE = 2000
CHECKPOINTS_DISTANCE = 350
MIN_GRID_POS = -11700
MAX_GRID_POS = 14500

# Reward / shaping
SUCCESS_REWARD = 50.0
CHECKPOINT_AREA_SIZE = 5.0
TARGET_AREA_SIZE = 7.0

# Rendering constants
MAX_FIG_WIDTH = 1200
MAX_FIG_HEIGHT = 900
DPI = 100

# Constants definition
PERPENDICULAR_LINE_LENGTH = 50.0
HEADING_CHANGE_THRESHOLD = np.pi / 2.0
CROSS_TRACK_TERMINATION_MULTIPLIER = 2.0
EPSILON = 1e-9  # Small epsilon for numerical stability


def calculate_perpendicular_lines(
        checkpoints: List[dict],
        line_length: float = PERPENDICULAR_LINE_LENGTH,
) -> List[Tuple[np.ndarray, np.ndarray]]:
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

        # If there's only one checkpoint, fallback to an arbitrary tangent (unit x-axis)
        if len(check_points) == 1:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
        else:
            if index == 0:
                tangent = check_points[1]['pos'] - check_points[0]['pos']
            elif index == len(check_points) - 1:
                tangent = check_points[-1]['pos'] - check_points[-2]['pos']
            else:
                # Average forward and backward differences
                forward = check_points[index + 1]['pos'] - check_points[index]['pos']
                backward = check_points[index]['pos'] - check_points[index - 1]['pos']
                tangent = 0.5 * (forward + backward)

        norm = np.linalg.norm(tangent)
        # Avoid division by zero; if zero, return tangent (will be zeros) cast to float32
        return (tangent / norm if norm != 0.0 else tangent).astype(np.float32)

    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, cp in enumerate(checkpoints):
        pos = np.asarray(cp['pos'], dtype=np.float32)
        smth_tangent = smooth_tangent(checkpoints, i)
        perpendicular = np.array([-smth_tangent[1], smth_tangent[0]], dtype=np.float32)
        offset = perpendicular * (line_length / 2.0)
        lines.append((pos + offset, pos - offset))
    return lines


class PySimEnv(BaseEnv):
    """Custom Python Simulator environment for ship navigation with improved physics."""

    def __init__(
            self,
            render_mode: Optional[str] = None,
            time_step: float = 0.1,
            max_steps: int = 1500,
            verbose: Optional[bool] = None,
            start_pos: Optional[np.ndarray] = None,
            target_pos: Optional[np.ndarray] = None,
            dynamics_model: Optional[Model] = Model.FOSSEN,
            ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
            wind: bool = False,
            current: bool = False,
            numerical_damping: bool = False,
            num_ships: int = 1,
    ) -> None:
        """
        Initialize the ship navigation environment.

        Args:
            render_mode: 'human' for rendering, otherwise None.
            time_step: simulation time step (s).
            max_steps: maximum steps per episode (for truncation).
            verbose: whether to print informational logs.
            start_pos: optional initial ship position (2-array).
            target_pos: optional target position (unused if checkpoints available).
            dynamics_model: optional dynamics model to use.
            ship_model: optional ship model to use.
            wind: enable wind effects.
            current: enable current effects.
            numerical_damping: enable numerical damping.
        """

        super().__init__(render_mode)

        self.num_ships = int(num_ships)
        if self.num_ships <= 0:
            raise ValueError("num_ships must be >= 1")

        # Core parameters
        self.wind = bool(wind)
        self.current = bool(current)
        self.start_pos = start_pos
        self.ship_model = ship_model
        self.max_steps = int(max_steps)
        self.time_step = float(time_step)
        self.dynamics_model = dynamics_model
        self.numerical_damping = bool(numerical_damping)
        self.verbose = bool(verbose) if verbose is not None else False
        self.previous_ship_pos = []
        self.previous_heading = []
        self.performed_action = []
        self.background = None
        self.dynamics = None
        self.ship_pos = []

        # RNG: environment-local RNG for reproducibility when seeded via reset()
        self._rng: np.random.Generator = np.random.default_rng()

        if start_pos is not None:
            self.initial_ship_pos = [np.array(start_pos, dtype=np.float32) for _ in range(self.num_ships)]
        else:
            self.initial_ship_pos = [np.array([5.0, 5.0], dtype=np.float32) for _ in range(self.num_ships)]

        self.randomization_scale = 1.0
        self.max_dist = np.sqrt(2) * float(MAX_GRID_POS)

        # Load environment static data and build checkpoints/polygons
        self._load_environment_data(target_pos)

        # State and control placeholders
        self.ships = [create_ship(self.ship_model) for _ in range(self.num_ships)]
        self.phys_sims = [self.create_simulator(i) for i in range(self.num_ships)]

        self._initialize_simulation()

        # Gym spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._initialize_observation_space()

        # Rendering flag
        self.initialize_plots = True

    # -----------------------
    # Initialization helpers
    # -----------------------
    def _initialize_simulation(self) -> None:
        """Initialize state vectors and convenience values."""

        self.ship_pos = []
        self.ship_heading = []
        self.previous_action = []
        self.previous_ship_pos = []
        self.previous_heading = []
        self.performed_action = []

        self.checkpoint_index = []
        self.cross_error = []
        self.heading_error = []
        self.velocities = []
        self.step_count = []
        self.ukc = []

        # Set heading toward first checkpoint (safeguard if checkpoint list is short)
        for i in range(self.num_ships):
            self.ship_pos.append(self.initial_ship_pos[i].copy())
            self.previous_ship_pos.append(self.initial_ship_pos[i].copy())
            self.previous_action.append(np.zeros(2, dtype=np.float32))
            self.previous_heading.append(0.0)
            self.checkpoint_index.append(1)
            self.cross_error.append(0.0)
            self.heading_error.append(self._calculate_heading_error(self.los_guidance_heading(i=i), self.phys_sims[i].state.heading))
            self.velocities.append(self.phys_sims[i].state.u)
            self.step_count.append(0)
            self.ukc.append(0.0)

            direction_vector = self.checkpoints[self.checkpoint_index[i]]['pos'] - self.ship_pos[i]
            heading = self.normalize_angle(self._safe_heading_from_vector(direction_vector))
            self.ship_heading.append(heading)

            self.phys_sims[i].reset(VesselState(
                    terminated=False,
                    truncated=False,
                    x=self.ship_pos[i][0],
                    y=self.ship_pos[i][1],
                    heading=heading,
                    u=0.0, v=0.0, r=0.0))
            self.performed_action.append(np.zeros(2, dtype=np.float32))

    def create_simulator(self, ship_index):
        """Create and return a configured ship simulator.

        This factory method instantiates a `ShipSimulator` using the current
        ship model, dynamics configuration, environmental conditions, and time
        step stored on this object.

        Returns:
            ShipSimulator: A fully initialized simulator instance configured
            with the specified initial state and the current model parameters.
        """

        if self.dynamics_model == Model.FOSSEN:
            self.dynamics = Fossen(self.ships[ship_index].specifications)
        elif self.dynamics_model == Model.NOMOTO:
            self.dynamics = Nomoto(self.ships[ship_index].specifications)
        elif self.dynamics_model == Model.UGENT_SURROGATE:
            self.dynamics = UGentSurrogate(self.ships[ship_index].specifications)
        else:
            raise RuntimeError(f"Unknown dynamics model {self.dynamics_model} specified.")

        return ShipSimulator(self.ships[ship_index].specifications, self.dynamics, self.time_step, self.wind, self.current, self.numerical_damping)

    # -------------------------
    # Environment data loaders
    # -------------------------
    def _load_environment_data(self, target_pos: Optional[np.ndarray]) -> None:
        """
        Loads static environment data including obstacles, map outlines, and trajectory checkpoints.

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

        def load_csv_strict(name: str) -> np.ndarray:
            csv_path = os.path.join(base_path, name)
            try:
                data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing required environment file: {csv_path}")
            except Exception as exc:
                raise RuntimeError(f"Failed to load {csv_path}: {exc}")
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError(f"{csv_path} must be a 2D array with shape (N, 2), got {data.shape}")
            if data.size == 0:
                raise ValueError(f"{csv_path} is empty")
            return data

        # Load obstacles and overall map shapes
        self.overall = load_csv_strict('env_Sche_no_scale.csv')
        self.obstacles = load_csv_strict('env_Sche_250cm_no_scale.csv')
        self.polygon_shape = Polygon(self.obstacles)

        # Load trajectory
        self.path_name = 'trajectory_points_no_scale.csv'
        path = load_csv_strict(self.path_name)

        # Preprocess path (stub currently returns same path)
        path = self._reduce_path(path, self.initial_ship_pos[0])

        # Create checkpoints spaced along path
        path = create_checkpoints_from_simple_path(path, CHECKPOINTS_DISTANCE)

        # Ensure start is current ship position; handle empty path safely
        if len(path) == 0:
            # empty path: fallback to a single checkpoint at initial position
            path = [self.initial_ship_pos[0].copy()]
        else:
            path = np.insert(path, 0, self.initial_ship_pos[0].copy(), axis=0)

        # Build checkpoint dicts
        checkpoints = []
        for i, point in enumerate(path):
            cp_dict = {
                'pos': np.array(point, dtype=np.float32),
                'radius': float(CHECKPOINT_AREA_SIZE),
                'reward': (i / max(1, len(path))) * 10.0,
            }
            checkpoints.append(cp_dict)

        # Last checkpoint is the target
        if checkpoints:
            checkpoints[-1]['radius'] = float(TARGET_AREA_SIZE)
            checkpoints[-1]['reward'] = float(SUCCESS_REWARD)

        # Perpendicular lines for guidance
        lines = calculate_perpendicular_lines(checkpoints, line_length=PERPENDICULAR_LINE_LENGTH)

        # Add perpendicular lines to checkpoints
        self.checkpoints = []
        for cp, line in zip(checkpoints, lines):
            cp['perpendicular_line'] = line
            self.checkpoints.append(cp)

        # Calculate direction of last path segment
        p_last = self.checkpoints[len(self.checkpoints) - 1]['pos'].copy()
        p_before_last = self.checkpoints[len(self.checkpoints) - 2]['pos'].copy()
        self.direction_last_path = p_last - p_before_last

        # Navigation and counters
        # Safe handling when no checkpoints are present:
        if self.checkpoints:
            self.target_pos = self.checkpoints[-1]['pos']
        else:
            # If user provided target_pos, use it; otherwise fallback to initial_ship_pos
            if target_pos is not None:
                self.target_pos = np.array(target_pos, dtype=np.float32)
            else:
                self.target_pos = self.initial_ship_pos[0].copy()

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
        rng = self._rng
        samples = []
        tries = 0

        while len(samples) < nbr_samples and tries < max_tries:
            tries += 1
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)

            if self._ship_in_open_water(np.array([x, y])):
                samples.append((x, y))

        if len(samples) < nbr_samples:
            print(f"Warning: Only generated {len(samples)} points out of requested {nbr_samples} "
                  f"(polygon may be thin or max_tries reached).")

        return np.array(samples, dtype=np.float32)

    def get_keel_clearance(self) -> float:
        return 0.0

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

        x, y = ship_position[0], ship_position[1]
        min_x, min_y, max_x, max_y = self.polygon_shape.bounds
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

        contains = self.polygon_shape.contains(Point(ship_position))
        deep_enough = self._has_enough_keel_clearance()
        return bool(contains and deep_enough)

    @staticmethod
    def _reduce_path(path: np.ndarray, start_pos: np.ndarray) -> np.ndarray:
        """Reduces the path to start from the point closest to the given start position.

        This method finds the point in the path that is closest to `start_pos`, then slices
        the path so that it begins just after that point and continues to the original end.
        The `start_pos` itself is not added to the returned path.

        Args:
            path (list of np.ndarray or array-like): The original path as a list of coordinate points.
            start_pos (np.ndarray or array-like): The new starting position to align the path with.

        Returns:
            list of np.ndarray: The reduced path starting after the closest point to `start_pos`.

        Note:
            - If `start_pos` is not on the path, the method finds the closest match and starts from there.
            - The point closest to `start_pos` is **excluded** from the result.
        """
        # Find index of the point on the path closest to start_pos
        distances = [np.linalg.norm(p - start_pos) for p in path]
        closest_index = np.argmin(distances)

        # Slice the path from following point to the end
        reduced_path = path[closest_index+1:]

        return reduced_path

    @staticmethod
    @lru_cache(maxsize=1024)
    def _distance_from_point_to_line_cached(point_tuple: tuple, line_seg_start_tuple: tuple,
                                            line_seg_end_tuple: tuple) -> float:
        """
        Calculate perpendicular distance from point to line segment.

        Args:
            point_tuple: Point coordinates [x,y]
            line_seg_start_tuple: Line segment start point [x,y]
            line_seg_end_tuple: Line segment end point [x,y]

        Returns:
            float: Perpendicular distance from point to line
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
        """Calculate distance with caching wrapper."""

        return self._distance_from_point_to_line_cached(
            tuple(point.astype(np.float32)),
            tuple(line_start.astype(np.float32)),
            tuple(line_end.astype(np.float32)),
        )

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalizes an angle to the range [-π, π).

        This method maps any input angle (in radians) to its equivalent value
        within the canonical range of -π (inclusive) to π (exclusive).

        Args:
            angle: An angle in radians. Can be any finite floating-point value.

        Returns:
            The equivalent angle in radians within the range [-π, π).
        """

        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _safe_heading_from_vector(vec):
        """Return heading angle for vector, or 0.0 if vector is zero."""

        if np.linalg.norm(vec) > 0.0:
            return np.arctan2(vec[1], vec[0])
        return 0.0

    def _calculate_heading_error(self, target_heading: float, current_heading: float, dead_zone_deg: float = 5.0) -> float:
        """
        Calculate heading error with dead zone handling.

        Args:
            target_heading: Desired heading in radians
            current_heading: Current heading in radians
            dead_zone_deg: Angular dead zone in degrees

        Returns:
            float: Heading error in radians, 0 if within dead zone
        """

        error = self.normalize_angle(target_heading - current_heading)
        dead_zone = np.radians(dead_zone_deg)
        return 0.0 if abs(error) < dead_zone else error

    # -----------------------
    # Observation & reset
    # -----------------------
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """
        Initialize and return the observation space for the environment.

        Observation space includes:
        - Normalized ship velocities
        - Distances to current checkpoint
        - Cross-track and heading errors
        - Optional wind/current parameters if enabled

        Returns:
            gym.spaces.Box: The observation space definition
        """

        base_low = np.array(
            [
                -1.0,   # Surge velocity
                -1.0,   # Sway velocity
                -1.0,   # Yaw rate
                -1.0,   # Distance to current checkpoint
                -1.0,   # Distance to checkpoint+1
                -1.0,   # Cross-track error
                -1.0,   # Heading error
                -1.0,   # Heading error
                -1.0,   # Heading error
                -1.0,   # Heading error
                -1.0,   # Previous rudder angle
                -1.0,   # Optimal LOS guidance
            ],
            dtype=np.float32,
        )

        base_high = np.array(
            [
                1.0,    # Surge velocity
                1.0,    # Sway velocity
                1.0,    # Yaw rate
                1.0,    # Distance to current checkpoint
                1.0,    # Distance to checkpoint+1
                1.0,    # Cross-track error
                1.0,    # Heading error
                1.0,    # Heading error
                1.0,    # Heading error
                1.0,    # Heading error
                1.0,    # Previous rudder angle
                1.0,    # Optimal LOS guidance
            ],
            dtype=np.float32,
        )

        if self.wind:
            base_low = np.hstack([base_low, np.array([-1.0, -1.0], dtype=np.float32)])
            base_high = np.hstack([base_high, np.array([1.0, 1.0], dtype=np.float32)])

        if self.current:
            base_low = np.hstack([base_low, np.array([-1.0, -1.0], dtype=np.float32)])
            base_high = np.hstack([base_high, np.array([1.0, 1.0], dtype=np.float32)])

        single_obs_space = gym.spaces.Box(low=base_low, high=base_high, dtype=np.float32)

        return gym.spaces.Box(
            low=np.tile(single_obs_space.low, (self.num_ships, 1)),
            high=np.tile(single_obs_space.high, (self.num_ships, 1)),
            dtype=np.float32,
        )

    def randomize(self, randomization_scale: Optional[float] = None) -> None:
        """
        Randomize the ship's initial position within specified bounds.

        Args:
            randomization_scale: Maximum absolute value for position
                randomization. If None, uses the class's default scale.

        Raises:
            ValueError: If randomization_scale is not positive
        """

        if randomization_scale is not None:
            if randomization_scale <= 0:
                raise ValueError("randomization_scale must be positive")

            self.randomization_scale = randomization_scale

        ran_gen = np.random.default_rng()

        perturbation = ran_gen.uniform(
            low=-self.randomization_scale,
            high=self.randomization_scale,
            size=self.initial_ship_pos[0].shape
        )

        if self.start_pos is not None:
            self.initial_ship_pos = [np.clip(self.start_pos + perturbation, MIN_GRID_POS, MAX_GRID_POS) for _ in range(self.num_ships)]
        else:
            self.initial_ship_pos = [np.clip(np.array([5.0, 5.0], dtype=np.float32) + perturbation, MIN_GRID_POS, MAX_GRID_POS) for _ in range(self.num_ships)]


    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to its initial state.

        Args:
            seed: Optional seed for random number generation
            kwargs: Additional arguments

        Returns:
            tuple: (observation, info) where:
                observation: Initial observation
                info: Additional information dictionary
        """

        # Set RNG seed if provided for reproducibility
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        super().reset(seed=seed)

        # Hook for environment-specific reset operations
        self.env_specific_reset()
        self._initialize_simulation()

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        Construct and return the normalized observation vector.

        Returns:
            np.ndarray: Normalized observation array containing:
            - Velocities (normalized to [-1,1] relative to max)
            - Distance to current checkpoint (normalized)
            - Cross-track and heading errors (normalized)
            - Optional wind/current observations
        """

        observations = []

        for i in range(self.num_ships):
            # Normalize velocities
            norm_velocities = np.array([
                np.clip(self.phys_sims[i].state.u / self.ships[i].specifications.surge_limits[1], -1, 1),
                np.clip(self.phys_sims[i].state.v / self.ships[i].specifications.sway_limits[1], -1, 1),
                np.clip(self.phys_sims[i].state.r / self.ships[i].specifications.yaw_rate_limits[1], -1, 1)
            ], dtype=np.float32)

            # Clamp checkpoint indices
            current_chkp_idx = self.checkpoint_index[i]
            prev_chkp_idx = max(0, current_chkp_idx - 1)
            next_chkp_idx = current_chkp_idx + 1

            # Checkpoint positions
            prev_chkp_pos = self.checkpoints[prev_chkp_idx]['pos']
            if current_chkp_idx <= len(self.checkpoints) - 1:
                current_chkp_pos = self.checkpoints[current_chkp_idx]['pos']
            else:
                multiplier = current_chkp_idx - (len(self.checkpoints) - 1)
                current_chkp_pos = self.checkpoints[len(self.checkpoints) - 1]['pos'] + 2 * multiplier * self.direction_last_path

            if next_chkp_idx <= len(self.checkpoints) - 1:
                next_chkp_pos = self.checkpoints[next_chkp_idx]['pos']
            else:
                multiplier = next_chkp_idx - (len(self.checkpoints) - 1)
                next_chkp_pos = self.checkpoints[len(self.checkpoints) - 1]['pos'] + 2 * multiplier * self.direction_last_path

            # Distances to checkpoints
            norm_distance = self._logarithmic_scaling(
                value=float(np.linalg.norm(self.ship_pos[i] - current_chkp_pos)),
                value_range=(0, MAX_CHECKPOINTS_DISTANCE),
                half_value=0.5 * CHECKPOINTS_DISTANCE)

            norm_next_distance = self._logarithmic_scaling(
                value=float(np.linalg.norm(self.ship_pos[i] - next_chkp_pos)),
                value_range=(0, 2 * MAX_CHECKPOINTS_DISTANCE),
                half_value=1.5 * CHECKPOINTS_DISTANCE)

            # Cross-track error (normalized, signed)
            norm_cross_error = self._logarithmic_scaling(
                value=self._distance_from_point_to_line(self.ship_pos[i], prev_chkp_pos, current_chkp_pos),
                value_range=(0, MAX_CHECKPOINTS_DISTANCE),
                half_value=50.0)

            track_vec = current_chkp_pos - prev_chkp_pos
            ship_vec = self.ship_pos[i] - prev_chkp_pos
            norm_cross_error *= 1.0 if np.cross(track_vec, ship_vec) >= 0.0 else -1.0

            # --- Heading errors ---

            # To current checkpoint
            heading_error = self._calculate_heading_error(
                self._safe_heading_from_vector(current_chkp_pos - self.ship_pos[i]),
                self.phys_sims[i].state.heading,
                0.0
            )

            # Parallel to current checkpoint segment
            heading_error_parallel = self._calculate_heading_error(
                self._safe_heading_from_vector(current_chkp_pos - prev_chkp_pos),
                self.phys_sims[i].state.heading,
                0.0
            )

            # To next checkpoint
            heading_error2 = self._calculate_heading_error(
                self._safe_heading_from_vector(next_chkp_pos - self.ship_pos[i]),
                self.phys_sims[i].state.heading,
                0.0
            )

            # Parallel to next checkpoint segment
            heading_error_parallel2 = self._calculate_heading_error(
                self._safe_heading_from_vector(next_chkp_pos - prev_chkp_pos),
                self.phys_sims[i].state.heading,
                0.0
            )

            # LOS guidance error
            los_heading_error = self._calculate_heading_error(
                self.los_guidance_heading(i=i),
                self.phys_sims[i].state.heading,
                0.0
            )

            # Build observation
            obs = np.hstack([
                norm_velocities,                        # Normalized velocities
                [2 * norm_distance - 1],                # Distance to current checkpoint
                [2 * norm_next_distance - 1],           # Distance to next checkpoint
                [norm_cross_error],                     # Normalized cross-track error
                [heading_error / np.pi],                # Normalized heading error
                [heading_error2 / np.pi],               # Normalized heading error
                [heading_error_parallel / np.pi],       # Normalized heading error
                [heading_error_parallel2 / np.pi],      # Normalized heading error
                self.performed_action[i][0],            # Previous rudder angle
                los_heading_error / np.pi,              # Optimal LOS guidance
            ])

            # Add environmental observations if enabled
            if self.wind:
                obs = np.hstack([obs, self.phys_sims[i].environment.wind_direction * self.phys_sims[i].environment.wind_strength])

            if self.current:
                obs = np.hstack([obs, self.phys_sims[i].environment.current_direction * self.phys_sims[i].environment.current_strength])

            observations.append(obs)

        return np.array(observations, dtype=np.float32)

    # -----------------------
    # Dynamics update & step
    # -----------------------
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment timestep for all ships.

        Args:
            actions: Array-like with shape (num_ships, 2) [rudder, thrust] commands in [-1, 1]

        Returns:
            tuple: (observations, reward, terminated, truncated, info)
        """

        actions = np.asarray(actions, dtype=np.float32)

        if actions.shape == (2,):
            actions = actions[None, :]  # single-ship fallback

        # Preallocate arrays
        rewards = np.zeros(self.num_ships, dtype=np.float32)
        terminations = np.zeros(self.num_ships, dtype=bool)
        truncations = np.zeros(self.num_ships, dtype=bool)

        for i in range(self.num_ships):
            if not self.phys_sims[i].state.terminated and not self.phys_sims[i].state.truncated:
                action = actions[i]
                self.previous_action[i] = self.performed_action[i].copy()

                # Save previous values
                self.previous_ship_pos[i] = self.ship_pos[i].copy()
                self.previous_heading[i] = self.phys_sims[i].state.heading

                # Update state vector
                self.phys_sims[i].step(action, self.enable_smoothing)
                self.ship_pos[i] = np.array([self.phys_sims[i].state.x, self.phys_sims[i].state.y], dtype=np.float32)
                self.performed_action[i] = self.phys_sims[i].actuators.current_actuator_state.copy()
                self.step_count[i] += 1

                # Reward and termination check
                reward, terminated = self._calculate_reward_single(i)

                # Set truncated True when episode exceeded max_steps but not terminated by failure or success.
                truncated = False
                if self.step_count[i] >= self.max_steps and not terminated:
                    reward = -250
                    truncated = True

                # Update ship state
                self.phys_sims[i].state.terminated = terminated
                self.phys_sims[i].state.truncated = truncated

                # Store outputs
                rewards[i] = reward
                terminations[i] = terminated
                truncations[i] = truncated
            else:
                # Ship already terminated/truncated, output zero reward and keep flags
                rewards[i] = 0.0
                terminations[i] = self.phys_sims[i].state.terminated
                truncations[i] = self.phys_sims[i].state.truncated

        info = {
            "rewards": rewards,
            "terminations": terminations,
            "truncations": truncations,
        }

        return self._get_obs(), rewards[0], np.all(terminations | truncations), False, info

    # -------------------
    # Reward calculation
    # -------------------
    def _calculate_reward_single(self, i: int) -> Tuple[float, bool]:
        """
        Calculate reward and termination conditions.

        Returns:
            tuple: (reward, done) where:
                reward: Calculated reward value
                done: True if episode should terminate
        """

        # Defensive index checking for checkpoints
        prev_checkpoint_idx = max(0, self.checkpoint_index[i] - 1)
        prev_checkpoint_pos = self.checkpoints[prev_checkpoint_idx]['pos']

        current_checkpoint = self.checkpoints[self.checkpoint_index[i]]
        current_checkpoint_pos = current_checkpoint['pos']

        # Cross-track errors
        cross_track_error = self._distance_from_point_to_line(self.ship_pos[i], prev_checkpoint_pos, current_checkpoint_pos)
        self.cross_error[i] = cross_track_error
        self.heading_error[i] = self._calculate_heading_error(self.los_guidance_heading(i=i), self.phys_sims[i].state.heading)
        self.velocities[i] = self.phys_sims[i].state.u

        # Keel clearance
        self.ukc[i] = self.get_keel_clearance()

        # Ship information
        rudder_change = abs(self.performed_action[i][0] - self.previous_action[i][0])
        max_rudder_change = self.ships[i].specifications.max_rudder_rate * self.time_step / self.ships[i].specifications.max_rudder_angle  # rad/s * s * action/rad
        heading_error = self._calculate_heading_error(self.los_guidance_heading(i=i), self.phys_sims[i].state.heading)
        distance_traveled = abs(np.linalg.norm(self.ship_pos[i] - self.previous_ship_pos[i]))
        velocity = distance_traveled / self.time_step
        target_velocity = 4

        # Reward components
        r_forward = - 0.25 * abs(target_velocity - velocity) / target_velocity
        r_heading_alignment = - 0.5 * abs(heading_error)
        r_cte = - 0.05 * cross_track_error
        r_rudder_angle_change = - 0.4 * (rudder_change / max_rudder_change) ** 1 if rudder_change < max_rudder_change else -1
        r_rudder_angle = - 0.05 * (abs(self.performed_action[i][0]) / 0.2) ** (2 if abs(self.performed_action[i][0]) < 0.2 else 4)
        r_thrust_change = 0  # - 0.5 * thrust_change

        # Combine weighted rewards and penalties
        reward = (
                r_forward +
                r_heading_alignment +
                r_cte +
                r_rudder_angle_change +
                r_rudder_angle +
                r_thrust_change
        )
        reward = 1.0 + reward / 5.0     # reward = 0.9 + reward

        # Checkpoint handling (may increment checkpoint_index)
        checkpoint_distance = np.linalg.norm(self.ship_pos[i] - current_checkpoint["pos"])

        # Reached checkpoint circle
        if self.segment_point_distance(self.previous_ship_pos[i], self.ship_pos[i], current_checkpoint["pos"]) <= current_checkpoint['radius']:
            if self.checkpoint_index[i] == len(self.checkpoints) - 1:
                print(f"Target REACHED at distance {checkpoint_distance:.2f}")
            self.checkpoint_index[i] += 1
            self.step_count[i] = 0

        # Passed perpendicular guidance line
        elif self._is_near_perpendicular_line(current_checkpoint, i):
            # if self.checkpoint_index[i] == len(self.checkpoints) - 1:
            #     print(f"Target passed at distance {checkpoint_distance:.2f}")
            self.checkpoint_index[i] += 1
            self.step_count[i] = 0

        done = False
        heading_change = abs(self._calculate_heading_error(self.previous_heading[i], self.phys_sims[i].state.heading))

        # Early termination conditions
        termination_conditions = [
            cross_track_error > CROSS_TRACK_TERMINATION_MULTIPLIER * CHECKPOINTS_DISTANCE,
            not self._ship_in_open_water(self.ship_pos[i]),
            heading_change > HEADING_CHANGE_THRESHOLD,
        ]

        if any(termination_conditions):
            #reward = -250
            done = True

        # Normal completion (reached or passed all checkpoints)
        if self.checkpoint_index[i] >= len(self.checkpoints):
            done = True

        return reward, done

    @staticmethod
    def segment_point_distance(p1, p2, c):
        # project c onto segment p1-p2
        if np.allclose(p1, p2):
            return np.linalg.norm(c - p1)

        v = p2 - p1
        w = c - p1

        t = np.dot(w, v) / np.dot(v, v)
        t = np.clip(t, 0.0, 1.0)

        closest = p1 + t * v
        return np.linalg.norm(c - closest)

    @staticmethod
    def _orientation(a, b, c):
        """
        Signed area (2D cross product):
        > 0 : c is left of line ab
        < 0 : c is right of line ab
        = 0 : co-linear
        """
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    @lru_cache(maxsize=1024)
    def _segments_intersect_cached(self, prev_point_tuple: tuple, curr_point_tuple: tuple, line_seg_start_tuple: tuple,
                                   line_seg_end_tuple: tuple) -> float:
        """
        Returns True if segment p1-p2 intersects the infinite line through q1-q2.
        """

        o1 = self._orientation(line_seg_start_tuple, line_seg_end_tuple, prev_point_tuple)
        o2 = self._orientation(line_seg_start_tuple, line_seg_end_tuple, curr_point_tuple)

        # If either point is on the line → intersection
        if abs(o1) <= EPSILON or abs(o2) <= EPSILON:
            return True

        # If signs differ → segment crosses the line
        return (o1 > 0 > o2) or (o1 < 0 < o2)

    def _segments_intersect(self, prev_point: np.ndarray, curr_point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate distance with caching wrapper."""

        return self._segments_intersect_cached(
            tuple(prev_point.astype(np.float32)),
            tuple(curr_point.astype(np.float32)),
            tuple(line_start.astype(np.float32)),
            tuple(line_end.astype(np.float32)),
        )

    def _is_near_perpendicular_line(self, checkpoint: dict, ship_index) -> bool:
        """Check if ship is close enough to checkpoint's perpendicular line."""

        line_start, line_end = checkpoint['perpendicular_line']
        return self._segments_intersect(self.previous_ship_pos[ship_index], self.ship_pos[ship_index], line_start, line_end)

    def los_guidance_heading(self, i: int):
        """
        Compute desired LOS heading toward the current path segment.
        These formulas were adopted from Changyuan Chen's PhD thesis
        "Numerical and Experimental Study on Ship Motion Control Systems in Shallow Water" at Flanders Hydraulics (FH)
        """

        current_checkpoint_idx = self.checkpoint_index[i]
        current_checkpoint_idx = min(current_checkpoint_idx, len(self.checkpoints) - 1)
        pos = self.ship_pos[i]
        p_k = self.checkpoints[current_checkpoint_idx]['pos'].copy()
        if current_checkpoint_idx + 1 > len(self.checkpoints) - 1:
            p_k_plus_1 = self.checkpoints[len(self.checkpoints) - 1]['pos'].copy()
            direction = p_k_plus_1 - self.checkpoints[len(self.checkpoints) - 2]['pos']
            p_k_plus_1 += direction
        else:
            p_k_plus_1 = self.checkpoints[current_checkpoint_idx + 1]['pos'].copy()
        x, y = pos[:]
        x_k, y_k = p_k[:]
        x_k_plus_1, y_k_plus_1 = p_k_plus_1[:]

        # previous checkpoint
        if current_checkpoint_idx > 0:
            p_k_min_1 = self.checkpoints[current_checkpoint_idx - 1]['pos'].copy()
            x_k_min_1, y_k_min_1 = p_k_min_1[:]
        else:
            # if none exists, create a virtual one behind the ship
            p_k_min_1 = p_k - (p_k_plus_1 - p_k)
            x_k_min_1, y_k_min_1 = p_k_min_1[:]

        # Find the orthogonal projection A(x_A, y_A) of the ship on the path from the previous to the next checkpoint
        x_a = x_k_min_1 + (((x - x_k_min_1) * (x_k - x_k_min_1) + (y - y_k_min_1) * (y_k - y_k_min_1)) / (
                    (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2)) * (x_k - x_k_min_1)
        y_a = y_k_min_1 + (((x - x_k_min_1) * (x_k - x_k_min_1) + (y - y_k_min_1) * (y_k - y_k_min_1)) / (
                    (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2)) * (y_k - y_k_min_1)

        # Calculate the look ahead distance L_A
        c = 0.5  # Coefficient for LOS look ahead horizon (default is 0.5 according to FH PhD thesis p. 148)
        delta_y = ((x - x_a) ** 2 + (y - y_a) ** 2) ** 0.5  # Cross-track error
        l_pp = 0.95 * self.ships[
            i].specifications.length  # Length between perpendiculars (a little bit smaller than the ship length)
        l_a = ((delta_y + c * l_pp) ** 2 - delta_y ** 2) ** 0.5

        # Calculate the LOS position Ploss(x_loss, y_loss)
        x_loss = x_a + ((x_k - x_k_min_1) / ((x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2) ** 0.5) * l_a
        y_loss = y_a + ((y_k - y_k_min_1) / ((x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2) ** 0.5) * l_a

        # 4.3.3 Selection of current waypoint
        # Calculate radius of the current checkpoint's acceptance circle Rk
        w_los = 0.5  # Tunable parameter for acceptance circle (default is 0.5 according to FH PhD thesis p. 148)
        r_k = w_los * l_pp

        heading_p_k_min_1_p_k = self._safe_heading_from_vector(p_k - p_k_min_1)
        heading_p_k_p_k_plus_1 = self._safe_heading_from_vector(p_k_plus_1 - p_k)
        heading_difference = self.normalize_angle(heading_p_k_min_1_p_k - heading_p_k_p_k_plus_1)
        collinear = abs(heading_difference) < (1 / 57)  # Are the previous, current, and next checkpoint collinear
        if collinear:
            dist_a_pk = ((x_a - x_k) ** 2 + (y_a - y_k) ** 2) ** 0.5  # Distance between and the current checkpoint
            if dist_a_pk > r_k:  # |A Pk| > Rk
                # LOS angle (desired course angle) in radians
                los_angle = math.atan2(y_loss - y, x_loss - x)
            else:  # |A Pk| < Rk
                # LOS position Ploss is replaced by current waypoint Pk+1
                los_angle = math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)
        else:
            dist_p_pk = ((x - x_k) ** 2 + (y - y_k) ** 2) ** 0.5  # Distance between and the current checkpoint
            if dist_p_pk > 2 * r_k:  # |P Pk| > 2*Rk
                if l_a < dist_p_pk:  # LA < |P Pk|
                    los_angle = math.atan2(y_loss - y, x_loss - x)
                else:  # LA > |P Pk|
                    # Change look ahead distance from LA to L'A
                    l_accent_a = l_a - ((x_k - x_a) ** 2 + (y_k - y_a) ** 2) ** 0.5

                    # Calculate new heading point P'loss
                    x_accent_loss = x_k + ((x_k_plus_1 - x_k) / (
                                (x_k_plus_1 - x_k) ** 2 + (y_k_plus_1 - y_k) ** 2) ** 0.5) * l_accent_a
                    y_accent_loss = y_k + ((y_k_plus_1 - y_k) / (
                                (x_k_plus_1 - x_k) ** 2 + (y_k_plus_1 - y_k) ** 2) ** 0.5) * l_accent_a

                    # LOS angle (desired course angle) in radians
                    los_angle = math.atan2(y_accent_loss - y, x_accent_loss - x)
            else:  # |P Pk| < 2*Rk
                # The checkpoint Pk becomes replaced by Pk+1 (next checkpoint becomes current checkpoint), same for Pk-1
                x_k_min_1, y_k_min_1 = x_k, y_k
                x_k, y_k = x_k_plus_1, y_k_plus_1

                # Find the orthogonal projection A(x_A, y_A) of the ship on the path from the previous to the next checkpoint
                x_a = x_k_min_1 + (((x - x_k_min_1) * (x_k - x_k_min_1) + (y - y_k_min_1) * (y_k - y_k_min_1)) / (
                            (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2)) * (x_k - x_k_min_1)
                y_a = y_k_min_1 + (((x - x_k_min_1) * (x_k - x_k_min_1) + (y - y_k_min_1) * (y_k - y_k_min_1)) / (
                            (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2)) * (y_k - y_k_min_1)

                # Calculate the look ahead distance L_A
                c = 0.5  # Coefficient for LOS look ahead horizon (default is 0.5 according to FH PhD thesis p. 148)
                delta_y = ((x - x_a) ** 2 + (y - y_a) ** 2) ** 0.5  # Cross-track error
                l_pp = 0.95 * self.ships[
                    i].specifications.length  # Length between perpendiculars (a little bit smaller than the ship length)
                l_a = ((delta_y + c * l_pp) ** 2 - delta_y ** 2) ** 0.5

                # Calculate the LOS position Ploss(x_loss, y_loss)
                x_loss = x_a + ((x_k - x_k_min_1) / ((x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2) ** 0.5) * l_a
                y_loss = y_a + ((y_k - y_k_min_1) / ((x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2) ** 0.5) * l_a

                los_angle = math.atan2(y_loss - y, x_loss - x)

        return los_angle

    @staticmethod
    def _logarithmic_scaling(value: float, value_range: tuple, half_value: float = 10):
        # Unpack the range, they correspondent with values that return 0 and 1
        min_value, max_value = value_range

        if value < min_value or value > max_value:
            raise ValueError(f"value {value} exceeded the range [{min_value}, {max_value}]")

        # Clamp to range

        # Linear normalization
        x = (value - min_value) / (max_value - min_value)

        m = (half_value - min_value) / (max_value - min_value)

        # compute curvature parameter
        k = (1 - 2 * m) / (m ** 2)

        # Logarithmic scaling
        return math.log1p(k * x) / math.log1p(k)

    # -----------------------
    # Rendering
    # -----------------------
    def _initialize_rendering(self) -> None:
        """Set up the rendering elements for visualization."""

        self.initialize_plots = False

        # Create temporary Tk root to determine screen size when possible
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except tk.TclError:
            screen_width, screen_height = MAX_FIG_WIDTH, MAX_FIG_HEIGHT

        fig_width = min(MAX_FIG_WIDTH, screen_width)
        fig_height = min(MAX_FIG_HEIGHT, screen_height)
        dpi = DPI

        self.fig, self.ax = plt.subplots(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)

        # Try to set window geometry when backend supports it; ignore errors
        try:
            manager = plt.get_current_fig_manager()
            backend = plt.get_backend()
            if manager is not None:
                if backend in {'TkAgg'} and hasattr(manager, 'window'):
                    manager.window.wm_geometry(f"+0+0")
                elif backend in {'QtAgg', 'Qt5Agg'} and hasattr(manager, 'window'):
                    manager.window.setGeometry(0, 0, fig_width, fig_height)
                elif backend == 'GTK3Agg' and hasattr(manager, 'window'):
                    manager.window.move(0, 0)
        except (RuntimeError, AttributeError):
            # Do not raise on rendering setup failure
            pass

        # Plot handles
        self.ship_plot, = self.ax.plot([], [], 'bo', markersize=10, label='Ship')
        self.target_plot, = self.ax.plot([], [], 'ro', markersize=10, label='Target')
        self.heading_line, = self.ax.plot([], [], color='black', linewidth=2, label='Heading')

        self.ax.set_xlim(MIN_GRID_POS, MAX_GRID_POS)
        self.ax.set_ylim(MIN_GRID_POS, MAX_GRID_POS)
        self.ax.set_title("Ship Navigation in a Path Following Environment")
        self.ax.legend()

    def _draw_fixed_landmarks(self) -> None:
        """Draw static features: path, checkpoints, obstacles, polygons."""

        if not hasattr(self, 'ax') or not self.checkpoints:
            return

        # Path: start with ship_pos followed by checkpoints and target
        path_points = [self.start_pos] + [cp['pos'] for cp in self.checkpoints] + [self.target_pos]

        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            # label only the first segment to avoid duplicate legend entries
            self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                         'g-', label='Path' if i == 0 else "")

        # Draw checkpoint patches and perpendicular lines
        for i, checkpoint in enumerate(self.checkpoints):
            circle = patches.Circle((checkpoint['pos'][0], checkpoint['pos'][1]),
                                    10.0, color='black', alpha=0.3)
            self.ax.add_patch(circle)
            start_point, end_point = checkpoint['perpendicular_line']
            self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                         'g-', label='PerpLine' if i == 0 else "")

        # Draw polygons for environment outlines
        polygon_patch = patches.Polygon(self.obstacles, closed=True, edgecolor='r',
                                        facecolor='none', lw=2, label='Waterway')
        western_scheldt = patches.Polygon(self.overall, closed=True, edgecolor='brown',
                                          facecolor='none', lw=2, label='Western Scheldt')
        self.ax.add_patch(polygon_patch)
        self.ax.add_patch(western_scheldt)

        # Update target plot
        self.target_plot.set_data(self.target_pos[0:1], self.target_pos[1:2])

    def _on_draw(self, event) -> None:
        """
        Updates the cached background after a canvas redraw.

        This ensures blitting stays in sync after zoom or pan interactions.

        Args:
            event (matplotlib.backend_bases.DrawEvent): The matplotlib draw event.
        """

        if event.canvas is self.fig.canvas:
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def render(self) -> None:
        """Render the environment and visualize the ship's movement."""

        if self.render_mode != 'human':
            return

        # Initialize plotting objects once
        if self.initialize_plots and not hasattr(self, 'ship_plot'):
            self._initialize_rendering()
            self._draw_fixed_landmarks()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig.canvas.mpl_connect('draw_event', self._on_draw)
            plt.show(block=False)

        canvas = self.fig.canvas

        # --- Helper: safe execution without aborting the simulation ---
        def _safe(callable_obj):
            try:
                callable_obj()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass

        # Restore background for blit optimization when available
        if hasattr(self, 'background'):
            _safe(lambda: canvas.restore_region(self.background))

        # Update ship and heading graphics
        for i in range(self.num_ships):
            heading_line_length = 30.0
            self.ship_plot.set_data(self.ship_pos[i][0:1], self.ship_pos[i][1:2])
            heading_x = self.ship_pos[i][0] + np.cos(self.phys_sims[i].state.heading) * heading_line_length
            heading_y = self.ship_pos[i][1] + np.sin(self.phys_sims[i].state.heading) * heading_line_length
            self.heading_line.set_data([self.ship_pos[i][0], heading_x], [self.ship_pos[i][1], heading_y])

        # Try blitting for efficient updates; otherwise draw everything
        try:
            if hasattr(self.fig.canvas, 'blit'):
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
        """Close the environment."""

        if self.render_mode == 'human' and hasattr(self, 'fig'):
            try:
                plt.close(self.fig)
            except (RuntimeError, AttributeError):
                pass