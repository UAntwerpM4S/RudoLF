import os
from random import randrange, randint

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gymnasium as gym
import tkinter as tk
import numpy as np
from numpy import ndarray, dtype
from sympy.codegen.cnodes import static

# First try interactive backend
try:
    matplotlib.use('TkAgg')  # Try using the interactive backend
except ImportError:
    print("Warning: 'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use('Agg')  # Use non-interactive backend as fallback

from functools import lru_cache
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from typing import Dict, List, Optional, Tuple, Union, Any
from fw.environments.tools import create_checkpoints_from_simple_path
from fw.ships.ship_factory import create_ship, ShipModel
from fw.simulators.ship_simulator import ShipSimulator
from fw.ships.vessel_state import VesselState
from fw.environments.base_env import BaseEnv
from fw.dynamics.dynamics_model import Model
from fw.dynamics.fossen import Fossen
from fw.dynamics.nomoto import Nomoto

import time
import os
from typing import Tuple, Dict
import numpy as np


# -----------------------
# Module-level constants
# -----------------------
# Physical limits / config

# Grid limits
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
PERPENDICULAR_LINE_PROXIMITY = 2.0
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


class PySimColavEnv(BaseEnv):
    """Custom Python Simulator environment for ship navigation with improved physics."""

    def __init__(
            self,
            render_mode: Optional[str] = None,
            time_step: float = 1,
            max_steps: int = 1500,
            verbose: Optional[bool] = None,
            ship_pos: Optional[np.ndarray] = None,
            target_pos: Optional[np.ndarray] = None,
            dynamics_model: Optional[Model] = Model.FOSSEN,
            ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
            wind: bool = False,
            current: bool = False,
            numerical_damping: bool = False,
            static_ship: bool = True,
            use_polar_scan: bool = True,
            enable_mirror_augmentation: bool = True
    ) -> None:
        """
        Initialize the ship navigation environment.

        Args:
            render_mode: 'human' for rendering, otherwise None.
            time_step: simulation time step (s).
            max_steps: maximum steps per episode (for truncation).
            verbose: whether to print informational logs.
            ship_pos: optional initial ship position (2-array).
            target_pos: optional target position (unused if checkpoints available).
            dynamics_model: optional dynamics model to use.
            ship_model: optional ship model to use.
            wind: enable wind effects.
            current: enable current effects.
            numerical_damping: enable numerical damping.
            static_ship: enable static ships
            use_polar_scan= enable polar distances
        """

        super().__init__(render_mode)

        # Core parameters
        self.wind = bool(wind)
        self.current = bool(current)
        self.ship_model = ship_model
        self.max_steps = int(max_steps)
        self.time_step = float(time_step)
        self.dynamics_model = dynamics_model
        self.numerical_damping = bool(numerical_damping)
        self.verbose = bool(verbose) if verbose is not None else False
        self.previous_ship_pos = None
        self.previous_heading = None
        self.performed_action = None
        self.background = None
        self.dynamics = None
        self.ship_pos = None
        self.static_ship = bool(static_ship)
        self.static_ship_population = [0, 1, 2, 3] # min() ships at least, max() at most
        self.static_ship_weights = [50, 25, 15, 10] # likeliness of ships being present
        self._static_ship_probs = np.asarray(self.static_ship_weights, dtype=np.float32) / np.sum(self.static_ship_weights) # static ship probability
        self.static_ship_lateral_max = 150  # max lateral deviation off path for static ships
        self.eval_static_ship_count = None
        self.static_ship_radius = 20 # safety radius around the obstacle
        self.static_ship_radius_min = 20.0  # minimum safety radius around obstacle
        self.static_ship_radius_max = 50.0  # maximum safety radius around obstacle
        self.static_ship_radii = []  # sampled per-ship radii for current episode
        self.static_ship_segment_margin_frac = 0.1  # avoid exact checkpoint positions (% margin at each end)
        self.own_ship_radius = 0.0 # keep point-ship model for now; set >0 for hull clearance
        self.use_polar_scan = bool(use_polar_scan) # Enable/disable polar scan representation
        self.polar_scan_cached = None # polar scan of current env step
        self._polar_scan_geometry = None  # cached union of ray-hit geometries (built once per episode)
        self.polar_n_sectors: int = 54 # Number of angular sectors around the ship with resolution = 360/n (°)
        self.polar_fov = np.radians(270) # determines how "forward facing" the polar scan is
        self.polar_max_range: float = 250.0 # max range for ray casting (m)
        self.polar_obs_max: float = 0.1  # max value in observation when distance is ~0
        self.enable_mirror_augmentation = bool(enable_mirror_augmentation) # enables mirroring of the environment during training
        self.mirror_probability = 0.5 # mirror with probability
        self._mirror_this_episode = False # mirroring current episode
        self._mirror_axis_x = None

        # Precompute sector center angles in ship frame, spanning exactly the FOV
        bin_width = float(self.polar_fov) / float(self.polar_n_sectors)
        self._polar_rel_angles = (
                -(self.polar_fov / 2.0) + (np.arange(self.polar_n_sectors, dtype=np.float32) + 0.5) * bin_width
        ).astype(np.float32)

        self._polar_rel_dirs = np.stack(
            (np.cos(self._polar_rel_angles), np.sin(self._polar_rel_angles)), axis=1
        ).astype(np.float32)


        # RNG: environment-local RNG for reproducibility when seeded via reset()
        self._rng: np.random.Generator = np.random.default_rng()

        if ship_pos is not None:
            self.initial_ship_pos = np.array(ship_pos, dtype=np.float32)
        else:
            self.initial_ship_pos = np.array([5.0, 5.0], dtype=np.float32)

        self.randomization_scale = 1.0
        self.max_dist = np.sqrt(2) * float(MAX_GRID_POS)

        # Load environment static data and build checkpoints/polygons
        self._load_environment_data(target_pos)
        # Mirror around the center x-coordinate of the navigable area
        minx, _, maxx, _ = self.polygon_shape.bounds
        self._mirror_axis_x = 0.5 * (minx + maxx)

        # Cache original episode geometry once
        self._base_initial_ship_pos = self.initial_ship_pos.copy()
        self._base_target_pos = self.target_pos.copy()
        self._base_overall = np.asarray(self.overall, dtype=np.float32).copy()
        self._base_obstacles = np.asarray(self.obstacles, dtype=np.float32).copy()
        self._base_checkpoints = [
            {
                'pos': cp['pos'].copy(),
                'radius': float(cp['radius']),
                'reward': float(cp['reward']),
                'perpendicular_line': (
                    cp['perpendicular_line'][0].copy(),
                    cp['perpendicular_line'][1].copy(),
                ),
            }
            for cp in self.checkpoints
        ]
        self._sample_static_ship_scenario()

        # Build union geometry once (bank boundary + static ships) to speed up per-step ray casting
        if self.use_polar_scan:
            self._rebuild_polar_scan_geometries()

        # State and control placeholders
        self.ship = create_ship(self.ship_model)
        self.phys_sim = self.create_simulator()

        self._initialize_simulation()
        self.observation_space = self._initialize_observation_space()


        # Gym spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Reward component weights
        self.reward_weights = {
            'forward': 1.0,
            'alignment': 1.8,
            'cross_track': 1.5,
            'rudder': 0.03,
            'polar_distance': 3.0
        }

        # Rendering flag
        self.initialize_plots = True
        self._static_ship_patches: list = []
        self._static_ship_render_key = None


    # -----------------------
    # Initialization helpers
    # -----------------------
    def _initialize_simulation(self) -> None:
        """Initialize state vectors and convenience values."""

        self.ship_pos = self.initial_ship_pos.copy()
        self.previous_ship_pos = self.initial_ship_pos.copy()
        self.previous_heading = 0.0

        self.checkpoint_index = 1
        self.cross_error = 0.0
        self.step_count = 0
        self.ukc = 0.0

        # Set heading toward first checkpoint (safeguard if checkpoint list is short)
        if len(self.checkpoints) > self.checkpoint_index:
            direction_vector = self.checkpoints[self.checkpoint_index]['pos'] - self.ship_pos
            self.ship_heading = self.normalize_angle(self._safe_heading_from_vector(direction_vector))
        else:
            self.ship_heading = 0.0

        self.phys_sim.reset(VesselState(
            x=self.initial_ship_pos[0],
            y=self.initial_ship_pos[1],
            heading=self.ship_heading,
            u=0.0, v=0.0, r=0.0))
        self.performed_action = np.zeros(2, dtype=np.float32)

    def create_simulator(self):
        """Create and return a configured ship simulator.

        This factory method instantiates a `ShipSimulator` using the current
        ship model, dynamics configuration, environmental conditions, and time
        step stored on this object.

        Returns:
            ShipSimulator: A fully initialized simulator instance configured
            with the specified initial state and the current model parameters.
        """

        if self.dynamics_model == Model.FOSSEN:
            self.dynamics = Fossen(self.ship.specifications)
        elif self.dynamics_model == Model.NOMOTO:
            self.dynamics = Nomoto(self.ship.specifications)
        else:
            raise RuntimeError(f"Unknown dynamics model {self.dynamics_model} specified.")

        return ShipSimulator(self.ship.specifications, self.dynamics, self.time_step, self.wind, self.current, self.numerical_damping)

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
        path = self._reduce_path(path, self.initial_ship_pos)

        # Create checkpoints spaced along path
        path = create_checkpoints_from_simple_path(path, CHECKPOINTS_DISTANCE)

        # Ensure start is current ship position; handle empty path safely
        if len(path) == 0:
            # empty path: fallback to a single checkpoint at initial position
            path = [self.initial_ship_pos.copy()]
        else:
            path = np.insert(path, 0, self.initial_ship_pos.copy(), axis=0)

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

        # Navigation and counters
        # Safe handling when no checkpoints are present:
        if self.checkpoints:
            self.target_pos = self.checkpoints[-1]['pos']
        else:
            # If user provided target_pos, use it; otherwise fallback to initial_ship_pos
            if target_pos is not None:
                self.target_pos = np.array(target_pos, dtype=np.float32)
            else:
                self.target_pos = self.initial_ship_pos.copy()

    # -----------------------
    # Utility & math helpers
    # -----------------------
    def _mirror_x(self, arr: np.ndarray) -> np.ndarray:
        """
        Mirror point(s) across the vertical axis x = self._mirror_axis_x.
        Works for shape (2,) and (N, 2).
        """
        out = np.array(arr, dtype=np.float32).copy()
        out[..., 0] = 2.0 * float(self._mirror_axis_x) - out[..., 0]
        return out

    def _apply_episode_geometry(self, mirrored: bool) -> None:
        """
        Restore original geometry or its mirrored counterpart for the current episode.
        """
        if mirrored:
            self.initial_ship_pos = self._mirror_x(self._base_initial_ship_pos)
            self.target_pos = self._mirror_x(self._base_target_pos)
            self.overall = self._mirror_x(self._base_overall)
            self.obstacles = self._mirror_x(self._base_obstacles)

            self.checkpoints = []
            for cp in self._base_checkpoints:
                self.checkpoints.append({
                    'pos': self._mirror_x(cp['pos']),
                    'radius': float(cp['radius']),
                    'reward': float(cp['reward']),
                    'perpendicular_line': (
                        self._mirror_x(cp['perpendicular_line'][0]),
                        self._mirror_x(cp['perpendicular_line'][1]),
                    ),
                })
        else:
            self.initial_ship_pos = self._base_initial_ship_pos.copy()
            self.target_pos = self._base_target_pos.copy()
            self.overall = self._base_overall.copy()
            self.obstacles = self._base_obstacles.copy()

            self.checkpoints = []
            for cp in self._base_checkpoints:
                self.checkpoints.append({
                    'pos': cp['pos'].copy(),
                    'radius': float(cp['radius']),
                    'reward': float(cp['reward']),
                    'perpendicular_line': (
                        cp['perpendicular_line'][0].copy(),
                        cp['perpendicular_line'][1].copy(),
                    ),
                })

        # Rebuild polygon because obstacle coordinates changed
        self.polygon_shape = Polygon(self.obstacles)

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

    def _intersect_rays_with_segments(self, origins, ray_ends, seg_starts, seg_ends):
        """
        Vectorized calculation of distances from origins to the first intersection
        of rays with a set of line segments.

        origins: (2,) array
        ray_ends: (N, 2) array (N = number of sectors)
        seg_starts: (M, 2) array (M = number of segments)
        seg_ends: (M, 2) array
        """
        # Ray vectors (N, 2)
        r = ray_ends - origins
        # Segment vectors (M, 2)
        s = seg_ends - seg_starts

        # Origins to segment starts (M, 2)
        origin_to_starts = seg_starts - origins

        # Cross product of r and s for all pairs (N, M)
        # Using broadcasting: r is (N, 1, 2) and s is (1, M, 2)
        denom = r[:, np.newaxis, 0] * s[np.newaxis, :, 1] - r[:, np.newaxis, 1] * s[np.newaxis, :, 0]

        # Avoid division by zero for parallel lines
        denom[np.abs(denom) < 1e-10] = np.nan

        # t = (q - p) x s / (r x s)
        t = (origin_to_starts[np.newaxis, :, 0] * s[np.newaxis, :, 1] -
             origin_to_starts[np.newaxis, :, 1] * s[np.newaxis, :, 0]) / denom

        # u = (q - p) x r / (r x s)
        u = (origin_to_starts[np.newaxis, :, 0] * r[:, np.newaxis, 1] -
             origin_to_starts[np.newaxis, :, 1] * r[:, np.newaxis, 0]) / denom

        # Valid intersection if 0 <= t <= 1 (on ray) and 0 <= u <= 1 (on segment)
        valid = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

        # Get distances for valid intersections, otherwise set to max_range
        dist_matrix = np.where(valid, t * self.polar_max_range, self.polar_max_range)

        # Return minimum distance for each ray across all segments
        return np.min(dist_matrix, axis=1)

    def _rebuild_polar_scan_geometries(self) -> None:
        """Prepare the 3.9k segments for vectorized filtering."""
        # Ensure obstacles is a numpy array of shape (N, 2)
        coords = np.array(self.obstacles, dtype=np.float32)

        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        # Create segment start and end points
        # This turns the loop of points into a set of individual lines
        self._boundary_starts = coords[:-1]
        self._boundary_ends = coords[1:]

        if getattr(self, "static_ship", False) and self.static_ship_pos is not None:
            centers = np.array(self.static_ship_pos[:self.static_ship_count], dtype=np.float32)
            self._static_ship_centers = centers.reshape(-1, 2)

            radii = np.array(self.static_ship_radii[:self.static_ship_count], dtype=np.float32)
            self._static_ship_radii_sq = np.square(radii + float(self.own_ship_radius))
        else:
            self._static_ship_centers = None
            self._static_ship_radii_sq = None

    def _get_polar_scan(self) -> np.ndarray:
        x0, y0 = self.ship_pos
        heading = self.phys_sim.state.heading

        # calculate ray endpoints first
        ch, sh = np.cos(heading), np.sin(heading)
        dx = ch * self._polar_rel_dirs[:, 0] - sh * self._polar_rel_dirs[:, 1]
        dy = sh * self._polar_rel_dirs[:, 0] + ch * self._polar_rel_dirs[:, 1]

        ray_ends = np.stack([x0 + self.polar_max_range * dx,
                             y0 + self.polar_max_range * dy], axis=1)

        # Filter to relevant boundaries to reduce computation time
        buffer = 50.0
        limit_sq = (self.polar_max_range + buffer) ** 2
        d0 = (self._boundary_starts[:, 0] - x0) ** 2 + (self._boundary_starts[:, 1] - y0) ** 2
        d1 = (self._boundary_ends[:, 0] - x0) ** 2 + (self._boundary_ends[:, 1] - y0) ** 2

        mask = (d0 < limit_sq) | (d1 < limit_sq)
        local_starts = self._boundary_starts[mask]
        local_ends = self._boundary_ends[mask]

        # Initialize distances with max_range
        distances = np.full(self.polar_n_sectors, self.polar_max_range, dtype=np.float32)

        # Intersect with filtered boundaries
        if len(local_starts) > 0:
            distances = self._intersect_rays_with_segments(
                self.ship_pos, ray_ends, local_starts, local_ends
            )

        # Handle Static Ships (Circles)
        if self._static_ship_centers is not None and len(self._static_ship_centers) > 0:
            # Ray directions (N, 2)
            d = ray_ends - self.ship_pos

            # Ship to centers: (M, 2)
            f = self.ship_pos - self._static_ship_centers

            # Quadratic components
            a = np.sum(d ** 2, axis=1)[:, np.newaxis]
            b = 2 * (d @ f.T)
            c = (np.sum(f ** 2, axis=1) - self._static_ship_radii_sq)[np.newaxis, :]

            discriminant = b ** 2 - 4 * a * c
            hits = (discriminant >= 0)

            # Calculate t
            t = np.where(hits, (-b - np.sqrt(np.maximum(0, discriminant))) / (2 * a), 1.0)

            # Valid t is between 0 and 1
            valid_t = np.where((t >= 0) & (t <= 1), t * self.polar_max_range, self.polar_max_range)

            circle_distances = np.min(valid_t, axis=1)

            # Take the minimum of boundary hit vs circle hit
            distances = np.minimum(distances, circle_distances)

        dist = np.clip(distances / self.polar_max_range, 0.0, 1.0).astype(np.float32)
        closeness = 1.0 - dist
        scan = closeness * float(self.polar_obs_max)

        return np.clip(scan, 0.0, float(self.polar_obs_max)).astype(np.float32)


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
                0.0,    # Distance to current checkpoint
                0.0,    # Distance to checkpoint+1
                -2.0,   # Cross-track error
                -1.0,   # Heading error
                -1.0,   # Heading error
                -1.0,   # Heading error
                -1.0,   # Heading error
                #-1.0,   # Heading error
                #-1.0,   # Heading error
                #-1.0,   # Heading error
                #-1.0,   # Heading error
                #-1.0,  # Heading error
                #-1.0,  # Heading error
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
                2.0,    # Cross-track error
                1.0,    # Heading error
                1.0,    # Heading error
                1.0,    # Heading error
                1.0,    # Heading error
                #1.0,  # Heading error
                #1.0,  # Heading error
                #1.0,  # Heading error
                #1.0,  # Heading error
                #1.0,   # Heading error
                #1.0    # Heading error
            ],
            dtype=np.float32,
        )

        if self.wind:
            base_low = np.hstack([base_low, np.array([-1.0, -1.0], dtype=np.float32)])
            base_high = np.hstack([base_high, np.array([1.0, 1.0], dtype=np.float32)])

        if self.current:
            base_low = np.hstack([base_low, np.array([-1.0, -1.0], dtype=np.float32)])
            base_high = np.hstack([base_high, np.array([1.0, 1.0], dtype=np.float32)])

        # Polar scan sectors: normalized distances in [0, 1] for each sector
        if getattr(self, "use_polar_scan", False):
            base_low = np.hstack(
                [base_low, np.zeros(self.polar_n_sectors, dtype=np.float32)]
            )
            base_high = np.hstack(
                [base_high, np.full(self.polar_n_sectors, self.polar_obs_max, dtype=np.float32)]
            )

        return gym.spaces.Box(low=base_low, high=base_high, dtype=np.float32)

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

        perturbation = self._rng.uniform(
            low=-self.randomization_scale,
            high=self.randomization_scale,
            size=self.initial_ship_pos.shape
        )
        self.initial_ship_pos = np.clip(self.initial_ship_pos + perturbation, MIN_GRID_POS, MAX_GRID_POS)

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

        # Normalize velocities
        norm_velocities = np.array([
            np.clip(self.phys_sim.state.u / self.ship.specifications.surge_limits[1], -1, 1),
            np.clip(self.phys_sim.state.v / self.ship.specifications.sway_limits[1], -1, 1),
            np.clip(self.phys_sim.state.r / self.ship.specifications.yaw_rate_limits[1], -1, 1)
        ], dtype=np.float32)

        # Clamp checkpoint indices
        current_chkp_idx = min(self.checkpoint_index, len(self.checkpoints) - 1)
        prev_chkp_idx = max(0, current_chkp_idx - 1)
        next_chkp_idx = min(current_chkp_idx + 1, len(self.checkpoints) - 1)
        next_next_chkp_idx = min(self.checkpoint_index + 2, len(self.checkpoints) - 1)
        next_next_next_chkp_idx = min(self.checkpoint_index + 3, len(self.checkpoints) - 1)


        # Checkpoint positions
        current_chkp_pos = self.checkpoints[current_chkp_idx]['pos']
        prev_chkp_pos = self.checkpoints[prev_chkp_idx]['pos']
        next_chkp_pos = self.checkpoints[next_chkp_idx]['pos']
        next_next_chkp_pos = self.checkpoints[next_next_chkp_idx]['pos']
        next_next_next_chkp_pos = self.checkpoints[next_next_next_chkp_idx]['pos']

        # Distances to checkpoints
        norm_distance = np.linalg.norm(self.ship_pos - current_chkp_pos) / self.max_dist
        norm_next_distance = np.linalg.norm(self.ship_pos - next_chkp_pos) / self.max_dist
        norm_next_next_distance = np.linalg.norm(self.ship_pos - next_next_chkp_pos) / self.max_dist
        norm_next_next_next_distance = np.linalg.norm(self.ship_pos - next_next_next_chkp_pos) / self.max_dist

        # Cross-track error (normalized, signed)
        norm_cross_error = (
                self._distance_from_point_to_line(
                    self.ship_pos,
                    prev_chkp_pos,
                    current_chkp_pos
                )
                / (CHECKPOINTS_DISTANCE / 2.0)
        )

        track_vec = current_chkp_pos - prev_chkp_pos
        ship_vec = self.ship_pos - prev_chkp_pos
        norm_cross_error *= 1.0 if np.cross(track_vec, ship_vec) >= 0.0 else -1.0

        # --- Heading errors ---

        # To current checkpoint
        heading_error = self._calculate_heading_error(
            self._safe_heading_from_vector(current_chkp_pos - self.ship_pos),
            self.phys_sim.state.heading,
            0.0
        )

        # Parallel to current checkpoint segment
        heading_error_parallel = self._calculate_heading_error(
            self._safe_heading_from_vector(current_chkp_pos - prev_chkp_pos),
            self.phys_sim.state.heading,
            0.0
        )

        # To next checkpoint
        heading_error2 = self._calculate_heading_error(
            self._safe_heading_from_vector(next_chkp_pos - self.ship_pos),
            self.phys_sim.state.heading,
            0.0
        )

        # Parallel to next checkpoint segment
        heading_error_parallel2 = self._calculate_heading_error(
            self._safe_heading_from_vector(next_chkp_pos - prev_chkp_pos),
            self.phys_sim.state.heading,
            0.0
        )

        # To next_next checkpoint
        heading_error3 = self._calculate_heading_error(
            self._safe_heading_from_vector(next_next_chkp_pos - self.ship_pos),
            self.phys_sim.state.heading,
            0.0
        )

        # Parallel to current checkpoint segment
        heading_error_parallel3 = self._calculate_heading_error(
            self._safe_heading_from_vector(next_next_next_chkp_pos - prev_chkp_pos),
            self.phys_sim.state.heading,
            0.0
        )

        # Build observation
        obs = np.hstack([
            norm_velocities,                        # Normalized velocities
            [norm_distance],                        # Distance to current checkpoint
            [norm_next_distance],                   # Distance to next checkpoint
            [norm_cross_error],                     # Normalized cross-track error
            [heading_error / np.pi],                # Normalized heading error
            [heading_error2 / np.pi],               # Normalized heading error
            [heading_error_parallel / np.pi],       # Normalized heading error
            [heading_error_parallel2 / np.pi],      # Normalized heading error
            # [norm_next_next_distance],            # Normalized distance to next checkpoint
            # [norm_next_next_next_distance]        # Normalized distance to next next next checkpoint
            # [heading_error3 / np.pi],             # Normalized heading error
            # [heading_error_parallel3 / np.pi]     # Normalized heading error
        ])

        # Add environmental observations if enabled
        if self.wind:
            obs = np.hstack([obs, self.phys_sim.environment.wind_direction * self.phys_sim.environment.wind_strength])

        if self.current:
            obs = np.hstack([obs, self.phys_sim.environment.current_direction * self.phys_sim.environment.current_strength])

        # Add polar scan (simulated range sensor) if enabled
        if getattr(self, "use_polar_scan", False):
            if self.polar_scan_cached is None:
                self.polar_scan_cached = self._get_polar_scan()
            obs = np.hstack([obs, self.polar_scan_cached])

        return obs.astype(np.float32)

    # -----------------------
    # Dynamics update & step
    # -----------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment timestep.
        """

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must be shape (2,), got {action.shape} with values {action}")

        # Save previous values
        self.previous_ship_pos = self.ship_pos.copy()
        self.previous_heading = self.phys_sim.state.heading

        # Update state vector
        self.phys_sim.step(action, self.enable_smoothing)
        self.ship_pos = np.array([self.phys_sim.state.x, self.phys_sim.state.y], dtype=np.float32)
        self.performed_action = self.phys_sim.actuators.current_actuator_state.copy()

        if self.use_polar_scan:
            self.polar_scan_cached = self._get_polar_scan()

        self.step_count += 1

        # Reward and termination check
        reward, terminated, info = self._calculate_reward()

        # Set truncated True when episode exceeded max_steps but not terminated by failure or success.
        truncated = False
        if self.step_count >= self.max_steps and not terminated:
            reward = -250
            truncated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # -------------------
    # Reward calculation
    # -------------------
    def _calculate_reward(self) -> Tuple[float, bool, Dict[str, Union[bool, str]]]:
        """
        Calculate reward and termination conditions.

        Returns:
            tuple: (reward, done) where:
                reward: Calculated reward value
                done: True if episode should terminate
        """

        # Defensive index checking for checkpoints
        prev_checkpoint_idx = max(0, self.checkpoint_index - 1)
        prev_checkpoint_pos = self.checkpoints[prev_checkpoint_idx]['pos']

        current_checkpoint = self.checkpoints[self.checkpoint_index]
        current_checkpoint_pos = current_checkpoint['pos']

        # Path vector and unit vector
        path_vec = current_checkpoint_pos - prev_checkpoint_pos
        path_length = max(np.linalg.norm(path_vec), EPSILON)
        path_unit = path_vec / path_length

        # Relative projections along the path (progress)
        rel_prev = self.previous_ship_pos - prev_checkpoint_pos
        rel_now = self.ship_pos - prev_checkpoint_pos
        proj_prev = np.dot(rel_prev, path_unit)
        proj_now = np.dot(rel_now, path_unit)

        # Forward progress
        raw_progress = (proj_now - proj_prev)
        forward_reward = 4.0 * np.clip(raw_progress, -1.0, 1.0)

        # Heading alignment towards current checkpoint
        heading_error = self._calculate_heading_error(self._safe_heading_from_vector(path_vec), self.phys_sim.state.heading)
        heading_alignment_reward = abs(np.exp(-abs(heading_error))) - 1

        # Cross-track penalty (uses distance to segment)
        cross_track_error = self._distance_from_point_to_line(self.ship_pos, prev_checkpoint_pos,
                                                              current_checkpoint_pos)
        self.cross_error = cross_track_error
        cross_track_penalty = cross_track_error / 100.0
        cross_track_penalty = -abs(cross_track_penalty) ** 1.0

        self.ukc = self.get_keel_clearance()

        # Action penalty (rudder magnitude)
        rudder_penalty = -0.2 * abs(self.performed_action[0])

        # Combine weighted rewards and penalties
        reward = (
                self.reward_weights['forward'] * forward_reward +
                self.reward_weights['alignment'] * heading_alignment_reward +
                self.reward_weights['cross_track'] * cross_track_penalty +
                self.reward_weights['rudder'] * rudder_penalty
        )

        # Polar distances (closeness-based scan in [0..polar_obs_max])
        if getattr(self, "use_polar_scan", False):
            # scan is closeness; closest object => largest value
            max_close = float(np.max(self.polar_scan_cached))  # [0..polar_obs_max]
            # convert back to normalized distance in [0..1], where 0 = collision, 1 = far
            min_dist_n = 1.0 - (max_close / max(float(self.polar_obs_max), 1e-6))

            d_ramp_start = 100 # ramp starts at X meters
            ramp_max_penalty = 5.0 # penalty at the collision boundary edges of TS and OS
            min_dist = min_dist_n * self.polar_max_range

            if min_dist < d_ramp_start:
                collision_boundary = float(self.static_ship_radius + self.own_ship_radius)
                # Normalize distance within the ramp zone: 1.0 at ramp start, 0.0 at collision boundary
                ramp_t = (min_dist - collision_boundary) / max(d_ramp_start - collision_boundary, EPSILON)
                ramp_t = float(np.clip(ramp_t, 0.0, 1.0))

                # Penalty grows exponentially as ramp_t approaches 0
                # exp_t=1.0 at ramp start (no penalty), exp_t→0 at collision boundary (max penalty)
                exp_t = (np.exp(ramp_t) - 1.0) / (np.e - 1.0)  # maps [0,1] → [0,1] with exponential curve
                ramp_penalty = ramp_max_penalty * (1.0 - exp_t)
                reward -= self.reward_weights['polar_distance'] * ramp_penalty


        # Checkpoint handling (may increment checkpoint_index)
        checkpoint_distance = np.linalg.norm(self.ship_pos - current_checkpoint["pos"])

        # Reached checkpoint circle
        if checkpoint_distance <= current_checkpoint['radius']:
            if self.checkpoint_index == len(self.checkpoints) - 1:
                print(f"Target REACHED at distance {checkpoint_distance:.2f}")
            self.checkpoint_index += 1
            self.step_count = 0

        # Passed perpendicular guidance line
        elif self._is_near_perpendicular_line(current_checkpoint):
            # if self.checkpoint_index == len(self.checkpoints) - 1:
            #     print(f"Target passed at distance {checkpoint_distance:.2f}")
            self.checkpoint_index += 1
            self.step_count = 0

        done = False
        heading_change = abs(self._calculate_heading_error(self.previous_heading, self.phys_sim.state.heading))

        bank_collision = not self._ship_in_open_water(self.ship_pos)

        # Early termination conditions
        termination_conditions = [
            cross_track_error > CROSS_TRACK_TERMINATION_MULTIPLIER * CHECKPOINTS_DISTANCE,
            bank_collision,
            heading_change > HEADING_CHANGE_THRESHOLD,
        ]

        if getattr(self, "static_ship", False) and self.static_ship_pos is not None:
            static_ship_collision = self._check_static_ship_collision()
            termination_conditions.append(static_ship_collision)

        info = {}
        if bank_collision:
            info = {"collision": True, "collision_type": "bank"}

        if getattr(self, "static_ship", False) and static_ship_collision:
            info = {"collision": True, "collision_type": "ship"}

        if any(termination_conditions):
            reward = -250.0
            done = True

        # Normal completion (reached or passed all checkpoints)
        if self.checkpoint_index >= len(self.checkpoints):
            done = True

        return reward, done, info

    def _is_near_perpendicular_line(self, checkpoint: dict) -> bool:
        """Check if ship is close enough to checkpoint's perpendicular line."""

        line_start, line_end = checkpoint['perpendicular_line']
        return self._distance_from_point_to_line(self.ship_pos, line_start, line_end) <= PERPENDICULAR_LINE_PROXIMITY

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

        # polar scan visualization (ray lines)
        self.polar_beams = []
        if getattr(self, "use_polar_scan", False):
            for _ in range(self.polar_n_sectors):
                # Semi-transparent rays so they don't clutter the plot
                beam_line, = self.ax.plot(
                    [], [],
                    color='orange',
                    alpha=0.3,
                    linewidth=1,
                    label=''  # avoid extra legend entries
                )
                self.polar_beams.append(beam_line)

    def _get_static_ship_render_key(self):
        """Small hashable signature of current static ship layout for render updates."""
        if not getattr(self, "static_ship", False) or self.static_ship_pos is None:
            return ("none", 0)

        n = min(int(self.static_ship_count), len(self.static_ship_pos))
        arr = np.asarray(self.static_ship_pos[:n], dtype=np.float32)
        # rounding avoids tiny float diffs causing redraw spam
        rounded_pos = tuple(map(tuple, np.round(arr, 2)))
        rounded_radii = tuple(np.round(np.asarray(self.static_ship_radii[:n], dtype=np.float32), 2))
        return (int(n), rounded_pos, rounded_radii)

    def _clear_static_ship_patches(self) -> None:
        """Remove previous static ship patches from the axes."""
        if not hasattr(self, "_static_ship_patches"):
            self._static_ship_patches = []
            return

        for p in self._static_ship_patches:
            try:
                p.remove()
            except Exception:
                pass
        self._static_ship_patches = []

    def _draw_static_ships(self) -> None:
        """Draw current episode's static ships and store patch handles."""
        if not hasattr(self, "ax"):
            return

        self._clear_static_ship_patches()

        if not getattr(self, "static_ship", False) or self.static_ship_pos is None:
            return

        n = min(int(self.static_ship_count), len(self.static_ship_pos))
        for i in range(n):
            sx, sy = float(self.static_ship_pos[i][0]), float(self.static_ship_pos[i][1])
            radius = float(self.static_ship_radii[i])

            safety = patches.Circle(
                (sx, sy),
                radius,
                fill=False,
                edgecolor="gray",
                linewidth=2,
                alpha=0.8,
                label="Static ship" if i == 0 else ""
            )
            self.ax.add_patch(safety)
            self._static_ship_patches.append(safety)


    def _draw_fixed_landmarks(self) -> None:
        """Draw static features: path, checkpoints, obstacles, polygons."""

        if not hasattr(self, 'ax') or not self.checkpoints:
            return

        # Path: start with ship_pos followed by checkpoints and target
        path_points = [self.ship_pos] + [cp['pos'] for cp in self.checkpoints] + [self.target_pos]

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
            self._draw_static_ships()
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

        # If static ship layout changed (e.g., new episode), redraw ships + refresh background
        new_key = self._get_static_ship_render_key()
        if new_key != getattr(self, "_static_ship_render_key", None):
            self._static_ship_render_key = new_key
            self._draw_static_ships()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        # Restore background for blit optimization when available
        if hasattr(self, 'background'):
            _safe(lambda: canvas.restore_region(self.background))

        # Update ship and heading graphics
        heading_line_length = 30.0
        self.ship_plot.set_data(self.ship_pos[0:1], self.ship_pos[1:2])
        heading_x = self.ship_pos[0] + np.cos(self.phys_sim.state.heading) * heading_line_length
        heading_y = self.ship_pos[1] + np.sin(self.phys_sim.state.heading) * heading_line_length
        self.heading_line.set_data([self.ship_pos[0], heading_x], [self.ship_pos[1], heading_y])

        # Update polar beams
        if getattr(self, "use_polar_scan", False) and hasattr(self, "_get_polar_scan"):
            scan = self._get_polar_scan()  # shape (polar_n_sectors,), values in [0,1]

            ship_x = float(self.ship_pos[0])
            ship_y = float(self.ship_pos[1])
            heading = float(self.phys_sim.state.heading)

            # For each sector, draw a line from ship position to "range" in that direction
            for i, beam in enumerate(self.polar_beams):
                rel_angle = float(self._polar_rel_angles[i])
                angle = heading + rel_angle

                # Actual length in meters for this sector
                closeness01 = float(scan[i]) / max(float(self.polar_obs_max), 1e-6)  # [0..1]
                length = (1.0 - closeness01) * float(self.polar_max_range)  # meters

                end_x = ship_x + length * np.cos(angle)
                end_y = ship_y + length * np.sin(angle)

                beam.set_data([ship_x, end_x], [ship_y, end_y])


        # Try blitting for efficient updates; otherwise draw everything
        try:
            if hasattr(self.fig.canvas, 'blit'):
                self.ax.draw_artist(self.ship_plot)
                self.ax.draw_artist(self.heading_line)

                # Draw polar beams
                if getattr(self, "use_polar_scan", False):
                    for beam in self.polar_beams:
                        self.ax.draw_artist(beam)

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

    # -----------------------
    # Static obstacle helpers
    # -----------------------
    def _sample_static_ship_scenario(self) -> None:
        """Sample number of static ships for this episode and regenerate their positions."""
        # Sample count for this episode
        self.static_ship_count = int(self._rng.choice(self.static_ship_population, p=self._static_ship_probs))
        self.static_ship_pos = self._compute_default_static_ship_pos()

    def _compute_default_static_ship_pos(self) -> list[ndarray[Any, dtype[Any]]] | None:
        """Compute a deterministic static obstacle position.

        By default we place a single static ship on the path between the 2nd and 3rd checkpoint
        (i.e., between checkpoints index 2 and 3 in `self.checkpoints`, where index 0 is the initial ship position).

        Returns:
            np.ndarray of shape (2,) in world coordinates, or None if checkpoints are unavailable.
        """
        if not hasattr(self, "checkpoints") or len(self.checkpoints) < 2:
            return None

        static_ship_positions = []
        self.static_ship_radii = []

        max_lat = float(getattr(self, "static_ship_lateral_max", 0.0))
        # Keep ships away from exact checkpoint centers (0.0 means full segment allowed)
        segment_margin_frac = float(getattr(self, "static_ship_segment_margin_frac", 0.0))  # values 0.1, 0.2, ...

        for i in range(self.static_ship_count):
            # Build index range we want to sample from (skip first/last few segments)
            eligible_indices = list(range(3, len(self.checkpoints) - 2))

            # Compute the physical length of each eligible segment
            seg_lengths = np.array([
                np.linalg.norm(
                    np.asarray(self.checkpoints[i]["pos"], dtype=np.float32) -
                    np.asarray(self.checkpoints[i - 1]["pos"], dtype=np.float32)
                )
                for i in eligible_indices
            ], dtype=np.float32)

            # let the probability of choosing a segment, be proportional to its length
            # such that curves (that have more segments) are not more likely to contain ships than straight parts
            # of the path
            seg_lengths = np.maximum(seg_lengths, EPSILON)
            probs = seg_lengths / seg_lengths.sum()

            # Sample proportional to length
            random_seg = int(eligible_indices[int(self._rng.choice(len(eligible_indices), p=probs))])
            pa = np.asarray(self.checkpoints[random_seg - 1]["pos"], dtype=np.float32)
            pb = np.asarray(self.checkpoints[random_seg]["pos"], dtype=np.float32)

            seg = (pb - pa).astype(np.float32)
            seg_len = float(np.linalg.norm(seg))

            # Fallback if degenerate segment
            # if seg_len <= EPSILON:
            #     static_ship_positions.append(((pa + pb) * 0.5).astype(np.float32))
            #     continue

            tangent = seg / seg_len
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

            # Safety circle radius used for "inside water" check and overlap clearance
            ship_radius = float(
                self._rng.uniform(self.static_ship_radius_min, self.static_ship_radius_max)
            )

            inflated_r = float(ship_radius + self.own_ship_radius)

            # Default fallback: midpoint with no lateral offset
            pos = (0.5 * (pa + pb)).astype(np.float32)

            # Try several samples (random along-segment + random lateral)
            t_min = max(0.0, segment_margin_frac)
            t_max = min(1.0, 1.0 - segment_margin_frac)
            if t_min > t_max:
                t_min, t_max = 0.0, 1.0

            for _ in range(30):
                # Random location anywhere along this checkpoint segment
                t = float(self._rng.uniform(t_min, t_max))
                base = pa + tangent * (t * seg_len)  # equivalent to pa + t * (pb-pa)

                # Normally distributed lateral offset to either side
                lat = float(np.clip(self._rng.normal(0.0, max_lat / 3.0), -max_lat, max_lat)) if max_lat > 0.0 else 0.0
                cand = base + normal * lat

                # Keep the full safety circle inside navigable water
                if not self.polygon_shape.contains(Point(float(cand[0]), float(cand[1])).buffer(inflated_r)):
                    continue

                # prevent overlap with already placed static ships
                too_close = False
                for p, prev_radius in zip(static_ship_positions, self.static_ship_radii):
                    prev_inflated_r = float(prev_radius + self.own_ship_radius)
                    if float(np.linalg.norm(cand - p)) < (inflated_r + prev_inflated_r):
                        too_close = True
                        break
                if too_close:
                    continue

                pos = cand.astype(np.float32)
                break

            static_ship_positions.append(pos)
            self.static_ship_radii.append(ship_radius)

        return static_ship_positions

    def _check_static_ship_collision(self) -> bool:
        """Return True if the own ship intersects with any static obstacle safety circle."""
        collisions = []
        for i in range(self.static_ship_count):
            d = float(np.linalg.norm(self.ship_pos - self.static_ship_pos[i]))
            r = float(self.static_ship_radii[i] + self.own_ship_radius)
            collisions.append(d <= r)
        return any(collisions)

    def env_specific_reset(self) -> None:
        """
        Called from reset() every episode.
        calculates amount and position of static ships differently for evaluation/training
        """
        if getattr(self, "use_polar_scan", False):
            self.polar_scan_cached = None # invalidate polar scan at start of each episode

        # Sample whether this episode uses mirrored geometry
        if getattr(self, "eval_static_ship_count", None) is None:
            if self.enable_mirror_augmentation:
                self._mirror_this_episode = bool(self._rng.random() < self.mirror_probability)
            else:
                self._mirror_this_episode = False
        else:
            self._mirror_this_episode = True

        # Apply original or mirrored geometry BEFORE static ship placement
        self._apply_episode_geometry(self._mirror_this_episode)

        if getattr(self, "static_ship", False):
            # Evaluation (visualization) runs
            if getattr(self, "eval_static_ship_count", None) is not None:
                self.static_ship_count = int(self.eval_static_ship_count)
                self.static_ship_pos = self._compute_default_static_ship_pos()
            # Training runs
            else:
                self._sample_static_ship_scenario()

        if getattr(self, "use_polar_scan", False):
            self._rebuild_polar_scan_geometries()