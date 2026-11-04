import os
import math
import logging
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

matplotlib.rcParams['figure.raise_window'] = False

from shapely.geometry import Point, Polygon
from typing import Dict, List, Optional, Tuple
from fw.environments.tools import create_checkpoints_from_simple_path
from fw.ships.ship_factory import create_ship, require_full_scale_ship, require_hull_coefficients, ShipModel
from fw.ships.vessel_state import VesselState
from fw.environments.base_env import BaseEnv

from fw.simulators.ship_simulator import ShipSimulator
from fw.dynamics.ugent_surrogate import UGentSurrogate
from fw.dynamics.ugent_b01_surrogate import UGentB01Surrogate
from fw.dynamics.dynamics_model import Model
from fw.dynamics.fossen import Fossen
from fw.dynamics.nomoto import Nomoto

logger = logging.getLogger(__name__)


# -----------------------
# Module-level constants
# -----------------------
# Grid limits
CHECKPOINTS_DISTANCE = 350
MAX_CHECKPOINTS_DISTANCE = 10000
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

# Geometry / termination
PERPENDICULAR_LINE_LENGTH = 50.0
HEADING_CHANGE_THRESHOLD = np.pi / 2.0
CROSS_TRACK_TERMINATION_MULTIPLIER = 2.0
EPSILON = 1e-9


def calculate_perpendicular_lines(
        checkpoints: List[dict],
        line_length: float = PERPENDICULAR_LINE_LENGTH,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return a line perpendicular to the smoothed path tangent at each checkpoint.

    Args:
        checkpoints: Checkpoint dicts with a 'pos' entry.
        line_length: Length of each line, centred on the checkpoint.

    Returns:
        (start_point, end_point) array pairs, one per checkpoint.
    """

    def smooth_tangent(check_points: List[dict], index: int) -> np.ndarray:
        """
        Unit tangent at a checkpoint, averaging the neighboring segments.
        """

        # Single checkpoint: arbitrary tangent (unit x-axis)
        if len(check_points) == 1:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
        else:
            if index == 0:
                tangent = check_points[1]['pos'] - check_points[0]['pos']
            elif index == len(check_points) - 1:
                tangent = check_points[-1]['pos'] - check_points[-2]['pos']
            else:
                forward = check_points[index + 1]['pos'] - check_points[index]['pos']
                backward = check_points[index]['pos'] - check_points[index - 1]['pos']
                tangent = 0.5 * (forward + backward)

        norm = np.linalg.norm(tangent)
        # Zero-length tangent is returned unnormalized (zeros)
        return (tangent / norm if norm != 0.0 else tangent).astype(np.float32)

    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, cp in enumerate(checkpoints):
        pos = np.asarray(cp['pos'], dtype=np.float32)
        smth_tangent = smooth_tangent(checkpoints, i)
        perpendicular = np.array([-smth_tangent[1], smth_tangent[0]], dtype=np.float32)
        offset = perpendicular * (line_length / 2.0)
        lines.append((pos + offset, pos - offset))
    return lines


class PySimBaseEnv(BaseEnv):
    """
    Base Python-simulator environment for full-scale ship navigation along a
    checkpoint path (used by the collision-avoidance envs).
    """

    SUPPORTS_MODEL_SCALE_SHIPS: bool = False
    """
    False: map, distances, speeds and rewards are full scale (Western Scheldt),
    so model-scale ships (ShipModel.B01) are refused. A towing-tank env sets True.
    """


    def __init__(
            self,
            render_mode: Optional[str] = None,
            time_step: float = 0.1,
            max_steps: int = 1500,
            verbose: Optional[bool] = None,
            start_pos: Optional[np.ndarray] = None,
            target_pos: Optional[np.ndarray] = None,
            overall_contour: Optional[str] = None,
            depth_contour: Optional[str] = None,
            trajectory: Optional[str] = None,
            dynamics_model: Optional[Model] = Model.FOSSEN,
            ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
            wind: bool = False,
            current: bool = False,
            minimal_path_render: bool = False,
            ugent_surrogate_config: Optional[Dict] = None,
            ugent_b01_surrogate_config: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            render_mode: 'human' for rendering, otherwise None.
            time_step: Simulation time step (s).
            max_steps: Maximum steps per episode (truncation).
            verbose: Whether to log informational messages.
            start_pos: Optional initial ship position (2-array).
            target_pos: Optional target. The trajectory is cut to end here,
                walked forwards or (if target_pos lies before start_pos)
                backwards. If omitted, the trajectory's last point is the
                target, and it is walked forwards.
            overall_contour: Map outline CSV (default env_Sche_no_scale.csv).
            depth_contour: Navigable-area CSV (default env_Sche_250cm_no_scale.csv).
            trajectory: Path CSV (default trajectory_points_no_scale.csv).
            dynamics_model: Dynamics model to use.
            ship_model: Ship model to use.
            wind: Add 2 wind observation features (zeros: no model here has wind).
            current: Add 2 current observation features (zeros, as wind).
            minimal_path_render: Draw only a faint path line (no checkpoints).
            ugent_surrogate_config: Config for Model.UGENT_SURROGATE, e.g.
                {"ukc_percent": 363}; see fw.dynamics.ugent_surrogate.
            ugent_b01_surrogate_config: Config for Model.UGENT_B01_SURROGATE,
                e.g. {"ukc_percent": 379}; see fw.dynamics.ugent_b01_surrogate.
        """

        super().__init__(render_mode)

        # Core parameters
        self.wind = bool(wind)
        self.current = bool(current)
        self.minimal_path_render = bool(minimal_path_render)
        self.start_pos = start_pos
        self.overall_contour = overall_contour if overall_contour is not None else "env_Sche_no_scale.csv"
        self.depth_contour = depth_contour if depth_contour is not None else "env_Sche_250cm_no_scale.csv"
        self.trajectory = trajectory if trajectory is not None else "trajectory_points_no_scale.csv"
        self.ship_model = ship_model
        if not self.SUPPORTS_MODEL_SCALE_SHIPS:
            require_full_scale_ship(self.ship_model, type(self).__name__)
        self.max_steps = int(max_steps)
        self.time_step = float(time_step)
        self.dynamics_model = dynamics_model
        self.verbose = bool(verbose) if verbose is not None else False
        self.previous_ship_pos = None
        self.previous_heading = None
        self.performed_action = None
        self.previous_action = None
        self.background = None
        self.dynamics = None
        self.ship_pos = None

        self._agent_state = None
        self.other_agent_states = {}

        # Env-local RNG, reseeded via reset(seed=...)
        self._rng: np.random.Generator = np.random.default_rng()

        if start_pos is not None:
            self.initial_ship_pos = np.array(start_pos, dtype=np.float32)
        else:
            self.initial_ship_pos = np.array([5.0, 5.0], dtype=np.float32)

        self.randomization_scale = 1.0
        self.max_dist = np.sqrt(2) * float(MAX_GRID_POS)

        self._load_environment_data(target_pos)

        self.ship = create_ship(self.ship_model)
        self.phys_sim = self.create_simulator(ugent_surrogate_config, ugent_b01_surrogate_config)

        self._initialize_simulation()

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._initialize_observation_space()

        self.initialize_plots = True

    def neutral_action(self):
        """
        Return the neutral action: zeros of shape (2,), float32.
        """

        return np.zeros(2, dtype=np.float32)

    def get_agent_state(self):
        """
        Return this agent's simulator state (self.phys_sim.state).
        """

        return self.phys_sim.state

    def set_other_agents_states(self, agent_states):
        """
        Store the states of the other agents (all entries of agent_states except
        the one keyed by self.env_name).
        """

        filtered = {
            agent_name: state
            for agent_name, state in agent_states.items()
            if agent_name != self.env_name
        }

        self.other_agent_states.update(filtered)

    # -----------------------
    # Initialization helpers
    # -----------------------
    def _initialize_simulation(self) -> None:
        """
        Reset the ship to its start state and reset the simulator.
        """

        self.ship_pos = self.initial_ship_pos.copy()
        self.previous_ship_pos = self.initial_ship_pos.copy()
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.checkpoint_index = 1

        # Focused-encounter training: optionally start at a random mid-path
        # checkpoint (overwrites ship_pos, previous_ship_pos, checkpoint_index).
        if getattr(self, "random_segment_start", False):
            self._apply_random_starting_checkpoint()

        self.previous_heading = 0.0
        self.cross_error = 0.0
        self.step_count = 0
        # Episode step index; unlike step_count it is not reset at checkpoints.
        # Used for episode-timed metrics (e.g. COLAV time-to-first-action).
        self._episode_step = 0
        self.ukc = 0.0

        # Head toward the first checkpoint (if there is one)
        if len(self.checkpoints) > self.checkpoint_index:
            direction_vector = self.checkpoints[self.checkpoint_index]['pos'] - self.ship_pos
            self.ship_heading = self.normalize_angle(self._safe_heading_from_vector(direction_vector))
        else:
            self.ship_heading = 0.0

        self.phys_sim.reset(VesselState(
            terminated=False,
            truncated=False,
            x=self.ship_pos[0],
            y=self.ship_pos[1],
            heading=self.ship_heading,
            u=0.0, v=0.0, r=0.0))
        self._agent_state = self.phys_sim.state
        self.performed_action = np.zeros(2, dtype=np.float32)

        # Must follow phys_sim.reset() to reflect the new episode.
        self.heading_error = self._calculate_heading_error(self.los_guidance_heading(), self.phys_sim.state.heading)
        self.velocities = self.phys_sim.state.u

    def _apply_random_starting_checkpoint(self) -> None:
        """
        Start at the first checkpoint of a uniformly random segment (excluding
        the last skip_tail_segments); sets ship_pos, previous_ship_pos and
        checkpoint_index.
        """

        n_skip = int(getattr(self, "skip_tail_segments", 1))
        n_segments = len(self.checkpoints) - 1
        max_seg = n_segments - n_skip
        if max_seg <= 0:
            return  # path too short to randomize

        # Segment K: from checkpoints[K] to checkpoints[K+1], K in [0, max_seg - 1].
        chosen_seg = int(self._rng.integers(low=0, high=max_seg))
        self.ship_pos = np.asarray(
            self.checkpoints[chosen_seg]["pos"], dtype=np.float32,
        ).copy()
        self.previous_ship_pos = self.ship_pos.copy()
        self.checkpoint_index = chosen_seg + 1

    def create_simulator(self, ugent_surrogate_config: Optional[Dict] = None,
                         ugent_b01_surrogate_config: Optional[Dict] = None):
        """
        Create the ShipSimulator for the configured ship, dynamics model and
        time step (also sets self.dynamics).

        Args:
            ugent_surrogate_config: Configuration for the UGent surrogate model
                (required for Model.UGENT_SURROGATE).
            ugent_b01_surrogate_config: Configuration for the UGent B01 model-scale
                surrogate (required for Model.UGENT_B01_SURROGATE).

        Raises:
            ValueError: Model.UGENT_SURROGATE with a ship other than the Tripoli,
                Model.UGENT_B01_SURROGATE with a ship other than B01, Model.FOSSEN /
                Model.NOMOTO with a model-scale ship (ShipModel.B01), or an
                invalid ugent_surrogate_config / ugent_b01_surrogate_config.
            RuntimeError: Unsupported (UANTWERP, FH_TABULAR) or unknown dynamics model.
        """

        if self.dynamics_model == Model.FOSSEN:
            require_hull_coefficients(self.ship_model, f"{type(self).__name__} with Model.FOSSEN")
            self.dynamics = Fossen(self.ship.specifications)
        elif self.dynamics_model == Model.NOMOTO:
            require_hull_coefficients(self.ship_model, f"{type(self).__name__} with Model.NOMOTO")
            self.dynamics = Nomoto(self.ship.specifications)
        elif self.dynamics_model == Model.UGENT_SURROGATE:
            if self.ship_model != ShipModel.TRIPOLI:
                raise ValueError(
                    f"Model.UGENT_SURROGATE is identified for ShipModel.TRIPOLI only, "
                    f"got ship_model={self.ship_model}."
                )
            # No bank geometry here, so no bank-distance provider: bank forces
            # are zero (PySimEnv does set one).
            self.dynamics = UGentSurrogate(self.ship.specifications, ugent_surrogate_config)
        elif self.dynamics_model == Model.UGENT_B01_SURROGATE:
            if self.ship_model != ShipModel.B01:
                raise ValueError(
                    f"Model.UGENT_B01_SURROGATE is identified for ShipModel.B01 only, "
                    f"got ship_model={self.ship_model}."
                )
            # No bank geometry here: bank forces are zero.
            self.dynamics = UGentB01Surrogate(self.ship.specifications, ugent_b01_surrogate_config)
        elif self.dynamics_model in (Model.UANTWERP, Model.FH_TABULAR):
            raise RuntimeError(
                f"{type(self).__name__} (PySimBaseEnv / collision-avoidance envs) does not support "
                f"{self.dynamics_model} yet; supported: FOSSEN, NOMOTO, UGENT_SURROGATE, UGENT_B01_SURROGATE. "
                f"Use PySimEnv for UANTWERP / FH_TABULAR, or pick another dynamics model / profile."
            )
        else:
            raise RuntimeError(f"Unknown dynamics model {self.dynamics_model} specified.")

        return ShipSimulator(self.ship.specifications, self.dynamics, self.time_step)

    # -------------------------
    # Environment data loaders
    # -------------------------
    def _load_environment_data(self, target_pos: Optional[np.ndarray]) -> None:
        """
        Load the map outline, navigable polygon and trajectory CSVs, and build
        the checkpoints (evenly spaced, with perpendicular guidance lines) and
        the target position.

        Raises:
            FileNotFoundError: If a required CSV file is missing.
            ValueError: If a file has the wrong shape, or the path yields
                fewer than 2 checkpoints.
            RuntimeError: If reading a file fails otherwise.
        """

        base_path = os.path.dirname(os.path.abspath(__file__))

        def load_csv_strict(name: str) -> np.ndarray:
            """
            Load a required (N, 2) CSV from this module's directory, or raise.
            """

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

        self.overall = load_csv_strict(self.overall_contour)
        self.obstacles = load_csv_strict(self.depth_contour)
        self.polygon_shape = Polygon(self.obstacles)

        path = load_csv_strict(self.trajectory)

        target_pos_arr = np.array(target_pos, dtype=np.float32) if target_pos is not None else None

        # Path from just after start_pos to target_pos (or the trajectory end)
        path = self._reduce_path(path, self.initial_ship_pos, target_pos_arr)
        path = create_checkpoints_from_simple_path(path, CHECKPOINTS_DISTANCE)

        # First checkpoint is the ship's start position
        if len(path) == 0:
            path = [self.initial_ship_pos.copy()]
        else:
            path = np.insert(path, 0, self.initial_ship_pos.copy(), axis=0)

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

        lines = calculate_perpendicular_lines(checkpoints, line_length=PERPENDICULAR_LINE_LENGTH)

        self.checkpoints = []
        for cp, line in zip(checkpoints, lines):
            cp['perpendicular_line'] = line
            self.checkpoints.append(cp)

        if len(self.checkpoints) < 2:
            raise ValueError(
                f"Path reduction from start_pos={self.initial_ship_pos} resulted "
                f"in fewer than 2 checkpoints. Check that the CSV trajectory file "
                f"contains points ahead of the ship's start position, and that "
                f"CHECKPOINTS_DISTANCE ({CHECKPOINTS_DISTANCE}) is not larger than "
                f"the total path length."
            )

        # Direction of the last path segment
        p_last = self.checkpoints[-1]['pos'].copy()
        p_before_last = self.checkpoints[-2]['pos'].copy()
        self.direction_last_path = p_last - p_before_last

        if self.checkpoints:
            self.target_pos = self.checkpoints[-1]['pos']
        else:
            if target_pos_arr is not None:
                self.target_pos = target_pos_arr
            else:
                self.target_pos = self.initial_ship_pos.copy()

    # -----------------------
    # Utility & math helpers
    # -----------------------
    def sample_point_in_river(self, nbr_samples: int = 1, max_tries: int = 20000) -> np.ndarray:
        """
        Rejection-sample nbr_samples points inside the river polygon.

        Returns:
            (N, 2) array; N may be smaller (with a warning) if max_tries is hit.
        """

        minx, miny, maxx, maxy = self.polygon_shape.bounds
        rng = self._rng
        samples = []
        tries = 0

        while len(samples) < nbr_samples and tries < max_tries:
            tries += 1
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)

            if self.ship_in_open_water(np.array([x, y])):
                samples.append((x, y))

        if len(samples) < nbr_samples:
            logger.warning("Only generated %d points out of requested %d "
                            "(polygon may be thin or max_tries reached).", len(samples), nbr_samples)

        return np.array(samples, dtype=np.float32)

    @staticmethod
    def get_keel_clearance() -> float:
        """
        Keel clearance; always 0.0 here (no depth data).
        """

        return 0.0

    @staticmethod
    def _has_enough_keel_clearance(depth_threshold: float = 0.5) -> bool:
        """
        Keel-clearance check placeholder; always True (no depth data).
        """

        return True

    def ship_in_open_water(self, ship_position: np.ndarray) -> bool:
        """
        True if (x, y) lies inside the navigable polygon and has keel clearance.
        """

        x, y = ship_position[0], ship_position[1]
        min_x, min_y, max_x, max_y = self.polygon_shape.bounds
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

        contains = self.polygon_shape.contains(Point(ship_position))
        deep_enough = self._has_enough_keel_clearance()
        return bool(contains and deep_enough)

    @staticmethod
    def _reduce_path(path: np.ndarray, start_pos: np.ndarray, target_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cut the trajectory to run from just after start_pos to target_pos, walking
        it backwards if target_pos lies earlier in the stored point order. The last
        point is set exactly to target_pos; without target_pos the path runs forwards
        to the last point.

        Args:
            path: Full trajectory, (N, 2).
            start_pos: Ship's starting position.
            target_pos: Optional end position.

        Returns:
            The reduced path.

        Raises:
            ValueError: Empty result, or start_pos and target_pos map to the same
                trajectory point.
        """

        if len(path) == 0:
            raise ValueError("Path is empty.")

        start_distances = np.linalg.norm(path - start_pos, axis=1)
        start_index = int(np.argmin(start_distances))

        if target_pos is None:
            reduced_path = path[start_index+1:]

            if len(reduced_path) == 0:
                raise ValueError(
                    f"_reduce_path produced an empty path. The start position "
                    f"{np.asarray(start_pos).tolist()} is at or beyond the last point in the trajectory "
                    f"(closest index {start_index} of {len(path) - 1}). "
                    f"Check that start_pos lies near the beginning of the trajectory."
                )

            return reduced_path

        target_pos = np.asarray(target_pos)

        target_distances = np.linalg.norm(path - target_pos, axis=1)
        target_index = int(np.argmin(target_distances))

        if target_index == start_index:
            raise ValueError(
                f"start_pos {np.asarray(start_pos).tolist()} and target_pos "
                f"{target_pos.tolist()} both resolve to the same point on the "
                f"trajectory (closest index {start_index}); no path exists "
                f"between them."
            )

        if target_index > start_index:
            truncated_path = path[start_index + 1: target_index + 1].copy()
        else:
            # Backwards: from just before start_index down to target_index
            truncated_path = path[target_index:start_index][::-1].copy()

        if len(truncated_path) == 0:
            # Defensive; should not happen since target_index != start_index
            raise ValueError(
                f"_reduce_path produced an empty path between start_pos "
                f"{np.asarray(start_pos).tolist()} (index {start_index}) and target_pos "
                f"{target_pos.tolist()} (index {target_index})."
            )

        # End exactly at target_pos, not at the nearest trajectory point
        truncated_path[-1] = target_pos.astype(truncated_path.dtype)

        return truncated_path

    @staticmethod
    def _distance_from_point_to_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        Perpendicular distance from point to the infinite line through
        line_start and line_end (distance to line_start if they coincide).
        """

        line_vec = line_end - line_start
        point_vec = point - line_start
        line_mag_squared = np.dot(line_vec, line_vec)

        if line_mag_squared == 0.0:
            return float(np.linalg.norm(point - line_start))

        projection_scalar = np.dot(point_vec, line_vec) / line_mag_squared
        projected_point = line_start + projection_scalar * line_vec
        return float(np.linalg.norm(point - projected_point))

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Wrap an angle (rad) to [-pi, pi).
        """

        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _safe_heading_from_vector(vec):
        """
        Heading angle of vec (rad), or 0.0 for a zero vector.
        """

        if np.linalg.norm(vec) > 0.0:
            return np.arctan2(vec[1], vec[0])
        return 0.0

    def _calculate_heading_error(self, target_heading: float, current_heading: float, dead_zone_deg: float = 5.0) -> float:
        """
        Wrapped heading error target - current (rad); 0 inside the
        +/- dead_zone_deg dead zone.
        """

        error = self.normalize_angle(target_heading - current_heading)
        dead_zone = np.radians(dead_zone_deg)
        return 0.0 if abs(error) < dead_zone else error

    # -----------------------
    # Observation & reset
    # -----------------------
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """
        Observation space: 12 base features in [-1, 1] (see _get_obs), plus 2
        each for wind and current when enabled.
        """

        base_low = np.array(
            [
                -1.0,   # Surge velocity
                -1.0,   # Sway velocity
                -1.0,   # Yaw rate
                -1.0,   # Distance to current checkpoint
                -1.0,   # Distance to checkpoint+1
                -1.0,   # Cross-track error
                -1.0,   # Heading error to current checkpoint
                -1.0,   # Heading error to next checkpoint
                -1.0,   # Heading error vs current segment
                -1.0,   # Heading error vs next segment
                -1.0,   # Previous rudder angle
                -1.0,   # LOS heading error
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
                1.0,    # Heading error to current checkpoint
                1.0,    # Heading error to next checkpoint
                1.0,    # Heading error vs current segment
                1.0,    # Heading error vs next segment
                1.0,    # Previous rudder angle
                1.0,    # LOS heading error
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
        Perturb the initial ship position uniformly by up to
        +/- randomization_scale per axis (clipped to the grid).

        Args:
            randomization_scale: New scale; None keeps the current one.

        Raises:
            ValueError: If randomization_scale is not positive.
        """

        if randomization_scale is not None:
            if randomization_scale <= 0:
                raise ValueError("randomization_scale must be positive")

            self.randomization_scale = randomization_scale

        # Env RNG, so reset(seed=...) makes this reproducible
        ran_gen = self._rng

        perturbation = ran_gen.uniform(
            low=-self.randomization_scale,
            high=self.randomization_scale,
            size=self.initial_ship_pos.shape
        )

        if self.start_pos is not None:
            self.initial_ship_pos = np.clip(self.start_pos + perturbation, MIN_GRID_POS, MAX_GRID_POS)
        else:
            self.initial_ship_pos = np.clip(np.array([5.0, 5.0], dtype=np.float32) + perturbation, MIN_GRID_POS, MAX_GRID_POS)


    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset the episode; a seed reseeds the env RNG.

        Returns:
            (observation, info) with an empty info dict.
        """

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        super().reset(seed=seed)

        self.env_specific_reset()
        self._initialize_simulation()

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        Build the normalized observation: u, v, r, distances to the current and
        next checkpoint (log-scaled), signed cross-track error, four heading
        errors, previous rudder angle, LOS heading error, plus zero wind/current
        placeholders when enabled.
        """

        norm_velocities = np.array([
            np.clip(self.phys_sim.state.u / self.ship.specifications.surge_limits[1], -1, 1),
            np.clip(self.phys_sim.state.v / self.ship.specifications.sway_limits[1], -1, 1),
            np.clip(self.phys_sim.state.r / self.ship.specifications.yaw_rate_limits[1], -1, 1)
        ], dtype=np.float32)

        current_chkp_idx = self.checkpoint_index
        prev_chkp_idx = max(0, current_chkp_idx - 1)
        next_chkp_idx = current_chkp_idx + 1

        # Beyond the last checkpoint, extrapolate along the last path segment
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

        norm_distance = self._logarithmic_scaling(
            value=float(np.linalg.norm(self.ship_pos - current_chkp_pos)),
            value_range=(0, MAX_CHECKPOINTS_DISTANCE),
            half_value=0.5 * CHECKPOINTS_DISTANCE)

        norm_next_distance = self._logarithmic_scaling(
            value=float(np.linalg.norm(self.ship_pos - next_chkp_pos)),
            value_range=(0, 2 * MAX_CHECKPOINTS_DISTANCE),
            half_value=1.5 * CHECKPOINTS_DISTANCE)

        # Cross-track error (normalized, signed)
        norm_cross_error = self._logarithmic_scaling(
            value=self._distance_from_point_to_line(self.ship_pos, prev_chkp_pos, current_chkp_pos),
            value_range=(0, MAX_CHECKPOINTS_DISTANCE),
            half_value=50.0)

        track_vec = current_chkp_pos - prev_chkp_pos
        ship_vec = self.ship_pos - prev_chkp_pos
        # 2D cross product by hand (np.cross on 2D vectors is deprecated in NumPy 2)
        cross_z = track_vec[0] * ship_vec[1] - track_vec[1] * ship_vec[0]
        norm_cross_error *= 1.0 if cross_z >= 0.0 else -1.0

        # Heading errors: to current checkpoint
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

        # LOS guidance error
        los_heading_error = self._calculate_heading_error(
            self.los_guidance_heading(),
            self.phys_sim.state.heading,
            0.0
        )

        obs = np.hstack([
            norm_velocities,
            [2 * norm_distance - 1],
            [2 * norm_next_distance - 1],
            [norm_cross_error],
            [heading_error / np.pi],
            [heading_error2 / np.pi],
            [heading_error_parallel / np.pi],
            [heading_error_parallel2 / np.pi],
            self.performed_action[0],               # Previous rudder angle
            los_heading_error / np.pi,
        ])

        # Placeholders: the dynamics models of this env have no wind or current.
        if self.wind:
            obs = np.hstack([obs, np.zeros(2, dtype=np.float32)])

        if self.current:
            obs = np.hstack([obs, np.zeros(2, dtype=np.float32)])

        return obs.astype(np.float32)

    # -----------------------
    # Dynamics update & step
    # -----------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance one time step.

        Args:
            action: [rudder, thrust] in [-1, 1].

        Returns:
            (observation, reward, terminated, truncated, info). After the
            episode ended, reward is 0 and the flags are repeated.

        Raises:
            ValueError: If action does not have shape (2,).
        """

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(
                f"Expected action shape (2,), got {action.shape}. "
                f"Action values: {action}. Use [-1,1] for [rudder, thrust]."
            )

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            self.previous_action = self.performed_action.copy()
            self.previous_ship_pos = self.ship_pos.copy()
            self.previous_heading = self.phys_sim.state.heading

            self.phys_sim.step(action, self.enable_smoothing, self.enable_rudder_angle_scaling)
            self._agent_state = self.phys_sim.state
            self.ship_pos = np.array([self.phys_sim.state.x, self.phys_sim.state.y], dtype=np.float32)
            self.performed_action = self.phys_sim.actuators.current_actuator_state.copy()
            self.step_count += 1

            reward, terminated = self._calculate_reward()

            # step_count restarts at each checkpoint, so max_steps is per checkpoint
            truncated = False
            if self.step_count >= self.max_steps and not terminated:
                # reward = -250
                truncated = True

            self.phys_sim.state.terminated = terminated
            self.phys_sim.state.truncated = truncated
        else:
            reward = 0.0
            terminated = self.phys_sim.state.terminated
            truncated = self.phys_sim.state.truncated

        info = {
            "rewards": reward,
            "terminations": terminated,
            "truncations": truncated,
        }

        # Gymnasium contract: time-limit cut-off is truncated (bootstrap V(s')),
        # a real episode end is terminated. Flags are mirrored in info.
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # -------------------
    # Reward calculation
    # -------------------
    def _calculate_reward(self) -> Tuple[float, bool]:
        """
        Compute the step reward, advance the checkpoint index, and check
        termination (cross-track limit, grounding, sudden heading change, or
        all checkpoints passed).

        Returns:
            (reward, done)
        """

        prev_checkpoint_idx = max(0, self.checkpoint_index - 1)
        prev_checkpoint_pos = self.checkpoints[prev_checkpoint_idx]['pos']

        current_checkpoint = self.checkpoints[self.checkpoint_index]
        current_checkpoint_pos = current_checkpoint['pos']

        cross_track_error = self._distance_from_point_to_line(self.ship_pos, prev_checkpoint_pos, current_checkpoint_pos)
        self.cross_error = cross_track_error
        self.heading_error = self._calculate_heading_error(self.los_guidance_heading(), self.phys_sim.state.heading)
        self.velocities = self.phys_sim.state.u

        self.ukc = self.get_keel_clearance()

        rudder_change = abs(self.performed_action[0] - self.previous_action[0])
        # Both in rad. The 1e-6 tolerance keeps a move at exactly the maximum slew
        # rate out of the -1 penalty despite float32 rounding.
        max_rudder_change = self.ship.specifications.max_rudder_rate * self.time_step + 1e-6  # rad
        heading_error = self._calculate_heading_error(self.los_guidance_heading(), self.phys_sim.state.heading)
        distance_traveled = abs(np.linalg.norm(self.ship_pos - self.previous_ship_pos))
        velocity = distance_traveled / self.time_step
        target_velocity = 4

        r_forward = - 0.25 * abs(target_velocity - velocity) / target_velocity
        r_heading_alignment = - 0.5 * abs(heading_error)
        r_cte = - 0.0 * cross_track_error # TODO: 0.05
        r_rudder_angle_change = - 0.4 * (rudder_change / max_rudder_change) ** 1 if rudder_change < max_rudder_change else -1
        r_rudder_angle = - 0.05 * (abs(self.performed_action[0]) / 0.2) ** (2 if abs(self.performed_action[0]) < 0.2 else 4)
        r_thrust_change = 0  # - 0.5 * thrust_change

        reward = (
                r_forward +
                r_heading_alignment +
                r_cte +
                r_rudder_angle_change +
                r_rudder_angle +
                r_thrust_change
        )
        reward = 0.9 + reward

        # Checkpoint handling (may increment checkpoint_index)
        checkpoint_distance = np.linalg.norm(self.ship_pos - current_checkpoint["pos"])

        # Reached checkpoint circle
        if self.segment_point_distance(self.previous_ship_pos, self.ship_pos, current_checkpoint["pos"]) <= current_checkpoint['radius']:
            if self.checkpoint_index == len(self.checkpoints) - 1:
                logger.info("%s - target REACHED at distance %.2f", self.env_name, checkpoint_distance)
            self.checkpoint_index += 1
            self.step_count = 0

        # Crossed the perpendicular guidance line
        elif self._is_near_perpendicular_line(current_checkpoint):
            # if self.checkpoint_index == len(self.checkpoints) - 1:
            #     print(f"Target passed at distance {checkpoint_distance:.2f}")
            self.checkpoint_index += 1
            self.step_count = 0

        done = False
        heading_change = abs(self._calculate_heading_error(self.previous_heading, self.phys_sim.state.heading))

        termination_conditions = [
            cross_track_error > CROSS_TRACK_TERMINATION_MULTIPLIER * CHECKPOINTS_DISTANCE,
            not self.ship_in_open_water(self.ship_pos),
            heading_change > HEADING_CHANGE_THRESHOLD,
        ]

        if any(termination_conditions):
            # reward = -250
            done = True

        # All checkpoints reached or passed
        if self.checkpoint_index >= len(self.checkpoints):
            done = True

        return reward, done

    @staticmethod
    def segment_point_distance(p1, p2, c):
        """
        Shortest distance from point c to the segment p1-p2.
        """

        if np.allclose(p1, p2):
            return np.linalg.norm(c - p1)

        v = p2 - p1
        w = c - p1

        t = np.dot(w, v) / np.dot(v, v)
        t = np.clip(t, 0.0, 1.0)

        closest = p1 + t * v
        return np.linalg.norm(c - closest)

    @staticmethod
    def _segments_intersect(prev_point: np.ndarray, curr_point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> bool:
        """
        True if the segment prev_point-curr_point touches or crosses the
        infinite line through line_start and line_end.
        """

        def _orientation(a, b, c):
            """2D cross product: > 0 if c is left of line ab, < 0 if right, 0 if collinear."""

            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        o1 = _orientation(line_start, line_end, prev_point)
        o2 = _orientation(line_start, line_end, curr_point)

        if abs(o1) <= EPSILON or abs(o2) <= EPSILON:
            return True

        return (o1 > 0 > o2) or (o1 < 0 < o2)

    def _is_near_perpendicular_line(self, checkpoint: dict) -> bool:
        """
        True if the last ship move crossed the checkpoint's perpendicular line.
        """

        line_start, line_end = checkpoint['perpendicular_line']
        return self._segments_intersect(self.previous_ship_pos, self.ship_pos, line_start, line_end)

    def los_guidance_heading(self):
        """
        Desired LOS heading (rad) for the current path segment.

        LOS guidance law from Changyuan Chen's PhD thesis "Numerical and
        Experimental Study on Ship Motion Control Systems in Shallow Water"
        (Flanders Hydraulics).
        """

        current_checkpoint_idx = min(self.checkpoint_index, len(self.checkpoints) - 1)
        p_k = self.checkpoints[current_checkpoint_idx]['pos'].copy()
        pos = self.ship_pos

        if current_checkpoint_idx + 1 > len(self.checkpoints) - 1:
            p_k_plus_1 = self.checkpoints[-1]['pos'].copy()
            direction = p_k_plus_1 - self.checkpoints[-2]['pos']
            p_k_plus_1 += direction
        else:
            p_k_plus_1 = self.checkpoints[current_checkpoint_idx + 1]['pos'].copy()

        x, y = pos[:]
        x_k, y_k = p_k[:]
        x_k_plus_1, y_k_plus_1 = p_k_plus_1[:]

        # previous checkpoint
        if current_checkpoint_idx > 0:
            p_k_min_1 = self.checkpoints[current_checkpoint_idx - 1]['pos'].copy()
        else:
            # None: use a virtual one behind the ship
            p_k_min_1 = p_k - (p_k_plus_1 - p_k)
        x_k_min_1, y_k_min_1 = p_k_min_1[:]

        seg_sq = (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2
        if seg_sq < EPSILON:
            # Coincident checkpoints: point directly at p_k_plus_1
            return math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)

        # Orthogonal projection A of the ship onto the line p_k_min_1, p_k
        proj_scalar = (
                          (x - x_k_min_1) * (x_k - x_k_min_1) +
                          (y - y_k_min_1) * (y_k - y_k_min_1)
                      ) / seg_sq

        x_a = x_k_min_1 + proj_scalar * (x_k - x_k_min_1)
        y_a = y_k_min_1 + proj_scalar * (y_k - y_k_min_1)

        # Look-ahead distance L_A
        c = 0.5   # LOS look-ahead coefficient (thesis default, p. 148)
        delta_y = math.sqrt((x - x_a) ** 2 + (y - y_a) ** 2)
        l_pp = 0.95 * self.ship.specifications.length
        l_a_sq = (delta_y + c * l_pp) ** 2 - delta_y ** 2
        l_a = math.sqrt(max(l_a_sq, 0.0))   # clamp float negatives

        # LOS point Ploss
        seg_len = math.sqrt(seg_sq)
        x_loss = x_a + ((x_k - x_k_min_1) / seg_len) * l_a
        y_loss = y_a + ((y_k - y_k_min_1) / seg_len) * l_a

        # Acceptance-circle radius Rk of the current checkpoint
        w_los = 0.5   # acceptance-circle coefficient (thesis default, p. 148)
        r_k = w_los * l_pp

        heading_p_k_min_1_p_k = self._safe_heading_from_vector(p_k - p_k_min_1)
        heading_p_k_p_k_plus_1 = self._safe_heading_from_vector(p_k_plus_1 - p_k)
        heading_difference = self.normalize_angle(
            heading_p_k_min_1_p_k - heading_p_k_p_k_plus_1
        )
        collinear = abs(heading_difference) < (1.0 / 57.0)   # within ~1 deg

        if collinear:
            dist_a_pk = math.sqrt((x_a - x_k) ** 2.0 + (y_a - y_k) ** 2.0)
            if dist_a_pk > r_k:  # |A Pk| > Rk
                los_angle = math.atan2(y_loss - y, x_loss - x)
            else:  # |A Pk| < Rk: aim at Pk+1 instead of Ploss
                los_angle = math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)
        else:
            dist_p_pk = math.sqrt((x - x_k) ** 2 + (y - y_k) ** 2)
            if dist_p_pk > 2 * r_k:  # |P Pk| > 2*Rk
                if l_a < dist_p_pk:  # LA < |P Pk|
                    los_angle = math.atan2(y_loss - y, x_loss - x)
                else:  # LA > |P Pk|: carry the look-ahead L'A onto the next segment
                    next_seg_sq = (x_k_plus_1 - x_k) ** 2 + (y_k_plus_1 - y_k) ** 2
                    if next_seg_sq < EPSILON:
                        return math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)

                    next_seg_len = math.sqrt(next_seg_sq)
                    l_accent_a = l_a - math.sqrt(
                        (x_k - x_a) ** 2 + (y_k - y_a) ** 2
                    )
                    x_accent_loss = x_k + ((x_k_plus_1 - x_k) / next_seg_len) * l_accent_a
                    y_accent_loss = y_k + ((y_k_plus_1 - y_k) / next_seg_len) * l_accent_a

                    los_angle = math.atan2(y_accent_loss - y, x_accent_loss - x)
            else:  # |P Pk| < 2*Rk: switch to the next segment (Pk-1, Pk := Pk, Pk+1)
                x_k_min_1, y_k_min_1 = x_k, y_k
                x_k, y_k = x_k_plus_1, y_k_plus_1

                seg_sq2 = (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2
                if seg_sq2 < EPSILON:
                    return math.atan2(y_k - y, x_k - x)

                proj_scalar2 = (
                                   (x - x_k_min_1) * (x_k - x_k_min_1) +
                                   (y - y_k_min_1) * (y_k - y_k_min_1)
                               ) / seg_sq2

                x_a = x_k_min_1 + proj_scalar2 * (x_k - x_k_min_1)
                y_a = y_k_min_1 + proj_scalar2 * (y_k - y_k_min_1)

                delta_y = math.sqrt((x - x_a) ** 2 + (y - y_a) ** 2)   # Cross-track error
                l_a_sq = (delta_y + c * l_pp) ** 2 - delta_y ** 2
                l_a = math.sqrt(max(l_a_sq, 0.0))

                seg_len2 = math.sqrt(seg_sq2)
                x_loss = x_a + ((x_k - x_k_min_1) / seg_len2) * l_a
                y_loss = y_a + ((y_k - y_k_min_1) / seg_len2) * l_a
                los_angle = math.atan2(y_loss - y, x_loss - x)

        return los_angle

    @staticmethod
    def _logarithmic_scaling(value: float, value_range: tuple, half_value: float = 10):
        """
        Map value from value_range = (min, max) to [0, 1] with a log curve such
        that half_value maps to 0.5 (finer resolution near min).

        Raises:
            ValueError: If value lies outside value_range.
        """

        min_value, max_value = value_range

        if value < min_value or value > max_value:
            raise ValueError(f"value {value} exceeded the range [{min_value}, {max_value}]")

        x = (value - min_value) / (max_value - min_value)

        m = (half_value - min_value) / (max_value - min_value)

        # Curvature parameter
        k = (1 - 2 * m) / (m ** 2)

        return math.log1p(k * x) / math.log1p(k)

    # -----------------------
    # Rendering
    # -----------------------
    def _initialize_rendering(self) -> None:
        """
        Create the figure and plot handles.
        """

        self.initialize_plots = False

        # Screen size via a temporary Tk root. tkinter is imported lazily so the
        # framework stays importable on headless machines without Tk.
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except Exception:   # ImportError (no Tk) or tk.TclError (no display)
            screen_width, screen_height = MAX_FIG_WIDTH, MAX_FIG_HEIGHT

        fig_width = min(MAX_FIG_WIDTH, screen_width)
        fig_height = min(MAX_FIG_HEIGHT, screen_height)
        dpi = DPI

        self.fig, self.ax = plt.subplots(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)

        # Place the window at the top left where the backend allows it
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
            pass

        self.ship_plot, = self.ax.plot([], [], 'bo', markersize=10, label='Ship')
        self.target_plot, = self.ax.plot([], [], 'ro', markersize=10, label='Target')
        self.heading_line, = self.ax.plot([], [], color='black', linewidth=2, label='Heading')

        self.ax.set_xlim(MIN_GRID_POS, MAX_GRID_POS)
        self.ax.set_ylim(MIN_GRID_POS, MAX_GRID_POS)
        self.ax.set_title("Ship Navigation in a Path Following Environment")
        self.ax.legend()

    def _draw_fixed_landmarks(self) -> None:
        """
        Draw static features: path, checkpoints, polygons and target.
        """

        if not hasattr(self, 'ax') or not self.checkpoints:
            return

        path_points = [self.ship_pos] + [cp['pos'] for cp in self.checkpoints] + [self.target_pos]

        if getattr(self, "minimal_path_render", False):
            logger.debug("minimal path render")
            xs = [p[0] for p in path_points]
            ys = [p[1] for p in path_points]
            self.ax.plot(xs, ys, color='green', alpha=0.2,
                         linewidth=1.0, label='Path')
        else:
            logger.debug("no minimal path render")
            for i in range(len(path_points) - 1):
                start_point = path_points[i]
                end_point = path_points[i + 1]
                # Label only the first segment (single legend entry)
                self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                             'g-', label='Path' if i == 0 else "")

            for i, checkpoint in enumerate(self.checkpoints):
                circle = patches.Circle((checkpoint['pos'][0], checkpoint['pos'][1]),
                                        10.0, color='black', alpha=0.3)
                self.ax.add_patch(circle)
                start_point, end_point = checkpoint['perpendicular_line']
                self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                             'g-', label='PerpLine' if i == 0 else "")

        polygon_patch = patches.Polygon(self.obstacles, closed=True, edgecolor='r',
                                        facecolor='none', lw=2, label='Waterway')
        western_scheldt = patches.Polygon(self.overall, closed=True, edgecolor='brown',
                                          facecolor='none', lw=2, label='Western Scheldt')
        self.ax.add_patch(polygon_patch)
        self.ax.add_patch(western_scheldt)

        self.target_plot.set_data(self.target_pos[0:1], self.target_pos[1:2])

    def _on_draw(self, event) -> None:
        """
        Refresh the cached blit background after a redraw (e.g. zoom or pan).
        """

        if event.canvas is self.fig.canvas:
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def render(self) -> None:
        """
        Draw the ship and heading (human render mode only), using blitting.
        """

        if self.render_mode != 'human':
            return

        if self.initialize_plots and not hasattr(self, 'ship_plot'):
            self._initialize_rendering()
            self._draw_fixed_landmarks()
            plt.show(block=False)
            plt.pause(0.001)
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig.canvas.mpl_connect('draw_event', self._on_draw)

        canvas = self.fig.canvas

        def _safe(callable_obj):
            """
            Call callable_obj(), ignoring rendering errors so they cannot abort the run.
            """

            try:
                callable_obj()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass

        if hasattr(self, 'background'):
            _safe(lambda: canvas.restore_region(self.background))

        heading_line_length = 30.0
        self.ship_plot.set_data(self.ship_pos[0:1], self.ship_pos[1:2])
        heading_x = self.ship_pos[0] + np.cos(self.phys_sim.state.heading) * heading_line_length
        heading_y = self.ship_pos[1] + np.sin(self.phys_sim.state.heading) * heading_line_length
        self.heading_line.set_data([self.ship_pos[0], heading_x], [self.ship_pos[1], heading_y])

        try:
            if hasattr(self.fig.canvas, 'blit'):
                self.ax.draw_artist(self.ship_plot)
                self.ax.draw_artist(self.heading_line)
                self.fig.canvas.blit(self.ax.bbox)
            else:
                self.fig.canvas.draw()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            _safe(self.fig.canvas.draw)

        # Flush pending GUI events
        _safe(lambda: self.fig.canvas.start_event_loop(0.0001))

    def close(self) -> None:
        """
        Close the render window, if any.
        """

        if self.render_mode == 'human' and hasattr(self, 'fig'):
            try:
                plt.close(self.fig)
            except (RuntimeError, AttributeError):
                pass
