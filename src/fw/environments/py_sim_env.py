import os
import copy
import math
import shapely
import logging
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
    logging.getLogger(__name__).warning("'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use('Agg')  # Use non-interactive backend as fallback

from shapely import LineString
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Point, Polygon, MultiPoint, GeometryCollection
from fw.environments.tools import create_checkpoints_from_simple_path
from fw.ships.ship_factory import create_ship, ShipModel
from fw.ships.vessel_state import VesselState
from fw.environments.base_env import BaseEnv
from fw.reward.reward import Reward

from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate
from fw.dynamics.fh_tabular_model import FhTabularModel
from fw.simulators.ship_simulator import ShipSimulator
from fw.dynamics.ugent_surrogate import UGentSurrogate
from fw.dynamics.dynamics_model import Model
from fw.dynamics.fossen import Fossen
from fw.dynamics.nomoto import Nomoto

logger = logging.getLogger(__name__)


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
TARGET_SURGE_VELOCITY_MS = 4.0  # Target surge velocity in m/s used for velocity reward shaping

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
        """
        Compute a smoothed tangent at a checkpoint by averaging neighbor vectors.
        """

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
    """
    Custom ship navigation environment with improved physics.

    Supports both pure-Python dynamics models (Fossen, Nomoto,
    UGentSurrogate, UAntwerpSurrogate) and the external FH Simulator engine
    (Model.FH_TABULAR). When dynamics_model == Model.FH_TABULAR, the
    environment is backed by an externally-constructed FH engine bridge
    (self._fh_sim / self.fh_sim) that manages the FH Simulator math model;
    interaction in that mode is limited to setting normalized rudder and
    thrust commands in [-1, 1] via the SimAPI Python layer. For all other
    dynamics models, self._fh_sim is None and the environment behaves
    exactly as before.

    PySimEnv itself is agnostic to which concrete bridge implementation is
    used: any object exposing the same duck-typed interface as FhSimBridge
    (ship_model, ship_interface, math_model, initial_vessel_state, reset(),
    dispose()) is accepted via the `fh_bridge` constructor argument. This
    currently includes:
        - FhSimBridge (fw.simulators.fh_sim_bridge): local FH math model.
        - SimSyncBridge (fw.simulators.sync_sim_bridge): realtime UDP
          connection to a running simSync instance.

    Attributes:
        _fh_sim: Externally-constructed FH engine bridge (FhSimBridge,
            SimSyncBridge, or any duck-typed equivalent) when
            dynamics_model == Model.FH_TABULAR; None otherwise.
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
            ua_surrogate_config: Optional[Dict] = None,
            enable_bank_collision_check: bool = True,
            enable_ukc_check: bool = True,
            #Reward configuration
            reward_name: str = "default",
            reward_weights: Optional[Dict[str, float]] = None,
            # FH Simulator configuration (only used when dynamics_model == Model.FH_TABULAR)
            scenario: Optional[str] = "DDShip_scenario1_nowindnocurrent",
            fh_bridge: Optional[object] = None,
    ) -> None:
        """
        Initialize the ship navigation environment.

        Args:
            render_mode: 'human' for rendering, otherwise None.
            time_step: simulation time step (s).
            max_steps: maximum steps per episode (for truncation).
            verbose: whether to print informational logs.
            start_pos: optional initial ship position (2-array).
            target_pos: optional target position. If given, the trajectory
                path is truncated to end exactly here, walked in whichever
                direction reaches it - forwards if target_pos lies further
                along the trajectory than start_pos, backwards (reversing
                the trajectory's point order) if target_pos lies earlier -
                shortened either way to reach it. If omitted, the
                trajectory's own last point is used as the target and the
                path is walked forwards, as before.
            dynamics_model: optional dynamics model to use. Pass Model.FH_TABULAR
                to drive the environment with the external FH Simulator engine
                instead of the pure-Python dynamics models.
            ship_model: optional ship model to use.
            wind: enable wind effects.
            current: enable current effects.
            ua_surrogate_config: optional configuration for the UA surrogate model.
            reward_name: name of reward profile (combination of components).
            reward_weights: optional per-component reward weights.
            scenario: FH Simulator scenario name. Retained for record-keeping
                (self.scenario) only; PySimEnv no longer uses it to construct
                a bridge. Only meaningful when dynamics_model == Model.FH_TABULAR.
            fh_bridge: Externally-constructed FH engine bridge instance
                (e.g. FhSimBridge or SimSyncBridge) to drive the environment
                when dynamics_model == Model.FH_TABULAR. Required in that
                case - PySimEnv does not construct a bridge itself. Ignored
                for all other dynamics models.

        Raises:
            RuntimeError: If dynamics_model == Model.FH_TABULAR and no
                fh_bridge was provided.
        """

        # ---------------------------------------------------------------
        # FH Simulator bridge setup (only when dynamics_model is FH_TABULAR).
        # Done first, before any other initialization, so a missing bridge
        # fails fast instead of after doing potentially expensive setup work.
        # PySimEnv is intentionally agnostic here: it accepts whatever
        # duck-typed bridge object the caller (env_factory) constructed -
        # it does not know about FhSimBridge or SimSyncBridge specifically.
        # ---------------------------------------------------------------
        self._fh_sim = None
        if dynamics_model == Model.FH_TABULAR:
            if fh_bridge is None:
                raise RuntimeError(
                    "Model.FH_TABULAR requires an externally constructed FH engine "
                    "bridge passed via `fh_bridge`. Construct one via "
                    "fw.simulators.fh_sim_bridge.FhSimBridge(...) (local math model) "
                    "or fw.simulators.sync_sim_bridge.SimSyncBridge(...) (realtime "
                    "simSync connection) before creating PySimEnv, or use a "
                    "non-FH_TABULAR dynamics_model for pure-Python simulation "
                    "without external dependencies."
                )
            self._fh_sim = fh_bridge

        super().__init__(render_mode)

        # Core parameters
        self.wind = bool(wind)
        self.current = bool(current)
        self.start_pos = start_pos
        self.overall_contour = overall_contour if overall_contour is not None else "env_Sche_no_scale.csv"
        self.depth_contour = depth_contour if depth_contour is not None else "env_Sche_250cm_no_scale.csv"
        self.trajectory = trajectory if trajectory is not None else "trajectory_points_no_scale.csv"
        self.ship_model: ShipModel = ship_model if ship_model is not None else ShipModel.MYZAKO_DEFAULT
        self.enable_bank_collision_check = bool(enable_bank_collision_check)
        self.enable_ukc_check = bool(enable_ukc_check)
        self.max_steps = int(max_steps)
        self.time_step = float(time_step)
        self.dynamics_model: Model = dynamics_model if dynamics_model is not None else Model.FOSSEN
        self.scenario = scenario
        self.verbose = bool(verbose) if verbose is not None else False
        self.previous_ship_pos = None
        self.previous_heading = None
        self.performed_action = None
        self.previous_action = None
        self._boundary_cache = None
        self.background = None
        self.dynamics = None
        
        self.ship_pos = None

        self._agent_state = None
        self.other_agent_states = {}

        #Reward configuration
        self.reward_name = str(reward_name)
        self.reward_weights = dict(reward_weights) if reward_weights is not None else None
        self.reward_calculator = Reward(
            reward_name=self.reward_name,
            reward_weights=self.reward_weights,
        )

        # RNG: environment-local RNG for reproducibility when seeded via reset()
        self._rng: np.random.Generator = np.random.default_rng()

        if start_pos is not None:
            self.initial_ship_pos = np.array(start_pos, dtype=np.float32)
        else:
            self.initial_ship_pos = np.array([5.0, 5.0], dtype=np.float32)

        self.randomization_scale = 1.0
        self.max_dist = np.sqrt(2) * float(MAX_GRID_POS)

        # Load environment static data and build checkpoints/polygons
        self._load_environment_data(target_pos)

        # State and control placeholders
        self.ship = create_ship(self.ship_model)
        self.phys_sim = self.create_simulator(ua_surrogate_config)

        self._initialize_simulation()

        # When driven by the FH Simulator, set the initial vessel state on the
        # bridge so the math model can be initialized with the correct
        # position, heading, and velocities on the first reset() call.
        if self._fh_sim is not None:
            self._fh_sim.initial_vessel_state = self.phys_sim.state

        # Gym spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._initialize_observation_space()

        # Rendering flag
        self.initialize_plots = True

    def neutral_action(self):
        """
        Return the neutral control action for the simulation.

        In this environment, the neutral action corresponds to zero control
        inputs (e.g., no thrust and no steering), resulting in passive system
        behavior such as drift.

        Returns:
            np.ndarray: A zero-valued action vector of shape (2,) with dtype
            np.float32.
        """

        return np.zeros(2, dtype=np.float32)

    def get_agent_state(self):
        """
        Retrieves the current state of this agent from the physics simulator.

        Returns:
            Any: The current state of the agent as provided by the underlying
            physics simulator (`self.phys_sim.state`). The exact structure
            depends on the simulator implementation.
        """

        return self.phys_sim.state

    def set_other_agents_states(self, agent_states):
        """
        Updates the stored states of all other agents in the environment.

        Filters out this agent's own state (identified by `self.env_name`)
        and updates `self.other_agent_states` with the remaining entries.

        Args:
            agent_states (dict): A mapping of agent identifiers (e.g., names or IDs)
                to their corresponding states.
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
        Initialize state vectors and convenience values.
        """

        self.ship_pos = self.initial_ship_pos.copy()
        self.previous_ship_pos = self.initial_ship_pos.copy()
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.checkpoint_index = 1

        self.heading_error = self._calculate_heading_error(self.los_guidance_heading(), self.phys_sim.state.heading)
        self.velocities = self.phys_sim.state.u
        self.previous_heading = 0.0
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
            terminated=False,
            truncated=False,
            x=self.initial_ship_pos[0],
            y=self.initial_ship_pos[1],
            heading=self.ship_heading,
            u=self.initial_speed, v=0.0, r=0.0))
        self._agent_state = self.phys_sim.state
        self.performed_action = np.zeros(2, dtype=np.float32)

    def create_simulator(self, ua_surrogate_config: Optional[Dict]=None):
        """
        Create and return a configured ship simulator.

        This factory method instantiates a `ShipSimulator` using the current
        ship model, dynamics configuration, environmental conditions, and time
        step stored on this object.

        Returns:
            ShipSimulator: A fully initialized simulator instance configured
                with the specified initial state and the current model parameters.
            ua_surrogate_config: Configuration for the UA surrogate model.
        """

        if self.dynamics_model == Model.FOSSEN:
            self.dynamics = Fossen(self.ship.specifications)
        elif self.dynamics_model == Model.NOMOTO:
            self.dynamics = Nomoto(self.ship.specifications)
        elif self.dynamics_model == Model.UGENT_SURROGATE:
            self.dynamics = UGentSurrogate(self.ship.specifications)
        elif self.dynamics_model == Model.UANTWERP:
            self.dynamics = UAntwerpSurrogate(self.ship.specifications, ua_surrogate_config)
        elif self.dynamics_model == Model.FH_TABULAR:
            # Explicit guard so any future code path that bypasses __init__
            # does not silently pass None to FhTabularModel.
            if self._fh_sim is None:
                raise RuntimeError(
                    "FH Simulator engine (_fh_sim) is not available. "
                    "Model.FH_TABULAR requires a working FH Simulator installation."
                )
            self.dynamics = FhTabularModel(self.ship.specifications, self._fh_sim)
        else:
            raise RuntimeError(f"Unknown dynamics model {self.dynamics_model} specified.")

        return ShipSimulator(self.ship.specifications, self.dynamics, self.time_step, self.wind, self.current)

    def __deepcopy__(self, memo):
        """
        Deep copy this instance, sharing the FH Simulator reference (if any).

        The FH Simulator bridge (_fh_sim) is not deep-copied because it holds
        non-picklable external resources (native library handles, COM
        objects). All other attributes are deep-copied normally. When
        _fh_sim is None (non-FH dynamics models), this is equivalent to a
        normal deep copy.

        Args:
            memo (dict): Memoization dict used by copy.deepcopy.

        Returns:
            PySimEnv: Deep copy with shared _fh_sim reference (including all
            other references to the same bridge in the object graph).
        """

        new_instance = self.__class__.__new__(self.__class__)
        if self._fh_sim is not None:
            memo[id(self._fh_sim)] = self._fh_sim

        for key, value in self.__dict__.items():
            if key == '_fh_sim':
                setattr(new_instance, key, value)   # Retain the original reference
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))  # Deep copy all other attributes

        return new_instance

    @property
    def fh_sim(self):
        """
        The FH engine bridge instance driving this environment.

        Returns:
            The externally-constructed bridge (FhSimBridge, SimSyncBridge,
            or any duck-typed equivalent) passed in via `fh_bridge` at
            construction time, or None if dynamics_model is not
            Model.FH_TABULAR.
        """

        return self._fh_sim

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
            """
            Load a 2-column (x, y) CSV file from this module's directory,
            validating its shape and raising clear, specific errors instead
            of letting a malformed or missing file fail deeper in the stack.

            Args:
                name: CSV filename (relative to this module's directory),
                    e.g. 'env_Sche_no_scale.csv'. Expected format: one
                    header row, followed by comma-separated (x, y) rows.

            Returns:
                np.ndarray: Float array of shape (N, 2).

            Raises:
                FileNotFoundError: If the file does not exist.
                RuntimeError: If the file exists but cannot be parsed.
                ValueError: If the parsed array is not a non-empty (N, 2) array.
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

        # Load obstacles and overall map shapes
        self.overall = load_csv_strict(self.overall_contour)
        self.obstacles = load_csv_strict(self.depth_contour)
        self.polygon_shape = Polygon(self.obstacles)

        # Load trajectory
        path = load_csv_strict(self.trajectory)

        # Resolve the target position up front: if the caller provided one,
        # it now dictates where the path ends (path is shortened to reach
        # it if needed). Otherwise fall back to the trajectory's own last
        # point, preserving the previous behavior.
        target_pos_arr = np.array(target_pos, dtype=np.float32) if target_pos is not None else None

        # Preprocess path: trim the start to begin after start_pos, and
        # trim the end so the path finishes exactly at target_pos_arr (or
        # keep the trajectory's own last point if no target_pos was given).
        path = self._reduce_path(path, self.initial_ship_pos, target_pos_arr)

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

        # Calculate direction of last path segment
        if len(self.checkpoints) < 2:
            raise ValueError(
                f"Path reduction from start_pos={self.initial_ship_pos.tolist()} resulted "
                f"in fewer than 2 checkpoints. Check that the CSV trajectory file "
                f"contains points ahead of the ship's start position, and that "
                f"CHECKPOINTS_DISTANCE ({CHECKPOINTS_DISTANCE}) is not larger than "
                f"the total path length."
            )

        p_last = self.checkpoints[-1]['pos'].copy()
        p_before_last = self.checkpoints[-2]['pos'].copy()
        self.direction_last_path = p_last - p_before_last

        # Navigation and counters
        # Safe handling when no checkpoints are present:
        if self.checkpoints:
            self.target_pos = self.checkpoints[-1]['pos']
        else:
            # If user provided target_pos, use it; otherwise fallback to initial_ship_pos
            if target_pos_arr is not None:
                self.target_pos = target_pos_arr
            else:
                self.target_pos = self.initial_ship_pos.copy()

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

            if self.ship_in_open_water(np.array([x, y])):
                samples.append((x, y))

        if len(samples) < nbr_samples:
            logger.warning("Only generated %d points out of requested %d "
                            "(polygon may be thin or max_tries reached).", len(samples), nbr_samples)

        return np.array(samples, dtype=np.float32)

    def get_keel_clearance(self, pos: Optional[np.ndarray] = None) -> float:
        """
        Get keel clearance.

        Args:
            pos (np.array, optional): The position for which to query the
                keel clearance. Ignored when driven by the FH Simulator,
                which reports clearance for its own internal state.

        Returns:
            float: Keel clearance in metres.

        Raises:
            RuntimeError: If dynamics_model is Model.FH_TABULAR and _fh_sim
                is unexpectedly unavailable.
        """

        if self._fh_sim is not None:
            # noinspection PyUnresolvedReferences
            return self._fh_sim.ship_interface.getKeelClearance()
        elif isinstance(self.dynamics, UAntwerpSurrogate):
            return self.dynamics.get_ukc(pos)
        else:
            return 0.0

    def ray_cast_distance(
            self,
            origin: Tuple[float, float],
            end_point: Tuple[float, float],
            max_distance: float = 200.0,
    ) -> float:
        """
        Return the distance from *origin* to the nearest river-boundary
        intersection along the ray `origin -> end_point`.

        The ray is defined by the two points; only the portion within
        [0, max_distance] from *origin* is considered.  If the ray does not
        intersect the polygon boundary within that range, *max_distance* is
        returned (i.e. no obstacle detected).

        Args:
            origin:       (x, y) start point of the ray (world frame, metres).
            end_point:    (x, y) point that defines the ray direction.  Its
                          distance from *origin* should be >= max_distance so
                          the full sensor range is covered.
            max_distance: Cut-off range in metres.  Hits beyond this distance
                          are ignored and the method returns max_distance.

        Returns:
            float: Distance to the nearest boundary intersection in metres,
                   clipped to [0, max_distance].
        """

        ray = LineString([origin, end_point])
        hit = self.polygon_shape.boundary.intersection(ray)

        if hit.is_empty:
            return max_distance

        origin_pt = Point(origin)

        if hit.geom_type == 'Point':
            dist = origin_pt.distance(hit)
        elif isinstance(hit, (MultiPoint, GeometryCollection)):
            # May include the origin itself (distance ≈ 0) when the ship sits
            # exactly on the boundary; skip those and take the nearest far hit.
            dists = [origin_pt.distance(g) for g in hit.geoms if not g.is_empty]
            far = [d for d in dists if d > 1e-6]
            dist = min(far) if far else 0.0
        else:
            # LineString / MultiLineString can appear when the ray is collinear
            # with a boundary segment; treat as touching the boundary.
            dist = origin_pt.distance(hit)

        return float(min(dist, max_distance))

    def ray_cast_distances_batch(
            self,
            origin: Tuple[float, float],
            end_points: np.ndarray,
            max_distance: float = 200.0,
    ) -> np.ndarray:
        """
        Vectorised version of :meth:`ray_cast_distance` for multiple rays.

        Fires *N* rays simultaneously from a common *origin* and returns all
        distances in a single call.

        Unlike a Python loop calling :meth:`ray_cast_distance` N times, this
        builds all N ray geometries as a single Shapely array and calls
        `shapely.intersection` / `shapely.distance` once each, which run
        in a compiled GEOS loop (Shapely >= 2.0) rather than once per ray in
        Python. The only remaining Python-level loop is over the rare subset
        of rays whose intersection is a MultiPoint/GeometryCollection (e.g.
        the ship sitting exactly on the boundary), which need the near-zero
        self-hit filtered out before taking the nearest distance.

        Each ray is processed against the pre-cached boundary object
        (`self._boundary_cache`), which is created on the first call and
        reused thereafter.

        Args:
            origin:       (x, y) start point shared by all rays (world frame).
            end_points:   Float array of shape (N, 2) — one end-point per ray.
                          Each row should be at least *max_distance* metres from
                          *origin* to cover the full sensor range.
            max_distance: Cut-off range in metres.

        Returns:
            np.ndarray: Float32 array of shape (N,) with distances clipped to
                        [0, max_distance].  Index *i* corresponds to the ray
                        toward `end_points[i]`.
        """

        # Cache the boundary object to avoid re-extracting it every call.
        if self._boundary_cache is None:
            self._boundary_cache = self.polygon_shape.boundary

        boundary = self._boundary_cache
        origin_pt = Point(origin)
        n = len(end_points)

        # Build all N ray LineStrings in one vectorized call instead of a
        # Python loop of LineString(...) constructors.
        origins_arr = np.tile(np.asarray(origin, dtype=np.float64), (n, 1))
        ends_arr = np.asarray(end_points, dtype=np.float64)
        coords = np.stack([origins_arr, ends_arr], axis=1)  # (n, 2, 2)
        rays = shapely.linestrings(coords)

        # Single vectorized intersection call against the (scalar) boundary,
        # broadcasting boundary against all n rays at the GEOS level.
        hits = np.asarray(shapely.intersection(boundary, rays))

        # Vectorised distance call. Works correctly for Point/LineString/
        # empty results; gives the WRONG answer for MultiPoint/
        # GeometryCollection hits (it would return the distance to the
        # nearest point in the set, including a spurious near-zero self-hit
        # when the origin sits on the boundary) so those are corrected below.
        dists = shapely.distance(origin_pt, hits)

        # shapely.distance on an empty geometry returns nan -> no obstacle
        # detected within range, so treat as max_distance.
        dists = np.where(np.isnan(dists), max_distance, dists)

        # Fix up the rare MultiPoint / GeometryCollection cases individually
        # (mirrors the scalar fallback in ray_cast_distance).
        geom_types = shapely.get_type_id(hits)
        # GEOS type ids: 4 = MultiPoint, 7 = GeometryCollection
        mixed_mask = np.isin(geom_types, (4, 7))
        if mixed_mask.any():
            for i in np.flatnonzero(mixed_mask):
                hit = hits[i]
                assert isinstance(hit, (MultiPoint, GeometryCollection)), (
                    f"GEOS type id {geom_types[i].tolist()} should imply MultiPoint/GeometryCollection"
                )
                sub_dists = [origin_pt.distance(g) for g in hit.geoms if not g.is_empty]
                far = [d for d in sub_dists if d > 1e-6]
                dists[i] = min(far) if far else 0.0

        distances = np.minimum(dists, max_distance).astype(np.float32)
        return distances

    def _has_enough_keel_clearance(self, depth_threshold: float = 0.5) -> bool:
        """
        Check whether the vessel has sufficient keel clearance.

        When driven by the FH Simulator, queries the bridge's ship interface
        directly. Otherwise, this is a placeholder returning True by default
        to preserve existing behavior for the pure-Python dynamics models.

        Args:
            depth_threshold (float): Minimum acceptable keel clearance in
                metres. Only used in FH Simulator mode. Defaults to 0.5 m.

        Returns:
            bool: True if keel clearance >= depth_threshold (FH mode), or
            True unconditionally (pure-Python dynamics models).
        """

        if self._fh_sim is not None:
            # noinspection PyUnresolvedReferences
            return self._fh_sim.ship_interface.getKeelClearance() >= depth_threshold

        return True

    def ship_in_open_water(self,
                           ship_position: np.ndarray,
                           enable_bank_collision_check: bool = True,
                           enable_ukc_check: bool = True) -> bool:
        """
        Check if the ship is inside the polygon bounds and has keel clearance.

        Args:
            ship_position: (x, y) coordinates
            enable_bank_collision_check: enables bank collision checking
            enable_ukc_check: enables ukc checking

        Returns:
            bool: True if ship is inside polygon and keel clearance check passes
        """

        if enable_bank_collision_check:
            x, y = ship_position[0], ship_position[1]
            min_x, min_y, max_x, max_y = self.polygon_shape.bounds
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                return False

            contains = self.polygon_shape.contains(Point(ship_position))
        else:
            contains = True

        deep_enough = self._has_enough_keel_clearance() if enable_ukc_check else True
        return bool(contains and deep_enough)

    @staticmethod
    def _reduce_path(path: np.ndarray, start_pos: np.ndarray, target_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reduce the path to run from start_pos to target_pos, in whichever
        direction along the trajectory's stored point order gets you there.

        The trajectory's points are stored in one fixed order (e.g. south
        to north). The path can be walked forwards - start_pos comes
        before target_pos in that order - or backwards, by swapping which
        end is start_pos and which is target_pos (e.g. sailing
        north-to-south along a trajectory that was recorded
        south-to-north). Either way, the returned path always begins just
        after start_pos and ends exactly at target_pos, shortened as
        needed to reach it - target_pos dictates where the path ends,
        rather than being inferred from the trajectory's stored last
        point.

        If target_pos is None, the trajectory's own last point is used as
        the end and the path is walked forwards (previous default
        behavior).

        Args:
            path (np.ndarray): Full trajectory as (N, 2) array.
            start_pos (np.ndarray): Ship's starting position.
            target_pos (Optional[np.ndarray]): Desired end position of the
                path. If None, the trajectory's own last point is used.

        Returns:
            np.ndarray: Reduced path beginning just after start_pos and
            ending at target_pos - in forward or reverse order, whichever
            direction target_pos lies in - or ending at the trajectory's
            original last point if target_pos is None.

        Raises:
            ValueError: If the reduction results in an empty path, or if
                start_pos and target_pos resolve to the same point on the
                trajectory.
        """

        if len(path) == 0:
            raise ValueError("Path is empty.")

        # Find index of the point on the path closest to start_pos
        start_distances = np.linalg.norm(path - start_pos, axis=1)
        start_index = int(np.argmin(start_distances))

        if target_pos is None:
            # Slice the path from following point to the end (forward, as before)
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

        # Find index of the point on the path closest to target_pos, over
        # the full original trajectory (direction-agnostic).
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
            # Forward traversal: target_pos lies further along the
            # trajectory's stored order than start_pos.
            truncated_path = path[start_index + 1: target_index + 1].copy()
        else:
            # Backward traversal: target_pos lies earlier in the
            # trajectory's stored order than start_pos, so walk the
            # trajectory in reverse - e.g. sailing north-to-south along a
            # trajectory recorded south-to-north. path[target_index:start_index]
            # is everything strictly between (and including) target_index up
            # to just before start_index, in stored order; reversing it gives
            # a path starting just before start_index and ending at target_index.
            truncated_path = path[target_index:start_index][::-1].copy()

        if len(truncated_path) == 0:
            # Defensive: shouldn't happen given target_index != start_index,
            # but guard against it explicitly rather than let it fail deeper.
            raise ValueError(
                f"_reduce_path produced an empty path between start_pos "
                f"{np.asarray(start_pos).tolist()} (index {start_index}) and target_pos "
                f"{target_pos.tolist()} (index {target_index})."
            )

        # Force the final point to be exactly target_pos, rather than
        # merely the closest existing trajectory point to it - this is
        # what allows target_pos to shorten the path.
        truncated_path[-1] = target_pos.astype(truncated_path.dtype)

        return truncated_path

    @staticmethod
    def _distance_from_point_to_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        Calculate perpendicular distance from point to line segment.

        Args:
            point: Point coordinates (x, y).
            line_start: Line segment start (x, y).
            line_end: Line segment end (x, y).

        Returns:
            float: Perpendicular distance from point to line segment.
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
        Normalizes an angle to the range [-π, π).

        This method maps any input angle (in radians) to its equivalent value
        within the canonical range of -π (inclusive) to π (exclusive).

        Args:
            angle: An angle in radians. Can be any finite floating-point value.

        Returns:
            The equivalent angle in radians within the range [-π, π).
        """

        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _safe_heading_from_vector(vec) -> float:
        """
        Return heading angle for vector, or 0.0 if vector is zero.
        """

        if np.linalg.norm(vec) > 0.0:
            return float(np.arctan2(vec[1], vec[0]))
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
                -1.0,   # Heading error to current checkpoint
                -1.0,   # Heading error to next checkpoint
                -1.0,   # Heading error, parallel to current checkpoint segment
                -1.0,   # Heading error, parallel to next checkpoint segment
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
                1.0,    # Heading error to current checkpoint
                1.0,    # Heading error to next checkpoint
                1.0,    # Heading error, parallel to current checkpoint segment
                1.0,    # Heading error, parallel to next checkpoint segment
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

        return gym.spaces.Box(low=base_low, high=base_high, dtype=np.float32)

    def env_specific_reset(self):
        """
        Hook for environment-specific reset operations, called by reset()
        before _initialize_simulation().

        When driven by the FH Simulator, resets the FH Simulator to its
        initial state (rewinding the math model to t=0.25s, the earliest
        supported rewind point). Otherwise, defers to the base class hook
        to preserve prior (non-FH) behavior unchanged.

        Raises:
            RuntimeError: If dynamics_model is Model.FH_TABULAR and _fh_sim
                is unexpectedly unavailable.
        """

        if self._fh_sim is not None:
            # noinspection PyUnresolvedReferences
            self._fh_sim.reset()
            return

        super().env_specific_reset()

    def randomize(self, randomization_scale: Optional[float] = None) -> None:
        """
        Randomize the ship's initial position within specified bounds.

        Intentionally a no-op when driven by the FH Simulator: the FH
        Simulator always rewinds to the fixed initial state baked into the
        exercise file at math model initialization time. The initial
        position, heading, and velocities are set once via
        FhSimBridge.initial_vessel_state and cannot be changed between
        episodes without reinitializing the math model from scratch. If
        per-episode starting position variation is needed in FH mode, it
        must be implemented by loading a different exercise file or by
        extending this method to call dispose() + reset() with a new
        initial_vessel_state.

        Args:
            randomization_scale: Maximum absolute value for position
                randomization. If None, uses the class's default scale.
                Ignored when driven by the FH Simulator.

        Raises:
            ValueError: If randomization_scale is not positive (non-FH mode).
        """

        if self._fh_sim is not None:
            return  # Intentional no-op when driven by the FH Simulator

        if randomization_scale is not None:
            if randomization_scale <= 0:
                raise ValueError("randomization_scale must be positive")

            self.randomization_scale = randomization_scale

        ran_gen = np.random.default_rng()

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
        current_chkp_idx = self.checkpoint_index
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

        # LOS guidance error
        los_heading_error = self._calculate_heading_error(
            self.los_guidance_heading(),
            self.phys_sim.state.heading,
            0.0
        )

        # Build observation
        obs = np.hstack([
            norm_velocities,                        # Normalized velocities
            [2 * norm_distance - 1],                # Distance to current checkpoint
            [2 * norm_next_distance - 1],           # Distance to next checkpoint
            [norm_cross_error],                     # Normalized cross-track error
            [heading_error / np.pi],                # Heading error to current checkpoint
            [heading_error2 / np.pi],               # Heading error to next checkpoint
            [heading_error_parallel / np.pi],       # Heading error, parallel to current segment
            [heading_error_parallel2 / np.pi],      # Heading error, parallel to next segment
            self.performed_action[0],               # Previous rudder angle
            los_heading_error / np.pi,              # Optimal LOS guidance
        ])

        # Add environmental observations if enabled
        if self.wind:
            obs = np.hstack([obs, self.phys_sim.environment.wind_direction * self.phys_sim.environment.wind_strength])

        if self.current:
            obs = np.hstack([obs, self.phys_sim.environment.current_direction * self.phys_sim.environment.current_strength])

        return obs.astype(np.float32)

    # -----------------------
    # Dynamics update & step
    # -----------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment timestep.

        Args:
            action: Array-like with [rudder, thrust] commands in [-1, 1]

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(
                f"Expected action shape (2,), got {action.shape}. "
                f"Action values: {action.tolist()}. Use [-1,1] for [rudder, thrust]."
            )

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            # Save previous values
            self.previous_action = self.performed_action.copy()
            self.previous_ship_pos = self.ship_pos.copy()
            self.previous_heading = self.phys_sim.state.heading

            # Update state vector
            self.phys_sim.step(action, self.enable_smoothing, self.enable_rudder_angle_scaling)
            self._agent_state = self.phys_sim.state
            self.ship_pos = np.array([self.phys_sim.state.x, self.phys_sim.state.y], dtype=np.float32)
            self.performed_action = self.phys_sim.actuators.current_actuator_state.copy()
            self.step_count += 1

            # Reward and termination check
            reward, terminated = self._calculate_reward()

            # Set truncated True when episode exceeded max_steps but not terminated by failure or success.
            truncated = False
            if self.step_count >= self.max_steps and not terminated:
                # reward = -250
                truncated = True

            # Update ship state
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

        return self._get_obs(), reward, terminated | truncated, False, info

    # -------------------
    # Reward calculation
    # -------------------
    def _calculate_reward(self) -> Tuple[float, bool]:
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

        # Cross-track errors
        cross_track_error = self._distance_from_point_to_line(self.ship_pos, prev_checkpoint_pos, current_checkpoint_pos)
        self.cross_error = cross_track_error
        self.heading_error = self._calculate_heading_error(self.los_guidance_heading(), self.phys_sim.state.heading)
        self.velocities = self.phys_sim.state.u

        # Keel clearance
        self.ukc = self.get_keel_clearance(self.ship_pos)

        # Ship information
        rudder_change = abs(self.performed_action[0] - self.previous_action[0])
        max_rudder_change = self.ship.specifications.max_rudder_rate * self.time_step / self.ship.specifications.max_rudder_angle  # rad/s * s * action/rad
        heading_error = self._calculate_heading_error(self.los_guidance_heading(), self.phys_sim.state.heading)
        distance_traveled = abs(np.linalg.norm(self.ship_pos - self.previous_ship_pos))
        velocity = distance_traveled / self.time_step
        target_velocity = TARGET_SURGE_VELOCITY_MS
        thrust_change = abs(self.performed_action[1] - self.previous_action[1])
        max_thrust_change = self.ship.specifications.max_thrust_rate * self.time_step / self.ship.specifications.max_thrust 

        reward = self.reward_calculator.calculate_reward(
            velocity=velocity,
            target_velocity=target_velocity,
            heading_error=heading_error,
            cross_track_error=cross_track_error,
            rudder_change=rudder_change,
            max_rudder_change=max_rudder_change,
            rudder_angle=float(self.performed_action[0]),
            thrust_change=thrust_change,
            max_thrust_change=max_thrust_change,
        )
        
        num_non_zero = sum(1 for w in self.reward_calculator.reward_weights.values() if w != 0)
        reward = 1.0 + (reward / num_non_zero if num_non_zero > 0 else 0.0)
        
        # Checkpoint handling (may increment checkpoint_index)
        checkpoint_distance = float(np.linalg.norm(self.ship_pos - current_checkpoint["pos"]))

        # Reached checkpoint circle
        if self.segment_point_distance(self.previous_ship_pos, self.ship_pos, current_checkpoint["pos"]) <= current_checkpoint['radius']:
            if self.checkpoint_index == len(self.checkpoints) - 1:
                logger.info("%s - target REACHED at distance %.2f", self.env_name, checkpoint_distance)
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

        # Early termination conditions
        termination_conditions = [
            cross_track_error > CROSS_TRACK_TERMINATION_MULTIPLIER * CHECKPOINTS_DISTANCE,
            not self.ship_in_open_water(self.ship_pos, self.enable_bank_collision_check, self.enable_ukc_check),
            heading_change > HEADING_CHANGE_THRESHOLD,
        ]

        if any(termination_conditions):
            # reward = -250
            done = True

        # Normal completion (reached or passed all checkpoints)
        if self.checkpoint_index >= len(self.checkpoints):
            done = True

        return reward, done

    @staticmethod
    def segment_point_distance(p1, p2, c) -> float:
        """
        Compute the shortest distance from a point to a line segment.

        Args:
            p1: Start point of the segment as a 2D vector.
            p2: End point of the segment as a 2D vector.
            c: Query point as a 2D vector.

        Returns:
            The Euclidean distance from `c` to the segment `p1`-`p2`.
        """

        # project c onto segment p1-p2
        if np.allclose(p1, p2):
            return float(np.linalg.norm(c - p1))

        v = p2 - p1
        w = c - p1

        t = float(np.dot(w, v) / np.dot(v, v))
        t = min(1.0, max(0.0, t))

        closest = p1 + t * v
        return float(np.linalg.norm(c - closest))

    @staticmethod
    def _segments_intersect(prev_point: np.ndarray, curr_point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> bool:
        """
        Return True if segment prev — curr intersects the infinite line through
        line_seg_start — line_seg_end.
        """

        def _orientation(a, b, c):
            """
            Signed area (2D cross product):
            > 0 : c is left of line ab
            < 0 : c is right of line ab
            = 0 : co-linear
            """

            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        o1 = _orientation(line_start, line_end, prev_point)
        o2 = _orientation(line_start, line_end, curr_point)

        # If either point is on the line -> intersection
        if abs(o1) <= EPSILON or abs(o2) <= EPSILON:
            return True

        # If signs differ -> segment crosses the line
        return (o1 > 0 > o2) or (o1 < 0 < o2)

    def _is_near_perpendicular_line(self, checkpoint: dict) -> bool:
        """
        Check if ship is close enough to checkpoint's perpendicular line.
        """

        line_start, line_end = checkpoint['perpendicular_line']
        return self._segments_intersect(self.previous_ship_pos, self.ship_pos, line_start, line_end)

    def los_guidance_heading(self):
        """
        Compute the desired LOS heading toward the current path segment.

        Based on the LOS guidance law from Changyuan Chen's PhD thesis
        "Numerical and Experimental Study on Ship Motion Control Systems
        in Shallow Water" at Flanders Hydraulics.

        Returns:
            float: Desired heading in radians.
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
            # if none exists, create a virtual one behind the ship
            p_k_min_1 = p_k - (p_k_plus_1 - p_k)
        x_k_min_1, y_k_min_1 = p_k_min_1[:]

        # Find the orthogonal projection A(x_A, y_A) of the ship on the path from the previous to the next checkpoint
        seg_sq = (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2
        if seg_sq < EPSILON:
            # Coincident checkpoints: fall back to pointing directly at p_k_plus_1
            return math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)

        # Orthogonal projection of ship onto segment p_k_min_1 -> p_k
        proj_scalar = (
                          (x - x_k_min_1) * (x_k - x_k_min_1) +
                          (y - y_k_min_1) * (y_k - y_k_min_1)
                      ) / seg_sq

        x_a = x_k_min_1 + proj_scalar * (x_k - x_k_min_1)
        y_a = y_k_min_1 + proj_scalar * (y_k - y_k_min_1)

        # Calculate the look ahead distance L_A
        c = 0.5   # Coefficient for LOS look ahead horizon (default is 0.5 according to FH PhD thesis p. 148)
        delta_y = math.sqrt((x - x_a) ** 2 + (y - y_a) ** 2)
        l_pp = 0.95 * self.ship.specifications.length
        l_a_sq = (delta_y + c * l_pp) ** 2 - delta_y ** 2
        l_a = math.sqrt(max(l_a_sq, 0.0))   # clamp against floating-point negatives

        # Calculate the LOS position Ploss(x_loss, y_loss)
        seg_len = math.sqrt(seg_sq)
        x_loss = x_a + ((x_k - x_k_min_1) / seg_len) * l_a
        y_loss = y_a + ((y_k - y_k_min_1) / seg_len) * l_a

        # Selection of current waypoint
        # Calculate radius of the current checkpoint's acceptance circle Rk
        w_los = 0.5   # Tunable parameter for acceptance circle (default is 0.5 according to FH PhD thesis p. 148)
        r_k = w_los * l_pp

        heading_p_k_min_1_p_k = self._safe_heading_from_vector(p_k - p_k_min_1)
        heading_p_k_p_k_plus_1 = self._safe_heading_from_vector(p_k_plus_1 - p_k)
        heading_difference = self.normalize_angle(
            heading_p_k_min_1_p_k - heading_p_k_p_k_plus_1
        )
        collinear = abs(heading_difference) < (1.0 / 57.0)   # Are the previous, current, and next checkpoint collinear

        if collinear:
            dist_a_pk = math.sqrt((x_a - x_k) ** 2.0 + (y_a - y_k) ** 2.0)   # Distance to the current checkpoint
            if dist_a_pk > r_k:  # |A Pk| > Rk
                # LOS angle (desired course angle) in radians
                los_angle = math.atan2(y_loss - y, x_loss - x)
            else:  # |A Pk| < Rk
                # LOS position Ploss is replaced by current waypoint Pk+1
                los_angle = math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)
        else:
            dist_p_pk = math.sqrt((x - x_k) ** 2 + (y - y_k) ** 2)   # Distance to the current checkpoint
            if dist_p_pk > 2 * r_k:  # |P Pk| > 2*Rk
                if l_a < dist_p_pk:  # LA < |P Pk|
                    los_angle = math.atan2(y_loss - y, x_loss - x)
                else:  # LA > |P Pk|
                    # Change look ahead distance from LA to L'A
                    next_seg_sq = (x_k_plus_1 - x_k) ** 2 + (y_k_plus_1 - y_k) ** 2
                    if next_seg_sq < EPSILON:
                        return math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)

                    # Calculate new heading point P'loss
                    next_seg_len = math.sqrt(next_seg_sq)
                    l_accent_a = l_a - math.sqrt(
                        (x_k - x_a) ** 2 + (y_k - y_a) ** 2
                    )
                    x_accent_loss = x_k + ((x_k_plus_1 - x_k) / next_seg_len) * l_accent_a
                    y_accent_loss = y_k + ((y_k_plus_1 - y_k) / next_seg_len) * l_accent_a

                    # LOS angle (desired course angle) in radians
                    los_angle = math.atan2(y_accent_loss - y, x_accent_loss - x)
            else:  # |P Pk| < 2*Rk
                # The checkpoint Pk becomes replaced by Pk+1 (next checkpoint becomes current checkpoint), same for Pk-1
                x_k_min_1, y_k_min_1 = x_k, y_k
                x_k, y_k = x_k_plus_1, y_k_plus_1

                # Find the orthogonal projection A(x_A, y_A) of the ship on the path from the previous to the next checkpoint
                seg_sq2 = (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2
                if seg_sq2 < EPSILON:
                    return math.atan2(y_k - y, x_k - x)

                # Calculate the look ahead distance L_A
                proj_scalar2 = (
                                   (x - x_k_min_1) * (x_k - x_k_min_1) +
                                   (y - y_k_min_1) * (y_k - y_k_min_1)
                               ) / seg_sq2

                x_a = x_k_min_1 + proj_scalar2 * (x_k - x_k_min_1)
                y_a = y_k_min_1 + proj_scalar2 * (y_k - y_k_min_1)

                delta_y = math.sqrt((x - x_a) ** 2 + (y - y_a) ** 2)   # Cross-track error
                l_a_sq = (delta_y + c * l_pp) ** 2 - delta_y ** 2
                l_a = math.sqrt(max(l_a_sq, 0.0))

                # Calculate the LOS position Ploss(x_loss, y_loss)
                seg_len2 = math.sqrt(seg_sq2)
                x_loss = x_a + ((x_k - x_k_min_1) / seg_len2) * l_a
                y_loss = y_a + ((y_k - y_k_min_1) / seg_len2) * l_a
                los_angle = math.atan2(y_loss - y, x_loss - x)

        return los_angle

    @staticmethod
    def _logarithmic_scaling(value: float, value_range: tuple, half_value: float = 10):
        """
        Maps a value to [0, 1] using a logarithmic curve.

        The curve is parameterized by `half_value`, which controls the midpoint:
        the input value that should map to 0.5 on the output scale. A lower
        half_value produces a curve that rises quickly then flattens; a higher
        half_value produces a more gradual rise.

        Args:
            value: The input value to scale.
            value_range: A (min, max) tuple. min maps to 0, max maps to 1.
            half_value: The input value that should map to 0.5 on the output scale.
        """

        # min_value -> 0, max_value -> 1 after scaling
        min_value, max_value = value_range

        # Clamp to range
        clamped = max(min_value, min(value, max_value))
        if clamped != value:
            logger.debug(
                "Value %s clamped to [%s, %s]", value, min_value, max_value
            )

        # Normalize value and half_value to [0, 1]
        x = (clamped - min_value) / (max_value - min_value)
        m = (half_value - min_value) / (max_value - min_value)

        if math.isclose(m, 0.5, abs_tol=1e-9):
            return x

        # Compute k: curvature parameter derived from the desired midpoint
        k = (1 - 2 * m) / (m ** 2)

        # Apply log curve: log1p(k*x) / log1p(k) maps [0,1] -> [0,1] with f(m) = 0.5
        return math.log1p(k * x) / math.log1p(k)

    # -----------------------
    # Rendering
    # -----------------------
    def _initialize_rendering(self) -> None:
        """
        Set up the rendering elements for visualization.
        """

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
        """
        Draw static features: path, checkpoints, obstacles, polygons.
        """

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
            # noinspection PyUnresolvedReferences
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def render(self) -> None:
        """
        Render the environment and visualize the ship's movement.
        """

        if self.render_mode != 'human':
            return

        # Initialize plotting objects once
        if self.initialize_plots and not hasattr(self, 'ship_plot'):
            self._initialize_rendering()
            self._draw_fixed_landmarks()
            self.fig.canvas.draw()
            # noinspection PyUnresolvedReferences
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig.canvas.mpl_connect('draw_event', self._on_draw)
            plt.show(block=False)

        canvas = self.fig.canvas

        # --- Helper: safe execution without aborting the simulation ---
        def _safe(callable_obj):
            """
            Call `callable_obj()`, swallowing rendering-related errors.

            Used to keep a single flaky matplotlib/Tk draw call from
            aborting the whole simulation.

            Args:
                callable_obj: A zero-argument callable to invoke; its
                    return value is discarded.
            """

            try:
                callable_obj()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass

        # Restore background for blit optimization when available
        if hasattr(self, 'background'):
            # noinspection PyUnresolvedReferences
            _safe(lambda: canvas.restore_region(self.background))

        # Update ship and heading graphics
        heading_line_length = 30.0
        self.ship_plot.set_data(self.ship_pos[0:1], self.ship_pos[1:2])
        heading_x = self.ship_pos[0] + np.cos(self.phys_sim.state.heading) * heading_line_length
        heading_y = self.ship_pos[1] + np.sin(self.phys_sim.state.heading) * heading_line_length
        self.heading_line.set_data([self.ship_pos[0], heading_x], [self.ship_pos[1], heading_y])

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
        """
        Close the environment.
        """

        if self.render_mode == 'human' and hasattr(self, 'fig'):
            try:
                plt.close(self.fig)
            except (RuntimeError, AttributeError):
                pass
