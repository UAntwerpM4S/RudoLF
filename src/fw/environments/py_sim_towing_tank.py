import os
import copy
import math
import shapely
import logging
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

matplotlib.rcParams['figure.raise_window'] = False

from shapely import LineString
from typing import Dict, List, Optional, Tuple
from fw.environments.ray_features import RAY_ANGLES_DEG, RAY_MAX_DISTANCE, ray_world_angles
from shapely.geometry import Point, Polygon, MultiPoint, GeometryCollection
from fw.environments.tools import create_checkpoints_from_simple_path
from fw.ships.ship_factory import create_ship, require_full_scale_ship, require_hull_coefficients, ShipModel
from fw.ships.vessel_state import VesselState
from fw.environments.base_env import BaseEnv
from fw.reward.reward import Reward

from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate
from fw.dynamics.fh_tabular_model import FhTabularModel
from fw.simulators.ship_simulator import ShipSimulator
from fw.dynamics.ugent_surrogate import UGentSurrogate
from fw.dynamics.ugent_b01_surrogate import UGentB01Surrogate
from fw.dynamics.dynamics_model import Model
from fw.dynamics.fossen import Fossen
from fw.dynamics.nomoto import Nomoto

logger = logging.getLogger(__name__)


# Grid limits
MAX_CHECKPOINTS_DISTANCE = 2000
CHECKPOINTS_DISTANCE = 350
MIN_GRID_POS = -10
MAX_GRID_POS = 150

# Reward / shaping
SUCCESS_REWARD = 50.0
CHECKPOINT_AREA_SIZE = 1.0
TARGET_AREA_SIZE = 1.0
TARGET_SURGE_VELOCITY_MS = 0.8  # m/s, target for velocity reward shaping

# Rendering constants
MAX_FIG_WIDTH = 1200
MAX_FIG_HEIGHT = 900
DPI = 100

# Guidance and termination
PERPENDICULAR_LINE_LENGTH = 10.0
HEADING_CHANGE_THRESHOLD = np.pi / 2.0
CROSS_TRACK_TERMINATION_MULTIPLIER = 2.0

# Map cut-off edges per depth-contour file: the contour is clipped at the map edge
# while FH has water beyond it, so these are not real banks.
MAP_CUTOFF_LINES: Dict[str, Tuple[Tuple[Tuple[float, float], ...], ...]] = {
    "env_Sche_250cm_no_scale.csv": (
        ((-10790.0, 9620.0), (-10770.0, 8760.0)),    # west end of the main channel
        ((-10810.0, 11880.0), (-10710.0, 12960.0)),  # north-west branch tip
        ((7992.0, -4640.0), (7992.0, -5320.0)),      # east (upstream) end
    ),
}
CUTOFF_LINE_TOLERANCE = 100.0  # m
EPSILON = 1e-9

# Port/starboard beam rays (ray_features layout), used for UGent bank distances.
_PORT_BEAM_RAY_IDX = int(np.flatnonzero(RAY_ANGLES_DEG == 270)[0])
_STARBOARD_BEAM_RAY_IDX = int(np.flatnonzero(RAY_ANGLES_DEG == 90)[0])


def calculate_perpendicular_lines(
        checkpoints: List[dict],
        line_length: float = PERPENDICULAR_LINE_LENGTH,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Perpendicular line, centred on each checkpoint, to the smoothed path tangent.

    Args:
        checkpoints: Checkpoint dicts with a 'pos' entry.
        line_length: Line length [m].

    Returns:
        (start_point, end_point) arrays per checkpoint.
    """

    def smooth_tangent(check_points: List[dict], index: int) -> np.ndarray:
        """
        Unit tangent at a checkpoint, averaging the neighboring segments (x-axis for a single checkpoint).
        """

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
        return (tangent / norm if norm != 0.0 else tangent).astype(np.float32)

    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, cp in enumerate(checkpoints):
        pos = np.asarray(cp['pos'], dtype=np.float32)
        smth_tangent = smooth_tangent(checkpoints, i)
        perpendicular = np.array([-smth_tangent[1], smth_tangent[0]], dtype=np.float32)
        offset = perpendicular * (line_length / 2.0)
        lines.append((pos + offset, pos - offset))
    return lines



class PySimTowingTankEnv(BaseEnv):
    """
    Ship navigation environment on the Western Scheldt map (full scale).

    Supports the Python dynamics models (Fossen, Nomoto, UGentSurrogate,
    UGentB01Surrogate, UAntwerpSurrogate) and the external FH Simulator
    (Model.FH_TABULAR). For FH_TABULAR an externally constructed bridge is passed
    as `fh_bridge` and stored as self._fh_sim; only normalized rudder and thrust
    commands in [-1, 1] are set through it. Any object with FhSimBridge's duck-typed
    interface (ship_model, ship_interface, math_model, initial_vessel_state, reset(),
    dispose()) works, e.g. FhSimBridge (local math model) or SimSyncBridge (realtime
    UDP link to simSync). For other models self._fh_sim is None.
    """

    SUPPORTS_MODEL_SCALE_SHIPS: bool = False
    """
    False: the env runs require_full_scale_ship at construction. That check is
    disabled and accepts every preset, also ShipModel.B01.
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
            ugent_surrogate_config: Optional[Dict] = None,
            ugent_b01_surrogate_config: Optional[Dict] = None,
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
        Args:
            render_mode: 'human' for rendering, otherwise None.
            time_step: Simulation time step [s].
            max_steps: Maximum steps per episode (truncation).
            verbose: Print informational logs.
            start_pos: Initial ship position (2-array).
            target_pos: Target position. The trajectory is cut to end here and walked
                forwards or backwards, whichever reaches it from start_pos. If omitted,
                the trajectory's last point is the target.
            overall_contour, depth_contour, trajectory: Map CSV file names (defaults:
                env_Sche_no_scale.csv, env_Sche_250cm_no_scale.csv,
                trajectory_points_no_scale.csv).
            dynamics_model: Dynamics model; Model.FH_TABULAR uses the FH Simulator.
            ship_model: Ship model (must be full scale).
            wind: Enable wind effects.
            current: Enable current effects.
            ua_surrogate_config: Model.UANTWERP configuration, e.g.
                {"preset": "tripoli_confined"} (see fw.surrogate.presets).
            ugent_surrogate_config: UGent surrogate configuration, e.g. {"ukc_percent": 363}.
            ugent_b01_surrogate_config: UGent B01 configuration, e.g. {"ukc_percent": 379}.
            enable_bank_collision_check: Terminate on bank collision.
            enable_ukc_check: Terminate on insufficient UKC.
            reward_name: Reward profile name.
            reward_weights: Optional per-component reward weights.
            scenario: FH scenario name, stored for record-keeping only.
            fh_bridge: FH engine bridge, required for Model.FH_TABULAR and ignored
                otherwise.

        Raises:
            RuntimeError: Model.FH_TABULAR without fh_bridge.
        """

        # FH bridge check first, so a missing bridge fails before expensive setup.
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
        if not self.SUPPORTS_MODEL_SCALE_SHIPS:
            require_full_scale_ship(self.ship_model, type(self).__name__)
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
        self._bank_lines_cache = None
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
        self.phys_sim = self.create_simulator(ua_surrogate_config, ugent_surrogate_config, ugent_b01_surrogate_config)

        self._initialize_simulation()

        # The FH bridge initializes its math model from this state on the first reset().
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
        Neutral action: zero rudder and thrust, float32 array of shape (2,).
        """

        return np.zeros(2, dtype=np.float32)

    def get_agent_state(self):
        """
        Current state of this agent (`self.phys_sim.state`).
        """

        return self.phys_sim.state

    def set_other_agents_states(self, agent_states):
        """
        Update `other_agent_states` from a name -> state mapping, skipping this agent's own entry.
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
    def _start_thrust(self, state: VesselState) -> float:
        """
        Thrust the actuators start the episode with: initial_thrust if set, else
        the thrust that holds the start speed (DynamicsBase.steady_thrust), else
        min_thrust. A start speed no thrust can hold is logged once.
        """

        if self.initial_thrust is not None:
            return self.initial_thrust
        specs = self.ship.specifications
        if state.u <= 0.0:
            return specs.min_thrust
        thrust = self.dynamics.steady_thrust(state, specs.min_thrust, specs.max_thrust)
        if thrust is not None:
            return thrust
        if self.dynamics.derives_steady_thrust and not getattr(self, "_warned_start_thrust", False):
            logger.warning(
                "%s: no thrust in [%.3f, %.3f] holds the initial speed %.3f m/s with %s; "
                "starting at min_thrust. Further occurrences are not logged.",
                type(self).__name__, specs.min_thrust, specs.max_thrust, state.u,
                type(self.dynamics).__name__)
            self._warned_start_thrust = True
        return specs.min_thrust

    def _initialize_simulation(self) -> None:
        """
        Reset the episode state and place the ship at the start, heading toward the first checkpoint.
        """

        self.ship_pos = self.initial_ship_pos.copy()
        self.previous_ship_pos = self.initial_ship_pos.copy()
        self.checkpoint_index = 1

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

        start_state = VesselState(
            terminated=False,
            truncated=False,
            x=self.initial_ship_pos[0],
            y=self.initial_ship_pos[1],
            heading=self.ship_heading,
            u=self.initial_speed, v=0.0, r=0.0)
        self.phys_sim.reset(start_state, initial_thrust=self._start_thrust(start_state))
        self._agent_state = self.phys_sim.state
        # Rudder at zero, thrust at the start thrust.
        self.performed_action = self.phys_sim.actuators.current_actuator_state.copy()
        self.previous_action = self.performed_action.copy()

        # Must follow phys_sim.reset(), else these describe the previous episode.
        self.heading_error = self._calculate_heading_error(self.los_guidance_heading(), self.phys_sim.state.heading)
        self.velocities = self.phys_sim.state.u

    def create_simulator(self, ua_surrogate_config: Optional[Dict]=None,
                         ugent_surrogate_config: Optional[Dict]=None,
                         ugent_b01_surrogate_config: Optional[Dict]=None):
        """
        Create the dynamics model for `self.dynamics_model` and wrap it in a ShipSimulator.

        Args:
            ua_surrogate_config: Configuration for the UA surrogate model.
            ugent_surrogate_config: Configuration for the UGent surrogate model.
            ugent_b01_surrogate_config: Configuration for the UGent B01 surrogate model.

        Returns:
            ShipSimulator using this env's ship, time step and wind/current flags.

        Raises:
            ValueError: Ship model not supported by the selected dynamics model.
            RuntimeError: Unknown dynamics model, or FH_TABULAR without a bridge.
        """

        if self.dynamics_model == Model.FOSSEN:
            require_hull_coefficients(self.ship_model, f"{type(self).__name__} with Model.FOSSEN")
            self.dynamics = Fossen(self.ship.specifications)
        elif self.dynamics_model == Model.NOMOTO:
            require_hull_coefficients(self.ship_model, f"{type(self).__name__} with Model.NOMOTO")
            self.dynamics = Nomoto(self.ship.specifications)
        elif self.dynamics_model == Model.UGENT_SURROGATE:
            # UGent's coefficients are identified for the Tripoli only.
            if self.ship_model != ShipModel.TRIPOLI:
                raise ValueError(
                    f"Model.UGENT_SURROGATE is identified for ShipModel.TRIPOLI only, "
                    f"got ship_model={self.ship_model}."
                )
            self.dynamics = UGentSurrogate(self.ship.specifications, ugent_surrogate_config)
            # Bound method, so deepcopy re-binds it to the copied env.
            self.dynamics.set_bank_distance_provider(self.bank_beam_distances)
        elif self.dynamics_model == Model.UGENT_B01_SURROGATE:
            # The B01 coefficients are identified for the model-scale B01 only.
            if self.ship_model != ShipModel.B01:
                raise ValueError(
                    f"Model.UGENT_B01_SURROGATE is identified for ShipModel.B01 only, "
                    f"got ship_model={self.ship_model}."
                )
            self.dynamics = UGentB01Surrogate(self.ship.specifications, ugent_b01_surrogate_config)
            # Bound method, so deepcopy re-binds it to the copied env.
            self.dynamics.set_bank_distance_provider(self.bank_beam_distances)
        elif self.dynamics_model == Model.UANTWERP:
            # Rays are cast against this environment's own river polygon.
            self.dynamics = UAntwerpSurrogate(self.ship.specifications, ua_surrogate_config,
                                              ship_model=self.ship_model, ray_caster=self)
            # full_env presets take wind/current from their lookup CSV
            # (models_own_wind_current); other presets get EnvironmentModel's via
            # ShipSimulator, according to the wind/current flags.
            if self.dynamics.models_own_wind_current:
                if not (self.wind and self.current):
                    logger.warning(
                        "wind=%s/current=%s, but the UAntwerp surrogate (preset '%s') always takes "
                        "wind and current from its lookup CSV: these flags have no effect on it.",
                        self.wind, self.current, self.dynamics.preset.name,
                    )
            elif self.wind or self.current:
                logger.info(
                    "wind=%s/current=%s: the UAntwerp surrogate (preset '%s') has no wind/current "
                    "inputs, so ShipSimulator adds EnvironmentModel wind/current for it.",
                    self.wind, self.current, self.dynamics.preset.name,
                )
        elif self.dynamics_model == Model.FH_TABULAR:
            # Guard for code paths that bypass __init__.
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
        Deep copy, except that the FH bridge (_fh_sim) is shared: it holds non-picklable
        native resources. All other references to the bridge also resolve to the original.
        """

        new_instance = self.__class__.__new__(self.__class__)
        # Register first, so references back to this env (e.g. the UGent bank
        # provider) resolve to the copy instead of recursing.
        memo[id(self)] = new_instance
        if self._fh_sim is not None:
            memo[id(self._fh_sim)] = self._fh_sim

        for key, value in self.__dict__.items():
            if key == '_fh_sim':
                setattr(new_instance, key, value)
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))

        return new_instance

    @property
    def fh_sim(self):
        """
        The FH engine bridge passed as `fh_bridge`, or None if not Model.FH_TABULAR.
        """

        return self._fh_sim

    # -------------------------
    # Environment data loaders
    # -------------------------
    def _load_environment_data(self, target_pos: Optional[np.ndarray]) -> None:
        """
        Load the map contours and trajectory, and build the checkpoints (with perpendicular
        guidance lines) from start_pos to the target.

        Raises:
            FileNotFoundError: A required CSV file is missing.
            ValueError: A file has the wrong shape, or fewer than 2 checkpoints result.
            RuntimeError: A file cannot be parsed.
        """

        base_path = os.path.dirname(os.path.abspath(__file__))

        def load_csv_strict(name: str) -> np.ndarray:
            """
            Load a non-empty (N, 2) x, y CSV (one header row) from this module's directory.

            Raises:
                FileNotFoundError: The file does not exist.
                RuntimeError: The file cannot be parsed.
                ValueError: The data is not a non-empty (N, 2) array.
            """

            csv_path = os.path.join(base_path, name)
            try:
                data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Missing required environment file: {csv_path}") from exc
            except Exception as exc:   # anything else from loadtxt, re-raised with the path
                raise RuntimeError(f"Failed to load {csv_path}: {exc}") from exc
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

        # A given target_pos defines where the path ends; otherwise the trajectory's last point.
        target_pos_arr = np.array(target_pos, dtype=np.float32) if target_pos is not None else None

        # Trim the path to run from start_pos to the target.
        path = self._reduce_path(path, self.initial_ship_pos, target_pos_arr)

        # Create checkpoints spaced along path
        path = create_checkpoints_from_simple_path(path, CHECKPOINTS_DISTANCE)

        # The first checkpoint is the start position.
        if len(path) == 0:
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

        self.checkpoints = []
        for cp, line in zip(checkpoints, lines):
            cp['perpendicular_line'] = line
            self.checkpoints.append(cp)

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
        Sample up to nbr_samples points inside the river polygon (rejection sampling); float32 (N, 2).
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

    def bank_segments(
            self,
            edge_tol: float = 0.01,
            min_cutoff_length: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the river-polygon boundary into real-bank and map cut-off segments.

        Map cut-offs are straight edges where the depth contour is clipped at
        the map limit; they are not banks. Uses MAP_CUTOFF_LINES when the
        contour has an entry there; otherwise a cut-off is a run of
        consecutive segments on one bounding-box side, at least
        `min_cutoff_length` long (short touches, e.g. a bend crest, are banks).
        Cut-offs not aligned with the bounding box are not detected.

        Args:
            edge_tol: Tolerance (m) for lying on a bounding-box side.
            min_cutoff_length: Minimum run length (m) to count as cut-off.

        Returns:
            (bank_segments, cutoff_segments): float64 arrays (K, 2, 2).
        """

        explicit_lines = MAP_CUTOFF_LINES.get(os.path.basename(str(self.depth_contour)))
        if explicit_lines:
            return self._bank_segments_from_lines(explicit_lines)

        minx, miny, maxx, maxy = self.polygon_shape.bounds
        bank_parts, cutoff_parts = [], []

        for ring in [self.polygon_shape.exterior, *self.polygon_shape.interiors]:
            c = np.asarray(ring.coords, dtype=np.float64)[:, :2]
            if len(c) < 2:
                continue
            segs = np.stack([c[:-1], c[1:]], axis=1)
            seg_len = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
            keep = seg_len > 1e-9                      # drop zero-length
            segs, seg_len = segs[keep], seg_len[keep]
            if len(segs) == 0:
                continue

            # Bounding-box side each segment lies on (0 = none, 1..4 = side).
            x, y = segs[..., 0], segs[..., 1]
            side = np.zeros(len(segs), dtype=int)
            for k, (v, ref) in enumerate(((x, minx), (x, maxx), (y, miny), (y, maxy)), start=1):
                side[(side == 0) & np.all(np.abs(v - ref) <= edge_tol, axis=1)] = k

            # Runs of consecutive same-side segments (may wrap around the ring).
            is_cutoff = np.zeros(len(segs), dtype=bool)
            if np.all(side == side[0]) and side[0] != 0:
                runs = [np.arange(len(segs))]          # whole ring on one side
            else:
                start = int(np.flatnonzero(side != np.roll(side, 1))[0]) if np.any(side != np.roll(side, 1)) else 0
                order = np.roll(np.arange(len(segs)), -start)
                runs, current = [], [order[0]]
                for i in order[1:]:
                    if side[i] == side[current[-1]]:
                        current.append(i)
                    else:
                        runs.append(np.array(current)); current = [i]
                runs.append(np.array(current))
            for run in runs:
                if side[run[0]] != 0 and seg_len[run].sum() >= min_cutoff_length:
                    is_cutoff[run] = True

            bank_parts.append(segs[~is_cutoff])
            cutoff_parts.append(segs[is_cutoff])

        def cat(parts):
            """
            Concatenate the non-empty (n, 2, 2) segment arrays (empty (0, 2, 2) if none).
            """

            parts = [p for p in parts if len(p)]
            return np.concatenate(parts) if parts else np.zeros((0, 2, 2))

        return cat(bank_parts), cat(cutoff_parts)

    def _bank_segments_from_lines(
            self,
            lines: Tuple[Tuple[Tuple[float, float], ...], ...],
            tolerance: float = CUTOFF_LINE_TOLERANCE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        As `bank_segments`, using explicit cut-off lines (MAP_CUTOFF_LINES).

        A segment is a cut-off when its end points and midpoint all lie within
        `tolerance` of the same line. Warns for a line that matches nothing.

        Args:
            lines: Cut-off polylines in world coordinates (m).
            tolerance: Maximum distance (m) to a line.

        Returns:
            (bank_segments, cutoff_segments): float64 arrays (K, 2, 2).
        """

        line_geoms = [LineString(np.asarray(line, dtype=np.float64)) for line in lines]

        segments = []
        for ring in [self.polygon_shape.exterior, *self.polygon_shape.interiors]:
            c = np.asarray(ring.coords, dtype=np.float64)[:, :2]
            if len(c) < 2:
                continue
            segs = np.stack([c[:-1], c[1:]], axis=1)
            segs = segs[np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) > 1e-9]
            if len(segs):
                segments.append(segs)
        if not segments:
            return np.zeros((0, 2, 2)), np.zeros((0, 2, 2))
        segs = np.concatenate(segments)

        starts = shapely.points(segs[:, 0])
        ends = shapely.points(segs[:, 1])
        mids = shapely.points(segs.mean(axis=1))
        is_cutoff = np.zeros(len(segs), dtype=bool)
        for k, line in enumerate(line_geoms):
            near = np.asarray((shapely.distance(line, starts) <= tolerance)
                              & (shapely.distance(line, ends) <= tolerance)
                              & (shapely.distance(line, mids) <= tolerance), dtype=bool)
            is_cutoff |= near
            if not near.any():
                logger.warning("MAP_CUTOFF_LINES entry %d %s matches no boundary segment within "
                               "%.0f m; check its coordinates.", k, list(line.coords), tolerance)

        return segs[~is_cutoff], segs[is_cutoff]

    def sample_point_near_bank(
            self,
            nbr_samples: int = 1,
            min_distance: float = 20.0,
            max_distance: float = 135.0,
            edge_tol: float = 0.01,
            min_cutoff_length: float = 50.0,
            max_tries: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample river points whose distance to the nearest real bank is in [min_distance, max_distance].

        Steps a random distance perpendicular from a random bank point into the water and
        keeps candidates inside the polygon within the band; points closer than
        `max_distance` to a map cut-off edge are rejected. Uses `self._rng`.

        Args:
            nbr_samples: Number of points to return.
            min_distance: Lower bound of the distance band to the nearest real bank (m).
            max_distance: Upper bound of that band (m); also the minimum distance to a map cut-off edge.
            edge_tol: Passed to `bank_segments`.
            min_cutoff_length: Passed to `bank_segments`.
            max_tries: Candidate budget; default 200 * nbr_samples + 1000.

        Returns:
            Float32 array (M, 2), M <= nbr_samples (warns when M < nbr_samples).

        Raises:
            ValueError: If the distance band is invalid.
        """

        if nbr_samples <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if not (0.0 <= min_distance < max_distance):
            raise ValueError(
                f"Need 0 <= min_distance < max_distance, got [{min_distance}, {max_distance}]"
            )

        bank, cutoff = self.bank_segments(edge_tol, min_cutoff_length)
        if len(bank) == 0:
            logger.warning("No real-bank segments found (all boundary segments look like map "
                           "cut-off edges); no near-bank points sampled.")
            return np.zeros((0, 2), dtype=np.float32)

        bank_geom = shapely.multilinestrings(bank)
        cutoff_geom = shapely.multilinestrings(cutoff) if len(cutoff) else None

        a, b = bank[:, 0], bank[:, 1]
        seg_vec = b - a
        seg_len = np.linalg.norm(seg_vec, axis=1)
        unit = seg_vec / seg_len[:, None]
        normal = np.stack([-unit[:, 1], unit[:, 0]], axis=1)   # left-hand normal
        seg_prob = seg_len / seg_len.sum()

        rng = self._rng
        max_tries = int(max_tries) if max_tries is not None else 200 * nbr_samples + 1000
        samples = []
        n_found, tries = 0, 0

        while n_found < nbr_samples and tries < max_tries:
            batch = int(min(max(4 * (nbr_samples - n_found), 64), max_tries - tries))
            tries += batch

            idx = rng.choice(len(bank), size=batch, p=seg_prob)
            foot = a[idx] + rng.uniform(0.0, 1.0, size=(batch, 1)) * seg_vec[idx]
            d = rng.uniform(min_distance, max_distance, size=(batch, 1))

            # Water side is unknown (differs for islands): try left, else right.
            left = foot + normal[idx] * d
            right = foot - normal[idx] * d
            left_in = shapely.contains_xy(self.polygon_shape, left[:, 0], left[:, 1])
            cand = np.where(left_in[:, None], left, right)
            inside = shapely.contains_xy(self.polygon_shape, cand[:, 0], cand[:, 1])
            cand = cand[inside]
            if len(cand) == 0:
                continue

            pts = shapely.points(cand)
            d_bank = shapely.distance(bank_geom, pts)
            ok = (d_bank >= min_distance) & (d_bank <= max_distance)
            if cutoff_geom is not None:
                ok &= shapely.distance(cutoff_geom, pts) >= max_distance

            for p in cand[ok][: nbr_samples - n_found]:
                samples.append(p)
            n_found = len(samples)

        if n_found < nbr_samples:
            logger.warning("Only generated %d near-bank points out of requested %d "
                           "(band [%.1f, %.1f] m may not fit the polygon, or max_tries reached).",
                           n_found, nbr_samples, min_distance, max_distance)

        return np.asarray(samples, dtype=np.float32).reshape(-1, 2)

    def get_keel_clearance(self, pos: Optional[np.ndarray] = None) -> float:
        """
        Under-keel clearance [m] at `pos`.

        From the FH Simulator (pos ignored) or the UA/UGent surrogates; 0.0 for models
        without UKC.
        """

        if self._fh_sim is not None:
            # noinspection PyUnresolvedReferences
            return self._fh_sim.ship_interface.getKeelClearance()
        elif isinstance(self.dynamics, (UAntwerpSurrogate, UGentSurrogate, UGentB01Surrogate)):
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
        Distance [m] from `origin` to the nearest river-boundary hit on the segment
        origin to end_point, clipped to [0, max_distance] (max_distance if no hit).

        Args:
            origin: (x, y) ray start, world frame [m].
            end_point: (x, y) ray end; place it at least max_distance from origin.
            max_distance: Sensor range [m].
        """

        ray = LineString([origin, end_point])
        hit = self.polygon_shape.boundary.intersection(ray)

        if hit.is_empty:
            return max_distance

        origin_pt = Point(origin)

        if hit.geom_type == 'Point':
            dist = origin_pt.distance(hit)
        elif isinstance(hit, (MultiPoint, GeometryCollection)):
            # Skip a near-zero self-hit (ship exactly on the boundary).
            dists = [origin_pt.distance(g) for g in hit.geoms if not g.is_empty]
            far = [d for d in dists if d > 1e-6]
            dist = min(far) if far else 0.0
        else:
            # (Multi)LineString: ray collinear with a boundary segment.
            dist = origin_pt.distance(hit)

        return float(min(dist, max_distance))

    def ray_cast_distances_batch(
            self,
            origin: Tuple[float, float],
            end_points: np.ndarray,
            max_distance: float = 200.0,
    ) -> np.ndarray:
        """
        Vectorized `ray_cast_distance` for N rays from a common origin.

        Uses Shapely 2 array operations (one GEOS call each); only the rare
        MultiPoint/GeometryCollection hits are fixed up in Python.

        Args:
            origin: (x, y) start point shared by all rays (world frame).
            end_points: (N, 2) ray end points, each at least max_distance from origin.
            max_distance: Sensor range [m].

        Returns:
            float32 (N,) distances clipped to [0, max_distance].
        """


        if self._boundary_cache is None:
            self._boundary_cache = self.polygon_shape.boundary

        boundary = self._boundary_cache
        origin_pt = Point(origin)
        n = len(end_points)

        origins_arr = np.tile(np.asarray(origin, dtype=np.float64), (n, 1))
        ends_arr = np.asarray(end_points, dtype=np.float64)
        coords = np.stack([origins_arr, ends_arr], axis=1)  # (n, 2, 2)
        rays = shapely.linestrings(coords)

        hits = np.asarray(shapely.intersection(boundary, rays))

        # Correct except for MultiPoint/GeometryCollection hits, which may contain
        # a near-zero self-hit; those are fixed below.
        dists = shapely.distance(origin_pt, hits)

        # NaN (empty hit) means nothing within range.
        dists = np.where(np.isnan(dists), max_distance, dists)

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

    def bank_beam_distances(self, state: Optional[VesselState] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Distances [m] to the real port and starboard banks along the beam rays (map
        cut-offs ignored). Bank-distance provider of the UGent surrogates, called every sub-step.

        Args:
            state: Vessel state to cast from; defaults to the env's own state.

        Returns:
            (dist_port, dist_starboard); None if no bank within RAY_MAX_DISTANCE.
        """

        if self._bank_lines_cache is None:
            bank_segs, _ = self.bank_segments()
            lines = np.asarray(shapely.linestrings(bank_segs)) if len(bank_segs) else np.empty(0, dtype=object)
            self._bank_lines_cache = (lines, shapely.STRtree(lines))
        lines, tree = self._bank_lines_cache

        if state is None:
            state = self.phys_sim.state
        origin = np.array([state.x, state.y], dtype=np.float64)
        angles = ray_world_angles(state.heading)[[_PORT_BEAM_RAY_IDX, _STARBOARD_BEAM_RAY_IDX]]
        ends = origin + RAY_MAX_DISTANCE * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        rays = np.asarray(shapely.linestrings(np.stack([np.tile(origin, (2, 1)), ends], axis=1)))

        # Nearest hit per ray, among the segments the index returns.
        ray_idx, seg_idx = tree.query(rays, predicate="intersects")
        nearest: List[Optional[float]] = [None, None]
        if len(ray_idx):
            dists = np.asarray(shapely.distance(Point(origin), shapely.intersection(lines[seg_idx], rays[ray_idx])),
                               dtype=np.float64)
            for i, d in zip(ray_idx, dists):
                if not np.isnan(d) and (nearest[i] is None or d < nearest[i]):
                    nearest[i] = float(d)
        return nearest[0], nearest[1]

    def _has_enough_keel_clearance(self, depth_threshold: float = 0.5) -> bool:
        """
        True if the FH Simulator reports keel clearance >= depth_threshold [m].

        Always True for the Python dynamics models (no check).
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
        True if `ship_position` is inside the river polygon and has keel clearance (each check optional).
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
        Cut the trajectory to run from just after start_pos to exactly target_pos.

        The trajectory is walked forwards or backwards (reversed point order), whichever
        direction reaches target_pos. Without target_pos, the path runs forwards to the
        trajectory's last point, or backwards to its first point if start_pos is closest
        to the last point.

        Args:
            path: Full trajectory, (N, 2).
            start_pos: Ship's start position.
            target_pos: End position of the path, or None.

        Raises:
            ValueError: Empty result, a one-point trajectory, or start_pos and
                target_pos resolving to the same trajectory point.
        """

        if len(path) == 0:
            raise ValueError("Path is empty.")

        start_distances = np.linalg.norm(path - start_pos, axis=1)
        start_index = int(np.argmin(start_distances))

        if target_pos is None:
            if start_index < len(path) - 1:
                return path[start_index + 1:]

            # Nothing lies ahead of start_pos: walk backwards to the first point.
            if len(path) < 2:
                raise ValueError(
                    f"_reduce_path: the trajectory has only one point, so no path "
                    f"can be built from start_pos {np.asarray(start_pos).tolist()}."
                )
            logger.info(
                "_reduce_path: start_pos %s is closest to the trajectory's last point "
                "(index %d); walking the trajectory backwards to its first point.",
                np.asarray(start_pos).tolist(), start_index,
            )
            return path[:start_index][::-1].copy()

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
            # Backwards: from just before start_index down to target_index.
            truncated_path = path[target_index:start_index][::-1].copy()

        if len(truncated_path) == 0:
            raise ValueError(
                f"_reduce_path produced an empty path between start_pos "
                f"{np.asarray(start_pos).tolist()} (index {start_index}) and target_pos "
                f"{target_pos.tolist()} (index {target_index})."
            )

        # End exactly at target_pos, not at the nearest trajectory point.
        truncated_path[-1] = target_pos.astype(truncated_path.dtype)

        return truncated_path

    @staticmethod
    def _distance_from_point_to_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        Perpendicular distance from `point` to the infinite line through line_start and
        line_end (distance to line_start if they coincide).
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
        Wrap an angle [rad] to [-π, π).
        """

        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _safe_heading_from_vector(vec) -> float:
        """
        Heading angle [rad] of a vector, or 0.0 for a zero vector.
        """

        if np.linalg.norm(vec) > 0.0:
            return float(np.arctan2(vec[1], vec[0]))
        return 0.0

    def _calculate_heading_error(self, target_heading: float, current_heading: float, dead_zone_deg: float = 5.0) -> float:
        """
        Wrapped heading error target - current [rad]; 0 within +-dead_zone_deg.
        """

        error = self.normalize_angle(target_heading - current_heading)
        dead_zone = np.radians(dead_zone_deg)
        return 0.0 if abs(error) < dead_zone else error

    # -----------------------
    # Observation & reset
    # -----------------------
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """
        Box in [-1, 1]: 12 base features (velocities, checkpoint distances, cross-track and
        heading errors, previous rudder, LOS error) plus 2 each for wind and current if enabled.
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
        Reset hook, called by reset() before _initialize_simulation().

        With the FH Simulator, rewinds its math model to the initial state (t = 0.25 s,
        the earliest rewind point); otherwise defers to the base class.
        """

        if self._fh_sim is not None:
            # noinspection PyUnresolvedReferences
            self._fh_sim.reset()
            return

        super().env_specific_reset()

    def randomize(self, randomization_scale: Optional[float] = None) -> None:
        """
        Perturb the initial position uniformly by up to +-randomization_scale per axis.

        No-op with the FH Simulator, which always rewinds to the initial state set at
        math model initialization (changing it needs a re-initialization).

        Args:
            randomization_scale: New scale; None keeps the current one.

        Raises:
            ValueError: If randomization_scale is not positive.
        """

        if self._fh_sim is not None:
            return

        if randomization_scale is not None:
            if randomization_scale <= 0:
                raise ValueError("randomization_scale must be positive")

            self.randomization_scale = randomization_scale

        # Use the env's own generator so `reset(seed=...)` makes this reproducible.
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
        Reset the episode (re-seeding the env RNG if `seed` is given); returns (observation, {}).
        """

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        super().reset(seed=seed)

        self.env_specific_reset()
        self._initialize_simulation()

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        Normalized float32 observation (layout: see _initialize_observation_space).

        Checkpoints beyond the last one are extrapolated along the last path segment.
        """

        # Normalize velocities
        norm_velocities = np.array([
            np.clip(self.phys_sim.state.u / self.ship.specifications.surge_limits[1], -1, 1),
            np.clip(self.phys_sim.state.v / self.ship.specifications.sway_limits[1], -1, 1),
            np.clip(self.phys_sim.state.r / self.ship.specifications.yaw_rate_limits[1], -1, 1)
        ], dtype=np.float32)

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
        # 2-D cross product written out (np.cross on 2-D vectors is deprecated since NumPy 2.0).
        cross_z = track_vec[0] * ship_vec[1] - track_vec[1] * ship_vec[0]
        norm_cross_error *= 1.0 if cross_z >= 0.0 else -1.0

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
        Advance one time step with action [rudder, thrust] in [-1, 1].

        Returns:
            (observation, reward, terminated, truncated, info); after the episode has
            ended, the state is kept and the reward is 0.
        """

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(
                f"Expected action shape (2,), got {action.shape}. "
                f"Action values: {action.tolist()}. Use [-1,1] for [rudder, thrust]."
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

            # Reward and termination check
            reward, terminated = self._calculate_reward()

            # Truncate at max_steps unless terminated.
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

        #self.render()

        # Gymnasium: time-limit cut-off is `truncated` (bootstrap V(s')), a real end is
        # `terminated`; mirrored in `info` for the framework's own agents.
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # -------------------
    # Reward calculation
    # -------------------
    def _calculate_reward(self) -> Tuple[float, bool]:
        """
        Compute the step reward, advance the checkpoint index and check termination.

        Returns:
            (reward, done)
        """

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
        # Both in rad (performed_action holds the actuator's rudder angle). The +1e-6 rad
        # tolerance keeps a move at exactly the maximum slew rate out of the 'exceeded'
        # penalty branch despite float32 rounding.
        max_rudder_change = self.ship.specifications.max_rudder_rate * self.time_step + 1e-6  # rad
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
        Shortest distance from point c to the segment p1-p2.
        """

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
        True if segment prev_point-curr_point touches or crosses the infinite line through line_start and line_end.
        """

        def _orientation(a, b, c):
            """
            2D cross product: > 0 if c is left of line ab, < 0 if right, 0 if collinear.
            """

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
        Desired LOS heading [rad] toward the current path segment.

        LOS guidance law from Changyuan Chen's PhD thesis "Numerical and Experimental
        Study on Ship Motion Control Systems in Shallow Water" (Flanders Hydraulics).
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
            # Virtual checkpoint behind the first one.
            p_k_min_1 = p_k - (p_k_plus_1 - p_k)
        x_k_min_1, y_k_min_1 = p_k_min_1[:]

        seg_sq = (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2
        if seg_sq < EPSILON:
            # Coincident checkpoints: head for p_k_plus_1.
            return math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)

        # Orthogonal projection A of the ship onto the line p_k_min_1, p_k.
        proj_scalar = (
                          (x - x_k_min_1) * (x_k - x_k_min_1) +
                          (y - y_k_min_1) * (y_k - y_k_min_1)
                      ) / seg_sq

        x_a = x_k_min_1 + proj_scalar * (x_k - x_k_min_1)
        y_a = y_k_min_1 + proj_scalar * (y_k - y_k_min_1)

        # Calculate the look ahead distance L_A
        c = 0.5   # Look-ahead coefficient (thesis default, p. 148)
        delta_y = math.sqrt((x - x_a) ** 2 + (y - y_a) ** 2)
        l_pp = 0.95 * self.ship.specifications.length
        l_a_sq = (delta_y + c * l_pp) ** 2 - delta_y ** 2
        l_a = math.sqrt(max(l_a_sq, 0.0))   # clamp against floating-point negatives

        # Calculate the LOS position Ploss(x_loss, y_loss)
        seg_len = math.sqrt(seg_sq)
        x_loss = x_a + ((x_k - x_k_min_1) / seg_len) * l_a
        y_loss = y_a + ((y_k - y_k_min_1) / seg_len) * l_a

        # Acceptance circle radius Rk of the current checkpoint.
        w_los = 0.5   # Acceptance-circle coefficient (thesis default, p. 148)
        r_k = w_los * l_pp

        heading_p_k_min_1_p_k = self._safe_heading_from_vector(p_k - p_k_min_1)
        heading_p_k_p_k_plus_1 = self._safe_heading_from_vector(p_k_plus_1 - p_k)
        heading_difference = self.normalize_angle(
            heading_p_k_min_1_p_k - heading_p_k_p_k_plus_1
        )
        collinear = abs(heading_difference) < (1.0 / 57.0)   # previous, current and next checkpoint within ~1 deg

        if collinear:
            dist_a_pk = math.sqrt((x_a - x_k) ** 2.0 + (y_a - y_k) ** 2.0)
            if dist_a_pk > r_k:  # |A Pk| > Rk
                los_angle = math.atan2(y_loss - y, x_loss - x)
            else:  # |A Pk| < Rk
                # Aim at Pk+1 instead of Ploss.
                los_angle = math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)
        else:
            dist_p_pk = math.sqrt((x - x_k) ** 2 + (y - y_k) ** 2)
            if dist_p_pk > 2 * r_k:  # |P Pk| > 2*Rk
                if l_a < dist_p_pk:  # LA < |P Pk|
                    los_angle = math.atan2(y_loss - y, x_loss - x)
                else:  # LA > |P Pk|
                    # Shortened look-ahead L'A along the next segment.
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
            else:  # |P Pk| < 2*Rk
                # Shift to the next segment: Pk-1 := Pk, Pk := Pk+1.
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

                # Calculate the LOS position Ploss(x_loss, y_loss)
                seg_len2 = math.sqrt(seg_sq2)
                x_loss = x_a + ((x_k - x_k_min_1) / seg_len2) * l_a
                y_loss = y_a + ((y_k - y_k_min_1) / seg_len2) * l_a
                los_angle = math.atan2(y_loss - y, x_loss - x)

        return los_angle

    @staticmethod
    def _logarithmic_scaling(value: float, value_range: tuple, half_value: float = 10):
        """
        Map `value` (clamped to value_range = (min, max)) to [0, 1] on a logarithmic
        curve with f(min) = 0, f(max) = 1 and f(half_value) = 0.5.
        """

        min_value, max_value = value_range

        clamped = max(min_value, min(value, max_value))
        if clamped != value:
            logger.debug(
                "Value %s clamped to [%s, %s]", value, min_value, max_value
            )

        x = (clamped - min_value) / (max_value - min_value)
        m = (half_value - min_value) / (max_value - min_value)

        if math.isclose(m, 0.5, abs_tol=1e-9):
            return x

        # Curvature k such that log1p(k*m) / log1p(k) = 0.5.
        k = (1 - 2 * m) / (m ** 2)

        return math.log1p(k * x) / math.log1p(k)

    # -----------------------
    # Rendering
    # -----------------------
    def _initialize_rendering(self) -> None:
        """
        Create the figure, plot handles and axes for rendering.
        """

        self.initialize_plots = False

        # Screen size via a temporary Tk root. tkinter is imported lazily so the
        # framework still imports on headless machines without Tk (e.g. HPC nodes).
        # Any failure (no Tk, no display, Tk off the main thread, ...) falls back.
        # noinspection PyBroadException
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
        except Exception:
            screen_width, screen_height = MAX_FIG_WIDTH, MAX_FIG_HEIGHT

        fig_width = min(MAX_FIG_WIDTH, screen_width)
        fig_height = min(MAX_FIG_HEIGHT, screen_height)
        dpi = DPI

        self.fig, self.ax = plt.subplots(figsize=(fig_width / dpi, fig_height / dpi), dpi=dpi)

        # Move the window to the top-left where the backend allows it.
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
        Draw the static features: path, checkpoints, perpendicular lines and map polygons.
        """

        if not hasattr(self, 'ax') or not self.checkpoints:
            return

        path_points = [self.ship_pos] + [cp['pos'] for cp in self.checkpoints] + [self.target_pos]

        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                         'g-', label='Path' if i == 0 else "")

        for i, checkpoint in enumerate(self.checkpoints):
            circle = patches.Circle((checkpoint['pos'][0], checkpoint['pos'][1]),
                                    1.0, color='black', alpha=0.3)
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
            # noinspection PyUnresolvedReferences
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def render(self) -> None:
        """
        Draw the ship and heading (human render mode only), using blitting where possible.
        """

        #if self.render_mode != 'human':
        #    return

        if self.initialize_plots and not hasattr(self, 'ship_plot'):
            self._initialize_rendering()
            self._draw_fixed_landmarks()
            plt.show(block=False)
            plt.pause(0.001)
            self.fig.canvas.draw()
            # noinspection PyUnresolvedReferences
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig.canvas.mpl_connect('draw_event', self._on_draw)

        canvas = self.fig.canvas

        def _safe(callable_obj):
            """
            Call `callable_obj()`, ignoring rendering errors so they cannot abort the simulation.
            """

            try:
                callable_obj()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass

        if hasattr(self, 'background'):
            # noinspection PyUnresolvedReferences
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

        # Flush pending GUI events.
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