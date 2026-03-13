"""Gymnasium environment for RL training inside UE5 via CosysAirSim.

This module provides a :class:`AirSimEnv`, a drop-in replacement for
``PySimEnv`` that uses Unreal Engine 5's physics via the CosysAirSim API
instead of the Fossen mathematical model.

Coordinate frames
-----------------
- **Fossen**: x = east, y = north, heading from east (CCW positive).
- **AirSim**: x = north, y = east (NED), yaw from north (CW positive).
- **Transform**: ``AirSim_yaw = pi/2 - fossen_heading``.

Usage
-----
>>> from airsim.environments.airsim_env import AirSimEnv
>>> env = AirSimEnv(ship_pos=[2245, -47], target_pos=[11530, 13000])
>>> obs, info = env.reset()
>>> obs, reward, done, truncated, info = env.step([0.0, 0.5])
>>> env.close()
"""

import csv
import os
import math
import time
import json
import logging
import numpy as np
import gymnasium as gym

from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from fw.environments.base_env import BaseEnv
from fw.environments.tools import create_checkpoints_from_simple_path
from airsim.config.airsim_config import (
    DEFAULT_AIRSIM_SAVE_SLOT,
    DEFAULT_AIRSIM_SIMULATION_SPEED,
)
from airsim.communication.coord_transform import CoordTransform, _normalize_angle
from airsim.communication.control_mapping import (
    map_fossen_policy_action,
    validate_policy_action,
)
from fw.ships.ship_factory import create_ship, ShipModel
from fw.ships.actuator_model import ActuatorModel

logger = logging.getLogger(__name__)

# ----------------
# Constants (mirror PySimEnv for consistency)
# ----------------
CHECKPOINTS_DISTANCE = 350
MIN_GRID_POS = -11700
MAX_GRID_POS = 14500
SUCCESS_REWARD = 50.0
CHECKPOINT_AREA_SIZE = 5.0
TARGET_AREA_SIZE = 7.0
PERPENDICULAR_LINE_LENGTH = 50.0
PERPENDICULAR_LINE_PROXIMITY = 2.0
HEADING_CHANGE_THRESHOLD = np.pi / 2.0
CROSS_TRACK_TERMINATION_MULTIPLIER = 2.0
EPSILON = 1e-9

# Default calibration file location
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIBRATION_PATH = _THIS_DIR.parent / "config" / "calibration.json"
_ENVIRONMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "fw" / "environments"


# ----------------
# Helpers
# ----------------
def _yaw_from_quaternion(q) -> float:
    """Extract yaw (in radians) from an AirSim Quaternionr."""
    return math.atan2(
        2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
        1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2),
    )


# _normalize_angle is imported from coord_transform


def _safe_heading_from_vector(vec: np.ndarray) -> float:
    """Return heading angle for a 2D vector, or 0 if near-zero."""
    n = np.linalg.norm(vec)
    return float(np.arctan2(vec[1], vec[0])) if n > 0.0 else 0.0


def _calculate_perpendicular_lines(
    checkpoints: List[dict],
    line_length: float = PERPENDICULAR_LINE_LENGTH,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute perpendicular guidance lines for checkpoint list."""
    def smooth_tangent(cps, idx):
        if len(cps) == 1:
            return np.array([1.0, 0.0], dtype=np.float32)
        if idx == 0:
            t = cps[1]["pos"] - cps[0]["pos"]
        elif idx == len(cps) - 1:
            t = cps[-1]["pos"] - cps[-2]["pos"]
        else:
            t = 0.5 * ((cps[idx + 1]["pos"] - cps[idx]["pos"]) +
                        (cps[idx]["pos"] - cps[idx - 1]["pos"]))
        n = np.linalg.norm(t)
        return (t / n if n != 0.0 else t).astype(np.float32)

    lines = []
    for i, cp in enumerate(checkpoints):
        pos = np.asarray(cp["pos"], dtype=np.float32)
        tang = smooth_tangent(checkpoints, i)
        perp = np.array([-tang[1], tang[0]], dtype=np.float32)
        offset = perp * (line_length / 2.0)
        lines.append((pos + offset, pos - offset))
    return lines


# ----------------
# Main Environment
# ----------------
class AirSimEnv(BaseEnv):
    """Gymnasium environment that trains an RL agent inside UE5 via CosysAirSim.

    This environment mirrors ``PySimEnv`` but replaces the Fossen physics
    model with UE5's physics engine. The agent sends ``[steering, throttle]``
    commands, and the environment reads back position, heading, speed, and
    collision info from UE5.

    Attributes:
        client: The active ``cosysairsim.CarClient`` connection.
        coord: The coordinate transform between Fossen and AirSim frames.
    """

    def __init__(
        self,
        ship_pos: Optional[list] = None,
        target_pos: Optional[list] = None,
        calibration: Optional[dict] = None,
        calibration_path: Optional[str] = None,
        use_airsim: bool = True,
        ip: str = "",
        port: int = 41451,
        timeout: float = 10.0,
        max_retries: int = 3,
        health_check_interval: float = 30.0,
        csv_log_path: str = "",
        save_slot: str = DEFAULT_AIRSIM_SAVE_SLOT,
        render_mode: Optional[str] = None,
        max_steps: int = 14000,
        time_step: float = 0.1,
        frames_per_step: int = 10,
        use_pause_stepping: bool = True,
        training_mode: bool = False,
        verbose: bool = False,
        episode_log_path: str = "",
        simulation_speed: float = DEFAULT_AIRSIM_SIMULATION_SPEED,
        physics_update_timeout: float = 1.0,
        state_poll_interval: float = 0.005,
        latency_warning_threshold: Optional[float] = None,
    ) -> None:
        """Initialise the AirSim training environment.

        Args:
            ship_pos: Initial ship position in Fossen coords [x_east, y_north].
            target_pos: Target position in Fossen coords.
            calibration: Optional calibration dict with scale/rotation/origins.
            calibration_path: Path to calibration JSON. If None, tries default.
            use_airsim: Whether the AirSim bridge should be enabled.
            ip: AirSim server IP. Empty string = localhost.
            port: AirSim RPC port.
            timeout: RPC connection timeout in seconds.
            max_retries: Maximum number of connection retries.
            health_check_interval: Seconds between bridge health checks.
            csv_log_path: Optional bridge telemetry CSV path.
            save_slot: UE5 save slot loaded before training or visualization.
            render_mode: 'human' or None.
            max_steps: Maximum steps per episode before truncation.
            time_step: Nominal simulation time step (seconds).
            frames_per_step: Physics frames to advance per RL step.
            use_pause_stepping: If True, pause sim and step frame-by-frame
                (fastest training mode).
            verbose: Print debug info.
            episode_log_path: Optional path to a CSV file where per-step
                episode telemetry (position, speed, cross-track error,
                reward, action) is appended.  One file collects all
                episodes across resets.  Set to ``""`` to disable.
            simulation_speed: Simulation speed multiplier set via
                ``simSetSimulationSpeed``.  ``1.0`` = real-time (inference);
                higher values (e.g. ``20.0``) speed up training.
        """
        super().__init__(render_mode)

        self.max_steps = max_steps
        self.time_step = time_step
        self.frames_per_step = frames_per_step
        self.use_pause_stepping = use_pause_stepping
        self.verbose = verbose
        self._physics_update_timeout = float(physics_update_timeout)
        self._state_poll_interval = float(state_poll_interval)
        self._latency_warning_threshold = float(
            latency_warning_threshold
            if latency_warning_threshold is not None
            else max(0.25, self.time_step)
        )
        self._last_ue5_timestamp: Optional[int] = None
        self.last_physics_wait_s: float = 0.0
        self.last_step_latency_s: float = 0.0

        # -- Episode-level telemetry CSV (T-012) --
        self._episode_log_path: str = episode_log_path
        self._episode_log_file = None
        self._episode_csv_writer = None
        self._episode_number: int = 0

        # -- Ship specifications & actuator model --
        self.ship = create_ship(ShipModel.MYZAKO_DEFAULT)
        self.ship_specs = self.ship.specifications
        self._actuators = ActuatorModel(self.ship_specs)

        # Match Fossen model surge limit for identical observation normalisation.
        # MYZAKO_DEFAULT: surge_limits = (0.0, 5.0).  UE5 top speed is 5.6 m/s
        # but we cap at the same value the policy was trained against.
        self.MAX_SPEED = self.ship_specs.surge_limits[1]  # 5.0 m/s

        # Positions in Fossen frame
        self.initial_ship_pos = np.array(
            ship_pos if ship_pos is not None else [2245, -47],
            dtype=np.float32,
        )
        self.target_fossen = np.array(
            target_pos if target_pos is not None else [11530.0, 13000.0],
            dtype=np.float32,
        )

        # Coordinate transform
        if calibration is not None:
            self.coord = CoordTransform(calibration)
        elif calibration_path is not None and Path(calibration_path).exists():
            self.coord = CoordTransform.load(Path(calibration_path))
        elif _DEFAULT_CALIBRATION_PATH.exists():
            self.coord = CoordTransform.load()
            logger.info("Loaded calibration from %s", _DEFAULT_CALIBRATION_PATH)
        else:
            self.coord = CoordTransform.identity(tuple(self.initial_ship_pos))
            logger.info("No calibration found - using identity axis-swap transform.")

        # -- Connect to AirSim via Bridge --
        try:
            import cosysairsim as airsim
        except ImportError as e:
            raise ImportError(
                "The 'cosysairsim' package is required to use AirSimEnv but is "
                "not installed. Install it with:  pip install cosysairsim"
            ) from e
        self._airsim = airsim

        from airsim.config.airsim_config import AirSimConfig
        from airsim.communication.airsim_bridge import AirSimBridge

        self._save_slot = save_slot
        cfg = AirSimConfig(
            use_airsim=use_airsim,
            ip=ip,
            port=port,
            timeout=timeout,
            max_retries=max_retries,
            health_check_interval=health_check_interval,
            csv_log_path=csv_log_path,
        )
        self.bridge = AirSimBridge(cfg)

        if not self.bridge.is_connected:
            raise ConnectionError(
                f"AirSimBridge could not connect to UE5 at {ip or '127.0.0.1'}:{port}. "
                "Make sure Unreal Engine 5 is open and in Play mode."
            )
        
        # -- Load world save + set simulation speed --
        self._simulation_speed = simulation_speed
        self.bridge.load_world_save(save_slot=self._save_slot)
        self.bridge.set_simulation_speed(simulation_speed)
        logger.info("Simulation speed set to %.1fx.", simulation_speed)

        # -- Auto drift correction --
        # Capture spawn pose AFTER world load so it reflects the loaded position.
        self._spawn_pose = self.bridge.get_vehicle_pose()
        cal_path = Path(calibration_path) if calibration_path else _DEFAULT_CALIBRATION_PATH
        if cal_path.exists():
            with open(cal_path, "r") as f:
                raw_cal = json.load(f)
            if "spawn_position" in raw_cal:
                saved_spawn = np.array(raw_cal["spawn_position"])
                current_spawn = np.array([
                    self._spawn_pose.position.x_val,
                    self._spawn_pose.position.y_val,
                ])
                drift = current_spawn - saved_spawn
                mag = np.linalg.norm(drift)
                if mag > 0.5:
                    logger.info(
                        "Origin drift detected: dx=%.2f, dy=%.2f (%.1f m) -> auto-correcting.",
                        drift[0], drift[1], mag,
                    )
                    self.coord._airsim_origin = self.coord._airsim_origin + drift
                else:
                    logger.debug("No origin drift (%.2f m).", mag)

        if self.use_pause_stepping:
            self.bridge.set_pause(True)
            logger.info("Simulation paused - using deterministic frame stepping.")

        # -- Draw trajectory in UE5 --
        self._draw_visualization()

        # -- Load checkpoints (same data as PySimEnv) --
        self._load_checkpoints()

        # -- State tracking --
        self.ship_pos = self.initial_ship_pos.copy()
        self.previous_ship_pos = self.initial_ship_pos.copy()
        self.ship_heading = 0.0
        self.previous_heading = 0.0
        self.ship_speed = 0.0
        self.ship_u = 0.0  # surge
        self.ship_v = 0.0  # sway
        self.ship_r = 0.0  # yaw rate
        self.performed_action = np.zeros(2, dtype=np.float32)
        self.last_controls = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}
        self.checkpoint_index = 1
        self.cross_error = 0.0
        self.step_count = 0
        self.max_dist = np.sqrt(2) * float(MAX_GRID_POS)
        self._smooth_action1: float = 0.0  # EMA state for Fossen throttle mapping

        # -- Gym spaces (identical to PySimEnv) --
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._initialize_observation_space()

        # Reward weights (same as PySimEnv)
        self.reward_weights = {
            "forward": 1.0,
            "alignment": 1.8,
            "cross_track": 1.5,
            "rudder": 0.03,
        }

        # -- Open episode log if configured --
        if self._episode_log_path:
            self._open_episode_log()

    # AirSimEnv holds a live network connection and a module reference
    # that cannot be pickled.
    def __deepcopy__(self, memo):
        """Return *self* - AirSimEnv cannot be deep-copied (live socket)."""
        return self

    def __copy__(self):
        """Return *self* - AirSimEnv cannot be shallow-copied."""
        return self

    # -------------
    # Episode telemetry CSV  (T-012)
    # -------------

    _EPISODE_LOG_FIELDS = [
        "episode", "episode_step",
        "ship_x", "ship_y", "heading_deg", "speed",
        "cross_track_error", "checkpoint_index",
        "reward", "steering", "throttle",
        "physics_wait_s", "step_latency_s", "ue5_timestamp",
    ]

    def _open_episode_log(self) -> None:
        """Open (or re-open) the episode telemetry CSV for appending.

        Creates the file with a header row when it does not yet exist.
        Safe to call repeatedly - closes any previously open handle first.
        """
        self._close_episode_log()
        path = self._episode_log_path
        try:
            file_exists = os.path.isfile(path)
            self._episode_log_file = open(path, "a", newline="")
            self._episode_csv_writer = csv.DictWriter(
                self._episode_log_file,
                fieldnames=self._EPISODE_LOG_FIELDS,
            )
            if not file_exists:
                self._episode_csv_writer.writeheader()
            logger.info("Episode log opened: %s", path)
        except Exception as exc:
            logger.warning("Could not open episode log '%s': %s", path, exc)
            self._episode_log_file = None
            self._episode_csv_writer = None

    def _write_episode_log_row(self, reward: float) -> None:
        """Append one row of telemetry to the episode CSV.

        Args:
            reward: The scalar reward computed for this step.
        """
        if self._episode_csv_writer is None:
            return
        try:
            self._episode_csv_writer.writerow({
                "episode": self._episode_number,
                "episode_step": self.step_count,
                "ship_x": float(self.ship_pos[0]),
                "ship_y": float(self.ship_pos[1]),
                "heading_deg": float(math.degrees(self.ship_heading)),
                "speed": float(self.ship_speed),
                "cross_track_error": float(self.cross_error),
                "checkpoint_index": int(self.checkpoint_index),
                "reward": float(reward),
                "steering": float(self.performed_action[0]),
                "throttle": float(self.performed_action[1]),
                "physics_wait_s": float(self.last_physics_wait_s),
                "step_latency_s": float(self.last_step_latency_s),
                "ue5_timestamp": int(self._last_ue5_timestamp or 0),
            })
        except Exception as exc:
            logger.warning("Episode log write failed: %s", exc)

    def _close_episode_log(self) -> None:
        """Flush and close the episode log file handle, if open."""
        if self._episode_log_file is not None:
            try:
                self._episode_log_file.flush()
                self._episode_log_file.close()
            except Exception:
                pass
            self._episode_log_file = None
            self._episode_csv_writer = None

    # -------------
    # BaseEnv adapter  (T-010)
    # -------------
    def env_specific_reset(self) -> None:
        """Perform AirSim-specific reset actions.

        Called by ``BaseEnv`` / the training pipeline *after* the
        generic state reset.  Delegates to the bridge reset and
        re-pauses the simulation when deterministic stepping is used.
        """
        self.bridge.reset()
        if self.use_pause_stepping:
            self.bridge.set_pause(True)

    def randomize(self, randomization_scale: Optional[float] = None) -> None:
        """Apply domain randomisation to the initial ship position.

        Adds a uniform random perturbation to ``initial_ship_pos`` and
        ensures the new position stays inside the navigable polygon.

        Args:
            randomization_scale: Maximum absolute perturbation (meters).
                Defaults to ``1.0`` if *None*.
        """
        scale = randomization_scale if randomization_scale is not None else 1.0
        if scale <= 0:
            raise ValueError("randomization_scale must be positive")

        perturbation = np.random.default_rng().uniform(
            low=-scale, high=scale, size=self.initial_ship_pos.shape,
        ).astype(np.float32)

        candidate = np.clip(
            self.initial_ship_pos + perturbation,
            MIN_GRID_POS, MAX_GRID_POS,
        )

        # Only accept if the candidate is inside the river polygon.
        if self._ship_in_open_water(candidate):
            self.initial_ship_pos = candidate
        else:
            logger.debug(
                "Randomised position (%.0f, %.0f) outside polygon - keeping original.",
                candidate[0], candidate[1],
            )

    # -------------
    # Data loading  (T-011)
    # -------------
    def _load_checkpoints(self) -> None:
        """Load trajectory checkpoints from CSV and add 3D AirSim coords.

        Reads ``trajectory_points_no_scale.csv`` (2-column Fossen x/y),
        reduces the path to start near ``initial_ship_pos``, spaces
        checkpoints evenly, and converts each one to 3D AirSim
        coordinates via ``coord.fossen_to_airsim``.

        Each checkpoint dict contains:
            * ``pos``              - 2D Fossen position ``(x_east, y_north)``.
            * ``pos_airsim_3d``    - 3D AirSim position ``(ax, ay, az)``.
            * ``radius``           - Proximity radius for checkpoint hit.
            * ``reward``           - Reward granted on reaching this checkpoint.
            * ``perpendicular_line`` - Guidance gate line endpoints.

        Raises:
            ValueError: If the CSV has an unexpected shape.
        """
        csv_path = _ENVIRONMENTS_DIR / "trajectory_points_no_scale.csv"
        path = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError(f"Trajectory CSV has unexpected shape: {path.shape}")

        distances = [np.linalg.norm(p - self.initial_ship_pos) for p in path]
        closest_idx = int(np.argmin(distances))
        path = path[closest_idx + 1:]

        path = create_checkpoints_from_simple_path(path, CHECKPOINTS_DISTANCE)

        if len(path) == 0:
            path = [self.initial_ship_pos.copy()]
        else:
            path = np.insert(path, 0, self.initial_ship_pos.copy(), axis=0)

        checkpoints = []
        for i, point in enumerate(path):
            fossen_pos = np.array(point, dtype=np.float32)

            # T-011: 3D coordinate transformation (Fossen -> AirSim)
            ax, ay, az = self.coord.fossen_to_airsim(
                float(fossen_pos[0]), float(fossen_pos[1]),
            )

            checkpoints.append({
                "pos": fossen_pos,
                "pos_airsim_3d": np.array([ax, ay, az], dtype=np.float32),
                "radius": float(CHECKPOINT_AREA_SIZE),
                "reward": (i / max(1, len(path))) * 10.0,
            })

        if checkpoints:
            checkpoints[-1]["radius"] = float(TARGET_AREA_SIZE)
            checkpoints[-1]["reward"] = float(SUCCESS_REWARD)

        lines = _calculate_perpendicular_lines(checkpoints)
        for cp, line in zip(checkpoints, lines):
            cp["perpendicular_line"] = line

        self.checkpoints = checkpoints
        self.target_pos = (
            checkpoints[-1]["pos"] if checkpoints
            else self.target_fossen
        )

        from shapely.geometry import Polygon
        obstacles_csv = _ENVIRONMENTS_DIR / "env_Sche_250cm_no_scale.csv"
        obstacles = np.loadtxt(obstacles_csv, delimiter=",", skiprows=1)
        self.polygon_shape = Polygon(obstacles)

    def _draw_visualization(self) -> None:
        """Draw trajectory and polygon lines in UE5 for visual reference."""
        airsim = self._airsim

        self.bridge.flush_persistent_markers()
        time.sleep(0.2)

        # -- Polygon outline (green) --
        csv_path = _ENVIRONMENTS_DIR / "env_Sche_no_scale.csv"
        polygon = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        poly_points = []
        for p in polygon[::2]:
            ux, uy = self.coord.fossen_to_airsim_2d(p[0], p[1])
            poly_points.append(airsim.Vector3r(ux, uy, -2.0))

        chunk_size = 500
        for i in range(0, len(poly_points), chunk_size - 1):
            chunk = poly_points[i:i + chunk_size]
            if len(chunk) > 1:
                self.bridge.plot_line_strip(
                    chunk,
                    color_rgba=[0, 1, 0, 0.8],
                    thickness=5.0,
                    is_persistent=True,
                )

        # -- Trajectory (yellow) --
        traj_csv = _ENVIRONMENTS_DIR / "trajectory_points_no_scale.csv"
        traj_data = np.loadtxt(traj_csv, delimiter=",", skiprows=1)

        distances = [np.linalg.norm(p - self.initial_ship_pos) for p in traj_data]
        closest_idx = int(np.argmin(distances))
        traj_data = traj_data[closest_idx:]

        traj_points = []
        for p in traj_data:
            ux, uy = self.coord.fossen_to_airsim_2d(p[0], p[1])
            traj_points.append(airsim.Vector3r(ux, uy, -3.0))

        for i in range(0, len(traj_points), chunk_size - 1):
            chunk = traj_points[i:i + chunk_size]
            if len(chunk) > 1:
                self.bridge.plot_line_strip(
                    chunk,
                    color_rgba=[1, 1, 0, 1.0],
                    thickness=8.0,
                    is_persistent=True,
                )

        # -- Start (blue) and end (red) dots --
        start_ux, start_uy = self.coord.fossen_to_airsim_2d(
            float(self.initial_ship_pos[0]), float(self.initial_ship_pos[1])
        )
        end_ux, end_uy = self.coord.fossen_to_airsim_2d(
            float(traj_data[-1][0]), float(traj_data[-1][1])
        )

        self.bridge.plot_points(
            [airsim.Vector3r(start_ux, start_uy, -3.0)],
            color_rgba=[0, 0, 1, 1],
            size=30.0,
            is_persistent=True,
        )
        self.bridge.plot_points(
            [airsim.Vector3r(end_ux, end_uy, -3.0)],
            color_rgba=[1, 0, 0, 1],
            size=30.0,
            is_persistent=True,
        )

        logger.info("Trajectory and polygon drawn in UE5.")

    # -------------
    # Observation space  (T-009)
    # -------------
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """Create the 10-dimensional observation space.

        The layout is **identical** to ``PySimEnv`` so that policies
        trained in either environment can be transferred directly.

        Observation channels (all min-max normalised):
            0. **surge velocity**  ``u / surge_max``          -> [-1, 1]
            1. **sway  velocity**  ``v / sway_max``           -> [-1, 1]
            2. **yaw rate**        ``r / yaw_rate_max``       -> [-1, 1]
            3. **dist to current checkpoint** ``/ max_dist``  -> [ 0, 1]
            4. **dist to next checkpoint**    ``/ max_dist``  -> [ 0, 1]
            5. **signed cross-track error** ``/ (CP_DIST/2)`` -> [-2, 2]
            6. **heading error  -> current CP** ``/ pi``        -> [-1, 1]
            7. **heading error  -> next CP**    ``/ pi``        -> [-1, 1]
            8. **heading error || current seg** ``/ pi``        -> [-1, 1]
            9. **heading error || next seg**    ``/ pi``        -> [-1, 1]

        Returns:
            A ``gym.spaces.Box`` with shape ``(10,)`` and dtype
            ``float32``.
        """
        low = np.array([
            -1.0,  # 0  surge velocity
            -1.0,  # 1  sway velocity
            -1.0,  # 2  yaw rate
             0.0,  # 3  distance to current checkpoint
             0.0,  # 4  distance to next checkpoint
            -2.0,  # 5  cross-track error
            -1.0,  # 6  heading error (to current)
            -1.0,  # 7  heading error (to next)
            -1.0,  # 8  heading error parallel (current)
            -1.0,  # 9  heading error parallel (next)
        ], dtype=np.float32)

        high = np.array([
            1.0, 1.0, 1.0,
            1.0, 1.0,
            2.0,
            1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)

        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # -------------
    # AirSim state reading
    # -------------
    @staticmethod
    def _extract_ue5_timestamp(car_state) -> Optional[int]:
        """Return the UE5 state timestamp when available.

        Args:
            car_state: AirSim ``CarState`` object returned by ``getCarState``.

        Returns:
            The integer timestamp reported by UE5, or ``None`` when the field
            is missing or not parseable.
        """
        timestamp = getattr(car_state, "timestamp", None)
        if timestamp is None:
            return None
        try:
            return int(timestamp)
        except (TypeError, ValueError):
            return None

    def _wait_for_next_physics_update(self, previous_timestamp: Optional[int]):
        """Wait until UE5 exposes a newer physics state or the watchdog fires.

        Args:
            previous_timestamp: The most recently observed UE5 physics
                timestamp before issuing the new control command.

        Returns:
            A tuple ``(car_state, wait_time_s)`` containing the freshest
            ``CarState`` observed and the total wait duration in seconds.
        """
        started = time.monotonic()
        latest_car_state = None

        while True:
            latest_car_state = self.bridge.get_car_state()
            current_timestamp = self._extract_ue5_timestamp(latest_car_state)
            if previous_timestamp is None or current_timestamp is None:
                break
            if current_timestamp > previous_timestamp:
                break
            if time.monotonic() - started >= self._physics_update_timeout:
                logger.warning(
                    "UE5 physics update watchdog expired after %.3f s; continuing with the latest state.",
                    time.monotonic() - started,
                )
                break
            time.sleep(self._state_poll_interval)

        wait_time = time.monotonic() - started
        if wait_time > self._latency_warning_threshold:
            logger.warning(
                "UE5 physics update latency %.3f s exceeded threshold %.3f s.",
                wait_time,
                self._latency_warning_threshold,
            )
        return latest_car_state, wait_time

    def _read_airsim_state(self, car_state=None) -> None:
        """Read vehicle state from UE5 and convert to Fossen frame.

        Queries ``simGetVehiclePose`` and ``getCarState`` from the
        AirSim client, then maps AirSim NED coordinates -> Fossen
        east/north via ``CoordTransform``.  Also derives the yaw rate
        from consecutive headings.
        """
        pose = self.bridge.get_vehicle_pose()
        if car_state is None:
            car_state = self.bridge.get_car_state()

        fx, fy = self.coord.airsim_to_fossen(
            pose.position.x_val, pose.position.y_val
        )
        self.ship_pos = np.array([fx, fy], dtype=np.float32)

        airsim_yaw = _yaw_from_quaternion(pose.orientation)
        self.ship_heading = self.coord.airsim_yaw_to_fossen_heading(airsim_yaw)

        self.ship_speed = min(car_state.speed, self.MAX_SPEED)
        self.ship_u = self.ship_speed
        self.ship_v = 0.0
        heading_diff = _normalize_angle(self.ship_heading - self.previous_heading)
        self.ship_r = heading_diff / max(self.time_step, EPSILON)
        self._last_ue5_timestamp = self._extract_ue5_timestamp(car_state)

    def _check_collision(self) -> bool:
        """Check for collision via AirSim API.

        Returns:
            ``True`` if the vehicle has collided with any object.
        """
        info = self.bridge.get_collision_info()
        return info.has_collided

    # -------------
    # Observation  (T-009)
    # -------------
    def _get_obs(self) -> np.ndarray:
        """Build the 10-dim normalised observation vector.

        All values are min-max normalised to match the ranges defined
        in ``_initialize_observation_space`` so that a policy trained on
        ``PySimEnv`` can be deployed here without any rescaling.

        Normalisation strategy:
            * **Velocities** - divided by the ship-spec hard limits and
              clipped to [-1, 1].
            * **Distances** - divided by the diagonal of the grid
              (``max_dist``).
            * **Cross-track error** - divided by half the checkpoint
              spacing; sign encodes port/starboard.
            * **Heading errors** - divided by pi.

        Returns:
            A ``np.float32`` array of shape ``(10,)`` clipped to
            ``observation_space``.
        """
        norm_velocities = np.array([
            np.clip(self.ship_u / self.ship_specs.surge_limits[1], -1, 1),
            np.clip(self.ship_v / self.ship_specs.sway_limits[1], -1, 1),
            np.clip(self.ship_r / self.ship_specs.yaw_rate_limits[1], -1, 1),
        ], dtype=np.float32)

        cur_idx = min(self.checkpoint_index, len(self.checkpoints) - 1)
        prev_idx = max(0, cur_idx - 1)
        next_idx = min(cur_idx + 1, len(self.checkpoints) - 1)

        cur_pos = self.checkpoints[cur_idx]["pos"]
        prev_pos = self.checkpoints[prev_idx]["pos"]
        next_pos = self.checkpoints[next_idx]["pos"]

        norm_distance = np.linalg.norm(self.ship_pos - cur_pos) / self.max_dist
        norm_next_distance = np.linalg.norm(self.ship_pos - next_pos) / self.max_dist

        cross_error = self._distance_from_point_to_line(
            self.ship_pos, prev_pos, cur_pos
        )
        norm_cross_error = cross_error / (CHECKPOINTS_DISTANCE / 2.0)
        track_vec = cur_pos - prev_pos
        ship_vec = self.ship_pos - prev_pos
        
        # 2D cross product: x1*y2 - y1*x2
        cross_2d = track_vec[0] * ship_vec[1] - track_vec[1] * ship_vec[0]
        norm_cross_error *= 1.0 if cross_2d >= 0.0 else -1.0
        self.cross_error = cross_error

        heading_error = self._calc_heading_error(
            _safe_heading_from_vector(cur_pos - self.ship_pos),
            self.ship_heading, 0.0
        )
        heading_error2 = self._calc_heading_error(
            _safe_heading_from_vector(next_pos - self.ship_pos),
            self.ship_heading, 0.0
        )
        heading_error_parallel = self._calc_heading_error(
            _safe_heading_from_vector(cur_pos - prev_pos),
            self.ship_heading, 0.0
        )
        heading_error_parallel2 = self._calc_heading_error(
            _safe_heading_from_vector(next_pos - prev_pos),
            self.ship_heading, 0.0
        )

        obs = np.array([
            norm_velocities[0],
            norm_velocities[1],
            norm_velocities[2],
            norm_distance,
            norm_next_distance,
            norm_cross_error,
            heading_error / np.pi,
            heading_error2 / np.pi,
            heading_error_parallel / np.pi,
            heading_error_parallel2 / np.pi,
        ], dtype=np.float32)

        # Safety clamp: guarantee the vector lies inside observation_space.
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # -------------
    # Step & Reset
    # -------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one RL step in UE5.

        Applies the action through the actuator model, advances the
        UE5 simulation, reads back the vehicle state, and computes
        the reward.

        Args:
            action: Array of shape ``(2,)`` containing
                ``[steering, throttle]`` each in ``[-1, 1]``.

        Returns:
            A tuple ``(observation, reward, terminated, truncated, info)``
            where *info* contains ``cross_track_error`` and
            ``checkpoint_index``.

        Raises:
            ValueError: If *action* does not have shape ``(2,)``.
        """
        step_started = time.monotonic()
        action = validate_policy_action(action, action_name="AirSimEnv.step action")

        self.previous_ship_pos = self.ship_pos.copy()
        self.previous_heading = self.ship_heading
        previous_timestamp = self._last_ue5_timestamp

        actuator = self._actuators.apply_smoothing(action, self.enable_smoothing)

        # Throttle: apply the same affine mapping used by Fossen ActionMapper so
        # that the same policy action produces equivalent thrust in both envs.
        #   action[1] = -1  ->  min_thrust = 0.1  (slow ahead)
        #   action[1] =  0  ->  0.55             (half ahead)
        #   action[1] = +1  ->  max_thrust = 1.0  (full ahead)
        # We keep a separate EMA on the *signed* action[1] (not abs()) so that
        # near-zero actions correctly map to mid-range throttle (~0.55), not idle.
        _alpha = 0.3
        if self.enable_smoothing:
            self._smooth_action1 = _alpha * float(action[1]) + (1.0 - _alpha) * self._smooth_action1
        else:
            self._smooth_action1 = float(action[1])
        controls = map_fossen_policy_action(
            self._airsim,
            self.ship_specs,
            steering=float(actuator.rudder_angle),
            signed_throttle_action=self._smooth_action1,
        )
        self.last_controls = {
            "steering": float(controls.steering),
            "throttle": float(controls.throttle),
            "brake": float(controls.brake),
        }

        if self.use_pause_stepping:
            # When the sim is paused, setCarControls blocks indefinitely waiting
            # for a physics-thread acknowledgement that never arrives.
            # Fix: unpause first so the RPC returns immediately, then
            # simContinueForFrames advances N frames and auto-pauses again.
            self.bridge.set_pause(False)
            self.bridge.set_controls(controls)
            self.bridge.continue_for_frames(self.frames_per_step)
            car_state, physics_wait_s = self._wait_for_next_physics_update(previous_timestamp)
        else:
            self.bridge.set_controls(controls)
            time.sleep(self.time_step)
            car_state = self.bridge.get_car_state()
            physics_wait_s = 0.0

        self._read_airsim_state(car_state=car_state)
        self.performed_action = self._actuators.current_actuator_state.copy()
        self.step_count += 1
        self.last_physics_wait_s = float(physics_wait_s)
        self.last_step_latency_s = float(time.monotonic() - step_started)

        reward, terminated = self._calculate_reward()

        # T-012: log per-step telemetry to episode CSV
        self._write_episode_log_row(reward)

        truncated = False
        if self.step_count >= self.max_steps and not terminated:
            reward = -250.0
            truncated = True

        # T-012: expose cross-track error and checkpoint progress
        info = {
            "cross_track_error": float(self.cross_error),
            "checkpoint_index": int(self.checkpoint_index),
            "physics_wait_s": float(self.last_physics_wait_s),
            "step_latency_s": float(self.last_step_latency_s),
            "ue5_timestamp": int(self._last_ue5_timestamp or 0),
        }

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the UE5 environment for a new episode.

        Resets the AirSim vehicle, re-pauses the simulation (if
        ``use_pause_stepping``), and teleports the ship back to the
        initial Fossen position.

        Args:
            seed: Optional RNG seed for reproducibility.
            **kwargs: Forwarded to ``gym.Env.reset``.

        Returns:
            A tuple ``(observation, info)``.
        """
        if seed is not None:
            super().reset(seed=seed)

        self.bridge.reset()

        if self.use_pause_stepping:
            self.bridge.set_pause(True)

        self.ship_pos = self.initial_ship_pos.copy()
        self.previous_ship_pos = self.initial_ship_pos.copy()
        self.previous_heading = 0.0
        self.ship_heading = 0.0
        self.ship_speed = 0.0
        self.ship_u = 0.0
        self.ship_v = 0.0
        self.ship_r = 0.0
        self._actuators.reset()
        self.performed_action = np.zeros(2, dtype=np.float32)
        self.last_controls = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}
        self.checkpoint_index = 1
        self.cross_error = 0.0
        self.step_count = 0
        self._smooth_action1 = 0.0  # reset throttle EMA for new episode
        self._episode_number += 1  # T-012: advance episode counter for CSV grouping

        initial_heading = 0.0
        if len(self.checkpoints) > self.checkpoint_index:
            direction = self.checkpoints[self.checkpoint_index]["pos"] - self.ship_pos
            initial_heading = _normalize_angle(_safe_heading_from_vector(direction))
        self.ship_heading = initial_heading

        current_pose = self.bridge.get_vehicle_pose()
        current_z = current_pose.position.z_val

        start_ax, start_ay, _ = self.coord.fossen_to_airsim(
            float(self.initial_ship_pos[0]),
            float(self.initial_ship_pos[1]),
        )
        initial_yaw = self.coord.fossen_heading_to_airsim_yaw(initial_heading)
        pose = self._airsim.Pose(
            self._airsim.Vector3r(start_ax, start_ay, current_z),
            self._airsim.Quaternionr(0, 0, math.sin(initial_yaw / 2), math.cos(initial_yaw / 2)),
        )
        self.bridge.set_vehicle_pose(pose, ignore_collision=True)
        time.sleep(0.2)

        self._read_airsim_state()

        if self.verbose:
            logger.debug(
                "Reset - ship at (%.0f, %.0f), heading %.1f deg",
                self.ship_pos[0], self.ship_pos[1],
                math.degrees(self.ship_heading),
            )

        return self._get_obs(), {}

    def close(self) -> None:
        """Disconnect from AirSim and restore real-time simulation speed.

        Safe to call multiple times.  Resets the simulation speed to 1.0
        (real-time) before disconnecting so UE5 is left in a usable state.
        """
        try:
            self.bridge.flush_persistent_markers()
            self.bridge.set_vehicle_pose(self._spawn_pose, ignore_collision=True)
            self.bridge.set_simulation_speed(1.0)

            if self.use_pause_stepping:
                self.bridge.set_pause(False)

            self.bridge.disconnect()
            logger.info("Vehicle reset to spawn, simulation speed restored to 1.0x.")
        except Exception as e:
            logger.warning("Error during AirSimEnv.close(): %s", e)
        finally:
            # T-012: always flush and close episode log on env teardown
            self._close_episode_log()

    # -------------
    # Reward (mirrors PySimEnv exactly)
    # -------------
    def _calculate_reward(self) -> Tuple[float, bool]:
        """Compute reward and termination - same logic as PySimEnv."""
        prev_idx = max(0, self.checkpoint_index - 1)
        prev_pos = self.checkpoints[prev_idx]["pos"]

        current_cp = self.checkpoints[self.checkpoint_index]
        cur_pos = current_cp["pos"]

        path_vec = cur_pos - prev_pos
        path_length = max(np.linalg.norm(path_vec), EPSILON)
        path_unit = path_vec / path_length

        rel_prev = self.previous_ship_pos - prev_pos
        rel_now = self.ship_pos - prev_pos
        proj_prev = np.dot(rel_prev, path_unit)
        proj_now = np.dot(rel_now, path_unit)
        raw_progress = proj_now - proj_prev
        forward_reward = 4.0 * np.clip(raw_progress, -1.0, 1.0)

        heading_error = self._calc_heading_error(
            _safe_heading_from_vector(path_vec), self.ship_heading
        )
        heading_alignment_reward = abs(np.exp(-abs(heading_error))) - 1

        cross_track_error = self._distance_from_point_to_line(
            self.ship_pos, prev_pos, cur_pos
        )
        self.cross_error = cross_track_error
        cross_track_penalty = -(cross_track_error / 100.0)

        rudder_penalty = -0.2 * abs(self.performed_action[0])

        reward = (
            self.reward_weights["forward"] * forward_reward
            + self.reward_weights["alignment"] * heading_alignment_reward
            + self.reward_weights["cross_track"] * cross_track_penalty
            + self.reward_weights["rudder"] * rudder_penalty
        )

        cp_dist = np.linalg.norm(self.ship_pos - cur_pos)
        if cp_dist <= current_cp["radius"]:
            if self.checkpoint_index == len(self.checkpoints) - 1:
                logger.info("Target REACHED at distance %.2f", cp_dist)
            self.checkpoint_index += 1
            self.step_count = 0
        elif self._is_near_perpendicular_line(current_cp):
            self.checkpoint_index += 1
            self.step_count = 0

        done = False
        heading_change = abs(self._calc_heading_error(
            self.previous_heading, self.ship_heading
        ))

        collision = self._check_collision()
        out_of_water = not self._ship_in_open_water(self.ship_pos)

        termination_conditions = [
            cross_track_error > CROSS_TRACK_TERMINATION_MULTIPLIER * CHECKPOINTS_DISTANCE,
            collision,
            out_of_water,
            heading_change > HEADING_CHANGE_THRESHOLD,
        ]

        if any(termination_conditions):
            reward = -250
            done = True
            if self.verbose:
                reasons = ["cross-track", "collision", "out-of-water", "heading-change"]
                triggered = [r for r, c in zip(reasons, termination_conditions) if c]
                logger.debug("Terminated: %s", ", ".join(triggered))

        if self.checkpoint_index >= len(self.checkpoints):
            done = True

        return reward, done

    # -------------
    # Utility methods (match PySimEnv)
    # -------------
    @staticmethod
    def _calc_heading_error(target: float, current: float, dead_zone_deg: float = 0.0) -> float:
        """Compute the wrapped heading error between two angles.

        Args:
            target: Desired heading angle in radians.
            current: Current heading angle in radians.
            dead_zone_deg: Optional symmetric dead zone in degrees. Errors with
                smaller magnitude are clamped to ``0.0``.

        Returns:
            The wrapped heading error in radians.
        """
        error = _normalize_angle(target - current)
        if dead_zone_deg > 0:
            dead_zone = np.radians(dead_zone_deg)
            return 0.0 if abs(error) < dead_zone else error
        return error

    def _ship_in_open_water(self, pos: np.ndarray) -> bool:
        """Check if ship is inside the navigable water polygon."""
        from shapely.geometry import Point
        x, y = pos[0], pos[1]
        minx, miny, maxx, maxy = self.polygon_shape.bounds
        if not (minx <= x <= maxx and miny <= y <= maxy):
            return False
        return self.polygon_shape.contains(Point(pos))

    def _is_near_perpendicular_line(self, checkpoint: dict) -> bool:
        """Check whether the ship is close enough to cross a checkpoint line.

        Args:
            checkpoint: Checkpoint dictionary containing a
                ``"perpendicular_line"`` entry.

        Returns:
            ``True`` when the ship is within
            ``PERPENDICULAR_LINE_PROXIMITY`` of the checkpoint line.
        """
        line_start, line_end = checkpoint["perpendicular_line"]
        return self._distance_from_point_to_line(
            self.ship_pos, line_start, line_end
        ) <= PERPENDICULAR_LINE_PROXIMITY

    @staticmethod
    def _distance_from_point_to_line(
        point: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray,
    ) -> float:
        """Distance from a point to a line segment."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        mag_sq = np.dot(line_vec, line_vec)
        if mag_sq == 0.0:
            return float(np.linalg.norm(point - line_start))
        proj = np.dot(point_vec, line_vec) / mag_sq
        projected = line_start + proj * line_vec
        return float(np.linalg.norm(point - projected))
