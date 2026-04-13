"""Gymnasium environment for RL training inside UE5 via CosysAirSim.

This module provides a :class:`AirSimEnv`, a drop-in replacement for
``PySimEnv`` that uses Unreal Engine 5's physics via the CosysAirSim API
instead of the local framework dynamics model.

Coordinate frames
-----------------
- **Framework**: x = east, y = north, heading from east (CCW positive).
- **AirSim**: x = north, y = east (NED), yaw from north (CW positive).
- **Transform**: ``AirSim_yaw = pi/2 - framework_heading``.

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
from typing import Dict, List, Optional, Tuple

from fw.environments.base_env import BaseEnv
from fw.environments.tools import create_checkpoints_from_simple_path
from airsim.config.airsim_config import (
    DEFAULT_AIRSIM_SAVE_SLOT,
    DEFAULT_AIRSIM_SIMULATION_SPEED,
)
from airsim.communication.coord_transform import CoordTransform
from fw.common.airsim.tools.angles import (
    normalize_angle as _normalize_angle,
    yaw_from_quaternion as _yaw_from_quaternion,
)
from airsim.communication.control_mapping import (
    map_framework_actuator_state,
    validate_policy_action,
)
from fw.ships.action_mapper import ActionMapper
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
CHECKPOINT_AREA_SIZE = 9.0
TARGET_AREA_SIZE = 14.0
PERPENDICULAR_LINE_LENGTH = 50.0
PERPENDICULAR_LINE_PROXIMITY = 8.0
HEADING_CHANGE_THRESHOLD = np.pi / 2.0
CROSS_TRACK_TERMINATION_MULTIPLIER = 2.0
EPSILON = 1e-9
EARLY_CROSS_TRACK_TERMINATION_DISTANCE = 400.0
HIGH_CROSS_TRACK_PENALTY_START = 175.0
HIGH_CROSS_TRACK_TIME_PENALTY_PATIENCE = 80
LOW_SPEED_THRESHOLD_MPS = 0.05
LOW_SPEED_PATIENCE_STEPS = 250
MIN_PROGRESS_STEP_M = 0.015
NO_PROGRESS_PATIENCE_STEPS = 350
LATE_CHECKPOINT_STRONGER_CROSS_TRACK_START = 3
CENTERLINE_BONUS_WIDTH_M = 45.0
CHECKPOINT_ENTRY_BASE_REWARD = 30.0
CHECKPOINT_ENTRY_REWARD_SCALE = 2.0
FINAL_CHECKPOINT_ENTRY_REWARD = 120.0
LOOKAHEAD_BLEND_START_PROGRESS = 0.55
LOOKAHEAD_BLEND_FULL_PROGRESS = 0.90
OFFTRACK_THROTTLE_ERROR_START_M = 60.0
COLLISION_TERMINATION_REWARD = -400.0
CHECKPOINT_PROGRESS_IMPROVEMENT_EPSILON_M = 2.5
COLLISION_LIKE_STOP_SPEED_MPS = 0.05
COLLISION_LIKE_PREVIOUS_SPEED_MPS = 1.5
COLLISION_LIKE_STOP_PATIENCE_STEPS = 8
COLLISION_LIKE_STOP_MIN_THROTTLE = 0.35
OBSTACLE_CONTACT_PATIENCE_STEPS = 12
OBSTACLE_CONTACT_MIN_THROTTLE = 0.35
OBSTACLE_CONTACT_MAX_PROGRESS_M = 0.05
OBSTACLE_CONTACT_LOW_SPEED_MPS = 0.50
OBSTACLE_CONTACT_MIN_CROSS_TRACK_M = 30.0
OBSTACLE_CONTACT_SPEED_DROP_MPS = 1.25
CHECKPOINT_DISTANCE_PROGRESS_SCALE_M = 5.0
CROSS_TRACK_CONVERGENCE_SCALE_M = 4.0
OFFTRACK_RECOVERY_BLEND_START_M = 20.0
OFFTRACK_RECOVERY_BLEND_FULL_M = 160.0
RECOVERY_LOOKAHEAD_DISTANCE_M = 70.0
OFFTRACK_SPEED_PENALTY_START_FRACTION = 0.70
PY_SIM_TARGET_SPEED_MPS = 3.4
HEADLESS_WATCHDOG_FALLBACK_PATIENCE = 2
MIN_TURN_TARGET_SPEED_MPS = 1.5
HEADING_CONVERGENCE_SCALE_RAD = math.radians(8.0)
UNDERSTEER_STEERING_THRESHOLD = 0.18
MAX_TURN_TARGET_SPEED_REDUCTION = 0.78
TURN_SPEED_EXCESS_PENALTY_SCALE = 0.60
TURN_STEERING_RELIEF_SCALE = 0.90
BASE_REQUIRED_TURN_STEERING = 0.08
MAX_REQUIRED_TURN_STEERING = 0.42
TURN_ACTION_GOVERNOR_START = 0.55
TURN_ACTION_MAX_STEERING_GAIN = 1.35
TURN_ACTION_OFFTRACK_THROTTLE_SCALE = 0.55
TURN_ACTION_HEADING_INJECTION_DEAD_ZONE_RAD = math.radians(2.0)
CURRICULUM_DEFAULT_WINDOW_SIZE = 8
CURRICULUM_DEFAULT_FRONTIER_INCREMENT = 4
CURRICULUM_DEFAULT_WINDOW_INCREMENT = 6
CURRICULUM_DEFAULT_SUCCESS_PATIENCE = 2
CHECKPOINT_STAGNATION_PATIENCE_STEPS = 420
CHECKPOINT_STAGNATION_MIN_DISTANCE_M = 120.0
GUIDANCE_ASSIST_DEFAULT_ENABLED = True
GUIDANCE_RESIDUAL_STEERING_SCALE = 0.45
GUIDANCE_RESIDUAL_THROTTLE_SCALE = 0.55
GUIDANCE_STEERING_KP = 1.10
GUIDANCE_STEERING_KD = 0.35
GUIDANCE_THROTTLE_KP = 2.00
GUIDANCE_OFFTRACK_SLOWDOWN_START_M = 25.0
GUIDANCE_OFFTRACK_SLOWDOWN_FULL_M = 180.0
GUIDANCE_MIN_FORWARD_THROTTLE = 0.12
GUIDANCE_MAX_REVERSE_THROTTLE = -0.30
# Default calibration file location
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIBRATION_PATH = _THIS_DIR.parent / "config" / "calibration.json"
_ENVIRONMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "fw" / "environments"


# ----------------
# Helpers
# ----------------
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
        """Estimate a smoothed tangent vector at a checkpoint index.

        Args:
            cps: Ordered checkpoint dictionaries containing ``pos`` vectors.
            idx: Checkpoint index to evaluate.

        Returns:
            A normalized tangent vector in ``float32`` when possible.
        """
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

    This environment mirrors ``PySimEnv`` but replaces the local framework physics
    model with UE5's physics engine. The agent sends ``[steering, throttle]``
    commands, and the environment reads back position, heading, speed, and
    collision info from UE5.

    Attributes:
        bridge: The :class:`~airsim.communication.airsim_bridge.AirSimBridge`
            that manages the connection to UE5.  The underlying
            ``cosysairsim.CarClient`` is accessible via ``bridge.client``.
        coord: The coordinate transform between framework and AirSim frames.
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
        draw_calibration_reference: bool = True,
        no_rendering_mode: bool = False,
        verbose: bool = False,
        episode_log_path: str = "",
        progress_log_interval_steps: int = 0,
        log_pause_watchdog_warnings: bool = False,
        use_sim_time_stepping: bool = False,
        sim_time_per_action: Optional[float] = None,
        simulation_speed: float = DEFAULT_AIRSIM_SIMULATION_SPEED,
        physics_update_timeout: float = 1.0,
        state_poll_interval: float = 0.005,
        curriculum_enabled: bool = False,
        curriculum_window_size: int = CURRICULUM_DEFAULT_WINDOW_SIZE,
        curriculum_target_checkpoint: Optional[int] = None,
        curriculum_frontier_increment: int = CURRICULUM_DEFAULT_FRONTIER_INCREMENT,
        curriculum_window_increment: int = CURRICULUM_DEFAULT_WINDOW_INCREMENT,
        curriculum_success_patience: int = CURRICULUM_DEFAULT_SUCCESS_PATIENCE,
        curriculum_randomization_scale: float = 0.0,
        guidance_assist_enabled: bool = GUIDANCE_ASSIST_DEFAULT_ENABLED,
        guidance_residual_steering_scale: float = GUIDANCE_RESIDUAL_STEERING_SCALE,
        guidance_residual_throttle_scale: float = GUIDANCE_RESIDUAL_THROTTLE_SCALE,
        latency_warning_threshold: Optional[float] = None,
    ) -> None:
        """Initialise the AirSim training environment.

        Args:
            ship_pos: Initial ship position in framework coords [x_east, y_north].
            target_pos: Target position in framework coords.
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
            draw_calibration_reference: Whether to draw the shared
                calibration overlay (ridges, route line, start/end markers)
                used by the visualization mode.
            no_rendering_mode: Whether to disable full UE5 scene rendering
                through Unreal's ``ShowFlag.Rendering`` console flag.
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
        self.training_mode = bool(training_mode)
        self.draw_calibration_reference = bool(draw_calibration_reference)
        self.no_rendering_mode = bool(no_rendering_mode)
        self.verbose = verbose
        self._progress_log_interval_steps = max(0, int(progress_log_interval_steps))
        self._log_pause_watchdog_warnings = bool(log_pause_watchdog_warnings)
        self._use_sim_time_stepping = bool(use_sim_time_stepping)
        self._sim_time_per_action = float(
            sim_time_per_action if sim_time_per_action is not None else time_step
        )
        self._last_sim_step_dt = max(self._sim_time_per_action, EPSILON)
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
        self.episode_step_count: int = 0

        # -- Episode-level telemetry CSV (T-012) --
        self._episode_log_path: str = episode_log_path
        self._episode_log_file = None
        self._episode_csv_writer = None
        self._episode_number: int = 0
        self._episode_started: bool = False
        self._curriculum_enabled = bool(curriculum_enabled)
        self._curriculum_window_size = max(1, int(curriculum_window_size))
        self._curriculum_target_checkpoint = (
            int(curriculum_target_checkpoint)
            if curriculum_target_checkpoint is not None
            else None
        )
        self._curriculum_frontier_increment = max(1, int(curriculum_frontier_increment))
        self._curriculum_window_increment = max(
            self._curriculum_frontier_increment,
            int(curriculum_window_increment),
        )
        self._curriculum_success_patience = max(1, int(curriculum_success_patience))
        self._curriculum_randomization_scale = max(0.0, float(curriculum_randomization_scale))
        self._curriculum_success_streak = 0
        self._curriculum_reference_checkpoints: List[dict] = []
        self._curriculum_active_start_index = 0
        self._curriculum_active_end_index = 0
        self._guidance_assist_enabled = bool(guidance_assist_enabled)
        self._guidance_residual_steering_scale = float(
            np.clip(guidance_residual_steering_scale, 0.0, 1.0)
        )
        self._guidance_residual_throttle_scale = float(
            np.clip(guidance_residual_throttle_scale, 0.0, 1.0)
        )
        self.last_policy_action = np.zeros(2, dtype=np.float32)
        self.last_guidance_action = np.zeros(2, dtype=np.float32)
        self.last_command_action = np.zeros(2, dtype=np.float32)

        # -- Ship specifications & actuator model --
        self.ship = create_ship(ShipModel.MYZAKO_DEFAULT)
        self.ship_specs = self.ship.specifications
        self._actuators = ActuatorModel(self.ship_specs)

        # Match the local framework surge limit for identical observation normalisation.
        # MYZAKO_DEFAULT: surge_limits = (0.0, 5.0).  UE5 top speed is 5.6 m/s
        # but we cap at the same value the policy was trained against.
        self.MAX_SPEED = self.ship_specs.surge_limits[1]  # 5.0 m/s

        # Positions in framework coordinates
        self.initial_ship_pos = np.array(
            ship_pos if ship_pos is not None else [2245, -47],
            dtype=np.float32,
        )
        self._base_initial_ship_pos = self.initial_ship_pos.copy()
        self.randomization_scale = 1.0
        self.target_framework = np.array(
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
        self.bridge.set_rendering_enabled(not self.no_rendering_mode)
        self.bridge.set_simulation_speed(simulation_speed)
        logger.info("Simulation speed set to %.1fx.", simulation_speed)

        # -- Auto drift correction --
        # Capture spawn pose AFTER world load so it reflects the loaded position.
        self._spawn_pose = self.bridge.get_vehicle_pose()
        self._live_debug_z = float(self._spawn_pose.position.z_val - 2.0)
        cal_path = Path(calibration_path) if calibration_path else _DEFAULT_CALIBRATION_PATH
        if calibration is None and cal_path.exists():
            from fw.common.airsim.tools.calibrate_coords import load_saved_calibration_reference

            self.coord, _ = load_saved_calibration_reference(
                self.bridge.client,
                calibration_path=cal_path,
                draw_reference=self.draw_calibration_reference,
                ship_pos=(float(self.initial_ship_pos[0]), float(self.initial_ship_pos[1])),
                target_pos=(float(self.target_framework[0]), float(self.target_framework[1])),
            )
        elif cal_path.exists():
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

        self._headless_watchdog_hits = 0
        self._force_continuous_stepping = False
        if self.use_pause_stepping and not self._force_continuous_stepping:
            self.bridge.set_pause(True)
            logger.info("Simulation paused - using deterministic frame stepping.")
        else:
            self.bridge.set_pause(False)
            logger.info("Simulation running continuously for accelerated training.")

        # -- Draw fallback visualization when no saved calibration overlay is available --
        if not (calibration is None and cal_path.exists()) and self.draw_calibration_reference:
            self._draw_visualization()

        # -- Load checkpoints (same data as PySimEnv) --
        self._load_checkpoints()
        self._curriculum_reference_checkpoints = self._build_checkpoints_for_initial_position(
            self._base_initial_ship_pos,
        )
        total_reference_targets = max(len(self._curriculum_reference_checkpoints) - 1, 1)
        seeded_target_checkpoint = (
            self._curriculum_target_checkpoint
            if self._curriculum_target_checkpoint is not None
            else self._curriculum_window_size
        )
        self._curriculum_target_checkpoint = int(
            np.clip(seeded_target_checkpoint, 1, total_reference_targets)
        )
        self._curriculum_window_size = int(
            np.clip(self._curriculum_window_size, 1, total_reference_targets)
        )

        # -- State tracking --
        self.ship_pos = self.initial_ship_pos.copy()
        self.previous_ship_pos = self.initial_ship_pos.copy()
        self.ship_heading = 0.0
        self.previous_heading = 0.0
        self.ship_speed = 0.0
        self.previous_speed = 0.0
        self.ship_u = 0.0  # surge
        self.ship_v = 0.0  # sway
        self.ship_r = 0.0  # yaw rate
        self.performed_action = np.zeros(2, dtype=np.float32)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.last_controls = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}
        self.checkpoint_index = 1
        self.cross_error = 0.0
        self.step_count = 0
        self._low_speed_steps = 0
        self._stalled_progress_steps = 0
        self._high_cross_track_steps = 0
        self._hard_stop_steps = 0
        self._obstacle_contact_steps = 0
        self._collision_latched = False
        self._headless_watchdog_hits = 0
        self._force_continuous_stepping = False
        self._best_checkpoint_distance = float("inf")
        self._checkpoint_stagnation_steps = 0
        self._last_raw_progress = 0.0
        self.max_dist = np.sqrt(2) * float(MAX_GRID_POS)
        self._curriculum_best_master_checkpoint = int(self._curriculum_target_checkpoint)

        # -- Gym spaces (identical to PySimEnv) --
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._initialize_observation_space()

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
        "checkpoint_step",
        "ship_x", "ship_y", "heading_deg", "speed",
        "cross_track_error", "checkpoint_index",
        "current_checkpoint_x", "current_checkpoint_y",
        "current_checkpoint_airsim_x", "current_checkpoint_airsim_y",
        "current_checkpoint_distance",
        "reward", "steering", "throttle",
        "physics_wait_s", "step_latency_s", "ue5_timestamp",
    ]

    def _get_active_checkpoint(self) -> dict:
        """Return the currently active checkpoint with index clamping."""
        return self.checkpoints[min(self.checkpoint_index, len(self.checkpoints) - 1)]

    def _open_episode_log(self) -> None:
        """Open (or re-open) the episode telemetry CSV for appending.

        Creates the file with a header row when it does not yet exist.
        Safe to call repeatedly - closes any previously open handle first.
        """
        self._close_episode_log()
        path = self._episode_log_path
        try:
            file_exists = os.path.isfile(path)
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
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
            current_checkpoint = self._get_active_checkpoint()
            self._episode_csv_writer.writerow({
                "episode": self._episode_number,
                "episode_step": self.episode_step_count,
                "checkpoint_step": self.step_count,
                "ship_x": float(self.ship_pos[0]),
                "ship_y": float(self.ship_pos[1]),
                "heading_deg": float(math.degrees(self.ship_heading)),
                "speed": float(self.ship_speed),
                "cross_track_error": float(self.cross_error),
                "checkpoint_index": int(self.checkpoint_index),
                "current_checkpoint_x": float(current_checkpoint["pos"][0]),
                "current_checkpoint_y": float(current_checkpoint["pos"][1]),
                "current_checkpoint_airsim_x": float(current_checkpoint["pos_airsim_3d"][0]),
                "current_checkpoint_airsim_y": float(current_checkpoint["pos_airsim_3d"][1]),
                "current_checkpoint_distance": float(
                    np.linalg.norm(self.ship_pos - current_checkpoint["pos"])
                ),
                "reward": float(reward),
                "steering": float(self.performed_action[0]),
                "throttle": float(self.performed_action[1]),
                "physics_wait_s": float(self.last_physics_wait_s),
                "step_latency_s": float(self.last_step_latency_s),
                "ue5_timestamp": int(self._last_ue5_timestamp or 0),
            })
            self._episode_log_file.flush()
        except Exception as exc:
            logger.warning("Episode log write failed: %s", exc)

    def _ensure_episode_started(self) -> None:
        """Advance the user-facing episode counter on the first real step only.

        ``DummyVecEnv`` auto-resets environments immediately after termination.
        If we incremented the counter inside ``reset()``, those hidden resets
        would make the console/log episode numbers jump (1, 3, 5, ...). By
        advancing the counter lazily on the first post-reset step, the episode
        number reflects only episodes that actually produced rollout data.
        """
        if not self._episode_started:
            self._episode_number += 1
            self._episode_started = True

    def _emit_progress_heartbeat(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        termination_reason: str = "",
    ) -> None:
        """Print a concise headless-training heartbeat to stdout.

        This is intentionally independent from logging levels so a user can
        still verify progress when UE5 rendering is disabled and the viewport
        looks frozen.
        """
        interval = self._progress_log_interval_steps
        should_print = (
            interval > 0
            and self.episode_step_count > 0
            and self.episode_step_count % interval == 0
        ) or terminated or truncated
        if not should_print:
            return

        print(
            (
                "[AirSimEnv][progress] "
                f"episode={self._episode_number} "
                f"step={self.episode_step_count} "
                f"checkpoint_step={self.step_count} "
                f"checkpoint={self.checkpoint_index}/{max(len(self.checkpoints) - 1, 0)} "
                f"reward={float(reward):.3f} "
                f"pos=({float(self.ship_pos[0]):.1f}, {float(self.ship_pos[1]):.1f}) "
                f"speed={float(self.ship_speed):.2f}m/s "
                f"xte={float(self.cross_error):.2f} "
                f"cp_dist={float(np.linalg.norm(self.ship_pos - self._get_active_checkpoint()['pos'])):.2f} "
                f"progress={float(self._last_raw_progress):.3f} "
                f"low_speed_steps={self._low_speed_steps} "
                f"stalled_steps={self._stalled_progress_steps} "
                f"hard_stop_steps={self._hard_stop_steps} "
                f"obstacle_contact_steps={self._obstacle_contact_steps} "
                f"cp_stagnation_steps={self._checkpoint_stagnation_steps} "
                f"wait={float(self.last_physics_wait_s):.3f}s "
                f"latency={float(self.last_step_latency_s):.3f}s "
                f"status={'done' if terminated else 'truncated' if truncated else 'running'}"
                f"{f' reason={termination_reason}' if termination_reason else ''}"
            ),
            flush=True,
        )

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

    def _checkpoint_entry_reward(self, checkpoint_index: int) -> float:
        """Return the bonus reward for entering a trajectory checkpoint."""
        cp = self.checkpoints[checkpoint_index]
        if checkpoint_index >= len(self.checkpoints) - 1:
            return FINAL_CHECKPOINT_ENTRY_REWARD
        return max(
            CHECKPOINT_ENTRY_BASE_REWARD,
            float(cp.get("reward", 0.0)) * CHECKPOINT_ENTRY_REWARD_SCALE,
        )

    def _halt_vehicle(self) -> None:
        """Immediately command neutral controls to stop post-termination drift."""
        try:
            controls = self._airsim.CarControls()
            controls.steering = 0.0
            controls.throttle = 0.0
            controls.brake = 1.0
            self.bridge.set_controls(controls)
            if self.use_pause_stepping:
                self.bridge.set_pause(True)
        except Exception as exc:
            logger.debug("Could not halt vehicle after termination: %s", exc)

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
        if self.use_pause_stepping and not self._force_continuous_stepping:
            self.bridge.set_pause(True)
        else:
            self.bridge.set_pause(False)

    def randomize(self, randomization_scale: Optional[float] = None) -> None:
        """Apply domain randomisation to the initial ship position.

        Adds a uniform random perturbation to ``initial_ship_pos`` and
        ensures the new position stays inside the navigable polygon.

        Args:
            randomization_scale: Maximum absolute perturbation (meters).
                Defaults to ``1.0`` if *None*.
        """
        if randomization_scale is not None:
            if randomization_scale <= 0:
                raise ValueError("randomization_scale must be positive")
            self.randomization_scale = float(randomization_scale)

        if self._curriculum_enabled and self._curriculum_reference_checkpoints:
            base_spawn, start_index, end_index = self._select_curriculum_route_window()
            scale = self._curriculum_randomization_scale
            if scale > 0.0:
                perturbation = np.random.default_rng().uniform(
                    low=-scale,
                    high=scale,
                    size=base_spawn.shape,
                ).astype(np.float32)
                candidate = np.clip(base_spawn + perturbation, MIN_GRID_POS, MAX_GRID_POS)
                if self._ship_in_open_water(candidate):
                    base_spawn = candidate.astype(np.float32)

            self.initial_ship_pos = np.asarray(base_spawn, dtype=np.float32)
            self._curriculum_active_start_index = int(start_index)
            self._curriculum_active_end_index = int(end_index)
            self._load_checkpoints(max_targets=end_index - start_index)
            return

        perturbation = np.random.default_rng().uniform(
            low=-self.randomization_scale,
            high=self.randomization_scale,
            size=self._base_initial_ship_pos.shape,
        ).astype(np.float32)

        candidate = np.clip(
            self._base_initial_ship_pos + perturbation,
            MIN_GRID_POS, MAX_GRID_POS,
        )

        # Only accept if the candidate is inside the river polygon.
        if self._ship_in_open_water(candidate):
            self.initial_ship_pos = candidate
            self._load_checkpoints()
        else:
            logger.debug(
                "Randomised position (%.0f, %.0f) outside polygon - keeping original.",
                candidate[0], candidate[1],
            )

    # -------------
    # Data loading  (T-011)
    # -------------
    def _build_checkpoints_for_initial_position(
        self,
        initial_ship_pos: np.ndarray,
        max_targets: Optional[int] = None,
    ) -> List[dict]:
        """Build a checkpoint list for a given spawn position."""
        csv_path = _ENVIRONMENTS_DIR / "trajectory_points_no_scale.csv"
        path = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError(f"Trajectory CSV has unexpected shape: {path.shape}")

        distances = [np.linalg.norm(p - initial_ship_pos) for p in path]
        closest_idx = int(np.argmin(distances))
        path = path[closest_idx + 1:]

        path = create_checkpoints_from_simple_path(path, CHECKPOINTS_DISTANCE)

        if len(path) == 0:
            path = [np.asarray(initial_ship_pos, dtype=np.float32).copy()]
        else:
            path = np.insert(path, 0, np.asarray(initial_ship_pos, dtype=np.float32).copy(), axis=0)

        if max_targets is not None:
            capped_length = max(2, int(max_targets) + 1)
            path = path[:capped_length]

        checkpoints = []
        for i, point in enumerate(path):
            framework_pos = np.array(point, dtype=np.float32)
            ax, ay, az = self.coord.framework_to_airsim(
                float(framework_pos[0]),
                float(framework_pos[1]),
            )
            checkpoints.append(
                {
                    "pos": framework_pos,
                    "pos_airsim_3d": np.array([ax, ay, az], dtype=np.float32),
                    "radius": float(CHECKPOINT_AREA_SIZE),
                    "reward": (i / max(1, len(path))) * 10.0,
                }
            )

        if checkpoints:
            checkpoints[-1]["radius"] = float(TARGET_AREA_SIZE)
            checkpoints[-1]["reward"] = float(SUCCESS_REWARD)

        lines = _calculate_perpendicular_lines(checkpoints)
        for cp, line in zip(checkpoints, lines):
            cp["perpendicular_line"] = line

        return checkpoints

    def _load_checkpoints(self, max_targets: Optional[int] = None) -> None:
        """Load active checkpoints and the water polygon for the current spawn."""
        checkpoints = self._build_checkpoints_for_initial_position(
            self.initial_ship_pos,
            max_targets=max_targets,
        )

        self.checkpoints = checkpoints
        self.target_pos = checkpoints[-1]["pos"] if checkpoints else self.target_framework

        from shapely.geometry import Polygon

        obstacles_csv = _ENVIRONMENTS_DIR / "env_Sche_250cm_no_scale.csv"
        obstacles = np.loadtxt(obstacles_csv, delimiter=",", skiprows=1)
        self.polygon_shape = Polygon(obstacles)

    def _select_curriculum_route_window(self) -> Tuple[np.ndarray, int, int]:
        """Select the active curriculum slice from the reference checkpoint route."""
        total_targets = max(len(self._curriculum_reference_checkpoints) - 1, 1)
        end_index = int(np.clip(self._curriculum_target_checkpoint, 1, total_targets))
        window_targets = int(np.clip(self._curriculum_window_size, 1, total_targets))
        start_index = max(0, end_index - window_targets)
        return (
            self._curriculum_reference_checkpoints[start_index]["pos"].copy(),
            start_index,
            end_index,
        )

    def _record_curriculum_result(
        self,
        terminated: bool,
        truncated: bool,
        termination_reason: str,
        reached_checkpoint_count: int,
    ) -> None:
        """Advance the curriculum only after repeated success at the frontier."""
        if not self._curriculum_enabled:
            return

        reached_master_checkpoint = int(
            min(
                self._curriculum_active_start_index + reached_checkpoint_count,
                len(self._curriculum_reference_checkpoints) - 1,
            )
        )
        self._curriculum_best_master_checkpoint = max(
            self._curriculum_best_master_checkpoint,
            reached_master_checkpoint,
        )

        success = bool(terminated and not truncated and termination_reason == "completed")
        if success:
            self._curriculum_success_streak += 1
        else:
            self._curriculum_success_streak = 0

        total_targets = max(len(self._curriculum_reference_checkpoints) - 1, 1)
        if (
            self._curriculum_success_streak >= self._curriculum_success_patience
            and self._curriculum_target_checkpoint < total_targets
        ):
            self._curriculum_target_checkpoint = min(
                total_targets,
                self._curriculum_target_checkpoint + self._curriculum_frontier_increment,
            )
            self._curriculum_window_size = min(
                total_targets,
                self._curriculum_window_size + self._curriculum_window_increment,
            )
            self._curriculum_success_streak = 0

    def _draw_visualization(self) -> None:
        """Draw a fallback trajectory overlay when no saved calibration exists."""
        airsim = self._airsim

        self.bridge.flush_persistent_markers()
        time.sleep(0.2)

        overlay_z = -60.0

        # -- Polygon outline (green) --
        csv_path = _ENVIRONMENTS_DIR / "env_Sche_no_scale.csv"
        polygon = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        poly_points = []
        for p in polygon[::2]:
            ux, uy = self.coord.framework_to_airsim_2d(p[0], p[1])
            poly_points.append(airsim.Vector3r(ux, uy, overlay_z))

        chunk_size = 500
        for i in range(0, len(poly_points), chunk_size - 1):
            chunk = poly_points[i:i + chunk_size]
            if len(chunk) > 1:
                self.bridge.plot_line_strip(
                    chunk,
                    color_rgba=[0.65, 1.0, 0.65, 1.0],
                    thickness=28.0,
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
            ux, uy = self.coord.framework_to_airsim_2d(p[0], p[1])
            traj_points.append(airsim.Vector3r(ux, uy, overlay_z - 2.0))

        for i in range(0, len(traj_points), chunk_size - 1):
            chunk = traj_points[i:i + chunk_size]
            if len(chunk) > 1:
                self.bridge.plot_line_strip(
                    chunk,
                    color_rgba=[1.0, 1.0, 0.6, 1.0],
                    thickness=36.0,
                    is_persistent=True,
                )

        # -- Start (blue) and end (red) dots --
        start_ux, start_uy = self.coord.framework_to_airsim_2d(
            float(self.initial_ship_pos[0]), float(self.initial_ship_pos[1])
        )
        end_ux, end_uy = self.coord.framework_to_airsim_2d(
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

    def _draw_live_debug_markers(self) -> None:
        """Draw non-persistent ship and active-checkpoint markers in UE5."""
        if self.no_rendering_mode:
            return

        ax, ay = self.coord.framework_to_airsim_2d(
            float(self.ship_pos[0]),
            float(self.ship_pos[1]),
        )
        self.bridge.plot_points(
            [self._airsim.Vector3r(ax, ay, self._live_debug_z)],
            color_rgba=[0.05, 0.85, 1.0, 1.0],
            size=18.0,
            is_persistent=False,
        )

        current_checkpoint = self._get_active_checkpoint()
        cp_ax, cp_ay = self.coord.framework_to_airsim_2d(
            float(current_checkpoint["pos"][0]),
            float(current_checkpoint["pos"][1]),
        )
        self.bridge.plot_points(
            [self._airsim.Vector3r(cp_ax, cp_ay, self._live_debug_z)],
            color_rgba=[1.0, 0.65, 0.1, 1.0],
            size=14.0,
            is_persistent=False,
        )

        line_start, line_end = current_checkpoint["perpendicular_line"]
        start_ax, start_ay = self.coord.framework_to_airsim_2d(
            float(line_start[0]),
            float(line_start[1]),
        )
        end_ax, end_ay = self.coord.framework_to_airsim_2d(
            float(line_end[0]),
            float(line_end[1]),
        )
        self.bridge.plot_line_strip(
            [
                self._airsim.Vector3r(start_ax, start_ay, self._live_debug_z),
                self._airsim.Vector3r(end_ax, end_ay, self._live_debug_z),
            ],
            color_rgba=[1.0, 0.45, 0.1, 1.0],
            thickness=8.0,
            is_persistent=False,
        )

    # -------------
    # Observation space  (T-009)
    # -------------
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """Create the 12-dimensional observation space.

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
            10. **previous rudder angle** (critical for policy)-> [-1, 1]
            11. **optimal LOS guidance heading** ``/ pi``        -> [-1, 1]

        Returns:
            A ``gym.spaces.Box`` with shape ``(12,)`` and dtype
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
            -1.0,  # 10 previous rudder angle
            -1.0,  # 11 optimal LOS guidance heading
        ], dtype=np.float32)

        high = np.array([
            1.0, 1.0, 1.0,
            1.0, 1.0,
            2.0,
            1.0, 1.0, 1.0, 1.0,
            1.0,  # 10 previous rudder angle
            1.0,  # 11 optimal LOS guidance heading
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

    def _wait_for_non_pause_state_update(
        self,
        previous_timestamp: Optional[int],
        previous_pose_xy: Tuple[float, float],
        previous_speed: float,
        minimum_wait_s: float,
    ):
        """Wait for a fresh state while UE5 runs continuously.

        Without this, very small wall-clock sleeps can sample the exact same
        stale pose/state before UE5 has processed the latest controls.
        """
        started = time.monotonic()
        latest_car_state = None

        initial_wait = max(0.0, float(minimum_wait_s))
        if initial_wait > 0.0:
            time.sleep(initial_wait)

        while True:
            latest_car_state = self.bridge.get_car_state()
            latest_pose = self.bridge.get_vehicle_pose()
            current_timestamp = self._extract_ue5_timestamp(latest_car_state)
            pose_dx = float(latest_pose.position.x_val) - float(previous_pose_xy[0])
            pose_dy = float(latest_pose.position.y_val) - float(previous_pose_xy[1])
            pose_delta = math.hypot(pose_dx, pose_dy)
            speed_delta = abs(float(latest_car_state.speed) - float(previous_speed))

            if previous_timestamp is not None and current_timestamp is not None:
                if current_timestamp != previous_timestamp:
                    break
                if current_timestamp == 0:
                    break

            if pose_delta > 1e-3 or speed_delta > 1e-3:
                break

            if time.monotonic() - started >= self._physics_update_timeout:
                logger.debug(
                    "Continuous-mode state watchdog expired after %.3f s; reading latest state.",
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
        """Read vehicle state from UE5 and convert to the framework frame.

        Queries ``simGetVehiclePose`` and ``getCarState`` from the
        AirSim client, then maps AirSim NED coordinates -> framework
        east/north via ``CoordTransform``.  Also derives the yaw rate
        from consecutive headings.
        """
        pose = self.bridge.get_vehicle_pose()
        if car_state is None:
            car_state = self.bridge.get_car_state()

        fx, fy = self.coord.airsim_to_framework(
            pose.position.x_val, pose.position.y_val
        )
        self.ship_pos = np.array([fx, fy], dtype=np.float32)

        airsim_yaw = _yaw_from_quaternion(pose.orientation)
        self.ship_heading = self.coord.airsim_yaw_to_framework_heading(airsim_yaw)

        self.ship_speed = min(car_state.speed, self.MAX_SPEED)
        self.ship_u = self.ship_speed
        self.ship_v = 0.0
        heading_diff = _normalize_angle(self.ship_heading - self.previous_heading)
        self.ship_r = heading_diff / max(self._last_sim_step_dt, EPSILON)
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
        """Build the 12-dim normalised observation vector.

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
            * **Previous rudder angle** - directly from performed_action[0].
            * **LOS guidance** - optimal heading error divided by pi.

        Returns:
            A ``np.float32`` array of shape ``(12,)`` clipped to
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

        # Compute optimal LOS guidance heading
        los_heading = self._los_guidance_heading()
        los_heading_error = self._calc_heading_error(
            los_heading, self.ship_heading, 0.0
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
            self.performed_action[0],  # Previous rudder angle (CRITICAL)
            los_heading_error / np.pi,  # Optimal LOS guidance heading
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
        self._ensure_episode_started()
        step_started = time.monotonic()
        policy_action = validate_policy_action(action, action_name="AirSimEnv.step action")
        guided_action = self._apply_guidance_residual_policy(policy_action)
        action = self._apply_turn_action_governor(guided_action)
        self.last_command_action = np.asarray(action, dtype=np.float32).copy()
        effective_pause_stepping = self.use_pause_stepping and not self._force_continuous_stepping
        sim_step_dt = (
            self._sim_time_per_action
            if effective_pause_stepping and self._use_sim_time_stepping
            else self.time_step
        )
        self._last_sim_step_dt = max(float(sim_step_dt), EPSILON)

        self.previous_ship_pos = self.ship_pos.copy()
        self.previous_heading = self.ship_heading
        self.previous_speed = float(self.ship_speed)
        self.previous_action = self.performed_action.copy()

        command = ActionMapper.map(action, self.ship_specs)
        actuator = self._actuators.step(
            command,
            dt=self._last_sim_step_dt,
            enable_smoothing=self.enable_smoothing,
        )
        controls = map_framework_actuator_state(
            self._airsim,
            self.ship_specs,
            rudder_angle=float(actuator.rudder_angle),
            thrust=float(actuator.thrust),
        )
        self.last_controls = {
            "steering": float(controls.steering),
            "throttle": float(controls.throttle),
            "brake": float(controls.brake),
        }

        if effective_pause_stepping:
            # When the sim is paused, setCarControls blocks indefinitely waiting
            # for a physics-thread acknowledgement that never arrives.
            # Fix: unpause first so the RPC returns immediately, then
            # simContinueForFrames schedules N frames and auto-pauses the sim.
            # simContinueForFrames is NON-blocking: poll simIsPause() to know
            # when UE5 has finished those N frames and re-paused.
            _pause_wait_started = time.monotonic()
            pause_watchdog_expired = False
            self.bridge.set_pause(False)
            self.bridge.set_controls(controls)
            if self._use_sim_time_stepping and hasattr(self.bridge, "continue_for_time"):
                self.bridge.continue_for_time(self._last_sim_step_dt)
            else:
                self.bridge.continue_for_frames(self.frames_per_step)
            while not self.bridge.is_paused():
                if time.monotonic() - _pause_wait_started >= self._physics_update_timeout:
                    message = "Pause watchdog expired after %.3f s; reading latest state."
                    if self._log_pause_watchdog_warnings:
                        logger.warning(
                            message,
                            time.monotonic() - _pause_wait_started,
                        )
                    else:
                        logger.debug(
                            message,
                            time.monotonic() - _pause_wait_started,
                        )
                    pause_watchdog_expired = True
                    break
                time.sleep(self._state_poll_interval)
            car_state = self.bridge.get_car_state()
            physics_wait_s = time.monotonic() - _pause_wait_started
            if pause_watchdog_expired and self.no_rendering_mode:
                self._headless_watchdog_hits += 1
                if self._headless_watchdog_hits >= HEADLESS_WATCHDOG_FALLBACK_PATIENCE:
                    self._force_continuous_stepping = True
                    self.bridge.set_pause(False)
                    logger.warning(
                        "Headless pause stepping stalled repeatedly; falling back to continuous stepping."
                    )
            else:
                self._headless_watchdog_hits = 0
        else:
            previous_timestamp = self._last_ue5_timestamp
            previous_pose = self.bridge.get_vehicle_pose()
            previous_pose_xy = (
                float(previous_pose.position.x_val),
                float(previous_pose.position.y_val),
            )
            previous_speed = float(self.ship_speed)
            self.bridge.set_pause(False)
            self.bridge.set_controls(controls)
            real_time_step = self.time_step / max(float(self._simulation_speed), EPSILON)
            car_state, physics_wait_s = self._wait_for_non_pause_state_update(
                previous_timestamp=previous_timestamp,
                previous_pose_xy=previous_pose_xy,
                previous_speed=previous_speed,
                minimum_wait_s=real_time_step,
            )
            self._headless_watchdog_hits = 0

        self._read_airsim_state(car_state=car_state)
        self._draw_live_debug_markers()
        self.performed_action = self._actuators.current_actuator_state.copy()
        self.step_count += 1
        self.episode_step_count += 1
        self.last_physics_wait_s = float(physics_wait_s)
        self.last_step_latency_s = float(time.monotonic() - step_started)

        reward, terminated, termination_reason = self._calculate_reward()
        current_checkpoint = self._get_active_checkpoint()

        # T-012: log per-step telemetry to episode CSV
        self._write_episode_log_row(reward)

        truncated = False
        if self.episode_step_count >= self.max_steps and not terminated:
            reward = -250.0
            truncated = True

        total_checkpoints = max(len(self.checkpoints) - 1, 1)
        reached_checkpoint_count = min(max(self.checkpoint_index - 1, 0), total_checkpoints)
        if terminated or truncated:
            self._record_curriculum_result(
                terminated=bool(terminated),
                truncated=bool(truncated),
                termination_reason=str(termination_reason),
                reached_checkpoint_count=int(reached_checkpoint_count),
            )

        # T-012: expose cross-track error and checkpoint progress
        info = {
            "episode_step": int(self.episode_step_count),
            "checkpoint_step": int(self.step_count),
            "cross_track_error": float(self.cross_error),
            "checkpoint_index": int(self.checkpoint_index),
            "reached_checkpoint_count": int(reached_checkpoint_count),
            "total_checkpoints": int(total_checkpoints),
            "checkpoint_fraction": float(reached_checkpoint_count / max(total_checkpoints, 1)),
            "physics_wait_s": float(self.last_physics_wait_s),
            "step_latency_s": float(self.last_step_latency_s),
            "ue5_timestamp": int(self._last_ue5_timestamp or 0),
            "current_checkpoint_pos": current_checkpoint["pos"].astype(np.float32).copy(),
            "current_checkpoint_airsim_3d": current_checkpoint["pos_airsim_3d"].astype(np.float32).copy(),
            "current_checkpoint_distance": float(
                np.linalg.norm(self.ship_pos - current_checkpoint["pos"])
            ),
            "raw_progress": float(self._last_raw_progress),
            "low_speed_steps": int(self._low_speed_steps),
            "stalled_progress_steps": int(self._stalled_progress_steps),
            "high_cross_track_steps": int(self._high_cross_track_steps),
            "hard_stop_steps": int(self._hard_stop_steps),
            "obstacle_contact_steps": int(self._obstacle_contact_steps),
            "checkpoint_stagnation_steps": int(self._checkpoint_stagnation_steps),
            "collision": bool(self._collision_latched),
            "completed": bool(termination_reason == "completed"),
            "termination_reason": str(termination_reason),
            "curriculum_enabled": bool(self._curriculum_enabled),
            "curriculum_active_start_checkpoint": int(self._curriculum_active_start_index),
            "curriculum_active_end_checkpoint": int(self._curriculum_active_end_index),
            "curriculum_target_checkpoint": int(self._curriculum_target_checkpoint),
            "curriculum_success_streak": int(self._curriculum_success_streak),
            "policy_action": self.last_policy_action.astype(np.float32).copy(),
            "guidance_action": self.last_guidance_action.astype(np.float32).copy(),
            "command_action": self.last_command_action.astype(np.float32).copy(),
            "rewards": float(reward),
            "terminations": bool(terminated),
            "truncations": bool(truncated),
        }

        self._emit_progress_heartbeat(
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            termination_reason=str(termination_reason),
        )

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the UE5 environment for a new episode.

        Resets the AirSim vehicle, re-pauses the simulation (if
        ``use_pause_stepping``), and teleports the ship back to the
        initial framework position.

        Args:
            seed: Optional RNG seed for reproducibility.
            **kwargs: Forwarded to ``gym.Env.reset``.

        Returns:
            A tuple ``(observation, info)``.
        """
        if seed is not None:
            super().reset(seed=seed)

        if self._curriculum_enabled and self._curriculum_active_end_index == 0:
            self.randomize(randomization_scale=self.randomization_scale)

        self.bridge.reset()

        if self.use_pause_stepping:
            self.bridge.set_pause(True)
        else:
            self.bridge.set_pause(False)

        self.ship_pos = self.initial_ship_pos.copy()
        self.previous_ship_pos = self.initial_ship_pos.copy()
        self.previous_heading = 0.0
        self.ship_heading = 0.0
        self.ship_speed = 0.0
        self.previous_speed = 0.0
        self.ship_u = 0.0
        self.ship_v = 0.0
        self.ship_r = 0.0
        self._actuators.reset()
        self.performed_action = np.zeros(2, dtype=np.float32)
        self.last_policy_action = np.zeros(2, dtype=np.float32)
        self.last_guidance_action = np.zeros(2, dtype=np.float32)
        self.last_command_action = np.zeros(2, dtype=np.float32)
        self.last_controls = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}
        self.checkpoint_index = 1
        self.cross_error = 0.0
        self.step_count = 0
        self.episode_step_count = 0
        self._low_speed_steps = 0
        self._stalled_progress_steps = 0
        self._high_cross_track_steps = 0
        self._hard_stop_steps = 0
        self._obstacle_contact_steps = 0
        self._collision_latched = False
        self._headless_watchdog_hits = 0
        self._best_checkpoint_distance = float("inf")
        self._checkpoint_stagnation_steps = 0
        self._last_raw_progress = 0.0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self._episode_started = False

        initial_heading = 0.0
        if len(self.checkpoints) > self.checkpoint_index:
            direction = self.checkpoints[self.checkpoint_index]["pos"] - self.ship_pos
            initial_heading = _normalize_angle(_safe_heading_from_vector(direction))
        self.ship_heading = initial_heading

        current_pose = self.bridge.get_vehicle_pose()
        current_z = current_pose.position.z_val

        start_ax, start_ay, _ = self.coord.framework_to_airsim(
            float(self.initial_ship_pos[0]),
            float(self.initial_ship_pos[1]),
        )
        initial_yaw = self.coord.framework_heading_to_airsim_yaw(initial_heading)
        pose = self._airsim.Pose(
            self._airsim.Vector3r(start_ax, start_ay, current_z),
            self._airsim.Quaternionr(0, 0, math.sin(initial_yaw / 2), math.cos(initial_yaw / 2)),
        )
        self.bridge.set_vehicle_pose(pose, ignore_collision=True)
        time.sleep(0.2)

        self._read_airsim_state()
        self._draw_live_debug_markers()

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
            self.bridge.set_rendering_enabled(True, force=True)
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
    # Reward and termination tuning for UE5 training
    # -------------
    def _update_stuck_trackers(
        self,
        raw_progress: float,
        cross_track_error: float,
        checkpoint_distance: float,
    ) -> None:
        """Track persistent low-speed, no-progress, and off-track behavior."""
        if abs(float(self.ship_speed)) <= LOW_SPEED_THRESHOLD_MPS:
            self._low_speed_steps += 1
        else:
            self._low_speed_steps = 0

        if float(raw_progress) <= MIN_PROGRESS_STEP_M:
            self._stalled_progress_steps += 1
        else:
            self._stalled_progress_steps = 0

        if float(cross_track_error) >= HIGH_CROSS_TRACK_PENALTY_START:
            self._high_cross_track_steps += 1
        else:
            self._high_cross_track_steps = 0

        throttle = abs(float(self.last_controls.get("throttle", 0.0)))
        collision_like_stop = (
            float(self.previous_speed) >= COLLISION_LIKE_PREVIOUS_SPEED_MPS
            and float(self.ship_speed) <= COLLISION_LIKE_STOP_SPEED_MPS
            and throttle >= COLLISION_LIKE_STOP_MIN_THROTTLE
        )
        if collision_like_stop:
            self._hard_stop_steps += 1
        else:
            self._hard_stop_steps = 0

        obstacle_contact = (
            throttle >= OBSTACLE_CONTACT_MIN_THROTTLE
            and float(raw_progress) <= OBSTACLE_CONTACT_MAX_PROGRESS_M
            and (
                float(self.ship_speed) <= OBSTACLE_CONTACT_LOW_SPEED_MPS
                or (float(self.previous_speed) - float(self.ship_speed)) >= OBSTACLE_CONTACT_SPEED_DROP_MPS
            )
            and float(cross_track_error) >= OBSTACLE_CONTACT_MIN_CROSS_TRACK_M
        )
        if obstacle_contact:
            self._obstacle_contact_steps += 1
        else:
            self._obstacle_contact_steps = 0

        if checkpoint_distance + CHECKPOINT_PROGRESS_IMPROVEMENT_EPSILON_M < self._best_checkpoint_distance:
            self._best_checkpoint_distance = float(checkpoint_distance)
            self._checkpoint_stagnation_steps = 0
        else:
            self._checkpoint_stagnation_steps += 1

    def _compute_turn_guidance(
        self,
        ship_pos: np.ndarray,
        ship_heading: float,
        cross_track_error: Optional[float] = None,
    ) -> Dict[str, float]:
        """Estimate local turn severity and a safer speed/steering target."""
        cp_idx = min(self.checkpoint_index, len(self.checkpoints) - 1)
        prev_idx = max(0, cp_idx - 1)
        prev_pos = self.checkpoints[prev_idx]["pos"]
        current_cp = self.checkpoints[cp_idx]
        cur_pos = current_cp["pos"]

        path_vec = cur_pos - prev_pos
        path_length = max(np.linalg.norm(path_vec), EPSILON)
        path_unit = path_vec / path_length
        cp_dist = float(np.linalg.norm(ship_pos - cur_pos))
        segment_progress = float(
            np.clip(np.dot(ship_pos - prev_pos, path_unit) / path_length, 0.0, 1.0)
        )
        heading_error = self._calc_heading_error(
            _safe_heading_from_vector(path_vec),
            ship_heading,
        )
        checkpoint_heading_error = self._calc_heading_error(
            _safe_heading_from_vector(cur_pos - ship_pos),
            ship_heading,
        )

        if cross_track_error is None:
            cross_track_error = float(
                self._distance_from_point_to_line(ship_pos, prev_pos, cur_pos)
            )

        lookahead_mix = 0.0
        turn_intensity = 0.0
        if cp_idx < len(self.checkpoints) - 1:
            next_pos = self.checkpoints[cp_idx + 1]["pos"]
            next_vec = next_pos - cur_pos
            next_length = np.linalg.norm(next_vec)
            if next_length > EPSILON:
                next_unit = next_vec / next_length
                turn_intensity = float(
                    np.clip(
                        abs(
                            self._calc_heading_error(
                                _safe_heading_from_vector(next_unit),
                                _safe_heading_from_vector(path_unit),
                            )
                        ) / (math.pi / 2.0),
                        0.0,
                        1.0,
                    )
                )
                lookahead_mix = float(
                    np.clip(
                        (segment_progress - LOOKAHEAD_BLEND_START_PROGRESS)
                        / max(
                            LOOKAHEAD_BLEND_FULL_PROGRESS - LOOKAHEAD_BLEND_START_PROGRESS,
                            EPSILON,
                        ),
                        0.0,
                        1.0,
                    )
                )

        recovery_mix = float(
            np.clip(
                (cross_track_error - OFFTRACK_RECOVERY_BLEND_START_M)
                / max(OFFTRACK_RECOVERY_BLEND_FULL_M - OFFTRACK_RECOVERY_BLEND_START_M, EPSILON),
                0.0,
                1.0,
            )
        )
        checkpoint_bearing_mix = float(
            np.clip(max(recovery_mix, 1.0 - (cp_dist / path_length)), 0.0, 1.0)
        )
        turn_demand = float(
            np.clip(
                max(
                    recovery_mix,
                    checkpoint_bearing_mix * abs(float(checkpoint_heading_error)) / math.radians(30.0),
                    lookahead_mix * (0.35 + turn_intensity),
                    turn_intensity,
                    abs(float(heading_error)) / math.radians(25.0),
                ),
                0.0,
                1.5,
            )
        )
        target_speed_mps = max(
            MIN_TURN_TARGET_SPEED_MPS,
            PY_SIM_TARGET_SPEED_MPS * (
                1.0 - MAX_TURN_TARGET_SPEED_REDUCTION * min(turn_demand, 1.0)
            ),
        )
        required_turn_steering = float(
            np.clip(
                BASE_REQUIRED_TURN_STEERING
                + 0.12 * checkpoint_bearing_mix
                + 0.14 * recovery_mix
                + 0.20 * turn_intensity
                + 0.06 * min(turn_demand, 1.0),
                BASE_REQUIRED_TURN_STEERING,
                MAX_REQUIRED_TURN_STEERING,
            )
        )
        required_turn_action = float(
            np.clip(
                required_turn_steering / max(float(self.ship_specs.max_rudder_angle), EPSILON),
                0.0,
                1.0,
            )
        )

        return {
            "turn_demand": turn_demand,
            "turn_intensity": turn_intensity,
            "recovery_mix": recovery_mix,
            "checkpoint_bearing_mix": checkpoint_bearing_mix,
            "checkpoint_heading_error": float(checkpoint_heading_error),
            "target_speed_mps": float(target_speed_mps),
            "required_turn_action": required_turn_action,
            "cross_track_error": float(cross_track_error),
        }

    def _apply_turn_action_governor(self, action: np.ndarray) -> np.ndarray:
        """Reduce throttle and amplify steering when the live route geometry demands a real turn."""
        if len(self.checkpoints) < 2:
            return np.asarray(action, dtype=np.float32)

        cp_idx = min(self.checkpoint_index, len(self.checkpoints) - 1)
        prev_pos = self.checkpoints[max(0, cp_idx - 1)]["pos"]
        cur_pos = self.checkpoints[cp_idx]["pos"]
        cross_track_error = float(
            self._distance_from_point_to_line(self.ship_pos, prev_pos, cur_pos)
        )
        guidance = self._compute_turn_guidance(
            self.ship_pos,
            self.ship_heading,
            cross_track_error=cross_track_error,
        )

        governed = np.asarray(action, dtype=np.float32).copy()
        turn_demand = float(guidance["turn_demand"])
        if turn_demand < TURN_ACTION_GOVERNOR_START:
            return governed

        turn_fraction = float(
            np.clip(
                (turn_demand - TURN_ACTION_GOVERNOR_START)
                / max(1.0 - TURN_ACTION_GOVERNOR_START, EPSILON),
                0.0,
                1.0,
            )
        )

        steering_gain = 1.0 + (TURN_ACTION_MAX_STEERING_GAIN - 1.0) * turn_fraction
        governed[0] = float(np.clip(governed[0] * steering_gain, -1.0, 1.0))

        steering_floor = guidance["required_turn_action"] * turn_fraction
        preferred_turn_sign = 0.0
        if abs(float(governed[0])) >= 0.02:
            preferred_turn_sign = float(np.sign(governed[0]))
        elif abs(float(guidance["checkpoint_heading_error"])) >= TURN_ACTION_HEADING_INJECTION_DEAD_ZONE_RAD:
            preferred_turn_sign = float(np.sign(guidance["checkpoint_heading_error"]))
        elif abs(float(self.previous_action[0])) >= 0.02:
            preferred_turn_sign = float(np.sign(self.previous_action[0]))

        if preferred_turn_sign != 0.0 and steering_floor > 0.0:
            governed[0] = float(
                np.clip(
                    preferred_turn_sign * max(abs(float(governed[0])), steering_floor),
                    -1.0,
                    1.0,
                )
            )

        if governed[1] > 0.0:
            speed_scale = float(
                np.clip(
                    guidance["target_speed_mps"] / max(float(self.MAX_SPEED), EPSILON),
                    0.25,
                    1.0,
                )
            )
            offtrack_scale = 1.0 - TURN_ACTION_OFFTRACK_THROTTLE_SCALE * float(
                np.clip(
                    (cross_track_error - OFFTRACK_RECOVERY_BLEND_START_M)
                    / max(OFFTRACK_RECOVERY_BLEND_FULL_M - OFFTRACK_RECOVERY_BLEND_START_M, EPSILON),
                    0.0,
                    1.0,
                )
            )
            governed[1] = float(
                np.clip(governed[1] * min(speed_scale, offtrack_scale), -1.0, 1.0)
            )

        return governed.astype(np.float32, copy=False)

    def _apply_guidance_residual_policy(self, policy_action: np.ndarray) -> np.ndarray:
        """Blend a deterministic path-following baseline with the policy residual.

        This makes training substantially more stable on the long 86-checkpoint
        route: the baseline keeps the vessel near the path while PPO learns
        higher-quality corrections.
        """
        policy_action = np.asarray(policy_action, dtype=np.float32)
        self.last_policy_action = policy_action.copy()

        if not self._guidance_assist_enabled or len(self.checkpoints) < 2:
            self.last_guidance_action = np.zeros(2, dtype=np.float32)
            return policy_action

        cp_idx = min(self.checkpoint_index, len(self.checkpoints) - 1)
        prev_pos = self.checkpoints[max(0, cp_idx - 1)]["pos"]
        cur_pos = self.checkpoints[cp_idx]["pos"]
        cross_track_error = float(
            self._distance_from_point_to_line(self.ship_pos, prev_pos, cur_pos)
        )
        guidance = self._compute_turn_guidance(
            self.ship_pos,
            self.ship_heading,
            cross_track_error=cross_track_error,
        )

        los_heading = self._los_guidance_heading()
        los_heading_error = self._calc_heading_error(los_heading, self.ship_heading)
        yaw_rate_limit = max(float(self.ship_specs.yaw_rate_limits[1]), EPSILON)
        yaw_rate_ratio = float(np.clip(self.ship_r / yaw_rate_limit, -2.0, 2.0))
        max_rudder_angle = max(float(self.ship_specs.max_rudder_angle), EPSILON)

        base_steering = float(
            np.clip(
                (GUIDANCE_STEERING_KP * float(los_heading_error) / max_rudder_angle)
                - (GUIDANCE_STEERING_KD * yaw_rate_ratio),
                -1.0,
                1.0,
            )
        )

        target_speed = float(
            np.clip(
                guidance["target_speed_mps"],
                MIN_TURN_TARGET_SPEED_MPS,
                PY_SIM_TARGET_SPEED_MPS,
            )
        )
        offtrack_mix = float(
            np.clip(
                (cross_track_error - GUIDANCE_OFFTRACK_SLOWDOWN_START_M)
                / max(
                    GUIDANCE_OFFTRACK_SLOWDOWN_FULL_M - GUIDANCE_OFFTRACK_SLOWDOWN_START_M,
                    EPSILON,
                ),
                0.0,
                1.0,
            )
        )
        target_speed *= 1.0 - 0.6 * offtrack_mix
        speed_error = target_speed - float(self.ship_speed)
        base_throttle = float(
            np.clip(
                (GUIDANCE_THROTTLE_KP * speed_error) / max(float(self.MAX_SPEED), EPSILON),
                GUIDANCE_MAX_REVERSE_THROTTLE,
                1.0,
            )
        )
        base_throttle *= 1.0 - 0.35 * abs(base_steering)
        if float(self.ship_speed) < 0.35 and abs(float(los_heading_error)) < math.radians(35.0):
            base_throttle = max(base_throttle, GUIDANCE_MIN_FORWARD_THROTTLE)

        guidance_action = np.array([base_steering, base_throttle], dtype=np.float32)
        self.last_guidance_action = guidance_action.copy()

        blended = np.array(
            [
                guidance_action[0] + (policy_action[0] * self._guidance_residual_steering_scale),
                guidance_action[1] + (policy_action[1] * self._guidance_residual_throttle_scale),
            ],
            dtype=np.float32,
        )
        blended = np.clip(blended, -1.0, 1.0)

        if guidance["turn_demand"] >= 0.65:
            preferred_turn_sign = float(np.sign(guidance["checkpoint_heading_error"]))
            if preferred_turn_sign != 0.0 and abs(float(blended[0])) < 0.12:
                blended[0] = float(
                    np.clip(
                        preferred_turn_sign * max(0.12, abs(float(guidance_action[0]))),
                        -1.0,
                        1.0,
                    )
                )

        return blended.astype(np.float32, copy=False)

    def _calculate_reward(self) -> Tuple[float, bool, str]:
        """Compute dense full-route reward for stable no-curriculum training."""
        if self.checkpoint_index >= len(self.checkpoints):
            return 200.0, True, "completed"

        prev_idx = max(0, self.checkpoint_index - 1)
        prev_pos = self.checkpoints[prev_idx]["pos"]
        current_cp = self.checkpoints[self.checkpoint_index]
        cur_pos = current_cp["pos"]

        path_vec = cur_pos - prev_pos
        path_length = max(np.linalg.norm(path_vec), EPSILON)
        path_unit = path_vec / path_length

        rel_prev = self.previous_ship_pos - prev_pos
        rel_now = self.ship_pos - prev_pos
        proj_prev = float(np.dot(rel_prev, path_unit))
        proj_now = float(np.dot(rel_now, path_unit))
        raw_progress = proj_now - proj_prev
        self._last_raw_progress = float(raw_progress)
        route_progress = float(self.checkpoint_index / max(len(self.checkpoints) - 1, 1))

        cp_dist = float(np.linalg.norm(self.ship_pos - cur_pos))
        prev_cp_dist = float(np.linalg.norm(self.previous_ship_pos - cur_pos))

        cross_track_error = float(
            self._distance_from_point_to_line(self.ship_pos, prev_pos, cur_pos)
        )
        prev_cross_track_error = float(
            self._distance_from_point_to_line(self.previous_ship_pos, prev_pos, cur_pos)
        )
        self.cross_error = cross_track_error

        los_heading = self._los_guidance_heading()
        los_heading_prev = _safe_heading_from_vector(cur_pos - self.previous_ship_pos)
        los_heading_error = self._calc_heading_error(los_heading, self.ship_heading)
        prev_los_heading_error = self._calc_heading_error(
            los_heading_prev,
            self.previous_heading,
        )

        guidance = self._compute_turn_guidance(
            self.ship_pos,
            self.ship_heading,
            cross_track_error=cross_track_error,
        )
        target_speed_mps = max(float(guidance["target_speed_mps"]), MIN_TURN_TARGET_SPEED_MPS)

        self._update_stuck_trackers(raw_progress, cross_track_error, cp_dist)

        progress_reward = float(np.clip(raw_progress / 2.2, -2.5, 2.5))
        checkpoint_approach_reward = float(
            np.clip((prev_cp_dist - cp_dist) / 4.0, -2.0, 2.0)
        )
        cross_track_convergence_reward = float(
            np.clip((prev_cross_track_error - cross_track_error) / 3.5, -2.0, 2.0)
        )
        heading_alignment_reward = float(math.cos(float(los_heading_error)))
        heading_convergence_reward = float(
            np.clip(
                (abs(float(prev_los_heading_error)) - abs(float(los_heading_error)))
                / max(math.radians(8.0), EPSILON),
                -2.0,
                2.0,
            )
        )
        centerline_bonus = float(
            math.exp(-((cross_track_error / max(CENTERLINE_BONUS_WIDTH_M, EPSILON)) ** 2))
        )
        cross_track_scale = 1.0 + 0.7 * route_progress
        cross_track_penalty = float(
            -cross_track_scale * np.clip(cross_track_error / 180.0, 0.0, 3.0)
        )

        speed_excess = max(float(self.ship_speed) - target_speed_mps, 0.0) / max(
            target_speed_mps,
            EPSILON,
        )
        speed_shortfall = max(target_speed_mps - float(self.ship_speed), 0.0) / max(
            target_speed_mps,
            EPSILON,
        )
        overspeed_penalty = -(
            (0.35 + 0.65 * float(guidance["turn_demand"])) * speed_excess
        )
        heading_slowdown_penalty = -(
            0.25
            * speed_shortfall
            * float(np.clip(abs(float(los_heading_error)) / math.radians(45.0), 0.0, 1.0))
        )

        rudder_delta = abs(float(self.performed_action[0]) - float(self.previous_action[0]))
        max_rudder_delta = float(self.ship_specs.max_rudder_rate) * max(
            self._last_sim_step_dt,
            EPSILON,
        )
        rudder_smooth_penalty = float(
            -0.10 * np.clip(rudder_delta / max(max_rudder_delta, EPSILON), 0.0, 1.5)
        )
        rudder_magnitude_penalty = float(
            -0.08 * min(abs(float(self.performed_action[0])) ** 2, 1.0)
        )

        throttle_mag = abs(float(self.last_controls.get("throttle", 0.0)))
        heading_throttle_penalty = float(
            -0.25
            * throttle_mag
            * np.clip(abs(float(los_heading_error)) / math.radians(35.0), 0.0, 1.5)
        )
        offtrack_throttle_penalty = float(
            -0.55
            * throttle_mag
            * np.clip((cross_track_error - 90.0) / 120.0, 0.0, 2.0)
        )
        turn_demand = float(guidance["turn_demand"])
        required_turn_action = max(float(guidance["required_turn_action"]), 0.10)
        steering_mag = abs(float(self.performed_action[0]))
        understeer_penalty = float(
            -1.0
            * max(0.0, turn_demand - 0.2)
            * np.clip((required_turn_action - steering_mag) / required_turn_action, 0.0, 1.0)
        )
        persistence_penalty = float(
            -0.35
            * min(
                self._high_cross_track_steps / max(HIGH_CROSS_TRACK_TIME_PENALTY_PATIENCE, 1),
                2.0,
            )
        )
        wrong_way_penalty = -0.75 if raw_progress < -0.05 else 0.0

        reward = (
            0.35
            + 0.90 * progress_reward
            + 0.50 * checkpoint_approach_reward
            + 0.35 * cross_track_convergence_reward
            + 0.30 * heading_alignment_reward
            + 0.25 * heading_convergence_reward
            + 0.20 * centerline_bonus
            + cross_track_penalty
            + overspeed_penalty
            + heading_slowdown_penalty
            + rudder_smooth_penalty
            + rudder_magnitude_penalty
            + heading_throttle_penalty
            + offtrack_throttle_penalty
            + understeer_penalty
            + persistence_penalty
            + wrong_way_penalty
        )
        if cp_dist <= current_cp["radius"]:
            reward += self._checkpoint_entry_reward(self.checkpoint_index)
            if self.checkpoint_index == len(self.checkpoints) - 1:
                logger.info("Target REACHED at distance %.2f", cp_dist)
            self.checkpoint_index += 1
            self.step_count = 0
            self._low_speed_steps = 0
            self._stalled_progress_steps = 0
            self._high_cross_track_steps = 0
            self._hard_stop_steps = 0
            self._obstacle_contact_steps = 0
            self._best_checkpoint_distance = float("inf")
            self._checkpoint_stagnation_steps = 0
        elif self._is_near_perpendicular_line(current_cp):
            reward += self._checkpoint_entry_reward(self.checkpoint_index)
            self.checkpoint_index += 1
            self.step_count = 0
            self._low_speed_steps = 0
            self._stalled_progress_steps = 0
            self._high_cross_track_steps = 0
            self._hard_stop_steps = 0
            self._obstacle_contact_steps = 0
            self._best_checkpoint_distance = float("inf")
            self._checkpoint_stagnation_steps = 0

        done = False
        termination_reason = ""

        collision = self._check_collision()
        self._collision_latched = self._collision_latched or bool(collision)
        out_of_water = not self._ship_in_open_water(self.ship_pos)
        cross_track_limit = min(380.0, CROSS_TRACK_TERMINATION_MULTIPLIER * CHECKPOINTS_DISTANCE)
        checkpoint_stagnation = (
            self._checkpoint_stagnation_steps >= CHECKPOINT_STAGNATION_PATIENCE_STEPS
            and cp_dist > CHECKPOINT_STAGNATION_MIN_DISTANCE_M
        )
        collision_like_stop = (
            self._hard_stop_steps >= COLLISION_LIKE_STOP_PATIENCE_STEPS
            and cp_dist > max(float(current_cp["radius"]) * 3.0, 20.0)
        )
        obstacle_contact_stall = (
            self._obstacle_contact_steps >= OBSTACLE_CONTACT_PATIENCE_STEPS
            and cp_dist > max(float(current_cp["radius"]) * 2.0, 20.0)
        )
        low_speed_stall = (
            self._low_speed_steps >= LOW_SPEED_PATIENCE_STEPS
            and cp_dist > max(float(current_cp["radius"]) * 2.0, 20.0)
        )
        no_progress_stall = (
            self._stalled_progress_steps >= NO_PROGRESS_PATIENCE_STEPS
            and cp_dist > max(float(current_cp["radius"]) * 1.5, 20.0)
        )

        termination_conditions = {
            "collision": bool(self._collision_latched),
            "collision-like-stop": collision_like_stop,
            "obstacle-contact-stall": obstacle_contact_stall,
            "cross-track": cross_track_error > cross_track_limit,
            "out-of-water": out_of_water,
            "low-speed-stall": low_speed_stall,
            "no-progress-stall": no_progress_stall,
            "checkpoint-stagnation": checkpoint_stagnation,
        }

        if termination_conditions["collision"]:
            reward = COLLISION_TERMINATION_REWARD
            done = True
            termination_reason = "collision"
            self._halt_vehicle()
        elif termination_conditions["collision-like-stop"]:
            reward = COLLISION_TERMINATION_REWARD
            done = True
            termination_reason = "collision-like-stop"
            self._halt_vehicle()
        elif termination_conditions["obstacle-contact-stall"]:
            reward = COLLISION_TERMINATION_REWARD
            done = True
            termination_reason = "obstacle-contact-stall"
            self._halt_vehicle()
        elif any(termination_conditions.values()):
            reward = -250
            done = True
            for name, active in termination_conditions.items():
                if active:
                    termination_reason = name
                    break
            self._halt_vehicle()
            if self.verbose:
                triggered = [name for name, active in termination_conditions.items() if active]
                logger.debug("Terminated: %s", ", ".join(triggered))

        if self.checkpoint_index >= len(self.checkpoints):
            done = True
            termination_reason = "completed"
            reward += 180.0

        return reward, done, termination_reason

    # -------------
    # Utility methods (match PySimEnv)
    # -------------
    def _los_guidance_heading(self) -> float:
        """Compute desired LOS (Line-of-Sight) heading toward the current path segment.

        This implements the LOS guidance algorithm from Changyuan Chen's PhD thesis
        "Numerical and Experimental Study on Ship Motion Control Systems in Shallow Water"
        at Flanders Hydraulics (FH). The algorithm computes an optimal heading that steers
        the vessel toward a look-ahead point on the trajectory.

        Returns:
            The optimal heading angle in radians that the ship should aim toward.
        """
        import math

        current_checkpoint_idx = self.checkpoint_index
        current_checkpoint_idx = min(current_checkpoint_idx, len(self.checkpoints) - 1)
        pos = self.ship_pos
        p_k = self.checkpoints[current_checkpoint_idx]["pos"].copy()
        if current_checkpoint_idx + 1 > len(self.checkpoints) - 1:
            p_k_plus_1 = self.checkpoints[len(self.checkpoints) - 1]["pos"].copy()
            direction = p_k_plus_1 - self.checkpoints[len(self.checkpoints) - 2]["pos"]
            p_k_plus_1 += direction
        else:
            p_k_plus_1 = self.checkpoints[current_checkpoint_idx + 1]["pos"].copy()
        x, y = pos[:]
        x_k, y_k = p_k[:]
        x_k_plus_1, y_k_plus_1 = p_k_plus_1[:]

        # previous checkpoint
        if current_checkpoint_idx > 0:
            p_k_min_1 = self.checkpoints[current_checkpoint_idx - 1]["pos"].copy()
            x_k_min_1, y_k_min_1 = p_k_min_1[:]
        else:
            # if none exists, create a virtual one behind the ship
            p_k_min_1 = p_k - (p_k_plus_1 - p_k)
            x_k_min_1, y_k_min_1 = p_k_min_1[:]

        # Find the orthogonal projection A(x_A, y_A) of the ship on the path segment
        denom_sq = (x_k - x_k_min_1) ** 2 + (y_k - y_k_min_1) ** 2
        if denom_sq < EPSILON:
            # Degenerate segment, just head toward current checkpoint
            return _safe_heading_from_vector(p_k - pos)

        t = ((x - x_k_min_1) * (x_k - x_k_min_1) + (y - y_k_min_1) * (y_k - y_k_min_1)) / denom_sq
        x_a = x_k_min_1 + t * (x_k - x_k_min_1)
        y_a = y_k_min_1 + t * (y_k - y_k_min_1)

        # Calculate the look ahead distance L_A
        c = 0.5  # Coefficient for LOS look ahead horizon
        delta_y = np.sqrt((x - x_a) ** 2 + (y - y_a) ** 2)  # Cross-track error
        l_pp = 0.95 * self.ship_specs.length  # Length between perpendiculars
        l_a = np.sqrt((delta_y + c * l_pp) ** 2 - delta_y ** 2) if (delta_y + c * l_pp) >= 0 else 0.0

        # Calculate the LOS position Ploss(x_loss, y_loss)
        path_length = np.sqrt(denom_sq)
        if path_length > EPSILON:
            x_loss = x_a + ((x_k - x_k_min_1) / path_length) * l_a
            y_loss = y_a + ((y_k - y_k_min_1) / path_length) * l_a
        else:
            x_loss, y_loss = x_a, y_a

        # Calculate radius of acceptance circle Rk
        w_los = 0.5
        r_k = w_los * l_pp

        heading_p_k_min_1_p_k = _safe_heading_from_vector(p_k - p_k_min_1)
        heading_p_k_p_k_plus_1 = _safe_heading_from_vector(p_k_plus_1 - p_k)
        heading_difference = _normalize_angle(heading_p_k_min_1_p_k - heading_p_k_p_k_plus_1)
        collinear = abs(heading_difference) < (1 / 57)  # ~1 degree

        if collinear:
            dist_a_pk = np.sqrt((x_a - x_k) ** 2 + (y_a - y_k) ** 2)
            if dist_a_pk > r_k:
                los_angle = math.atan2(y_loss - y, x_loss - x)
            else:
                los_angle = math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)
        else:
            dist_p_pk = np.sqrt((x - x_k) ** 2 + (y - y_k) ** 2)
            if dist_p_pk > 2 * r_k:
                if l_a < dist_p_pk:
                    los_angle = math.atan2(y_loss - y, x_loss - x)
                else:
                    # Change look ahead distance from LA to L'A
                    l_accent_a = l_a - np.sqrt((x_k - x_a) ** 2 + (y_k - y_a) ** 2)
                    path_next_length = np.sqrt((x_k_plus_1 - x_k) ** 2 + (y_k_plus_1 - y_k) ** 2)
                    if path_next_length > EPSILON:
                        x_accent_loss = x_k + ((x_k_plus_1 - x_k) / path_next_length) * l_accent_a
                        y_accent_loss = y_k + ((y_k_plus_1 - y_k) / path_next_length) * l_accent_a
                    else:
                        x_accent_loss, y_accent_loss = x_k, y_k
                    los_angle = math.atan2(y_accent_loss - y, x_accent_loss - x)
            else:
                # Ship is close to current checkpoint; use next checkpoint
                los_angle = math.atan2(y_k_plus_1 - y, x_k_plus_1 - x)

        return los_angle

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
