"""Gymnasium environment for RL training inside UE5 via CosysAirSim.

This module provides a :class:`AirSimEnv`, a drop-in replacement for
``PySimEnv`` that uses Unreal Engine 5's physics via the CosysAirSim API
instead of the Fossen mathematical model.

Coordinate frames
-----------------
- **Fossen**: x = east, y = north, heading from east (CCW positive).
- **AirSim**: x = north, y = east (NED), yaw from north (CW positive).
- **Transform**: ``AirSim_yaw = π/2 − fossen_heading``.

Usage
-----
>>> from airsim.environments.airsim_env import AirSimEnv
>>> env = AirSimEnv(ship_pos=[2245, -47], target_pos=[11530, 13000])
>>> obs, info = env.reset()
>>> obs, reward, done, truncated, info = env.step([0.0, 0.5])
>>> env.close()
"""

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
from airsim.communication.coord_transform import CoordTransform, _normalize_angle
from fw.ships.ship_factory import create_ship, ShipModel
from fw.ships.actuator_model import ActuatorModel

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────
# Constants (mirror PySimEnv for consistency)
# ───────────────────────────────────────────
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


# ───────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────
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


# ───────────────────────────────────────────
# Main Environment
# ───────────────────────────────────────────
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
        ip: str = "",
        port: int = 41451,
        render_mode: Optional[str] = None,
        max_steps: int = 14000,
        time_step: float = 0.1,
        frames_per_step: int = 10,
        use_pause_stepping: bool = True,
        training_mode: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialise the AirSim training environment.

        Args:
            ship_pos: Initial ship position in Fossen coords [x_east, y_north].
            target_pos: Target position in Fossen coords.
            calibration: Optional calibration dict with scale/rotation/origins.
            calibration_path: Path to calibration JSON. If None, tries default.
            ip: AirSim server IP. Empty string = localhost.
            port: AirSim RPC port.
            render_mode: 'human' or None.
            max_steps: Maximum steps per episode before truncation.
            time_step: Nominal simulation time step (seconds).
            frames_per_step: Physics frames to advance per RL step.
            use_pause_stepping: If True, pause sim and step frame-by-frame
                (fastest training mode).
            verbose: Print debug info.
        """
        super().__init__(render_mode)

        self.max_steps = max_steps
        self.time_step = time_step
        self.frames_per_step = frames_per_step
        self.use_pause_stepping = use_pause_stepping
        self.verbose = verbose

        # ── Ship specifications & actuator model ──
        self.ship = create_ship(ShipModel.MYZAKO_DEFAULT)
        self.ship_specs = self.ship.specifications
        self._actuators = ActuatorModel(self.ship_specs)

        # UE5-specific limits (may differ from Fossen model)
        self.MAX_SPEED = 5.6  # m/s — top speed of the UE5 ship

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
            logger.info("No calibration found — using identity axis-swap transform.")

        # ── Connect to AirSim via Bridge ──
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

        cfg = AirSimConfig(ip=ip, port=port)
        self.bridge = AirSimBridge(cfg)

        if not self.bridge.is_connected:
            raise ConnectionError(
                f"AirSimBridge could not connect to UE5 at {ip or '127.0.0.1'}:{port}. "
                "Make sure Unreal Engine 5 is open and in Play mode."
            )
        
        # Expose client for environment-specific features (plotting, etc)
        self.client = self.bridge.client

        # ── Auto drift correction ──
        self._spawn_pose = self.client.simGetVehiclePose()
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
                    print(f"[AirSimEnv] Origin drift detected: "
                          f"Δx={drift[0]:.2f}, Δy={drift[1]:.2f} ({mag:.1f} m) → auto-correcting.")
                    self.coord._airsim_origin = self.coord._airsim_origin + drift
                else:
                    print(f"[AirSimEnv] No origin drift ({mag:.2f} m).")

        if self.use_pause_stepping:
            self.client.simPause(True)
            print("[AirSimEnv] Simulation paused — using deterministic frame stepping.")

        # ── Training mode: disable UE5 rendering for speed ──
        self._training_mode = training_mode
        if training_mode:
            self.client.simRunConsoleCommand("r.SetRes 64x64w")
            self.client.simRunConsoleCommand("t.MaxFPS 5")
            self.client.simRunConsoleCommand("ShowFlag.Rendering 0")
            self.client.simRunConsoleCommand("ShowFlag.PostProcessing 0")
            self.client.simRunConsoleCommand("ShowFlag.Lighting 0")
            self.client.simRunConsoleCommand("ShowFlag.Translucency 0")
            self.client.simRunConsoleCommand("ShowFlag.Fog 0")
            self.client.simRunConsoleCommand("ShowFlag.Particles 0")
            self.client.simRunConsoleCommand("ShowFlag.SkeletalMeshes 0")
            self.client.simRunConsoleCommand("ShowFlag.Bloom 0")
            self.client.simRunConsoleCommand("ShowFlag.EyeAdaptation 0")
            self.client.simRunConsoleCommand("sg.PostProcessQuality 0")
            self.client.simRunConsoleCommand("sg.ShadowQuality 0")
            self.client.simRunConsoleCommand("sg.TextureQuality 0")
            self.client.simRunConsoleCommand("sg.EffectsQuality 0")
            self.client.simRunConsoleCommand("sg.FoliageQuality 0")
            print("[AirSimEnv] Training mode: UE5 rendering minimized for speed.")

        # ── Draw trajectory in UE5 ──
        self._draw_visualization()

        # ── Load checkpoints (same data as PySimEnv) ──
        self._load_checkpoints()

        # ── State tracking ──
        self.ship_pos = self.initial_ship_pos.copy()
        self.previous_ship_pos = self.initial_ship_pos.copy()
        self.ship_heading = 0.0
        self.previous_heading = 0.0
        self.ship_speed = 0.0
        self.ship_u = 0.0  # surge
        self.ship_v = 0.0  # sway
        self.ship_r = 0.0  # yaw rate
        self.performed_action = np.zeros(2, dtype=np.float32)
        self.checkpoint_index = 1
        self.cross_error = 0.0
        self.step_count = 0
        self.max_dist = np.sqrt(2) * float(MAX_GRID_POS)

        # ── Gym spaces (identical to PySimEnv) ──
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

    # AirSimEnv holds a live network connection and a module reference
    # that cannot be pickled.
    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    # ─────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────
    def _load_checkpoints(self) -> None:
        """Load trajectory checkpoints from the same CSV as PySimEnv."""
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
            checkpoints.append({
                "pos": np.array(point, dtype=np.float32),
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

        self.client.simFlushPersistentMarkers()
        time.sleep(0.2)

        # ── Polygon outline (green) ──
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
                self.client.simPlotLineStrip(
                    chunk,
                    color_rgba=[0, 1, 0, 0.8],
                    thickness=5.0,
                    is_persistent=True,
                )

        # ── Trajectory (yellow) ──
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
                self.client.simPlotLineStrip(
                    chunk,
                    color_rgba=[1, 1, 0, 1.0],
                    thickness=8.0,
                    is_persistent=True,
                )

        # ── Start (blue) and end (red) dots ──
        start_ux, start_uy = self.coord.fossen_to_airsim_2d(
            float(self.initial_ship_pos[0]), float(self.initial_ship_pos[1])
        )
        end_ux, end_uy = self.coord.fossen_to_airsim_2d(
            float(traj_data[-1][0]), float(traj_data[-1][1])
        )

        self.client.simPlotPoints(
            [airsim.Vector3r(start_ux, start_uy, -3.0)],
            color_rgba=[0, 0, 1, 1],
            size=30.0,
            is_persistent=True,
        )
        self.client.simPlotPoints(
            [airsim.Vector3r(end_ux, end_uy, -3.0)],
            color_rgba=[1, 0, 0, 1],
            size=30.0,
            is_persistent=True,
        )

        print("[AirSimEnv] Trajectory and polygon drawn in UE5.")

    # ─────────────────────────────────────
    # Observation space
    # ─────────────────────────────────────
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """10-dim observation space identical to PySimEnv."""
        low = np.array([
            -1.0,  # surge velocity
            -1.0,  # sway velocity
            -1.0,  # yaw rate
             0.0,  # distance to current checkpoint
             0.0,  # distance to next checkpoint
            -2.0,  # cross-track error
            -1.0,  # heading error (to current)
            -1.0,  # heading error (to next)
            -1.0,  # heading error parallel (current)
            -1.0,  # heading error parallel (next)
        ], dtype=np.float32)

        high = np.array([
            1.0, 1.0, 1.0,
            1.0, 1.0,
            2.0,
            1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)

        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # ─────────────────────────────────────
    # AirSim state reading
    # ─────────────────────────────────────
    def _read_airsim_state(self) -> None:
        """Read vehicle state from UE5 and convert to Fossen frame."""
        pose = self.client.simGetVehiclePose()
        car_state = self.client.getCarState()

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

    def _check_collision(self) -> bool:
        """Check for collision via AirSim API."""
        info = self.client.simGetCollisionInfo()
        return info.has_collided

    # ─────────────────────────────────────
    # Observation
    # ─────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """Build 10-dim observation vector (same format as PySimEnv)."""
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
        norm_cross_error *= 1.0 if np.cross(track_vec, ship_vec) >= 0.0 else -1.0
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

        obs = np.hstack([
            norm_velocities,
            [norm_distance],
            [norm_next_distance],
            [norm_cross_error],
            [heading_error / np.pi],
            [heading_error2 / np.pi],
            [heading_error_parallel / np.pi],
            [heading_error_parallel2 / np.pi],
        ])

        return obs.astype(np.float32)

    # ─────────────────────────────────────
    # Step & Reset
    # ─────────────────────────────────────
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one RL step in UE5.

        Args:
            action: [steering, throttle] in [-1, 1].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must be shape (2,), got {action.shape}")

        self.previous_ship_pos = self.ship_pos.copy()
        self.previous_heading = self.ship_heading

        actuator = self._actuators.apply_smoothing(action, self.enable_smoothing)
        self.bridge.send_action([actuator.rudder_angle, actuator.thrust])

        if self.use_pause_stepping:
            self.client.simContinueForFrames(self.frames_per_step)
            time.sleep(0.001)
        else:
            time.sleep(self.time_step)

        self._read_airsim_state()
        self.performed_action = self._actuators.current_actuator_state.copy()
        self.step_count += 1

        reward, terminated = self._calculate_reward()

        truncated = False
        if self.step_count >= self.max_steps and not terminated:
            reward = -250.0
            truncated = True

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {}

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the UE5 environment for a new episode."""
        if seed is not None:
            super().reset(seed=seed)

        self.bridge.reset()

        if self.use_pause_stepping:
            self.client.simPause(True)

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
        self.checkpoint_index = 1
        self.cross_error = 0.0
        self.step_count = 0

        initial_heading = 0.0
        if len(self.checkpoints) > self.checkpoint_index:
            direction = self.checkpoints[self.checkpoint_index]["pos"] - self.ship_pos
            initial_heading = _normalize_angle(_safe_heading_from_vector(direction))
        self.ship_heading = initial_heading

        current_pose = self.client.simGetVehiclePose()
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
        self.client.simSetVehiclePose(pose, ignore_collision=True)
        time.sleep(0.2)

        self._read_airsim_state()

        if self.verbose:
            print(f"[AirSimEnv] Reset — ship at ({self.ship_pos[0]:.0f}, {self.ship_pos[1]:.0f}), "
                  f"heading {math.degrees(self.ship_heading):.1f}°")

        return self._get_obs(), {}

    def close(self) -> None:
        """Disconnect from AirSim, unfreeze, and restore vehicle to spawn."""
        try:
            self.client.simFlushPersistentMarkers()
            self.client.simSetVehiclePose(self._spawn_pose, ignore_collision=True)

            if getattr(self, '_training_mode', False):
                self.client.simRunConsoleCommand("ShowFlag.Rendering 1")
                self.client.simRunConsoleCommand("ShowFlag.PostProcessing 1")
                self.client.simRunConsoleCommand("ShowFlag.Lighting 1")
                self.client.simRunConsoleCommand("ShowFlag.Translucency 1")
                self.client.simRunConsoleCommand("ShowFlag.Fog 1")
                self.client.simRunConsoleCommand("ShowFlag.Particles 1")
                self.client.simRunConsoleCommand("ShowFlag.SkeletalMeshes 1")
                self.client.simRunConsoleCommand("ShowFlag.Bloom 1")
                self.client.simRunConsoleCommand("ShowFlag.EyeAdaptation 1")
                self.client.simRunConsoleCommand("sg.PostProcessQuality 3")
                self.client.simRunConsoleCommand("sg.ShadowQuality 3")
                self.client.simRunConsoleCommand("sg.TextureQuality 3")
                self.client.simRunConsoleCommand("sg.EffectsQuality 3")
                self.client.simRunConsoleCommand("sg.FoliageQuality 3")
                self.client.simRunConsoleCommand("t.MaxFPS 0")
                print("[AirSimEnv] UE5 rendering quality restored.")

            if self.use_pause_stepping:
                self.client.simPause(False)

            self.bridge.disconnect()
            print("[AirSimEnv] Vehicle reset to spawn, simulation unpaused.")
        except Exception as e:
            logger.warning("Error during AirSimEnv.close(): %s", e)

    # ─────────────────────────────────────
    # Reward (mirrors PySimEnv exactly)
    # ─────────────────────────────────────
    def _calculate_reward(self) -> Tuple[float, bool]:
        """Compute reward and termination — same logic as PySimEnv."""
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
                print(f"[AirSimEnv] Target REACHED at distance {cp_dist:.2f}")
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
                print(f"[AirSimEnv] Terminated: {', '.join(triggered)}")

        if self.checkpoint_index >= len(self.checkpoints):
            done = True

        return reward, done

    # ─────────────────────────────────────
    # Utility methods (match PySimEnv)
    # ─────────────────────────────────────
    @staticmethod
    def _calc_heading_error(target: float, current: float, dead_zone_deg: float = 0.0) -> float:
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
