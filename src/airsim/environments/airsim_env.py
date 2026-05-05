"""Gymnasium AirSim environment for UE5 vessel training and inference.

This environment is the missing bridge between the existing AirSim connection
layer and the framework PPO stack. It exposes a standard Gymnasium interface,
defines an explicit continuous observation space for PPO, and keeps the route,
reward, and telemetry model close to the rest of the project so training,
inference, and reporting all use the same state contract.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium as gym
import numpy as np
try:
    from msgpackrpc.error import TransportError as _MsgpackTransportError
except Exception:  # pragma: no cover - optional dependency fallback
    class _MsgpackTransportError(Exception):
        """Fallback TransportError type when msgpackrpc is unavailable."""

        pass

from airsim.communication.airsim_bridge import AirSimBridge
from airsim.communication.control_mapping import (
    build_car_controls,
    map_direct_policy_action,
    validate_policy_action,
)
from airsim.communication.coord_transform import CoordTransform
from airsim.config.airsim_config import AirSimConfig, DEFAULT_AIRSIM_SAVE_SLOT
from fw.common.airsim.tools.calibrate_coords import load_saved_calibration_reference
from fw.common.airsim.tools.angles import yaw_from_quaternion
from fw.common.default_positions import (
    DEFAULT_FRAMEWORK_SHIP_POS,
    DEFAULT_FRAMEWORK_TARGET_POS,
)
from fw.environments.base_env import BaseEnv
from fw.environments.tools import create_checkpoints_from_simple_path

logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parents[2]
_FW_ENV_DIR = _SRC_DIR / "fw" / "environments"
_DEFAULT_TRAJECTORY_CSV = _FW_ENV_DIR / "trajectory_points_no_scale.csv"
_DEFAULT_OVERALL_CSV = _FW_ENV_DIR / "env_Sche_no_scale.csv"
_DEFAULT_OBSTACLES_CSV = _FW_ENV_DIR / "env_Sche_250cm_no_scale.csv"

_ROUTE_POINT_SPACING_M = 80.0
_CHECKPOINT_RADIUS_M = 60.0
_TARGET_RADIUS_M = 80.0
_CROSS_TRACK_LIMIT_M = 150.0
_SPEED_SCALE_MPS = 15.0
_YAW_RATE_SCALE_RADPS = math.radians(45.0)
_CROSS_TRACK_DELTA_SCALE_M = 20.0
_MIN_DISTANCE_SCALE_M = 500.0
_CHECKPOINT_REWARD = 1.0
_SUCCESS_REWARD = 5.0
_COLLISION_PENALTY = -5.0
_OFFTRACK_PENALTY = -4.0
_DISCONNECT_PENALTY = -5.0
_TIMEOUT_PENALTY = -2.5
_DEFAULT_REWARD_NORMALIZATION_SCALE = 5.0
_RESET_SETTLE_TIMEOUT_S = 1.5
_RESET_SETTLE_POLL_S = 0.05
_RESET_POSITION_TOLERANCE_M = 25.0
_WORLD_LOAD_SETTLE_TIMEOUT_S = 6.0
_WORLD_LOAD_SETTLE_POLL_S = 0.2
_WORLD_LOAD_SETTLE_POSITION_EPS_M = 0.25
_WORLD_LOAD_SETTLE_YAW_EPS_RAD = math.radians(2.0)
_EARLY_FAILURE_GRACE_STEPS = 4
_STATE_PROGRESS_POSITION_EPS_M = 0.05
_STATE_PROGRESS_YAW_EPS_RAD = math.radians(0.5)
_OBSERVATION_PROFILE_ROUTE_RELATIVE_CAR = "ppo_route_relative_car"
_CONTROL_INTERFACE_CAR_PHYSICS = "car_physics"
_RESET_MODE_REWIND_POSE = "rewind_pose"
_RESET_MODE_BRIDGE_RESET = "bridge_reset"
_DEFAULT_STEERING_SMOOTHING_ALPHA = 0.65
_DEFAULT_MAX_STEERING_RATE_PER_SECOND = 3.2
_CHECKPOINT_PASS_MAX_CROSS_TRACK_M = 120.0
_CHECKPOINT_PASS_MAX_HEADING_RAD = math.radians(75.0)
_SUCCESS_MAX_CROSS_TRACK_M = 60.0
_SUCCESS_MAX_HEADING_RAD = math.radians(45.0)
_STRICT_CTE_LIMIT_M = 10.0
_HARD_OFFTRACK_CTE_LIMIT_M = 150.0
_SOFT_OFFTRACK_CTE_LIMIT_M = 90.0
_OFFTRACK_LIMIT_DECAY_EPISODES = 450
_OFFTRACK_NEXT_CHECKPOINT_FACTOR = 1.6
_STRICT_TRACK_DECAY_M = 45.0
_PRECISION_BONUS_THRESHOLD_M = 8.0
_PRECISION_BONUS = 0.35
_MIN_ADAPTIVE_OFFTRACK_M = 90.0
_STRICT_PROGRESS_SCALE_M = 20.0
_STRICT_SUCCESS_BONUS = 5.0
_STRICT_FAILURE_REWARD = -1.0
_STEERING_MAGNITUDE_PENALTY_WEIGHT = 0.04
_STEERING_RATE_PENALTY_WEIGHT = 0.06
_LOS_LOOKAHEAD_FRACTION = 0.5
_LOG_SCALE_HALF_VALUE_FRACTION = 0.3


def _log_scale_signed(value: float, scale: float) -> float:
    """Map a signed distance to [-1, 1] with logarithmic compression.

    Values near zero get finer resolution; the input magnitude equal to
    ``scale`` maps to exactly ±1. The input magnitude equal to
    ``_LOG_SCALE_HALF_VALUE_FRACTION * scale`` maps to approximately ±0.5,
    giving the policy more precision when close to a checkpoint.
    """
    if scale <= 1e-6:
        return 0.0
    half_value = _LOG_SCALE_HALF_VALUE_FRACTION * scale
    k = math.log(3.0) / max(half_value, 1e-6)
    log_max = math.log1p(k * scale)
    scaled = math.log1p(k * abs(value)) / log_max
    return math.copysign(min(scaled, 1.0), value)


_OBSERVATION_FIELDS_ROUTE_RELATIVE_CAR = (
    "current_checkpoint_forward",
    "current_checkpoint_lateral",
    "next_checkpoint_forward",
    "next_checkpoint_lateral",
    "target_forward",
    "target_lateral",
    "heading_error_cos",
    "heading_error_sin",
    "cross_track_error",
    "cross_track_error_delta",
    "forward_speed",
    "yaw_rate",
    "executed_action_steering",
    "executed_action_throttle",
    "route_progress_ratio",
    "checkpoint_progress_ratio",
    "previous_action_steering",
    "los_heading_error",
)
_OBSERVATION_FIELDS_BY_PROFILE = {
    _OBSERVATION_PROFILE_ROUTE_RELATIVE_CAR: _OBSERVATION_FIELDS_ROUTE_RELATIVE_CAR,
}
_OBSERVATION_BUILDERS = {
    (_CONTROL_INTERFACE_CAR_PHYSICS, _OBSERVATION_PROFILE_ROUTE_RELATIVE_CAR): "_build_route_relative_car_observation",
}
_CONTROL_MAPPERS = {
    _CONTROL_INTERFACE_CAR_PHYSICS: map_direct_policy_action,
}
_AIRSIM_READ_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    ConnectionError,
    TimeoutError,
    _MsgpackTransportError,
)
_AIRSIM_DECODE_EXCEPTIONS = (AttributeError, TypeError, ValueError)


def normalize_navigation_state(
    state: dict[str, float | int],
    *,
    xy_center: tuple[float, float] = (2200.0, 2200.0),
    xy_half_span: float = 2500.0,
    route_progress_scale_m: float = 4000.0,
    speed_scale_mps: float = _SPEED_SCALE_MPS,
    checkpoint_count: int = 1,
) -> np.ndarray:
    """Normalize raw ship telemetry into a compact ``[-1, 1]`` vector.

    This helper is intended for pipelines that consume the run-log schema
    directly (large ``ship_x``/``ship_y`` values and heading in degrees).
    """
    safe_half_span = max(float(xy_half_span), 1e-6)
    safe_progress_scale = max(float(route_progress_scale_m), 1e-6)
    safe_speed_scale = max(float(speed_scale_mps), 1e-6)
    safe_checkpoint_count = max(int(checkpoint_count), 1)

    ship_x = float(state.get("ship_x", 0.0))
    ship_y = float(state.get("ship_y", 0.0))
    heading_deg = float(state.get("heading_deg", 0.0)) % 360.0
    speed = max(float(state.get("speed", 0.0)), 0.0)
    cross_track_error = float(state.get("cross_track_error", 0.0))
    route_progress_m = float(state.get("route_progress_m", 0.0))
    checkpoint_index = int(state.get("checkpoint_index", 0))

    heading_norm = (heading_deg / 180.0) - 1.0
    checkpoint_ratio = checkpoint_index / safe_checkpoint_count

    return np.array(
        [
            np.clip((ship_x - float(xy_center[0])) / safe_half_span, -1.0, 1.0),
            np.clip((ship_y - float(xy_center[1])) / safe_half_span, -1.0, 1.0),
            np.clip(heading_norm, -1.0, 1.0),
            np.clip(speed / safe_speed_scale, -1.0, 1.0),
            np.clip(cross_track_error / _CROSS_TRACK_LIMIT_M, -1.0, 1.0),
            np.clip(route_progress_m / safe_progress_scale, -1.0, 1.0),
            np.clip((2.0 * checkpoint_ratio) - 1.0, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def _normalize_angle(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi]``."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _safe_load_csv(path: Path, fallback: np.ndarray) -> np.ndarray:
    """Load a two-column CSV file or return the provided fallback array."""
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception:
        return np.asarray(fallback, dtype=np.float32)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) == 0:
        return np.asarray(fallback, dtype=np.float32)
    return data


def _calculate_perpendicular_lines(
    checkpoints: list[dict[str, Any]],
    line_length: float = 120.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build short perpendicular guidance lines for each checkpoint."""
    if not checkpoints:
        return []

    def _tangent_at(index: int) -> np.ndarray:
        """Return a normalized local path tangent vector at a checkpoint index."""
        if len(checkpoints) == 1:
            return np.array([1.0, 0.0], dtype=np.float32)
        if index == 0:
            tangent = checkpoints[1]["pos"] - checkpoints[0]["pos"]
        elif index == len(checkpoints) - 1:
            tangent = checkpoints[-1]["pos"] - checkpoints[-2]["pos"]
        else:
            tangent = 0.5 * (
                checkpoints[index + 1]["pos"] - checkpoints[index - 1]["pos"]
            )
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-9:
            return np.array([1.0, 0.0], dtype=np.float32)
        return (tangent / norm).astype(np.float32)

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for index, checkpoint in enumerate(checkpoints):
        tangent = _tangent_at(index)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        offset = normal * (line_length / 2.0)
        lines.append((checkpoint["pos"] + offset, checkpoint["pos"] - offset))
    return lines


class AirSimEnv(BaseEnv):
    """Live AirSim route-following environment for PPO training and inference."""

    def __init__(
        self,
        render_mode: Optional[str] = None,
        ship_pos: Optional[list[float] | tuple[float, float]] = None,
        target_pos: Optional[list[float] | tuple[float, float]] = None,
        max_steps: int = 14000,
        time_step: float = 0.1,
        training_mode: bool = True,
        no_rendering_mode: bool = False,
        use_pause_stepping: bool = True,
        use_sim_time_stepping: bool = False,
        sim_time_per_action: float = 0.1,
        frames_per_step: int = 8,
        physics_update_timeout: float = 1.0,
        state_poll_interval: float = 0.005,
        draw_calibration_reference: bool = True,
        simulation_speed: float = 1.0,
        save_slot: str = DEFAULT_AIRSIM_SAVE_SLOT,
        load_world_on_start: bool | None = None,
        capture_live_start_pose: bool | None = None,
        use_airsim: bool = True,
        ip: str = "",
        port: int = 41451,
        timeout: float = 10.0,
        max_retries: int = 3,
        health_check_interval: float = 30.0,
        csv_log_path: str = "",
        episode_log_path: str = "",
        normalize_reward: bool = True,
        reward_normalization_scale: float = _DEFAULT_REWARD_NORMALIZATION_SCALE,
        reset_mode: str = _RESET_MODE_REWIND_POSE,
        progress_log_interval_steps: int = 0,
        log_pause_watchdog_warnings: bool = False,
        latency_warning_threshold: float | None = None,
        curriculum_enabled: bool = False,
        curriculum_window_size: int = 8,
        curriculum_target_checkpoint: int | None = None,
        curriculum_frontier_increment: int = 4,
        curriculum_window_increment: int = 6,
        curriculum_success_patience: int = 2,
        curriculum_randomization_scale: float = 0.0,
        guidance_assist_enabled: bool = False,
        guidance_residual_steering_scale: float = 0.45,
        guidance_residual_throttle_scale: float = 0.55,
        steering_dynamics_enabled: bool = True,
        steering_smoothing_alpha: float = _DEFAULT_STEERING_SMOOTHING_ALPHA,
        max_steering_rate_per_second: float = _DEFAULT_MAX_STEERING_RATE_PER_SECOND,
        control_interface: str = _CONTROL_INTERFACE_CAR_PHYSICS,
        observation_profile: str = _OBSERVATION_PROFILE_ROUTE_RELATIVE_CAR,
    ) -> None:
        """Create a live UE5-backed RL environment."""
        super().__init__(render_mode=render_mode)

        self._constructor_kwargs = {
            "render_mode": render_mode,
            "ship_pos": list(ship_pos) if ship_pos is not None else None,
            "target_pos": list(target_pos) if target_pos is not None else None,
            "max_steps": int(max_steps),
            "time_step": float(time_step),
            "training_mode": bool(training_mode),
            "no_rendering_mode": bool(no_rendering_mode),
            "use_pause_stepping": bool(use_pause_stepping),
            "use_sim_time_stepping": bool(use_sim_time_stepping),
            "sim_time_per_action": float(sim_time_per_action),
            "frames_per_step": int(frames_per_step),
            "physics_update_timeout": float(physics_update_timeout),
            "state_poll_interval": float(state_poll_interval),
            "draw_calibration_reference": bool(draw_calibration_reference),
            "simulation_speed": float(simulation_speed),
            "save_slot": str(save_slot),
            "load_world_on_start": load_world_on_start,
            "capture_live_start_pose": capture_live_start_pose,
            "use_airsim": bool(use_airsim),
            "ip": str(ip),
            "port": int(port),
            "timeout": float(timeout),
            "max_retries": int(max_retries),
            "health_check_interval": float(health_check_interval),
            "csv_log_path": str(csv_log_path),
            "episode_log_path": str(episode_log_path),
            "normalize_reward": bool(normalize_reward),
            "reward_normalization_scale": float(reward_normalization_scale),
            "reset_mode": str(reset_mode),
            "progress_log_interval_steps": int(progress_log_interval_steps),
            "log_pause_watchdog_warnings": bool(log_pause_watchdog_warnings),
            "latency_warning_threshold": latency_warning_threshold,
            "curriculum_enabled": bool(curriculum_enabled),
            "curriculum_window_size": int(curriculum_window_size),
            "curriculum_target_checkpoint": curriculum_target_checkpoint,
            "curriculum_frontier_increment": int(curriculum_frontier_increment),
            "curriculum_window_increment": int(curriculum_window_increment),
            "curriculum_success_patience": int(curriculum_success_patience),
            "curriculum_randomization_scale": float(curriculum_randomization_scale),
            "guidance_assist_enabled": bool(guidance_assist_enabled),
            "guidance_residual_steering_scale": float(guidance_residual_steering_scale),
            "guidance_residual_throttle_scale": float(guidance_residual_throttle_scale),
            "steering_dynamics_enabled": bool(steering_dynamics_enabled),
            "steering_smoothing_alpha": float(steering_smoothing_alpha),
            "max_steering_rate_per_second": float(max_steering_rate_per_second),
            "control_interface": str(control_interface),
            "observation_profile": str(observation_profile),
        }

        self.max_steps = int(max_steps)
        self.time_step = float(time_step)
        self.training_mode = bool(training_mode)
        self.no_rendering_mode = bool(no_rendering_mode)
        self.use_pause_stepping = bool(use_pause_stepping)
        self.use_sim_time_stepping = bool(use_sim_time_stepping)
        self.sim_time_per_action = float(sim_time_per_action)
        self.frames_per_step = int(frames_per_step)
        self.physics_update_timeout = float(physics_update_timeout)
        self.state_poll_interval = float(state_poll_interval)
        self.draw_calibration_reference = bool(draw_calibration_reference)
        self.simulation_speed = float(simulation_speed)
        self.save_slot = str(save_slot)
        self.load_world_on_start = (
            bool(load_world_on_start)
            if load_world_on_start is not None
            else not self.training_mode
        )
        self.capture_live_start_pose = (
            bool(capture_live_start_pose)
            if capture_live_start_pose is not None
            else (self.training_mode and not self.load_world_on_start)
        )
        self.episode_log_path = str(episode_log_path)
        self.normalize_reward = bool(normalize_reward)
        self.reward_normalization_scale = max(
            float(reward_normalization_scale),
            1e-6,
        )
        self.reset_mode = str(reset_mode).strip().lower() or _RESET_MODE_REWIND_POSE
        self.progress_log_interval_steps = int(progress_log_interval_steps)
        self.log_pause_watchdog_warnings = bool(log_pause_watchdog_warnings)
        self.latency_warning_threshold = (
            float(latency_warning_threshold)
            if latency_warning_threshold is not None
            else None
        )

        self.curriculum_enabled = bool(curriculum_enabled)
        self.curriculum_window_size = max(1, int(curriculum_window_size))
        self.curriculum_frontier_increment = max(1, int(curriculum_frontier_increment))
        self.curriculum_window_increment = max(1, int(curriculum_window_increment))
        self.curriculum_success_patience = max(1, int(curriculum_success_patience))
        self.curriculum_randomization_scale = float(curriculum_randomization_scale)
        self._curriculum_success_streak = 0

        self.guidance_assist_enabled = bool(guidance_assist_enabled)
        self.guidance_residual_steering_scale = float(np.clip(guidance_residual_steering_scale, 0.0, 1.0))
        self.guidance_residual_throttle_scale = float(np.clip(guidance_residual_throttle_scale, 0.0, 1.0))
        self.steering_dynamics_enabled = bool(steering_dynamics_enabled)
        self.steering_smoothing_alpha = float(np.clip(steering_smoothing_alpha, 0.0, 1.0))
        self.max_steering_rate_per_second = max(float(max_steering_rate_per_second), 0.0)
        self.control_interface = str(control_interface).strip().lower() or _CONTROL_INTERFACE_CAR_PHYSICS
        self.observation_profile = (
            str(observation_profile).strip().lower()
            or _OBSERVATION_PROFILE_ROUTE_RELATIVE_CAR
        )
        self.mdp_step_seconds = self._resolve_mdp_step_seconds()
        if self.use_pause_stepping and self.use_sim_time_stepping:
            self.time_step = self.mdp_step_seconds

        self.initial_ship_pos = np.asarray(
            ship_pos if ship_pos is not None else DEFAULT_FRAMEWORK_SHIP_POS,
            dtype=np.float32,
        )
        self._requested_target_pos = (
            np.asarray(target_pos, dtype=np.float32)
            if target_pos is not None
            else None
        )

        self._rng = np.random.default_rng()
        self._other_agent_states: dict[str, Any] = {}
        self._world_prepared = False
        self._captured_startup_pose_airsim: dict[str, float] | None = None
        self._captured_startup_framework_heading: float | None = None
        self._pending_run_events: list[dict[str, Any]] = []
        self._log_next_reset_event = True
        self._episode_index = 0
        self._episode_start_checkpoint_index = 0
        self._episode_success = False
        self._last_pause_timeout = False
        self._last_collision = False
        self._route_projection_segment = 0
        self._route_progress_m = 0.0
        self._route_progress_ratio = 0.0
        self._distance_scale_m = _MIN_DISTANCE_SCALE_M
        self._last_ue5_timestamp: int = 0
        self._last_state_signature: tuple[float, float, float, int] | None = None
        self._repeated_state_count = 0
        self._state_update_reliable = True
        self.last_physics_wait_s = 0.0
        self.last_step_latency_s = 0.0
        self.last_controls: dict[str, float] = {"steering": 0.0, "throttle": 0.0, "brake": 1.0}
        self.last_policy_action = np.array([0.0, -1.0], dtype=np.float32)
        self.last_steering_target = 0.0
        self.last_steering_rate_limited = False
        self.last_checkpoint_reward_raw = 0.0
        self.last_checkpoint_reward_effective = 0.0
        self.previous_action = self.neutral_action()
        self.performed_action = self.neutral_action()
        self.ship_pos = self.initial_ship_pos.copy()
        self.target_pos = (
            self._requested_target_pos.copy()
            if self._requested_target_pos is not None
            else np.asarray(DEFAULT_FRAMEWORK_TARGET_POS, dtype=np.float32)
        )
        self.ship_heading = 0.0
        self.ship_speed = 0.0
        self.ship_yaw_rate = 0.0
        self.heading_error = 0.0
        self.cross_error = 0.0
        self.cross_track_error_delta = 0.0
        self.ukc = 0.0
        self.step_count = 0
        self.checkpoint_index = 1
        self._episode_abs_cross_track_sum = 0.0
        self._episode_quality_samples = 0
        self._episode_max_abs_cross_track_error = 0.0
        self._episode_reward_sum = 0.0
        self._episode_reward_steps = 0

        self.overall = self._load_outline_points(_DEFAULT_OVERALL_CSV)
        self.obstacles = self._load_outline_points(_DEFAULT_OBSTACLES_CSV)
        self._rebuild_route_geometry()
        self._curriculum_frontier_index = self._initial_curriculum_frontier(
            curriculum_target_checkpoint
        )
        self._validate_configuration()

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._build_observation_space()

        bridge_config = AirSimConfig(
            use_airsim=use_airsim,
            ip=ip,
            port=port,
            timeout=timeout,
            max_retries=max_retries,
            health_check_interval=health_check_interval,
            csv_log_path=csv_log_path,
        )
        self.bridge = AirSimBridge(bridge_config)
        if not self.bridge.is_connected:
            raise ConnectionError(
                f"AirSimEnv could not connect to UE5 at {bridge_config.resolved_ip}:{bridge_config.port}."
            )

        self.coord_transform = CoordTransform.identity(
            framework_origin=(float(self.initial_ship_pos[0]), float(self.initial_ship_pos[1]))
        )
        self.phys_sim = SimpleNamespace(
            state=SimpleNamespace(
                x=float(self.ship_pos[0]),
                y=float(self.ship_pos[1]),
                heading=0.0,
                u=0.0,
                v=0.0,
                r=0.0,
            )
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "AirSimEnv":
        """Create a fresh UE5-backed copy for vectorized PPO wrappers."""
        return type(self)(**copy.deepcopy(self._constructor_kwargs, memo))

    def neutral_action(self) -> np.ndarray:
        """Return a neutral AirSim action with zero propulsion."""
        return np.array([0.0, -1.0], dtype=np.float32)

    def get_agent_state(self) -> Any:
        """Return the latest lightweight simulator state proxy."""
        return self.phys_sim.state

    def set_other_agents_states(self, agent_states: dict[str, Any]) -> None:
        """Store other-agent state snapshots for framework compatibility."""
        self._other_agent_states = {
            agent_name: state
            for agent_name, state in dict(agent_states or {}).items()
            if agent_name != self.env_name
        }

    def randomize(self, randomization_scale: Optional[float] = None) -> None:
        """Adjust the curriculum start randomization scale."""
        if randomization_scale is not None:
            self.curriculum_randomization_scale = float(randomization_scale)

    def render(self) -> None:
        """UE5 handles rendering directly, so this method is a no-op."""
        return None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the UE5 episode and return the first observation."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._prepare_world_if_needed()
        self._episode_index += 1
        self._episode_success = False
        self._last_pause_timeout = False
        self._last_collision = False
        self._reset_sync_metrics()

        start_index, start_pos = self._select_episode_start()
        start_heading = self._select_episode_start_heading(start_index)

        if self._world_prepared and self.reset_mode == _RESET_MODE_BRIDGE_RESET:
            try:
                self.bridge.reset()
            except Exception:
                logger.debug("AirSim reset failed during episode reset.", exc_info=True)

        self.bridge.set_simulation_speed(self.simulation_speed)
        self.bridge.set_rendering_enabled(not self.no_rendering_mode, force=True)
        self._set_pause_mode(self.use_pause_stepping)

        startup_pose = self._startup_reset_pose_for(start_index, start_pos, start_heading)
        if startup_pose is not None:
            airsim_x = float(startup_pose["x"])
            airsim_y = float(startup_pose["y"])
            airsim_z = float(startup_pose["z"])
            airsim_yaw = float(startup_pose["yaw"])
        else:
            airsim_x, airsim_y, airsim_z = self.coord_transform.framework_to_airsim(
                float(start_pos[0]),
                float(start_pos[1]),
                float(getattr(self.bridge, "render_z", 0.0)),
            )
            airsim_yaw = self.coord_transform.framework_heading_to_airsim_yaw(start_heading)
        reset_settled = False
        for reset_attempt in range(2):
            self.bridge.set_pose(
                airsim_x,
                airsim_y,
                airsim_z,
                airsim_yaw,
                preserve_current_z=False,
                preserve_roll_pitch=False,
            )
            self._apply_parked_controls()
            reset_settled = self._settle_after_reset(start_pos=start_pos, start_heading=start_heading)
            if reset_settled:
                break
            logger.warning(
                "AirSimEnv reset pose did not settle on attempt %d/2; retrying teleport.",
                reset_attempt + 1,
            )
        if not reset_settled:
            raise RuntimeError("AirSimEnv reset failed because the post-teleport state never stabilized.")

        # Keep reset behavior unchanged while forcing a deterministic zero-velocity start state.
        self._enforce_zero_velocity_after_reset()

        self.step_count = 0
        self.previous_action = self.neutral_action()
        self.performed_action = self.neutral_action()
        self.last_policy_action = self.neutral_action()
        self.last_steering_target = 0.0
        self.last_steering_rate_limited = False
        self.ship_yaw_rate = 0.0
        self.cross_track_error_delta = 0.0
        self.last_checkpoint_reward_raw = 0.0
        self.last_checkpoint_reward_effective = 0.0
        self._episode_abs_cross_track_sum = 0.0
        self._episode_quality_samples = 0
        self._episode_max_abs_cross_track_error = 0.0
        self._episode_reward_sum = 0.0
        self._episode_reward_steps = 0
        self._episode_start_checkpoint_index = start_index
        self.checkpoint_index = min(start_index + 1, len(self.checkpoints) - 1)
        self._route_progress_m = float(self.route_cumulative_distances[start_index])
        self._route_progress_ratio = self._route_progress_m / max(self.route_length_m, 1e-6)

        self._reset_sync_metrics()
        if not self._refresh_state_from_bridge(track_sync=False):
            raise RuntimeError("AirSimEnv reset failed because the initial simulator state could not be read reliably.")
        observation = self._build_observation()
        info = self._build_info(
            reward=0.0,
            raw_reward=0.0,
            terminated=False,
            truncated=False,
            outcome="reset",
        )
        if self._log_next_reset_event:
            self._record_run_event(
                "respawn",
                start_checkpoint_index=int(start_index),
                used_startup_anchor=bool(startup_pose is not None),
                start_x=float(start_pos[0]),
                start_y=float(start_pos[1]),
                start_heading_deg=float(math.degrees(start_heading)),
                airsim_x=float(airsim_x),
                airsim_y=float(airsim_y),
                airsim_z=float(airsim_z),
                airsim_yaw_deg=float(math.degrees(airsim_yaw)),
                outcome="reset",
            )
        self._log_next_reset_event = False
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one policy action to UE5 and return the next transition."""
        if not self.bridge.is_connected:
            observation = self._build_observation()
            reward = self._normalize_reward(_DISCONNECT_PENALTY)
            info = self._build_info(
                reward=reward,
                raw_reward=_DISCONNECT_PENALTY,
                terminated=True,
                truncated=False,
                outcome="disconnected",
            )
            return observation, float(reward), True, False, info

        step_start = time.perf_counter()
        action = validate_policy_action(action)
        self.previous_action = self.performed_action.copy()
        self.last_policy_action = action.copy()
        effective_action = self._compose_effective_action(action)
        controls = self._map_policy_action_to_controls(effective_action)
        self.last_controls = {
            "steering": float(getattr(controls, "steering", 0.0)),
            "throttle": float(getattr(controls, "throttle", 0.0)),
            "brake": float(getattr(controls, "brake", 0.0)),
        }

        previous_state = self._build_reward_state()
        previous_cross_error = float(self.cross_error)

        self.bridge.set_controls(controls)
        self.last_physics_wait_s = self._advance_simulation()
        state_updated = self._refresh_state_from_bridge(track_sync=True)
        self.step_count += 1
        self.performed_action = effective_action.copy()
        self.cross_track_error_delta = float(self.cross_error - previous_cross_error) if state_updated else 0.0

        if not state_updated:
            logger.warning(
                "AirSimEnv could not read a valid simulator state after stepping; truncating the episode."
            )
            raw_reward = _TIMEOUT_PENALTY
            reward = self._normalize_reward(raw_reward)
            self.last_step_latency_s = time.perf_counter() - step_start
            observation = self._build_observation()
            info = self._build_info(
                reward=reward,
                raw_reward=raw_reward,
                terminated=False,
                truncated=True,
                outcome="state_unavailable",
            )
            self._record_run_event(
                "episode_finished",
                outcome="state_unavailable",
                reward=float(reward),
                raw_reward=float(raw_reward),
                terminated=False,
                truncated=True,
            )
            return observation, float(reward), False, True, info

        self._episode_abs_cross_track_sum += abs(self.cross_error)
        self._episode_quality_samples += 1
        self._episode_max_abs_cross_track_error = max(
            self._episode_max_abs_cross_track_error,
            abs(float(self.cross_error)),
        )

        checkpoint_reward = self._update_checkpoint_progress()
        self.last_checkpoint_reward_raw = float(checkpoint_reward)
        self.last_checkpoint_reward_effective = float(checkpoint_reward)

        collision_status = self._detect_collision()
        collision_read_failed = collision_status is None
        collision = bool(collision_status) if collision_status is not None else False

        current_state = self._build_reward_state()
        raw_reward, offtrack = self.calculate_reward(
            current_state=current_state,
            previous_state=previous_state,
            checkpoint_reward=checkpoint_reward,
        )
        all_checkpoints_passed = bool(self.checkpoint_index >= len(self.checkpoints) - 1)
        reached_target = self._has_reached_target()
        success = bool(
            reached_target
            and not offtrack
            and not collision
            and self.bridge.is_connected
        )
        terminated = bool(success or collision or offtrack or not self.bridge.is_connected)
        truncated = bool(
            self.step_count >= self.max_steps
            or self._last_pause_timeout
            or collision_read_failed
        )

        if success:
            raw_reward += _SUCCESS_REWARD
            all_checkpoints_passed = True
        if collision:
            raw_reward += _COLLISION_PENALTY
            raw_reward = min(raw_reward, _COLLISION_PENALTY)
        if offtrack:
            raw_reward = min(raw_reward, _OFFTRACK_PENALTY)
        if not self.bridge.is_connected:
            raw_reward += _DISCONNECT_PENALTY
        if collision_read_failed:
            raw_reward = min(raw_reward, -0.5)
        if truncated and self._last_pause_timeout:
            raw_reward += _TIMEOUT_PENALTY

        reward = self._normalize_reward(raw_reward)

        self._episode_reward_sum += reward
        self._episode_reward_steps += 1

        self.last_step_latency_s = time.perf_counter() - step_start
        if (
            self.latency_warning_threshold is not None
            and self.last_step_latency_s > self.latency_warning_threshold
        ):
            logger.warning(
                "AirSimEnv step latency %.3fs exceeded threshold %.3fs.",
                self.last_step_latency_s,
                self.latency_warning_threshold,
            )

        if self.progress_log_interval_steps > 0 and self.step_count % self.progress_log_interval_steps == 0:
            logger.info(
                "AirSimEnv step=%d progress=%.1f%% checkpoint=%d/%d reward=%+.3f",
                self.step_count,
                self._route_progress_ratio * 100.0,
                self.checkpoint_index,
                len(self.checkpoints) - 1,
                reward,
            )

        if terminated or truncated:
            curriculum_success = bool(success and not collision and not offtrack and not truncated)
            self._episode_success = curriculum_success
            self._update_curriculum_after_episode(success=curriculum_success)

        outcome = "running"
        if collision:
            outcome = "collision"
        elif offtrack:
            outcome = "offtrack"
        elif success:
            outcome = "success"
        elif not self.bridge.is_connected:
            outcome = "disconnected"
        elif collision_read_failed:
            outcome = "collision_unavailable"
        elif self._last_pause_timeout:
            outcome = "timeout"
        elif truncated:
            outcome = "truncated"

        observation = self._build_observation()
        info = self._build_info(
            reward=reward,
            raw_reward=raw_reward,
            terminated=terminated,
            truncated=truncated,
            outcome=outcome,
            all_checkpoints_passed=all_checkpoints_passed,
        )
        if terminated or truncated:
            self._record_run_event(
                "episode_finished",
                outcome=outcome,
                reward=float(reward),
                raw_reward=float(raw_reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Disconnect from AirSim and release bridge resources."""
        try:
            if getattr(self, "bridge", None) is not None:
                self.bridge.disconnect()
        except Exception:
            logger.debug("AirSimEnv close() swallowed a bridge disconnect error.", exc_info=True)

    def _prepare_world_if_needed(self) -> None:
        """Load the requested save slot and coordinate transform once per env."""
        if self._world_prepared:
            return

        if self.load_world_on_start and self.save_slot:
            try:
                load_started_at = time.perf_counter()
                self.bridge.load_world_save(save_slot=self.save_slot)
                settled_pose = self._settle_after_world_load()
                self._record_run_event(
                    "world_loaded",
                    save_slot=self.save_slot,
                    load_duration_s=float(time.perf_counter() - load_started_at),
                    airsim_x=float(settled_pose["x"]) if settled_pose is not None else None,
                    airsim_y=float(settled_pose["y"]) if settled_pose is not None else None,
                    airsim_z=float(settled_pose["z"]) if settled_pose is not None else None,
                    airsim_yaw_deg=(
                        float(math.degrees(float(settled_pose["yaw"])))
                        if settled_pose is not None
                        else None
                    ),
                )
            except Exception as exc:
                logger.warning("Could not load AirSim save slot '%s': %s", self.save_slot, exc)
                self._record_run_event(
                    "world_load_failed",
                    save_slot=self.save_slot,
                    details={"error": str(exc)},
                )

        self.bridge.set_simulation_speed(self.simulation_speed)
        self.bridge.set_rendering_enabled(not self.no_rendering_mode, force=True)

        try:
            self.coord_transform, _ = load_saved_calibration_reference(
                self.bridge.client,
                draw_reference=self.draw_calibration_reference,
                ship_pos=(float(self.initial_ship_pos[0]), float(self.initial_ship_pos[1])),
                target_pos=(float(self.target_pos[0]), float(self.target_pos[1])),
            )
        except FileNotFoundError:
            logger.warning("No AirSim calibration.json found; falling back to identity transform.")
            self.coord_transform = CoordTransform.identity(
                framework_origin=(float(self.initial_ship_pos[0]), float(self.initial_ship_pos[1]))
            )
        except Exception as exc:
            logger.warning("Could not load saved AirSim calibration; using identity transform. %s", exc)
            self.coord_transform = CoordTransform.identity(
                framework_origin=(float(self.initial_ship_pos[0]), float(self.initial_ship_pos[1]))
            )

        self.bridge.coord_transform = self.coord_transform
        try:
            pose = self.bridge.get_vehicle_pose()
            self.bridge.render_z = float(pose.position.z_val)
        except _AIRSIM_READ_EXCEPTIONS:
            self.bridge.render_z = 0.0

        self._capture_startup_anchor_if_needed()
        self._record_run_event(
            "world_prepared",
            used_startup_anchor=bool(self._captured_startup_pose_airsim is not None),
            save_slot=self.save_slot,
            load_world_on_start=bool(self.load_world_on_start),
            capture_live_start_pose=bool(self.capture_live_start_pose),
            no_rendering_mode=bool(self.no_rendering_mode),
            simulation_speed=float(self.simulation_speed),
            reset_mode=str(self.reset_mode),
        )

        self._world_prepared = True

    def _settle_after_world_load(self) -> dict[str, float] | None:
        """Wait briefly for the freshly loaded save-slot spawn pose to stabilize."""
        self._apply_parked_controls()
        deadline = time.monotonic() + _WORLD_LOAD_SETTLE_TIMEOUT_S
        previous_sample: tuple[float, float, float, float] | None = None
        stable_samples = 0

        while time.monotonic() < deadline:
            try:
                pose = self.bridge.get_vehicle_pose()
                sample = (
                    float(pose.position.x_val),
                    float(pose.position.y_val),
                    float(pose.position.z_val),
                    float(yaw_from_quaternion(pose.orientation)),
                )
            except _AIRSIM_READ_EXCEPTIONS:
                logger.debug("Could not read vehicle pose while settling world load.", exc_info=True)
                time.sleep(_WORLD_LOAD_SETTLE_POLL_S)
                continue

            if previous_sample is not None:
                position_delta = math.hypot(
                    sample[0] - previous_sample[0],
                    sample[1] - previous_sample[1],
                )
                yaw_delta = abs(_normalize_angle(sample[3] - previous_sample[3]))
                if (
                    position_delta <= _WORLD_LOAD_SETTLE_POSITION_EPS_M
                    and yaw_delta <= _WORLD_LOAD_SETTLE_YAW_EPS_RAD
                ):
                    stable_samples += 1
                    if stable_samples >= 2:
                        return {
                            "x": sample[0],
                            "y": sample[1],
                            "z": sample[2],
                            "yaw": sample[3],
                        }
                else:
                    stable_samples = 0

            previous_sample = sample
            time.sleep(_WORLD_LOAD_SETTLE_POLL_S)

        if previous_sample is None:
            return None

        logger.info(
            "Continuing after save-slot load settle timeout; using latest observed startup pose."
        )
        return {
            "x": previous_sample[0],
            "y": previous_sample[1],
            "z": previous_sample[2],
            "yaw": previous_sample[3],
        }

    def _capture_startup_anchor_if_needed(self) -> None:
        """Adopt the live simulator spawn pose as the reset anchor once."""
        if not self.capture_live_start_pose or self._captured_startup_pose_airsim is not None:
            return

        try:
            pose = self.bridge.get_vehicle_pose()
        except _AIRSIM_READ_EXCEPTIONS:
            logger.debug("Could not read the live AirSim startup pose.", exc_info=True)
            return

        try:
            airsim_x = float(pose.position.x_val)
            airsim_y = float(pose.position.y_val)
            airsim_z = float(pose.position.z_val)
            airsim_yaw = float(yaw_from_quaternion(pose.orientation))
        except _AIRSIM_DECODE_EXCEPTIONS:
            logger.debug("Could not decode the live AirSim startup pose.", exc_info=True)
            return

        framework_x, framework_y = self.coord_transform.airsim_to_framework(
            airsim_x,
            airsim_y,
        )
        framework_heading = self.coord_transform.airsim_yaw_to_framework_heading(
            airsim_yaw
        )
        captured_framework_pos = np.asarray(
            [framework_x, framework_y],
            dtype=np.float32,
        )

        drift_m = float(
            np.linalg.norm(captured_framework_pos - np.asarray(self.initial_ship_pos, dtype=np.float32))
        )
        logger.info(
            "Adopting live AirSim startup pose as the training reset anchor "
            "(drift %.1fm, heading %.1fdeg).",
            drift_m,
            math.degrees(framework_heading),
        )

        self._captured_startup_pose_airsim = {
            "x": airsim_x,
            "y": airsim_y,
            "z": airsim_z,
            "yaw": airsim_yaw,
        }
        self._captured_startup_framework_heading = float(framework_heading)
        self.bridge.render_z = airsim_z
        self.initial_ship_pos = captured_framework_pos
        self.ship_pos = captured_framework_pos.copy()
        self._rebuild_route_geometry()
        self._record_run_event(
            "startup_anchor_captured",
            used_startup_anchor=True,
            ship_x=float(framework_x),
            ship_y=float(framework_y),
            heading_deg=float(math.degrees(framework_heading)),
            airsim_x=float(airsim_x),
            airsim_y=float(airsim_y),
            airsim_z=float(airsim_z),
            airsim_yaw_deg=float(math.degrees(airsim_yaw)),
        )

    def _rebuild_route_geometry(self) -> None:
        """Recompute route geometry after the chosen route anchor changes."""
        self.route_points = self._build_route_points()
        self.route_cumulative_distances = self._compute_cumulative_distances(
            self.route_points
        )
        self.route_length_m = (
            float(self.route_cumulative_distances[-1])
            if len(self.route_cumulative_distances)
            else 1.0
        )
        self._distance_scale_m = min(
            max(_MIN_DISTANCE_SCALE_M, 3.0 * _ROUTE_POINT_SPACING_M),
            800.0,
        )
        self.checkpoints = self._build_checkpoints(
            self.route_points,
            self.route_cumulative_distances,
        )
        self.target_pos = self.checkpoints[-1]["pos"].copy()

    def _startup_reset_pose_for(
        self,
        start_index: int,
        start_pos: np.ndarray,
        start_heading: float,
    ) -> dict[str, float] | None:
        """Return the exact captured AirSim reset pose for the canonical start."""
        if self._captured_startup_pose_airsim is None:
            return None
        if int(start_index) != 0:
            return None
        if self.curriculum_randomization_scale > 0.0:
            return None
        if float(np.linalg.norm(np.asarray(start_pos, dtype=np.float32) - self.initial_ship_pos)) > 1e-3:
            return None
        if self._captured_startup_framework_heading is None:
            return None
        heading_error = abs(
            _normalize_angle(
                float(start_heading) - float(self._captured_startup_framework_heading)
            )
        )
        if heading_error > math.radians(45.0):
            return None
        return dict(self._captured_startup_pose_airsim)

    def _apply_parked_controls(self) -> None:
        """Stop the AirSim vehicle after a reset teleport."""
        controls = build_car_controls(
            self.bridge.airsim,
            steering=0.0,
            throttle=0.0,
            brake=1.0,
        )
        self.bridge.set_controls(controls)
        self.last_controls = {"steering": 0.0, "throttle": 0.0, "brake": 1.0}

    def _enforce_zero_velocity_after_reset(self) -> None:
        """Explicitly zero linear/angular velocity immediately after reset settle."""
        try:
            pose = self.bridge.get_vehicle_pose()
            airsim_x = float(pose.position.x_val)
            airsim_y = float(pose.position.y_val)
            airsim_z = float(pose.position.z_val)
            airsim_yaw = float(yaw_from_quaternion(pose.orientation))
        except _AIRSIM_READ_EXCEPTIONS:
            logger.debug("Could not read pose while enforcing reset zero-velocity state.", exc_info=True)
            self._apply_parked_controls()
            return

        try:
            self.bridge.set_kinematics(
                airsim_x,
                airsim_y,
                airsim_z,
                airsim_yaw,
                linear_velocity_x=0.0,
                linear_velocity_y=0.0,
                linear_velocity_z=0.0,
                angular_velocity_x=0.0,
                angular_velocity_y=0.0,
                angular_velocity_z=0.0,
                preserve_current_z=False,
                preserve_roll_pitch=False,
            )
        except Exception:
            logger.debug("Could not enforce reset zero-velocity state via kinematics.", exc_info=True)

        self._apply_parked_controls()

    def _set_pause_mode(self, is_paused: bool) -> None:
        """Toggle paused stepping mode."""
        try:
            self.bridge.set_pause(bool(is_paused))
        except Exception:
            logger.debug("Could not change AirSim pause mode.", exc_info=True)

    def _advance_simulation(self) -> float:
        """Advance the UE5 simulation according to the configured step mode."""
        step_start = time.perf_counter()
        self._last_pause_timeout = False

        if not self.use_pause_stepping:
            sleep_s = max(
                self.mdp_step_seconds / max(self.simulation_speed, 1e-6),
                self.state_poll_interval,
            )
            time.sleep(sleep_s)
            return time.perf_counter() - step_start

        self.bridge.set_pause(True)
        before_signature = self._snapshot_progress_signature()
        if self.use_sim_time_stepping:
            self.bridge.continue_for_time(self.mdp_step_seconds)
        else:
            self.bridge.continue_for_frames(self.frames_per_step)

        deadline = step_start + self.physics_update_timeout
        while time.perf_counter() < deadline:
            if self._has_progressed_since(before_signature):
                return time.perf_counter() - step_start
            time.sleep(self.state_poll_interval)

        self._last_pause_timeout = True
        if self.log_pause_watchdog_warnings:
            logger.warning(
                "AirSim pause-step watchdog timed out after %.3fs.",
                self.physics_update_timeout,
            )
        return time.perf_counter() - step_start

    def _snapshot_progress_signature(self) -> dict[str, float | int] | None:
        """Capture a lightweight telemetry signature for step-progress checks."""
        signature: dict[str, float | int] = {}
        try:
            state = self.bridge.get_state()
        except Exception:
            state = None

        if isinstance(state, dict) and not bool(state.get("state_read_failed")):
            try:
                signature["timestamp"] = int(state.get("timestamp", 0) or 0)
                signature["pos_x"] = float(state["pos_x"])
                signature["pos_y"] = float(state["pos_y"])
                signature["yaw"] = float(state["yaw"])
            except (KeyError, TypeError, ValueError):
                signature.clear()

        try:
            pose = self.bridge.get_vehicle_pose()
        except Exception:
            pose = None

        if pose is not None:
            try:
                signature["pose_x"] = float(pose.position.x_val)
                signature["pose_y"] = float(pose.position.y_val)
            except Exception:
                pass
            try:
                signature["pose_yaw"] = float(yaw_from_quaternion(pose.orientation))
            except Exception:
                pass

        return signature or None

    def _has_progressed_since(self, before_signature: dict[str, float | int] | None) -> bool:
        """Return whether UE5 telemetry changed after a step command."""
        if before_signature is None:
            return False

        after_signature = self._snapshot_progress_signature()
        if not after_signature:
            return False

        if int(after_signature.get("timestamp", 0)) != int(before_signature.get("timestamp", 0)):
            return True

        for x_key, y_key, yaw_key in (
            ("pose_x", "pose_y", "pose_yaw"),
            ("pos_x", "pos_y", "yaw"),
        ):
            if x_key not in before_signature or x_key not in after_signature:
                continue
            dx = float(after_signature[x_key]) - float(before_signature[x_key])
            dy = float(after_signature[y_key]) - float(before_signature[y_key])
            if math.hypot(dx, dy) > _STATE_PROGRESS_POSITION_EPS_M:
                return True
            if yaw_key in before_signature and yaw_key in after_signature:
                yaw_delta = abs(
                    _normalize_angle(
                        float(after_signature[yaw_key]) - float(before_signature[yaw_key])
                    )
                )
                if yaw_delta > _STATE_PROGRESS_YAW_EPS_RAD:
                    return True

        return False

    def _refresh_state_from_bridge(self, track_sync: bool = True) -> bool:
        """Pull the latest pose and velocity state from AirSim.

        Returns:
            bool: ``True`` when at least one valid telemetry snapshot was
            available for the update, ``False`` otherwise.
        """
        state = self.bridge.get_state()
        state_failed = not isinstance(state, dict) or bool(state.get("state_read_failed"))
        airsim_x: float | None = None
        airsim_y: float | None = None
        airsim_yaw: float | None = None
        pose = None
        try:
            pose = self.bridge.get_vehicle_pose()
        except _AIRSIM_READ_EXCEPTIONS:
            pose = None

        if pose is not None:
            try:
                airsim_x = float(pose.position.x_val)
                airsim_y = float(pose.position.y_val)
            except _AIRSIM_DECODE_EXCEPTIONS:
                airsim_x = None
                airsim_y = None
            try:
                airsim_yaw = float(yaw_from_quaternion(pose.orientation))
            except _AIRSIM_DECODE_EXCEPTIONS:
                airsim_yaw = None

        if (
            airsim_x is None
            or airsim_y is None
            or airsim_yaw is None
        ) and not state_failed:
            try:
                airsim_x = float(state["pos_x"])
                airsim_y = float(state["pos_y"])
                airsim_yaw = float(state["yaw"])
            except (KeyError, TypeError, ValueError):
                airsim_x = None
                airsim_y = None
                airsim_yaw = None

        if airsim_x is None or airsim_y is None or airsim_yaw is None:
            self._repeated_state_count += 1
            self._state_update_reliable = False
            logger.debug("AirSimEnv state refresh skipped because no valid telemetry snapshot was available.")
            return False

        framework_x, framework_y = self.coord_transform.airsim_to_framework(airsim_x, airsim_y)
        framework_heading = self.coord_transform.airsim_yaw_to_framework_heading(
            airsim_yaw
        )

        self.ship_pos = np.array([framework_x, framework_y], dtype=np.float32)
        self.ship_heading = float(framework_heading)
        self.ship_speed = 0.0
        self._last_ue5_timestamp = 0
        yaw_rate = 0.0
        if not state_failed:
            try:
                self.ship_speed = float(max(float(state.get("speed", 0.0)), 0.0))
            except (TypeError, ValueError):
                self.ship_speed = 0.0
            try:
                self._last_ue5_timestamp = int(state.get("timestamp", 0) or 0)
            except (TypeError, ValueError):
                self._last_ue5_timestamp = 0
            try:
                yaw_rate = self._airsim_yaw_rate_to_framework(float(state.get("ang_vel_z", 0.0)))
            except (TypeError, ValueError):
                yaw_rate = 0.0
        self.ship_yaw_rate = float(yaw_rate)
        if track_sync:
            self._update_sync_metrics()
        projection = self._project_onto_route(self.ship_pos)
        self._route_projection_segment = int(projection["segment_index"])
        self._route_progress_m = float(projection["progress_m"])
        self._route_progress_ratio = float(
            np.clip(self._route_progress_m / max(self.route_length_m, 1e-6), 0.0, 1.0)
        )
        self.cross_error = float(projection["cross_track_error"])
        self.heading_error = self._current_heading_error()
        self.phys_sim.state = SimpleNamespace(
            x=float(self.ship_pos[0]),
            y=float(self.ship_pos[1]),
            heading=float(self.ship_heading),
            u=float(self.ship_speed),
            v=0.0,
            r=float(self.ship_yaw_rate),
        )
        if state_failed:
            self._state_update_reliable = False
            return False
        return True

    def _validate_configuration(self) -> None:
        """Validate the control and timing model used for the active env."""
        if self.control_interface not in _CONTROL_MAPPERS:
            raise ValueError(
                "AirSimEnv currently supports only these control interfaces: "
                f"{sorted(_CONTROL_MAPPERS)}."
            )
        if (self.control_interface, self.observation_profile) not in _OBSERVATION_BUILDERS:
            raise ValueError(
                "AirSimEnv currently does not support observation_profile="
                f"'{self.observation_profile}' for control_interface='{self.control_interface}'."
            )
        if self.reset_mode not in {_RESET_MODE_REWIND_POSE, _RESET_MODE_BRIDGE_RESET}:
            raise ValueError(
                "AirSimEnv reset_mode must be either "
                f"'{_RESET_MODE_REWIND_POSE}' or '{_RESET_MODE_BRIDGE_RESET}'."
            )
        if self.training_mode and self.use_pause_stepping and not self.use_sim_time_stepping:
            raise ValueError(
                "AirSimEnv training with paused stepping requires "
                "use_sim_time_stepping=True so one RL step has an explicit simulated duration."
            )
        if self.training_mode and not self.use_pause_stepping:
            logger.warning(
                "AirSimEnv training is running without pause stepping; transition timing may be less deterministic."
            )

    def _resolve_mdp_step_seconds(self) -> float:
        """Resolve the simulated duration represented by one RL step."""
        if self.use_pause_stepping and self.use_sim_time_stepping:
            step_seconds = float(self.sim_time_per_action)
        else:
            step_seconds = float(self.time_step)
        if step_seconds <= 0.0:
            raise ValueError("AirSimEnv requires a positive RL step duration.")
        return step_seconds

    def _reset_sync_metrics(self) -> None:
        """Reset state-synchronization metrics at episode boundaries."""
        self._last_state_signature = None
        self._repeated_state_count = 0
        self._state_update_reliable = True

    def _update_sync_metrics(self) -> None:
        """Track whether AirSim returned a fresh state for this RL step."""
        signature = (
            round(float(self.ship_pos[0]), 3),
            round(float(self.ship_pos[1]), 3),
            round(float(self.ship_heading), 4),
            int(self._last_ue5_timestamp),
        )
        if self._last_state_signature == signature:
            self._repeated_state_count += 1
            self._state_update_reliable = False
        else:
            self._repeated_state_count = 0
            self._state_update_reliable = True
        self._last_state_signature = signature

    def _settle_after_reset(self, start_pos: np.ndarray, start_heading: float) -> bool:
        """Wait briefly until the teleported reset pose is reflected in telemetry."""
        deadline = time.perf_counter() + _RESET_SETTLE_TIMEOUT_S
        while time.perf_counter() < deadline:
            if not self._refresh_state_from_bridge():
                time.sleep(_RESET_SETTLE_POLL_S)
                continue
            pos_error = float(np.linalg.norm(self.ship_pos - np.asarray(start_pos, dtype=np.float32)))
            heading_error = abs(_normalize_angle(self.ship_heading - float(start_heading)))
            if pos_error <= _RESET_POSITION_TOLERANCE_M and heading_error <= math.radians(20.0):
                return True
            time.sleep(_RESET_SETTLE_POLL_S)

        logger.warning(
            "AirSimEnv reset pose did not fully settle within %.1fs; "
            "marking reset as failed with pos_error=%.1fm.",
            _RESET_SETTLE_TIMEOUT_S,
            float(np.linalg.norm(self.ship_pos - np.asarray(start_pos, dtype=np.float32))),
        )
        return False

    def _compose_effective_action(self, action: np.ndarray) -> np.ndarray:
        """Blend residual policy output with an optional guidance prior."""
        action = validate_policy_action(action)
        if self.guidance_assist_enabled:
            guidance = self._guidance_action()
            effective = np.array(
                [
                    guidance[0] + action[0] * self.guidance_residual_steering_scale,
                    guidance[1] + action[1] * self.guidance_residual_throttle_scale,
                ],
                dtype=np.float32,
            )
        else:
            effective = action

        effective = np.clip(effective, -1.0, 1.0).astype(np.float32, copy=False)
        return self._apply_steering_dynamics(effective)

    def _apply_steering_dynamics(self, action: np.ndarray) -> np.ndarray:
        """Apply vessel-like steering dynamics to reduce left-right snap commands."""
        action = validate_policy_action(action, action_name="effective action")
        target_steering = float(action[0])
        self.last_steering_target = target_steering
        self.last_steering_rate_limited = False

        if not self.steering_dynamics_enabled:
            return action

        previous_steering = float(self.performed_action[0])
        alpha = float(np.clip(self.steering_smoothing_alpha, 0.0, 1.0))
        smoothed_steering = previous_steering + alpha * (target_steering - previous_steering)

        max_step_delta = float(self.max_steering_rate_per_second) * float(self.mdp_step_seconds)
        if max_step_delta > 0.0:
            limited_steering = float(
                np.clip(
                    smoothed_steering,
                    previous_steering - max_step_delta,
                    previous_steering + max_step_delta,
                )
            )
            self.last_steering_rate_limited = abs(limited_steering - smoothed_steering) > 1e-6
            smoothed_steering = limited_steering

        filtered = action.copy()
        filtered[0] = float(np.clip(smoothed_steering, -1.0, 1.0))
        return filtered.astype(np.float32, copy=False)

    def _guidance_action(self) -> np.ndarray:
        """Build a conservative route-following prior in policy-action space."""
        heading_error = self._current_heading_error()
        cross_track_term = float(np.clip(self.cross_error / _CROSS_TRACK_LIMIT_M, -1.0, 1.0))
        steering = np.clip((heading_error / math.pi) * 0.95 + cross_track_term * 0.35, -1.0, 1.0)

        throttle_01 = 0.55 + 0.30 * max(0.0, math.cos(heading_error))
        throttle_01 *= max(0.25, 1.0 - 0.55 * min(abs(cross_track_term), 1.0))
        throttle_signed = np.clip(2.0 * throttle_01 - 1.0, -1.0, 1.0)
        return np.array([steering, throttle_signed], dtype=np.float32)

    def _distance_to_target(self) -> float:
        """Return current Euclidean distance to the route target."""
        return float(np.linalg.norm(self.target_pos - self.ship_pos))

    def _distance_to_checkpoint(self, checkpoint_index: int) -> float:
        """Return Euclidean distance to a checkpoint by index."""
        checkpoint = self.checkpoints[min(max(checkpoint_index, 0), len(self.checkpoints) - 1)]
        return float(np.linalg.norm(checkpoint["pos"] - self.ship_pos))

    def _update_checkpoint_progress(self) -> float:
        """Advance the current checkpoint index from route progress."""
        reward = 0.0
        while self.checkpoint_index < len(self.checkpoints) - 1:
            if abs(self.cross_error) > _CHECKPOINT_PASS_MAX_CROSS_TRACK_M:
                break
            if abs(self.heading_error) > _CHECKPOINT_PASS_MAX_HEADING_RAD:
                break

            checkpoint = self.checkpoints[self.checkpoint_index]
            threshold = checkpoint["distance_m"] - checkpoint["radius"]
            if self._route_progress_m < threshold:
                break
            passed_checkpoint_index = int(self.checkpoint_index)
            self.checkpoint_index += 1
            checkpoint_reward = float(checkpoint.get("reward", _CHECKPOINT_REWARD))
            reward += checkpoint_reward
            self._record_run_event(
                "checkpoint_passed",
                checkpoint_index=passed_checkpoint_index,
                reward=checkpoint_reward,
                ship_x=float(self.ship_pos[0]),
                ship_y=float(self.ship_pos[1]),
                heading_deg=float(math.degrees(self.ship_heading)),
            )
            # Avoid reward bursts when progress jumps across many checkpoints in one RL step.
            break
        return reward

    def _has_reached_target(self) -> bool:
        """Return whether the ship has reached the route target."""
        if abs(self.cross_error) > _SUCCESS_MAX_CROSS_TRACK_M:
            return False
        if abs(self.heading_error) > _SUCCESS_MAX_HEADING_RAD:
            return False
        return (
            self._distance_to_target() <= self.checkpoints[-1]["radius"]
            or self._route_progress_m >= self.route_length_m - self.checkpoints[-1]["radius"]
        )

    def _distance_to_next_checkpoint(self) -> float:
        """Return Euclidean distance to the next active checkpoint."""
        next_idx = min(self.checkpoint_index + 1, len(self.checkpoints) - 1)
        next_pos = self.checkpoints[next_idx]["pos"]
        return float(np.linalg.norm(next_pos - self.ship_pos))

    def _effective_offtrack_limit_m(self) -> float:
        """Return the active off-track threshold for the current episode step.

        The threshold follows a two-part schedule:
            1) A curriculum anneal from a generous limit to a stricter limit
               across early episodes.
            2) A next-checkpoint-aware expansion so the threshold does not
               collapse too aggressively in wide turns.
        """
        anneal_ratio = float(
            np.clip(
                (self._episode_index - 1) / max(_OFFTRACK_LIMIT_DECAY_EPISODES, 1),
                0.0,
                1.0,
            )
        )
        scheduled_limit = float(
            _HARD_OFFTRACK_CTE_LIMIT_M
            - (_HARD_OFFTRACK_CTE_LIMIT_M - _SOFT_OFFTRACK_CTE_LIMIT_M) * anneal_ratio
        )
        checkpoint_aware_limit = float(
            np.clip(
                _OFFTRACK_NEXT_CHECKPOINT_FACTOR * self._distance_to_next_checkpoint(),
                _MIN_ADAPTIVE_OFFTRACK_M,
                _HARD_OFFTRACK_CTE_LIMIT_M,
            )
        )
        return float(
            np.clip(
                max(scheduled_limit, checkpoint_aware_limit),
                _SOFT_OFFTRACK_CTE_LIMIT_M,
                _HARD_OFFTRACK_CTE_LIMIT_M,
            )
        )

    def _detect_collision(self) -> bool | None:
        """Return whether AirSim reported a collision this step."""
        try:
            collision = self.bridge.get_collision_info()
        except _AIRSIM_READ_EXCEPTIONS:
            logger.warning("AirSimEnv could not read collision info for the current step.", exc_info=True)
            self._last_collision = False
            return None
        self._last_collision = bool(getattr(collision, "has_collided", False))
        return self._last_collision

    def _build_reward_state(self) -> dict[str, float | int]:
        """Capture the run-log state schema used by the strict reward."""
        return {
            "route_progress_m": float(self._route_progress_m),
            "cross_track_error": float(self.cross_error),
            "ship_x": float(self.ship_pos[0]),
            "ship_y": float(self.ship_pos[1]),
            "heading_deg": float(math.degrees(self.ship_heading)),
            "speed": float(self.ship_speed),
            "checkpoint_index": int(self.checkpoint_index),
        }

    def calculate_reward(
        self,
        current_state: dict[str, float | int],
        previous_state: dict[str, float | int],
        checkpoint_reward: float = 0.0,
    ) -> tuple[float, bool]:
        """Return a stable dense reward and whether this step is off-track."""
        current_cte = abs(float(current_state.get("cross_track_error", 0.0)))
        previous_cte = abs(float(previous_state.get("cross_track_error", current_cte)))
        effective_limit = self._effective_offtrack_limit_m()
        enforce_offtrack = self.step_count > _EARLY_FAILURE_GRACE_STEPS
        offtrack = bool(enforce_offtrack and current_cte > effective_limit)

        current_progress = float(current_state.get("route_progress_m", 0.0))
        previous_progress = float(previous_state.get("route_progress_m", current_progress))
        progress_delta = float(current_progress - previous_progress)
        progress_reward = float(
            np.clip(progress_delta / max(_STRICT_PROGRESS_SCALE_M, 1e-6), -0.25, 1.0)
        )
        checkpoint_bonus = float(
            np.clip(checkpoint_reward / max(_CHECKPOINT_REWARD, 1e-6), 0.0, 1.0)
        )

        track_reward = float(np.exp(-current_cte / max(_STRICT_TRACK_DECAY_M, 1e-6)))
        heading_ratio = float(np.clip(abs(self.heading_error) / math.pi, 0.0, 1.0))
        heading_alignment = max(0.0, math.cos(self.heading_error))
        speed_ratio = float(np.clip(self.ship_speed / _SPEED_SCALE_MPS, 0.0, 1.0))
        cte_ratio = float(np.clip(current_cte / max(effective_limit, 1e-6), 0.0, 1.5))
        previous_cte_ratio = float(
            np.clip(previous_cte / max(effective_limit, 1e-6), 0.0, 1.5)
        )
        cte_growth = max(0.0, cte_ratio - previous_cte_ratio)

        current_steering = float(self.last_policy_action[0])
        previous_steering = (
            float(self.previous_action[0])
            if hasattr(self, "previous_action")
            else current_steering
        )
        steering_delta = abs(current_steering - previous_steering)
        steering_penalty = float(
            _STEERING_MAGNITUDE_PENALTY_WEIGHT * abs(current_steering) +
            _STEERING_RATE_PENALTY_WEIGHT * steering_delta
        )
        yaw_penalty = 0.05 * float(
            np.clip(abs(self.ship_yaw_rate) / max(_YAW_RATE_SCALE_RADPS, 1e-6), 0.0, 1.0)
        )

        precision_bonus = 0.0
        if current_cte < _PRECISION_BONUS_THRESHOLD_M:
            precision_bonus = _PRECISION_BONUS * max(0.0, math.cos(self.heading_error))

        dense_reward = (
            0.30
            + 0.75 * track_reward
            + 0.70 * progress_reward
            + 0.90 * checkpoint_bonus
            + 0.35 * heading_alignment * speed_ratio
            + precision_bonus
            - 0.25 * cte_ratio
            - 0.20 * cte_growth
            - 0.20 * heading_ratio
            - steering_penalty
            - yaw_penalty
        )

        if offtrack:
            dense_reward += _OFFTRACK_PENALTY

        self.last_checkpoint_reward_effective = float(
            checkpoint_reward * max(0.0, 1.0 - min(cte_ratio, 1.0))
        )
        return float(dense_reward), offtrack


    def _normalize_reward(self, reward: float) -> float:
        """Scale route rewards into the PPO-friendly ``[-1, 1]`` interval."""
        if not self.normalize_reward:
            return float(reward)
        normalized = float(reward) / self.reward_normalization_scale
        return float(np.clip(normalized, -1.0, 1.0))

    def _map_policy_action_to_controls(self, action: np.ndarray) -> Any:
        """Map the policy action to the active UE5 control interface."""
        try:
            mapper = _CONTROL_MAPPERS[self.control_interface]
        except KeyError as exc:
            raise ValueError(
                "Unsupported AirSim control interface "
                f"'{self.control_interface}'."
            ) from exc
        return mapper(self.bridge.airsim, action)

    def _build_observation(self) -> np.ndarray:
        """Build the PPO observation vector from the current route state.

        The current car-physics training profile keeps the observation
        route-relative while also exposing the minimum extra dynamics needed
        for tighter turn control: yaw-rate and cross-track trend.
        """
        builder_name = _OBSERVATION_BUILDERS.get(
            (self.control_interface, self.observation_profile)
        )
        if builder_name is None:
            raise ValueError(
                "Unsupported AirSim observation profile "
                f"'{self.observation_profile}' for control interface "
                f"'{self.control_interface}'."
            )
        builder = getattr(self, builder_name)
        observation = builder()
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def _build_route_relative_car_observation(self) -> np.ndarray:
        """Build the current route-relative car-physics observation vector."""
        current_checkpoint = self.checkpoints[min(self.checkpoint_index, len(self.checkpoints) - 1)]["pos"]
        next_index = min(self.checkpoint_index + 1, len(self.checkpoints) - 1)
        next_checkpoint = self.checkpoints[next_index]["pos"]

        current_local = self._world_to_ship_frame(current_checkpoint)
        next_local = self._world_to_ship_frame(next_checkpoint)
        target_local = self._world_to_ship_frame(self.target_pos)

        lookahead_m = max(self._distance_scale_m * _LOS_LOOKAHEAD_FRACTION, 20.0)
        los_correction = math.atan2(-float(self.cross_error), lookahead_m)
        los_heading_error = _normalize_angle(float(self.heading_error) + los_correction)

        observation = np.array(
            [
                _log_scale_signed(float(current_local[0]), self._distance_scale_m),
                _log_scale_signed(float(current_local[1]), self._distance_scale_m),
                _log_scale_signed(float(next_local[0]), self._distance_scale_m),
                _log_scale_signed(float(next_local[1]), self._distance_scale_m),
                _log_scale_signed(float(target_local[0]), self._distance_scale_m),
                _log_scale_signed(float(target_local[1]), self._distance_scale_m),
                math.cos(self.heading_error),
                math.sin(self.heading_error),
                float(np.clip(self.cross_error / _CROSS_TRACK_LIMIT_M, -1.0, 1.0)),
                float(
                    np.clip(
                        self.cross_track_error_delta / _CROSS_TRACK_DELTA_SCALE_M,
                        -1.0,
                        1.0,
                    )
                ),
                float(np.clip(self.ship_speed / _SPEED_SCALE_MPS, 0.0, 1.0)),
                float(np.clip(self.ship_yaw_rate / _YAW_RATE_SCALE_RADPS, -1.0, 1.0)),
                float(self.performed_action[0]),
                float(self.performed_action[1]),
                float(np.clip(self._route_progress_ratio, 0.0, 1.0)),
                float(np.clip(self.checkpoint_index / max(len(self.checkpoints) - 1, 1), 0.0, 1.0)),
                float(self.previous_action[0]),
                float(np.clip(los_heading_error / math.pi, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return observation

    def _build_observation_space(self) -> gym.spaces.Box:
        """Define the explicit continuous PPO observation space."""
        try:
            fields = _OBSERVATION_FIELDS_BY_PROFILE[self.observation_profile]
        except KeyError as exc:
            raise ValueError(
                "Unsupported AirSim observation profile "
                f"'{self.observation_profile}'."
            ) from exc
        high = np.ones(len(fields), dtype=np.float32)
        return gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def _world_to_ship_frame(self, point: np.ndarray) -> np.ndarray:
        """Express a world-frame point in the ship-local frame."""
        delta = np.asarray(point, dtype=np.float32) - self.ship_pos
        cos_h = math.cos(self.ship_heading)
        sin_h = math.sin(self.ship_heading)
        forward = cos_h * float(delta[0]) + sin_h * float(delta[1])
        lateral = -sin_h * float(delta[0]) + cos_h * float(delta[1])
        return np.array([forward, lateral], dtype=np.float32)

    def _current_heading_error(self) -> float:
        """Compute heading error against the current route segment direction."""
        route_heading = self._route_heading_at_index(self._route_projection_segment)
        return _normalize_angle(route_heading - self.ship_heading)

    def _route_heading_at_index(self, index: int) -> float:
        """Return the route tangent heading at a route-point index."""
        idx = min(max(int(index), 0), len(self.route_points) - 2)
        segment = self.route_points[idx + 1] - self.route_points[idx]
        if float(np.linalg.norm(segment)) <= 1e-9:
            return 0.0
        return float(math.atan2(float(segment[1]), float(segment[0])))

    def _project_onto_route(self, ship_pos: np.ndarray) -> dict[str, float | int]:
        """Project a ship position onto the route polyline."""
        if len(self.route_points) < 2:
            return {"progress_m": 0.0, "cross_track_error": 0.0, "segment_index": 0}

        best_distance = float("inf")
        best_progress = 0.0
        best_cross_track = 0.0
        best_segment_index = 0

        for index in range(len(self.route_points) - 1):
            start = self.route_points[index]
            end = self.route_points[index + 1]
            segment = end - start
            length_sq = float(np.dot(segment, segment))
            if length_sq <= 1e-9:
                continue

            relative = np.asarray(ship_pos, dtype=np.float32) - start
            t = float(np.clip(np.dot(relative, segment) / length_sq, 0.0, 1.0))
            projection = start + t * segment
            delta = np.asarray(ship_pos, dtype=np.float32) - projection
            distance = float(np.linalg.norm(delta))
            signed_area = float(segment[0] * relative[1] - segment[1] * relative[0])
            segment_length = math.sqrt(length_sq)
            lateral_distance = signed_area / max(segment_length, 1e-9)
            if abs(lateral_distance) <= distance + 1e-6:
                signed_distance = lateral_distance
            elif abs(lateral_distance) > 1e-9:
                signed_distance = math.copysign(distance, lateral_distance)
            else:
                signed_distance = 0.0
            progress = float(self.route_cumulative_distances[index] + t * math.sqrt(length_sq))

            if distance < best_distance:
                best_distance = distance
                best_progress = progress
                best_cross_track = signed_distance
                best_segment_index = index

        return {
            "progress_m": best_progress,
            "cross_track_error": best_cross_track,
            "segment_index": best_segment_index,
        }

    def _build_info(
        self,
        *,
        reward: float,
        raw_reward: float,
        terminated: bool,
        truncated: bool,
        outcome: str,
        all_checkpoints_passed: bool | None = None,
    ) -> dict[str, Any]:
        """Build the step info dictionary shared by training and inference."""
        reached_target = bool(self._has_reached_target())
        if all_checkpoints_passed is None:
            all_checkpoints_passed = bool(
                self.checkpoint_index >= len(self.checkpoints) - 1
                or reached_target
            )
        return {
            "checkpoint_index": int(self.checkpoint_index),
            "route_total_checkpoints": int(max(len(self.checkpoints) - 1, 1)),
            "cross_track_error": float(self.cross_error),
            "cross_track_error_delta": float(self.cross_track_error_delta),
            "strict_cte_limit_m": float(_STRICT_CTE_LIMIT_M),
            "hard_offtrack_cte_limit_m": float(_HARD_OFFTRACK_CTE_LIMIT_M),
            "effective_offtrack_limit_m": float(self._effective_offtrack_limit_m()),
            "strict_cte_violation": bool(abs(self.cross_error) > _STRICT_CTE_LIMIT_M),
            "hard_offtrack_violation": bool(abs(self.cross_error) > _HARD_OFFTRACK_CTE_LIMIT_M),
            "episode_max_abs_cross_track_error": float(self._episode_max_abs_cross_track_error),
            "all_checkpoints_passed": bool(all_checkpoints_passed),
            "heading_error": float(self.heading_error),
            "route_progress_m": float(self._route_progress_m),
            "route_progress_ratio": float(self._route_progress_ratio),
            "target_distance": float(self._distance_to_target()),
            "speed": float(self.ship_speed),
            "yaw_rate": float(self.ship_yaw_rate),
            "steering_target": float(self.last_steering_target),
            "steering_rate_limited": bool(self.last_steering_rate_limited),
            "checkpoint_reward_raw": float(self.last_checkpoint_reward_raw),
            "checkpoint_reward_effective": float(self.last_checkpoint_reward_effective),
            "episode_mean_abs_cross_track_error": float(
                self._episode_abs_cross_track_sum / max(self._episode_quality_samples, 1)
            ),
            "mdp_step_seconds": float(self.mdp_step_seconds),
            "physics_wait_s": float(self.last_physics_wait_s),
            "step_latency_s": float(self.last_step_latency_s),
            "collision": bool(self._last_collision),
            "pause_timeout": bool(self._last_pause_timeout),
            "state_update_reliable": bool(self._state_update_reliable),
            "repeated_state_count": int(self._repeated_state_count),
            "guidance_assist_enabled": bool(self.guidance_assist_enabled),
            "steering_dynamics_enabled": bool(self.steering_dynamics_enabled),
            "steering_smoothing_alpha": float(self.steering_smoothing_alpha),
            "max_steering_rate_per_second": float(self.max_steering_rate_per_second),
            "control_interface": str(self.control_interface),
            "observation_profile": str(self.observation_profile),
            "outcome": str(outcome),
            "reward": float(reward),
            "raw_reward": float(raw_reward),
            "episode_average_reward": float(
                self._episode_reward_sum / max(self._episode_reward_steps, 1)
            ),
            "strict_success": bool(
                outcome == "success"
                and self._episode_max_abs_cross_track_error <= _STRICT_CTE_LIMIT_M
                and bool(all_checkpoints_passed)
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def mark_explicit_reset_context(self, enabled: bool = True) -> None:
        """Allow the next explicit reset to emit a respawn event."""
        self._log_next_reset_event = bool(enabled)

    def drain_run_events(self) -> list[dict[str, Any]]:
        """Return and clear pending training run events."""
        events = list(self._pending_run_events)
        self._pending_run_events.clear()
        return events

    def get_training_export_snapshot(self) -> dict[str, Any]:
        """Return a serializable route snapshot for training exports."""
        return {
            "route_points": self.route_points.astype(float).tolist(),
            "checkpoints": [
                {
                    "index": int(index),
                    "x": float(checkpoint["pos"][0]),
                    "y": float(checkpoint["pos"][1]),
                    "radius": float(checkpoint["radius"]),
                    "reward": float(checkpoint["reward"]),
                    "distance_m": float(checkpoint["distance_m"]),
                    "progress_ratio": float(checkpoint["progress_ratio"]),
                }
                for index, checkpoint in enumerate(self.checkpoints)
            ],
            "overall": self.overall.astype(float).tolist(),
            "obstacles": self.obstacles.astype(float).tolist(),
            "initial_ship_pos": self.initial_ship_pos.astype(float).tolist(),
            "target_pos": self.target_pos.astype(float).tolist(),
            "captured_startup_pose_airsim": self._serialize_run_event_value(
                self._captured_startup_pose_airsim
            ),
            "captured_startup_framework_heading_deg": (
                None
                if self._captured_startup_framework_heading is None
                else float(math.degrees(self._captured_startup_framework_heading))
            ),
            "save_slot": self.save_slot,
            "load_world_on_start": bool(self.load_world_on_start),
            "capture_live_start_pose": bool(self.capture_live_start_pose),
            "no_rendering_mode": bool(self.no_rendering_mode),
            "simulation_speed": float(self.simulation_speed),
            "reset_mode": str(self.reset_mode),
        }

    def _record_run_event(self, event_type: str, **fields: Any) -> None:
        """Queue one structured training event for PPO-side export."""
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": str(event_type),
            "env_pid": int(os.getpid()),
            "episode": int(self._episode_index),
            "episode_step": int(self.step_count),
            "checkpoint_index": int(self.checkpoint_index),
            "route_progress_m": float(self._route_progress_m),
            "route_progress_ratio": float(self._route_progress_ratio),
            "cross_track_error": float(self.cross_error),
            "ship_x": float(self.ship_pos[0]),
            "ship_y": float(self.ship_pos[1]),
            "heading_deg": float(math.degrees(self.ship_heading)),
            "speed": float(self.ship_speed),
        }
        event.update(fields)
        self._pending_run_events.append(self._serialize_run_event_value(event))

    def _serialize_run_event_value(self, value: Any) -> Any:
        """Convert nested event values into JSON-safe structures."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {
                str(key): self._serialize_run_event_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._serialize_run_event_value(item) for item in value]
        return str(value)

    def _initial_curriculum_frontier(self, target_checkpoint: int | None) -> int:
        """Resolve the initial curriculum frontier checkpoint."""
        if not self.curriculum_enabled:
            return len(self.checkpoints) - 2
        if target_checkpoint is not None:
            return min(max(int(target_checkpoint), 1), len(self.checkpoints) - 2)
        return min(self.curriculum_window_size, len(self.checkpoints) - 2)

    def _select_episode_start(self) -> tuple[int, np.ndarray]:
        """Choose the episode start checkpoint and framework position."""
        start_index = 0
        if self.curriculum_enabled and self.training_mode and len(self.checkpoints) > 2:
            window_end = min(self._curriculum_frontier_index, len(self.checkpoints) - 2)
            window_start = max(0, window_end - self.curriculum_window_size + 1)
            start_index = int(self._rng.integers(window_start, window_end + 1))

        start_pos = self.checkpoints[start_index]["pos"].copy()
        if self.curriculum_randomization_scale > 0.0:
            start_pos = start_pos + self._rng.normal(
                loc=0.0,
                scale=self.curriculum_randomization_scale,
                size=2,
            ).astype(np.float32)
        return start_index, start_pos.astype(np.float32)

    def _select_episode_start_heading(self, start_index: int) -> float:
        """Choose the reset heading for the selected episode start."""
        if (
            int(start_index) == 0
            and self._captured_startup_framework_heading is not None
            and self.curriculum_randomization_scale <= 0.0
        ):
            return float(self._captured_startup_framework_heading)
        return self._route_heading_at_index(start_index)

    def _update_curriculum_after_episode(self, success: bool) -> None:
        """Expand the curriculum frontier after repeated success."""
        if not self.curriculum_enabled:
            return
        if success:
            self._curriculum_success_streak += 1
        else:
            self._curriculum_success_streak = 0

        if self._curriculum_success_streak < self.curriculum_success_patience:
            return

        self._curriculum_success_streak = 0
        self._curriculum_frontier_index = min(
            len(self.checkpoints) - 2,
            self._curriculum_frontier_index + self.curriculum_frontier_increment,
        )
        self.curriculum_window_size = min(
            len(self.checkpoints) - 1,
            self.curriculum_window_size + self.curriculum_window_increment,
        )

    def _airsim_yaw_rate_to_framework(self, yaw_rate: float) -> float:
        """Invert the yaw-rate sign convention used by the coordinate transform."""
        sign = float(self.coord_transform.framework_yaw_rate_to_airsim(1.0))
        return float(yaw_rate * sign)

    def _build_route_points(self) -> np.ndarray:
        """Load, trim, and densify the framework route used by AirSimEnv."""
        raw_path = self._load_route_points(target_pos=self._requested_target_pos)
        if len(raw_path) == 0:
            raw_path = np.array(
                [self.initial_ship_pos, self.target_pos],
                dtype=np.float32,
            )

        start_index = int(
            np.argmin(np.linalg.norm(raw_path - self.initial_ship_pos.reshape(1, 2), axis=1))
        )
        trimmed = raw_path[start_index:].astype(np.float32)
        if len(trimmed) == 0:
            trimmed = raw_path.astype(np.float32)

        if float(np.linalg.norm(trimmed[0] - self.initial_ship_pos)) > 1e-3:
            trimmed = np.vstack([self.initial_ship_pos, trimmed]).astype(np.float32)

        if self._requested_target_pos is not None and float(np.linalg.norm(trimmed[-1] - self._requested_target_pos)) > 1e-3:
            trimmed = np.vstack([trimmed, self._requested_target_pos]).astype(np.float32)

        dense = np.asarray(
            create_checkpoints_from_simple_path(trimmed, _ROUTE_POINT_SPACING_M),
            dtype=np.float32,
        )
        if len(dense) < 2:
            dense = np.vstack([self.initial_ship_pos, self.target_pos]).astype(np.float32)
        return dense

    def _load_route_points(self, target_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """Load the route CSV used for AirSim training and inference."""
        fallback_target = (
            np.asarray(target_pos, dtype=np.float32)
            if target_pos is not None
            else np.asarray(DEFAULT_FRAMEWORK_TARGET_POS, dtype=np.float32)
        )
        fallback = np.vstack([self.initial_ship_pos, fallback_target]).astype(np.float32)
        return _safe_load_csv(_DEFAULT_TRAJECTORY_CSV, fallback=fallback)

    def _load_outline_points(self, path: Path) -> np.ndarray:
        """Load an outline CSV used by framework plotting helpers."""
        fallback = np.array(
            [
                [self.initial_ship_pos[0] - 1000.0, self.initial_ship_pos[1] - 1000.0],
                [self.initial_ship_pos[0] + 1000.0, self.initial_ship_pos[1] - 1000.0],
                [self.initial_ship_pos[0] + 1000.0, self.initial_ship_pos[1] + 1000.0],
                [self.initial_ship_pos[0] - 1000.0, self.initial_ship_pos[1] + 1000.0],
            ],
            dtype=np.float32,
        )
        return _safe_load_csv(path, fallback=fallback)

    @staticmethod
    def _compute_cumulative_distances(points: np.ndarray) -> np.ndarray:
        """Compute cumulative route distances for a polyline."""
        if len(points) == 0:
            return np.array([0.0], dtype=np.float32)
        if len(points) == 1:
            return np.array([0.0], dtype=np.float32)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(segment_lengths)]).astype(np.float32)

    def _build_checkpoints(
        self,
        route_points: np.ndarray,
        cumulative_distances: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Build checkpoint metadata for route following and plotting."""
        checkpoints: list[dict[str, Any]] = []
        total_points = max(len(route_points) - 1, 1)
        for index, point in enumerate(route_points):
            checkpoints.append(
                {
                    "pos": np.asarray(point, dtype=np.float32),
                    "radius": float(_TARGET_RADIUS_M if index == len(route_points) - 1 else _CHECKPOINT_RADIUS_M),
                    "reward": float(_CHECKPOINT_REWARD if index < len(route_points) - 1 else 0.0),
                    "distance_m": float(cumulative_distances[index]),
                    "progress_ratio": float(index / total_points),
                }
            )

        for checkpoint, line in zip(checkpoints, _calculate_perpendicular_lines(checkpoints)):
            checkpoint["perpendicular_line"] = line

        return checkpoints
