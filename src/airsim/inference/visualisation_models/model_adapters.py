"""Adapter registry for framework-model visualisation in UE5.

The UE5 replay loop is intentionally model-family agnostic. Framework-specific
details such as policy loading, environment construction, state extraction, and
control animation live behind these adapters so new model families can be added
without rewriting the runner.
"""

from __future__ import annotations

import time
from abc import ABC
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from airsim.communication.airsim_bridge import AirSimBridge
from airsim.communication.control_mapping import (
    map_fossen_actuator_state,
    map_fossen_steering_visual_state,
    map_fossen_visual_actuator_state,
)
from fw.common.airsim.tools.angles import interpolate_angle, wrapped_angle_delta
from fw.common.airsim.tools.path_utils import resolve_model_path
from fw.environments.fh_sim_env import FhSimEnv
from fw.environments.py_sim_env import PySimEnv
from fw.policies.ppo_policy import PPOPolicy

_DEFAULT_SHIP_POS = [2245, -47]
_DEFAULT_TARGET_POS = [-1559, 11990]
_SRC_DIR = Path(__file__).resolve().parents[3]


def _interpolate_actuator_state(
    previous_state: Optional[np.ndarray],
    target_state: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linearly interpolate between two actuator-state vectors."""
    if previous_state is None:
        return target_state
    return previous_state + alpha * (target_state - previous_state)


def load_policy(model_path: str, device: str = "cpu") -> PPOPolicy:
    """Load a PPO policy from a saved framework model."""
    path = resolve_model_path(model_path, _SRC_DIR)
    stem = str(path.with_suffix("")) if path.suffix == ".zip" else str(path)
    policy = PPOPolicy.load(stem, device=device)
    policy.network.eval()
    return policy


def _framework_pose_from_env(env, bridge: AirSimBridge) -> tuple[float, float, float]:
    """Convert the current framework ship pose into AirSim coordinates."""
    coord = bridge.coord_transform
    if coord is None:
        raise RuntimeError("Bridge has no active coordinate transform.")

    ship_pos = env.ship_pos[0]
    heading = float(env.phys_sims[0].state.heading)
    airsim_x, airsim_y, _ = coord.fossen_to_airsim(
        float(ship_pos[0]),
        float(ship_pos[1]),
        bridge.render_z,
    )
    airsim_yaw = coord.fossen_heading_to_airsim_yaw(heading)
    return airsim_x, airsim_y, airsim_yaw


def _framework_motion_state_from_env(env, bridge: AirSimBridge) -> np.ndarray:
    """Extract AirSim-frame pose and velocities from the framework state."""
    x, y, yaw = _framework_pose_from_env(env, bridge)
    state = env.phys_sims[0].state

    fossen_vx = float(state.u) * np.cos(float(state.heading)) - float(state.v) * np.sin(float(state.heading))
    fossen_vy = float(state.u) * np.sin(float(state.heading)) + float(state.v) * np.cos(float(state.heading))
    airsim_vx, airsim_vy = bridge.coord_transform.fossen_velocity_to_airsim(fossen_vx, fossen_vy)
    airsim_yaw_rate = bridge.coord_transform.fossen_yaw_rate_to_airsim(float(state.r))

    return np.array([x, y, yaw, airsim_vx, airsim_vy, airsim_yaw_rate], dtype=np.float64)


def _resolve_interpolation_steps(
    previous_motion_state: np.ndarray,
    target_motion_state: np.ndarray,
    interpolation_steps: int,
    max_translation_step: float,
    max_yaw_step_rad: float,
) -> int:
    """Compute how many UE5 interpolation steps are needed for a motion update."""
    steps = max(1, int(interpolation_steps))
    delta_x = float(target_motion_state[0] - previous_motion_state[0])
    delta_y = float(target_motion_state[1] - previous_motion_state[1])
    translation = float(np.hypot(delta_x, delta_y))
    yaw_delta = abs(wrapped_angle_delta(float(previous_motion_state[2]), float(target_motion_state[2])))

    if max_translation_step > 0.0:
        steps = max(steps, int(np.ceil(translation / max_translation_step)))
    if max_yaw_step_rad > 0.0:
        steps = max(steps, int(np.ceil(yaw_delta / max_yaw_step_rad)))
    return max(1, steps)


def _should_snap_visual_update(total_duration: float, min_render_hz: float) -> bool:
    """Return whether a visual update should skip time-based interpolation."""
    total_duration = max(float(total_duration), 0.0)
    min_step_interval = 1.0 / float(min_render_hz) if min_render_hz > 0.0 else 0.0
    snap_threshold = max(1e-3, min_step_interval)
    return total_duration <= snap_threshold


def _animate_framework_state(
    env,
    bridge: AirSimBridge,
    target_motion_state: np.ndarray,
    previous_motion_state: Optional[np.ndarray],
    total_duration: float,
    interpolation_steps: int,
    mirror_controls_in_ue5: bool = False,
    animate_steering_in_ue5: bool = True,
    animate_throttle_in_ue5: bool = False,
    previous_actuator_state: Optional[np.ndarray] = None,
    steering_visual_gain: float = 0.35,
    throttle_visual_gain: float = 1.0,
    use_kinematics_in_ue5: bool = False,
    min_render_hz: float = 90.0,
    max_translation_step: float = 0.15,
    max_yaw_step_rad: float = np.deg2rad(1.5),
    lock_path_in_ue5: bool = False,
    vehicle_object_name: str = "",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Mirror one framework motion update into UE5 with optional smoothing."""
    if previous_motion_state is None:
        previous_motion_state = target_motion_state

    total_duration = max(total_duration, 0.0)

    target_actuator_state = None
    if getattr(env, "performed_action", None):
        target_actuator_state = np.asarray(env.performed_action[0], dtype=np.float32)

    start_x, start_y, start_yaw, start_vx, start_vy, start_yaw_rate = previous_motion_state
    target_x, target_y, target_yaw, target_vx, target_vy, target_yaw_rate = target_motion_state
    _prefetched_pose = None

    def _send_pose(alpha: float) -> None:
        """Push an interpolated motion state to UE5."""
        ix = start_x + alpha * (target_x - start_x)
        iy = start_y + alpha * (target_y - start_y)
        iyaw = interpolate_angle(start_yaw, target_yaw, alpha)
        if vehicle_object_name:
            bridge.set_object_pose(vehicle_object_name, ix, iy, bridge.render_z, iyaw)
        elif use_kinematics_in_ue5:
            ivx = start_vx + alpha * (target_vx - start_vx)
            ivy = start_vy + alpha * (target_vy - start_vy)
            iyr = start_yaw_rate + alpha * (target_yaw_rate - start_yaw_rate)
            bridge.set_kinematics(
                ix, iy, bridge.render_z, iyaw,
                linear_velocity_x=ivx, linear_velocity_y=ivy, linear_velocity_z=0.0,
                angular_velocity_x=0.0, angular_velocity_y=0.0, angular_velocity_z=iyr,
                preserve_current_z=True, preserve_roll_pitch=True,
            )
        else:
            bridge.set_pose(
                ix, iy, bridge.render_z, iyaw,
                preserve_current_z=True, preserve_roll_pitch=True,
                _current_pose=_prefetched_pose,
            )

    def _send_controls(alpha: float) -> None:
        """Send interpolated control animation for the current frame."""
        if vehicle_object_name or target_actuator_state is None:
            return
        if not (mirror_controls_in_ue5 or animate_steering_in_ue5 or animate_throttle_in_ue5):
            return
        interp_act = _interpolate_actuator_state(previous_actuator_state, target_actuator_state, alpha)
        if mirror_controls_in_ue5:
            controls = map_fossen_actuator_state(
                bridge._airsim,
                env.ships[0].specifications,
                rudder_angle=float(interp_act[0]),
                thrust=float(interp_act[1]),
                steering_gain=steering_visual_gain,
            )
        elif animate_throttle_in_ue5:
            controls = map_fossen_visual_actuator_state(
                bridge._airsim,
                env.ships[0].specifications,
                rudder_angle=float(interp_act[0]),
                thrust=float(interp_act[1]),
                steering_gain=steering_visual_gain,
                throttle_gain=throttle_visual_gain,
            )
        else:
            controls = map_fossen_steering_visual_state(
                bridge._airsim,
                env.ships[0].specifications,
                rudder_angle=float(interp_act[0]),
                steering_gain=steering_visual_gain,
            )
        bridge.set_controls(controls)

    interpolation_steps = _resolve_interpolation_steps(
        previous_motion_state=previous_motion_state,
        target_motion_state=target_motion_state,
        interpolation_steps=interpolation_steps,
        max_translation_step=max_translation_step,
        max_yaw_step_rad=max_yaw_step_rad,
    )
    if lock_path_in_ue5 and min_render_hz > 0.0:
        total_duration = max(total_duration, interpolation_steps / float(min_render_hz))

    min_step_interval = 1.0 / min_render_hz if min_render_hz > 0.0 else 0.0
    if _should_snap_visual_update(total_duration, min_render_hz):
        _send_pose(1.0)
        _send_controls(1.0)
        return target_motion_state, target_actuator_state

    if not vehicle_object_name and not use_kinematics_in_ue5 and bridge.is_connected:
        try:
            _prefetched_pose = bridge.get_vehicle_pose()
        except Exception:
            pass

    start_time = time.perf_counter()
    deadline = start_time + total_duration
    _last_pose_time = start_time - min_step_interval

    while True:
        now = time.perf_counter()
        alpha = min((now - start_time) / total_duration, 1.0)
        is_final = alpha >= 1.0

        if is_final or (now - _last_pose_time >= min_step_interval):
            _send_pose(alpha)
            _last_pose_time = now

        if is_final:
            _send_controls(1.0)
            break

        remaining = deadline - time.perf_counter()
        if remaining > 0.002:
            time.sleep(0.001)

    return target_motion_state, target_actuator_state


class ModelVisualisationAdapter(ABC):
    """Base adapter for one visualised model family."""

    family: str = ""
    model_origin: str = ""
    supported_environment_names: tuple[str, ...] = ()
    filename_keywords: tuple[str, ...] = ()

    def matches_model_path(self, model_path: str) -> bool:
        """Return whether the model filename looks like this family."""
        stem = Path(model_path).stem.lower()
        return any(keyword in stem for keyword in self.filename_keywords)

    def resolve_environment_name(self, model_path: str, requested_env_name: str) -> str:
        """Resolve the framework environment name used for playback."""
        requested = str(requested_env_name).strip() or "auto"
        if requested.lower() != "auto":
            if requested not in self.supported_environment_names:
                supported = ", ".join(self.supported_environment_names)
                raise ValueError(
                    f"Model family '{self.family}' does not support environment '{requested}'. "
                    f"Supported environments: {supported}."
                )
            return requested
        if len(self.supported_environment_names) == 1:
            return self.supported_environment_names[0]
        raise ValueError(
            f"Model family '{self.family}' requires an explicit environment name."
        )

    def load_policy(self, model_path: str, device: str = "cpu") -> PPOPolicy:
        """Load a policy checkpoint for this model family."""
        return load_policy(model_path, device=device)

    def build_environment(self, environment_name: str, max_steps: int):
        """Construct the framework environment used during replay."""
        builder = self.environment_builders().get(environment_name)
        if builder is None:
            supported = ", ".join(self.environment_builders())
            raise ValueError(
                f"Unsupported environment '{environment_name}' for '{self.family}'. "
                f"Supported environments: {supported}."
            )
        return builder(max_steps)

    def environment_builders(self) -> dict[str, Callable[[int], object]]:
        """Return the environment factories supported by this adapter."""
        raise NotImplementedError

    def motion_state_from_env(self, env, bridge: AirSimBridge) -> np.ndarray:
        """Extract a generic motion state from the current framework env."""
        return _framework_motion_state_from_env(env, bridge)

    def animate_state(self, env, bridge: AirSimBridge, **kwargs) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Mirror one framework state update into UE5."""
        return _animate_framework_state(env, bridge, **kwargs)

    def cleanup_environment(self, env) -> None:
        """Dispose optional backend resources owned by the environment."""
        if env is not None and hasattr(env, "fh_sim"):
            env.fh_sim.dispose()


class PySimVisualisationAdapter(ModelVisualisationAdapter):
    """Adapter for PySimEnv framework-model visualisation."""

    family = "pysim"
    model_origin = "pysim"
    supported_environment_names = ("PySimEnv",)
    filename_keywords = ("pysimenv", "py_sim", "fossen")

    def environment_builders(self) -> dict[str, Callable[[int], object]]:
        return {
            "PySimEnv": lambda max_steps: PySimEnv(
                max_steps=max_steps,
                start_pos=_DEFAULT_SHIP_POS,
                target_pos=_DEFAULT_TARGET_POS,
            ),
        }


class FhSimVisualisationAdapter(ModelVisualisationAdapter):
    """Adapter for FHSim-backed framework models."""

    family = "fhsim"
    model_origin = "fhsim"
    supported_environment_names = ("FhSimEnv",)
    filename_keywords = ("fhsimenv", "fh_sim", "fhsim")

    def environment_builders(self) -> dict[str, Callable[[int], object]]:
        return {
            "FhSimEnv": lambda max_steps: FhSimEnv(
                max_steps=max_steps,
                start_pos=_DEFAULT_SHIP_POS,
                target_pos=_DEFAULT_TARGET_POS,
            ),
        }


_VISUALISATION_ADAPTERS: dict[str, ModelVisualisationAdapter] = {
    "pysim": PySimVisualisationAdapter(),
    "fhsim": FhSimVisualisationAdapter(),
}
_AUTO_DETECTION_ORDER = ("fhsim", "pysim")


def get_supported_model_families() -> tuple[str, ...]:
    """Return the registered model-family names."""
    return tuple(_VISUALISATION_ADAPTERS.keys())


def get_supported_environment_names() -> tuple[str, ...]:
    """Return the registered framework environment names."""
    names: set[str] = set()
    for adapter in _VISUALISATION_ADAPTERS.values():
        names.update(adapter.supported_environment_names)
    return tuple(sorted(names))


def resolve_visualisation_adapter(
    model_family: str = "auto",
    model_path: str = "",
    env_name: str = "auto",
) -> ModelVisualisationAdapter:
    """Resolve the adapter used for a visualised model replay."""
    requested_family = str(model_family).strip().lower() or "auto"
    if requested_family != "auto":
        try:
            return _VISUALISATION_ADAPTERS[requested_family]
        except KeyError as exc:
            supported = ", ".join(get_supported_model_families())
            raise ValueError(
                f"Unsupported model family '{model_family}'. Supported families: {supported}."
            ) from exc

    requested_env = str(env_name).strip()
    if requested_env and requested_env.lower() != "auto":
        for adapter in _VISUALISATION_ADAPTERS.values():
            if requested_env in adapter.supported_environment_names:
                return adapter
        supported_envs = ", ".join(get_supported_environment_names())
        raise ValueError(
            f"Unsupported environment '{env_name}'. Supported environments: {supported_envs}."
        )

    for family_name in _AUTO_DETECTION_ORDER:
        adapter = _VISUALISATION_ADAPTERS[family_name]
        if model_path and adapter.matches_model_path(model_path):
            return adapter

    return _VISUALISATION_ADAPTERS["pysim"]
