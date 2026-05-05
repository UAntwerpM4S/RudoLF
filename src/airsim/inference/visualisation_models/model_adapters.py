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
from typing import Any, Callable, Optional

import numpy as np

from airsim.communication.airsim_bridge import AirSimBridge
from airsim.communication.control_mapping import (
    map_framework_actuator_state,
    map_framework_steering_visual_state,
    map_framework_visual_actuator_state,
)
from airsim.config.airsim_config import DEFAULT_THROTTLE_VISUAL_GAIN
from fw.common.airsim.tools.angles import interpolate_angle, wrapped_angle_delta
from fw.common.airsim.tools.path_utils import resolve_model_path
from fw.common.default_positions import (
    DEFAULT_FRAMEWORK_SHIP_POS,
    DEFAULT_FRAMEWORK_TARGET_POS,
)
from fw.environments.fh_sim_env import FhSimEnv
from fw.environments.py_sim_env import PySimEnv
from fw.policies.ppo_policy import PPOPolicy

_DEFAULT_SHIP_POS = list(DEFAULT_FRAMEWORK_SHIP_POS)
_DEFAULT_TARGET_POS = list(DEFAULT_FRAMEWORK_TARGET_POS)
_SRC_DIR = Path(__file__).resolve().parents[3]
_MAX_VISUAL_UPDATE_HZ = 240.0


def _interpolate_actuator_state(
    previous_state: Optional[np.ndarray],
    target_state: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linearly interpolate between two actuator-state vectors.

    Args:
        previous_state: Previously mirrored actuator state. When ``None``,
            interpolation starts directly at ``target_state``.
        target_state: Latest framework actuator state.
        alpha: Interpolation factor in the inclusive range ``[0, 1]``.

    Returns:
        np.ndarray: Interpolated actuator vector with the same shape as the
        provided state vectors.
    """
    if previous_state is None:
        return target_state
    return previous_state + alpha * (target_state - previous_state)


def load_policy(model_path: str, device: str = "cpu") -> PPOPolicy:
    """Load a PPO policy from a saved framework model.

    Args:
        model_path: User-provided checkpoint path, relative or absolute.
        device: Torch device selector forwarded to ``PPOPolicy.load``.

    Returns:
        PPOPolicy: Loaded policy with its network placed in evaluation mode.
    """
    path = resolve_model_path(model_path, _SRC_DIR)
    stem = str(path.with_suffix("")) if path.suffix == ".zip" else str(path)
    policy = PPOPolicy.load(stem, device=device)
    policy.network.eval()
    return policy


def _extract_indexed_value(value: Any, ship_index: int = 0) -> Any:
    """Return one per-ship item while tolerating common container layouts.

    Args:
        value: Scalar, NumPy array, or list/tuple carrying framework state.
        ship_index: Preferred ship index when ``value`` stores per-ship items.

    Returns:
        Any: A single ship-specific item, the original flat value, or ``None``
        when extraction is not possible.
    """
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.ndim == 1:
            return value
        if len(value) > ship_index:
            return value[ship_index]
        return None

    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if ship_index < len(value):
            candidate = value[ship_index]
        else:
            candidate = value[0]
        if isinstance(candidate, (list, tuple, np.ndarray)) or not np.isscalar(candidate):
            return candidate
        return value

    return value


def _extract_physics_entry(env: Any, ship_index: int = 0) -> Any:
    """Return one physics-simulator entry from common env layouts.

    Args:
        env: Framework environment exposing either ``phys_sims`` or
            ``phys_sim``.
        ship_index: Preferred ship index for multi-ship environments.

    Returns:
        Any: Physics simulator entry for one ship, or ``None`` when no known
        simulator attribute is available.
    """
    for attr_name in ("phys_sim", "phys_sims"):
        candidate = _extract_indexed_value(getattr(env, attr_name, None), ship_index=ship_index)
        if candidate is not None:
            return candidate
    return None


def _build_framework_motion_state(
    adapter: "ModelVisualisationAdapter",
    env,
    bridge: AirSimBridge,
    ship_index: int = 0,
) -> np.ndarray:
    """Extract AirSim-frame pose and velocities from adapter-provided state.

    Args:
        adapter: Adapter responsible for framework-specific extraction logic.
        env: Active framework environment.
        bridge: Connected AirSim bridge exposing the coordinate transform.
        ship_index: Ship index to extract when the environment is multi-ship.

    Returns:
        np.ndarray: Motion vector ``[x, y, yaw, vx, vy, yaw_rate]`` in the
        AirSim reference frame.

    Raises:
        RuntimeError: If the AirSim bridge has no active coordinate transform.
    """
    coord = bridge.coord_transform
    if coord is None:
        raise RuntimeError("Bridge has no active coordinate transform.")

    framework_x, framework_y, framework_heading = adapter.extract_pose(env, ship_index=ship_index)
    surge, sway, yaw_rate = adapter.extract_velocity(env, ship_index=ship_index)

    airsim_x, airsim_y, _ = coord.framework_to_airsim(
        framework_x,
        framework_y,
        bridge.render_z,
    )
    airsim_yaw = coord.framework_heading_to_airsim_yaw(framework_heading)

    framework_vx = surge * np.cos(framework_heading) - sway * np.sin(framework_heading)
    framework_vy = surge * np.sin(framework_heading) + sway * np.cos(framework_heading)
    airsim_vx, airsim_vy = coord.framework_velocity_to_airsim(framework_vx, framework_vy)
    airsim_yaw_rate = coord.framework_yaw_rate_to_airsim(yaw_rate)

    return np.array([airsim_x, airsim_y, airsim_yaw, airsim_vx, airsim_vy, airsim_yaw_rate], dtype=np.float64)


def _resolve_interpolation_steps(
    previous_motion_state: np.ndarray,
    target_motion_state: np.ndarray,
    interpolation_steps: int,
    max_translation_step: float,
    max_yaw_step_rad: float,
) -> int:
    """Compute how many UE5 interpolation steps are needed for a motion update.

    Args:
        previous_motion_state: Previously mirrored AirSim-frame motion state.
        target_motion_state: Newly extracted AirSim-frame motion state.
        interpolation_steps: Minimum interpolation step count requested by the
            caller.
        max_translation_step: Maximum allowed translation per interpolation
            step in meters.
        max_yaw_step_rad: Maximum allowed yaw delta per interpolation step in
            radians.

    Returns:
        int: Interpolation step count clamped to at least ``1``.
    """
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


def _resolve_visual_update_interval(
    total_duration: float,
    interpolation_steps: int,
    min_render_hz: float,
    max_update_hz: float = _MAX_VISUAL_UPDATE_HZ,
) -> float:
    """Resolve the UE5 pose-update interval for one replay segment.

    Args:
        total_duration: Planned total interpolation duration in seconds.
        interpolation_steps: Requested interpolation step count for the motion.
        min_render_hz: Minimum desired render frequency in Hertz.
        max_update_hz: Hard cap on pose pushes to UE5 in Hertz.

    Returns:
        float: Update interval in seconds used for pose pushes.
    """
    total_duration = max(float(total_duration), 0.0)
    interpolation_steps = max(1, int(interpolation_steps))

    render_interval = 1.0 / float(min_render_hz) if min_render_hz > 0.0 else 0.0
    interpolation_interval = (
        total_duration / float(interpolation_steps) if total_duration > 0.0 else 0.0
    )

    if render_interval > 0.0 and interpolation_interval > 0.0:
        update_interval = min(render_interval, interpolation_interval)
    else:
        update_interval = max(render_interval, interpolation_interval)

    if update_interval <= 0.0:
        update_interval = 1e-3

    if max_update_hz > 0.0:
        update_interval = max(update_interval, 1.0 / float(max_update_hz))

    return float(update_interval)


def _should_snap_visual_update(
    total_duration: float,
    update_interval_s: float,
    interpolation_steps: int,
) -> bool:
    """Return whether a visual update should skip time-based interpolation.

    Args:
        total_duration: Planned total interpolation duration in seconds.
        update_interval_s: Effective pose-update interval for the segment.
        interpolation_steps: Requested interpolation step count.

    Returns:
        bool: ``True`` when interpolation should snap directly to the final
        visual state.
    """
    total_duration = max(float(total_duration), 0.0)
    update_interval_s = max(float(update_interval_s), 1e-3)
    if total_duration <= 1e-9:
        return True
    return total_duration <= update_interval_s and int(interpolation_steps) <= 1


def _build_visual_controls(
    adapter: "ModelVisualisationAdapter",
    env,
    bridge: AirSimBridge,
    actuator_state: np.ndarray,
    mirror_controls_in_ue5: bool,
    animate_steering_in_ue5: bool,
    animate_throttle_in_ue5: bool,
    steering_visual_gain: float,
    throttle_visual_gain: float,
    ship_index: int = 0,
):
    """Build the UE5 control payload for one interpolated actuator state.

    Args:
        adapter: Adapter responsible for framework-specific ship extraction.
        env: Active framework environment.
        bridge: Connected AirSim bridge.
        actuator_state: Interpolated framework actuator vector.
        mirror_controls_in_ue5: Whether to mirror propulsion and steering.
        animate_steering_in_ue5: Whether to animate steering-only visuals.
        animate_throttle_in_ue5: Whether to animate throttle visuals.
        steering_visual_gain: Steering gain used during mapping.
        throttle_visual_gain: Throttle gain used during mapping.
        ship_index: Ship index to extract when the environment is multi-ship.

    Returns:
        Any: AirSim ``CarControls``-compatible payload, or ``None`` when the
        bridge is disconnected.
    """
    if not bridge.is_connected:
        return None

    airsim_module = bridge.airsim
    specs = adapter.extract_ship_specifications(env, ship_index=ship_index)
    replay_brake = 1.0
    speed_ratio = None
    try:
        surge_speed, _, _ = adapter.extract_velocity(env, ship_index=ship_index)
        surge_limits = getattr(specs, "surge_limits", None)
        if surge_limits is not None and len(surge_limits) >= 2:
            max_speed = max(abs(float(surge_limits[0])), abs(float(surge_limits[1])))
            if max_speed > 1e-6:
                speed_ratio = float(np.clip(abs(float(surge_speed)) / max_speed, 0.0, 1.0))
    except Exception:
        speed_ratio = None

    def _apply_visual_speed_matching(controls: Any):
        """Align the replay throttle visual with the framework speed envelope."""
        if controls is None or speed_ratio is None:
            return controls
        controls.throttle = float(
            np.clip(float(speed_ratio) * max(float(throttle_visual_gain), 0.0), 0.0, 1.0)
        )
        return controls

    if mirror_controls_in_ue5:
        controls = map_framework_actuator_state(
            airsim_module,
            specs,
            rudder_angle=float(actuator_state[0]),
            thrust=float(actuator_state[1]),
            brake=replay_brake,
            steering_gain=steering_visual_gain,
            throttle_gain=throttle_visual_gain,
        )
        controls.brake = 0.0
        return _apply_visual_speed_matching(controls)
    if animate_throttle_in_ue5:
        controls = map_framework_visual_actuator_state(
            airsim_module,
            specs,
            rudder_angle=float(actuator_state[0]),
            thrust=float(actuator_state[1]),
            brake=replay_brake,
            steering_gain=steering_visual_gain,
            throttle_gain=throttle_visual_gain,
        )
        controls.brake = 0.0
        return _apply_visual_speed_matching(controls)
    if animate_steering_in_ue5:
        return map_framework_steering_visual_state(
            airsim_module,
            specs,
            rudder_angle=float(actuator_state[0]),
            brake=replay_brake,
            steering_gain=steering_visual_gain,
        )
    return None


def _animate_framework_state(
    adapter: "ModelVisualisationAdapter",
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
    throttle_visual_gain: float = DEFAULT_THROTTLE_VISUAL_GAIN,
    use_kinematics_in_ue5: bool = False,
    min_render_hz: float = 90.0,
    max_translation_step: float = 0.15,
    max_yaw_step_rad: float = np.deg2rad(1.5),
    lock_path_in_ue5: bool = False,
    vehicle_object_name: str = "",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Mirror one framework motion update into UE5 with optional smoothing.

    Args:
        adapter: Adapter responsible for framework-specific extraction logic.
        env: Active framework environment.
        bridge: AirSim bridge used to push pose and control updates.
        target_motion_state: Latest AirSim-frame motion state.
        previous_motion_state: Previously mirrored AirSim-frame motion state.
        total_duration: Intended playback duration for the update in seconds.
        interpolation_steps: Minimum number of visual interpolation steps.
        mirror_controls_in_ue5: Whether to mirror full propulsion controls.
        animate_steering_in_ue5: Whether to animate steering visuals.
        animate_throttle_in_ue5: Whether to animate throttle visuals.
        previous_actuator_state: Previously mirrored actuator state.
        steering_visual_gain: Steering gain used for visual control mapping.
        throttle_visual_gain: Throttle gain used for visual control mapping.
        use_kinematics_in_ue5: Whether to use kinematic updates instead of
            direct pose teleportation.
        min_render_hz: Minimum desired render frequency in Hertz.
        max_translation_step: Maximum allowed translation per interpolation
            step in meters.
        max_yaw_step_rad: Maximum allowed yaw delta per interpolation step in
            radians.
        lock_path_in_ue5: Whether to stretch duration to preserve the full
            interpolated path.
        vehicle_object_name: Optional named UE5 actor to move instead of the
            AirSim vehicle.

    Returns:
        tuple[np.ndarray, Optional[np.ndarray]]: Final mirrored motion state and
        the actuator state used for visual control interpolation.
    """
    if previous_motion_state is None:
        previous_motion_state = target_motion_state

    total_duration = max(total_duration, 0.0)
    target_actuator_state = adapter.extract_actuator_state(env)

    start_x, start_y, start_yaw, start_vx, start_vy, start_yaw_rate = previous_motion_state
    target_x, target_y, target_yaw, target_vx, target_vy, target_yaw_rate = target_motion_state
    _prefetched_pose = None

    def _send_pose(alpha: float) -> None:
        """Push an interpolated motion state to UE5."""
        if not bridge.is_connected:
            return
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
        if not bridge.is_connected:
            return
        interp_act = _interpolate_actuator_state(previous_actuator_state, target_actuator_state, alpha)
        controls = _build_visual_controls(
            adapter=adapter,
            env=env,
            bridge=bridge,
            actuator_state=interp_act,
            mirror_controls_in_ue5=mirror_controls_in_ue5,
            animate_steering_in_ue5=animate_steering_in_ue5,
            animate_throttle_in_ue5=animate_throttle_in_ue5,
            steering_visual_gain=steering_visual_gain,
            throttle_visual_gain=throttle_visual_gain,
        )
        if controls is not None:
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

    visual_update_interval = _resolve_visual_update_interval(
        total_duration=total_duration,
        interpolation_steps=interpolation_steps,
        min_render_hz=min_render_hz,
    )
    if _should_snap_visual_update(total_duration, visual_update_interval, interpolation_steps):
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
    _last_pose_time = start_time - visual_update_interval

    while True:
        now = time.perf_counter()
        alpha = min((now - start_time) / total_duration, 1.0)
        is_final = alpha >= 1.0

        if not bridge.is_connected:
            break

        if is_final or (now - _last_pose_time >= visual_update_interval):
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
    """Base adapter for one visualised model family.

    Subclasses isolate family-specific policy loading, environment creation,
    and state extraction so the replay runner can remain framework-agnostic.
    """

    family: str = ""
    model_origin: str = ""
    supported_environment_names: tuple[str, ...] = ()
    filename_keywords: tuple[str, ...] = ()

    def matches_model_path(self, model_path: str) -> bool:
        """Return whether the model filename looks like this family.

        Args:
            model_path: Candidate model path supplied by the user or runner.

        Returns:
            bool: ``True`` when the stem contains one of the family keywords.
        """
        stem = Path(model_path).stem.lower()
        return any(keyword in stem for keyword in self.filename_keywords)

    def resolve_environment_name(self, model_path: str, requested_env_name: str) -> str:
        """Resolve the framework environment name used for playback.

        Args:
            model_path: Model path being visualised. Included for adapter APIs
                that may want filename-aware resolution.
            requested_env_name: User-requested environment name, or ``"auto"``.

        Returns:
            str: Canonical framework environment name to instantiate.

        Raises:
            ValueError: If the requested environment is incompatible with this
                adapter family.
        """
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
        """Load a policy checkpoint for this model family.

        Args:
            model_path: User-provided checkpoint path.
            device: Torch device selector forwarded to the policy loader.

        Returns:
            PPOPolicy: Loaded policy instance.
        """
        return load_policy(model_path, device=device)

    def build_environment(self, environment_name: str, max_steps: int):
        """Construct the framework environment used during replay.

        Args:
            environment_name: Canonical framework environment name.
            max_steps: Maximum number of environment steps for the replay run.

        Returns:
            object: Instantiated framework environment.

        Raises:
            ValueError: If the environment name is unsupported by this adapter.
        """
        builder = self.environment_builders().get(environment_name)
        if builder is None:
            supported = ", ".join(self.environment_builders())
            raise ValueError(
                f"Unsupported environment '{environment_name}' for '{self.family}'. "
                f"Supported environments: {supported}."
            )
        return builder(max_steps)

    def environment_builders(self) -> dict[str, Callable[[int], object]]:
        """Return the environment factories supported by this adapter.

        Returns:
            dict[str, Callable[[int], object]]: Mapping from environment name to
            one-argument factory taking ``max_steps``.
        """
        raise NotImplementedError

    def extract_pose(self, env, ship_index: int = 0) -> tuple[float, float, float]:
        """Extract framework-frame ``(x, y, heading)`` for one ship.

        Args:
            env: Active framework environment.
            ship_index: Ship index to extract when the environment is
                multi-ship.

        Returns:
            tuple[float, float, float]: Framework-frame pose values.

        Raises:
            ValueError: If the environment does not expose enough state to
                extract a pose.
        """
        ship_pos = _extract_indexed_value(getattr(env, "ship_pos", None), ship_index=ship_index)
        state_entry = _extract_physics_entry(env, ship_index=ship_index)
        state = getattr(state_entry, "state", state_entry)

        if ship_pos is None or state is None:
            raise ValueError(f"Adapter '{self.family}' could not extract pose from the environment.")

        ship_pos_arr = np.asarray(ship_pos, dtype=np.float64).reshape(-1)
        heading = getattr(state, "heading", None)
        if ship_pos_arr.size < 2 or heading is None:
            raise ValueError(f"Adapter '{self.family}' could not extract pose from the environment.")

        return (
            float(ship_pos_arr[0]),
            float(ship_pos_arr[1]),
            float(heading),
        )

    def extract_velocity(self, env, ship_index: int = 0) -> tuple[float, float, float]:
        """Extract framework-frame ``(surge, sway, yaw_rate)`` for one ship.

        Args:
            env: Active framework environment.
            ship_index: Ship index to extract when the environment is
                multi-ship.

        Returns:
            tuple[float, float, float]: Surge, sway, and yaw-rate values.

        Raises:
            ValueError: If the environment does not expose enough state to
                extract velocity.
        """
        phys_sim = _extract_physics_entry(env, ship_index=ship_index)
        state = getattr(phys_sim, "state", phys_sim)
        if state is None or getattr(state, "u", None) is None:
            raise ValueError(f"Adapter '{self.family}' could not extract velocity from the environment.")
        return (
            float(state.u),
            float(getattr(state, "v", 0.0)),
            float(getattr(state, "r", 0.0)),
        )

    def extract_actuator_state(self, env, ship_index: int = 0) -> Optional[np.ndarray]:
        """Extract the framework actuator state for one ship when available.

        Args:
            env: Active framework environment.
            ship_index: Ship index to extract when the environment is
                multi-ship.

        Returns:
            Optional[np.ndarray]: Flat actuator vector, or ``None`` when no
            actuator state is available.
        """
        action = _extract_indexed_value(getattr(env, "performed_action", None), ship_index=ship_index)
        if action is None:
            return None
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size == 0:
            return None
        return action_arr

    def extract_ship_specifications(self, env, ship_index: int = 0):
        """Extract ship specifications needed for visual control mapping.

        Args:
            env: Active framework environment.
            ship_index: Ship index to extract when the environment is
                multi-ship.

        Returns:
            Any: Ship specification object compatible with control-mapping
            helpers.

        Raises:
            ValueError: If the environment does not expose ship
                specifications.
        """
        ship = _extract_indexed_value(getattr(env, "ships", None), ship_index=ship_index)
        if ship is None:
            ship = _extract_indexed_value(getattr(env, "ship", None), ship_index=ship_index)
        if ship is None or not hasattr(ship, "specifications"):
            raise ValueError(
                f"Adapter '{self.family}' could not extract ship specifications from the environment."
            )
        return ship.specifications

    def motion_state_from_env(self, env, bridge: AirSimBridge) -> np.ndarray:
        """Extract a generic motion state from the current framework env.

        Args:
            env: Active framework environment.
            bridge: AirSim bridge providing coordinate transforms.

        Returns:
            np.ndarray: AirSim-frame motion state.
        """
        return _build_framework_motion_state(self, env, bridge)

    def animate_state(self, env, bridge: AirSimBridge, **kwargs) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Mirror one framework state update into UE5.

        Args:
            env: Active framework environment.
            bridge: AirSim bridge used to push visual updates.
            **kwargs: Playback parameters forwarded to
                ``_animate_framework_state``.

        Returns:
            tuple[np.ndarray, Optional[np.ndarray]]: Final mirrored motion state
            and the actuator state used for interpolation.
        """
        return _animate_framework_state(self, env, bridge, **kwargs)

    def cleanup_environment(self, env) -> None:
        """Dispose optional backend resources owned by the environment.

        Args:
            env: Framework environment created by this adapter.
        """
        if env is not None and hasattr(env, "fh_sim"):
            env.fh_sim.dispose()


class PySimVisualisationAdapter(ModelVisualisationAdapter):
    """Adapter for PySimEnv framework-model visualisation."""

    family = "pysim"
    model_origin = "pysim"
    supported_environment_names = ("PySimEnv",)
    filename_keywords = ("pysimenv", "py_sim", "framework")

    def environment_builders(self) -> dict[str, Callable[[int], object]]:
        """Return the supported PySim environment factories.

        Returns:
            dict[str, Callable[[int], object]]: Factory mapping for PySim-backed
            replay environments.
        """
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
        """Return the supported FHSim environment factories.

        Returns:
            dict[str, Callable[[int], object]]: Factory mapping for FHSim-backed
            replay environments.
        """
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
    """Return the registered model-family names.

    Returns:
        tuple[str, ...]: Registered adapter family keys in registry order.
    """
    return tuple(_VISUALISATION_ADAPTERS.keys())


def get_supported_environment_names() -> tuple[str, ...]:
    """Return the registered framework environment names.

    Returns:
        tuple[str, ...]: Sorted unique environment names supported by the
        registered adapters.
    """
    names: set[str] = set()
    for adapter in _VISUALISATION_ADAPTERS.values():
        names.update(adapter.supported_environment_names)
    return tuple(sorted(names))


def resolve_visualisation_adapter(
    model_family: str = "auto",
    model_path: str = "",
    env_name: str = "auto",
) -> ModelVisualisationAdapter:
    """Resolve the adapter used for a visualised model replay.

    Args:
        model_family: Requested adapter family, or ``"auto"`` for detection.
        model_path: Candidate model path used during automatic family
            detection.
        env_name: Explicit environment name, or ``"auto"``.

    Returns:
        ModelVisualisationAdapter: Resolved adapter instance.

    Raises:
        ValueError: If the requested family or environment is unsupported.
    """
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
