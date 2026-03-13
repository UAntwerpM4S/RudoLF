"""Visualise a Fossen-trained PPO policy in UE5 using the framework simulator.

This runner keeps the Fossen dynamics inside the framework and uses AirSim / UE5
only as a real-time renderer. The bridge loads the saved world, applies the saved
coordinate calibration, and mirrors the framework vessel pose into UE5 each step.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from airsim.communication.airsim_bridge import AirSimBridge  # noqa: E402
from airsim.communication.control_mapping import (  # noqa: E402
    map_fossen_actuator_state,
    map_fossen_steering_visual_state,
    map_fossen_visual_actuator_state,
)
from airsim.config.airsim_config import (  # noqa: E402
    AirSimConfig,
    DEFAULT_AIRSIM_SAVE_SLOT,
    DEFAULT_AIRSIM_SIMULATION_SPEED,
)
from airsim.inference.minimap_fossen_airsim import MinimapFossenAirsim  # noqa: E402
from fw.environments.fh_sim_env import FhSimEnv  # noqa: E402
from fw.environments.py_sim_env import PySimEnv  # noqa: E402
from fw.policies.ppo_policy import PPOPolicy  # noqa: E402

logger = logging.getLogger(__name__)


def _wrapped_angle_delta(start: float, end: float) -> float:
    """Return the shortest signed angular delta from ``start`` to ``end``.

    Args:
        start: Start angle in radians.
        end: End angle in radians.

    Returns:
        Signed wrapped delta in radians within ``[-pi, pi]``.
    """
    return (end - start + np.pi) % (2.0 * np.pi) - np.pi


def _interpolate_angle(start: float, end: float, alpha: float) -> float:
    """Interpolate between two angles while respecting wrap-around.

    Args:
        start: Start angle in radians.
        end: End angle in radians.
        alpha: Interpolation factor in ``[0, 1]``.

    Returns:
        Interpolated angle in radians.
    """
    wrapped_delta = _wrapped_angle_delta(start, end)
    return start + alpha * wrapped_delta


def _interpolate_actuator_state(
    previous_state: Optional[np.ndarray],
    target_state: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linearly interpolate between two actuator-state vectors.

    Args:
        previous_state: Previous actuator vector, or ``None`` when no prior
            state exists.
        target_state: Target actuator vector.
        alpha: Interpolation factor in ``[0, 1]``.

    Returns:
        Interpolated actuator state. When ``previous_state`` is ``None``, the
        target state is returned unchanged.
    """
    if previous_state is None:
        return target_state
    return previous_state + alpha * (target_state - previous_state)


def _resolve_model_path(model_path: str) -> Path:
    """Resolve a model path against the standard repository search roots.

    Args:
        model_path: User-supplied model path, absolute or relative.

    Returns:
        A resolved filesystem path. If the file is not found in the search
        roots, the repository-root candidate is returned unchanged.
    """
    path = Path(model_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        Path.cwd() / path,
        _REPO_ROOT / path,
        _SRC_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (_REPO_ROOT / path).resolve()


def load_policy(model_path: str, device: str = "cpu") -> PPOPolicy:
    """Load a PPO policy for framework-driven Fossen visualisation.

    Args:
        model_path: Path to the saved model, with or without the ``.zip``
            suffix.
        device: Torch device string, typically ``"cpu"`` or ``"cuda"``.

    Returns:
        A loaded ``PPOPolicy`` with its network switched to eval mode.
    """
    path = _resolve_model_path(model_path)
    stem = str(path.with_suffix("")) if path.suffix == ".zip" else str(path)
    policy = PPOPolicy.load(stem, device=device)
    policy.network.eval()
    return policy


def _detect_fossen_environment(model_path: str, env_name: str) -> str:
    """Choose the framework environment used to replay a Fossen policy.

    Args:
        model_path: Saved policy path used for auto-detection.
        env_name: Requested environment name, or ``"auto"``.

    Returns:
        Either ``"FhSimEnv"`` or ``"PySimEnv"``.
    """
    if env_name.lower() != "auto":
        return env_name

    stem = Path(model_path).stem.lower()
    if "fhsimenv" in stem or "fh_sim" in stem:
        return "FhSimEnv"
    return "PySimEnv"


def _build_fossen_environment(env_name: str, max_steps: int):
    """Instantiate the selected framework environment for playback.

    Args:
        env_name: Name of the environment class to construct.
        max_steps: Episode step limit forwarded to the environment.

    Returns:
        An initialised ``FhSimEnv`` or ``PySimEnv`` instance.

    Raises:
        ValueError: If ``env_name`` is not supported.
    """
    ship_pos = [2245, -47]
    target_pos = [-1559, 11990]
    if env_name == "FhSimEnv":
        return FhSimEnv(max_steps=max_steps, start_pos=ship_pos, target_pos=target_pos)
    if env_name == "PySimEnv":
        return PySimEnv(max_steps=max_steps, start_pos=ship_pos, target_pos=target_pos)
    raise ValueError(f"Unsupported Fossen environment '{env_name}'.")


def _framework_pose_from_env(env, bridge: AirSimBridge) -> tuple[float, float, float]:
    """Convert the current framework ship pose into AirSim coordinates.

    Args:
        env: Active framework environment containing ship state.
        bridge: Connected AirSim bridge with an active coordinate transform.

    Returns:
        A tuple ``(airsim_x, airsim_y, airsim_yaw)``.

    Raises:
        RuntimeError: If the bridge has no loaded coordinate transform.
    """
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
    """Extract AirSim-frame pose and velocities from the framework state.

    Args:
        env: Active framework environment containing Fossen state.
        bridge: Connected AirSim bridge with a coordinate transform.

    Returns:
        A six-element array containing AirSim-frame position, yaw, planar
        velocity, and yaw rate.
    """
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
    """Compute how many UE5 interpolation steps are needed for a motion update.

    Args:
        previous_motion_state: Previous AirSim-frame motion state.
        target_motion_state: Target AirSim-frame motion state.
        interpolation_steps: Minimum requested interpolation step count.
        max_translation_step: Maximum allowed translation per UE5 update.
        max_yaw_step_rad: Maximum allowed yaw delta per UE5 update.

    Returns:
        The effective interpolation step count after enforcing translation and
        yaw-rate limits.
    """
    steps = max(1, int(interpolation_steps))
    delta_x = float(target_motion_state[0] - previous_motion_state[0])
    delta_y = float(target_motion_state[1] - previous_motion_state[1])
    translation = float(np.hypot(delta_x, delta_y))
    yaw_delta = abs(_wrapped_angle_delta(float(previous_motion_state[2]), float(target_motion_state[2])))

    if max_translation_step > 0.0:
        steps = max(steps, int(np.ceil(translation / max_translation_step)))
    if max_yaw_step_rad > 0.0:
        steps = max(steps, int(np.ceil(yaw_delta / max_yaw_step_rad)))
    return max(1, steps)


def _should_snap_visual_update(total_duration: float, min_render_hz: float) -> bool:
    """Return whether a visual update should skip time-based interpolation.

    When the allocated wall-clock duration for one mirrored framework step is
    shorter than the minimum render interval, interpolation cannot produce more
    than a single visible pose push anyway. In that case, sleeping inside the
    interpolation loop only adds jitter and can make both UE5 and the minimap
    appear to slow down abruptly.

    Args:
        total_duration: Wall-clock duration allocated to the update.
        min_render_hz: Minimum render cadence used for UE5 pose pushes.

    Returns:
        ``True`` when the update should snap directly to the target pose.
    """
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
    """Mirror one framework motion update into UE5 with optional smoothing.

    Args:
        env: Active framework environment providing actuator state.
        bridge: Connected AirSim bridge used to send pose and control updates.
        target_motion_state: Target AirSim-frame motion state for this step.
        previous_motion_state: Previous AirSim-frame motion state.
        total_duration: Wall-clock duration allocated to the interpolation.
        interpolation_steps: Minimum number of interpolation segments.
        mirror_controls_in_ue5: Whether to mirror full Fossen actuator output.
        animate_steering_in_ue5: Whether to animate steering visuals in UE5.
        animate_throttle_in_ue5: Whether to animate throttle visuals in UE5.
        previous_actuator_state: Previous actuator-state vector.
        steering_visual_gain: Gain applied to visual steering commands.
        throttle_visual_gain: Gain applied to visual throttle commands.
        use_kinematics_in_ue5: Whether to use ``simSetKinematics`` instead of
            pose updates when available.
        min_render_hz: Minimum wall-clock interval between pose pushes.
        max_translation_step: Maximum translation per UE5 update in metres.
        max_yaw_step_rad: Maximum yaw delta per UE5 update in radians.
        lock_path_in_ue5: Reserved compatibility flag for path locking.
        vehicle_object_name: Optional non-physics UE5 actor name to move with
            ``simSetObjectPose``.

    Returns:
        A tuple ``(target_motion_state, target_actuator_state)`` representing
        the new interpolation baseline.
    """
    if previous_motion_state is None:
        previous_motion_state = target_motion_state

    total_duration = max(total_duration, 0.0)

    target_actuator_state = None
    if getattr(env, "performed_action", None):
        target_actuator_state = np.asarray(env.performed_action[0], dtype=np.float32)

    start_x, start_y, start_yaw, start_vx, start_vy, start_yaw_rate = previous_motion_state
    target_x, target_y, target_yaw, target_vx, target_vy, target_yaw_rate = target_motion_state

    # _prefetched_pose is populated after the snap check (skipping the RPC on
    # high-speed snap steps).  It must be declared here so the _send_pose
    # closure can always reference it regardless of which code path runs.
    _prefetched_pose = None

    def _send_pose(alpha: float) -> None:
        ix = start_x + alpha * (target_x - start_x)
        iy = start_y + alpha * (target_y - start_y)
        iyaw = _interpolate_angle(start_yaw, target_yaw, alpha)
        if vehicle_object_name:
            # Object-pose path: moves a non-physics mesh actor - no water ripple,
            # no collision response, no buoyancy reactions.
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
        if vehicle_object_name:
            # Object-pose mode: the actor is a non-physics mesh with no vehicle
            # controls. Skip setCarControls to save an RPC round-trip per step.
            return
        if target_actuator_state is None:
            return
        if not (mirror_controls_in_ue5 or animate_steering_in_ue5 or animate_throttle_in_ue5):
            return
        interp_act = _interpolate_actuator_state(previous_actuator_state, target_actuator_state, alpha)
        if mirror_controls_in_ue5:
            controls = map_fossen_actuator_state(
                bridge._airsim, env.ships[0].specifications,
                rudder_angle=float(interp_act[0]), thrust=float(interp_act[1]),
                steering_gain=steering_visual_gain,
            )
        elif animate_throttle_in_ue5:
            controls = map_fossen_visual_actuator_state(
                bridge._airsim, env.ships[0].specifications,
                rudder_angle=float(interp_act[0]), thrust=float(interp_act[1]),
                steering_gain=steering_visual_gain, throttle_gain=throttle_visual_gain,
            )
        else:
            controls = map_fossen_steering_visual_state(
                bridge._airsim, env.ships[0].specifications,
                rudder_angle=float(interp_act[0]), steering_gain=steering_visual_gain,
            )
        bridge.set_controls(controls)

    # -- Instant snap when interpolation would be invisible or too short --
    # If the requested wall-clock duration is shorter than the effective render
    # interval, time-based interpolation only burns CPU and introduces jitter.
    # Snapping directly preserves the intended high-speed cadence more reliably.
    min_step_interval = 1.0 / min_render_hz if min_render_hz > 0.0 else 0.0
    if _should_snap_visual_update(total_duration, min_render_hz):
        _send_pose(1.0)
        _send_controls(1.0)
        return target_motion_state, target_actuator_state

    # Pre-fetch UE5 vehicle pose once per Fossen step so set_pose never calls
    # simGetVehiclePose() inside the tight interpolation loop.
    # Placed after the snap check so high-speed instant steps skip this RPC.
    # Skipped in object-pose and kinematics modes (no vehicle body to query).
    _prefetched_pose = None
    if not vehicle_object_name and not use_kinematics_in_ue5 and bridge.is_connected:
        try:
            _prefetched_pose = bridge.get_vehicle_pose()
        except Exception:
            pass

    # -- Elapsed-time driven interpolation --
    # Alpha is computed from actual wall-clock time, so the loop is immune
    # to per-RPC latency variance.  Controls are sent once at the end to
    # avoid mid-interpolation physics interference from setCarControls().
    # A short sleep() yields CPU to UE5's render/physics thread between
    # pose updates, preventing the busy-wait that starved UE5 previously.
    start_time = time.perf_counter()
    deadline = start_time + total_duration
    _last_pose_time = start_time - min_step_interval  # allow first update immediately

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


def run_visualise_fossen(
    model_path: str,
    env_name: str = "auto",
    ip: str = "",
    port: int = 41451,
    max_steps: int = 14000,
    num_episodes: int = 1,
    device: str = "cpu",
    draw_calibration_reference: bool = True,
    log_every: int = 50,
    save_slot: str = DEFAULT_AIRSIM_SAVE_SLOT,
    simulation_speed: float = DEFAULT_AIRSIM_SIMULATION_SPEED,
    mirror_controls_in_ue5: bool = False,
    animate_steering_in_ue5: bool = True,
    animate_throttle_in_ue5: bool = True,
    render_interpolation_steps: int = 4,
    steering_visual_gain: float = 0.35,
    throttle_visual_gain: float = 1.0,
    use_kinematics_in_ue5: bool = False,
    min_render_hz: float = 90.0,
    max_translation_step: float = 0.15,
    max_yaw_step_deg: float = 1.5,
    lock_path_in_ue5: bool = False,
    vehicle_object_name: str = "",
    show_minimap: bool = True,
) -> None:
    """Run a Fossen-trained PPO policy and mirror it into UE5 in real time."""
    selected_env = _detect_fossen_environment(model_path, env_name)
    banner = "=" * 62
    print(banner)
    print("  FOSSEN POLICY VISUALISATION IN UE5")
    print(f"  Model      : {Path(model_path).name}")
    print(f"  Framework  : {selected_env}")
    print(f"  Save slot  : {save_slot}")
    if vehicle_object_name:
        print(f"  Mode       : framework physics -> UE5 object pose ('{vehicle_object_name}') - no water physics")
    elif use_kinematics_in_ue5:
        print("  Mode       : framework physics -> UE5 mirrored kinematics")
    else:
        print("  Mode       : framework physics -> UE5 mirrored pose")
    if mirror_controls_in_ue5:
        control_mode = "full Fossen actuator mapping"
    elif animate_throttle_in_ue5:
        control_mode = "visual rudder + throttle animation"
    elif animate_steering_in_ue5:
        control_mode = "steering-only visual mapping"
    else:
        control_mode = "kinematic pose only"
    print(f"  Controls   : {control_mode}")
    render_label = "kinematic updates" if use_kinematics_in_ue5 else "pose updates"
    print(f"  Render     : {max(1, int(render_interpolation_steps))} interpolated {render_label} per Fossen step")
    print(f"  Smoothing  : min {min_render_hz:.0f} Hz, max {max_translation_step:.2f} m / {max_yaw_step_deg:.1f} deg per UE5 update")
    print(f"  Steering   : UE5 visual gain {steering_visual_gain:.2f}")
    print(f"  Throttle   : UE5 visual gain {throttle_visual_gain:.2f}")
    print("  PID        : NOT USED")
    print(banner)

    policy = load_policy(model_path, device=device)
    bridge = AirSimBridge(AirSimConfig(ip=ip, port=port))
    if not bridge.is_connected:
        raise ConnectionError(
            f"AirSimBridge could not connect to UE5 at {ip or '127.0.0.1'}:{port}."
        )

    env = _build_fossen_environment(selected_env, max_steps=max_steps)
    max_yaw_step_rad = np.deg2rad(max_yaw_step_deg)

    try:
        bridge.prepare_fossen_visualisation(
            save_slot=save_slot,
            simulation_speed=simulation_speed,
            draw_reference=draw_calibration_reference,
        )
        logger.info("Loaded saved calibration reference overlay.")

        minimap: Optional[MinimapFossenAirsim] = None

        if not mirror_controls_in_ue5 and not animate_steering_in_ue5 and not animate_throttle_in_ue5:
            neutral_controls = bridge._airsim.CarControls()
            neutral_controls.throttle = 0.0
            neutral_controls.steering = 0.0
            neutral_controls.brake = 0.0
            bridge.set_controls(neutral_controls)

        for episode in range(num_episodes):
            print(f"\n-- Visualisation episode {episode + 1}/{num_episodes} --")
            obs, _ = env.reset()

            # Build (or rebuild) the live minimap once per episode so the
            # static background reflects the freshly reset environment.
            if show_minimap:
                if minimap is not None:
                    minimap.close()
                try:
                    minimap = MinimapFossenAirsim(env, save_dir=".")
                except Exception as _mm_exc:
                    logger.warning("Minimap unavailable: %s", _mm_exc)
                    minimap = None

            previous_motion_state = _framework_motion_state_from_env(env, bridge)
            previous_actuator_state = None
            previous_motion_state, previous_actuator_state = _animate_framework_state(
                env,
                bridge,
                target_motion_state=previous_motion_state,
                previous_motion_state=None,
                total_duration=0.0,
                interpolation_steps=1,
                mirror_controls_in_ue5=False,
                animate_steering_in_ue5=animate_steering_in_ue5,
                animate_throttle_in_ue5=animate_throttle_in_ue5,
                previous_actuator_state=None,
                steering_visual_gain=steering_visual_gain,
                throttle_visual_gain=throttle_visual_gain,
                use_kinematics_in_ue5=use_kinematics_in_ue5,
                min_render_hz=min_render_hz,
                max_translation_step=max_translation_step,
                max_yaw_step_rad=max_yaw_step_rad,
                lock_path_in_ue5=lock_path_in_ue5,
                vehicle_object_name=vehicle_object_name,
            )

            # Park the vehicle physics and let the water surface settle after the
            # initial teleport to the starting position before the policy runs.
            if bridge.is_connected and not vehicle_object_name:
                _park = bridge._airsim.CarControls()
                _park.throttle = 0.0
                _park.steering = 0.0
                _park.brake = 1.0
                bridge.set_controls(_park)
            time.sleep(0.4)

            total_reward = 0.0
            step = 0
            # Rate-limit UE5 pose updates to min_render_hz wall-clock frequency.
            # At normal speeds Python is the bottleneck so UE5 is updated every
            # step anyway.  At high simulation_speed Python runs faster than the
            # UE5 render rate; batching Fossen steps between updates removes the
            # RPC bottleneck and lets the simulation advance much faster.
            _ue5_update_dt = 1.0 / min_render_hz if min_render_hz > 0.0 else 0.0
            _last_ue5_t = time.perf_counter() - _ue5_update_dt  # allow first update immediately
            while step < max_steps:
                policy_obs = obs[0] if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs
                action, _, _ = policy.predict(policy_obs, deterministic=True)
                action = np.asarray(action).reshape(-1)

                obs, reward, done, truncated, _ = env.step(action)
                total_reward += float(reward)
                step += 1

                if minimap is not None:
                    minimap.update(env)

                target_motion_state = _framework_motion_state_from_env(env, bridge)
                _now = time.perf_counter()
                if (_now - _last_ue5_t) >= _ue5_update_dt or done or truncated:
                    _last_ue5_t = _now
                    previous_motion_state, previous_actuator_state = _animate_framework_state(
                        env,
                        bridge,
                        target_motion_state=target_motion_state,
                        previous_motion_state=previous_motion_state,
                        total_duration=env.time_step / max(simulation_speed, 1e-6),
                        interpolation_steps=max(1, int(render_interpolation_steps)),
                        mirror_controls_in_ue5=mirror_controls_in_ue5,
                        animate_steering_in_ue5=animate_steering_in_ue5,
                        animate_throttle_in_ue5=animate_throttle_in_ue5,
                        previous_actuator_state=previous_actuator_state,
                        steering_visual_gain=steering_visual_gain,
                        throttle_visual_gain=throttle_visual_gain,
                        use_kinematics_in_ue5=use_kinematics_in_ue5,
                        min_render_hz=min_render_hz,
                        max_translation_step=max_translation_step,
                        max_yaw_step_rad=max_yaw_step_rad,
                        lock_path_in_ue5=lock_path_in_ue5,
                        vehicle_object_name=vehicle_object_name,
                    )
                else:
                    # Skip UE5 update this step; advance the motion-state baseline
                    # so the next update interpolates from a recent position.
                    previous_motion_state = target_motion_state
                if log_every > 0 and step % log_every == 0:
                    logger.info(
                        "step=%d pos=(%.0f, %.0f) speed=%.2f steer=%+.3f throttle=%.3f brake=0.000 reward=%+.2f",
                        step,
                        env.ship_pos[0][0],
                        env.ship_pos[0][1],
                        env.phys_sims[0].state.u,
                        bridge._controls.steering if bridge._controls is not None else 0.0,
                        bridge._controls.throttle if bridge._controls is not None else 0.0,
                        reward,
                    )

                if done or truncated:
                    break

            print(
                f"   Finished: {step} steps | "
                f"reward={total_reward:+.1f} | "
                f"end=({env.ship_pos[0][0]:.1f}, {env.ship_pos[0][1]:.1f})"
            )

            if minimap is not None:
                minimap.finish(total_reward, episode=episode + 1)

    finally:
        if minimap is not None:
            minimap.close()
        # Reset UE5 to real-time so the engine does not stay at an elevated
        # simulation speed after the visualisation script exits.
        try:
            if bridge.is_connected:
                bridge.set_simulation_speed(1.0)
        except Exception:
            pass
        if hasattr(env, "fh_sim"):
            env.fh_sim.dispose()
        bridge.disconnect()
        print("\nVisualisation complete - AirSim connection closed.")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Fossen-in-UE5 visualisation.

    Returns:
        A configured ``ArgumentParser`` for the Fossen visualisation entry
        point.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Visualise a Fossen-trained PPO policy in the UE5 AirSim world in "
            "real time using the saved calibration and simulator_1 world save."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the Fossen-trained .zip model file.",
    )
    parser.add_argument(
        "--env",
        default="auto",
        choices=["auto", "PySimEnv", "FhSimEnv"],
        help="Framework simulator to use.",
    )
    parser.add_argument(
        "--ip",
        default="",
        help="AirSim server IP address (empty string = 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=41451,
        help="AirSim RPC port.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of visualisation episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=14000,
        dest="max_steps",
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device used for policy inference.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log mapped UE5 controls every N steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--save-slot",
        default="simulator_1",
        help="UE5 save slot to load before visualisation.",
    )
    parser.add_argument(
        "--simulation-speed",
        type=float,
        default=15.0,
        help="UE5 simulation speed during visualisation.",
    )
    parser.add_argument(
        "--mirror-controls",
        action="store_true",
        default=False,
        help="Also send mapped Fossen controls into UE5. Disabled by default to keep rendering kinematic and stable.",
    )
    parser.add_argument(
        "--no-steering-visual",
        action="store_true",
        default=False,
        help="Disable steering-only visual mapping during kinematic playback.",
    )
    parser.add_argument(
        "--no-throttle-visual",
        action="store_true",
        default=False,
        help="Disable throttle animation during kinematic playback.",
    )
    parser.add_argument(
        "--render-interpolation-steps",
        type=int,
        default=8,
        help="Number of interpolated UE5 kinematic updates per Fossen step.",
    )
    parser.add_argument(
        "--steering-visual-gain",
        type=float,
        default=0.35,
        help="Scale factor for UE5 steering animation relative to full Fossen rudder deflection.",
    )
    parser.add_argument(
        "--throttle-visual-gain",
        type=float,
        default=1.0,
        help="Scale factor for UE5 throttle animation during visual playback.",
    )
    parser.add_argument(
        "--use-kinematics-in-ue5",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use experimental AirSim kinematics mirroring instead of the default pose mirroring.",
    )
    parser.add_argument(
        "--min-render-hz",
        type=float,
        default=90.0,
        help="Minimum effective UE5 update rate for visual playback. Higher values preserve real-time speed less aggressively.",
    )
    parser.add_argument(
        "--max-translation-step",
        type=float,
        default=0.15,
        help="Maximum translation distance in metres for each mirrored UE5 update.",
    )
    parser.add_argument(
        "--max-yaw-step-deg",
        type=float,
        default=1.5,
        help="Maximum yaw delta in degrees for each mirrored UE5 update.",
    )
    parser.add_argument(
        "--lock-path-in-ue5",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enforce minimum render Hz by slowing down Fossen if necessary. Disabled by default.",
    )
    parser.add_argument(
        "--no-calibration-reference",
        action="store_true",
        default=False,
        help="Skip redrawing the saved calibration overlay in UE5.",
    )
    parser.add_argument(
        "--vehicle-object-name",
        default="",
        dest="vehicle_object_name",
        help=(
            "UE5 Outliner name of a non-physics Static/Skeletal Mesh Actor to move "
            "instead of the vehicle pawn. When set, simSetObjectPose is used and "
            "water physics is completely bypassed. Leave empty to use the default "
            "vehicle pawn (simSetVehiclePose)."
        ),
    )
    parser.add_argument(
        "--no-minimap",
        action="store_true",
        default=False,
        dest="no_minimap",
        help="Disable the live matplotlib minimap window.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = _build_parser().parse_args()
    run_visualise_fossen(
        model_path=args.model,
        env_name=args.env,
        ip=args.ip,
        port=args.port,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        device=args.device,
        draw_calibration_reference=not args.no_calibration_reference,
        log_every=args.log_every,
        save_slot=args.save_slot,
        simulation_speed=args.simulation_speed,
        mirror_controls_in_ue5=args.mirror_controls,
        animate_steering_in_ue5=not args.no_steering_visual,
        animate_throttle_in_ue5=not args.no_throttle_visual,
        render_interpolation_steps=args.render_interpolation_steps,
        steering_visual_gain=args.steering_visual_gain,
        throttle_visual_gain=args.throttle_visual_gain,
        use_kinematics_in_ue5=args.use_kinematics_in_ue5,
        min_render_hz=args.min_render_hz,
        max_translation_step=args.max_translation_step,
        max_yaw_step_deg=args.max_yaw_step_deg,
        lock_path_in_ue5=args.lock_path_in_ue5,
        vehicle_object_name=args.vehicle_object_name,
        show_minimap=not args.no_minimap,
    )