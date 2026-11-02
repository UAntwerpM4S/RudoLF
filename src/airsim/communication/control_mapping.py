"""
Helpers for mapping policy actions and actuator states to UE5 CarControls.
"""

from __future__ import annotations

import numpy as np

from typing import Any
from fw.ships.ship_specs import ShipSpecifications

_ACTION_BOUNDS_TOLERANCE = 1e-4


def validate_policy_action(
    action: np.ndarray,
    action_name: str = "action",
) -> np.ndarray:
    """
    Validate a policy action and clip it to [-1, 1].

    Args:
        action: Candidate action vector.
        action_name: Name used in error messages.

    Returns:
        float32 array of shape (2,).

    Raises:
        ValueError: Wrong shape, non-finite values, or outside [-1, 1] by more
            than a small tolerance.
    """

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (2,):
        raise ValueError(f"{action_name} must be shape (2,), got {action.shape}")

    if not np.all(np.isfinite(action)):
        raise ValueError(f"{action_name} must contain only finite values, got {action}")

    if np.any(action < -1.0 - _ACTION_BOUNDS_TOLERANCE) or np.any(
        action > 1.0 + _ACTION_BOUNDS_TOLERANCE
    ):
        raise ValueError(
            f"{action_name} must stay within [-1, 1], got {action}"
        )

    return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


def _normalize_framework_thrust(specs: ShipSpecifications, thrust: float) -> float:
    """
    Normalize framework thrust from [min_thrust, max_thrust] to a UE5 throttle in [0, 1].

    A degenerate range (max <= min) just clips thrust to [0, 1].
    """

    min_thrust = float(specs.min_thrust)
    max_thrust = float(specs.max_thrust)
    if max_thrust <= min_thrust:
        return float(np.clip(thrust, 0.0, 1.0))

    clamped_thrust = float(np.clip(thrust, min_thrust, max_thrust))
    normalized_throttle = (clamped_thrust - min_thrust) / (max_thrust - min_thrust)
    return float(np.clip(normalized_throttle, 0.0, 1.0))


def build_car_controls(
    airsim_module: Any,
    steering: float,
    throttle: float,
    brake: float = 0.0,
    is_manual_gear: bool = False,
    manual_gear: int = 0,
    gear_immediate: bool = False,
):
    """
    Create a `CarControls` object; numeric inputs are clipped to their valid ranges.

    Args:
        airsim_module: The imported `cosysairsim` module.
        steering: Steering command in [-1, 1].
        throttle: Throttle command in [0, 1].
        brake: Brake command in [0, 1].
        is_manual_gear: Engage manual gear mode.
        manual_gear: Gear index used when `is_manual_gear` is True.
        gear_immediate: Apply the gear change immediately (if supported).

    Returns:
        A populated `cosysairsim.CarControls` instance.
    """

    controls = airsim_module.CarControls()
    controls.steering = float(np.clip(steering, -1.0, 1.0))
    controls.throttle = float(np.clip(throttle, 0.0, 1.0))
    controls.brake = float(np.clip(brake, 0.0, 1.0))
    controls.is_manual_gear = bool(is_manual_gear)
    controls.manual_gear = int(manual_gear)
    if hasattr(controls, "gear_immediate"):
        controls.gear_immediate = bool(gear_immediate)
    return controls


def map_direct_policy_action(airsim_module: Any, action: np.ndarray):
    """
    Map an action from a policy trained directly in AirSim to `CarControls`.

    Args:
        airsim_module: The imported `cosysairsim` module.
        action: [steering, throttle_raw], both in [-1, 1]; throttle is rescaled to [0, 1].

    Returns:
        `cosysairsim.CarControls` for `bridge.set_controls()`.

    Raises:
        ValueError: If the action fails `validate_policy_action`.
    """

    action = validate_policy_action(action)

    return build_car_controls(
        airsim_module,
        steering=float(action[0]),
        throttle=float((action[1] + 1.0) / 2.0),
        brake=0.0,
    )


def map_framework_policy_action(
    airsim_module: Any,
    specs: ShipSpecifications,
    steering: float,
    signed_throttle_action: float,
    brake: float = 0.0,
):
    """
    Map a framework policy action to UE5 `CarControls`.

    The signed throttle action is mapped linearly onto the ship's thrust range
    (-1 to min_thrust, +1 to max_thrust) and then normalized to a UE5 throttle.

    Args:
        airsim_module: The imported `cosysairsim` module.
        specs: Ship specifications supplying `min_thrust` and `max_thrust`.
        steering: Normalized steering command in [-1, 1].
        signed_throttle_action: Policy throttle output in [-1, 1].
        brake: Brake command in [0, 1].

    Returns:
        `cosysairsim.CarControls` for `bridge.set_controls()`.
    """

    signed_action = validate_policy_action(
        np.array([steering, signed_throttle_action], dtype=np.float32),
        action_name="Framework policy action",
    )
    thrust = (
        specs.min_thrust
        + 0.5 * (float(signed_action[1]) + 1.0)
        * (specs.max_thrust - specs.min_thrust)
    )
    return build_car_controls(
        airsim_module,
        steering=float(signed_action[0]),
        throttle=_normalize_framework_thrust(specs, thrust),
        brake=brake,
    )


def map_framework_actuator_state(
    airsim_module: Any,
    specs: ShipSpecifications,
    rudder_angle: float,
    thrust: float,
    brake: float = 0.0,
    steering_sign: float = 1.0,
    steering_gain: float = 1.0,
    throttle_gain: float = 1.0,
):
    """
    Map physical framework actuator outputs to UE5 `CarControls`.

    Rudder angle is normalized by `max_rudder_angle`, thrust by the thrust range;
    forward gear (manual_gear=1) is engaged so UE5 animates propulsion.

    Args:
        airsim_module: The imported `cosysairsim` module.
        specs: Ship specifications supplying rudder and thrust limits.
        rudder_angle: Rudder angle in radians.
        thrust: Thrust in the framework actuator space.
        brake: Brake command in [0, 1].
        steering_sign: Sign applied to the normalized steering (+1 or -1).
        steering_gain: Multiplier on the normalized steering.
        throttle_gain: Multiplier on the normalized throttle.

    Returns:
        `cosysairsim.CarControls` for `bridge.set_controls()`.
    """

    normalized_steering = 0.0
    if specs.max_rudder_angle != 0.0:
        normalized_steering = (
            steering_sign
            * float(steering_gain)
            * float(rudder_angle)
            / float(specs.max_rudder_angle)
        )

    return build_car_controls(
        airsim_module,
        steering=normalized_steering,
        throttle=_normalize_framework_thrust(specs, thrust) * float(throttle_gain),
        brake=brake,
        is_manual_gear=True,
        manual_gear=1,
        gear_immediate=True,
    )


def map_framework_steering_visual_state(
    airsim_module: Any,
    specs: ShipSpecifications,
    rudder_angle: float,
    brake: float = 0.0,
    steering_sign: float = 1.0,
    steering_gain: float = 0.35,
):
    """
    Map only the rudder angle to UE5 steering, for visual playback.

    Propulsion stays neutral (throttle 0) so UE5 does not fight the kinematic
    pose mirror.

    Args:
        airsim_module: The imported `cosysairsim` module.
        specs: Ship specifications supplying `max_rudder_angle`.
        rudder_angle: Rudder angle in radians.
        brake: Brake command in [0, 1].
        steering_sign: Sign applied to the normalized steering.
        steering_gain: Multiplier on the normalized steering (reduced default,
            visual only).

    Returns:
        `cosysairsim.CarControls` with throttle 0.
    """

    normalized_steering = 0.0
    if specs.max_rudder_angle != 0.0:
        normalized_steering = (
            steering_sign
            * float(steering_gain)
            * float(rudder_angle)
            / float(specs.max_rudder_angle)
        )

    return build_car_controls(
        airsim_module,
        steering=normalized_steering,
        throttle=0.0,
        brake=brake,
    )


def map_framework_visual_actuator_state(
    airsim_module: Any,
    specs: ShipSpecifications,
    rudder_angle: float,
    thrust: float,
    brake: float = 0.0,
    steering_sign: float = 1.0,
    steering_gain: float = 0.35,
    throttle_gain: float = 1.0,
):
    """
    Map rudder and thrust to UE5 controls in forward gear, for visual animation only.

    The mirrored framework motion stays the source of truth for replay.
    Arguments as in `map_framework_actuator_state` (steering_gain defaults to 0.35).

    Returns:
        `cosysairsim.CarControls` with forward gear engaged.
    """

    normalized_steering = 0.0
    if specs.max_rudder_angle != 0.0:
        normalized_steering = (
            steering_sign
            * float(steering_gain)
            * float(rudder_angle)
            / float(specs.max_rudder_angle)
        )

    return build_car_controls(
        airsim_module,
        steering=normalized_steering,
        throttle=_normalize_framework_thrust(specs, thrust) * float(throttle_gain),
        brake=brake,
        is_manual_gear=True,
        manual_gear=1,
        gear_immediate=True,
    )
