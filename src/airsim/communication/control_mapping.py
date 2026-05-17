"""
Helpers for mapping policy actions to UE5 CarControls.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fw.ships.ship_specs import ShipSpecifications


_ACTION_BOUNDS_TOLERANCE = 1e-4


def validate_policy_action(
    action: np.ndarray,
    action_name: str = "action",
) -> np.ndarray:
    """
    Validate and lightly clamp a policy action in the expected space.

    Args:
        action: Candidate action vector.
        action_name: Human-readable name used in error messages.

    Returns:
        A float32 action vector of shape ``(2,)`` clipped to ``[-1, 1]``.

    Raises:
        ValueError: If the action shape is wrong, contains non-finite values,
            or exceeds the valid range by more than a small numeric tolerance.
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
    Convert a framework thrust value to a UE5 throttle command in [0, 1].

    Args:
        specs: Ship specifications containing ``min_thrust`` and ``max_thrust``.
        thrust: Raw thrust value in the framework actuator space.

    Returns:
        Throttle value normalized to the range ``[0, 1]``.
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
    Create a populated ``CarControls`` object with clipped values.

    All numeric inputs are clipped to their valid ranges before being written
    to the ``CarControls`` object so callers do not need to guard against
    out-of-range values.

    Args:
        airsim_module: The imported ``cosysairsim`` module.
        steering: Steering command in ``[-1, 1]``.
        throttle: Throttle command in ``[0, 1]``.
        brake: Brake command in ``[0, 1]``.  Defaults to ``0.0``.
        is_manual_gear: Whether to engage manual gear mode.
        manual_gear: Gear index to engage when ``is_manual_gear`` is ``True``.
        gear_immediate: Whether the gear change takes effect immediately.

    Returns:
        A fully populated ``cosysairsim.CarControls`` instance.
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
    Map a normalized UE5-trained action to direct ``CarControls``.

    Expects an action produced by an RL policy trained directly in AirSim
    (not an external framework replay). The throttle dimension is rescaled from ``[-1, 1]`` to
    ``[0, 1]``.

    Args:
        airsim_module: The imported ``cosysairsim`` module.
        action: Array of shape ``(2,)`` containing
            ``[steering, throttle_raw]`` each in ``[-1, 1]``.

    Returns:
        A ``cosysairsim.CarControls`` object ready to pass to
        ``bridge.set_controls()``.

    Raises:
        ValueError: If *action* does not have shape ``(2,)``.
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
    Map a framework policy action to the equivalent UE5 ``CarControls``.

    The throttle dimension is mapped from the signed policy action
    ``[-1, 1]`` to the ship-spec thrust range using a linear affine
    transform: ``-1 -> min_thrust``, ``+1 -> max_thrust``.

    Args:
        airsim_module: The imported ``cosysairsim`` module.
        specs: Ship specifications supplying ``min_thrust`` and ``max_thrust``.
        steering: Normalised steering command in ``[-1, 1]``.
        signed_throttle_action: Raw policy throttle output in ``[-1, 1]``.
        brake: Brake command in ``[0, 1]``.  Defaults to ``0.0``.

    Returns:
        A ``cosysairsim.CarControls`` object ready to pass to
        ``bridge.set_controls()``.
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
    Map physical framework actuator outputs to UE5 ``CarControls``.

    Normalizes the rudder angle against the ship's ``max_rudder_angle`` and
    the thrust against ``min_thrust`` / ``max_thrust``, then engages forward
    gear (``manual_gear=1``) so UE5 produces visible propulsion animation.

    Args:
        airsim_module: The imported ``cosysairsim`` module.
        specs: Ship specifications supplying rudder and thrust limits.
        rudder_angle: Physical rudder angle in radians.
        thrust: Physical thrust in the framework actuator space.
        brake: Brake command in ``[0, 1]``.  Defaults to ``0.0``.
        steering_sign: Sign applied to the normalized steering (``+1`` or
            ``-1``).  Defaults to ``1.0``.
        steering_gain: Additional multiplier on the normalized steering.
            Defaults to ``1.0``.
        throttle_gain: Additional multiplier on the normalized throttle.
            Defaults to ``1.0``.

    Returns:
        A ``cosysairsim.CarControls`` object ready to pass to
        ``bridge.set_controls()``.
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
    Map only the framework rudder angle to UE5 steering for visual playback.

    This keeps propulsion neutral so UE5 does not fight the kinematic pose
    mirror while still animating the visible steering / rudder state.

    Args:
        airsim_module: The imported ``cosysairsim`` module.
        specs: Ship specifications supplying ``max_rudder_angle``.
        rudder_angle: Physical rudder angle in radians.
        brake: Brake command in ``[0, 1]``.  Defaults to ``0.0``.
        steering_sign: Sign applied to the normalized steering.  Defaults
            to ``1.0``.
        steering_gain: Additional multiplier on the normalized steering.
            Defaults to ``0.35`` (visual-only, reduced to avoid fighting UE5
            physics).

    Returns:
        A ``cosysairsim.CarControls`` object with ``throttle=0.0``.
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
    Map framework rudder and thrust for visual animation in forward gear.

    This exposes steering and throttle animation in UE5 while the mirrored
    framework motion remains the source of truth for replay.

    Args:
        airsim_module: The imported ``cosysairsim`` module.
        specs: Ship specifications supplying rudder and thrust limits.
        rudder_angle: Physical rudder angle in radians.
        thrust: Physical thrust in the framework actuator space.
        brake: Brake command in ``[0, 1]``.  Defaults to ``0.0``.
        steering_sign: Sign applied to the normalized steering.  Defaults
            to ``1.0``.
        steering_gain: Additional multiplier on the normalized steering.
            Defaults to ``0.35``.
        throttle_gain: Multiplier applied to the normalized throttle.
            Defaults to ``1.0``.

    Returns:
        A ``cosysairsim.CarControls`` object with forward gear engaged.
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
