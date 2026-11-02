import types

import numpy as np
import pytest

from airsim.communication.control_mapping import (
    map_direct_policy_action,
    map_framework_actuator_state,
    map_framework_policy_action,
    validate_policy_action,
)
from fw.ships.ship_factory import create_ship


def _make_airsim_module():
    """
    Build a stub airsim module exposing only CarControls.
    """

    module = types.SimpleNamespace()

    class CarControls:
        """
        Minimal stand-in for airsim.CarControls.
        """

        def __init__(self):
            """
            Neutral controls.
            """

            self.steering = 0.0
            self.throttle = 0.0
            self.brake = 0.0
            self.is_manual_gear = False
            self.manual_gear = 0
            self.gear_immediate = False

    module.CarControls = CarControls
    return module


def test_validate_policy_action_clips_small_numeric_overshoot():
    """
    Small numeric overshoot is clipped to [-1, 1].
    """

    action = validate_policy_action(np.array([1.0 + 1e-6, -1.0 - 1e-6], dtype=np.float32))
    np.testing.assert_allclose(action, np.array([1.0, -1.0], dtype=np.float32))


def test_validate_policy_action_rejects_large_out_of_bounds():
    """
    Large out-of-bounds actions are rejected.
    """

    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        validate_policy_action(np.array([0.0, 1.2], dtype=np.float32))


def test_validate_policy_action_rejects_non_finite_values():
    """
    Non-finite actions are rejected.
    """

    with pytest.raises(ValueError, match="finite values"):
        validate_policy_action(np.array([0.0, np.nan], dtype=np.float32))


def test_map_framework_policy_action_normalizes_to_ue5_throttle_range():
    """
    Framework policy actions are normalized to UE5 throttle.
    """

    airsim_module = _make_airsim_module()
    specs = create_ship().specifications

    controls = map_framework_policy_action(
        airsim_module,
        specs,
        steering=0.0,
        signed_throttle_action=0.0,
    )

    assert 0.0 <= controls.throttle <= 1.0
    assert controls.throttle == pytest.approx(0.5, abs=1e-6)


def test_map_direct_policy_action_preserves_full_lock_left_and_full_throttle():
    """
    AirSim-trained actions should reach the UE5 control extremes.
    """

    airsim_module = _make_airsim_module()

    controls = map_direct_policy_action(
        airsim_module,
        np.array([-1.0, 1.0], dtype=np.float32),
    )

    assert controls.steering == pytest.approx(-1.0, abs=1e-6)
    assert controls.throttle == pytest.approx(1.0, abs=1e-6)
    assert controls.brake == pytest.approx(0.0, abs=1e-6)


def test_map_direct_policy_action_preserves_full_lock_right_and_zero_throttle():
    """
    Signed throttle should map -1 to zero propulsion without clipping steering.
    """

    airsim_module = _make_airsim_module()

    controls = map_direct_policy_action(
        airsim_module,
        np.array([1.0, -1.0], dtype=np.float32),
    )

    assert controls.steering == pytest.approx(1.0, abs=1e-6)
    assert controls.throttle == pytest.approx(0.0, abs=1e-6)
    assert controls.brake == pytest.approx(0.0, abs=1e-6)


def test_map_framework_actuator_state_applies_throttle_gain():
    """
    Mirrored framework throttle should support calibrated visual scaling.
    """

    airsim_module = _make_airsim_module()
    specs = create_ship().specifications

    controls = map_framework_actuator_state(
        airsim_module,
        specs,
        rudder_angle=0.0,
        thrust=specs.max_thrust,
        throttle_gain=0.5,
    )

    assert controls.throttle == pytest.approx(0.5, abs=1e-6)
