import types

import numpy as np
import pytest

from airsim.communication.control_mapping import (
    map_fossen_policy_action,
    validate_policy_action,
)
from fw.ships.ship_factory import create_ship


def _make_airsim_module():
    module = types.SimpleNamespace()

    class CarControls:
        def __init__(self):
            self.steering = 0.0
            self.throttle = 0.0
            self.brake = 0.0
            self.is_manual_gear = False
            self.manual_gear = 0
            self.gear_immediate = False

    module.CarControls = CarControls
    return module


def test_validate_policy_action_clips_small_numeric_overshoot():
    action = validate_policy_action(np.array([1.0 + 1e-6, -1.0 - 1e-6], dtype=np.float32))
    np.testing.assert_allclose(action, np.array([1.0, -1.0], dtype=np.float32))


def test_validate_policy_action_rejects_large_out_of_bounds():
    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        validate_policy_action(np.array([0.0, 1.2], dtype=np.float32))


def test_validate_policy_action_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite values"):
        validate_policy_action(np.array([0.0, np.nan], dtype=np.float32))


def test_map_fossen_policy_action_normalizes_to_ue5_throttle_range():
    airsim_module = _make_airsim_module()
    specs = create_ship().specifications

    controls = map_fossen_policy_action(
        airsim_module,
        specs,
        steering=0.0,
        signed_throttle_action=0.0,
    )

    assert 0.0 <= controls.throttle <= 1.0
    assert controls.throttle == pytest.approx(0.5, abs=1e-6)
