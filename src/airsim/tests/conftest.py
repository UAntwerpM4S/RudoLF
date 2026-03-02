"""Shared pytest fixtures for the AirSim test suite.

Provides mock objects that replace the ``cosysairsim`` module so tests
can run without a live UE5/AirSim instance.
"""

import math
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace


class MockQuaternionr:
    """Minimal mock for ``cosysairsim.Quaternionr``."""

    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
        self.w_val = w_val

    @classmethod
    def from_yaw(cls, yaw: float) -> "MockQuaternionr":
        """Create a quaternion representing a pure yaw rotation."""
        return cls(
            x_val=0.0,
            y_val=0.0,
            z_val=math.sin(yaw / 2),
            w_val=math.cos(yaw / 2),
        )


class MockPose:
    """Minimal mock for ``cosysairsim.Pose``."""

    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.position = SimpleNamespace(x_val=x, y_val=y, z_val=z)
        self.orientation = MockQuaternionr.from_yaw(yaw)


class MockCarState:
    """Minimal mock for ``cosysairsim.CarState``."""

    def __init__(self, speed=0.0, gear=1, timestamp=0):
        self.speed = speed
        self.gear = gear
        self.timestamp = timestamp


@pytest.fixture
def mock_car_client():
    """Return a MagicMock simulating ``cosysairsim.CarClient``."""
    client = MagicMock()
    client.confirmConnection.return_value = True
    client.enableApiControl.return_value = None
    client.simGetVehiclePose.return_value = MockPose()
    client.getCarState.return_value = MockCarState()
    client.simGetCollisionInfo.return_value = SimpleNamespace(has_collided=False)
    return client
