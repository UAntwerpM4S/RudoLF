"""Unit tests for airsim.communication.airsim_bridge."""

import math
import pytest

from airsim.communication.airsim_bridge import (
    _normalize_angle,
    _yaw_from_quaternion,
    AirSimBridge,
)
from airsim.tests.conftest import MockQuaternionr


# ── _normalize_angle ──────────────────────────────────────

class TestNormalizeAngle:
    """Tests for the bridge's _normalize_angle helper."""

    def test_zero(self):
        assert _normalize_angle(0.0) == pytest.approx(0.0)

    def test_positive_wrap(self):
        result = _normalize_angle(math.pi + 0.1)
        assert -math.pi <= result <= math.pi
        assert result == pytest.approx(-math.pi + 0.1, abs=1e-10)

    def test_negative_wrap(self):
        result = _normalize_angle(-math.pi - 0.1)
        assert -math.pi <= result <= math.pi
        assert result == pytest.approx(math.pi - 0.1, abs=1e-10)

    def test_two_pi(self):
        result = _normalize_angle(2 * math.pi)
        assert result == pytest.approx(0.0, abs=1e-12)


# ── _yaw_from_quaternion ─────────────────────────────────

class TestYawFromQuaternion:
    """Tests for quaternion → yaw extraction."""

    def test_identity_quaternion(self):
        """Identity quaternion (w=1, x=y=z=0) should give yaw=0."""
        q = MockQuaternionr(w_val=1.0, x_val=0.0, y_val=0.0, z_val=0.0)
        assert _yaw_from_quaternion(q) == pytest.approx(0.0, abs=1e-10)

    def test_90_degree_yaw(self):
        """A 90° yaw rotation."""
        yaw = math.pi / 2
        q = MockQuaternionr.from_yaw(yaw)
        assert _yaw_from_quaternion(q) == pytest.approx(yaw, abs=1e-10)

    def test_negative_yaw(self):
        """A -45° yaw rotation."""
        yaw = -math.pi / 4
        q = MockQuaternionr.from_yaw(yaw)
        assert _yaw_from_quaternion(q) == pytest.approx(yaw, abs=1e-10)

    def test_180_degree_yaw(self):
        """A 180° yaw rotation."""
        yaw = math.pi
        q = MockQuaternionr.from_yaw(yaw)
        result = _yaw_from_quaternion(q)
        # π and -π are equivalent
        assert abs(result) == pytest.approx(math.pi, abs=1e-10)


# ── AirSimBridge ──────────────────────────────────────────

class TestAirSimBridgeInit:
    """Tests for AirSimBridge initialization without cosysairsim."""

    def test_disconnected_without_cosysairsim(self):
        """Bridge should gracefully start in disconnected state
        when cosysairsim is not importable."""
        bridge = AirSimBridge(ip="192.0.2.1", port=99999)
        assert bridge.is_connected is False

    def test_get_state_when_disconnected(self):
        bridge = AirSimBridge(ip="192.0.2.1", port=99999)
        assert bridge.get_state() == {}

    def test_send_action_when_disconnected(self):
        """send_action should be a no-op when not connected."""
        import numpy as np
        bridge = AirSimBridge(ip="192.0.2.1", port=99999)
        # Should not raise
        bridge.send_action(np.array([0.0, 0.5]), ship_speed=1.0)

    def test_reset_when_disconnected(self):
        bridge = AirSimBridge(ip="192.0.2.1", port=99999)
        # Should not raise
        bridge.reset()

    def test_disconnect_when_disconnected(self):
        bridge = AirSimBridge(ip="192.0.2.1", port=99999)
        # Should not raise
        bridge.disconnect()
