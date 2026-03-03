"""Unit tests for the AirSim Bridge refactoring (Max 12 tests)."""

import pytest
from unittest.mock import patch
import numpy as np

from airsim.config.airsim_config import AirSimConfig
from airsim.communication.airsim_bridge import AirSimBridge, is_ue5_reachable


@pytest.fixture
def mock_car_client():
    from unittest.mock import MagicMock
    client_mock = MagicMock()
    # Ensure client.getCarState() works for health check
    state_mock = MagicMock()
    state_mock.speed = 0.0
    state_mock.gear = 1
    state_mock.timestamp = 123456789
    client_mock.getCarState.return_value = state_mock
    return client_mock

@pytest.fixture
def mock_cosysairsim_module(mock_car_client):
    from unittest.mock import MagicMock
    import sys
    
    mock_module = MagicMock()
    mock_module.CarClient = MagicMock(return_value=mock_car_client)
    mock_module.CarControls = MagicMock()
    
    sys.modules["cosysairsim"] = mock_module
    yield mock_module
    if "cosysairsim" in sys.modules:
        del sys.modules["cosysairsim"]


# ── T-003: API-parameter validatie & foutmeldingen ──

def test_config_valid_defaults():
    """Test 1: Check if the default IP and ports are properly assigned without PID values."""
    cfg = AirSimConfig()
    assert cfg.port == 41451
    assert cfg.timeout == 10.0
    assert cfg.max_retries == 3
    assert not hasattr(cfg, "airsim_kp")  # PID values should be removed

def test_config_invalid_port():
    """Test 2: Check if invalid ports throw ValueError."""
    with pytest.raises(ValueError, match="Port must be an integer in range"):
        AirSimConfig(port=-1)

def test_config_invalid_ip():
    """Test 3: Check if bad IP formats throw ValueError."""
    with pytest.raises(ValueError, match="Invalid IP address"):
        AirSimConfig(ip="invalid_address")


# ── T-004: UE5 detectie ──

@patch("socket.create_connection")
def test_ue5_reachable(mock_socket):
    """Test 4: TCP probe returns True when connection succeeds."""
    assert is_ue5_reachable("127.0.0.1", 41451) is True

@patch("socket.create_connection", side_effect=ConnectionRefusedError)
def test_ue5_unreachable(mock_socket):
    """Test 5: TCP probe returns False when connection is refused."""
    assert is_ue5_reachable("127.0.0.1", 41451) is False

@patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=False)
def test_bridge_init_ue5_unreachable(mock_reachable):
    """Test 6: Bridge disables itself immediately if UE5 is unreachable."""
    bridge = AirSimBridge()
    assert bridge.is_connected is False
    mock_reachable.assert_called_once()


# ── T-001: Retry-logica & health-check ──

@patch("airsim.communication.airsim_bridge.time.sleep")
@patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=True)
def test_bridge_retry_success(_, mock_sleep, mock_cosysairsim_module, mock_car_client):
    """Test 7: Bridge successfully connects after a failure via exponential backoff."""
    mock_car_client.confirmConnection.side_effect = [ConnectionError("Fail 1"), None]
    bridge = AirSimBridge(AirSimConfig(max_retries=1))
    
    assert bridge.is_connected is True
    assert mock_car_client.confirmConnection.call_count == 2
    mock_sleep.assert_called_once_with(1)  # 2^0 = 1s sleep

@patch("airsim.communication.airsim_bridge.time.sleep")
@patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=True)
def test_bridge_retry_exhausted(_, mock_sleep, mock_cosysairsim_module, mock_car_client):
    """Test 8: Bridge disables itself if all connection retries fail."""
    mock_car_client.confirmConnection.side_effect = Exception("Fail")
    bridge = AirSimBridge(AirSimConfig(max_retries=2))
    
    assert bridge.is_connected is False
    assert mock_car_client.confirmConnection.call_count == 3
    assert mock_sleep.call_count == 2


class TestBridgeHealthAndControl:
    @pytest.fixture(autouse=True)
    def setup_bridge(self, mock_cosysairsim_module, mock_car_client):
        with patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=True):
            self.mock_client = mock_car_client
            self.bridge = AirSimBridge()
            self.mock_client.reset_mock()
            yield

    def test_health_check_success(self):
        """Test 9: Health check returns True and keeps bridge connected."""
        assert self.bridge.health_check() is True
        self.mock_client.getCarState.assert_called_once()
        assert self.bridge.is_connected is True

    def test_health_check_failure(self):
        """Test 10: Health check disables bridge on failure."""
        self.mock_client.getCarState.side_effect = Exception("Disconnected")
        assert self.bridge.health_check() is False
        assert self.bridge.is_connected is False

    def test_send_action(self):
        """Test 11: send_action passes raw bounds directly (T-001 pass-through constraint)."""
        action = np.array([-0.5, 0.8])
        self.bridge.send_action(action)

        controls = self.bridge._controls
        assert controls.steering == pytest.approx(-0.5)
        assert controls.throttle == pytest.approx(0.8)
        assert controls.brake == pytest.approx(0.0)
        self.mock_client.setCarControls.assert_called_once()

# ── T-005: AirSimEnv & AirSimBridge Integratie ──

def test_client_property_exposure(mock_cosysairsim_module, mock_car_client):
    """Test 12: Ensure AirSimEnv can safely access the underlying UE5 client."""
    with patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=True):
        bridge = AirSimBridge()
        # Ensure it works
        client = bridge.client
        assert client is mock_car_client
        
        # Test failure state
        bridge._connected = False
        with pytest.raises(ConnectionError, match="Cannot access AirSim client"):
            _ = bridge.client
