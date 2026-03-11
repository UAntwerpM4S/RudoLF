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
    
    kinematics_mock = MagicMock()
    kinematics_mock.position.x_val = 1.0
    kinematics_mock.position.y_val = 2.0
    kinematics_mock.position.z_val = 3.0
    kinematics_mock.linear_velocity.x_val = 4.0
    kinematics_mock.linear_velocity.y_val = 5.0
    kinematics_mock.linear_velocity.z_val = 6.0
    kinematics_mock.angular_velocity.x_val = 0.1
    kinematics_mock.angular_velocity.y_val = 0.2
    kinematics_mock.angular_velocity.z_val = 0.3
    kinematics_mock.orientation.w_val = 0.7071
    kinematics_mock.orientation.x_val = 0.0
    kinematics_mock.orientation.y_val = 0.0
    kinematics_mock.orientation.z_val = 0.7071
    state_mock.kinematics_estimated = kinematics_mock
    
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
        with patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=True), patch(
            "airsim.communication.airsim_bridge.time.sleep",
            return_value=None,
        ):
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
        self.mock_client.confirmConnection.side_effect = Exception("Reconnection Fail") # To prevent successful reconnect
        assert self.bridge.health_check() is False
        assert self.bridge.is_connected is False

    def test_set_controls(self, mock_cosysairsim_module):
        """Test 11: set_controls forwards predefined CarControls (T-001 pass-through constraint modified to pure middleware)."""
        controls = mock_cosysairsim_module.CarControls()
        controls.steering = -0.5
        controls.throttle = 0.8
        
        self.bridge.set_controls(controls)

        assert self.bridge._controls is controls
        self.mock_client.setCarControls.assert_called_once_with(controls)

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

# ── Extension Tests (T-005 - T-008) ──

@patch("airsim.communication.airsim_bridge.AirSimBridge._connect_with_retry")
@patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=True)
def test_health_check_reconnect(mock_reachable, mock_connect, mock_cosysairsim_module, mock_car_client):
    """T-005: Verify health_check triggers reconnect and cleans up when connection fails."""
    mock_car_client.getCarState.side_effect = Exception("Connection lost")
    bridge = AirSimBridge(AirSimConfig(max_retries=1))
    
    # Fake being connected so health_check can try
    bridge._connected = True
    
    # Act
    mock_connect.reset_mock()
    bridge.health_check()
    
    # Assert
    assert mock_connect.call_count == 1
    assert bridge._client is None  # Should be cleared before reconnect
    assert bridge._controls is None

def test_disconnect_behavior(mock_cosysairsim_module, mock_car_client):
    """T-005 & T-008: Disconnect clears state, sends brakes, and flushes CSV."""
    with patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=True):
        bridge = AirSimBridge(AirSimConfig(csv_log_path="dummy.csv"))
        bridge._sensor_buffer.append({"dummy": 1})
        
        with patch("airsim.communication.airsim_bridge.AirSimBridge._flush_csv") as mock_flush:
            bridge.disconnect()
            
            # Assert flush was called
            mock_flush.assert_called_once()
            
            # Assert brakes were applied
            args, kwargs = mock_car_client.setCarControls.call_args
            controls = args[0]
            assert controls.throttle == 0.0
            assert controls.brake == 1.0
            assert bridge._connected is False

def test_get_state_telemetry_and_yaw(mock_cosysairsim_module, mock_car_client):
    """T-007: get_state returns full telemetry and valid yaw from quaternion."""
    with patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=True):
        bridge = AirSimBridge()
        state = bridge.get_state()
        
        # Validate positions
        assert state["pos_x"] == 1.0
        assert state["pos_y"] == 2.0
        assert state["pos_z"] == 3.0
        
        # Validate yaw (w=0.7071, z=0.7071 -> ~90 degrees or pi/2 radians)
        assert state["yaw"] == pytest.approx(1.5707, abs=0.01)

@patch("os.path.isfile", return_value=False)
@patch("builtins.open")
def test_csv_logging_buffer(mock_open, mock_isfile, mock_cosysairsim_module, mock_car_client):
    """T-008: Verify buffered data logic flushes to CSV."""
    with patch("airsim.communication.airsim_bridge.is_ue5_reachable", return_value=True):
        bridge = AirSimBridge(AirSimConfig(csv_log_path="test_log.csv"))
        import csv
        with patch("csv.DictWriter") as mock_writer_cls:
            mock_writer_inst = mock_writer_cls.return_value
            
            # Add one item to buffer
            bridge._sensor_buffer.append({"a": 1, "b": 2})
            
            # Flush
            bridge._flush_csv()
            
            # Assert empty buffer and file opened
            assert len(bridge._sensor_buffer) == 0
            mock_open.assert_called_once_with("test_log.csv", mode='a', newline='')
            mock_writer_inst.writeheader.assert_called_once()
            mock_writer_inst.writerows.assert_called_once_with([{"a": 1, "b": 2}])
