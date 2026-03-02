"""Unit tests for airsim.config.airsim_config."""

import pytest

from airsim.config.airsim_config import get_airsim_config, AIRSIM_CONFIG


class TestGetAirsimConfig:
    """Tests for the get_airsim_config function."""

    def test_returns_dict(self):
        cfg = get_airsim_config()
        assert isinstance(cfg, dict)

    def test_returns_copy(self):
        """Modifying the returned dict should not affect the original."""
        cfg = get_airsim_config()
        cfg["airsim_port"] = 99999
        assert AIRSIM_CONFIG["airsim_port"] == 41451

    def test_has_expected_keys(self):
        cfg = get_airsim_config()
        expected = [
            "use_airsim",
            "airsim_ip",
            "airsim_port",
            "airsim_kp",
            "airsim_ki",
            "airsim_kd",
            "airsim_ff",
            "airsim_lat_gain",
            "airsim_speed_scale",
            "airsim_throttle_kp",
            "airsim_throttle_ki",
            "airsim_dist_gain",
            "airsim_dist_ki",
            "airsim_log_telemetry",
        ]
        for key in expected:
            assert key in cfg, f"Missing key: {key}"

    def test_port_is_positive(self):
        cfg = get_airsim_config()
        assert cfg["airsim_port"] > 0

    def test_gains_are_non_negative(self):
        cfg = get_airsim_config()
        for key in ["airsim_kp", "airsim_ki", "airsim_kd",
                     "airsim_throttle_kp", "airsim_throttle_ki"]:
            assert cfg[key] >= 0, f"{key} should be non-negative"

    def test_speed_scale_positive(self):
        cfg = get_airsim_config()
        assert cfg["airsim_speed_scale"] > 0
