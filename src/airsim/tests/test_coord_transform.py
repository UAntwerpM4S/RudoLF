"""Unit tests for airsim.communication.coord_transform."""

import math
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from airsim.communication.coord_transform import CoordTransform, _normalize_angle


# ── _normalize_angle ──────────────────────────────────────

class TestNormalizeAngle:
    """Tests for the _normalize_angle helper."""

    def test_zero(self):
        assert _normalize_angle(0.0) == pytest.approx(0.0)

    def test_pi(self):
        # π should stay at π (or wrap to -π — both are valid boundaries)
        result = _normalize_angle(math.pi)
        assert abs(result) == pytest.approx(math.pi, abs=1e-12)

    def test_large_positive(self):
        result = _normalize_angle(3 * math.pi)
        assert -math.pi <= result <= math.pi
        assert abs(result) == pytest.approx(math.pi, abs=1e-12)

    def test_large_negative(self):
        result = _normalize_angle(-3 * math.pi)
        assert -math.pi <= result <= math.pi

    def test_small_positive(self):
        assert _normalize_angle(0.1) == pytest.approx(0.1)

    def test_two_pi(self):
        result = _normalize_angle(2 * math.pi)
        assert result == pytest.approx(0.0, abs=1e-12)


# ── CoordTransform ────────────────────────────────────────

class TestCoordTransformIdentity:
    """Tests for the identity / axis-swap transform."""

    def test_identity_round_trip(self):
        ct = CoordTransform.identity(fossen_origin=(100.0, 200.0))
        fx, fy = 500.0, 600.0
        ax, ay, _ = ct.fossen_to_airsim(fx, fy)
        fx2, fy2 = ct.airsim_to_fossen(ax, ay)
        assert fx2 == pytest.approx(fx, abs=1e-6)
        assert fy2 == pytest.approx(fy, abs=1e-6)

    def test_origin_maps_correctly(self):
        ct = CoordTransform.identity(fossen_origin=(0.0, 0.0))
        ax, ay, _ = ct.fossen_to_airsim(0.0, 0.0)
        # At origin, should map to airsim origin (0, 0)
        assert ax == pytest.approx(0.0, abs=1e-6)
        assert ay == pytest.approx(0.0, abs=1e-6)


class TestCoordTransformCalibrated:
    """Tests with a realistic calibration dict."""

    @pytest.fixture
    def calibration(self):
        return {
            "scale": 1.0,
            "rotation": 0.0,
            "mirror_x": False,
            "mirror_y": True,
            "fossen_origin": [2245.0, -47.0],
            "airsim_origin": [-1000.0, 6000.0],
        }

    def test_round_trip(self, calibration):
        ct = CoordTransform(calibration)
        for fx, fy in [(2245.0, -47.0), (5000.0, 3000.0), (-1000.0, 10000.0)]:
            ax, ay, _ = ct.fossen_to_airsim(fx, fy)
            fx2, fy2 = ct.airsim_to_fossen(ax, ay)
            assert fx2 == pytest.approx(fx, abs=1e-6), f"Failed for ({fx}, {fy})"
            assert fy2 == pytest.approx(fy, abs=1e-6), f"Failed for ({fx}, {fy})"

    def test_heading_round_trip(self, calibration):
        ct = CoordTransform(calibration)
        for heading in [0.0, math.pi / 4, -math.pi / 3, math.pi]:
            yaw = ct.fossen_heading_to_airsim_yaw(heading)
            heading2 = ct.airsim_yaw_to_fossen_heading(yaw)
            assert heading2 == pytest.approx(
                _normalize_angle(heading), abs=1e-6
            ), f"Failed for heading={heading}"

    def test_mirror_x_round_trip(self):
        cal = {
            "scale": 1.0,
            "rotation": 0.0,
            "mirror_x": True,
            "mirror_y": False,
            "fossen_origin": [0.0, 0.0],
            "airsim_origin": [0.0, 0.0],
        }
        ct = CoordTransform(cal)
        ax, ay, _ = ct.fossen_to_airsim(100.0, 200.0)
        fx, fy = ct.airsim_to_fossen(ax, ay)
        assert fx == pytest.approx(100.0, abs=1e-6)
        assert fy == pytest.approx(200.0, abs=1e-6)

    def test_2d_convenience(self, calibration):
        ct = CoordTransform(calibration)
        ax3, ay3, _ = ct.fossen_to_airsim(3000.0, 1000.0)
        ax2, ay2 = ct.fossen_to_airsim_2d(3000.0, 1000.0)
        assert ax2 == pytest.approx(ax3)
        assert ay2 == pytest.approx(ay3)


class TestCoordTransformLoad:
    """Tests for loading calibration from JSON."""

    def test_load_from_file(self, tmp_path):
        cal = {
            "scale": 0.999,
            "rotation": 0.02,
            "mirror_x": False,
            "mirror_y": True,
            "fossen_origin": [100.0, 200.0],
            "airsim_origin": [300.0, 400.0],
        }
        path = tmp_path / "test_cal.json"
        path.write_text(json.dumps(cal))

        ct = CoordTransform.load(path)
        assert ct._scale == pytest.approx(0.999)
        assert ct._mirror_y is True

    def test_default_no_calibration(self):
        ct = CoordTransform()
        assert ct._scale == pytest.approx(1.0)
        assert ct._rotation == pytest.approx(-math.pi / 2)
