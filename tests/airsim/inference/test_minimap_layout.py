"""Pure-logic tests for minimap placement and follow-view behaviour."""

import numpy as np

from airsim.inference.minimap_fossen_airsim import (
    _estimate_view_half_span,
    _preferred_view_center,
    _resolve_minimap_pixels,
    _resolve_minimap_position,
    _should_shift_view,
)


def _make_snapshot(checkpoints=None, target_pos=(3000.0, 0.0)):
    """ make snapshot.

    Args:
        checkpoints: Description.
        target_pos: Description.
    """
    return {
        "ship_pos_0": np.array([0.0, 0.0], dtype=np.float64),
        "target_pos": np.array(target_pos, dtype=np.float64),
        "checkpoints": checkpoints or [],
    }


def test_resolve_minimap_pixels_uses_ue5_monitor_geometry():
    """Test behavior for resolve minimap pixels uses ue5 monitor geometry.
    """
    context = {
        "window_rect": (100, 100, 2000, 1200),
        "work_rect": (0, 0, 2560, 1400),
    }

    assert _resolve_minimap_pixels(context) == 360


def test_resolve_minimap_position_docks_to_top_right_of_work_area():
    """Test behavior for resolve minimap position docks to top right of work area.
    """
    context = {
        "window_rect": (2000, 100, 1800, 1000),
        "work_rect": (1920, 0, 1920, 1040),
    }

    assert _resolve_minimap_position(context, minimap_px=480, inset=20) == (3340, 20)


def test_estimate_view_half_span_prefers_waypoint_spacing():
    """Test behavior for estimate view half span prefers waypoint spacing.
    """
    snap = _make_snapshot(
        checkpoints=[
            {"pos": [0.0, 0.0]},
            {"pos": [500.0, 0.0]},
            {"pos": [1000.0, 0.0]},
        ]
    )

    assert _estimate_view_half_span(snap) == 1000.0


def test_estimate_view_half_span_falls_back_to_target_distance():
    """Test behavior for estimate view half span falls back to target distance.
    """
    snap = _make_snapshot(checkpoints=[] , target_pos=(8000.0, 0.0))

    assert _estimate_view_half_span(snap) == np.float64(8000.0 / 9.0)


def test_preferred_view_center_looks_ahead_on_route():
    """Test behavior for preferred view center looks ahead on route.
    """
    center = _preferred_view_center(
        np.array([0.0, 0.0], dtype=np.float64),
        checkpoints=[
            {"pos": [0.0, 0.0]},
            {"pos": [250.0, 0.0]},
            {"pos": [500.0, 0.0]},
            {"pos": [750.0, 0.0]},
        ],
        half_span=900.0,
    )

    assert center[0] > 0.0
    assert center[0] == 90.0
    assert center[1] == 0.0


def test_should_shift_view_when_ship_nears_view_edge():
    """Test behavior for should shift view when ship nears view edge.
    """
    center = np.array([0.0, 0.0], dtype=np.float64)

    assert _should_shift_view(np.array([600.0, 0.0]), center, 1000.0) is True
    assert _should_shift_view(np.array([400.0, 0.0]), center, 1000.0) is False

