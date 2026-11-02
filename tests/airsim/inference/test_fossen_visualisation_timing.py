"""
Tests for the visualization snap/interpolate timing helper.
"""

from airsim.inference.visualisation_models.model_adapters import _should_snap_visual_update


def test_should_snap_visual_update_when_duration_is_shorter_than_render_interval():
    """
    Snap when the motion is shorter than one render interval.
    """

    assert _should_snap_visual_update(total_duration=0.005, update_interval_s=1.0/90.0, interpolation_steps=1) is True


def test_should_not_snap_visual_update_when_duration_exceeds_render_interval():
    """
    Interpolate when the motion spans several render intervals.
    """

    assert _should_snap_visual_update(total_duration=0.030, update_interval_s=1.0/90.0, interpolation_steps=3) is False


def test_should_snap_visual_update_for_near_zero_duration_even_without_render_limit():
    """
    Snap near-zero durations regardless of the render interval.
    """

    assert _should_snap_visual_update(total_duration=0.0005, update_interval_s=0.001, interpolation_steps=1) is True
