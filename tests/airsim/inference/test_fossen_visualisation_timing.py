"""Pure timing helper tests for Fossen visualisation smoothing."""

from airsim.inference.visualisation_models.run_visualise_fossen import _should_snap_visual_update


def test_should_snap_visual_update_when_duration_is_shorter_than_render_interval():
    assert _should_snap_visual_update(total_duration=0.005, min_render_hz=90.0) is True


def test_should_not_snap_visual_update_when_duration_exceeds_render_interval():
    assert _should_snap_visual_update(total_duration=0.030, min_render_hz=90.0) is False


def test_should_snap_visual_update_for_near_zero_duration_even_without_render_limit():
    assert _should_snap_visual_update(total_duration=0.0005, min_render_hz=0.0) is True