"""Pure timing helper tests for framework visualisation smoothing."""

from airsim.inference.visualisation_models.model_adapters import _should_snap_visual_update


def test_should_snap_visual_update_when_duration_is_shorter_than_render_interval():
    """Test behavior for should snap visual update when duration is shorter than render interval.
    """
    assert _should_snap_visual_update(total_duration=0.005, min_render_hz=90.0) is True


def test_should_not_snap_visual_update_when_duration_exceeds_render_interval():
    """Test behavior for should not snap visual update when duration exceeds render interval.
    """
    assert _should_snap_visual_update(total_duration=0.030, min_render_hz=90.0) is False


def test_should_snap_visual_update_for_near_zero_duration_even_without_render_limit():
    """Test behavior for should snap visual update for near zero duration even without render limit.
    """
    assert _should_snap_visual_update(total_duration=0.0005, min_render_hz=0.0) is True
