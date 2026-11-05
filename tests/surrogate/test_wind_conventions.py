"""
fw.dynamics.uantwerp_surrogate.apparent_wind_for_state: re-expressing the
apparent wind recorded on one ship for another ship (full_env lookups).

Convention (FH WindState, as stored in the datasets): the direction the wind
comes FROM, relative to the bow, positive = from starboard. Sway is positive
to PORT, heading counter-clockwise from East.
"""

import pytest
import numpy as np

from fw.dynamics.uantwerp_surrogate import apparent_wind_for_state


def _wind(app_speed, app_dir, rec, own):
    """
    apparent_wind_for_state with (heading, u, v) tuples for the recording and own ship.
    """

    return apparent_wind_for_state(app_speed, app_dir, *rec, *own)


def test_identical_ships_give_the_same_wind():
    """
    Same heading and velocity: the apparent wind is unchanged.
    """

    for app_dir in (-2.5, -0.3, 0.0, 1.0, 3.0):
        speed, direction = _wind(7.0, app_dir, (0.4, 3.0, -0.2), (0.4, 3.0, -0.2))
        assert speed == pytest.approx(7.0, abs=1e-9)
        assert np.cos(direction - app_dir) == pytest.approx(1.0, abs=1e-9)


def test_calm_air_gives_head_wind_from_own_surge():
    """
    No true wind (the recording ship was stopped in calm air): moving ahead makes a head wind = surge.
    """

    speed, direction = _wind(0.0, 0.0, (1.2, 0.0, 0.0), (-0.7, 4.0, 0.0))
    assert speed == pytest.approx(4.0, abs=1e-9)
    assert direction == pytest.approx(0.0, abs=1e-9)


def test_calm_air_sway_to_port_gives_wind_from_port():
    """
    No true wind, own sway to PORT (v > 0): the apparent wind comes from port (negative direction).
    """

    speed, direction = _wind(0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.5))
    assert speed == pytest.approx(1.5, abs=1e-9)
    assert direction == pytest.approx(-np.pi / 2, abs=1e-9)


def test_stopped_recorder_measured_true_wind():
    """
    A stopped recording ship measures the true wind; a stopped ship turned 90 deg to port sees it rotated.
    """

    # Recorder heading East, wind from starboard (i.e. from the South, blowing North).
    speed, direction = _wind(5.0, np.pi / 2, (0.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0))
    # Own ship heading North: a wind from the South comes from astern.
    assert speed == pytest.approx(5.0, abs=1e-9)
    assert abs(direction) == pytest.approx(np.pi, abs=1e-9)


def test_rotating_both_ships_changes_nothing():
    """
    Only the relative geometry matters: rotating both headings by the same angle keeps the result.
    """

    a = _wind(6.0, 0.8, (0.3, 2.0, 0.1), (1.1, 3.0, -0.4))
    b = _wind(6.0, 0.8, (0.3 + 2.0, 2.0, 0.1), (1.1 + 2.0, 3.0, -0.4))
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    assert np.cos(a[1] - b[1]) == pytest.approx(1.0, abs=1e-9)


def test_direction_range():
    """
    The returned direction lies in [-pi, pi).
    """

    rng = np.random.default_rng(0)
    for _ in range(200):
        rec = (rng.uniform(-4, 4), rng.uniform(0, 5), rng.normal())
        own = (rng.uniform(-4, 4), rng.uniform(0, 5), rng.normal())
        _, direction = _wind(rng.uniform(0, 20), rng.uniform(-np.pi, np.pi), rec, own)
        assert -np.pi <= direction < np.pi
