"""
Bank-distance behaviour and configuration defaults of fw.dynamics.ugent_surrogate (Tripoli, full scale).
"""

import math

import pytest

from fw.dynamics.ugent_surrogate import (
    DEFAULT_MAX_BANK_CLEARANCE, DEFAULT_MIN_BANK_CLEARANCE, DEFAULT_N_MAX_RPM, UGentSurrogate)
from fw.ships.ship_factory import ShipModel, create_ship
from fw.ships.vessel_state import VesselState

BEAM = 17.1  # Tripoli beam [m]


@pytest.fixture
def spec():
    """
    Specifications of the Tripoli preset.
    """

    return create_ship(ShipModel.TRIPOLI).specifications


def _state(u=4.0):
    """
    Straight running at surge u [m/s], no sway or yaw.
    """

    return VesselState(terminated=False, truncated=False, x=0.0, y=0.0, heading=0.0, u=u, v=0.0, r=0.0)


def _dist(y_nondim):
    """
    Bank distance from midship [m] for a non-dimensional clearance y' = (d - B/2) / B.
    """

    return BEAM / 2.0 + y_nondim * BEAM


def _sway(model, y_port, y_stbd):
    """
    Sway acceleration [m/s^2] with banks at the given y' (None = no bank), zero rudder and thrust.
    """

    model.set_bank_distances(None if y_port is None else _dist(y_port),
                             None if y_stbd is None else _dist(y_stbd))
    return model._compute_accelerations(_state(), 0.0, 0.0)[1]


def test_defaults(spec):
    """
    n_max 320 rpm, banks held at y' = 1.0 below it, no upper cut-off.
    """

    model = UGentSurrogate(spec, {"ukc_percent": 125})
    assert DEFAULT_N_MAX_RPM == 320.0 and model.n_max_rpm == DEFAULT_N_MAX_RPM
    assert DEFAULT_MIN_BANK_CLEARANCE == 1.0 and model.min_bank_clearance == 1.0
    assert DEFAULT_MAX_BANK_CLEARANCE is None and math.isinf(model.max_bank_clearance)


def test_far_bank_acts_without_cut_off(spec):
    """
    A bank at y' = 10 still acts by default (both banks count).
    """

    model = UGentSurrogate(spec, {"ukc_percent": 125})
    assert _sway(model, 10.0, None) != _sway(model, None, None)
    assert _sway(model, 10.0, None) == pytest.approx(_sway(model, 1.0, None) / 10.0)


def test_no_jump_when_far_bank_crosses_four(spec):
    """
    Without cut-off the force is continuous at y' = 4; with the old cut-off 4.0 it jumps.
    """

    default = UGentSurrogate(spec, {"ukc_percent": 125})
    assert _sway(default, 2.0, 3.999) == pytest.approx(_sway(default, 2.0, 4.001), rel=1e-3)

    cut = UGentSurrogate(spec, {"ukc_percent": 125, "max_bank_clearance": 4.0})
    jump = abs(_sway(cut, 2.0, 4.001) - _sway(cut, 2.0, 3.999))
    assert jump == pytest.approx(abs(_sway(cut, 1.0, None)) / 4.0, rel=1e-2)


def test_optional_cut_off(spec):
    """
    max_bank_clearance ignores banks farther than that y'.
    """

    model = UGentSurrogate(spec, {"ukc_percent": 125, "max_bank_clearance": 4.0})
    assert _sway(model, 10.0, 10.0) == _sway(model, None, None)
    assert _sway(model, 2.0, None) != _sway(model, None, None)


def test_close_bank_held_at_min_clearance(spec):
    """
    Banks closer than y' = 1.0 act as at y' = 1.0 (default).
    """

    model = UGentSurrogate(spec, {"ukc_percent": 125})
    assert _sway(model, 0.3, None) == pytest.approx(_sway(model, 1.0, None))


def test_bank_attracts_towards_port_bank(spec):
    """
    A port bank pulls the ship to port (framework v positive to port).
    """

    model = UGentSurrogate(spec, {"ukc_percent": 125})
    assert _sway(model, 2.0, None) > 0.0


@pytest.mark.parametrize("config", [
    {"ukc_percent": 125, "max_bank_clearance": 1.0},
    {"ukc_percent": 125, "max_bank_clearance": "4"},
    {"ukc_percent": 125, "max_bank_clearance": float("nan")},
    {"ukc_percent": 125, "min_bank_clearance": 2.0, "max_bank_clearance": 1.5},
    {"ukc_percent": 125, "n_max_rpm": 0},
])
def test_invalid_bank_config_raises(spec, config):
    """
    Invalid limits raise ValueError; max must exceed min.
    """

    with pytest.raises(ValueError):
        UGentSurrogate(spec, config)


def test_explicit_none_cut_off(spec):
    """
    max_bank_clearance None is accepted and means no cut-off.
    """

    assert math.isinf(UGentSurrogate(spec, {"ukc_percent": 125, "max_bank_clearance": None}).max_bank_clearance)
