"""
Regression tests for the review fixes on the UAntwerp surrogate integration:
lookup CSV layout check, position-aware get_ukc, non-finite input guard and
the ShipSimulator sub-step grid.
"""

import numpy as np
import pytest

from fw.ships.ship_factory import ShipModel, create_ship
from fw.surrogate.schema import get_scenario

from surrogate_test_helpers import make_state

CONFINED = get_scenario("confined_ukc")
FULL = get_scenario("full_env")


def _csv(path, scenario, points):
    """A lookup CSV in `scenario`'s dataset layout (raw row, 3 targets, episode_id)."""

    rows = np.zeros((len(points), scenario.raw_input_dim + 4))
    for i, (x, y, ukc) in enumerate(points):
        rows[i, scenario.col("x")], rows[i, scenario.col("y")], rows[i, scenario.col("ukc")] = x, y, ukc
    rows[:, -1] = -1
    np.savetxt(path, rows, delimiter=",", fmt="%.6g")
    return path


def _confined(tiny_models, lookup):
    """A myzako_confined UAntwerpSurrogate on the tiny checkpoint."""

    from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate

    spec = create_ship(ShipModel.MYZAKO_37).specifications
    return UAntwerpSurrogate(spec, {"preset": "myzako_confined", "checkpoint": str(tiny_models["confined_ukc"]),
                                    "lookup_csv": str(lookup)}, ship_model=ShipModel.MYZAKO_37)


def test_lookup_with_wrong_layout_is_refused(tiny_models, tmp_path):
    """A full_env-layout CSV given to a confined_ukc preset is refused (layout identified by width)."""

    wrong = _csv(tmp_path / "full.csv", FULL, [(0.0, 0.0, 5.0)])
    with pytest.raises(ValueError, match="full_env"):
        _confined(tiny_models, wrong)


def test_get_ukc_uses_the_position(tiny_models, tmp_path):
    """get_ukc(pos) looks up pos instead of returning the last model input."""

    model = _confined(tiny_models, _csv(tmp_path / "lk.csv", CONFINED,
                                        [(0.0, 0.0, 5.0), (100.0, 0.0, 9.0)]))
    model.get_env(np.array([0.0, 0.0]), make_state(x=0.0, y=0.0, heading=0.0, u=1.0, v=0.0, r=0.0))
    assert model.get_ukc(np.array([100.0, 0.0])) == pytest.approx(9.0)
    assert model.get_ukc(None) == pytest.approx(5.0)          # no position: last value used


def test_non_finite_input_fails_fast(tiny_models, tmp_path):
    """A NaN in the state raises instead of silently producing NaN velocities."""

    model = _confined(tiny_models, _csv(tmp_path / "lk.csv", CONFINED, [(0.0, 0.0, 5.0)]))
    state = make_state(x=0.0, y=0.0, heading=0.0, u=1.0, v=float("nan"), r=0.0)
    with pytest.raises(FloatingPointError, match="sway"):
        model._compute_accelerations(state, 0.0, 0.5)


def test_nan_position_with_lookup_raises_the_clear_error(tiny_models, tmp_path):
    """A NaN position (velocity and position turn NaN in the same sub-step) must
    give the clear FloatingPointError, not scipy's KD-tree "'x' must be finite";
    get_ukc at a NaN position returns the default instead of raising."""

    model = _confined(tiny_models, _csv(tmp_path / "lk.csv", CONFINED, [(0.0, 0.0, 5.0)]))
    state = make_state(x=float("nan"), y=float("nan"), heading=0.0, u=1.0, v=0.0, r=0.0)
    with pytest.raises(FloatingPointError, match=r"\bx\b"):
        model._compute_accelerations(state, 0.0, 0.5)
    assert model.get_ukc(np.array([np.nan, np.nan])) == pytest.approx(model.default_ukc)


def test_ship_simulator_sub_step_grid():
    """dt is validated at construction: UA needs a multiple of 0.1 s; other models get a remainder step."""

    from fw.dynamics.fossen import Fossen
    from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate
    from fw.simulators.ship_simulator import ShipSimulator

    spec = create_ship(ShipModel.MYZAKO_37).specifications
    sim = ShipSimulator(spec, Fossen(spec), dt=0.25)
    assert sim._n_sub_steps == 2 and sim._dt_remainder == pytest.approx(0.05)
    sim = ShipSimulator(spec, Fossen(spec), dt=2.0)
    assert sim._n_sub_steps == 20 and sim._dt_remainder == 0.0
    sim = ShipSimulator(spec, Fossen(spec), dt=0.3)            # 0.3/0.1 = 2.9999999999999996
    assert sim._n_sub_steps == 3 and sim._dt_remainder == 0.0

    ua = UAntwerpSurrogate.__new__(UAntwerpSurrogate)           # isinstance check only
    with pytest.raises(ValueError, match="multiple of 0.1"):
        ShipSimulator(spec, ua, dt=0.25)
