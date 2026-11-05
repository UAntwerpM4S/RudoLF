"""
End to end: checkpoint -> SurrogatePredictor -> UAntwerpSurrogate.step().

One step of UAntwerpSurrogate at SURROGATE_DT must give exactly the state
plus the deltas the predictor returns for the raw row it builds; any other
time step is refused. Also, the raw input row (rudder scaling) and the
preset/checkpoint consistency checks.
"""

import pytest
import numpy as np

from fw.surrogate.schema import SURROGATE_DT
from surrogate_test_helpers import make_state
from fw.ships.ship_factory import ShipModel, create_ship


def _surrogate(ckpt, preset="myzako_clean", **config):
    """
    A UAntwerpSurrogate for the Myzako on the given checkpoint.
    """

    from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate

    spec = create_ship(ShipModel.MYZAKO_37).specifications
    return UAntwerpSurrogate(spec, {"preset": preset, "checkpoint": str(ckpt), **config},
                             ship_model=ShipModel.MYZAKO_37)


def test_one_step_equals_predicted_deltas(tiny_models):
    """
    step(dt = SURROGATE_DT) returns u + du, v + dv, r + dr of the predictor for the same raw row.
    """

    model = _surrogate(tiny_models["open_water"])
    state = make_state(x=10.0, y=-5.0, heading=0.7, u=3.2, v=-0.15, r=0.012)
    rudder, thrust = np.radians(20.0), 0.6
    raw = model._build_input_vector(state, rudder, thrust)
    du, dv, dr = model.surrogate.predict_single(raw)
    u, v, r = model.step(state, rudder, thrust, SURROGATE_DT)
    assert u == pytest.approx(state.u + du, abs=1e-6)
    assert v == pytest.approx(state.v + dv, abs=1e-6)
    assert r == pytest.approx(state.r + dr, abs=1e-8)


@pytest.mark.parametrize("dt", [0.05, 0.2, 1.0, float("nan")])
def test_other_time_steps_are_refused(tiny_models, dt):
    """
    Any dt other than SURROGATE_DT raises ValueError (not an assert: also under python -O).
    """

    model = _surrogate(tiny_models["open_water"])
    state = make_state(x=0.0, y=0.0, heading=0.0, u=2.0, v=0.0, r=0.0)
    with pytest.raises(ValueError):
        model.step(state, 0.0, 0.5, dt)


@pytest.mark.parametrize("rudder_deg, expected", [(-75.0, -1.0), (0.0, 0.0), (37.5, 0.5), (75.0, 1.0)])
def test_rudder_action_scaling(tiny_models, rudder_deg, expected):
    """
    rudder_action = rudder angle / the checkpoint's rudder scale (75 deg for the Myzako data here).
    """

    model = _surrogate(tiny_models["open_water"])
    state = make_state(x=0.0, y=0.0, heading=0.0, u=2.0, v=0.0, r=0.0)
    raw = model._build_input_vector(state, np.radians(rudder_deg), 0.3)
    sc = model.scenario
    assert raw[sc.col("rudder_action")] == pytest.approx(expected, abs=1e-6)
    assert raw[sc.col("thrust_action")] == pytest.approx(0.3, abs=1e-7)


def test_raw_row_matches_scenario_layout(tiny_models):
    """
    The row UAntwerpSurrogate builds has the checkpoint scenario's raw layout, state values in place.
    """

    model = _surrogate(tiny_models["open_water"])
    state = make_state(x=1.5, y=2.5, heading=-1.0, u=3.0, v=0.2, r=-0.01)
    raw = model._build_input_vector(state, 0.0, 0.5)
    sc = model.scenario
    assert raw.shape == (sc.raw_input_dim,)
    for col, value in (("x", 1.5), ("y", 2.5), ("heading", -1.0), ("surge", 3.0), ("sway", 0.2),
                       ("yaw_rate", -0.01)):
        assert raw[sc.col(col)] == pytest.approx(value, abs=1e-6)


def test_wrong_scenario_checkpoint_is_refused(tiny_models):
    """
    A confined_ukc checkpoint in an open_water preset is refused when the surrogate is built.
    """

    with pytest.raises(ValueError):
        _surrogate(tiny_models["confined_ukc"], preset="myzako_clean")


def test_wrong_ship_is_refused(tiny_models):
    """
    A Tripoli caller with a Myzako preset is refused.
    """

    from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate

    spec = create_ship(ShipModel.TRIPOLI).specifications
    with pytest.raises(ValueError):
        UAntwerpSurrogate(spec, {"preset": "myzako_clean", "checkpoint": str(tiny_models["open_water"])},
                          ship_model=ShipModel.TRIPOLI)


def test_predictor_refuses_mismatching_data(tiny_models):
    """
    SurrogatePredictor.data_mismatch: other ship or other rudder scale -> a reason; same -> None.
    """

    from fw.surrogate.inference_surrogate import SurrogatePredictor
    from surrogate_test_helpers import identity

    pred = SurrogatePredictor(str(tiny_models["open_water"]), device="cpu")
    assert pred.data_mismatch(identity("open_water")) is None
    assert "TRIPOLI" in pred.data_mismatch(identity("open_water", ship_model="TRIPOLI", rudder_max_deg=75.0))
    assert "rudder" in pred.data_mismatch(identity("open_water", rudder_max_deg=35.0))
