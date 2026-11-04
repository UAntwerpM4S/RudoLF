"""
Wind and current in the RL observation: the shared body-component encoding,
the UAntwerp surrogate's smoothing and observation, and PySimEnv's flag rules
and observation layout.
"""

import numpy as np
import pytest

from fw.dynamics.dynamics_model import Model
from fw.dynamics.wcu_lookup import EnvironmentData
from fw.environments.wind_current_obs import (
    CURRENT_OBS_SCALE,
    WIND_CURRENT_OBS_DIM,
    WIND_OBS_SCALE,
    normalize_wind_current,
)
from fw.ships.ship_factory import ShipModel, create_ship
from fw.surrogate.schema import (
    ENV_ENCODING_BODY,
    SURROGATE_DT,
    encoded_feature_names,
    env_body_components,
    get_scenario,
    select_and_encode,
)

from surrogate_test_helpers import make_state, synthetic_rows, train_tiny

BASE_OBS_DIM = 12


@pytest.fixture(scope="module")
def full_env_model(tmp_path_factory):
    """
    One-epoch full_env Myzako checkpoint on synthetic data.
    """

    ckpt, _ = train_tiny(tmp_path_factory.mktemp("full_env_model"), "full_env")
    return ckpt


def _ua(ckpt, preset="myzako_full", **config):
    """
    UAntwerpSurrogate for the Myzako without a lookup CSV.
    """

    from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate

    spec = create_ship(ShipModel.MYZAKO_37).specifications
    return UAntwerpSurrogate(spec, {"preset": preset, "checkpoint": str(ckpt), "lookup_csv": None, **config},
                             ship_model=ShipModel.MYZAKO_37)


def _env_sequence(values):
    """
    get_env replacement returning EnvironmentData(**values[i]) on call i (the last one repeated).
    """

    calls = []

    def get_env(pos, state=None):
        """
        Next environment of the sequence.
        """

        calls.append(pos)
        return EnvironmentData(**values[min(len(calls), len(values)) - 1])

    return get_env


def _calm(ukc, wind_speed=0.0, wind_direction=0.0, current_speed=0.0, current_direction=0.0):
    """
    EnvironmentData fields as a dict.
    """

    return dict(ukc=ukc, wind_speed=wind_speed, wind_direction=wind_direction,
                current_speed=current_speed, current_direction=current_direction)


# ---------------------------------------------------------------- encoding

def test_body_components_equal_the_dataset_encoding():
    """
    env_body_components gives exactly the body_components columns of select_and_encode.
    """

    sc = get_scenario("full_env")
    x, _, _ = synthetic_rows("full_env", n_episodes=3, steps=5, seed=1)
    encoded = select_and_encode(x, sc, ENV_ENCODING_BODY)
    names = encoded_feature_names(sc, ENV_ENCODING_BODY)
    xd = x.astype(np.float64)
    comps = env_body_components(xd[:, sc.col("wind_speed")], xd[:, sc.col("wind_direction")],
                                xd[:, sc.col("current_speed")], xd[:, sc.col("current_direction")],
                                xd[:, sc.col("heading")])
    for name, comp in zip(("wind_from_ahead", "wind_from_stbd", "current_to_ahead", "current_to_stbd"), comps):
        np.testing.assert_array_equal(encoded[:, names.index(name)], comp.astype(np.float32))


def test_normalize_scales_and_clips():
    """
    Wind / WIND_OBS_SCALE, current / CURRENT_OBS_SCALE, clipped to [-1, 1].
    """

    out = normalize_wind_current([0.5 * WIND_OBS_SCALE, -3.0 * WIND_OBS_SCALE, 0.25 * CURRENT_OBS_SCALE, 9.0])
    np.testing.assert_allclose(out, [0.5, -1.0, 0.25, 1.0])
    assert out.dtype == np.float32
    with pytest.raises(ValueError):
        normalize_wind_current([1.0, 2.0])


# ---------------------------------------------------------------- UAntwerp surrogate

@pytest.mark.parametrize("tau", [0, -1.0, float("nan"), True, "10"])
def test_invalid_smoothing_tau_is_refused(full_env_model, tau):
    """
    env_smoothing_tau must be None or a positive number of seconds.
    """

    with pytest.raises(ValueError, match="env_smoothing_tau"):
        _ua(full_env_model, env_smoothing_tau=tau)


def test_without_smoothing_the_model_input_is_get_env(full_env_model):
    """
    No env_smoothing_tau: the raw row holds get_env's values unchanged.
    """

    model = _ua(full_env_model)
    model.get_env = _env_sequence([_calm(4.0, 3.0, 0.4, 0.7, 2.0), _calm(9.0, 6.0, -1.0, 0.2, -2.0)])
    state = make_state(x=1.0, y=2.0, heading=0.3, u=2.0, v=0.1, r=0.0)
    sc = model.scenario
    for expected in ((4.0, 3.0, 0.4, 0.7, 2.0), (9.0, 6.0, -1.0, 0.2, -2.0)):
        raw = model._build_input_vector(state, 0.0, 0.5)
        got = [raw[sc.col(c)] for c in ("ukc", "wind_speed", "wind_direction", "current_speed",
                                        "current_direction")]
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_smoothing_is_an_exponential_moving_average(full_env_model):
    """
    A UKC step from 4 to 10 m approaches 10 with alpha = 1 - exp(-dt / tau) per model step.
    """

    tau = 2.0
    model = _ua(full_env_model, env_smoothing_tau=tau)
    model.get_env = _env_sequence([_calm(4.0), _calm(10.0)])
    state = make_state(x=1.0, y=2.0, heading=0.3, u=2.0, v=0.0, r=0.0)
    alpha = 1.0 - np.exp(-SURROGATE_DT / tau)
    raw = model._build_input_vector(state, 0.0, 0.5)
    assert raw[model.scenario.col("ukc")] == pytest.approx(4.0)
    expected = 4.0
    for _ in range(5):
        raw = model._build_input_vector(state, 0.0, 0.5)
        expected += alpha * (10.0 - expected)
        assert raw[model.scenario.col("ukc")] == pytest.approx(expected, rel=1e-6)
    assert model.ukc == pytest.approx(expected, rel=1e-6)

    model.reset_episode()   # restarts at the next raw value
    raw = model._build_input_vector(state, 0.0, 0.5)
    assert raw[model.scenario.col("ukc")] == pytest.approx(10.0)


def test_smoothing_keeps_a_steady_wind_and_current(full_env_model):
    """
    A steady environment passes the smoothing (apparent -> true -> apparent) unchanged.
    """

    model = _ua(full_env_model, env_smoothing_tau=5.0)
    env = _calm(6.0, 7.5, 1.1, 0.8, -2.5)
    model.get_env = _env_sequence([env])
    state = make_state(x=1.0, y=2.0, heading=-0.4, u=3.0, v=0.2, r=0.0)
    sc = model.scenario
    for _ in range(3):
        raw = model._build_input_vector(state, 0.0, 0.5)
    got = [raw[sc.col(c)] for c in ("ukc", "wind_speed", "wind_direction", "current_speed", "current_direction")]
    np.testing.assert_allclose(got, list(env.values()), atol=1e-5)


def test_smoothing_is_earth_fixed(full_env_model):
    """
    A constant true wind and current seen by a ship that turns and changes speed
    stay constant after smoothing: the model input is the apparent wind of the
    ship's current state.
    """

    from fw.dynamics.uantwerp_surrogate import apparent_wind_from_true_world

    true_wind = np.array([3.0, -4.0])          # earth-fixed, pointing where the wind comes from
    model = _ua(full_env_model, env_smoothing_tau=5.0)

    def get_env(pos, state=None):
        """
        Apparent wind of `state` for true_wind; current 0.6 m/s toward 1.0 rad.
        """

        ws, wd = apparent_wind_from_true_world(true_wind, float(state.heading), float(state.u), float(state.v))
        return EnvironmentData(**_calm(5.0, ws, wd, 0.6, 1.0))

    model.get_env = get_env
    sc = model.scenario
    for heading, u, v in ((0.0, 1.0, 0.0), (1.2, 3.0, 0.3), (-2.5, 5.0, -0.2)):
        state = make_state(x=1.0, y=2.0, heading=heading, u=u, v=v, r=0.0)
        raw = model._build_input_vector(state, 0.0, 0.5)
        ws, wd = apparent_wind_from_true_world(true_wind, heading, u, v)
        got = [raw[sc.col(c)] for c in ("wind_speed", "wind_direction", "current_speed", "current_direction")]
        np.testing.assert_allclose(got, [ws, wd, 0.6, 1.0], atol=1e-5)


def test_observation_is_the_last_model_input_for_this_state(full_env_model):
    """
    observed_wind_current: the body components of the last model input; before
    the first step, of the environment at the state.
    """

    model = _ua(full_env_model, env_smoothing_tau=5.0)
    env = _calm(6.0, 7.5, 1.1, 0.8, -2.5)
    model.get_env = _env_sequence([env])
    state = make_state(x=1.0, y=2.0, heading=-0.4, u=3.0, v=0.2, r=0.0)
    expected = env_body_components(7.5, 1.1, 0.8, -2.5, -0.4)
    np.testing.assert_allclose(model.observed_wind_current(state), expected, atol=1e-6)
    model._build_input_vector(state, 0.0, 0.5)
    np.testing.assert_allclose(model.observed_wind_current(state), expected, atol=1e-6)


def test_confined_surrogate_observes_no_wind_or_current(tiny_models):
    """
    Presets without wind and current observe zeros.
    """

    model = _ua(tiny_models["confined_ukc"], preset="myzako_confined")
    state = make_state(x=1.0, y=2.0, heading=0.3, u=2.0, v=0.0, r=0.0)
    np.testing.assert_array_equal(model.observed_wind_current(state), np.zeros(4))


# ---------------------------------------------------------------- PySimEnv

def _env(**kwargs):
    """
    PySimEnv for the Myzako 37.
    """

    from fw.environments.py_sim_env import PySimEnv

    kwargs.setdefault("ship_model", ShipModel.MYZAKO_37)
    return PySimEnv(render_mode=None, **kwargs)


@pytest.mark.parametrize("wind, current", [(True, False), (False, True)])
def test_wind_and_current_must_be_equal(wind, current):
    """
    Only both or neither.
    """

    with pytest.raises(ValueError, match="together"):
        _env(wind=wind, current=current)


@pytest.mark.parametrize("flags, dim", [(False, BASE_OBS_DIM), (True, BASE_OBS_DIM + WIND_CURRENT_OBS_DIM)])
def test_observation_layout_and_zeros_for_white_box_models(flags, dim):
    """
    The flags add 4 features; Fossen observes zeros.
    """

    env = _env(dynamics_model=Model.FOSSEN, wind=flags, current=flags)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape == (dim,) and obs.shape == (dim,)
    if flags:
        np.testing.assert_array_equal(obs[BASE_OBS_DIM:], np.zeros(WIND_CURRENT_OBS_DIM))
        obs, *_ = env.step(np.array([0.2, 0.5], dtype=np.float32))
        np.testing.assert_array_equal(obs[BASE_OBS_DIM:], np.zeros(WIND_CURRENT_OBS_DIM))


def test_full_env_surrogate_requires_the_flags(full_env_model):
    """
    A full_env preset with wind/current off is refused; on, its observation has the 4 features.
    """

    config = {"preset": "myzako_full", "checkpoint": str(full_env_model), "lookup_csv": None}
    with pytest.raises(ValueError, match="must be enabled"):
        _env(dynamics_model=Model.UANTWERP, ua_surrogate_config=config)
    env = _env(dynamics_model=Model.UANTWERP, ua_surrogate_config=config, wind=True, current=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (BASE_OBS_DIM + WIND_CURRENT_OBS_DIM,)
    np.testing.assert_array_equal(obs[BASE_OBS_DIM:], np.zeros(WIND_CURRENT_OBS_DIM))  # no lookup: calm


class _Bridge:
    """
    Minimal FH bridge stand-in (only the attributes PySimEnv checks at construction).
    """

    ship_model = ShipModel.MYZAKO_37

    def __init__(self, wind_current=None):
        """
        wind_current None: no such attribute (as SimSyncBridge).
        """

        if wind_current is not None:
            self.wind_current = wind_current


@pytest.mark.parametrize("bridge_wc, flags", [(False, True), (None, True), (True, False)])
def test_fh_bridge_exercise_must_match_the_flags(bridge_wc, flags):
    """
    The bridge's wind_current (absent = False) must equal the flags.
    """

    with pytest.raises(ValueError, match="wind_current"):
        _env(dynamics_model=Model.FH_TABULAR, fh_bridge=_Bridge(bridge_wc), wind=flags, current=flags)


def test_env_reset_restarts_the_smoothing(full_env_model):
    """
    PySimEnv.reset() restarts the UA moving average (ShipSimulator.reset -> reset_episode).
    """

    config = {"preset": "myzako_full", "checkpoint": str(full_env_model), "lookup_csv": None,
              "env_smoothing_tau": 10.0}
    env = _env(dynamics_model=Model.UANTWERP, ua_surrogate_config=config, wind=True, current=True)
    env.reset(seed=0)
    env.dynamics._env_ema = np.full(5, 999.0)   # stale state of a previous episode
    env.reset(seed=1)
    assert env.dynamics._env_ema[0] == pytest.approx(env.dynamics.default_ukc)
    np.testing.assert_array_equal(env.dynamics._env_ema[1:], np.zeros(4))
