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
    raw = None

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

    def get_env(pos, _state=None):
        """
        Apparent wind of `state` for true_wind; current 0.6 m/s toward 1.0 rad.
        """

        _ws, _wd = apparent_wind_from_true_world(true_wind, float(_state.heading), float(_state.u), float(_state.v))
        return EnvironmentData(**_calm(5.0, _ws, _wd, 0.6, 1.0))

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


def test_reset_episode_restores_the_default_ukc(full_env_model):
    """
    After reset_episode, get_ukc(None) no longer returns the previous episode's (smoothed) UKC.
    """

    model = _ua(full_env_model, env_smoothing_tau=2.0)
    model.get_env = _env_sequence([_calm(4.0)])
    model._build_input_vector(make_state(x=1.0, y=2.0, heading=0.0, u=2.0, v=0.0, r=0.0), 0.0, 0.5)
    assert model.get_ukc(None) == pytest.approx(4.0)
    model.reset_episode()
    assert model.get_ukc(None) == model.default_ukc


def test_scale_script_refuses_bad_input():
    """
    compute_wind_current_obs_scales: percentile outside [0, 100] or no finite rows raise ValueError.
    """

    from fw.surrogate.compute_wind_current_obs_scales import suggest_scales

    cols = {c: np.ones(3) for c in ("heading", "wind_speed", "wind_direction", "current_speed", "current_direction")}
    with pytest.raises(ValueError, match="percentile"):
        suggest_scales(cols, 101.0)
    with pytest.raises(ValueError, match="No finite wind"):
        suggest_scales(dict(cols, wind_speed=np.full(3, np.nan)), 99.9)
    assert suggest_scales(cols, 99.9)["wind"]["rows"] == 3


# ---------------------------------------------------------------- random wind and current

def _random_ua(ckpt, seed=0, **config):
    """
    full_env surrogate with wind_current="random", a stand-in lookup and an episode drawn from `seed`.
    """

    model = _ua(ckpt, wind_current="random", **config)
    model.wcu_lookup = object()     # only checked for presence; get_env is replaced in the tests
    model.reset_episode(np.random.default_rng(seed))
    return model


def _raw_env(model, raw):
    """
    (ukc, wind speed, wind dir, current speed, current dir) of a raw input row.
    """

    sc = model.scenario
    return [float(raw[sc.col(c)]) for c in ("ukc", "wind_speed", "wind_direction", "current_speed",
                                            "current_direction")]


def test_invalid_wind_current_source_is_refused(full_env_model):
    """
    wind_current must be "lookup" or "random".
    """

    with pytest.raises(ValueError, match="wind_current"):
        _ua(full_env_model, wind_current="csv")


def test_random_mode_needs_an_episode_draw(full_env_model):
    """
    Without reset_episode(rng) the first model step raises.
    """

    model = _ua(full_env_model, wind_current="random")
    model.get_env = _env_sequence([_calm(5.0)])
    with pytest.raises(RuntimeError, match="no wind/current drawn"):
        model._build_input_vector(make_state(x=1.0, y=2.0, heading=0.0, u=2.0, v=0.0, r=0.0), 0.0, 0.5)


def test_random_draw_is_reproducible_and_in_range(full_env_model):
    """
    Same seed, same episode; wind speed within [0, RANDOM_WIND_SPEED_MAX], current within [0, RANDOM_CURRENT_SPEED_MAX].
    """

    from fw.dynamics import uantwerp_surrogate as ua

    a, b = _random_ua(full_env_model, seed=4), _random_ua(full_env_model, seed=4)
    np.testing.assert_array_equal(a._random_wind_world, b._random_wind_world)
    assert (a._random_current_speed, a._random_current_sign) == (b._random_current_speed, b._random_current_sign)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a.reset_episode(rng)
        assert 0.0 <= np.linalg.norm(a._random_wind_world) <= ua.RANDOM_WIND_SPEED_MAX
        assert 0.0 <= a._random_current_speed <= ua.RANDOM_CURRENT_SPEED_MAX
        assert a._random_current_sign in (1.0, -1.0)


def test_random_wind_is_constant_earth_fixed(full_env_model):
    """
    The model input is the apparent wind of the drawn true wind for every ship state.
    """

    from fw.dynamics.uantwerp_surrogate import apparent_wind_from_true_world

    model = _random_ua(full_env_model, seed=1)
    model.get_env = _env_sequence([_calm(5.0, 9.0, 0.4, 0.8, 1.0)])     # lookup wind must be ignored
    for heading, u, v in ((0.0, 1.0, 0.0), (1.2, 3.0, 0.3), (-2.5, 5.0, -0.2)):
        state = make_state(x=1.0, y=2.0, heading=heading, u=u, v=v, r=0.0)
        got = _raw_env(model, model._build_input_vector(state, 0.0, 0.5))
        ws, wd = apparent_wind_from_true_world(model._random_wind_world, heading, u, v)
        np.testing.assert_allclose(got[1:3], [ws, wd], atol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_current_follows_the_lookup_line(full_env_model, seed):
    """
    Drawn speed along the lookup's current line: drawn sign at the first step,
    then continuous through a flipped row; a slack row keeps the previous direction.
    """

    model = _random_ua(full_env_model, seed=seed)
    model.get_env = _env_sequence([_calm(5.0, current_speed=1.0, current_direction=0.3),
                                   _calm(5.0, current_speed=0.7, current_direction=0.3 + np.pi),  # flipped row
                                   _calm(5.0, current_speed=1.2, current_direction=0.6),          # river bends
                                   _calm(5.0, current_speed=0.01, current_direction=-2.0)])       # slack row
    state = make_state(x=1.0, y=2.0, heading=0.0, u=2.0, v=0.0, r=0.0)
    first = 0.3 if model._random_current_sign > 0 else 0.3 - np.pi
    expected = [first, first, first + 0.3, first + 0.3]
    for exp in expected:
        got = _raw_env(model, model._build_input_vector(state, 0.0, 0.5))
        assert got[3] == pytest.approx(model._random_current_speed, abs=1e-6)
        assert np.cos(got[4] - exp) == pytest.approx(1.0, abs=1e-6)


def test_random_current_without_lookup_is_zero(full_env_model):
    """
    Without a lookup CSV: random wind, zero current.
    """

    model = _ua(full_env_model, wind_current="random")
    model.reset_episode(np.random.default_rng(0))
    got = _raw_env(model, model._build_input_vector(make_state(x=1.0, y=2.0, heading=0.0, u=2.0, v=0.0, r=0.0),
                                                    0.0, 0.5))
    assert got[3] == 0.0
    assert got[1] > 0.0


def test_random_mode_smooths_only_the_ukc(full_env_model):
    """
    With env_smoothing_tau, the UKC is averaged; wind and current stay the episode's exact values.
    """

    model = _random_ua(full_env_model, seed=2, env_smoothing_tau=2.0)
    model.get_env = _env_sequence([_calm(4.0, current_speed=1.0, current_direction=0.3),
                                   _calm(10.0, current_speed=1.0, current_direction=0.3)])
    state = make_state(x=1.0, y=2.0, heading=0.5, u=3.0, v=0.1, r=0.0)
    model._build_input_vector(state, 0.0, 0.5)
    got = _raw_env(model, model._build_input_vector(state, 0.0, 0.5))
    alpha = 1.0 - np.exp(-SURROGATE_DT / 2.0)
    assert got[0] == pytest.approx(4.0 + alpha * 6.0, rel=1e-6)
    assert got[3] == pytest.approx(model._random_current_speed, abs=1e-6)


def test_random_mode_has_no_effect_without_wind_current(tiny_models):
    """
    A confined preset accepts wind_current="random" and stays without wind and current.
    """

    model = _ua(tiny_models["confined_ukc"], preset="myzako_confined", wind_current="random")
    model.reset_episode(np.random.default_rng(0))
    assert model._random_wind_world is None
    np.testing.assert_array_equal(model.observed_wind_current(make_state(x=1.0, y=2.0, heading=0.0)), np.zeros(4))


def test_env_draws_a_new_episode_from_its_rng(full_env_model):
    """
    PySimEnv.reset(seed) draws reproducibly; the next reset draws a new wind and current.
    """

    config = {"preset": "myzako_full", "checkpoint": str(full_env_model), "lookup_csv": None,
              "wind_current": "random"}
    envs = [_env(dynamics_model=Model.UANTWERP, ua_surrogate_config=config, wind=True, current=True)
            for _ in range(2)]
    obs = [env.reset(seed=11)[0] for env in envs]
    np.testing.assert_array_equal(obs[0], obs[1])
    first = envs[0].dynamics._random_wind_world.copy()
    np.testing.assert_array_equal(first, envs[1].dynamics._random_wind_world)
    assert np.any(obs[0][BASE_OBS_DIM:BASE_OBS_DIM + 2] != 0.0)       # random wind observed
    envs[0].reset()
    assert not np.array_equal(first, envs[0].dynamics._random_wind_world)


# ---------------------------------------------------------------- current line, real lookup, episode log

def _full_lookup_csv(path, flow_dir=0.2, speed=1.0, slack_every=0):
    """
    full_env lookup CSV on a 5 m grid (x 0..300, y 0..100): current along `flow_dir`,
    rows alternately recorded in flood and ebb (direction flipped, checkerboard);
    every `slack_every`-th row is slack (0.01 m/s, random direction).
    """

    sc = get_scenario("full_env")
    rng = np.random.default_rng(0)
    rows = []
    for i, x in enumerate(np.arange(0.0, 301.0, 5.0)):
        for j, y in enumerate(np.arange(0.0, 101.0, 5.0)):
            row = np.zeros(sc.raw_input_dim + 3 + 1)
            row[sc.col("x")], row[sc.col("y")], row[sc.col("ukc")] = x, y, 6.0
            row[sc.col("wind_speed")], row[sc.col("wind_direction")] = 4.0, 0.5
            row[sc.col("surge")] = 2.0
            k = len(rows)
            if slack_every and k % slack_every == 0:
                row[sc.col("current_speed")], row[sc.col("current_direction")] = 0.01, rng.uniform(-np.pi, np.pi)
            else:
                d = flow_dir + (np.pi if (i + j) % 2 else 0.0)
                row[sc.col("current_speed")] = speed
                row[sc.col("current_direction")] = (d + np.pi) % (2 * np.pi) - np.pi
            rows.append(row)
    np.savetxt(path, np.array(rows), delimiter=",")
    return path


def test_follow_current_line_ignores_cross_flow_rows():
    """
    The reviewer's sequence 0, 0, 85, 170, 180, 0 deg (cross flow, then the other tide) never reverses the current.
    """

    from fw.dynamics.uantwerp_surrogate import follow_current_line

    ref, dirs = None, []
    for deg in (0.0, 0.0, 85.0, 170.0, 180.0, 0.0):
        d, ref = follow_current_line(np.radians(deg), 1.0, ref, 1.0)
        dirs.append(np.degrees(d))
    # 85 deg is cross flow (ignored); the 170 deg line is followed as -10 deg.
    np.testing.assert_allclose(dirs, [0.0, 0.0, 0.0, -10.0, 0.0, 0.0], atol=1e-9)


def test_follow_current_line_bends_slack_and_nan():
    """
    Gradual bends are followed; slack and non-finite rows keep the reference.
    """

    from fw.dynamics.uantwerp_surrogate import follow_current_line

    ref = 0.0
    for k in range(1, 31):                                                     # 30 x 3 deg bend, alternating tides
        d, ref = follow_current_line(np.radians(3.0 * k) + (np.pi if k % 2 else 0.0), 1.0, ref, 1.0)
    assert d == ref == pytest.approx(np.radians(90.0))
    assert follow_current_line(2.0, 0.01, ref, 1.0) == (ref, ref)              # slack
    assert follow_current_line(float("nan"), 1.0, ref, 1.0) == (ref, ref)      # non-finite line
    assert follow_current_line(1.6, float("nan"), ref, 1.0) == (ref, ref)      # non-finite speed


def test_slack_rows_before_the_first_reliable_row_are_provisional():
    """
    Slack rows at the start are used sign-oriented without becoming the reference;
    the first reliable row then sets the orientation with the drawn sign.
    """

    from fw.dynamics.uantwerp_surrogate import follow_current_line

    for sign in (1.0, -1.0):
        d, ref = follow_current_line(2.5, 0.01, None, sign)                    # slack, noise direction
        assert ref is None and np.cos(d - 2.5) == pytest.approx(sign)
        d, ref = follow_current_line(0.2, 1.0, ref, sign)                      # first reliable row
        assert ref == d and np.cos(d - 0.2) == pytest.approx(sign)
        d, ref = follow_current_line(0.2 + np.pi, 1.0, ref, sign)              # opposite-tide row: no flip
        assert np.cos(d - 0.2) == pytest.approx(sign)


def test_random_mode_with_a_real_lookup(full_env_model, tmp_path):
    """
    Real WcuLookup and get_env: the current keeps one orientation along the flow line
    through flood/ebb rows and slack rows, at the drawn speed; the wind is the drawn true wind.
    """

    from fw.dynamics.uantwerp_surrogate import apparent_wind_from_true_world

    csv = _full_lookup_csv(tmp_path / "lk.csv", flow_dir=0.2, slack_every=7)
    for seed in range(4):
        model = _ua(full_env_model, wind_current="random", lookup_csv=str(csv))
        model.reset_episode(np.random.default_rng(seed))
        dirs = []
        for x in np.arange(10.0, 290.0, 0.7):
            state = make_state(x=float(x), y=50.0, heading=0.1, u=3.0, v=0.0, r=0.0)
            got = _raw_env(model, model._build_input_vector(state, 0.0, 0.5))
            assert got[0] == pytest.approx(6.0)
            assert got[3] == pytest.approx(model._random_current_speed, abs=1e-6)
            np.testing.assert_allclose(got[1:3], apparent_wind_from_true_world(model._random_wind_world, 0.1, 3.0,
                                                                               0.0), atol=1e-5)
            dirs.append(got[4])
        np.testing.assert_allclose(np.cos(np.array(dirs) - dirs[0]), 1.0, atol=1e-9)
        assert np.cos(dirs[0] - 0.2) == pytest.approx(model._random_current_sign, abs=1e-6)


def test_random_observation_equals_the_model_input(full_env_model, tmp_path):
    """
    In random mode the observed wind/current are the body components of the model input.
    """

    model = _ua(full_env_model, wind_current="random", lookup_csv=str(_full_lookup_csv(tmp_path / "lk.csv")))
    model.reset_episode(np.random.default_rng(9))
    state = make_state(x=40.0, y=20.0, heading=-0.7, u=2.5, v=0.2, r=0.0)
    got = _raw_env(model, model._build_input_vector(state, 0.0, 0.5))
    np.testing.assert_allclose(model.observed_wind_current(state),
                               env_body_components(got[1], got[2], got[3], got[4], -0.7), atol=1e-5)


def test_diagnostic_script_on_a_synthetic_lookup(tmp_path):
    """
    check_random_current_lookup.follow_route: no reversals along the flow, slack rows counted.
    """

    from fw.dynamics.uantwerp_surrogate import make_scenario_lookup
    from fw.surrogate.check_random_current_lookup import follow_route

    lookup = make_scenario_lookup(_full_lookup_csv(tmp_path / "lk.csv", flow_dir=0.0, slack_every=5),
                                  get_scenario("full_env"))
    points = np.column_stack([np.arange(5.0, 295.0, 0.5), np.full(580, 50.0)])
    for sign in (1.0, -1.0):
        s = follow_route(lookup, points, sign, jump_deg=10.0)
        assert s["reversals"] == 0 and s["jumps"] == 0
        assert s["slack"] > 0
        assert s["along_share"] == pytest.approx(1.0)


def test_episode_end_is_logged_with_outcome_and_conditions(full_env_model, caplog):
    """
    One log line per finished episode with outcome and drawn conditions; the same in info.
    """

    import logging

    config = {"preset": "myzako_full", "checkpoint": str(full_env_model), "lookup_csv": None,
              "wind_current": "random"}
    from fw.common.default_positions import DEFAULT_FRAMEWORK_SHIP_POS, DEFAULT_FRAMEWORK_TARGET_POS

    env = _env(dynamics_model=Model.UANTWERP, ua_surrogate_config=config, wind=True, current=True, max_steps=2,
               start_pos=np.array(DEFAULT_FRAMEWORK_SHIP_POS, dtype=np.float32),
               target_pos=np.array(DEFAULT_FRAMEWORK_TARGET_POS, dtype=np.float32))
    env.reset(seed=0)
    with caplog.at_level(logging.INFO, logger="fw.environments.py_sim_env"):
        _, _, _, truncated, info = env.step(np.array([0.0, 0.5], dtype=np.float32))
        assert not truncated and "episode_outcome" not in info
        _, _, _, truncated, info = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert truncated and info["episode_outcome"] == "max_steps"
    assert set(info["episode_conditions"]) == {"wind_speed", "wind_from_deg", "current_speed", "current_sign"}
    lines = [r.getMessage() for r in caplog.records if "episode end" in r.getMessage()]
    assert len(lines) == 1 and "outcome=max_steps" in lines[0] and "success=False" in lines[0]
    assert "wind_speed" in lines[0]


def test_current_follower_smooths_cell_noise():
    """
    With tau, a line jittering +-17 deg between cells is averaged; without tau it is followed as is.
    """

    from fw.dynamics.uantwerp_surrogate import CurrentLineFollower

    axes = [0.3 + (0.3 if k % 2 else -0.3) for k in range(400)]
    raw, smooth = CurrentLineFollower(1.0), CurrentLineFollower(1.0, tau=10.0)
    raw_dirs = [raw.step(a, 1.0, 0.1) for a in axes]
    smooth_dirs = [smooth.step(a, 1.0, 0.1) for a in axes]
    np.testing.assert_allclose(raw_dirs, axes, atol=1e-12)
    assert np.max(np.abs(np.array(smooth_dirs[1:]) - 0.3)) < np.radians(18.0)       # first step only
    assert np.max(np.abs(np.array(smooth_dirs[300:]) - 0.3)) < np.radians(3.0)      # after 3 tau
    assert np.max(np.abs(np.diff(smooth_dirs))) < np.radians(4.0)


def test_current_follower_restarts_at_the_first_reliable_row():
    """
    A provisional slack direction is not averaged into the real one.
    """

    from fw.dynamics.uantwerp_surrogate import CurrentLineFollower

    f = CurrentLineFollower(1.0, tau=10.0)
    assert f.step(2.5, 0.01, 0.1) == pytest.approx(2.5)        # provisional
    assert f.step(0.2, 1.0, 0.1) == pytest.approx(0.2)         # restart, no blend with 2.5
    assert f.step(0.2, 1.0, 0.1) == pytest.approx(0.2)


def test_random_mode_smooths_the_current_direction(full_env_model):
    """
    In the model input, the current direction is smoothed with env_smoothing_tau; its speed is the drawn one.
    """

    model = _random_ua(full_env_model, seed=0, env_smoothing_tau=10.0)
    sign = model._random_current_sign
    rows = [_calm(5.0, current_speed=1.0, current_direction=0.3 + (0.3 if k % 2 else -0.3)) for k in range(400)]
    model.get_env = _env_sequence(rows)
    state = make_state(x=1.0, y=2.0, heading=0.0, u=2.0, v=0.0, r=0.0)
    dirs = []
    for _ in range(400):
        got = _raw_env(model, model._build_input_vector(state, 0.0, 0.5))
        assert got[3] == pytest.approx(model._random_current_speed, abs=1e-6)
        dirs.append(got[4])
    target = 0.3 if sign > 0 else 0.3 - np.pi
    assert np.max(np.abs(np.sin(np.array(dirs[300:]) - target))) < np.radians(3.0)     # after 3 tau
    assert np.min(np.cos(np.array(dirs[300:]) - target)) > 0.99
