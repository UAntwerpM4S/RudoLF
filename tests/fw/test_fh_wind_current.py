"""
FhSimBridge with wind and current, without an FH installation: exercise
selection by the wind/current flag, and the per-episode wind-field transform
(on a fake math model) including the one-time check after a rewind.
"""

import os
import pytest
import numpy as np

from types import SimpleNamespace
from fw.simulators import fh_sim_bridge
from fw.ships.ship_factory import ShipModel
from fw.simulators.fh_sim_bridge import FhSimBridge
from fw.simulators.fh_interface_utils import WIND_SCALE_MAX, draw_wind_field_transform

EXERCISE_SUBDIR = "database/areas/ScheldeSaeftinge_23_002/invoer/tra"


@pytest.mark.parametrize("ship, wind_current, expected", [
    (ShipModel.TRIPOLI, True, "DDShip_Tripoli_scenario_windcurrent.tab"),
    (ShipModel.TRIPOLI, False, "DDShip_Tripoli_scenario_nowindnocurrent.tab"),
    (ShipModel.MYZAKO_37, True, "DDShip_scenario3_windcurrent_myzako_37.tab"),
    (ShipModel.MYZAKO_37, False, "DDShip_scenario1_nowindnocurrent_myzako_37.tab"),
    (ShipModel.MYZAKO_29, True, "DDShip_scenario3_windcurrent_myzako_29.tab"),
    (ShipModel.MYZAKO_29, False, "DDShip_scenario1_nowindnocurrent_myzako_29.tab"),
    (ShipModel.MYZAKO_DEFAULT, True, "DDShip_scenario3_windcurrent_default.tab"),
    (ShipModel.MYZAKO_DEFAULT, False, "DDShip_scenario1_nowindnocurrent_default.tab"),
])
def test_exercise_follows_the_wind_current_flag(tmp_path, monkeypatch, ship, wind_current, expected):
    """
    The exercise file name per ship and flag (default scenario base).
    """

    folder = tmp_path / EXERCISE_SUBDIR
    folder.mkdir(parents=True)
    (folder / expected).write_text("")
    monkeypatch.setattr(fh_sim_bridge, "simexe_path", str(tmp_path))
    bridge = FhSimBridge(ship_model=ship, output_path=str(tmp_path / "out"), wind_current=wind_current)
    try:
        assert os.path.basename(bridge.exercise_path) == expected
        assert bridge.exercise_path_exists
        assert bridge.wind_current is wind_current
    finally:
        bridge.dispose()


@pytest.mark.parametrize("ship", [ShipModel.TRIPOLI, ShipModel.MYZAKO_37])
def test_exercises_match_the_dataset_generator(tmp_path, ship):
    """
    RL loads the exercises the generator used: "full" with, "confined" without wind/current.
    """

    from fw.surrogate.sim_fh_dataset_generator import SHIP_PROFILES

    for wind_current, configuration in ((True, "full"), (False, "confined")):
        bridge = FhSimBridge(ship_model=ship, output_path=str(tmp_path / "out"), wind_current=wind_current)
        try:
            assert os.path.basename(bridge.exercise_path) == SHIP_PROFILES[ship].exercises[configuration][0]
        finally:
            bridge.dispose()


def test_wind_draw_matches_the_generator_distribution():
    """
    Rotation in [0, 360), Scale in [0, WIND_SCALE_MAX]; reproducible with a seed.
    """

    draws = [draw_wind_field_transform(np.random.default_rng(7)) for _ in range(2)]
    assert draws[0] == draws[1]
    rng = np.random.default_rng(0)
    rot, scale = np.array([draw_wind_field_transform(rng) for _ in range(2000)]).T
    assert rot.min() >= 0.0 and rot.max() < 360.0
    assert scale.min() >= 0.0 and scale.max() <= WIND_SCALE_MAX


class _FakeMathModel:
    """
    FH math model stand-in: wind = Scale * BASE_SPEED (ship at rest, so
    apparent = true). As measured on FH (diagnose_fh_rewind_wind): a transform
    is applied, and WindState updated, only at the 1 Hz environment update
    (t = k + UPDATE_OFFSET, k >= 1); a rewind returns to t = 0, discards a
    transform not applied yet and keeps the applied one.
    """

    BASE_SPEED = 1.1
    UPDATE_OFFSET = 0.1

    def __init__(self, ignore_after_rewind: bool):
        """
        ignore_after_rewind: transforms set after a rewind are never applied.
        """

        self.ignore_after_rewind = ignore_after_rewind
        self.time = 0.0
        self.pending = None          # transform set, not applied yet
        self.active = (0.0, 1.0)     # applied transform
        self.shown = (0.0, 1.0)      # transform WindState reports
        self.init_transform = None
        self.rewound = False
        self.updates_enabled = True  # False: WindState never updates
        self.requested = SimpleNamespace(Rotation=0.0, Scale=1.0, Offset=0.0)
        control = SimpleNamespace(getFieldTransform=lambda: SimpleNamespace(**vars(self.requested)),
                                  setFieldTransform=self._set_transform)
        self._env = SimpleNamespace(getWindFieldControl=lambda: control)
        self.ship = SimpleNamespace(getWindState=self._wind_state,
                                    getShipVelocityOverGround=lambda: SimpleNamespace(x=0.0, y=0.0))

    def _wind_state(self):
        """
        WindState of the shown transform.
        """

        return SimpleNamespace(Speed=self.shown[1] * self.BASE_SPEED, Direction=(90.0 - self.shown[0]) % 360)

    def _set_transform(self, t):
        """
        setFieldTransform: pending until the next environment update.
        """

        self.requested = t
        if not (self.rewound and self.ignore_after_rewind):
            self.pending = (t.Rotation, t.Scale)

    def getEnvironmentInterface(self):
        """
        Environment interface with the wind field control.
        """

        return self._env

    def getSimInterface(self):
        """
        Sim interface with the simulation time.
        """

        return SimpleNamespace(getSimulationTime=lambda: self.time)

    def simulateSeconds(self, dt):
        """
        Advance time; at an environment update apply the pending transform.
        """

        before, self.time = self.time, self.time + dt
        n_before, n_after = (np.floor(t - self.UPDATE_OFFSET + 1e-9) for t in (before, self.time))
        if self.updates_enabled and n_after > n_before and n_after >= 1:
            if self.pending is not None:
                self.active, self.pending = self.pending, None
            self.shown = self.active

    def SimulateRewindTo(self, t):
        """
        Back to t = 0 (as FH does for t = 0.25); the pending transform is discarded.
        """

        self.time = 0.0
        self.pending = None
        self.shown = self.active
        self.rewound = True


def _bridge(tmp_path, monkeypatch, ignore_after_rewind=False, real_dispose=False):
    """
    FhSimBridge(wind_current=True) whose initialization creates a _FakeMathModel
    (a new one per initialization; `fake` is the first). real_dispose keeps the
    bridge's own _dispose_math_model (it logs the fake's missing Terminate/Dispose).
    """

    bridge = FhSimBridge(ship_model=ShipModel.MYZAKO_37, output_path=str(tmp_path / "out"), wind_current=True)
    fakes = [_FakeMathModel(ignore_after_rewind)]

    def init(**_kwargs):
        """
        Stand-in for _initialize_math_model (transform before the first simulateSeconds).
        """

        if fakes[-1].init_transform is not None:
            fakes.append(_FakeMathModel(ignore_after_rewind))
        fake = fakes[-1]
        bridge._math_model, bridge._ship_interface = fake, fake.ship
        bridge.math_model_initialized, bridge.initialize_math_model = True, False
        if bridge.initial_wind_field_transform is not None:
            bridge._apply_wind_field_transform(*bridge.initial_wind_field_transform)
        fake.init_transform = fake.pending
        fake.simulateSeconds(0.125)

    monkeypatch.setattr(bridge, "_initialize_math_model", init)
    if not real_dispose:
        monkeypatch.setattr(bridge, "_dispose_math_model", lambda: None)
    return bridge, fakes[0]


def test_every_episode_gets_its_own_wind(tmp_path, monkeypatch):
    """
    Initialization and every rewind apply a new draw, which holds after the warm-up.
    """

    bridge, fake = _bridge(tmp_path, monkeypatch)
    rng, ref = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(3):
        bridge.reset(wind_rng=rng)
        rot, scale = draw_wind_field_transform(ref)
        assert bridge.episode_wind_field_transform == (rot, scale, 0.0)
        assert fake.active == fake.shown == (rot, scale)
    assert bridge._rewind_wind_verified
    bridge.dispose()


def test_rewind_warm_up_reaches_the_environment_update(tmp_path, monkeypatch):
    """
    With the 1.0 s warm-up after a rewind the probe saw the previous transform
    (FH run of 2026-11-05); REWIND_WIND_WARMUP_TIME must pass the first update.
    """

    monkeypatch.setattr(FhSimBridge, "REWIND_WIND_WARMUP_TIME", 1.0)
    bridge, _ = _bridge(tmp_path, monkeypatch)
    rng = np.random.default_rng(3)
    bridge.reset(wind_rng=rng)
    with pytest.raises(RuntimeError, match="ignores a wind-field transform"):
        bridge.reset(wind_rng=rng)
    bridge.dispose()


def test_ignored_transform_after_rewind_is_detected(tmp_path, monkeypatch):
    """
    If FH ignores transforms set after SimulateRewindTo, the first rewind raises.
    """

    bridge, _ = _bridge(tmp_path, monkeypatch, ignore_after_rewind=True)
    rng = np.random.default_rng(1)
    bridge.reset(wind_rng=rng)
    with pytest.raises(RuntimeError, match="ignores a wind-field transform"):
        bridge.reset(wind_rng=rng)
    bridge.dispose()


def test_wind_current_needs_an_rng(tmp_path):
    """
    reset() without wind_rng is refused with wind_current.
    """

    bridge = FhSimBridge(ship_model=ShipModel.MYZAKO_37, output_path=str(tmp_path / "out"), wind_current=True)
    with pytest.raises(ValueError, match="wind_rng"):
        bridge.reset()
    bridge.dispose()


class _FakeShipInterface:
    """
    FH ship interface stand-in: heading 90 deg (north), wind 5 m/s from 30 deg
    to starboard, sailing 2 m/s through water with a 0.5 m/s current to port.
    """

    @staticmethod
    def getShipHeading():
        """
        Heading [deg], math convention.
        """

        return 90.0

    @staticmethod
    def getWindState():
        """
        Apparent wind.
        """

        return SimpleNamespace(Speed=5.0, Direction=30.0)

    @staticmethod
    def getShipVelocityOverGround():
        """
        Body-frame velocity over ground (sway port positive).
        """

        return SimpleNamespace(x=2.0, y=0.5)

    @staticmethod
    def getShipVelocityThroughWater():
        """
        Body-frame velocity through water.
        """

        return SimpleNamespace(x=2.0, y=0.0)

    @staticmethod
    def getShipPosition():
        """
        World position [m], away from the configured start.
        """

        return SimpleNamespace(x=2250.0, y=-40.0)

    @staticmethod
    def getShipYawRate():
        """
        Yaw rate [deg/s], counter-clockwise positive.
        """

        return 0.5


class _FakeEnvBridge:
    """
    Bridge stand-in for PySimEnv: records reset() keyword arguments.
    """

    ship_model = ShipModel.MYZAKO_37
    wind_current = True

    def __init__(self):
        """
        No math model; the ship interface is _FakeShipInterface.
        """

        self.initial_vessel_state = None
        self.ship_interface = _FakeShipInterface()
        self.resets = []

    def reset(self, **kwargs):
        """
        Record the call.
        """

        self.resets.append(kwargs)

    def dispose(self):
        """
        Nothing to release.
        """


def test_fh_observed_wind_current_in_the_dataset_convention():
    """
    Wind from 30 deg starboard; current 0.5 m/s to port (negative to starboard).
    """

    from fw.dynamics.fh_tabular_model import FhTabularModel
    from fw.ships.ship_factory import create_ship

    model = FhTabularModel(create_ship(ShipModel.MYZAKO_37).specifications, _FakeEnvBridge())
    np.testing.assert_allclose(model.observed_wind_current(None),
                               [5.0 * np.cos(np.radians(30.0)), 2.5, 0.0, -0.5], atol=1e-9)


def test_env_reset_draws_the_wind_from_its_rng_and_observes_fh():
    """
    PySimEnv passes its RNG to the bridge and appends FH's normalized wind/current.
    """

    from fw.dynamics.dynamics_model import Model
    from fw.environments.py_sim_env import PySimEnv
    from fw.environments.wind_current_obs import normalize_wind_current

    bridge = _FakeEnvBridge()
    env = PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=bridge,
                   wind=True, current=True)
    obs, _ = env.reset(seed=0)
    assert bridge.resets == [{"wind_rng": env._rng}]
    np.testing.assert_allclose(obs[-4:], normalize_wind_current([5.0 * np.cos(np.radians(30.0)), 2.5, 0.0, -0.5]),
                               atol=1e-6)


def test_env_reset_starts_from_fh_state():
    """
    After reset() the env state, positions for the first step's reward and the
    observed velocities are FH's state after its reset, not the configured start.
    """

    from fw.dynamics.dynamics_model import Model
    from fw.environments.py_sim_env import PySimEnv, YAW_RATE_OBS_SCALES

    env = PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=_FakeEnvBridge(),
                   wind=True, current=True)
    obs, _ = env.reset(seed=0)
    s = env.phys_sim.state
    assert (s.x, s.y, s.u, s.v) == (2250.0, -40.0, 2.0, 0.5)
    assert s.heading == pytest.approx(np.pi / 2)
    assert s.r == pytest.approx(np.radians(0.5))
    np.testing.assert_array_equal(env.ship_pos, np.array([2250.0, -40.0], dtype=np.float32))
    np.testing.assert_array_equal(env.previous_ship_pos, env.ship_pos)
    spec = env.ship.specifications
    np.testing.assert_allclose(obs[:3], [2.0 / env._surge_cap(), 0.5 / spec.sway_limits[1],
                                         np.radians(0.5) / YAW_RATE_OBS_SCALES[Model.FH_TABULAR]], rtol=1e-6)


def test_bridge_without_wind_current_support_is_named_in_the_error():
    """
    A bridge without wind_current (as SimSyncBridge) is refused with wind on, accepted without.
    """

    from fw.dynamics.dynamics_model import Model
    from fw.environments.py_sim_env import PySimEnv

    class _NoWindBridge(_FakeEnvBridge):
        """
        _FakeEnvBridge without the wind_current attribute.
        """

        wind_current = property(lambda self: (_ for _ in ()).throw(AttributeError()))

    with pytest.raises(ValueError, match="does not support wind and current"):
        PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=_NoWindBridge(),
                 wind=True, current=True)
    PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=_NoWindBridge())


def test_non_finite_rewind_probe_fails(tmp_path, monkeypatch):
    """
    A NaN true-wind measurement does not pass the rewind check.
    """

    bridge, _ = _bridge(tmp_path, monkeypatch)
    rng = np.random.default_rng(1)
    bridge.reset(wind_rng=rng)
    monkeypatch.setattr(fh_sim_bridge, "fh_true_wind_speed", lambda _si: float("nan"))
    with pytest.raises(RuntimeError, match="ignores a wind-field transform"):
        bridge.reset(wind_rng=rng)
    bridge.dispose()


def test_user_set_initial_transform_is_refused_with_wind_current(tmp_path, monkeypatch):
    """
    reset() draws the wind itself: a caller's initial_wind_field_transform is an error.
    """

    bridge, _ = _bridge(tmp_path, monkeypatch)
    bridge.initial_wind_field_transform = (90.0, 2.0, 0.0)
    with pytest.raises(ValueError, match="initial_wind_field_transform"):
        bridge.reset(wind_rng=np.random.default_rng(0))
    bridge.dispose()


def test_reinitialization_after_dispose_draws_again(tmp_path, monkeypatch):
    """
    After dispose(), reset() re-initializes with a new draw (its own transform is not refused).
    """

    bridge, _ = _bridge(tmp_path, monkeypatch, real_dispose=True)
    rng = np.random.default_rng(5)
    bridge.reset(wind_rng=rng)
    bridge.reset(wind_rng=rng)                  # rewind
    first = bridge.initial_wind_field_transform
    bridge._dispose_math_model()
    bridge.reset(wind_rng=rng)                  # re-initialization
    assert bridge.initial_wind_field_transform != first
    assert bridge.episode_wind_field_transform == bridge.initial_wind_field_transform
    bridge.dispose()


def test_failed_initial_warm_up_disposes_the_model(tmp_path, monkeypatch):
    """
    A wind warm-up failure at initialization disposes the model, so the next reset() re-initializes.
    """

    bridge, fake = _bridge(tmp_path, monkeypatch, real_dispose=True)
    fake.updates_enabled = False                # WindState never shows the transform
    with pytest.raises(RuntimeError, match="WindState unchanged"):
        bridge.reset(wind_rng=np.random.default_rng(0))
    assert bridge.initialize_math_model and not bridge.math_model_initialized
    bridge.dispose()


def test_read_state_in_framework_conventions():
    """
    FhTabularModel.read_state: position, heading in rad (math convention), body velocities, yaw rate in rad/s.
    """

    from fw.dynamics.fh_tabular_model import FhTabularModel
    from fw.ships.ship_factory import create_ship

    model = FhTabularModel(create_ship(ShipModel.MYZAKO_37).specifications, _FakeEnvBridge())
    np.testing.assert_allclose(model.read_state(), [2250.0, -40.0, np.pi / 2, 2.0, 0.5, np.radians(0.5)])
