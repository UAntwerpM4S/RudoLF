"""
Episode start thrust: initial_thrust (thrust units) if set, else the fixed thrust
of a fixed-thrust profile, else the thrust that holds the initial speed (Fossen,
Nomoto, UGent and B01 surrogates), else min_thrust. With FH the actuator starts
there too; nothing is sent to FH.
"""

import copy
import logging
import dataclasses
import numpy as np
import pytest

from collections import defaultdict
from types import SimpleNamespace
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model

START = np.array([2245.0, -47.0], dtype=np.float32)
TARGET = np.array([-1559.0, 11990.0], dtype=np.float32)


def _fossen_env():
    """
    Small Fossen/Myzako PySimEnv.
    """

    from fw.environments.py_sim_env import PySimEnv

    return PySimEnv(time_step=2.0, max_steps=50, start_pos=START, target_pos=TARGET,
                    dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)


def _thrust_action(thrust, spec):
    """
    Thrust action that ActionMapper maps to *thrust*.
    """

    return 2.0 * (thrust - spec.min_thrust) / (spec.max_thrust - spec.min_thrust) - 1.0


def test_actuators_start_at_min_thrust_by_default():
    """
    Without a start thrust and at 0 m/s the actuators hold min_thrust, not zero.
    """

    env = _fossen_env()
    env.reset(seed=0)
    spec = env.ship.specifications
    np.testing.assert_allclose(env.phys_sim.actuators.current_actuator_state, [0.0, spec.min_thrust])
    np.testing.assert_allclose(env.performed_action, [0.0, spec.min_thrust])
    np.testing.assert_allclose(env.previous_action, env.performed_action)


@pytest.mark.parametrize("smoothing", [False, True])
def test_start_thrust_is_held_from_reset_on(smoothing):
    """
    With initial_thrust equal to the commanded thrust, the first steps have no
    thrust change (with and without smoothing).
    """

    env = _fossen_env()
    env.enable_smoothing = smoothing
    spec = env.ship.specifications
    start = 0.5 * (spec.min_thrust + spec.max_thrust)
    env.initial_thrust = start
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(start)
    for _ in range(2):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert float(env.performed_action[1]) == pytest.approx(start, abs=1e-6)
        assert float(env.performed_action[1]) == pytest.approx(float(env.previous_action[1]), abs=1e-6)


@pytest.mark.parametrize("model, ship, config", [
    (Model.NOMOTO, ShipModel.MYZAKO_37, {}),
    (Model.UGENT_SURROGATE, ShipModel.TRIPOLI, {"ugent_surrogate_config": {"ukc_percent": 363}}),
])
def test_start_thrust_for_other_python_models(model, ship, config):
    """
    Nomoto and the UGent surrogate start at an explicit initial_thrust too.
    """

    from fw.environments.py_sim_env import PySimEnv

    env = PySimEnv(time_step=2.0, max_steps=50, start_pos=START, target_pos=TARGET, dynamics_model=model,
                   ship_model=ship, **config)
    env.initial_thrust = 1.0
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.max_thrust)


def test_colav_env_starts_at_the_start_thrust():
    """
    The COLAV envs (PySimBaseEnv) start at 0 m/s, so at min_thrust whatever
    initial_speed is, or at initial_thrust.
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv

    env = PySimColavEnv(time_step=1.0, max_steps=20, ship_pos=START, target_pos=TARGET,
                        dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    env.initial_speed = 3.0
    env.reset(seed=0)
    assert env.phys_sim.state.u == 0.0
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.min_thrust)
    env.initial_thrust = 1.0
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.max_thrust)


class _FakeShipInterface:
    """
    FH ship interface stand-in: the ship at START, heading east, at rest
    (PySimEnv reads FH's state after reset).
    """

    @staticmethod
    def getShipPosition():
        """
        World position [m].
        """

        return SimpleNamespace(x=float(START[0]), y=float(START[1]))

    @staticmethod
    def getShipHeading():
        """
        Heading [deg], math convention.
        """

        return 0.0

    @staticmethod
    def getShipVelocityOverGround():
        """
        Body-frame velocity over ground.
        """

        return SimpleNamespace(x=0.0, y=0.0)

    @staticmethod
    def getShipYawRate():
        """
        Yaw rate [deg/s].
        """

        return 0.0


class _FakeFhBridge:
    """
    FH bridge stand-in: records reset() calls and accepts no start hooks.
    """

    ship_model = ShipModel.MYZAKO_37

    def __init__(self):
        """
        No math model.
        """

        self.initial_vessel_state = None
        self.ship_interface = _FakeShipInterface()
        self.math_model = None
        self.reset_calls = []

    def reset(self, *args, **kwargs):
        """
        Record the call and its arguments.
        """

        self.reset_calls.append((args, kwargs))

    def dispose(self):
        """
        Nothing to release.
        """


def test_fh_actuator_starts_at_the_start_thrust_and_nothing_is_sent_to_fh():
    """
    With FH the actuator also starts at the start thrust; the bridge gets no
    start lever or propeller rate. No steady thrust is derived for FH: at speed
    it starts at min_thrust, without a warning.
    """

    from fw.environments.py_sim_env import PySimEnv

    bridge = _FakeFhBridge()
    env = PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=bridge)
    env.initial_speed = 3.0
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.min_thrust)
    assert not getattr(env, "_warned_start_thrust", False)
    bridge.reset_calls.clear()
    env.initial_thrust = 1.0
    before = {k: v for k, v in vars(bridge).items() if k != "reset_calls"}
    env.reset(seed=0)
    assert bridge.reset_calls == [((), {})]
    assert {k: v for k, v in vars(bridge).items() if k != "reset_calls"} == before
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.max_thrust)
    assert float(env.phys_sim.actuators.current_actuator_state[1]) == pytest.approx(env.ship.specifications.max_thrust)


def test_fh_state_sync_keeps_the_exact_start_thrust():
    """
    The second reset after reading FH's state keeps the start thrust in double
    precision, also at a limit whose float32 value lies above it (0.8597).
    """

    from fw.environments.py_sim_env import PySimEnv

    env = PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=_FakeFhBridge())
    spec = dataclasses.replace(env.ship.specifications, max_thrust=0.8597)
    assert float(np.float32(spec.max_thrust)) > spec.max_thrust
    env.ship.specifications = spec
    env.phys_sim.actuators.specifications = spec
    env.initial_thrust = spec.max_thrust
    env.reset(seed=0)
    assert env.phys_sim.actuators._thrust == spec.max_thrust


def test_start_thrust_outside_its_range_is_refused():
    """
    initial_thrust must be finite; a thrust outside the ship's limits is refused
    when set, by the env and by the actuator model.
    """

    from fw.ships.actuator_model import ActuatorModel

    env = _fossen_env()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            env.initial_thrust = bad
    env.initial_thrust = None
    assert env.initial_thrust is None
    for bad in (0.05, 1.5):
        with pytest.raises(ValueError, match="outside"):
            env.initial_thrust = bad
        assert env.initial_thrust is None
    actuators = ActuatorModel(env.ship.specifications)
    for bad in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError):
            actuators.reset(bad)


def test_start_thrust_never_leaves_the_limits_by_rounding():
    """
    ActionMapper.map_thrust stays within the limits where a float32 mapping overshoots.
    """

    from fw.ships.action_mapper import ActionMapper
    from fw.ships.actuator_model import ActuatorModel
    from fw.ships.ship_factory import create_ship

    for lo, hi in ((0.0, 0.3), (0.1, 1.0), (0.2, 0.7), (0.05, 0.35), (0.3, 0.8597)):
        spec = dataclasses.replace(create_ship(ShipModel.MYZAKO_37).specifications, min_thrust=lo, max_thrust=hi)
        actuators = ActuatorModel(spec)
        for action in (-1.0, np.float32(-1.0), 1.0, np.float32(1.0), 0.3, np.float32(0.1)):
            thrust = ActionMapper.map_thrust(action, spec)
            assert lo <= thrust <= hi
            actuators.reset(thrust)
        assert ActionMapper.map_thrust(-1.0, spec) == lo and ActionMapper.map_thrust(1.0, spec) == hi


def test_profile_start_thrust_rule():
    """
    scenario.initial_thrust (thrust units) wins; else the fixed thrust mapped with
    the env's ship; else None (derived from initial_speed by the env).
    """

    from fw.configuration.config_profile import (ProfileRegistry, ScenarioConfig, ThrustConfig, ThrustMode,
                                                 VesselProfile)
    from fw.configuration.profile_applier import apply_profile

    env = _fossen_env()
    apply_profile(ProfileRegistry.get("default_profile"), env=env)
    assert env.initial_thrust is None
    fixed = VesselProfile(name="f", thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=0.0))
    apply_profile(fixed, env=env)
    assert env.initial_thrust == pytest.approx(0.55)
    both = VesselProfile(name="p", scenario=ScenarioConfig(initial_speed=3.0, initial_thrust=0.25),
                         thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=0.8))
    apply_profile(both, env=env)
    assert env.initial_thrust == 0.25
    assert VesselProfile.from_dict(both.to_dict()).scenario.initial_thrust == 0.25
    VesselProfile(name="ok", scenario=ScenarioConfig(initial_thrust=0.6)).validate()
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="initial_thrust"):
            VesselProfile(name="q", scenario=ScenarioConfig(initial_thrust=bad)).validate()


@pytest.mark.parametrize("fixed_value", [-1.0, -0.4, 0.1, 0.33, 0.8, 1.0])
@pytest.mark.parametrize("smoothing", [False, True])
def test_fixed_profile_starts_at_the_thrust_it_holds(fixed_value, smoothing):
    """
    With ThrustMode.FIXED the start thrust equals the thrust held once the fixed
    action (ActionAdapter) is applied, so the first steps have no thrust change.
    """

    from fw.configuration.config_profile import ThrustConfig, ThrustMode, VesselProfile
    from fw.configuration.profile_applier import ActionAdapter, apply_profile

    profile = VesselProfile(name="fixed", thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=fixed_value))
    env = _fossen_env()
    apply_profile(profile, env=env)
    env.enable_smoothing = smoothing
    env.reset(seed=0)
    start = float(env.performed_action[1])
    for _ in range(2):
        env.step(ActionAdapter.transform(profile, np.array([0.0, -0.7], dtype=np.float32)))
        assert float(env.performed_action[1]) == pytest.approx(start, abs=1e-6)


def test_profile_start_thrust_reaches_env_and_sub_environments():
    """
    apply_profile sets the start thrust on an env and on its sub-environments, and
    resets it for a later profile without one.
    """

    from fw.configuration.config_profile import ProfileRegistry
    from fw.configuration.profile_applier import apply_profile
    from fw.environments.multi_sim_env import MultiSimEnv
    from fw.environments.py_sim_colav_env import PySimColavEnv

    colav = PySimColavEnv(time_step=1.0, max_steps=5, ship_pos=START, target_pos=TARGET,
                          dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    multi = MultiSimEnv([_fossen_env(), _fossen_env()])
    fixed = ProfileRegistry.get("tripoli_pysim_to_fhsim_transfer_profile")     # fixed thrust 1.0
    for env in (colav, multi):
        apply_profile(fixed, env=env)
    colav.reset(seed=0)
    assert float(colav.performed_action[1]) == pytest.approx(colav.ship.specifications.max_thrust)
    multi.reset(seed=0)
    for sub in multi.sub_environments.values():
        assert float(sub.performed_action[1]) == pytest.approx(sub.ship.specifications.max_thrust)

    apply_profile(ProfileRegistry.get("default_profile"), env=colav)
    colav.reset(seed=0)
    assert float(colav.performed_action[1]) == pytest.approx(colav.ship.specifications.min_thrust)


def test_start_thrust_survives_a_deep_copy():
    """
    Worker copies of an env start at the same thrust as the template.
    """

    env = _fossen_env()
    env.initial_thrust = 0.55
    worker = copy.deepcopy(env)
    worker.reset(seed=1)
    assert worker.initial_thrust == 0.55
    assert float(worker.performed_action[1]) == pytest.approx(0.55)


def test_episode_collector_starts_at_the_real_actuator_state():
    """
    The recorded rudder/thrust series start at the actuator state after reset, so
    a held thrust shows no change at the first step.
    """

    from fw.postprocessing.episode_collector import ShipDataExtractor

    env = _fossen_env()
    env.initial_thrust = 1.0
    env.reset(seed=0)
    acc = defaultdict(list)
    ShipDataExtractor.record_initial(env, acc)
    env.step(np.array([0.0, 1.0], dtype=np.float32))
    ShipDataExtractor.record_step(env, acc)
    np.testing.assert_allclose(acc["thrust_actions"], [1.0, 1.0])
    np.testing.assert_allclose(acc["thrust_changes"], [0.0, 0.0])
    np.testing.assert_allclose(acc["rudder_actions"], [0.0, 0.0])


@pytest.mark.parametrize("fixed_value", [-1.0, 0.0, 1.0])
def test_fixed_value_accepts_the_action_range(fixed_value):
    """
    fixed_value is on the action scale: all of [-1, 1] is valid.
    """

    from fw.configuration.config_profile import VesselProfile, ThrustConfig, ThrustMode

    VesselProfile(name="fixed", thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=fixed_value)).validate()


@pytest.mark.parametrize("fixed_value", [-1.01, 1.01, float("nan")])
def test_fixed_value_outside_the_action_range_is_refused(fixed_value):
    """
    fixed_value outside [-1, 1], or NaN, is refused.
    """

    from fw.configuration.config_profile import VesselProfile, ThrustConfig, ThrustMode

    with pytest.raises(ValueError, match="fixed_value"):
        VesselProfile(name="fixed", thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=fixed_value)).validate()


@pytest.mark.parametrize("model, ship, config, speed", [
    (Model.FOSSEN, ShipModel.MYZAKO_37, {}, 3.0),
    (Model.UGENT_SURROGATE, ShipModel.TRIPOLI, {"ugent_surrogate_config": {"ukc_percent": 363}}, 3.0),
])
def test_start_thrust_holds_the_initial_speed(model, ship, config, speed):
    """
    Without initial_thrust the env starts at the thrust that holds initial_speed:
    zero surge acceleration at reset, and with that thrust commanded and the
    rudder at zero the speed stays put.
    """

    from fw.environments.py_sim_env import PySimEnv

    env = PySimEnv(time_step=1.0, max_steps=50, start_pos=START, target_pos=TARGET, dynamics_model=model,
                   ship_model=ship, **config)
    env.initial_speed = speed
    env.reset(seed=0)
    spec = env.ship.specifications
    thrust = float(env.performed_action[1])
    assert spec.min_thrust < thrust < spec.max_thrust
    straight = env.phys_sim.state
    du, dv, dr = env.dynamics._compute_accelerations(straight, 0.0, thrust)
    assert du == pytest.approx(0.0, abs=1e-6)
    action = np.array([0.0, _thrust_action(thrust, spec)], dtype=np.float32)
    for _ in range(5):
        env.step(action)
    assert env.phys_sim.state.u == pytest.approx(speed, rel=1e-3)
    assert not getattr(env, "_warned_start_thrust", False)


def test_start_thrust_falls_back_to_min_thrust_with_one_warning(caplog):
    """
    A speed no thrust can hold (Fossen/Myzako below its min-thrust speed; Nomoto,
    whose min thrust already exceeds the speed limit) starts at min_thrust and logs
    one warning over several resets.
    """

    from fw.environments.py_sim_env import PySimEnv

    for model, speed in ((Model.FOSSEN, 1.0), (Model.NOMOTO, 3.0)):
        env = PySimEnv(time_step=1.0, max_steps=50, start_pos=START, target_pos=TARGET, dynamics_model=model,
                       ship_model=ShipModel.MYZAKO_37)
        env.initial_speed = speed
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            for seed in range(3):
                env.reset(seed=seed)
                assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.min_thrust)
        assert sum("holds the initial speed" in r.getMessage() for r in caplog.records) == 1


def test_steady_thrust_is_not_derived_where_unsupported():
    """
    Models without _compute_accelerations (FH) or with supports_steady_thrust
    False (the UA surrogate) return None.
    """

    from fw.dynamics.dynamics_model import DynamicsBase
    from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate
    from fw.ships.vessel_state import VesselState

    class _NoAccelerations(DynamicsBase):
        """
        Engine-backed stand-in.
        """

    class _Unsupported(DynamicsBase):
        """
        Model that opts out.
        """

        supports_steady_thrust = False

        def _compute_accelerations(self, state, rudder_angle, thrust):
            """
            Would hold any speed.
            """

            return 0.0, 0.0, 0.0

    state = VesselState(terminated=False, truncated=False, x=0.0, y=0.0, heading=0.0, u=1.0, v=0.0, r=0.0)
    assert UAntwerpSurrogate.supports_steady_thrust is False
    assert _NoAccelerations().steady_thrust(state, 0.1, 1.0) is None
    assert _Unsupported().steady_thrust(state, 0.1, 1.0) is None


def test_explicit_and_fixed_thrust_win_over_the_derived_thrust():
    """
    At a start speed a thrust can hold, an explicit initial_thrust and then a
    fixed-thrust profile still decide the start thrust.
    """

    from fw.configuration.config_profile import ScenarioConfig, ThrustConfig, ThrustMode, VesselProfile
    from fw.configuration.profile_applier import apply_profile
    env, speed = _fossen_env(), 3.0
    spec = env.ship.specifications
    env.initial_speed = speed
    env.reset(seed=0)
    derived = float(env.performed_action[1])
    assert spec.min_thrust < derived < spec.max_thrust

    explicit = 0.5 * (derived + spec.max_thrust)
    apply_profile(VesselProfile(name="e", scenario=ScenarioConfig(initial_speed=speed, initial_thrust=explicit)),
                  env=env)
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(explicit)

    apply_profile(VesselProfile(name="f", scenario=ScenarioConfig(initial_speed=speed),
                                thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=1.0)), env=env)
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(spec.max_thrust)


def _b01_surrogate(ukc_percent):
    """
    UGent B01 surrogate at a tested UKC [%], and the B01 specifications. Tested
    at the dynamics level: the rudi_work envs refuse ShipModel.B01.
    """

    from fw.dynamics.ugent_b01_surrogate import UGentB01Surrogate
    from fw.ships.ship_factory import create_ship

    spec = create_ship(ShipModel.B01).specifications
    return UGentB01Surrogate(spec, {"ukc_percent": ukc_percent}), spec


def _straight(speed):
    """
    Straight-running start state at `speed` [m/s].
    """

    from fw.ships.vessel_state import VesselState

    return VesselState(terminated=False, truncated=False, x=0.0, y=0.0, heading=0.0, u=speed, v=0.0, r=0.0)


@pytest.mark.parametrize("ukc, speed", [(379, 0.8), (153, 0.5), (35, 0.2)])
def test_b01_steady_thrust_holds_the_speed(ukc, speed):
    """
    UGentB01Surrogate.steady_thrust: a thrust inside the limits with zero surge
    acceleration at the start speed.
    """

    model, spec = _b01_surrogate(ukc)
    state = _straight(speed)
    thrust = model.steady_thrust(state, spec.min_thrust, spec.max_thrust)
    assert thrust is not None and spec.min_thrust < thrust < spec.max_thrust
    assert model._surge_acceleration(state, thrust) == pytest.approx(0.0, abs=1e-6)


def test_b01_steady_thrust_is_none_for_an_unreachable_speed():
    """
    B01 at 35 % UKC cannot hold 0.8 m/s (at most ~0.43 m/s): no steady thrust.
    """

    model, spec = _b01_surrogate(35)
    assert model.steady_thrust(_straight(0.8), spec.min_thrust, spec.max_thrust) is None


def test_b01_steady_thrust_keeps_the_low_thrust_warning():
    """
    Deriving the steady thrust probes thrusts below B01's straight-running minimum
    without using up the low-thrust warning of the surrogate.
    """

    model, spec = _b01_surrogate(35)
    assert model.min_straight_running_thrust > spec.min_thrust
    model.steady_thrust(_straight(0.2), spec.min_thrust, spec.max_thrust)
    assert model._warned_low_thrust is False


def test_initial_speed_must_be_finite_and_non_negative():
    """
    NaN, inf and negative start speeds are refused by the env and by profile validation.
    """

    from fw.configuration.config_profile import ScenarioConfig, VesselProfile

    env = _fossen_env()
    for bad in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError, match="initial_speed"):
            env.initial_speed = bad
        with pytest.raises(ValueError, match="initial_speed"):
            VesselProfile(name="bad", scenario=ScenarioConfig(initial_speed=bad)).validate()
    assert env.initial_speed == 0.0


def test_initial_speed_outside_the_surge_limits_is_refused():
    """
    PySimEnv refuses a start speed above the ship's surge limit, also via a profile;
    the limit itself is accepted.
    """

    from fw.configuration.config_profile import ScenarioConfig, VesselProfile
    from fw.configuration.profile_applier import apply_profile

    env = _fossen_env()
    high = env.ship.specifications.surge_limits[1]
    with pytest.raises(ValueError, match="surge limits"):
        env.initial_speed = high + 0.1
    with pytest.raises(ValueError, match="surge limits"):
        apply_profile(VesselProfile(name="fast", scenario=ScenarioConfig(initial_speed=high + 2.0)), env=env)
    env.initial_speed = high
    assert env.initial_speed == high


def test_profile_initial_thrust_outside_the_limits_is_refused_when_applied():
    """
    A profile initial_thrust outside the ship's thrust limits fails in apply_profile,
    before any reset, for the env and for MultiSimEnv sub-environments.
    """

    from fw.configuration.config_profile import ScenarioConfig, VesselProfile
    from fw.configuration.profile_applier import apply_profile
    from fw.environments.multi_sim_env import MultiSimEnv
    from fw.environments.py_sim_colav_env import PySimColavEnv

    colav = PySimColavEnv(time_step=1.0, max_steps=5, ship_pos=START, target_pos=TARGET,
                          dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    profile = VesselProfile(name="hot", scenario=ScenarioConfig(initial_thrust=1.5))
    for env in (_fossen_env(), colav, MultiSimEnv([_fossen_env(), _fossen_env()])):
        with pytest.raises(ValueError, match="outside"):
            apply_profile(profile, env=env)
