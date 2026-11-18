"""
Episode start thrust: the actuators start an episode at a thrust they can hold
(initial_thrust, on the action scale; default min_thrust), for every dynamics
model and env. With FH the actuator starts there too; nothing is sent to FH.
"""

import copy
import dataclasses
import numpy as np
import pytest

from collections import defaultdict
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


def test_actuators_start_at_min_thrust_by_default():
    """
    Without a start thrust the actuators hold min_thrust, not zero.
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
    env.initial_thrust = 0.5
    env.reset(seed=0)
    spec = env.ship.specifications
    start = spec.min_thrust + 0.75 * (spec.max_thrust - spec.min_thrust)
    assert float(env.performed_action[1]) == pytest.approx(start)
    for _ in range(2):
        env.step(np.array([0.0, 0.5], dtype=np.float32))
        assert float(env.performed_action[1]) == pytest.approx(start, abs=1e-6)
        assert float(env.performed_action[1]) == pytest.approx(float(env.previous_action[1]), abs=1e-6)


@pytest.mark.parametrize("model, ship, config", [
    (Model.NOMOTO, ShipModel.MYZAKO_37, {}),
    (Model.UGENT_SURROGATE, ShipModel.TRIPOLI, {"ugent_surrogate_config": {"ukc_percent": 363}}),
])
def test_start_thrust_for_other_python_models(model, ship, config):
    """
    Nomoto and the UGent surrogate start at the start thrust too.
    """

    from fw.environments.py_sim_env import PySimEnv

    env = PySimEnv(time_step=2.0, max_steps=50, start_pos=START, target_pos=TARGET, dynamics_model=model,
                   ship_model=ship, **config)
    env.initial_thrust = 1.0
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.max_thrust)


def test_towing_tank_env_with_b01_starts_at_the_start_thrust():
    """
    PySimTowingTankEnv with the UGent B01 surrogate: default min_thrust (B01's own), or initial_thrust.
    """

    from fw.environments.py_sim_towing_tank import PySimTowingTankEnv

    env = PySimTowingTankEnv(dynamics_model=Model.UGENT_B01_SURROGATE, ship_model=ShipModel.B01,
                             ugent_b01_surrogate_config={"ukc_percent": 153})
    env.reset(seed=0)
    spec = env.ship.specifications
    assert float(env.performed_action[1]) == pytest.approx(spec.min_thrust)
    env.initial_thrust = 1.0
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(spec.max_thrust)
    np.testing.assert_allclose(env.previous_action, env.performed_action)


def test_colav_env_starts_at_the_start_thrust():
    """
    The COLAV envs (PySimBaseEnv) start at the start thrust.
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv

    env = PySimColavEnv(time_step=1.0, max_steps=20, ship_pos=START, target_pos=TARGET,
                        dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.min_thrust)
    env.initial_thrust = 1.0
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.max_thrust)


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
        self.ship_interface = None
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
    start lever or propeller rate.
    """

    from fw.environments.py_sim_env import PySimEnv

    bridge = _FakeFhBridge()
    env = PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=bridge)
    env.initial_thrust = 1.0
    before = {k: v for k, v in vars(bridge).items() if k != "reset_calls"}
    env.reset(seed=0)
    assert bridge.reset_calls == [((), {})]
    assert {k: v for k, v in vars(bridge).items() if k != "reset_calls"} == before
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.max_thrust)


def test_start_thrust_outside_its_range_is_refused():
    """
    initial_thrust is on the action scale; the actuator refuses a thrust outside its limits.
    """

    from fw.ships.actuator_model import ActuatorModel

    env = _fossen_env()
    for bad in (1.5, -1.5, float("nan")):
        with pytest.raises(ValueError):
            env.initial_thrust = bad
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

    for lo, hi in ((0.0, 0.3), (0.1, 1.0), (0.2, 0.7), (0.05, 0.35)):
        spec = dataclasses.replace(create_ship(ShipModel.MYZAKO_37).specifications, min_thrust=lo, max_thrust=hi)
        actuators = ActuatorModel(spec)
        for action in (-1.0, np.float32(-1.0), 1.0, np.float32(1.0), 0.3, np.float32(0.1)):
            thrust = ActionMapper.map_thrust(action, spec)
            assert lo <= thrust <= hi
            actuators.reset(thrust)
        assert ActionMapper.map_thrust(-1.0, spec) == lo and ActionMapper.map_thrust(1.0, spec) == hi


def test_profile_start_thrust_rule():
    """
    scenario.initial_thrust wins; else the fixed thrust value; else -1 (min_thrust).
    """

    from fw.configuration.config_profile import (ProfileRegistry, ScenarioConfig, ThrustConfig, ThrustMode,
                                                 VesselProfile)

    assert ProfileRegistry.get("default_profile").initial_thrust_action() == -1.0
    assert ProfileRegistry.get("ugent_surrogate_profile").initial_thrust_action() == 1.0
    profile = VesselProfile(name="p", scenario=ScenarioConfig(initial_speed=3.0, initial_thrust=0.25),
                            thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=0.8))
    assert profile.initial_thrust_action() == 0.25
    assert VesselProfile.from_dict(profile.to_dict()).scenario.initial_thrust == 0.25
    with pytest.raises(ValueError):
        VesselProfile(name="q", scenario=ScenarioConfig(initial_thrust=1.5)).validate()


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
    env.initial_thrust = 0.5
    worker = copy.deepcopy(env)
    worker.reset(seed=1)
    assert worker.initial_thrust == 0.5
    assert float(worker.performed_action[1]) == pytest.approx(0.1 + 0.75 * 0.9)


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


def test_towing_tank_multi_agent_env_starts_at_the_start_thrust():
    """
    PySimTowingTankMultiAgentEnv (towing-tank COLAV base) starts at the start thrust.
    """

    from fw.environments.py_sim_towing_tank_multiagent import PySimTowingTankMultiAgentEnv

    env = PySimTowingTankMultiAgentEnv(dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.min_thrust)
    env.initial_thrust = 1.0
    env.reset(seed=0)
    assert float(env.performed_action[1]) == pytest.approx(env.ship.specifications.max_thrust)
    np.testing.assert_allclose(env.previous_action, env.performed_action)


def test_b01_profile_starts_at_min_thrust():
    """
    ugent_b01_surrogate_profile sets initial_thrust -1 (min_thrust).
    """

    from fw.configuration.config_profile import ProfileRegistry

    assert ProfileRegistry.get("ugent_b01_surrogate_profile").scenario.initial_thrust == -1.0


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
