"""
Observation rework of PR 431 and its review fixes:

- yaw-rate observation scale per dynamics model (PySimEnv and, separately, the COLAV envs);
- COLAV observation: rudder and thrust on the action scale, inside the Box;
- a saved policy that does not fit the env is refused at load time;
- visualization raises on a missing or incompatible policy, training falls back;
- ShipSpecifications refuses actuator ranges the observation cannot normalize by.
"""

import dataclasses
import numpy as np
import pytest

from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model

START = np.array([2245.0, -47.0], dtype=np.float32)
TARGET = np.array([-1559.0, 11990.0], dtype=np.float32)


# ----------------------------------------------------------------------
# Yaw-rate scale per dynamics model
# ----------------------------------------------------------------------
@pytest.mark.parametrize("model", [Model.FOSSEN, Model.NOMOTO])
def test_pysimenv_yaw_rate_is_scaled_per_dynamics_model(model):
    """
    The observed yaw rate uses the scale of the env's dynamics model.
    """

    from fw.environments.py_sim_env import PySimEnv, YAW_RATE_OBS_SCALES

    env = PySimEnv(time_step=2.0, max_steps=50, start_pos=START, target_pos=TARGET,
                   dynamics_model=model, ship_model=ShipModel.MYZAKO_37)
    env.reset(seed=0)
    obs, *_ = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert env.yaw_rate_obs_scale == YAW_RATE_OBS_SCALES[model]
    assert float(obs[2]) == pytest.approx(float(np.clip(env.phys_sim.state.r / YAW_RATE_OBS_SCALES[model], -1, 1)))


def test_pysimenv_scales_cover_every_model_it_accepts():
    """
    Every dynamics model has a scale; FH and UA share theirs (UA is trained on FH data).
    """

    from fw.environments.py_sim_env import YAW_RATE_OBS_SCALES

    from fw.dynamics.ugent_surrogate import SUPPORTED_UKC_PERCENT
    from fw.environments.py_sim_env import UGENT_YAW_RATE_OBS_SCALES

    assert set(YAW_RATE_OBS_SCALES) == set(Model) - {Model.UGENT_SURROGATE, Model.UGENT_B01_SURROGATE}
    assert set(UGENT_YAW_RATE_OBS_SCALES) == set(SUPPORTED_UKC_PERCENT)
    assert YAW_RATE_OBS_SCALES[Model.FH_TABULAR] == YAW_RATE_OBS_SCALES[Model.UANTWERP]


def test_pysimenv_without_a_yaw_scale_raises(monkeypatch):
    """
    A dynamics model without a scale is refused at construction.
    """

    import fw.environments.py_sim_env as module

    monkeypatch.delitem(module.YAW_RATE_OBS_SCALES, Model.FOSSEN)
    with pytest.raises(ValueError, match="yaw-rate scale"):
        module.PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                        dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)


def test_colav_yaw_scales_are_separate_from_pysimenv():
    """
    The COLAV envs keep their own scale table.
    """

    import fw.environments.py_sim_base_env as colav_base
    import fw.environments.py_sim_env as py_sim_env

    from fw.dynamics.ugent_surrogate import SUPPORTED_UKC_PERCENT

    assert colav_base.YAW_RATE_OBS_SCALES is not py_sim_env.YAW_RATE_OBS_SCALES
    assert colav_base.UGENT_YAW_RATE_OBS_SCALES is not py_sim_env.UGENT_YAW_RATE_OBS_SCALES
    assert set(colav_base.YAW_RATE_OBS_SCALES) == {Model.FOSSEN, Model.NOMOTO}
    assert set(colav_base.UGENT_YAW_RATE_OBS_SCALES) == set(SUPPORTED_UKC_PERCENT)


def test_colav_without_a_yaw_scale_raises(monkeypatch):
    """
    A COLAV env with a dynamics model without a scale is refused at construction.
    """

    import fw.environments.py_sim_base_env as module
    from fw.environments.py_sim_colav_env import PySimColavEnv

    monkeypatch.delitem(module.YAW_RATE_OBS_SCALES, Model.FOSSEN)
    with pytest.raises(ValueError, match="yaw-rate scale"):
        PySimColavEnv(time_step=1.0, max_steps=5, ship_pos=START, target_pos=TARGET,
                      dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)


# ----------------------------------------------------------------------
# COLAV observation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("ship_model", [ShipModel.MYZAKO_37, ShipModel.TRIPOLI])
def test_colav_rudder_and_thrust_are_on_the_action_scale(ship_model):
    """
    Myzako's 75 deg rudder used to be observed in rad (up to 1.309), outside the Box.
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv
    from fw.environments.py_sim_base_env import PySimBaseEnv

    env = PySimColavEnv(time_step=1.0, max_steps=200, ship_pos=START, target_pos=TARGET,
                        dynamics_model=Model.FOSSEN, ship_model=ship_model)
    env.enable_smoothing = False
    obs, _ = env.reset(seed=0)
    base = PySimBaseEnv._get_obs(env)
    assert base.shape == (13,)
    assert base[12] == -1.0                                     # min_thrust after reset
    assert env.observation_space.contains(obs)
    for action in ([1.0, 0.5], [-1.0, 0.5]):
        obs, *_ = env.step(np.array(action, dtype=np.float32))
        assert env.observation_space.contains(obs)
        base = PySimBaseEnv._get_obs(env)
        assert base[10] == pytest.approx(action[0], abs=1e-6)
        assert base[12] == pytest.approx(0.5, abs=1e-6)


# ----------------------------------------------------------------------
# Policy size check
# ----------------------------------------------------------------------
def _fossen_env():
    """
    Small Fossen/Myzako PySimEnv.
    """

    from fw.environments.py_sim_env import PySimEnv

    return PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                    dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)


def test_single_agent_refuses_a_policy_of_another_observation_size(tmp_path):
    """
    A 12-input policy is refused with a ValueError and the current policy is kept.
    """

    from fw.policies.ppo_policy import PPOPolicy
    from fw.policies.single_ppo_agent import SinglePPOAgent

    PPOPolicy(12, 2, "cpu").save(str(tmp_path / "old_policy"))
    agent = SinglePPOAgent(environment=_fossen_env(), model_dir=str(tmp_path))
    before = agent.policy
    with pytest.raises(ValueError, match="12/2.*13/2"):
        agent.load_policy("old_policy")
    assert agent.policy is before


def test_single_agent_loads_a_fitting_policy(tmp_path):
    """
    A policy of the env's size loads.
    """

    from fw.policies.ppo_policy import PPOPolicy
    from fw.policies.single_ppo_agent import SinglePPOAgent

    PPOPolicy(13, 2, "cpu").save(str(tmp_path / "policy"))
    agent = SinglePPOAgent(environment=_fossen_env(), model_dir=str(tmp_path))
    agent.load_policy("policy")
    assert agent.policy.input_dim == 13


def _manager_with_old_policy(tmp_path, manager_cls):
    """
    Agent manager (without __init__) whose model dir holds a 12-input policy.
    """

    from fw.policies.ppo_policy import PPOPolicy

    PPOPolicy(12, 2, "cpu").save(str(tmp_path / "old_PPO_policy"))
    mgr = manager_cls.__new__(manager_cls)
    mgr._agent = None
    mgr._agent_type = "PPO"
    mgr._profile = None
    mgr._hyperparameters = {"model_dir": str(tmp_path)}
    return mgr


@pytest.mark.parametrize("manager", ["agent_manager", "agent_manager_simple"])
def test_training_falls_back_and_visualization_raises_on_an_old_policy(tmp_path, manager, monkeypatch):
    """
    Without require_policy a misfit policy falls back to a fresh one; with it, misfit and missing policies raise.
    """

    import importlib
    from fw.configuration.config import PolicyStrategy

    module = importlib.import_module(f"fw.scheduling.{manager}")
    cls = module.AgentManager if manager == "agent_manager" else module.AgentManagerSimple
    monkeypatch.setattr(module, "apply_profile", lambda *a, **k: None)
    mgr = _manager_with_old_policy(tmp_path, cls)

    mgr.set_environment(_fossen_env(), PolicyStrategy.REUSE_OTHER_POLICY, "old")
    assert mgr._agent.policy.input_dim == 13                    # fresh policy kept

    with pytest.raises(ValueError, match="observation/action size"):
        mgr.set_environment(_fossen_env(), PolicyStrategy.REUSE_OTHER_POLICY, "old", require_policy=True)
    with pytest.raises(FileNotFoundError):
        mgr.set_environment(_fossen_env(), PolicyStrategy.REUSE_OTHER_POLICY, "missing", require_policy=True)


# ----------------------------------------------------------------------
# ShipSpecifications
# ----------------------------------------------------------------------
@pytest.mark.parametrize("change", [dict(max_rudder_angle=0.0), dict(max_rudder_angle=float("nan")),
                                    dict(min_thrust=1.0, max_thrust=1.0), dict(max_thrust=float("nan"))])
def test_ship_specifications_refuse_degenerate_actuator_ranges(change):
    """
    max_rudder_angle must be positive and max_thrust must exceed min_thrust (NaN refused).
    """

    from fw.ships.ship_factory import create_ship

    spec = create_ship(ShipModel.MYZAKO_37).specifications
    with pytest.raises(ValueError):
        dataclasses.replace(spec, **change)


# ----------------------------------------------------------------------
# compute_wind_current_obs_scales
# ----------------------------------------------------------------------
def test_yaw_scale_suggestion_ignores_non_finite_rows():
    """
    suggest_yaw_scale skips non-finite rows and validates its inputs.
    """

    from fw.surrogate.compute_wind_current_obs_scales import suggest_yaw_scale

    r = suggest_yaw_scale({"yaw_rate": np.array([0.01, -0.03, np.nan, np.inf, 0.02])}, 100.0)
    assert r == {"percentile": pytest.approx(0.03), "max": pytest.approx(0.03), "rows": 3}
    with pytest.raises(ValueError):
        suggest_yaw_scale({"yaw_rate": np.array([np.nan])}, 99.9)
    with pytest.raises(ValueError):
        suggest_yaw_scale({"yaw_rate": np.array([0.01])}, 101.0)


# ----------------------------------------------------------------------
# Episode start thrust
# ----------------------------------------------------------------------
def test_actuators_start_at_min_thrust_by_default():
    """
    Without a start thrust the actuators hold min_thrust (observed as -1), not zero.
    """

    env = _fossen_env()
    obs, _ = env.reset(seed=0)
    spec = env.ship.specifications
    np.testing.assert_allclose(env.phys_sim.actuators.current_actuator_state, [0.0, spec.min_thrust])
    np.testing.assert_allclose(env.performed_action, [0.0, spec.min_thrust])
    np.testing.assert_allclose(env.previous_action, env.performed_action)
    assert float(obs[12]) == pytest.approx(-1.0)


@pytest.mark.parametrize("smoothing", [False, True])
def test_start_thrust_is_held_from_reset_on(smoothing):
    """
    With initial_thrust equal to the commanded thrust, the thrust is observed at
    that value from reset on and the first step has no thrust change.
    """

    env = _fossen_env()
    env.enable_smoothing = smoothing
    env.initial_thrust = 0.5
    obs, _ = env.reset(seed=0)
    assert float(obs[12]) == pytest.approx(0.5, abs=1e-6)
    obs, *_ = env.step(np.array([0.0, 0.5], dtype=np.float32))
    assert float(obs[12]) == pytest.approx(0.5, abs=1e-6)
    assert float(env.performed_action[1]) == pytest.approx(float(env.previous_action[1]))


def test_start_thrust_outside_the_action_range_is_refused():
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
    obs, _ = colav.reset(seed=0)
    assert float(colav.performed_action[1]) == pytest.approx(colav.ship.specifications.max_thrust)
    multi.reset(seed=0)
    for sub in multi.sub_environments.values():
        assert float(sub.performed_action[1]) == pytest.approx(sub.ship.specifications.max_thrust)

    apply_profile(ProfileRegistry.get("default_profile"), env=colav)
    colav.reset(seed=0)
    assert float(colav.performed_action[1]) == pytest.approx(colav.ship.specifications.min_thrust)


def test_fh_gets_lever_and_propeller_rate_of_the_start_thrust():
    """
    Before FH's reset the bridge gets the start thrust as engine lever and
    initial_propeller_rps = 5.5 * start thrust, as the dataset generator sets them.
    """

    from fw.environments.py_sim_env import PySimEnv
    from fw.simulators.fh_interface_utils import FH_PROPELLER_RPS_PER_THRUST
    from tests.fw.test_fh_wind_current import _FakeEnvBridge

    class _Bridge(_FakeEnvBridge):
        """
        Has FhSimBridge's start hooks; records them as they are at reset().
        """

        initial_propeller_rps = None
        initial_engine_lever = None

        def reset(self, **kwargs):
            """
            Record what the math model would be initialized with.
            """

            self.at_reset = (self.initial_engine_lever, self.initial_propeller_rps)
            super().reset(**kwargs)

    bridge = _Bridge()
    env = PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=bridge,
                   wind=True, current=True)
    assert FH_PROPELLER_RPS_PER_THRUST == 5.5
    env.reset(seed=0)
    spec = env.ship.specifications
    assert bridge.at_reset == (pytest.approx(spec.min_thrust), pytest.approx(5.5 * spec.min_thrust))
    assert float(env.performed_action[1]) == pytest.approx(spec.min_thrust)   # also after the FH state sync


def test_bridge_without_start_hooks_is_left_alone():
    """
    A bridge without the start hooks (e.g. simSync) gets no new attributes.
    """

    from fw.environments.py_sim_env import PySimEnv
    from tests.fw.test_fh_wind_current import _FakeEnvBridge

    bridge = _FakeEnvBridge()
    env = PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=bridge,
                   wind=True, current=True)
    env.reset(seed=0)
    assert not hasattr(bridge, "initial_propeller_rps") and not hasattr(bridge, "initial_engine_lever")


def test_fh_bridge_sets_the_lever_before_the_warm_up_and_warns_on_a_late_rps_change(tmp_path, monkeypatch, caplog):
    """
    FhSimBridge sets initial_engine_lever on the propellers after a rewind, before
    the wind warm-up, and warns once when initial_propeller_rps changes on an
    initialized model. (The initialization path needs FH itself.)
    """

    import logging
    from types import SimpleNamespace
    from tests.fw.test_fh_wind_current import _bridge

    bridge, fake = _bridge(tmp_path, monkeypatch)
    calls = []
    propeller = SimpleNamespace(engage=lambda: calls.append(("engage", fake.time)),
                                setEngineLeverValue=lambda v: calls.append((v, fake.time)))
    fake.ship.getPropellerControls = lambda: [propeller]
    rng = np.random.default_rng(0)
    bridge.initial_engine_lever, bridge.initial_propeller_rps = 0.4, 2.2
    bridge.reset(wind_rng=rng)                                  # stand-in initialization
    bridge._initialized_propeller_rps = 2.2
    calls.clear()

    bridge.reset(wind_rng=rng)                                  # rewind
    assert calls[:2] == [("engage", 0.0), (0.4, 0.0)]           # at the rewind point, before the warm-up
    assert fake.time > 0.0

    bridge.initial_propeller_rps = 5.5
    with caplog.at_level(logging.WARNING):
        bridge.reset(wind_rng=rng)
        bridge.reset(wind_rng=rng)
    assert sum("initial_propeller_rps changed" in r.getMessage() for r in caplog.records) == 1


def test_tripoli_start_lever_goes_to_the_azipods(tmp_path):
    """
    For the Tripoli the start lever is set on the azipods.
    """

    from types import SimpleNamespace
    from fw.simulators.fh_sim_bridge import FhSimBridge

    bridge = FhSimBridge(ship_model=ShipModel.TRIPOLI, output_path=str(tmp_path / "out"))
    levers = []
    azipod = SimpleNamespace(setEngineLeverValue=levers.append)
    bridge._ship_interface = SimpleNamespace(getAzipodControls=lambda: [azipod, azipod])
    bridge._apply_initial_engine_lever()
    assert levers == []                                         # no lever set: nothing sent
    bridge.initial_engine_lever = 1.0
    bridge._apply_initial_engine_lever()
    assert levers == [1.0, 1.0]


# ----------------------------------------------------------------------
# Second review round
# ----------------------------------------------------------------------
def test_ugent_yaw_scale_follows_the_ukc():
    """
    The UGent surrogate's yaw-rate scale is per UKC, in PySimEnv and in the COLAV envs.
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv
    from fw.environments.py_sim_env import PySimEnv

    for ukc, scale in ((363, 0.06), (125, 0.14), (43, 0.06)):
        kwargs = dict(dynamics_model=Model.UGENT_SURROGATE, ship_model=ShipModel.TRIPOLI,
                      ugent_surrogate_config={"ukc_percent": ukc}, target_pos=TARGET, max_steps=5)
        assert PySimEnv(time_step=2.0, start_pos=START, **kwargs).yaw_rate_obs_scale == scale
        assert PySimColavEnv(time_step=1.0, ship_pos=START, **kwargs).yaw_rate_obs_scale == scale


def test_start_thrust_never_leaves_the_limits_by_rounding():
    """
    ActionMapper.map_thrust stays within the limits for limits where the float32
    mapping overshoots (max_thrust 0.3: 0.30000001).
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
            actuators.reset(thrust)                             # not refused
        assert ActionMapper.map_thrust(-1.0, spec) == lo and ActionMapper.map_thrust(1.0, spec) == hi


@pytest.mark.parametrize("fixed_value", [0.1, 0.33, 0.8, 1.0])
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


def test_start_thrust_survives_a_deep_copy():
    """
    Worker copies of an env start at the same thrust as the template.
    """

    import copy

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

    from collections import defaultdict
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


def _multi_archive(tmp_path, name, dims):
    """
    Write a multi-agent policy archive with one policy per entry of dims (input sizes).
    """

    import torch
    import zipfile
    from fw.policies.ppo_policy import PPOPolicy

    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for i, dim in enumerate(dims):
            file = tmp_path / f"policy_{i}.pth"
            torch.save(PPOPolicy(dim, 2, "cpu").serialize_policy(), file)
            archive.write(file, f"policy_{i}.pth")
    return name


def test_multi_agent_size_mismatch_is_a_value_error_and_changes_nothing(tmp_path):
    """
    MultiPPOAgent.load_policy raises ValueError (not RuntimeError) for a policy of
    another size and leaves every agent's policy and optimizer as they were.
    """

    from fw.environments.multi_sim_env import MultiSimEnv
    from fw.policies.multi_ppo_agent import MultiPPOAgent
    from fw.policies.single_ppo_agent import SinglePPOAgent

    env = MultiSimEnv([_fossen_env(), _fossen_env()])
    agents = [SinglePPOAgent(environment=sub, model_dir=str(tmp_path)) for sub in env.sub_environments.values()]
    agent = MultiPPOAgent(None, environment=env, agents=agents, model_dir=str(tmp_path))
    before = [(a.policy, a.optimizer) for a in agents]
    with pytest.raises(ValueError, match="observation/action size"):
        agent.load_policy(_multi_archive(tmp_path, "mixed", [13, 12]))
    assert [(a.policy, a.optimizer) for a in agents] == before

    agent.load_policy(_multi_archive(tmp_path, "fitting", [13, 13]))
    assert all(a.policy is not p for a, (p, _) in zip(agents, before))


def test_trainer_and_scheduler_require_the_policy_for_visualization():
    """
    Trainer.visualize and "visualize" pipeline blocks ask set_environment to raise
    on a missing or unfitting policy; "train" blocks do not.
    """

    import inspect
    from fw.scheduling.pipeline_scheduler import PipelineScheduler
    from fw.scheduling.trainer import Trainer

    assert "require_policy=True" in inspect.getsource(Trainer.visualize)
    assert 'require_policy=block.action == "visualize"' in inspect.getsource(PipelineScheduler)


def test_trainer_visualize_raises_when_the_policy_cannot_be_loaded():
    """
    A policy that cannot be loaded raises out of Trainer.visualize; a RuntimeError
    of the visualization itself is only logged.
    """

    from types import SimpleNamespace
    from fw.scheduling.trainer import Trainer

    gym_env = SimpleNamespace(training_name="run", type_name="PySimEnv")
    trainer = Trainer.__new__(Trainer)
    trainer._config = {"visualize_best_trained": False, "live_animation": False}

    def failing_load(**_kwargs):
        """
        set_environment stand-in: the policy cannot be loaded.
        """

        raise RuntimeError("Failed to load the policy")

    trainer._agent_mgr = SimpleNamespace(set_environment=failing_load)
    with pytest.raises(RuntimeError, match="Failed to load"):
        trainer.visualize([SimpleNamespace(gym_env=gym_env)])

    def failing_run(**_kwargs):
        """
        visualize_trained_agent stand-in: the rollout fails.
        """

        raise RuntimeError("rollout failed")

    trainer._agent_mgr = SimpleNamespace(set_environment=lambda **_k: None, visualize_trained_agent=failing_run)
    trainer.visualize([SimpleNamespace(gym_env=gym_env)])       # logged, not raised


@pytest.mark.parametrize("error", [FileNotFoundError("no Best_ policy"), ValueError("size"), RuntimeError("load")])
def test_failed_evaluation_after_training_keeps_the_metrics(error, caplog):
    """
    If the evaluation after a successful training cannot load its policy, the
    error is logged and the training metrics are still returned.
    """

    import logging
    from types import SimpleNamespace
    from fw.scheduling.trainer import Trainer

    class _Env:
        """
        Env wrapper stand-in whose str() is its metrics key.
        """

        def __str__(self):
            """
            Metrics key.
            """

            return "PySimEnv"

    trainer = Trainer.__new__(Trainer)
    trainer._envs = [_Env()]
    trainer._scheduler = SimpleNamespace(execute=lambda _c: {"PySimEnv": {"status": "success", "duration": 1.0}})

    def visualize(*_args, **_kwargs):
        """
        Trainer.visualize stand-in: the policy cannot be loaded.
        """

        raise error

    trainer.visualize = visualize
    with caplog.at_level(logging.ERROR):
        metrics = trainer.train({})
    assert metrics == {"PySimEnv": {"status": "success", "duration": 1.0}}
    assert any("Evaluation after training" in r.getMessage() for r in caplog.records)


def test_multi_policy_agent_size_mismatch_changes_nothing(tmp_path):
    """
    MultiPolicySingleAgent.load_policy raises ValueError for a policy of another
    size and keeps all policies and optimizers; a fitting archive replaces both.
    """

    import torch
    from fw.policies.multi_policy_single_agent import MultiPolicySingleAgent
    from fw.policies.ppo_policy import PPOPolicy

    agent = MultiPolicySingleAgent.__new__(MultiPolicySingleAgent)
    agent.num_policies, agent.policy_obs_dim, agent.output_dim = 2, 13, 2
    agent.device, agent.model_dir, agent.learning_rate = torch.device("cpu"), str(tmp_path), 1e-3
    agent.policies = [PPOPolicy(13, 2, "cpu") for _ in range(2)]
    agent.optimizers = [torch.optim.Adam(p.network.parameters()) for p in agent.policies]
    policies, optimizers = list(agent.policies), list(agent.optimizers)

    with pytest.raises(ValueError, match="observation/action size"):
        agent.load_policy(_multi_archive(tmp_path, "mixed", [13, 12]))
    assert agent.policies == policies and agent.optimizers == optimizers

    agent.load_policy(_multi_archive(tmp_path, "fitting", [13, 13]))
    for policy, optimizer, old in zip(agent.policies, agent.optimizers, policies):
        assert policy is not old
        assert optimizer.param_groups[0]["params"][0] is next(policy.network.parameters())
