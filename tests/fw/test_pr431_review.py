"""
Review fixes kept on top of PR 431 (no change to observation or dynamics):

- a saved policy that does not fit the env is refused at load time, without
  changing any agent;
- visualization raises on a missing or incompatible policy, training falls back,
  and the evaluation after training keeps the training metrics;
- ShipSpecifications refuses actuator ranges the observation cannot normalize by;
- evaluation series start at the env's actuator state;
- compute_wind_current_obs_scales suggests a yaw-rate scale.
"""

import dataclasses
import numpy as np
import pytest

from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model

START = np.array([2245.0, -47.0], dtype=np.float32)
TARGET = np.array([-1559.0, 11990.0], dtype=np.float32)


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


def test_episode_collector_starts_at_the_env_actuator_state():
    """
    The recorded rudder/thrust series start at performed_action after reset.
    """

    from collections import defaultdict
    from fw.postprocessing.episode_collector import ShipDataExtractor

    env = _fossen_env()
    env.reset(seed=0)
    env.performed_action = np.array([0.05, 0.4], dtype=np.float32)
    acc = defaultdict(list)
    ShipDataExtractor.record_initial(env, acc)
    np.testing.assert_allclose(acc["rudder_actions"], [0.05])
    np.testing.assert_allclose(acc["thrust_actions"], [0.4])
    assert acc["rudder_changes"] == [0.0] and acc["thrust_changes"] == [0.0]


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


def test_fh_and_ua_keep_the_yaw_scale_of_pr431():
    """
    FH and UA keep 0.04; the per-model scales only concern the other models.
    """

    from fw.dynamics.ugent_surrogate import SUPPORTED_UKC_PERCENT
    from fw.environments.py_sim_env import UGENT_YAW_RATE_OBS_SCALES, YAW_RATE_OBS_SCALES

    assert YAW_RATE_OBS_SCALES[Model.FH_TABULAR] == YAW_RATE_OBS_SCALES[Model.UANTWERP] == 0.04
    assert set(YAW_RATE_OBS_SCALES) == set(Model) - {Model.UGENT_SURROGATE, Model.UGENT_B01_SURROGATE}
    assert set(UGENT_YAW_RATE_OBS_SCALES) == set(SUPPORTED_UKC_PERCENT)


def test_pysimenv_without_a_yaw_scale_raises(monkeypatch):
    """
    A dynamics model without a scale is refused at construction.
    """

    import fw.environments.py_sim_env as module

    monkeypatch.delitem(module.YAW_RATE_OBS_SCALES, Model.FOSSEN)
    with pytest.raises(ValueError, match="yaw-rate scale"):
        module.PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                        dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)


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


def test_colav_yaw_scales_are_separate_from_pysimenv():
    """
    The COLAV envs keep their own scale tables.
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


@pytest.mark.parametrize("ship_model", [ShipModel.MYZAKO_37, ShipModel.TRIPOLI])
def test_colav_rudder_and_thrust_are_on_the_action_scale(ship_model):
    """
    Myzako's 75 deg rudder used to be observed in rad (up to 1.309), outside the
    Box; rudder and thrust are now observed on the action scale, as in PySimEnv
    (13 base features, thrust -1 before the first step).
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv
    from fw.environments.py_sim_base_env import PySimBaseEnv

    env = PySimColavEnv(time_step=1.0, max_steps=200, ship_pos=START, target_pos=TARGET,
                        dynamics_model=Model.FOSSEN, ship_model=ship_model)
    env.enable_smoothing = False
    obs, _ = env.reset(seed=0)
    base = PySimBaseEnv._get_obs(env)
    assert base.shape == (13,)
    assert base[12] == -1.0                                     # zero thrust after reset, clipped
    assert env.observation_space.contains(obs)
    for rudder in (1.0, -1.0):
        obs, *_ = env.step(np.array([rudder, 0.5], dtype=np.float32))
        assert env.observation_space.contains(obs)
        base = PySimBaseEnv._get_obs(env)
        assert base[10] == pytest.approx(rudder, abs=1e-6)
        assert base[12] == pytest.approx(0.5, abs=1e-6)
