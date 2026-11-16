"""
Policy loading and the ship observation:

- a saved policy that does not fit the env is refused at load time, without
  changing any agent;
- visualization raises on a missing or incompatible policy, training falls back,
  and a failing evaluation after training keeps the training metrics (Trainer
  and pipeline);
- ShipSpecifications refuses actuator ranges the observation cannot normalize by;
- evaluation series start at the env's actuator state;
- compute_wind_current_obs_scales suggests a yaw-rate scale;
- observed yaw-rate scale per dynamics model (UGent per UKC), in PySimEnv and,
  in separate tables, in the COLAV envs (FH and UA keep 0.04);
- COLAV observation: rudder and thrust on the action scale (13 base features);
- a saved policy carries its yaw-rate scale, and the env it is loaded into takes
  it over (e.g. a Fossen policy validated in FH); files without one get the env's
  unstored_policy_yaw_rate_obs_scale;
- the UE5 replay checks the policy size and applies its yaw-rate scale.
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


def test_trainer_and_scheduler_require_the_policy_for_visualization(monkeypatch):
    """
    Trainer.visualize and "visualize" pipeline blocks ask set_environment to raise
    on a missing or unfitting policy; "train" blocks do not.
    """

    from types import SimpleNamespace
    import fw.scheduling.pipeline_scheduler as scheduler_module
    from fw.scheduling.trainer import Trainer

    calls = []

    def set_environment(**kwargs):
        """
        Record the call.
        """

        calls.append(kwargs)

    trainer = Trainer.__new__(Trainer)
    trainer._config = {"visualize_best_trained": False, "live_animation": False}
    trainer._agent_mgr = SimpleNamespace(set_environment=set_environment,
                                         visualize_trained_agent=lambda **_k: None)
    trainer.visualize([SimpleNamespace(gym_env=SimpleNamespace(training_name="run", type_name="PySimEnv"))])
    assert calls[-1]["require_policy"] is True

    gym_env = SimpleNamespace(training_name="block", phys_sim=None)
    monkeypatch.setattr(scheduler_module, "resolve_block_profile",
                        lambda _b: SimpleNamespace(model=SimpleNamespace(ship_model=None, dynamics_model=None)))
    monkeypatch.setattr(scheduler_module, "block_framework_config", lambda _b, _p: {})
    monkeypatch.setattr(scheduler_module, "build_block_environment",
                        lambda _b, _c: (SimpleNamespace(gym_env=gym_env), "PySimEnv"))
    monkeypatch.setattr(scheduler_module, "apply_profile", lambda *a, **k: None)
    scheduler = scheduler_module.PipelineScheduler.__new__(scheduler_module.PipelineScheduler)
    scheduler.agent_mgr = SimpleNamespace(set_environment=set_environment, visualize_trained_agent=lambda **_k: None,
                                          train=lambda **_k: None, agent_type="PPO",
                                          agent=SimpleNamespace(save_policy=lambda _n: None))
    monkeypatch.setattr(scheduler, "_resolve_transfer", lambda *a: (None, None), raising=False)
    monkeypatch.setattr(scheduler, "_resolve_hyperparameters", lambda *a: {}, raising=False)
    monkeypatch.setattr(scheduler, "_build_stop_condition", lambda _s: SimpleNamespace(reset=lambda: None),
                        raising=False)
    monkeypatch.setattr(scheduler, "_save_checkpoint", lambda *a: None, raising=False)
    for action, expected in (("visualize", True), ("train", False)):
        block = SimpleNamespace(name=action, env_type="PySimEnv", action=action, show_plots=False,
                                stop_condition=None)
        scheduler._run_block(block, set(), None)
        assert calls[-1]["require_policy"] is expected


def test_failed_evaluation_after_a_pipeline_train_block_keeps_the_block(monkeypatch):
    """
    A train block whose evaluation (show_plots) fails still succeeds, with an
    "evaluation_error" instead of an evaluation summary.
    """

    from types import SimpleNamespace
    import fw.scheduling.pipeline_scheduler as scheduler_module

    def failing_evaluation(**_kwargs):
        """
        visualize_trained_agent stand-in: the evaluation fails.
        """

        raise KeyError("summary")

    gym_env = SimpleNamespace(training_name="block", phys_sim=None)
    monkeypatch.setattr(scheduler_module, "resolve_block_profile",
                        lambda _b: SimpleNamespace(model=SimpleNamespace(ship_model=None, dynamics_model=None)))
    monkeypatch.setattr(scheduler_module, "block_framework_config", lambda _b, _p: {})
    monkeypatch.setattr(scheduler_module, "build_block_environment",
                        lambda _b, _c: (SimpleNamespace(gym_env=gym_env), "PySimEnv"))
    monkeypatch.setattr(scheduler_module, "apply_profile", lambda *a, **k: None)
    scheduler = scheduler_module.PipelineScheduler.__new__(scheduler_module.PipelineScheduler)
    scheduler.agent_mgr = SimpleNamespace(set_environment=lambda **_k: None, visualize_trained_agent=failing_evaluation,
                                          train=lambda **_k: None)
    monkeypatch.setattr(scheduler, "_resolve_transfer", lambda *a: (None, None), raising=False)
    monkeypatch.setattr(scheduler, "_resolve_hyperparameters", lambda *a: {}, raising=False)
    monkeypatch.setattr(scheduler, "_build_stop_condition", lambda _s: SimpleNamespace(reset=lambda: None),
                        raising=False)
    monkeypatch.setattr(scheduler, "_save_checkpoint", lambda *a: None, raising=False)
    block = SimpleNamespace(name="train_block", env_type="PySimEnv", action="train", show_plots=True,
                            stop_condition=None)
    result = scheduler._run_block(block, set(), None)
    assert "evaluation_summary" not in result and "summary" in result["evaluation_error"]


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


@pytest.mark.parametrize("error", [FileNotFoundError("no Best_ policy"), ValueError("size"), RuntimeError("load"),
                                   KeyError("summary")])
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
    agent.env = None                                            # no env: no yaw-rate scale to apply
    agent._vec_env = None
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


def test_fh_and_ua_yaw_scale_is_0_04():
    """
    FH and UA use 0.04; the other models have their own scales.
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
    Rudder and thrust are observed on the action scale, as in PySimEnv (13 base
    features, thrust -1 before the first step), so Myzako's 75 deg rudder stays
    inside the Box.
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


def _saved_scale(path):
    """
    The yaw-rate scale stored in a saved single policy (.zip with policy.pth), or None.
    """

    import io
    import torch
    import zipfile

    with zipfile.ZipFile(path) as archive:
        state = torch.load(io.BytesIO(archive.read("policy.pth")), weights_only=False)
    return (state.get("obs_scales") or {}).get("yaw_rate")


def test_saved_policy_carries_the_yaw_scale_it_was_trained_with(tmp_path):
    """
    A policy built for a Fossen env records 0.07 and saves it; reloading restores it.
    """

    from fw.policies.ppo_policy import PPOPolicy
    from fw.policies.single_ppo_agent import SinglePPOAgent

    agent = SinglePPOAgent(environment=_fossen_env(), model_dir=str(tmp_path))
    assert agent.policy.yaw_rate_obs_scale == 0.07
    agent.save_policy("fossen_policy")
    assert _saved_scale(tmp_path / "fossen_policy.zip") == 0.07
    assert PPOPolicy.load(str(tmp_path / "fossen_policy"), "cpu").yaw_rate_obs_scale == 0.07


def test_fossen_policy_in_fh_observes_yaw_with_the_fossen_scale(tmp_path):
    """
    Validation in FH: the FH env takes over the scale the Fossen policy was trained
    with, so the same yaw rate gives the same observation as in training.
    """

    from fw.environments.py_sim_env import PySimEnv
    from fw.policies.single_ppo_agent import SinglePPOAgent
    from tests.fw.test_fh_wind_current import _FakeEnvBridge

    fossen = PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET, dynamics_model=Model.FOSSEN,
                      ship_model=ShipModel.MYZAKO_37, wind=True, current=True)
    SinglePPOAgent(environment=fossen, model_dir=str(tmp_path)).save_policy("fossen_policy")
    fh_env = PySimEnv(dynamics_model=Model.FH_TABULAR, ship_model=ShipModel.MYZAKO_37, fh_bridge=_FakeEnvBridge(),
                      wind=True, current=True)
    assert fh_env.yaw_rate_obs_scale == fh_env.own_yaw_rate_obs_scale == 0.04
    agent = SinglePPOAgent(environment=fh_env, model_dir=str(tmp_path))
    agent.load_policy("fossen_policy")
    assert fh_env.yaw_rate_obs_scale == 0.07 and fh_env.own_yaw_rate_obs_scale == 0.04
    obs, _ = fh_env.reset(seed=0)
    np.testing.assert_allclose(obs[2], fh_env.phys_sim.state.r / 0.07, rtol=1e-6)


def test_fresh_agent_restores_the_env_own_scale(tmp_path):
    """
    After a loaded policy changed the env's scale, a new (untrained) agent on the
    same env trains with the env's own scale again.
    """

    from fw.environments.py_sim_env import PySimEnv
    from fw.policies.single_ppo_agent import SinglePPOAgent

    SinglePPOAgent(environment=_fossen_env(), model_dir=str(tmp_path)).save_policy("fossen_policy")
    nomoto = PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                      dynamics_model=Model.NOMOTO, ship_model=ShipModel.MYZAKO_37)
    SinglePPOAgent(environment=nomoto, model_dir=str(tmp_path)).load_policy("fossen_policy")
    assert nomoto.yaw_rate_obs_scale == 0.07
    fresh = SinglePPOAgent(environment=nomoto, model_dir=str(tmp_path))
    assert nomoto.yaw_rate_obs_scale == 0.15 and fresh.policy.yaw_rate_obs_scale == 0.15


def test_policy_file_without_a_scale_gets_0_04_in_pysimenv(tmp_path, caplog):
    """
    PySimEnv: a policy file without a stored scale gets 0.04 (the scale such
    files were trained with); the env takes it over (logged) and the next save
    stores it.
    """

    import logging
    from fw.environments.py_sim_env import UNSTORED_POLICY_YAW_RATE_OBS_SCALE
    from fw.policies.ppo_policy import PPOPolicy
    from fw.policies.single_ppo_agent import SinglePPOAgent

    assert UNSTORED_POLICY_YAW_RATE_OBS_SCALE == 0.04
    PPOPolicy(13, 2, "cpu").save(str(tmp_path / "unstored_policy"))
    assert _saved_scale(tmp_path / "unstored_policy.zip") is None
    env = _fossen_env()
    agent = SinglePPOAgent(environment=env, model_dir=str(tmp_path))
    with caplog.at_level(logging.INFO):
        agent.load_policy("unstored_policy")
    assert env.yaw_rate_obs_scale == 0.04 and env.own_yaw_rate_obs_scale == 0.07
    assert agent.policy.yaw_rate_obs_scale == 0.04
    assert any("no stored yaw-rate observation scale" in r.getMessage() for r in caplog.records)
    agent.save_policy("resaved_policy")
    assert _saved_scale(tmp_path / "resaved_policy.zip") == 0.04


def test_policy_file_without_a_scale_gets_the_own_scale_in_colav(tmp_path):
    """
    COLAV envs: a policy file without a stored scale gets the env's own scale
    (COLAV observations of that size used it), so nothing changes.
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv
    from fw.policies.ppo_policy import PPOPolicy
    from fw.policies.single_ppo_agent import SinglePPOAgent

    env = PySimColavEnv(time_step=1.0, max_steps=5, ship_pos=START, target_pos=TARGET,
                        dynamics_model=Model.NOMOTO, ship_model=ShipModel.MYZAKO_37)
    dim = env.observation_space.shape[0]
    PPOPolicy(dim, 2, "cpu").save(str(tmp_path / "unstored_colav_policy"))
    agent = SinglePPOAgent(environment=env, model_dir=str(tmp_path))
    agent.load_policy("unstored_colav_policy")
    assert env.yaw_rate_obs_scale == env.own_yaw_rate_obs_scale == 0.15
    assert agent.policy.yaw_rate_obs_scale == 0.15


def test_invalid_stored_scale_is_refused_before_anything_changes(tmp_path):
    """
    A corrupt stored scale (0.0) is refused when the file is read, so a multi-agent
    archive whose second policy has it leaves every agent and sub-environment as it was.
    """

    import torch
    import zipfile
    from fw.environments.multi_sim_env import MultiSimEnv
    from fw.policies.multi_ppo_agent import MultiPPOAgent
    from fw.policies.ppo_policy import PPOPolicy
    from fw.policies.single_ppo_agent import SinglePPOAgent

    with zipfile.ZipFile(tmp_path / "corrupt.zip", "w") as archive:
        for i, scale in enumerate((0.15, 0.0)):
            state = PPOPolicy(13, 2, "cpu").serialize_policy()
            state["obs_scales"] = {"yaw_rate": scale}
            torch.save(state, tmp_path / f"policy_{i}.pth")
            archive.write(tmp_path / f"policy_{i}.pth", f"policy_{i}.pth")
    env = MultiSimEnv([_fossen_env(), _fossen_env()])
    agents = [SinglePPOAgent(environment=sub, model_dir=str(tmp_path)) for sub in env.sub_environments.values()]
    before = [a.policy for a in agents]
    with pytest.raises(ValueError, match="positive and finite"):
        MultiPPOAgent(None, environment=env, agents=agents, model_dir=str(tmp_path)).load_policy("corrupt")
    assert [a.policy for a in agents] == before
    assert [sub.yaw_rate_obs_scale for sub in env.sub_environments.values()] == [0.07, 0.07]


def test_scale_related_load_failure_falls_back_for_training_and_raises_for_visualization(tmp_path, monkeypatch):
    """
    A policy file with an invalid stored scale: training falls back to a fresh
    policy at the env's own scale, visualization raises.
    """

    from fw.configuration.config import PolicyStrategy
    from fw.policies.ppo_policy import PPOPolicy
    import fw.scheduling.agent_manager_simple as module

    corrupt = PPOPolicy(13, 2, "cpu")
    corrupt.yaw_rate_obs_scale = -1.0
    corrupt.save(str(tmp_path / "bad_PPO_policy"))
    monkeypatch.setattr(module, "apply_profile", lambda *a, **k: None)
    mgr = module.AgentManagerSimple.__new__(module.AgentManagerSimple)
    mgr._agent, mgr._agent_type, mgr._profile = None, "PPO", None
    mgr._hyperparameters = {"model_dir": str(tmp_path)}
    env = _fossen_env()
    mgr.set_environment(env, PolicyStrategy.REUSE_OTHER_POLICY, "bad")
    assert env.yaw_rate_obs_scale == 0.07 and mgr._agent.policy.yaw_rate_obs_scale == 0.07
    with pytest.raises((ValueError, RuntimeError)):
        mgr.set_environment(env, PolicyStrategy.REUSE_OTHER_POLICY, "bad", require_policy=True)


def test_failing_worker_close_changes_nothing(tmp_path):
    """
    The worker copies are closed before the scale changes: if closing fails, the
    load fails (RuntimeError, caught by the agent managers) and env scale and
    policy stay as they were.
    """

    from fw.environments.py_sim_env import PySimEnv
    from fw.policies.single_ppo_agent import SinglePPOAgent

    SinglePPOAgent(environment=_fossen_env(), model_dir=str(tmp_path)).save_policy("fossen_policy")
    nomoto = PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                      dynamics_model=Model.NOMOTO, ship_model=ShipModel.MYZAKO_37)
    agent = SinglePPOAgent(environment=nomoto, model_dir=str(tmp_path))
    before = agent.policy

    class _Workers:
        """
        VecEnv stand-in whose close() fails.
        """

        def close(self):
            """
            Fail.
            """

            raise OSError("worker did not stop")

    agent._vec_env = _Workers()
    with pytest.raises(RuntimeError, match="close the worker copies"):
        agent.load_policy("fossen_policy")
    assert nomoto.yaw_rate_obs_scale == 0.15 and agent.policy is before


def test_changed_scale_rebuilds_existing_worker_copies(tmp_path):
    """
    Worker copies made before a policy with another scale was loaded are rebuilt,
    so they observe with the new scale too.
    """

    from fw.policies.single_ppo_agent import SinglePPOAgent
    from fw.environments.py_sim_env import PySimEnv

    SinglePPOAgent(environment=_fossen_env(), model_dir=str(tmp_path)).save_policy("fossen_policy")
    nomoto = PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                      dynamics_model=Model.NOMOTO, ship_model=ShipModel.MYZAKO_37)
    agent = SinglePPOAgent(environment=nomoto, model_dir=str(tmp_path))
    agent._reset_envs(1)
    old_workers = agent._vec_env
    agent.load_policy("fossen_policy")
    assert agent._vec_env is None
    agent._reset_envs(1)
    assert agent._vec_env is not old_workers
    assert agent._vec_env.get_attr("yaw_rate_obs_scale") == [0.07]


def test_multi_agent_sub_environments_take_their_policy_scale(tmp_path):
    """
    MultiPPOAgent: each sub-environment takes over the scale of its own policy.
    """

    import torch
    import zipfile
    from fw.environments.multi_sim_env import MultiSimEnv
    from fw.environments.py_sim_env import PySimEnv
    from fw.policies.multi_ppo_agent import MultiPPOAgent
    from fw.policies.ppo_policy import PPOPolicy
    from fw.policies.single_ppo_agent import SinglePPOAgent

    path = tmp_path / "pair.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for i, scale in enumerate((0.07, 0.15)):
            policy = PPOPolicy(13, 2, "cpu")
            policy.yaw_rate_obs_scale = scale
            torch.save(policy.serialize_policy(), tmp_path / f"policy_{i}.pth")
            archive.write(tmp_path / f"policy_{i}.pth", f"policy_{i}.pth")
    subs = [PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                     dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37) for _ in range(2)]
    env = MultiSimEnv(subs)
    agents = [SinglePPOAgent(environment=sub, model_dir=str(tmp_path)) for sub in env.sub_environments.values()]
    MultiPPOAgent(None, environment=env, agents=agents, model_dir=str(tmp_path)).load_policy("pair")
    assert [sub.yaw_rate_obs_scale for sub in env.sub_environments.values()] == [0.07, 0.15]


def test_multi_policy_agent_refuses_mixed_or_foreign_scales(tmp_path):
    """
    MultiPolicySingleAgent shares one env: an archive with different scales, or a
    pretrained slot policy with another scale than the env, is refused.
    """

    import torch
    import zipfile
    from fw.policies.multi_policy_single_agent import MultiPolicySingleAgent
    from fw.policies.ppo_policy import PPOPolicy

    env = _fossen_env()
    agent = MultiPolicySingleAgent.__new__(MultiPolicySingleAgent)
    agent.num_policies, agent.policy_obs_dim, agent.output_dim = 2, 13, 2
    agent.device, agent.model_dir, agent.learning_rate = torch.device("cpu"), str(tmp_path), 1e-3
    agent.env = env
    agent.policies = [PPOPolicy(13, 2, "cpu") for _ in range(2)]
    agent.optimizers = [torch.optim.Adam(p.network.parameters()) for p in agent.policies]
    agent.frozen_pids = set()
    agent._vec_env = None

    with zipfile.ZipFile(tmp_path / "mixed.zip", "w") as archive:
        for i, scale in enumerate((0.07, 0.15)):
            policy = PPOPolicy(13, 2, "cpu")
            policy.yaw_rate_obs_scale = scale
            torch.save(policy.serialize_policy(), tmp_path / f"policy_{i}.pth")
            archive.write(tmp_path / f"policy_{i}.pth", f"policy_{i}.pth")
    with pytest.raises(ValueError, match="different yaw-rate observation scales"):
        agent.load_policy("mixed")

    with zipfile.ZipFile(tmp_path / "legacy_and_new.zip", "w") as archive:
        for i, scale in enumerate((None, 0.15)):                # no stored scale counts as 0.04
            policy = PPOPolicy(13, 2, "cpu")
            policy.yaw_rate_obs_scale = scale
            torch.save(policy.serialize_policy(), tmp_path / f"policy_{i}.pth")
            archive.write(tmp_path / f"policy_{i}.pth", f"policy_{i}.pth")
    with pytest.raises(ValueError, match=r"different yaw-rate observation scales \[0.04, 0.15\]"):
        agent.load_policy("legacy_and_new")

    foreign = PPOPolicy(13, 2, "cpu")
    foreign.yaw_rate_obs_scale = 0.15
    foreign.save(str(tmp_path / "nomoto_policy"))
    with pytest.raises(ValueError, match="shared by all slots"):
        agent.load_individual_policy(0, str(tmp_path / "nomoto_policy"))
    assert env.yaw_rate_obs_scale == 0.07


def test_multi_policy_agent_takes_over_a_common_scale(tmp_path):
    """
    MultiPolicySingleAgent: an archive with one common scale is taken over by the
    shared env, the worker copies are closed for rebuilding, and the optimizers
    are bound to the new policies.
    """

    import torch
    import zipfile
    from fw.policies.multi_policy_single_agent import MultiPolicySingleAgent
    from fw.policies.ppo_policy import PPOPolicy

    env = _fossen_env()
    agent = MultiPolicySingleAgent.__new__(MultiPolicySingleAgent)
    agent.num_policies, agent.policy_obs_dim, agent.output_dim = 2, 13, 2
    agent.device, agent.model_dir, agent.learning_rate = torch.device("cpu"), str(tmp_path), 1e-3
    agent.env = env
    agent.policies = [PPOPolicy(13, 2, "cpu") for _ in range(2)]
    agent.optimizers = [torch.optim.Adam(p.network.parameters()) for p in agent.policies]
    closed = []
    agent._vec_env = type("Workers", (), {"close": lambda self: closed.append(True)})()

    with zipfile.ZipFile(tmp_path / "nomoto.zip", "w") as archive:
        for i in range(2):
            policy = PPOPolicy(13, 2, "cpu")
            policy.yaw_rate_obs_scale = 0.15
            torch.save(policy.serialize_policy(), tmp_path / f"policy_{i}.pth")
            archive.write(tmp_path / f"policy_{i}.pth", f"policy_{i}.pth")
    agent.load_policy("nomoto")
    assert env.yaw_rate_obs_scale == 0.15 and closed == [True] and agent._vec_env is None
    for policy, optimizer in zip(agent.policies, agent.optimizers):
        assert policy.yaw_rate_obs_scale == 0.15
        assert optimizer.param_groups[0]["params"][0] is next(policy.network.parameters())


def test_colav_env_yaw_scale_setter():
    """
    The COLAV env's setter validates, and None restores its own scale.
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv

    env = PySimColavEnv(time_step=1.0, max_steps=5, ship_pos=START, target_pos=TARGET,
                        dynamics_model=Model.NOMOTO, ship_model=ShipModel.MYZAKO_37)
    assert env.yaw_rate_obs_scale == env.own_yaw_rate_obs_scale == 0.15
    for bad in (0.0, float("nan")):
        with pytest.raises(ValueError):
            env.set_yaw_rate_obs_scale(bad)
    env.set_yaw_rate_obs_scale(0.07)
    obs, *_ = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert obs[2] == pytest.approx(np.clip(env.phys_sim.state.r / 0.07, -1, 1), abs=1e-6)
    env.set_yaw_rate_obs_scale(None)
    assert env.yaw_rate_obs_scale == 0.15


def test_ue5_replay_checks_the_policy_and_applies_its_scale(tmp_path, monkeypatch):
    """
    The UE5 replay refuses a policy of another size right after building its env
    (clear size error, no stepping) and otherwise applies the policy's yaw-rate scale.
    """

    from types import SimpleNamespace
    import airsim.inference.run_visualise_models as module
    from fw.policies.ppo_policy import PPOPolicy

    env = _fossen_env()
    fitting = PPOPolicy(13, 2, "cpu")
    fitting.yaw_rate_obs_scale = 0.15
    module._prepare_policy_for_env(fitting, env, "fitting")
    assert env.yaw_rate_obs_scale == 0.15
    with pytest.raises(ValueError, match="observation/action size"):
        module._prepare_policy_for_env(PPOPolicy(12, 2, "cpu"), _fossen_env(), "old")

    class _Exporter:
        """
        AirSimSessionExporter stand-in writing nothing.
        """

        def __init__(self, *args, **kwargs):
            """
            Paths under tmp_path.
            """

            self.bridge_log_path = tmp_path / "bridge.csv"
            self.export_dir = tmp_path

        def __enter__(self):
            """
            Enter.
            """

            return self

        def __exit__(self, *exc):
            """
            Do not swallow exceptions.
            """

            return False

        def update_metadata(self, *_args, **_kwargs):
            """
            Ignore.
            """

    class _Bridge:
        """
        Connected AirSimBridge stand-in; every other call is a no-op.
        """

        is_connected = True

        def __init__(self, *_args, **_kwargs):
            """
            Nothing to connect.
            """

        def __getattr__(self, _name):
            """
            Any bridge method: no-op.
            """

            return lambda *a, **k: None

    adapter = SimpleNamespace(family="pysim", model_origin="test", cleanup_environment=lambda *_a, **_k: None,
                              resolve_environment_name=lambda _p, _n: "PySimEnv",
                              load_policy=lambda _p, device="cpu": PPOPolicy(12, 2, "cpu"),
                              build_environment=lambda _n, max_steps: _fossen_env())
    monkeypatch.setattr(module, "AirSimSessionExporter", _Exporter)
    monkeypatch.setattr(module, "AirSimBridge", _Bridge)
    monkeypatch.setattr(module, "resolve_model_path", lambda p, _d: tmp_path / "old.zip")
    with pytest.raises(ValueError, match="observation/action size"):
        module.VisualiseModelsRunner(adapter).run("old.zip")


def test_env_scale_setter_validates():
    """
    set_yaw_rate_obs_scale refuses non-positive or non-finite scales and None restores the own scale.
    """

    env = _fossen_env()
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            env.set_yaw_rate_obs_scale(bad)
    env.set_yaw_rate_obs_scale(0.2)
    assert env.yaw_rate_obs_scale == 0.2
    env.set_yaw_rate_obs_scale(None)
    assert env.yaw_rate_obs_scale == env.own_yaw_rate_obs_scale == 0.07
