"""
Focused regression tests for SinglePPOAgent training safeguards.
"""

from __future__ import annotations

import pytest
import numpy as np
import gymnasium as gym

from types import SimpleNamespace
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.configuration.stop_condition import StopCondition


def test_filter_trainable_episodes_skips_one_step_rollouts():
    """
    One-step rollouts should be discarded before PPO optimization.
    """

    short_episode = {
        "rewards": np.array([-0.1], dtype=np.float32),
        "advantages": np.array([-0.1], dtype=np.float32),
    }
    valid_episode = {
        "rewards": np.array([-0.1, 0.2], dtype=np.float32),
        "advantages": np.array([0.05, -0.02], dtype=np.float32),
    }

    valid_episodes, skipped_count = SinglePPOAgent.filter_trainable_episodes(
        [short_episode, valid_episode]
    )

    assert skipped_count == 1
    assert valid_episodes == [valid_episode]


class _DummyEnv:
    """
    Minimal env stub for SinglePPOAgent unit tests.
    """

    type_name = "PySimEnv"
    training_name = type_name

    observation_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2,),
        dtype=np.float32,
    )


class _DummyAirSimEnv(_DummyEnv):
    """
    Minimal AirSim env stub for AirSim-specific PPO safeguards.
    """

    type_name = "AirSimEnv"


def test_learn_reraises_keyboard_interrupt_and_skips_final_checkpoint(tmp_path, monkeypatch):
    """
    Interrupted training should not look like a successful completed run.
    """

    import fw.policies.single_ppo_agent as single_ppo_agent_module
    from fw.configuration.profile_applier import SimulatorApplier, ActionAdapter

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))
    save_calls = []
    metrics_dumped = []

    def _raise_interrupt(_num_envs):
        """
        Simulate the user aborting training with Ctrl+C.
        """

        raise KeyboardInterrupt()

    # Stub profile-dependent methods.
    monkeypatch.setattr(SimulatorApplier, "apply", lambda *args, **kwargs: None)
    monkeypatch.setattr(ActionAdapter, "transform", lambda *args, **kwargs: args[1] if len(args) > 1 else None)
    
    monkeypatch.setattr(agent, "run_parallel_episodes", _raise_interrupt)
    monkeypatch.setattr(agent, "save_policy", lambda policy_name: save_calls.append(policy_name))
    monkeypatch.setattr(
        single_ppo_agent_module,
        "dump_metrics_to_csv",
        lambda *args, **kwargs: metrics_dumped.append((args, kwargs)),
    )

    with pytest.raises(KeyboardInterrupt):
        agent.learn(stop_condition=StopCondition(max_time_steps=3))

    assert save_calls == ["ppo_interrupt_0"]
    assert metrics_dumped == []
    assert not list(tmp_path.glob("*iterations*.csv"))


def test_learn_raises_after_repeated_untrainable_airsim_batches(tmp_path, monkeypatch):
    """
    AirSim training should fail fast instead of looping forever on empty batches.
    """

    import fw.policies.single_ppo_agent as single_ppo_agent_module
    from fw.configuration.profile_applier import SimulatorApplier, ActionAdapter

    agent = SinglePPOAgent(environment=_DummyAirSimEnv(), model_dir=str(tmp_path))
    fake_logger = SimpleNamespace(
        path=tmp_path / "iterations.csv",
        log_iteration=lambda **kwargs: None,
        save=lambda: None,
    )
    fake_session = SimpleNamespace(
        iteration_logger=fake_logger,
        run_log_path=tmp_path / "run_log.csv",
        metadata_path=tmp_path / "metadata.json",
        route_plot_path=tmp_path / "route_plot.png",
        training_metrics_plot_path=tmp_path / "training_metrics.png",
        log_run_events=lambda *args, **kwargs: None,
        set_route_snapshot=lambda *args, **kwargs: None,
        finalize=lambda *args, **kwargs: None,
    )
    short_episode = [
        {
            "rewards": np.array([0.0], dtype=np.float32),
            "advantages": np.array([0.0], dtype=np.float32),
        }
    ]

    # Stub profile-dependent methods.
    monkeypatch.setattr(SimulatorApplier, "apply", lambda *args, **kwargs: None)
    monkeypatch.setattr(ActionAdapter, "transform", lambda *args, **kwargs: args[1] if len(args) > 1 else None)
    
    monkeypatch.setattr(
        single_ppo_agent_module,
        "AirSimTrainingSession",
        lambda *args, **kwargs: fake_session,
    )
    monkeypatch.setattr(
        SinglePPOAgent,
        "MAX_CONSECUTIVE_UNTRAINABLE_AIRSIM_BATCHES",
        3,
    )
    monkeypatch.setattr(agent, "run_parallel_episodes", lambda _num_envs: short_episode)
    monkeypatch.setattr(agent, "_drain_env_run_events", lambda: [])
    monkeypatch.setattr(agent, "_get_training_export_snapshot", lambda: None)

    with pytest.raises(RuntimeError, match="untrainable rollout batches"):
        agent.learn(stop_condition=StopCondition(max_time_steps=10))


def test_single_ppo_agent_can_be_instantiated_without_profile(tmp_path):
    """
    SinglePPOAgent initializes cleanly with profile=None (profile-dependent
    steps such as scenario initialization and action transformation are skipped).
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))

    assert agent._profile is None
    assert agent.observation_space is not None
    assert agent.action_space is not None
    assert agent.policy is not None


def test_load_policy_rebinds_optimizer(tmp_path):
    """
    Optimizer must point at the loaded policy's parameters: PPOPolicy.load()
    returns a new object, so otherwise step() would update the old network.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))

    # Save the current policy so we have something to load back.
    agent.save_policy("test_policy")

    original_policy = agent.policy
    original_param_ids = {id(p) for p in original_policy.network.parameters()}

    agent.load_policy("test_policy")

    # After load_policy, self.policy must be a different object.
    assert agent.policy is not original_policy, "load_policy must replace the policy object"

    # The optimizer must reference the NEW policy's parameters, not the old ones.
    optimizer_param_ids = {id(p) for group in agent.optimizer.param_groups for p in group["params"]}
    new_param_ids = {id(p) for p in agent.policy.network.parameters()}

    assert optimizer_param_ids == new_param_ids, (
        "Optimizer parameters must match the loaded policy's network parameters"
    )
    assert optimizer_param_ids.isdisjoint(original_param_ids), (
        "Optimizer must not still reference the old (pre-load) network parameters"
    )
