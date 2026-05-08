"""Focused regression tests for SinglePPOAgent training safeguards."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from types import SimpleNamespace

from fw.stop_condition import StopCondition
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.config_profile import ProfileRegistry

_DEFAULT_PROFILE = ProfileRegistry.get("default_pysim_profile")


def test_filter_trainable_episodes_skips_one_step_rollouts():
    """One-step rollouts should be discarded before PPO optimization."""
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
    """Minimal env stub for SinglePPOAgent unit tests."""

    type_name = "PySimEnv"
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
    """Minimal AirSim env stub for AirSim-specific PPO safeguards."""

    type_name = "AirSimEnv"


def test_learn_reraises_keyboard_interrupt_and_skips_final_checkpoint(tmp_path, monkeypatch):
    """Interrupted training should not look like a successful completed run."""
    import fw.policies.single_ppo_agent as single_ppo_agent_module

    agent = SinglePPOAgent(profile=_DEFAULT_PROFILE, environment=_DummyEnv(), model_dir=str(tmp_path))
    save_calls = []
    metrics_dumped = []

    def _raise_interrupt(_num_envs):
        """Simulate the user aborting training with Ctrl+C."""
        raise KeyboardInterrupt()

    monkeypatch.setattr(agent, "run_parallel_episodes", _raise_interrupt)
    monkeypatch.setattr(agent, "save_policy", lambda policy_name: save_calls.append(policy_name))
    monkeypatch.setattr(
        single_ppo_agent_module,
        "dump_metrics_to_csv",
        lambda *args, **kwargs: metrics_dumped.append((args, kwargs)),
    )

    with pytest.raises(KeyboardInterrupt):
        agent.learn(stop_condition=StopCondition(max_time_steps=3))

    # Interrupt checkpoints (ppo_interrupt_*) are expected; final/success saves are not.
    non_interrupt_saves = [n for n in save_calls if "interrupt" not in n]
    assert non_interrupt_saves == []
    assert metrics_dumped == []
    assert not list(tmp_path.glob("*iterations*.csv"))


def test_learn_raises_after_repeated_untrainable_airsim_batches(tmp_path, monkeypatch):
    """AirSim training should fail fast instead of looping forever on empty batches."""
    import fw.policies.single_ppo_agent as single_ppo_agent_module

    agent = SinglePPOAgent(profile=_DEFAULT_PROFILE, environment=_DummyAirSimEnv(), model_dir=str(tmp_path))
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
