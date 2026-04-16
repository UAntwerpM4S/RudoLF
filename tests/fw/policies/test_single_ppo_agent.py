"""Focused regression tests for SinglePPOAgent training safeguards."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from fw.stop_condition import StopCondition
from fw.policies.single_ppo_agent import SinglePPOAgent


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


def test_learn_reraises_keyboard_interrupt_and_skips_final_checkpoint(tmp_path, monkeypatch):
    """Interrupted training should not look like a successful completed run."""
    import fw.policies.single_ppo_agent as single_ppo_agent_module

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))
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

    assert save_calls == []
    assert metrics_dumped == []
    assert not list(tmp_path.glob("*iterations*.csv"))
