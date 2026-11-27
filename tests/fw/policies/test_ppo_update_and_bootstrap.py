"""
PPO update safeguards and GAE bootstrap values of SinglePPOAgent,
MultiPolicySingleAgent (MPSA) and MultiPPOAgent.

The safeguards: a non-finite loss or gradient norm must skip the optimizer step,
so the weights and the Adam state stay exactly as they were. The bootstraps: the
value that closes a truncated trajectory (or, in MPSA, a segment at a scenario
switch) must come from the right critic, evaluated at the right state.
"""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import fw.policies.multi_policy_single_agent as mpsa_module
import fw.policies.single_ppo_agent as single_module
from fw.policies.base_ppo_agent import BasePPOAgent
from fw.policies.multi_policy_single_agent import MultiPolicySingleAgent
from fw.policies.multi_ppo_agent import MultiPPOAgent
from fw.policies.ppo_policy import PPOPolicy
from fw.policies.single_ppo_agent import SinglePPOAgent

OBS_DIM, ACT_DIM, NUM_POLICIES = 6, 2, 2


def _env(obs_dim: int) -> SimpleNamespace:
    """
    Minimal env: observation and action spaces only (no yaw-rate scale).
    """

    return SimpleNamespace(
        observation_space=gym.spaces.Box(-1.0, 1.0, (obs_dim,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32),
    )


def _single_agent(tmp_path) -> SinglePPOAgent:
    """
    SinglePPOAgent built through its constructor, on CPU.
    """

    torch.manual_seed(0)
    return SinglePPOAgent(None, environment=_env(OBS_DIM), model_dir=str(tmp_path), seed=0,
                          num_epochs=2, batch_size=16)


def _mpsa_agent(tmp_path) -> MultiPolicySingleAgent:
    """
    MultiPolicySingleAgent with 2 policies (observation = base features + one-hot).
    """

    torch.manual_seed(0)
    return MultiPolicySingleAgent(environment=_env(OBS_DIM + NUM_POLICIES), num_policies=NUM_POLICIES,
                                  model_dir=str(tmp_path), seed=0, num_epochs=2, batch_size=16,
                                  watchdog_enabled=False)


def _episode(obs_dim: int, n: int = 64, **overrides) -> dict:
    """
    Random trajectory in the format the PPO updates consume.
    """

    rng = np.random.default_rng(0)
    episode = {
        "states": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "raw_actions": rng.standard_normal((n, ACT_DIM)).astype(np.float32),
        "advantages": rng.standard_normal(n).astype(np.float32),
        "log_probs": rng.standard_normal(n).astype(np.float32),
        "returns": rng.standard_normal(n).astype(np.float32),
    }
    episode.update(overrides)
    return episode


def _snapshot(network: torch.nn.Module) -> list:
    """
    Copies of all parameters.
    """

    return [p.detach().clone() for p in network.parameters()]


def _unchanged(network: torch.nn.Module, snapshot: list) -> bool:
    """
    Whether every parameter still equals the snapshot exactly.
    """

    return all(torch.equal(p, s) for p, s in zip(network.parameters(), snapshot))


def _value(policy: PPOPolicy, state: np.ndarray) -> float:
    """
    Value of one state from the policy's training network.
    """

    with torch.no_grad():
        _, _, v = policy.network(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
    return float(v.squeeze())


# ---------------------------------------------------------------------------
# Update safeguards
# ---------------------------------------------------------------------------
def test_single_nan_loss_skips_every_step(tmp_path):
    """
    NaN old log-probabilities make every loss NaN: no step, weights and Adam untouched.
    """

    agent = _single_agent(tmp_path)
    before = _snapshot(agent.policy.network)
    episode = _episode(OBS_DIM, log_probs=np.full(64, np.nan, dtype=np.float32))
    assert np.isnan(agent.train([episode]))
    assert _unchanged(agent.policy.network, before)
    assert len(agent.optimizer.state) == 0


def test_single_non_finite_grad_norm_skips_the_step(tmp_path, monkeypatch):
    """
    A finite loss with a non-finite gradient norm must not step either.
    """

    agent = _single_agent(tmp_path)
    before = _snapshot(agent.policy.network)
    monkeypatch.setattr(single_module.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.tensor(float("inf")))
    assert np.isnan(agent.train([_episode(OBS_DIM)]))
    assert _unchanged(agent.policy.network, before)
    assert len(agent.optimizer.state) == 0


def test_single_clean_update_steps_and_returns_finite_loss(tmp_path):
    """
    Control for the two tests above: clean data does change the weights.
    """

    agent = _single_agent(tmp_path)
    before = _snapshot(agent.policy.network)
    assert np.isfinite(agent.train([_episode(OBS_DIM)]))
    assert not _unchanged(agent.policy.network, before)


def test_mpsa_nan_loss_skips_every_step(tmp_path):
    """
    As for SinglePPOAgent, in MPSA._train_policy (per slot); the other slot is untouched.
    """

    agent = _mpsa_agent(tmp_path)
    before = [_snapshot(p.network) for p in agent.policies]
    episode = _episode(OBS_DIM, log_probs=np.full(64, np.nan, dtype=np.float32))
    assert all(np.isnan(x) for x in agent._train_policy(0, [episode]))
    assert all(_unchanged(p.network, b) for p, b in zip(agent.policies, before))
    assert len(agent.optimizers[0].state) == 0


def test_mpsa_non_finite_grad_norm_skips_the_step(tmp_path, monkeypatch):
    """
    MPSA._train_policy with a finite loss but a non-finite gradient norm: no step.
    """

    agent = _mpsa_agent(tmp_path)
    before = _snapshot(agent.policies[1].network)
    monkeypatch.setattr(mpsa_module.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.tensor(float("nan")))
    assert all(np.isnan(x) for x in agent._train_policy(1, [_episode(OBS_DIM)]))
    assert _unchanged(agent.policies[1].network, before)
    assert len(agent.optimizers[1].state) == 0


def test_mpsa_clean_update_returns_its_loss_terms(tmp_path):
    """
    MPSA._train_policy on clean data: only the trained slot changes, and the
    returned (total, policy, value, entropy bonus) are finite with total =
    policy + value_coef * value - bonus on average.
    """

    agent = _mpsa_agent(tmp_path)
    before = [_snapshot(p.network) for p in agent.policies]
    total, policy_loss, value_loss, entropy_bonus = agent._train_policy(0, [_episode(OBS_DIM)])
    assert all(np.isfinite(x) for x in (total, policy_loss, value_loss, entropy_bonus))
    assert total == pytest.approx(policy_loss + agent.value_loss_coef * value_loss - entropy_bonus, rel=1e-5)
    assert not _unchanged(agent.policies[0].network, before[0])
    assert _unchanged(agent.policies[1].network, before[1])


def test_mpsa_frozen_slot_is_not_trained(tmp_path):
    """
    A frozen slot returns NaNs and keeps its weights.
    """

    agent = _mpsa_agent(tmp_path)
    agent.frozen_pids.add(0)
    before = _snapshot(agent.policies[0].network)
    assert all(np.isnan(x) for x in agent._train_policy(0, [_episode(OBS_DIM)]))
    assert _unchanged(agent.policies[0].network, before)


# ---------------------------------------------------------------------------
# Bootstrap values
# ---------------------------------------------------------------------------
def test_single_bootstrap_is_the_critic_value_of_the_state(tmp_path):
    """
    SinglePPOAgent._get_bootstrap_value returns the critic's value of that state.
    """

    agent = _single_agent(tmp_path)
    state = np.linspace(-1, 1, OBS_DIM).astype(np.float32)
    expected = _value(agent.policy, state)
    assert expected != 0.0
    assert agent._get_bootstrap_value(state) == pytest.approx(expected, abs=1e-6)


def test_mpsa_bootstrap_uses_the_scenario_policy_on_the_base_features(tmp_path):
    """
    MPSA._get_bootstrap_value strips the one-hot and asks the policy it selects.
    """

    agent = _mpsa_agent(tmp_path)
    base = np.linspace(-1, 1, OBS_DIM).astype(np.float32)
    values = [_value(p, base) for p in agent.policies]
    assert values[0] != pytest.approx(values[1])
    for pid in range(NUM_POLICIES):
        one_hot = np.eye(NUM_POLICIES, dtype=np.float32)[pid]
        assert agent._get_bootstrap_value(np.concatenate([base, one_hot])) == pytest.approx(values[pid], abs=1e-6)


def _mpsa_episode(terminated_at_end: bool) -> dict:
    """
    6 steps: scenario 0 for t=0..2, scenario 1 for t=3..5.
    """

    rng = np.random.default_rng(3)
    T = 6
    one_hot = np.eye(NUM_POLICIES, dtype=np.float32)[[0, 0, 0, 1, 1, 1]]
    states = np.concatenate([rng.standard_normal((T, OBS_DIM)).astype(np.float32), one_hot], axis=1)
    next_states = np.concatenate([rng.standard_normal((T, OBS_DIM)).astype(np.float32), one_hot], axis=1)
    terminations = np.zeros(T, dtype=bool)
    terminations[-1] = terminated_at_end
    return {
        "states": states, "next_states": next_states,
        "actions": rng.uniform(-1, 1, (T, ACT_DIM)).astype(np.float32),
        "raw_actions": rng.standard_normal((T, ACT_DIM)).astype(np.float32),
        "rewards": rng.standard_normal(T).astype(np.float32),
        "values": rng.standard_normal(T).astype(np.float32),
        "log_probs": rng.standard_normal(T).astype(np.float32),
        "terminations": terminations,
    }


@pytest.mark.parametrize("terminated_at_end", [False, True])
def test_mpsa_segment_bootstraps(tmp_path, terminated_at_end):
    """
    At the scenario switch the outgoing policy's value of the next segment's first
    state closes segment 0; the last segment is closed by its policy's value of the
    real next state if truncated, by 0 if terminated.
    """

    agent = _mpsa_agent(tmp_path)
    episode = _mpsa_episode(terminated_at_end)
    segments = agent._split_trajectory_and_compute_gae(episode)
    assert len(segments[0]) == len(segments[1]) == 1

    switch_value = _value(agent.policies[0], episode["states"][3, :OBS_DIM])
    end_value = 0.0 if terminated_at_end else _value(agent.policies[1], episode["next_states"][5, :OBS_DIM])
    for pid, (start, end, next_value) in ((0, (0, 3, switch_value)), (1, (3, 6, end_value))):
        expected = BasePPOAgent.compute_gae(
            episode["rewards"][start:end], episode["values"][start:end], episode["terminations"][start:end],
            next_value, agent.gamma, agent.gae_lambda)
        np.testing.assert_allclose(segments[pid][0]["advantages"], expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(segments[pid][0]["states"], episode["states"][start:end, :OBS_DIM])


def test_mppo_bootstrap_uses_the_env_s_own_agent(tmp_path):
    """
    MultiPPOAgent._get_bootstrap_value asks the critic of the agent with that env_id,
    and refuses a missing env_id.
    """

    agent = MultiPPOAgent.__new__(MultiPPOAgent)
    torch.manual_seed(0)
    agent.agents = [SimpleNamespace(policy=PPOPolicy(OBS_DIM, ACT_DIM, "cpu")) for _ in range(2)]
    state = np.linspace(-1, 1, OBS_DIM).astype(np.float32)
    for env_id in range(2):
        assert agent._get_bootstrap_value(state, env_id) == pytest.approx(
            _value(agent.agents[env_id].policy, state), abs=1e-6)
    with pytest.raises(TypeError):
        agent._get_bootstrap_value(state)
