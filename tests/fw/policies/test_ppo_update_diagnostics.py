"""
Tests for the PPO update diagnostics, the log_std clamp and the reward scaling.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import gymnasium as gym
import pytest
import torch

from fw.policies.ppo_policy import PPO_UPDATE_STAT_KEYS, PPOPolicy
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.policies.multi_policy_single_agent import MultiPolicySingleAgent
from fw.policies.actor_critic_network import ActorCriticNetwork


class _DummyEnv:
    """
    Minimal env stub: 4 observations, 2 continuous actions.
    """

    type_name = "PySimEnv"
    training_name = type_name
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


class _DummyOneHotEnv(_DummyEnv):
    """
    Env stub for MultiPolicySingleAgent: 4 base observations + 2 one-hot dims.
    """

    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)


def _episode(policy: PPOPolicy, length: int = 128, seed: int = 0) -> dict:
    """
    One synthetic trajectory collected with `policy`.
    """

    rng = np.random.default_rng(seed)
    states = rng.standard_normal((length, policy.input_dim)).astype(np.float32)
    _, raw_actions, log_probs, values = policy.predict_with_raw(states)
    advantages = rng.standard_normal(length).astype(np.float32)
    return {
        "states": states,
        "raw_actions": raw_actions,
        "log_probs": log_probs.astype(np.float32),
        "advantages": advantages,
        "returns": (advantages + values).astype(np.float32),
        "rewards": rng.standard_normal(length).astype(np.float32),
    }


def test_entropy_coef_default_is_zero(tmp_path):
    """
    The PPO agents default to no entropy bonus.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))
    assert agent.entropy_coef == 0.0


def test_single_agent_train_fills_the_update_stats(tmp_path):
    """
    train() keeps returning the mean loss and stores finite diagnostics.
    """

    torch.manual_seed(0)
    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path), num_epochs=2, batch_size=32)

    loss = agent.train([_episode(agent.policy)])

    assert isinstance(loss, float) and math.isfinite(loss)
    assert set(agent.last_update_stats) == set(PPO_UPDATE_STAT_KEYS)
    assert all(math.isfinite(v) for v in agent.last_update_stats.values())
    assert 0.0 <= agent.last_update_stats["clip_frac"] <= 1.0
    assert agent.last_update_stats["action_std"] > 0.0


def test_single_agent_train_without_data_reports_nan_but_the_current_std(tmp_path):
    """
    No update: the loss stats are NaN, action_std is the current std.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))

    assert math.isnan(agent.train([]))
    assert math.isnan(agent.last_update_stats["approx_kl"])
    assert agent.last_update_stats["action_std"] == pytest.approx(math.exp(-0.5), rel=1e-6)


def test_log_std_parameter_stays_within_bounds_and_keeps_its_gradient():
    """
    clamp_log_std_() keeps the parameter inside the bounds, where the clamp in
    action_std() still passes a gradient, so the std can move back.
    """

    network = ActorCriticNetwork(4, 2)
    with torch.no_grad():
        network.actor_log_std.fill_(5.0)        # e.g. pushed out by Adam momentum

    network.clamp_log_std_()
    assert torch.all(network.actor_log_std == ActorCriticNetwork.LOG_STD_MAX)

    network.action_std().sum().backward()
    assert torch.all(network.actor_log_std.grad != 0.0)


def test_single_agent_train_clamps_the_log_std_parameter(tmp_path):
    """
    After an update the log_std parameter lies within the bounds, even if it
    started outside them.
    """

    torch.manual_seed(0)
    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path), num_epochs=1, batch_size=64)
    episode = _episode(agent.policy)
    with torch.no_grad():
        agent.policy.network.actor_log_std.fill_(3.0)

    agent.train([episode])

    log_std = agent.policy.network.actor_log_std.detach()
    assert torch.all(log_std <= ActorCriticNetwork.LOG_STD_MAX)
    assert torch.all(log_std >= ActorCriticNetwork.LOG_STD_MIN)


class _RewardSequenceEnv(gym.Env):
    """
    Env that emits a fixed reward sequence per episode (for SB3 VecNormalize).
    """

    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, episodes):
        """
        Args:
            episodes: One reward sequence per episode, used in order.
        """

        self._episodes = list(episodes)
        self._rewards = []
        self._t = 0

    def reset(self, seed=None, options=None):
        """
        Start the next reward sequence.
        """

        self._rewards = self._episodes.pop(0) if self._episodes else [0.0]
        self._t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        """
        Emit the next reward; the episode ends with its sequence.
        """

        reward = float(self._rewards[self._t])
        self._t += 1
        return np.zeros(1, dtype=np.float32), reward, self._t >= len(self._rewards), False, {}


def test_normalize_rewards_matches_sb3_vec_normalize(tmp_path):
    """
    Parallel episodes are scaled exactly as SB3's VecNormalize scales the
    rewards of envs stepped together, carried over consecutive iterations.
    """

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    gamma, num_envs, length = 0.95, 3, 40
    rng = np.random.default_rng(1)
    iterations = [[(rng.standard_normal(length) * scale - 1.0).astype(np.float32)
                   for scale in (0.1, 1.0, 10.0)] for _ in range(3)]

    vec_env = VecNormalize(
        DummyVecEnv([lambda i=i: _RewardSequenceEnv([it[i] for it in iterations]) for i in range(num_envs)]),
        norm_obs=False, norm_reward=True, gamma=gamma,
    )
    vec_env.reset()
    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path), gamma=gamma, normalize=True)

    for episodes in iterations:
        expected = np.stack([vec_env.step(np.zeros((num_envs, 1), dtype=np.float32))[1] for _ in range(length)], axis=1)
        scaled = agent.normalize_episode_rewards(episodes)
        assert np.allclose(np.stack(scaled), expected, rtol=1e-5, atol=1e-6)


def test_normalize_episode_rewards_handles_unequal_lengths(tmp_path):
    """
    Envs that finished earlier drop out of the per-step statistics update.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path), normalize=True)
    scaled = agent.normalize_episode_rewards([np.full(5, -1.0, dtype=np.float32), np.full(12, 2.0, dtype=np.float32)])

    assert [len(x) for x in scaled] == [5, 12]
    assert np.all(scaled[0] < 0.0) and np.all(scaled[1] > 0.0)
    assert agent.return_count == pytest.approx(1e-4 + 17)


def test_normalize_rewards_does_not_subtract_the_mean(tmp_path):
    """
    Scaling only: constant negative per-step rewards stay negative.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path), normalize=True)

    for _ in range(5):
        scaled = agent.normalize_rewards(np.full(100, -0.5, dtype=np.float32))
        assert np.all(scaled < 0.0)


def test_multi_policy_train_fills_the_update_stats(tmp_path):
    """
    MultiPolicySingleAgent records the diagnostics per slot and clamps log_std.
    """

    torch.manual_seed(0)
    agent = MultiPolicySingleAgent(
        environment=_DummyOneHotEnv(), model_dir=str(tmp_path), num_policies=2,
        num_epochs=2, batch_size=32, watchdog_enabled=False,
    )
    with torch.no_grad():
        agent.policies[1].network.actor_log_std.fill_(3.0)

    loss, *_ = agent._train_policy(1, [_episode(agent.policies[1])])
    stats = agent._update_stats(1)

    assert math.isfinite(loss)
    assert set(stats) == set(PPO_UPDATE_STAT_KEYS)
    assert all(math.isfinite(v) for v in stats.values())
    assert torch.all(agent.policies[1].network.actor_log_std <= ActorCriticNetwork.LOG_STD_MAX)
    # Slot 0 never trained: NaN stats, current std.
    assert math.isnan(agent._update_stats(0)["approx_kl"])
    assert agent._update_stats(0)["action_std"] == pytest.approx(math.exp(-0.5), rel=1e-6)


def test_iteration_logger_writes_the_update_stats(tmp_path):
    """
    The diagnostics columns are written when given and NaN otherwise.
    """

    from fw.common.tools.training_iteration_logger import TrainingIterationLogger

    logger = TrainingIterationLogger(env_name="PySimEnv", agent_type="PPO", device="cpu", src_dir=tmp_path)
    common = dict(mean_reward=0.1, mean_advantage=0.0, loss=0.5, num_episode_batches=1, elapsed_seconds=1.0)
    logger.log_iteration(iteration=0, **common, **{key: 0.25 for key in PPO_UPDATE_STAT_KEYS})
    logger.log_iteration(iteration=1, **common)
    logger.save()

    with logger.path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for key in PPO_UPDATE_STAT_KEYS:
        assert float(rows[0][key]) == 0.25
        assert math.isnan(float(rows[1][key]))


def test_non_finite_reward_does_not_poison_the_return_statistics(tmp_path):
    """
    A NaN/Inf reward passes through (the update then skips the batch) but is
    left out of the return, so the other steps and later episodes stay scaled.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path), normalize=True)
    agent.normalize_rewards(np.full(20, -5.0, dtype=np.float32))

    scaled = agent.normalize_rewards(np.array([-5.0, np.nan, -5.0, np.inf, -5.0], dtype=np.float32))
    assert np.isnan(scaled[1]) and np.isinf(scaled[3])
    assert np.all(np.abs(scaled[[0, 2, 4]]) < 1.0)         # scaled, not raw -5
    assert np.isfinite([agent.return_mean, agent.return_var, agent.return_count]).all()
    assert np.isfinite(agent.normalize_rewards(np.full(20, -5.0, dtype=np.float32))).all()


@pytest.mark.parametrize("log_std", [ActorCriticNetwork.LOG_STD_MIN, -8.0, ActorCriticNetwork.LOG_STD_MAX])
@pytest.mark.parametrize("returns_scale", [1.0, 1e4, 1e18])
def test_extreme_updates_never_produce_nan_weights(tmp_path, log_std, returns_scale):
    """
    Std at (or near) a bound, huge value targets and off-policy samples: updates
    may be skipped, but the weights and the next rollout stay finite.
    """

    torch.manual_seed(0)
    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path), num_epochs=4, batch_size=64)
    with torch.no_grad():
        agent.policy.network.actor_log_std.fill_(log_std)

    finite_losses = 0
    for i in range(8):
        episode = _episode(agent.policy, length=128, seed=i)
        episode["returns"] = (episode["returns"] * returns_scale).astype(np.float32)
        episode["advantages"] = (episode["advantages"] * 1e6).astype(np.float32)
        # Stale behaviour log-probs: large positive and negative log-ratios.
        episode["log_probs"] = (episode["log_probs"] + np.linspace(-50, 50, 128)).astype(np.float32)
        finite_losses += math.isfinite(agent.train([episode]))

        for name, param in agent.policy.network.named_parameters():
            assert torch.isfinite(param).all(), f"{name} not finite after update {i}"

    assert finite_losses > 0                                  # updates actually ran
    actions, raw_actions, log_probs, values = agent.policy.predict_with_raw(
        np.random.default_rng(0).standard_normal((32, 4)).astype(np.float32))
    assert np.isfinite(actions).all() and np.isfinite(raw_actions).all() and np.isfinite(values).all()


@pytest.mark.parametrize("log_std", [-19.5, -0.5, 1.99])
def test_action_std_is_unchanged_inside_the_bounds(log_std):
    """
    Inside the bounds the std and its gradient equal the old plain clamp.
    """

    network = ActorCriticNetwork(4, 2)
    with torch.no_grad():
        network.actor_log_std.fill_(log_std)

    new_std = network.action_std()
    (new_grad,) = torch.autograd.grad(new_std.sum(), network.actor_log_std)
    old_std = torch.exp(torch.clamp(network.actor_log_std, -20, 2))
    (old_grad,) = torch.autograd.grad(old_std.sum(), network.actor_log_std)

    assert torch.equal(new_std, old_std)
    assert torch.equal(new_grad, old_grad)


def test_entropy_coef_in_the_config():
    """
    PPO and MultiPPO default to no entropy bonus; the tuned multi-policy and
    SB3 PPO2 values are kept.
    """

    from fw.configuration.config import (DEFAULT_HYPERPARAMETERS, MULTI_POLICY_SINGLE_NAME,
                                         MULTI_PPO_POLICY_NAME, PPO2_POLICY_NAME, PPO_POLICY_NAME)

    assert DEFAULT_HYPERPARAMETERS[PPO_POLICY_NAME]["entropy_coef"] == 0.0
    assert DEFAULT_HYPERPARAMETERS[MULTI_PPO_POLICY_NAME]["entropy_coef"] == 0.0
    assert DEFAULT_HYPERPARAMETERS[MULTI_POLICY_SINGLE_NAME]["entropy_coef"] == 0.025
    assert DEFAULT_HYPERPARAMETERS[PPO2_POLICY_NAME]["entropy_coef"] == 0.01


@pytest.mark.parametrize("log_std", [float("inf"), -float("inf"), 1e8, -1e8, 3.1, -25.0])
def test_action_std_equals_the_plain_clamp_outside_the_bounds(log_std):
    """
    Also for extreme or infinite log_std (e.g. a corrupt policy file) the std
    equals the old clamp, never NaN, and the gradient still flows.
    """

    network = ActorCriticNetwork(4, 2)
    with torch.no_grad():
        network.actor_log_std.fill_(log_std)

    std = network.action_std()
    expected = torch.exp(torch.clamp(network.actor_log_std.detach(), -20, 2))

    assert torch.equal(std.detach(), expected)
    (grad,) = torch.autograd.grad(std.sum(), network.actor_log_std)
    assert torch.all(torch.isfinite(grad)) and torch.all(grad != 0.0)
