"""
Device handling of the PPO stack: rollout network on `rollout_device`, update on
`device`, the weight-sync of the rollout copy, worker factories that do not carry
the policy, and the UAntwerp surrogate pinned to CPU.

Without CUDA the split is exercised by forcing a separate rollout copy on the CPU
(same code path, same device); the tests marked `cuda` run only where CUDA exists.
"""

from __future__ import annotations

import copy
import torch
import pytest
import cloudpickle
import numpy as np
import gymnasium as gym
import fw.policies.base_agent as base_agent_module

from types import SimpleNamespace
from fw.policies.ppo_policy import PPOPolicy
from fw.policies.multi_ppo_agent import MultiPPOAgent
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.policies.multi_policy_single_agent import MultiPolicySingleAgent

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _forced_split(policy: PPOPolicy) -> PPOPolicy:
    """
    Give a CPU policy a separate rollout copy, as PPOPolicy does when the devices differ.
    """

    policy._rollout_network = copy.deepcopy(policy.network).requires_grad_(False).eval()
    policy._rollout_weights_key = None
    return policy


def _train_values(policy: PPOPolicy, states: np.ndarray) -> np.ndarray:
    """
    Values of `states` from the training network itself.
    """

    with torch.no_grad():
        _, _, v = policy.network(torch.as_tensor(states, dtype=torch.float32, device=policy.device))
    return v.squeeze(-1).cpu().numpy()


def _fake_agent(policy: PPOPolicy) -> SinglePPOAgent:
    """
    SinglePPOAgent with just what train() and make_env() need (no env, no workers).
    """

    agent = SinglePPOAgent.__new__(SinglePPOAgent)
    agent.policy, agent.device = policy, policy.device
    agent.optimizer = torch.optim.Adam(policy.network.parameters(), lr=1e-2)
    agent.num_epochs, agent.batch_size = 2, 32
    agent.clip_range, agent.value_loss_coef, agent.entropy_coef, agent.max_grad_norm = 0.2, 0.5, 0.01, 0.5
    agent.env, agent._profile = {"dummy_env": True}, None
    agent._worker_seed_seq = np.random.SeedSequence(0)
    return agent


def _episode(obs_dim: int, n: int = 100) -> dict:
    """
    Random trajectory in the format run_parallel_episodes() produces.
    """

    rng = np.random.default_rng(0)
    return {
        "states": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "raw_actions": rng.standard_normal((n, 2)).astype(np.float32),
        "advantages": rng.standard_normal(n).astype(np.float32),
        "log_probs": rng.standard_normal(n).astype(np.float32),
        "returns": rng.standard_normal(n).astype(np.float32),
    }


def test_same_devices_use_the_training_network_itself():
    """
    rollout_device equal to device (or None) makes no copy.
    """

    for rollout_device in (None, "cpu"):
        policy = PPOPolicy(4, 2, "cpu", rollout_device=rollout_device)
        assert policy._rollout_network is None
        assert policy.rollout_network() is policy.network


def test_torch_bumps_parameter_versions_on_in_place_updates():
    """
    The sync relies on Tensor._version: guard against a torch release changing it.
    """

    policy = PPOPolicy(4, 2, "cpu")
    before = policy._weights_key()
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=1e-2)
    policy.network(torch.ones(1, 4))[2].sum().backward()
    optimizer.step()
    after_step = policy._weights_key()
    assert after_step != before
    policy.network.load_state_dict(PPOPolicy(4, 2, "cpu").network.state_dict())
    assert policy._weights_key() != after_step


def test_rollout_copy_follows_every_kind_of_weight_change():
    """
    After an update, load_state_dict or swapping the network, the public calls see
    the new weights, never stale ones.
    """

    policy = _forced_split(PPOPolicy(4, 2, "cpu"))
    states = np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32)
    np.testing.assert_allclose(policy.predict_value(states), _train_values(policy, states))

    _fake_agent(policy).train([_episode(4)])
    np.testing.assert_allclose(policy.predict_value(states), _train_values(policy, states))

    policy.network.load_state_dict(PPOPolicy(4, 2, "cpu").network.state_dict())
    np.testing.assert_allclose(policy.predict_value(states), _train_values(policy, states))

    policy.network = PPOPolicy(4, 2, "cpu").network
    np.testing.assert_allclose(policy.predict_value(states), _train_values(policy, states))

    # Deterministic actions match the training network as well.
    with torch.no_grad():
        mean, _, _ = policy.network(torch.as_tensor(states))
    np.testing.assert_allclose(policy.predict_with_raw(states, deterministic=True)[1], mean.numpy())


def test_rollout_copy_has_no_gradients_and_is_not_trained():
    """
    The rollout copy is never handed to an optimizer.
    """

    policy = _forced_split(PPOPolicy(4, 2, "cpu"))
    policy.rollout_network()
    assert all(not p.requires_grad for p in policy._rollout_network.parameters())
    assert {id(p) for p in policy._rollout_network.parameters()}.isdisjoint(
        id(p) for p in policy.network.parameters())


def test_invalid_rollout_device_is_rejected():
    """
    rollout_device is validated like device.
    """

    with pytest.raises(ValueError, match="rollout_device"):
        SinglePPOAgent(environment=None, rollout_device="gpu")


def test_worker_factory_does_not_carry_the_policy():
    """
    SubprocVecEnv pickles the factory into every worker: it must hold the env only,
    not the agent's network or optimizer (on CUDA: one CUDA context per worker).
    """

    agent = _fake_agent(PPOPolicy(6, 2, "cpu"))
    blob = cloudpickle.dumps(agent.make_env(0))
    assert b"actor_mean" not in blob


def test_ppo_hyperparameters_default_to_cpu_rollouts():
    """
    The PPO agents of RudoLF get rollout_device 'cpu' by default; SB3's PPO2 does not
    take it.
    """

    from fw.configuration.config import (MULTI_POLICY_SINGLE_NAME, MULTI_PPO_POLICY_NAME, PPO2_POLICY_NAME,
                                         PPO_POLICY_NAME, get_hyperparameters)

    for name in (PPO_POLICY_NAME, MULTI_PPO_POLICY_NAME, MULTI_POLICY_SINGLE_NAME):
        assert get_hyperparameters(name)["rollout_device"] == "cpu"
    assert "rollout_device" not in get_hyperparameters(PPO2_POLICY_NAME)


def test_uantwerp_surrogate_runs_on_cpu(monkeypatch, tmp_path):
    """
    The UAntwerp surrogate predictor is built on the CPU, whatever CUDA offers.
    """

    import fw.surrogate.inference_surrogate as inference_surrogate
    from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate
    from fw.surrogate.presets import PRESETS

    seen = {}

    class _Captured(Exception):
        """
        Stops the constructor once the predictor's device is known.
        """

    def fake_predictor(checkpoint, device=None):
        """
        Record the device and stop.
        """

        seen["device"] = device
        raise _Captured()

    monkeypatch.setattr(inference_surrogate, "SurrogatePredictor", fake_predictor)
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"")
    preset = next(iter(PRESETS))
    with pytest.raises(_Captured):
        UAntwerpSurrogate(ship_spec=None, ua_surrogate_config={"preset": preset, "checkpoint": str(checkpoint)})
    assert torch.device(seen["device"]).type == "cpu"


@cuda
def test_cuda_update_with_cpu_rollouts():
    """
    Training on CUDA with CPU rollouts: numpy results, CPU rollout copy that
    follows the CUDA weights, save/load round trip.
    """

    policy = PPOPolicy(6, 2, "cuda", rollout_device="cpu")
    assert next(policy.network.parameters()).device.type == "cuda"
    assert next(policy.rollout_network().parameters()).device.type == "cpu"
    states = np.random.default_rng(2).standard_normal((12, 6)).astype(np.float32)

    _fake_agent(policy).train([_episode(6, 300)])
    np.testing.assert_allclose(policy.predict_value(states), _train_values(policy, states), atol=1e-5)
    action, raw, log_prob, value = policy.predict_with_raw(states)
    assert action.shape == raw.shape == (12, 2) and log_prob.shape == value.shape == (12,)


@cuda
def test_cuda_policy_save_and_load_keep_rollout_device(tmp_path):
    """
    A policy saved from CUDA loads onto CUDA with CPU rollouts, and onto CPU alone.
    """

    policy = PPOPolicy(6, 2, "cuda", rollout_device="cpu")
    policy.save(str(tmp_path / "p"))
    loaded = PPOPolicy.load(str(tmp_path / "p"), "cuda", rollout_device="cpu")
    assert loaded.device.type == "cuda" and loaded.rollout_device.type == "cpu"
    states = np.ones((3, 6), dtype=np.float32)
    np.testing.assert_allclose(loaded.predict_value(states), _train_values(policy, states), atol=1e-5)
    assert PPOPolicy.load(str(tmp_path / "p"), "cpu").device.type == "cpu"


@cuda
def test_cuda_tensor_predict_still_uses_the_training_network():
    """
    airsim run_inference calls policy._predict with a tensor on `device`: unchanged.
    """

    policy = PPOPolicy(6, 2, "cuda", rollout_device="cpu")
    action, _, _ = policy._predict(torch.zeros(1, 6, device="cuda"), deterministic=True)
    assert action.device.type == "cuda"


# ---------------------------------------------------------------------------
# rollout_device through the agents, with a real split on a CPU-only machine:
# the fixture makes rollout_device "auto" resolve to torch.device("cpu", 0),
# which torch treats as another device than "cpu", so PPOPolicy builds and uses
# a separate rollout copy exactly as for device "cuda" + rollout_device "cpu".
# ---------------------------------------------------------------------------
SPLIT = torch.device("cpu", 0)


@pytest.fixture
def split_device(monkeypatch):
    """
    rollout_device="auto" -> cpu:0 in BaseAgent (other values unchanged).
    """

    real = base_agent_module.get_device
    monkeypatch.setattr(base_agent_module, "get_device", lambda d="auto": SPLIT if d == "auto" else real(d))
    return SPLIT


def _space_env(obs_dim: int) -> SimpleNamespace:
    """
    Minimal env with observation and action spaces.
    """

    return SimpleNamespace(
        observation_space=gym.spaces.Box(-1.0, 1.0, (obs_dim,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
    )


def _has_split(policy: PPOPolicy) -> bool:
    """
    Whether the policy uses a separate rollout copy on the split device.
    """

    return policy._rollout_network is not None and policy.rollout_device == SPLIT


def _follows(policy: PPOPolicy, obs_dim: int) -> None:
    """
    Assert the public prediction calls match the training network.
    """

    states = np.random.default_rng(5).standard_normal((4, obs_dim)).astype(np.float32)
    np.testing.assert_allclose(policy.predict_value(states), _train_values(policy, states), rtol=1e-6, atol=1e-6)


def test_single_agent_passes_rollout_device_to_new_and_loaded_policies(tmp_path, split_device):
    """
    SinglePPOAgent: constructor and load_policy both give the policy the agent's rollout_device.
    """

    agent = SinglePPOAgent(None, environment=_space_env(6), model_dir=str(tmp_path), seed=0,
                           rollout_device="auto")
    assert agent.rollout_device == SPLIT and _has_split(agent.policy)
    agent.save_policy("p")
    agent.load_policy("p")
    assert _has_split(agent.policy)
    _follows(agent.policy, 6)


def test_mpsa_passes_rollout_device_to_all_policies_and_loads(tmp_path, split_device):
    """
    MultiPolicySingleAgent: constructor, archive load and single-slot load.
    """

    agent = MultiPolicySingleAgent(environment=_space_env(8), num_policies=2, model_dir=str(tmp_path), seed=0,
                                   rollout_device="auto", watchdog_enabled=False)
    assert all(_has_split(p) for p in agent.policies)
    agent.save_policy("archive")
    agent.load_policy("archive")
    assert all(_has_split(p) for p in agent.policies)
    PPOPolicy(6, 2, "cpu").save(str(tmp_path / "slot"))
    agent.load_individual_policy(1, str(tmp_path / "slot"))
    assert _has_split(agent.policies[1])
    _follows(agent.policies[1], 6)


def test_mppo_load_keeps_each_agent_s_rollout_device(tmp_path, split_device):
    """
    MultiPPOAgent.load_policy rebuilds every agent's policy with that agent's rollout_device.
    """

    agents = [SinglePPOAgent(None, environment=_space_env(6), model_dir=str(tmp_path), seed=i,
                             rollout_device="auto") for i in range(2)]
    mppo = MultiPPOAgent.__new__(MultiPPOAgent)
    mppo.agents, mppo.model_dir, mppo.device = agents, str(tmp_path), torch.device("cpu")
    mppo.save_policy("multi")
    mppo.load_policy("multi")
    assert all(_has_split(a.policy) for a in agents)


def test_airsim_transfer_policy_gets_the_rollout_device(tmp_path):
    """
    build_transfer_policy passes target_rollout_device on, and the rollout copy
    carries the transferred weights.
    """

    from airsim.train.transfer_learning import build_transfer_policy

    PPOPolicy(6, 2, "cpu").save(str(tmp_path / "source"))
    policy = build_transfer_policy(str(tmp_path / "source.zip"), 6, 2, target_device=torch.device("cpu"),
                                   freeze_shared=False, target_rollout_device=SPLIT)
    assert _has_split(policy)
    _follows(policy, 6)


def test_update_refreshes_the_copy_even_with_a_fused_optimizer(tmp_path, split_device):
    """
    A fused Adam does not bump the parameters' version counters; the explicit
    refresh after train() must still bring the rollout copy up to date.
    """

    agent = SinglePPOAgent(None, environment=_space_env(6), model_dir=str(tmp_path), seed=0,
                           rollout_device="auto", num_epochs=2, batch_size=32)
    try:
        agent.optimizer = torch.optim.Adam(agent.policy.network.parameters(), lr=1e-2, fused=True)
    except (RuntimeError, TypeError) as exc:
        pytest.skip(f"fused Adam not available on CPU in this torch: {exc}")
    _follows(agent.policy, 6)  # copy taken
    versions = [p._version for p in agent.policy.network.parameters()]
    agent.train([_episode(6, 100)])
    if [p._version for p in agent.policy.network.parameters()] != versions:
        pytest.skip("this torch's fused Adam bumps version counters; nothing to prove here")
    _follows(agent.policy, 6)


def test_mark_weights_changed_covers_writes_torch_does_not_track():
    """
    `p.data` writes are invisible to the version counter: mark_weights_changed()
    is the documented way to refresh, and replacing the network is detected.
    """

    policy = _forced_split(PPOPolicy(4, 2, "cpu"))
    _follows(policy, 4)
    for p in policy.network.parameters():
        p.data = p.data + 0.1
    policy.mark_weights_changed()
    _follows(policy, 4)
    policy.network = PPOPolicy(4, 2, "cpu").network
    _follows(policy, 4)



def test_learn_runs_for_an_agent_without_rollout_device(tmp_path):
    """
    An agent built without rollout_device (e.g. via __new__ in older code) still gets
    through learn()'s logging; it falls back to device.
    """

    agent = SinglePPOAgent(None, environment=_space_env(6), model_dir=str(tmp_path), seed=0)
    agent.env.training_name = "no_rollout_device"
    del agent.rollout_device
    from fw.configuration.stop_condition import StopCondition
    agent.learn(StopCondition(max_time_steps=0))
