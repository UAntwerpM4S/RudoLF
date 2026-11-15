"""
RL-contract regression tests:

- envs return the real Gymnasium `terminated` / `truncated` flags, and keep
  returning them after the episode has ended;
- the PPO agents bootstrap a truncated episode from its real last
  observation (not SB3's auto-reset one), and do not treat a truncation as a
  termination in the info-less fallback;
- the tanh-squashed log-prob is evaluated at the same raw action in the rollout
  and in the update (ratio exactly 1 before any optimizer step);
- a policy saved on another device type loads on CPU;
- deep-copied vectorized workers get independent RNGs.
"""

import copy
import torch
import pytest
import numpy as np
import gymnasium as gym

from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model

START = np.array([2245.0, -47.0], dtype=np.float32)
TARGET = np.array([-1559.0, 11990.0], dtype=np.float32)


# ----------------------------------------------------------------------
# Environment step contract
# ----------------------------------------------------------------------

def _straight(env, n):
    """
    Step `n` times with a neutral rudder; return the last 5-tuple.
    """

    out = None
    for _ in range(n):
        out = env.step(np.array([0.0, 0.5], dtype=np.float32))
    return out


def test_pysimenv_time_limit_is_truncation():
    """
    At max_steps PySimEnv reports truncated=True, terminated=False, also after the end.
    """

    from fw.environments.py_sim_env import PySimEnv

    env = PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                   dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    env.reset(seed=0)
    _, reward, terminated, truncated, info = _straight(env, 5)
    assert (terminated, truncated) == (False, True)
    assert info["truncations"] is True and info["terminations"] is False
    assert isinstance(reward, float)

    _, _, terminated, truncated, _ = _straight(env, 2)          # stepping a finished episode
    assert (terminated, truncated) == (False, True)


def test_colav_flags_persist_after_episode_end():
    """
    The COLAV env does not 'un-terminate' when stepped after the end.
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv

    env = PySimColavEnv(time_step=1.0, max_steps=5, ship_pos=START, target_pos=TARGET,
                        dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    env.reset(seed=0)
    done_flags = None
    for _ in range(10):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.5], dtype=np.float32))
        if done_flags is None and (terminated or truncated):
            done_flags = (terminated, truncated)
        elif done_flags is not None:
            assert (terminated, truncated) == done_flags
    assert done_flags is not None
    assert not (done_flags[0] and done_flags[1])


# ----------------------------------------------------------------------
# Bootstrapping
# ----------------------------------------------------------------------

class _CountingEnv(gym.Env):
    """
    Observation = step index; truncates after 3 steps; info carries no framework keys.
    """

    observation_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    type_name = training_name = "CountingEnv"

    def __init__(self):
        """
        Create the toy env.
        """

        self.t = 0

    def reset(self, seed=None, options=None):
        """
        Restart counting at 0.
        """

        super().reset(seed=seed)
        self.t = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        """
        Advance one step; truncate at t == 3.
        """

        self.t += 1
        return np.full(4, float(self.t), dtype=np.float32), 1.0, False, self.t >= 3, {}

    def get_neutral_action(self, env_id, action):
        """
        Neutral action for finished envs (BasePPOAgent hook).
        """

        return np.zeros_like(action)


def test_step_envs_returns_terminal_observation_and_real_truncation(tmp_path):
    """
    Without framework info keys: truncation is not a termination, and the final state is the real one.
    """

    from stable_baselines3.common.vec_env import DummyVecEnv
    from fw.policies.single_ppo_agent import SinglePPOAgent

    agent = SinglePPOAgent(environment=_CountingEnv(), model_dir=str(tmp_path))
    agent._vec_env = DummyVecEnv([_CountingEnv])
    agent._vec_env.reset()
    for _ in range(2):
        agent._step_envs(np.zeros((1, 2), dtype=np.float32))
    next_states, _, terminations, truncations, final_states = agent._step_envs(np.zeros((1, 2), dtype=np.float32))

    assert truncations.tolist() == [True]
    assert terminations.tolist() == [False]                 # True would zero the bootstrap
    assert next_states[0][0] == 0.0                          # SB3 auto-reset observation
    assert final_states[0][0] == 3.0                         # real last observation


# ----------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------

def test_raw_log_prob_ratio_is_one_before_update():
    """
    Rollout and update evaluate the log-prob at the same raw action, also for
    saturated samples, after the NumPy round-trip and in shuffled minibatches.
    """

    from fw.policies.ppo_policy import PPOPolicy

    torch.manual_seed(0)
    policy = PPOPolicy(4, 2, "cpu")
    with torch.no_grad():
        policy.network.actor_log_std.fill_(2.0)              # std = 7.4: many saturated samples
    states = np.random.default_rng(0).standard_normal((512, 4)).astype(np.float32)

    # Rollout: NumPy outputs, as stored in the trajectories.
    actions, raw_actions, old_log_probs, _ = policy.predict_with_raw(states)
    assert (np.abs(actions) > 0.999).any()                   # saturation actually exercised
    assert np.allclose(actions, np.tanh(raw_actions), atol=1e-6)

    # Update: tensors rebuilt from the stored arrays, evaluated per minibatch.
    all_states = torch.as_tensor(states, dtype=torch.float32)
    all_raw = torch.as_tensor(raw_actions, dtype=torch.float32)
    all_old = torch.as_tensor(old_log_probs, dtype=torch.float32)
    indices = torch.randperm(len(states))
    with torch.no_grad():
        for start in range(0, len(states), 64):
            mb = indices[start:start + 64]
            mean, std, _ = policy.network(all_states[mb])
            new_log_probs = PPOPolicy.raw_log_prob(torch.distributions.Normal(mean, std), all_raw[mb])
            ratio = torch.exp(new_log_probs - all_old[mb])
            assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


def test_policy_trained_on_other_device_loads_on_cpu(tmp_path):
    """
    The device stored in a checkpoint is informational: a 'cuda' checkpoint loads on CPU.
    """

    from fw.policies import ppo_policy
    from fw.policies.ppo_policy import PPOPolicy

    policy = PPOPolicy(4, 2, "cpu")
    path = tmp_path / "p"
    policy.save(str(path))

    original = PPOPolicy.serialize_policy

    def _as_cuda(self):
        """
        Serialize as if the policy had been trained on a GPU.
        """

        data = original(self)
        data.setdefault("data", {})["device"] = "cuda"
        return data

    PPOPolicy.serialize_policy = _as_cuda
    try:
        policy.save(str(path))
    finally:
        PPOPolicy.serialize_policy = original

    loaded = ppo_policy.PPOPolicy.load(str(path), device="cpu")
    assert next(loaded.network.parameters()).device.type == "cpu"


def test_deep_copied_workers_get_independent_rngs():
    """
    reseed_env_rng gives every deep copy its own generator.
    """

    from fw.common.tools.ppo_utils import reseed_env_rng

    class _E:
        """
        Env stub with a generator.
        """

        def __init__(self):
            """
            Create the generator.
            """

            self._rng = np.random.default_rng(1)

    template = _E()
    a, b = copy.deepcopy(template), copy.deepcopy(template)
    assert a._rng.integers(1 << 30) == b._rng.integers(1 << 30)   # copies share the stream
    reseed_env_rng(a)
    reseed_env_rng(b)
    assert a._rng.integers(1 << 30, size=4).tolist() != b._rng.integers(1 << 30, size=4).tolist()


@pytest.mark.parametrize("ship_model", [ShipModel.MYZAKO_37, ShipModel.TRIPOLI])
def test_rudder_change_penalty_limit_is_in_radians(ship_model):
    """
    The rudder-change limit has the same unit (rad) as the rudder change, for every ship.
    """

    from fw.environments.py_sim_env import PySimEnv

    env = PySimEnv(time_step=2.0, max_steps=50, start_pos=START, target_pos=TARGET,
                   dynamics_model=Model.FOSSEN, ship_model=ship_model)
    env.reset(seed=0)
    captured = {}
    original = env.reward_calculator._component_functions["rudder_change"]

    def _spy(rudder_change, max_rudder_change, **kwargs):
        """
        Record the arguments of the rudder-change component.
        """

        captured.update(rudder_change=rudder_change, max_rudder_change=max_rudder_change)
        return original(rudder_change=rudder_change, max_rudder_change=max_rudder_change, **kwargs)

    env.reward_calculator._component_functions["rudder_change"] = _spy
    env.step(np.array([1.0, 0.5], dtype=np.float32))          # full rudder from zero

    spec = env.ship.specifications
    assert captured["max_rudder_change"] == pytest.approx(spec.max_rudder_rate * 2.0 + 1e-6)   # rad + rounding tolerance
    assert captured["rudder_change"] == pytest.approx(abs(float(env.performed_action[0])))   # rad


@pytest.mark.parametrize("ship_model", [ShipModel.MYZAKO_37, ShipModel.TRIPOLI])
def test_full_rate_rudder_move_is_not_a_violation(ship_model):
    """
    With smoothing on, a move at exactly the top slew rate gets the ratio branch, not the hard -1.
    """

    from fw.environments.py_sim_env import PySimEnv

    env = PySimEnv(time_step=2.0, max_steps=50, start_pos=START, target_pos=TARGET,
                   dynamics_model=Model.FOSSEN, ship_model=ship_model)
    env.enable_smoothing = True
    env.reset(seed=0)
    ratios = []
    original = env.reward_calculator._component_functions["rudder_change"]

    def _spy(rudder_change, max_rudder_change, **kwargs):
        """
        Record rudder_change / max_rudder_change.
        """

        ratios.append(rudder_change / max_rudder_change)
        return original(rudder_change=rudder_change, max_rudder_change=max_rudder_change, **kwargs)

    env.reward_calculator._component_functions["rudder_change"] = _spy
    for k in range(10):                                         # alternating full rudder: always rate-limited
        env.step(np.array([1.0 if k % 2 == 0 else -1.0, 0.5], dtype=np.float32))
    assert max(ratios) < 1.0
    assert max(ratios) == pytest.approx(1.0, abs=1e-4)          # really at the top rate


@pytest.mark.parametrize("ship_model", [ShipModel.MYZAKO_37, ShipModel.TRIPOLI])
@pytest.mark.parametrize("smoothing", [False, True])
def test_observation_stays_in_its_space(ship_model, smoothing):
    """
    Every observation lies in the Box; hard-over rudder is observed as +1 and -1
    (within the jitter threshold), commanded thrust 0.5 as 0.5, and the Fossen
    yaw rate does not saturate its scale.
    """

    from fw.environments.py_sim_env import PySimEnv

    env = PySimEnv(time_step=2.0, max_steps=50, start_pos=START, target_pos=TARGET,
                   dynamics_model=Model.FOSSEN, ship_model=ship_model, wind=True, current=True)
    env.enable_smoothing = smoothing
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert float(obs[12]) == -1.0                               # min_thrust after reset
    rudder_obs, yaw_obs = [], []
    for k in range(40):                                         # hard over to port, then to starboard
        obs, _, terminated, truncated, _ = env.step(np.array([1.0 if k < 20 else -1.0, 0.5], dtype=np.float32))
        assert env.observation_space.contains(obs)
        rudder_obs.append(float(obs[10]))
        yaw_obs.append(float(obs[2]))
        assert not (terminated or truncated)
    spec = env.ship.specifications
    tol = spec.rudder_jitter_threshold / spec.max_rudder_angle + 1e-5
    assert 1.0 - tol <= max(rudder_obs) <= 1.0
    assert -1.0 <= min(rudder_obs) <= -1.0 + tol
    # With smoothing the thrust stops within the jitter threshold of its command.
    thrust_tol = 2.0 * spec.thrust_jitter_threshold / (spec.max_thrust - spec.min_thrust) if smoothing else 0.0
    assert float(obs[12]) == pytest.approx(0.5, abs=thrust_tol + 1e-5)
    assert max(abs(y) for y in yaw_obs) < 1.0


def test_observation_9_is_parallel_to_the_previous_to_next_line():
    """
    Observation 9 is the heading error to the direction previous checkpoint -> next checkpoint.
    """

    from fw.environments.py_sim_env import PySimEnv

    env = PySimEnv(time_step=2.0, max_steps=50, start_pos=START, target_pos=TARGET,
                   dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    obs, _ = env.reset(seed=0)
    previous_cp = env.checkpoints[env.checkpoint_index - 1]["pos"]
    next_cp = env.checkpoints[env.checkpoint_index + 1]["pos"]
    line = next_cp - previous_cp
    expected = env.normalize_angle(np.arctan2(line[1], line[0]) - env.phys_sim.state.heading) / np.pi
    assert float(obs[9]) == pytest.approx(float(expected), abs=1e-6)


def test_vec_env_factory_reseeds_each_worker(tmp_path):
    """
    SinglePPOAgent.make_env gives every worker copy its own RNG (not a copy of the template's).
    """

    from fw.environments.py_sim_env import PySimEnv
    from fw.policies.single_ppo_agent import SinglePPOAgent

    env = PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                   dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    agent = SinglePPOAgent(environment=env, model_dir=str(tmp_path))
    w1, w2 = agent.make_env()(), agent.make_env()()            # Monitor-wrapped copies
    draws = [w.env._rng.integers(1 << 30, size=4).tolist() for w in (w1, w2)]
    assert draws[0] != draws[1]


def test_colav_static_obstacle_is_not_despawned():
    """
    The despawn rule only removes dynamic target ships.
    """

    from fw.environments.py_sim_colav_env import PySimColavEnv

    env = PySimColavEnv(time_step=1.0, max_steps=100000, ship_pos=START, target_pos=TARGET,
                        dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37,
                        scenario_types=["static"], scenario_probs=[1.0])
    env.scenario_builder.check_collision_entity_collision = lambda *a, **k: False
    env.reset(seed=1)
    seen, removed = False, False
    for _ in range(1200):
        err = env._calculate_heading_error(env.los_guidance_heading(), env.phys_sim.state.heading)
        _, _, terminated, truncated, _ = env.step(np.array([np.clip(2.0 * err, -1, 1), 0.6], dtype=np.float32))
        if env.collision_entities:
            seen = True
        elif seen:
            removed = True
        if terminated or truncated:
            break
    assert seen and not removed


def test_pipeline_stop_condition_type_error_is_a_spec_error():
    """
    A non-numeric patience is reported by the loader, not when the block starts.
    """

    from fw.scheduling.run_loader import PipelineSpecError, _parse_stop_condition

    with pytest.raises(PipelineSpecError):
        _parse_stop_condition({"criterion": "TARGET_REWARD", "target_reward": 0.9, "patience": "3"}, "block[0]", "b")


# ----------------------------------------------------------------------
# Seeded worker RNGs ('seed' hyperparameter)
# ----------------------------------------------------------------------

def _worker_draws(seed, tmp_path, n_workers=2):
    """
    First draws of the env RNGs of `n_workers` workers of a SinglePPOAgent seeded with `seed`.
    """

    from fw.environments.py_sim_env import PySimEnv
    from fw.policies.single_ppo_agent import SinglePPOAgent

    env = PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                   dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    agent = SinglePPOAgent(environment=env, model_dir=str(tmp_path), seed=seed)
    workers = [agent.make_env(s)() for s in agent._spawn_worker_seeds(n_workers)]   # as _reset_envs does
    return agent, [w.env._rng.integers(1 << 30, size=4).tolist() for w in workers]


def test_worker_rngs_are_reproducible_from_the_seed(tmp_path):
    """
    Same 'seed' -> same worker streams; workers differ from each other; another seed differs.
    """

    _, run_a = _worker_draws(42, tmp_path)
    _, run_b = _worker_draws(42, tmp_path)
    _, run_c = _worker_draws(43, tmp_path)
    assert run_a == run_b
    assert run_a[0] != run_a[1]
    assert run_a != run_c


def test_unset_seed_is_drawn_and_logged(tmp_path, caplog):
    """
    seed=None draws a base seed and logs it, so the run can be reproduced afterward.
    """

    with caplog.at_level("INFO", logger="fw.policies.base_agent"):
        agent, run = _worker_draws(None, tmp_path)
    assert isinstance(agent.seed, int) and agent.seed >= 0
    assert any(str(agent.seed) in r.getMessage() for r in caplog.records)
    _, rerun = _worker_draws(agent.seed, tmp_path)
    assert run == rerun


@pytest.mark.parametrize("bad_seed", [-1, True, "42", 4.2])
def test_invalid_seed_is_refused(bad_seed):
    """
    Only None or a non-negative int is accepted (fail fast, no silent bool -> 1).
    """

    from fw.common.tools.ppo_utils import resolve_base_seed

    with pytest.raises(ValueError):
        resolve_base_seed(bad_seed)


# ----------------------------------------------------------------------
# Target-ship despawn threshold
# ----------------------------------------------------------------------

def _despawn_after_pass(closest_offset, env_cls=None):
    """
    Place one dynamic TS at `encounter_range + closest_offset` from the OS,
    then beyond forget_range, running the ScenarioBuilder step each time.
    Returns True if the TS was despawned. `env_cls` defaults to the polar env.
    """

    from fw.collision_avoidance.collision_entity import CollisionEntity
    from fw.environments.py_sim_colav_env import PySimColavEnv

    env_cls = env_cls or PySimColavEnv
    env = env_cls(time_step=1.0, max_steps=100000, ship_pos=START, target_pos=TARGET,
                  dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37)
    env.reset(seed=1)
    if env.TRACKS_TARGET_SHIP:
        assert env.ts_idx is None
    builder = env.scenario_builder
    ts = CollisionEntity(x=0.0, y=0.0, radius=5.0, is_dynamic=True, entity_type="ship")
    env.collision_entities = [ts]
    env._sync_collision_entity_views()
    builder.step_counter = builder._active_cooldown()            # past the cooldown, as after a spawn

    os_x, os_y = float(env.ship_pos[0]), float(env.ship_pos[1])
    for distance in (env.encounter_range + closest_offset, env.forget_range + 50.0):
        ts.set_position(os_x + distance, os_y)
        builder.on_step(env.time_step)
    return not env.collision_entities


def test_ts_that_came_within_encounter_range_despawns_when_it_leaves():
    """
    Closest approach inside encounter_range, then beyond forget_range: despawned.
    """

    assert _despawn_after_pass(-10.0)


def test_ts_passing_only_through_the_hysteresis_band_is_not_despawned():
    """
    Closest approach between encounter_range and forget_range: kept.
    """

    assert not _despawn_after_pass(+10.0)


def test_planner_env_never_despawns_a_target_ship():
    """
    Envs without TRACKS_TARGET_SHIP (the planner env) keep their TS.
    """

    from fw.environments.py_sim_colav_planner import PySimColavPlannerEnv

    assert PySimColavPlannerEnv.TRACKS_TARGET_SHIP is False
    assert not _despawn_after_pass(-10.0, env_cls=PySimColavPlannerEnv)


def test_multi_ppo_seeds_its_sub_environments(tmp_path):
    """
    MultiPPOAgent steps its sub-envs directly: each gets its own child of 'seed', reproducibly.
    """

    from fw.environments.multi_sim_env import MultiSimEnv
    from fw.environments.py_sim_env import PySimEnv
    from fw.policies.multi_ppo_agent import MultiPPOAgent
    from fw.policies.single_ppo_agent import SinglePPOAgent

    def _draws(seed):
        """
        First RNG draws of every sub-env after building an MPPO agent with `seed`.
        """

        subs = [PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                         dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37) for _ in range(2)]
        env = MultiSimEnv(sub_environments=subs)
        agents = [SinglePPOAgent(environment=e, model_dir=str(tmp_path), seed=seed)
                  for e in env.sub_environments.values()]
        agent = MultiPPOAgent(None, environment=env, agents=agents, model_dir=str(tmp_path), seed=seed)
        return [e._rng.integers(1 << 30, size=3).tolist() for e in agent.env.sub_environments.values()]

    a, b = _draws(5), _draws(5)
    assert a == b and a[0] != a[1]
