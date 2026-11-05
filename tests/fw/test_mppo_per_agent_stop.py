"""
MultiPPOAgent per-agent stop condition: each agent is judged on its own reward, a
converged agent is frozen (no more updates), and the run ends when all are frozen.
"""

import numpy as np

from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.configuration.stop_condition import Criterion, StopCondition

START = np.array([2245.0, -47.0], dtype=np.float32)
TARGET = np.array([-1559.0, 11990.0], dtype=np.float32)


def _episode(env_id, reward):
    """
    A minimal trajectory dict as produced by run_parallel_episodes.
    """

    return {"env_id": env_id, "rewards": np.full(3, reward, dtype=np.float32),
            "advantages": np.zeros(3, dtype=np.float32)}


def _mppo(tmp_path):
    """
    MPPO agent on two PySimEnv sub-envs (no rollouts are run).
    """

    from fw.environments.multi_sim_env import MultiSimEnv
    from fw.environments.py_sim_env import PySimEnv
    from fw.policies.multi_ppo_agent import MultiPPOAgent
    from fw.policies.single_ppo_agent import SinglePPOAgent

    subs = [PySimEnv(time_step=2.0, max_steps=5, start_pos=START, target_pos=TARGET,
                     dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37) for _ in range(2)]
    env = MultiSimEnv(sub_environments=subs)
    agents = [SinglePPOAgent(environment=e, model_dir=str(tmp_path), seed=0) for e in env.sub_environments.values()]
    return MultiPPOAgent(None, environment=env, agents=agents, model_dir=str(tmp_path), seed=0)


def test_converged_agent_is_frozen_and_run_ends_when_all_are(tmp_path, monkeypatch):
    """
    Agent 0 converges at once, agent 1 from iteration 5: agent 0 gets exactly
    `patience` updates, agent 1 keeps training until it converges too.
    """

    agent = _mppo(tmp_path)
    iteration = {"n": 0}

    def _rollout(num_envs):
        """
        Agent 0 always on target; agent 1 on target from the 5th rollout.
        """

        n = iteration["n"]
        iteration["n"] += 1
        return [_episode(0, 1.0), _episode(1, 1.0 if n >= 5 else 0.5)]

    updates = [0, 0]
    for agent_id, sub in enumerate(agent.agents):
        def _train(episodes, agent_id=agent_id):
            """
            Count updates instead of running PPO.
            """

            updates[agent_id] += 1
            return 0.1
        monkeypatch.setattr(sub, "train", _train)
    monkeypatch.setattr(agent, "run_parallel_episodes", _rollout)

    captured = {}
    monkeypatch.setattr("fw.policies.multi_ppo_agent.dump_multi_agent_metrics",
                        lambda model_dir, metrics: captured.update(metrics))

    agent.learn(StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=0.99,
                              patience=2, max_time_steps=50))

    assert updates == [2, 7]                              # agent 0 frozen after 2 on-target updates
    assert iteration["n"] == 7                            # run ended when both were frozen, not at 50
    frozen_0 = [m["frozen"] for m in captured["agent0"]]
    assert frozen_0 == [False, False, True, True, True, True, True]
    assert all(np.isnan(m["loss"]) for m in captured["agent0"][2:])   # no updates while frozen
    assert captured["agent0"][-1]["reward"] == 1.0                    # reward still logged


def test_without_convergence_the_run_stops_at_max_time_steps(tmp_path, monkeypatch):
    """
    No agent reaches the target: nobody is frozen and the run lasts max_time_steps.
    """

    agent = _mppo(tmp_path)
    monkeypatch.setattr(agent, "run_parallel_episodes", lambda n: [_episode(0, 0.1), _episode(1, 0.1)])
    for sub in agent.agents:
        monkeypatch.setattr(sub, "train", lambda episodes: 0.1)
    captured = {}
    monkeypatch.setattr("fw.policies.multi_ppo_agent.dump_multi_agent_metrics",
                        lambda model_dir, metrics: captured.update(metrics))

    agent.learn(StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=0.99,
                              patience=2, max_time_steps=6))

    assert len(captured["agent0"]) == len(captured["agent1"]) == 6
    assert not any(m["frozen"] for m in captured["agent0"] + captured["agent1"])


def _run(agent, monkeypatch, rollout, stop):
    """
    Run learn() with a scripted rollout, stub updates/saves; return (metrics, best saves per agent).
    """

    for sub in agent.agents:
        monkeypatch.setattr(sub, "train", lambda episodes: 0.1)
    best_saves = [[] for _ in agent.agents]
    for agent_id, sub in enumerate(agent.agents):
        monkeypatch.setattr(sub, "save_policy", lambda name, agent_id=agent_id: best_saves[agent_id].append(name))
    monkeypatch.setattr(agent, "save_policy", lambda name: None)
    monkeypatch.setattr(agent, "run_parallel_episodes", rollout)
    captured = {}
    monkeypatch.setattr("fw.policies.multi_ppo_agent.dump_multi_agent_metrics",
                        lambda model_dir, metrics: captured.update(metrics))
    agent.learn(stop)
    return captured, best_saves


def test_missing_trajectory_breaks_the_patience_streak(tmp_path, monkeypatch, caplog):
    """
    on, on, missing, on, on, on, on with patience 4: agent 0 freezes at the 4th on-target
    iteration after the gap (iteration 6), not after 4 in total.
    """

    agent = _mppo(tmp_path)
    state = {"n": 0}

    def _rollout(num_envs):
        """
        Agent 0 has no trajectory in iteration 2; both agents are always on target.
        """

        n = state["n"]
        state["n"] += 1
        episodes = [_episode(1, 1.0)]
        if n != 2:
            episodes.insert(0, _episode(0, 1.0))
        return episodes

    with caplog.at_level("INFO", logger="fw.policies.multi_ppo_agent"):
        _run(agent, monkeypatch, _rollout,
             StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=0.99,
                           patience=4, max_time_steps=50))
    frozen_at = {r.getMessage().split(" | ")[1]: r.getMessage().split(" | ")[0]
                 for r in caplog.records if " frozen: " in r.getMessage()}
    assert frozen_at["agent 0 frozen: Reached target_reward = 0.99 for 4 consecutive steps"] == "Iteration 6"
    assert frozen_at["agent 1 frozen: Reached target_reward = 0.99 for 4 consecutive steps"] == "Iteration 3"
    assert state["n"] == 7                                 # run ended when agent 0 froze


def test_agent_frozen_before_best_save_threshold_still_gets_a_best_policy(tmp_path, monkeypatch):
    """
    Freezing at iteration 1 (< MINIMUM_ITERATIONS_FOR_BEST_SAVE): its policy is saved as best once.
    """

    agent = _mppo(tmp_path)
    state = {"n": 0}

    def _rollout(num_envs):
        """
        Agent 0 on target at once; agent 1 only from iteration 25 (after the threshold).
        """

        n = state["n"]
        state["n"] += 1
        return [_episode(0, 1.0), _episode(1, 1.0 if n >= 25 else 0.5)]

    _, best_saves = _run(agent, monkeypatch, _rollout,
                         StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=0.99,
                                       patience=2, max_time_steps=50))
    assert len(best_saves[0]) == 1 and best_saves[0][0].startswith("Best_")
    assert len(best_saves[1]) >= 1                         # normal best saves from iteration 20 on


def test_unconverged_agents_are_logged_at_max_time_steps(tmp_path, monkeypatch, caplog):
    """
    Run ends at max_time_steps with agent 1 never on target: the log names it.
    """

    agent = _mppo(tmp_path)
    with caplog.at_level("INFO", logger="fw.policies.multi_ppo_agent"):
        _run(agent, monkeypatch, lambda n: [_episode(0, 1.0), _episode(1, 0.1)],
             StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=0.99,
                           patience=2, max_time_steps=6))
    msg = [r.getMessage() for r in caplog.records if "did not meet" in r.getMessage()]
    assert len(msg) == 1 and "[1]" in msg[0] and "frozen: [0]" in msg[0]
