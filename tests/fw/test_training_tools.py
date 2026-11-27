"""
The training-analysis tools in src/: compare_training_runs.py (must score runs
exactly as StopCondition stops them) and benchmark_ppo_device.py (smoke test).
Both are scripts, not package modules, so they are loaded from their file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fw.configuration.stop_condition import Criterion, StopCondition

SRC = Path(__file__).resolve().parents[2] / "src"


def _load(name: str):
    """
    Import a script from src/ by path.
    """

    spec = importlib.util.spec_from_file_location(name, SRC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load("compare_training_runs")


def _stop_condition_hit(rewards, losses, criterion, target, target_loss, patience):
    """
    Iteration at which the training loop stops on the target, using StopCondition
    exactly as SinglePPOAgent.learn() does (one should_stop call per logged
    iteration, max_time_steps never reached inside the loop), or None.
    """

    condition = StopCondition(criterion, max_time_steps=10 ** 9, target_reward=target,
                              target_loss=target_loss, patience=patience)
    for i, (reward, loss) in enumerate(zip(rewards, losses)):
        if condition.should_stop(current_step=i, mean_reward=reward, mean_loss=loss)[0]:
            return i
    return None


@pytest.mark.parametrize("patience", [None, 0, 1, 2, 4])
@pytest.mark.parametrize("with_loss", [False, True])
def test_first_target_hit_matches_stop_condition(patience, with_loss):
    """
    On random reward/loss sequences the script finds the same iteration as StopCondition.
    """

    rng = np.random.default_rng(0)
    criterion = Criterion.TARGET_REWARD_AND_LOSS if with_loss else Criterion.TARGET_REWARD
    target_loss = 0.5 if with_loss else None
    for _ in range(300):
        n = int(rng.integers(1, 40))
        rewards = pd.Series(rng.choice([0.0, 0.92, 0.93, 0.95], size=n))
        losses = pd.Series(rng.choice([0.1, 0.5, 0.9], size=n))
        expected = _stop_condition_hit(rewards, losses, criterion, 0.93, target_loss, patience)
        got = compare.first_target_hit(rewards, 0.93, patience, losses, target_loss)
        assert got == expected


def test_block_target_follows_the_criterion_and_overrides():
    """
    Target per criterion (none for NBR_POLICY_UPDATES), patience like StopCondition,
    and --target / --patience replacing the spec's values.
    """

    assert compare.block_target({"criterion": "NBR_POLICY_UPDATES", "target_reward": 0.9}, None, None) == (None, None, 1)
    assert compare.block_target({"criterion": "TARGET_REWARD", "target_reward": 0.94, "patience": 4},
                                None, None) == (0.94, None, 4)
    assert compare.block_target({"criterion": "TARGET_REWARD", "target_reward": 0.94, "patience": 0},
                                None, None) == (0.94, None, 1)
    assert compare.block_target({"criterion": "TARGET_REWARD", "target_reward": 0.94, "patience": None},
                                None, None) == (0.94, None, 1)
    assert compare.block_target({"criterion": "TARGET_REWARD_AND_LOSS", "target_reward": 0.9, "target_loss": 0.2,
                                 "patience": 3}, 0.93, 2) == (0.93, 0.2, 2)
    assert compare.block_target({"criterion": "NBR_POLICY_UPDATES"}, 0.93, None) == (0.93, None, 1)


def test_latest_iteration_log_picks_the_newest_log_of_exactly_that_block(tmp_path):
    """
    The newest timestamp wins; a block whose name starts with this one's is ignored.
    """

    for name in ("bs64_seed1_PPO_iterations_20261127_160519.csv",
                 "bs64_seed1_PPO_iterations_20261128_090000.csv",
                 "bs64_seed10_PPO_iterations_20261129_000000.csv",
                 "bs64_seed1_extra_PPO_iterations_20261130_000000.csv"):
        (tmp_path / name).write_text("x")
    assert compare.latest_iteration_log(tmp_path, "bs64_seed1").name == "bs64_seed1_PPO_iterations_20261128_090000.csv"
    assert compare.latest_iteration_log(tmp_path, "bs32_seed1") is None


def _write_log(log_dir: Path, block: str, rewards: list, episodes_per_iter: int = 1) -> None:
    """
    Iteration log as TrainingIterationLogger writes it (the columns the script reads).
    """

    n = len(rewards)
    pd.DataFrame({
        "iteration": range(n), "mean_reward": rewards, "mean_advantage": 0.0, "loss": 0.3,
        "num_episode_batches": episodes_per_iter, "elapsed_seconds": np.arange(1, n + 1) * 6.0,
    }).to_csv(log_dir / f"{block}_PPO_iterations_20261127_120000.csv", index=False)


def test_compare_main_scores_two_specs_on_one_target(tmp_path, monkeypatch, capsys):
    """
    End to end: two specs with different targets, filtered with --blocks and scored
    on --target; table and figure are produced.
    """

    def spec(path, blocks, target):
        """
        Write a spec with TARGET_REWARD train blocks.
        """

        path.write_text(json.dumps({"blocks": [
            {"name": name, "action": "train", "hyperparameter_overrides": overrides,
             "stop_condition": {"criterion": "TARGET_REWARD", "target_reward": target, "patience": 2}}
            for name, overrides in blocks]}))

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    spec(tmp_path / "a.json", [("bs64_seed1", {"batch_size": 64, "seed": 1}),
                               ("bs512_seed1", {"batch_size": 512, "seed": 1})], 0.94)
    spec(tmp_path / "b.json", [("envs4_seed1", {"batch_size": 64, "num_envs": 4, "seed": 1})], 0.93)
    _write_log(log_dir, "bs64_seed1", [0.0, 0.935, 0.935, 0.95, 0.95])          # 0.93: hit at 2; 0.94: at 4
    _write_log(log_dir, "envs4_seed1", [0.0, 0.94, 0.94], episodes_per_iter=4)  # hit at 2, 12 episodes
    out = tmp_path / "cmp.png"
    monkeypatch.setattr(sys, "argv", ["compare_training_runs.py", "--spec", str(tmp_path / "a.json"),
                                      str(tmp_path / "b.json"), "--blocks", "bs64_*", "envs*",
                                      "--target", "0.93", "--log-dir", str(log_dir), "--out", str(out)])
    compare.main()
    table = capsys.readouterr().out
    assert "bs512_seed1" not in table
    rows = {line.split()[0]: line.split() for line in table.splitlines() if line.strip().startswith(("bs64", "envs4"))}
    header = next(line.split() for line in table.splitlines() if line.strip().startswith("block"))
    col = {name: header.index(name) for name in ("target", "target_reached_at_iter", "episodes_to_target")}
    assert rows["bs64_seed1"][col["target"]] == "0.93" and rows["bs64_seed1"][col["target_reached_at_iter"]] == "2"
    assert rows["envs4_seed1"][col["episodes_to_target"]] == "12"
    assert out.is_file()


def test_benchmark_functions_run_on_cpu():
    """
    Smoke test: the benchmark's timing functions run and return positive times.
    """

    bench = _load("benchmark_ppo_device")
    assert bench.time_rollout_inference("cpu", 5, 2, 3, 5) > 0
    assert bench.time_ppo_update("cpu", 5, 2, 64, 32, 1, repeats=2) > 0
