"""
Tests for the live PPO iteration CSV logger.
"""

from __future__ import annotations

import csv


def test_training_iteration_logger_writes_one_row_per_iteration(tmp_path):
    """
    One row per logged iteration; the CSV is only written on save().
    """

    from fw.common.tools.training_iteration_logger import TrainingIterationLogger

    logger = TrainingIterationLogger(
        env_name="AirSimEnv",
        agent_type="PPO",
        device="cuda",
        src_dir=tmp_path,
        log_dir="logs",
    )

    logger.log_iteration(
        iteration=0,
        mean_reward=-0.08,
        mean_advantage=0.12,
        loss=0.71,
        num_episode_batches=1,
        elapsed_seconds=3.5,
    )
    logger.log_iteration(
        iteration=1,
        mean_reward=0.15,
        mean_advantage=0.18,
        loss=0.55,
        num_episode_batches=1,
        elapsed_seconds=7.2,
    )

    assert not logger.path.exists()

    logger.save()

    assert logger.path.exists()
    assert logger.path.parent == tmp_path / "logs"

    with logger.path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["env_name"] == "AirSimEnv"
    assert rows[0]["agent_type"] == "PPO"
    assert rows[0]["device"] == "cuda"
    assert rows[0]["iteration"] == "0"
    assert rows[1]["iteration"] == "1"


def test_training_iteration_logger_records_the_rollout_device(tmp_path):
    """
    The rollout_device column holds the given rollout device, or the device without one.
    """

    from fw.common.tools.training_iteration_logger import TrainingIterationLogger

    for rollout_device, expected in (("cpu", "cpu"), (None, "cuda")):
        logger = TrainingIterationLogger(env_name="AirSimEnv", agent_type="PPO", device="cuda",
                                         src_dir=tmp_path, log_dir=f"logs_{expected}_{rollout_device}",
                                         rollout_device=rollout_device)
        logger.log_iteration(iteration=0, mean_reward=0.0, mean_advantage=0.0, loss=0.0,
                             num_episode_batches=1, elapsed_seconds=1.0)
        logger.save()
        with logger.path.open("r", newline="", encoding="utf-8") as file:
            row = next(csv.DictReader(file))
        assert row["device"] == "cuda" and row["rollout_device"] == expected
        assert list(row)[-1] == "rollout_device"   # appended last: readers by position are unaffected
