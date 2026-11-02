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
