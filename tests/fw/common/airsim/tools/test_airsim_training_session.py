"""
Regression tests for the AirSim training session exporter.
"""

from __future__ import annotations

import json
import numpy as np

from pathlib import Path
from fw.common.airsim.tools.airsim_training_session import AirSimTrainingSession


def test_airsim_training_session_writes_logs_metadata_and_plots(tmp_path):
    """
    Finalizing a training session should emit the expected bundle of files.
    """

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    session = AirSimTrainingSession(
        env_name="AirSimEnv",
        agent_type="PPO",
        device="cpu",
        src_dir=src_dir,
        model_dir=tmp_path / "models",
    )

    session.set_route_snapshot(
        {
            "route_points": [[0.0, 0.0], [100.0, 0.0], [200.0, 20.0]],
            "checkpoints": [
                {"index": 0, "x": 0.0, "y": 0.0, "radius": 20.0, "reward": 0.0, "distance_m": 0.0, "progress_ratio": 0.0},
                {"index": 1, "x": 100.0, "y": 0.0, "radius": 20.0, "reward": 8.0, "distance_m": 100.0, "progress_ratio": 0.5},
                {"index": 2, "x": 200.0, "y": 20.0, "radius": 30.0, "reward": 0.0, "distance_m": 205.0, "progress_ratio": 1.0},
            ],
            "overall": [[-10.0, -50.0], [240.0, -50.0], [240.0, 60.0], [-10.0, 60.0]],
            "obstacles": [[-5.0, -20.0], [220.0, -20.0], [220.0, 40.0], [-5.0, 40.0]],
            "initial_ship_pos": [0.0, 0.0],
            "target_pos": [200.0, 20.0],
            "captured_startup_pose_airsim": {"x": 1.0, "y": 2.0, "z": -1.0, "yaw": 0.1},
            "captured_startup_framework_heading_deg": 5.0,
            "save_slot": "simulator_1",
            "load_world_on_start": False,
            "capture_live_start_pose": True,
            "no_rendering_mode": True,
            "simulation_speed": 20.0,
            "reset_mode": "rewind_pose",
        }
    )
    session.log_run_events(
        [
            {
                "timestamp_utc": "2026-04-16T08:12:21+00:00",
                "event_type": "respawn",
                "episode": 1,
                "episode_step": 0,
                "checkpoint_index": 1,
                "ship_x": 0.0,
                "ship_y": 0.0,
                "heading_deg": 0.0,
                "speed": 0.0,
            },
            {
                "timestamp_utc": "2026-04-16T08:12:30+00:00",
                "event_type": "checkpoint_passed",
                "episode": 1,
                "episode_step": 12,
                "checkpoint_index": 1,
                "ship_x": 100.0,
                "ship_y": 0.0,
                "heading_deg": 2.0,
                "speed": 4.0,
                "reward": 8.0,
            },
        ]
    )
    session.log_iteration(
        iteration=0,
        mean_reward=-0.01,
        mean_advantage=-0.02,
        loss=-0.03,
        num_episode_batches=1,
        elapsed_seconds=12.5,
    )
    training_metrics = [
        {"loss": -0.03, "advantage": -0.02, "reward": -0.01},
        {"loss": -0.02, "advantage": -0.01, "reward": 0.05},
    ]

    session.finalize(
        training_metrics=training_metrics,
        interrupted=False,
        skipped_short_rollouts=1,
        last_completed_iteration=1,
        best_mean_reward=0.05,
        stop_rule="2000 policy updates",
    )

    assert session.run_log_path.exists()
    assert session.iteration_logger.path.exists()
    assert session.training_metrics_csv_path.exists()
    assert session.route_plot_path.exists()
    assert session.training_metrics_plot_path.exists()
    assert session.metadata_path.exists()

    metadata = json.loads(session.metadata_path.read_text(encoding="utf-8"))
    assert Path(metadata["iteration_log"]).name == session.iteration_logger.path.name
    assert metadata["interrupted"] is False
    assert metadata["run_event_count"] == 2
    assert metadata["route_summary"]["checkpoint_count"] == 3


def test_auto_normalize_series_maps_data_to_minus_one_plus_one():
    """
    Metric series should be rescaled to the full `[-1, 1]` interval.
    """

    normalized, value_min, value_max = AirSimTrainingSession._auto_normalize_series(
        np.array([2.0, 4.0, 6.0], dtype=float)
    )

    assert np.allclose(normalized, np.array([-1.0, 0.0, 1.0], dtype=float))
    assert value_min == 2.0
    assert value_max == 6.0


def test_auto_normalize_series_keeps_constant_series_centered():
    """
    Constant finite data should map to zero without crashing.
    """

    normalized, value_min, value_max = AirSimTrainingSession._auto_normalize_series(
        np.array([np.nan, 5.0, 5.0, np.nan], dtype=float)
    )

    assert np.isnan(normalized[0])
    assert normalized[1] == 0.0
    assert normalized[2] == 0.0
    assert np.isnan(normalized[3])
    assert value_min == 5.0
    assert value_max == 5.0


def test_airsim_training_session_records_the_rollout_device(tmp_path):
    """
    The session's metadata and iteration log name the rollout device next to the device.
    """

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    session = AirSimTrainingSession(env_name="AirSimEnv", agent_type="PPO", device="cuda", src_dir=src_dir,
                                    model_dir=tmp_path / "models", rollout_device="cpu")
    session.iteration_logger.log_iteration(iteration=0, mean_reward=0.0, mean_advantage=0.0, loss=0.0,
                                           num_episode_batches=1, elapsed_seconds=1.0)
    session.finalize(training_metrics=[{"loss": 0.0, "advantage": 0.0, "reward": 0.0}], interrupted=False,
                     skipped_short_rollouts=0, last_completed_iteration=0, best_mean_reward=0.0,
                     stop_rule="1 policy update")

    metadata = json.loads(session.metadata_path.read_text(encoding="utf-8"))
    assert metadata["config"]["device"] == "cuda" and metadata["config"]["rollout_device"] == "cpu"
    assert session.iteration_logger.path.read_text(encoding="utf-8").splitlines()[0].endswith(",rss_mb,rollout_device")
