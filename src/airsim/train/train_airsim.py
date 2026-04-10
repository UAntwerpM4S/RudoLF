"""Train an RL vessel policy inside UE5 via CosysAirSim.

Usage (from the ``src/`` directory):

    python -m airsim.train.train_airsim

Prerequisites:
    1. UE5 with CosysAirSim must be running and in Play mode.
    2. calibration.json must exist (run calibrate_coords.py once first).
"""

from __future__ import annotations

import logging
import multiprocessing

from pathlib import Path


def _build_training_env_config() -> dict:
    """Build a no-curriculum AirSim training configuration for the full route."""

    from airsim.train.training_log_analysis import analyze_training_log

    repo_root = Path(__file__).resolve().parents[2]
    episode_log_path = repo_root / "logs" / "airsim_training_episode_log.csv"
    route_csv = repo_root / "fw" / "environments" / "trajectory_points_no_scale.csv"

    route_total_checkpoints = max(1, sum(1 for _ in route_csv.open("r", encoding="utf-8")) - 1)
    training_log = analyze_training_log(
        episode_log_path,
        route_total_checkpoints=route_total_checkpoints,
    )

    if training_log.episode_count > 0:
        print("\n" + "=" * 60)
        print("  AIRSIM TRAINING LOG ANALYSIS")
        print("=" * 60)
        print(f"  Rows analysed:             {training_log.total_rows}")
        print(f"  Episodes analysed:         {training_log.episode_count}")
        print(f"  Median max checkpoint:     {training_log.median_max_checkpoint}")
        print(f"  Modal max checkpoint:      {training_log.modal_max_checkpoint}")
        print(f"  Best max checkpoint:       {training_log.best_max_checkpoint}")
        print(
            "  High-speed off-track rate: "
            f"{training_log.high_speed_offtrack_ratio:.1%}"
        )
        print("=" * 60 + "\n")

    return {
        "training_mode": True,
        "no_rendering_mode": True,
        "use_pause_stepping": True,
        "episode_log_path": str(episode_log_path),
        "progress_log_interval_steps": 200,
        "max_steps": 4000,
        "curriculum_enabled": True,
        "curriculum_window_size": 8,
        "curriculum_frontier_increment": 4,
        "curriculum_window_increment": 6,
        "curriculum_success_patience": 2,
        "guidance_assist_enabled": True,
        "guidance_residual_steering_scale": 0.50,
        "guidance_residual_throttle_scale": 0.60,
    }


if __name__ == "__main__":
    import os

    from fw.config import PPO_POLICY_NAME
    from fw.env_factory import airsim_env_name, get as get_environment
    from fw.stop_condition import Criterion, StopCondition
    from fw.trainer import Trainer

    verbose = str(os.environ.get("FW_AIRSIM_VERBOSE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.getLogger("sim").setLevel(logging.INFO if verbose else logging.ERROR)
    multiprocessing.set_start_method("spawn", force=True)

    action = "train"  # "train" or "visualize"
    env_config = _build_training_env_config() if action == "train" else {
        "training_mode": False,
    }

    env = get_environment(airsim_env_name, configuration=env_config)
    envs = [env]
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    if action == "train":
        configuration = {
            airsim_env_name: StopCondition(
                stop_criterion=Criterion.NBR_POLICY_UPDATES,
                max_time_steps=1500,
            ),
        }

        print("\n" + "=" * 60)
        print("  STARTING AIRSIM RL TRAINING")
        print("=" * 60)
        print(f"  Environment: {airsim_env_name}")
        print(f"  Policy:      {PPO_POLICY_NAME}")
        print("  Stop rule:   1500 policy updates")
        print("  Mode:        headless curriculum training")
        print("=" * 60 + "\n")

        metrics = trainer.train(configuration)
        print(f"\nTraining metrics: {metrics}")

    elif action == "visualize":
        print("\n" + "=" * 60)
        print("  VISUALIZING TRAINED POLICY IN AIRSIM")
        print("=" * 60 + "\n")
        trainer.visualize(envs)

    try:
        env.gym_env.close()
    except Exception:
        pass
