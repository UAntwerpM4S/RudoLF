"""Train or visualize the UE5 AirSim environment using runtime settings.

Usage:
    python -m airsim.train.train_airsim

This entrypoint now reads its AirSim environment configuration from
``airsim/config/runtime_settings.json`` so standalone AirSim training stays in
sync with ``fw.main_with_airsim``.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any

from airsim.config.airsim_config import get_runtime_settings, merge_runtime_settings


_TRAINING_SPEED_PROFILES = {
    "normal": {
        "simulation_speed": 1.0,
        "use_pause_stepping": False,
        "frames_per_step": 1,
        "time_step": 0.1,
        "physics_update_timeout": 1.0,
        "state_poll_interval": 0.01,
    },
    "safe": {
        "simulation_speed": 20.0,
        "use_pause_stepping": True,
        "use_sim_time_stepping": True,
        "sim_time_per_action": 0.05,
        "frames_per_step": 8,
        "time_step": 0.05,
        "physics_update_timeout": 1.2,
        "state_poll_interval": 0.003,
    },
    "fast": {
        "simulation_speed": 40.0,
        "use_pause_stepping": True,
        "use_sim_time_stepping": True,
        "sim_time_per_action": 0.1,
        "frames_per_step": 10,
        "time_step": 0.1,
        "physics_update_timeout": 1.0,
        "state_poll_interval": 0.003,
    },
    "turbo": {
        "simulation_speed": 80.0,
        "use_pause_stepping": True,
        "use_sim_time_stepping": True,
        "sim_time_per_action": 0.1,
        "frames_per_step": 12,
        "time_step": 0.1,
        "physics_update_timeout": 1.2,
        "state_poll_interval": 0.002,
    },
}


class _AsyncioWin995Filter(logging.Filter):
    """Suppress the noisy Windows asyncio self-pipe shutdown traceback."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for the known benign WinError 995 shutdown record."""
        exc = record.exc_info[1] if record.exc_info else None
        message = record.getMessage().lower()
        if (
            "event loop self pipe" in message
            and (
                (
                    isinstance(exc, ConnectionResetError)
                    and getattr(exc, "winerror", None) == 995
                )
                or "winerror 995" in message
                or "connectionreseterror" in message
            )
        ):
            return False
        return True


def _resolve_repo_relative_path(value: str, repo_root: Path) -> str:
    """Resolve a runtime-settings path relative to the repository root."""
    if not value:
        return ""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((repo_root / candidate).resolve())


def _emit_training_log_summary(
    episode_log_path: Path,
    route_csv: Path,
) -> None:
    """Print a concise summary of the existing AirSim training log."""
    from airsim.train.training_log_analysis import analyze_training_log

    route_total_checkpoints = max(
        1,
        sum(1 for _ in route_csv.open("r", encoding="utf-8")) - 1,
    )
    training_log = analyze_training_log(
        episode_log_path,
        route_total_checkpoints=route_total_checkpoints,
    )

    if training_log.episode_count <= 0:
        return

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


def _resolve_effective_training_settings(
    runtime_settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return merged runtime settings plus the effective AirSim training config."""
    runtime = merge_runtime_settings(
        runtime_settings if runtime_settings is not None else get_runtime_settings()
    )
    training_cfg = dict(runtime.get("training", {}))
    airsim_cfg = dict(runtime.get("airsim", {}))

    profile_name = str(training_cfg.get("speed_profile", "custom")).strip().lower()
    effective_training_cfg = dict(training_cfg)
    if profile_name in _TRAINING_SPEED_PROFILES:
        effective_training_cfg.update(_TRAINING_SPEED_PROFILES[profile_name])

    if bool(effective_training_cfg.get("use_pause_stepping", False)) and bool(
        effective_training_cfg.get("use_sim_time_stepping", False)
    ):
        effective_training_cfg["time_step"] = float(
            effective_training_cfg.get(
                "sim_time_per_action",
                effective_training_cfg.get("time_step", 0.1),
            )
        )

    return runtime, airsim_cfg, effective_training_cfg


def _build_training_env_config(runtime_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the standalone AirSim training config from runtime settings."""
    repo_root = Path(__file__).resolve().parents[2]
    _, airsim_cfg, training_cfg = _resolve_effective_training_settings(runtime_settings)

    episode_log_path = _resolve_repo_relative_path(
        str(training_cfg.get("episode_log_path", "logs/airsim_training_episode_log.csv")),
        repo_root,
    )
    route_csv = repo_root / "fw" / "environments" / "trajectory_points_no_scale.csv"
    if episode_log_path:
        try:
            _emit_training_log_summary(Path(episode_log_path), route_csv)
        except Exception as exc:
            print(f"[train_airsim] Skipping training log analysis: {exc}")

    env_config = dict(airsim_cfg)
    env_config.update(
        {
            "training_mode": True,
            "no_rendering_mode": bool(training_cfg.get("no_rendering_mode", True)),
            "use_pause_stepping": bool(training_cfg.get("use_pause_stepping", True)),
            "use_sim_time_stepping": bool(training_cfg.get("use_sim_time_stepping", False)),
            "sim_time_per_action": float(
                training_cfg.get(
                    "sim_time_per_action",
                    training_cfg.get("time_step", 0.1),
                )
            ),
            "frames_per_step": int(training_cfg.get("frames_per_step", 8)),
            "time_step": float(training_cfg.get("time_step", 0.1)),
            "physics_update_timeout": float(training_cfg.get("physics_update_timeout", 1.0)),
            "state_poll_interval": float(training_cfg.get("state_poll_interval", 0.005)),
            "draw_calibration_reference": bool(
                training_cfg.get("draw_calibration_reference", False)
            ),
            "simulation_speed": float(
                training_cfg.get(
                    "simulation_speed",
                    airsim_cfg.get("simulation_speed", 1.0),
                )
            ),
            "save_slot": str(airsim_cfg.get("save_slot", "simulator_1")),
            "load_world_on_start": bool(
                training_cfg.get("load_world_on_start", True)
            ),
            "capture_live_start_pose": bool(
                training_cfg.get("capture_live_start_pose", True)
            ),
            "episode_log_path": episode_log_path,
            "normalize_reward": bool(training_cfg.get("normalize_reward", True)),
            "reward_normalization_scale": float(
                training_cfg.get("reward_normalization_scale", 5.0)
            ),
            "reset_mode": str(training_cfg.get("reset_mode", "rewind_pose")),
            "progress_log_interval_steps": int(
                training_cfg.get("progress_log_interval_steps", 0)
            ),
            "log_pause_watchdog_warnings": bool(
                training_cfg.get("log_pause_watchdog_warnings", False)
            ),
            "max_steps": int(training_cfg.get("max_steps", 60000)),
            "curriculum_enabled": bool(training_cfg.get("curriculum_enabled", False)),
            "curriculum_window_size": int(training_cfg.get("curriculum_window_size", 8)),
            "curriculum_target_checkpoint": training_cfg.get(
                "curriculum_target_checkpoint"
            ),
            "curriculum_frontier_increment": int(
                training_cfg.get("curriculum_frontier_increment", 4)
            ),
            "curriculum_window_increment": int(
                training_cfg.get("curriculum_window_increment", 6)
            ),
            "curriculum_success_patience": int(
                training_cfg.get("curriculum_success_patience", 2)
            ),
            "curriculum_randomization_scale": float(
                training_cfg.get("curriculum_randomization_scale", 0.0)
            ),
            "guidance_assist_enabled": bool(
                training_cfg.get("guidance_assist_enabled", False)
            ),
            "guidance_residual_steering_scale": float(
                training_cfg.get("guidance_residual_steering_scale", 0.50)
            ),
            "guidance_residual_throttle_scale": float(
                training_cfg.get("guidance_residual_throttle_scale", 0.60)
            ),
            "steering_dynamics_enabled": bool(
                training_cfg.get("steering_dynamics_enabled", True)
            ),
            "steering_smoothing_alpha": float(
                training_cfg.get("steering_smoothing_alpha", 0.65)
            ),
            "max_steering_rate_per_second": float(
                training_cfg.get("max_steering_rate_per_second", 3.2)
            ),
            "control_interface": str(
                training_cfg.get("control_interface", "car_physics")
            ),
            "observation_profile": str(
                training_cfg.get("observation_profile", "ppo_route_relative_car")
            ),
        }
    )

    if "latency_warning_threshold" in training_cfg:
        env_config["latency_warning_threshold"] = float(
            training_cfg["latency_warning_threshold"]
    )

    return env_config


def _build_training_stop_condition(
    runtime_settings: dict[str, Any] | None = None,
):
    """Build the AirSim standalone stop condition."""
    _, _, training_cfg = _resolve_effective_training_settings(runtime_settings)
    max_policy_updates = int(training_cfg.get("max_policy_updates", 1500))

    from fw.stop_condition import Criterion, StopCondition

    return StopCondition(
        stop_criterion=Criterion.NBR_POLICY_UPDATES,
        max_time_steps=max_policy_updates,
    )


def _build_visualize_env_config(runtime_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a simple inference/visualization AirSim config from runtime settings."""
    runtime = merge_runtime_settings(
        runtime_settings if runtime_settings is not None else get_runtime_settings()
    )
    visualization_cfg = dict(runtime.get("visualization", {}))
    airsim_cfg = dict(runtime.get("airsim", {}))
    return {
        **airsim_cfg,
        "training_mode": False,
        "use_pause_stepping": False,
        "draw_calibration_reference": bool(
            visualization_cfg.get("draw_calibration_reference", True)
        ),
        "simulation_speed": 1.0,
        "max_steps": int(visualization_cfg.get("max_steps", 14000)),
        "load_world_on_start": True,
        "capture_live_start_pose": False,
    }


def _run_training_with_transfer(
    env,
    tl_cfg: dict[str, Any],
    stop_condition: Any,
    repo_root: Path,
) -> dict[str, Any]:
    """Create a SinglePPOAgent directly, inject a transfer policy, and run learn().

    Bypasses Trainer/Scheduler so the policy can be replaced before training
    starts — the Trainer creates the agent inside train() which is too late for
    weight injection.  This path is AirSim-only and has no effect on 2D envs.
    """
    import torch.optim as optim

    from fw.config import PPO_POLICY_NAME, get_hyperparameters
    from fw.policies.single_ppo_agent import SinglePPOAgent
    from airsim.train.transfer_learning import (
        TransferIncompatibilityError,
        build_transfer_policy,
    )

    hyperparams = get_hyperparameters(PPO_POLICY_NAME, env_name="AirSimEnv")
    agent = SinglePPOAgent(environment=env.gym_env, **hyperparams)

    source_path_raw = str(tl_cfg.get("source_policy_path", "")).strip()
    source_path = Path(source_path_raw)
    if not source_path.is_absolute():
        source_path = (repo_root / source_path).resolve()

    obs_dim = int(env.gym_env.observation_space.shape[0])
    act_dim = int(env.gym_env.action_space.shape[0])
    freeze = bool(tl_cfg.get("freeze_shared_layers", True))

    print("\n" + "=" * 60)
    print("  TRANSFER LEARNING — loading 2D policy weights")
    print("=" * 60)
    print(f"  Source : {source_path}")
    print(f"  Target : AirSimEnv  obs={obs_dim}  act={act_dim}  device={agent.device}")
    print(f"  Freeze : {freeze}")
    print("=" * 60 + "\n")

    try:
        transfer_policy = build_transfer_policy(
            source_zip_path=source_path,
            target_input_dim=obs_dim,
            target_output_dim=act_dim,
            target_device=agent.device,
            freeze_shared=freeze,
        )
        agent.policy = transfer_policy
        agent.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, transfer_policy.network.parameters()),
            lr=agent.learning_rate,
        )
    except (TransferIncompatibilityError, RuntimeError) as exc:
        print(f"[TransferLearning] Failed — training from scratch.\n  {exc}")

    agent.learn(stop_condition=stop_condition)
    return {}


def run_training(runtime_settings: dict[str, Any] | None = None):
    """Run AirSim PPO training from runtime settings and return trainer metrics."""
    from fw.config import PPO_POLICY_NAME
    from fw.env_factory import airsim_env_name, get as get_environment
    from fw.trainer import Trainer

    repo_root = Path(__file__).resolve().parents[2]
    env_config = _build_training_env_config(runtime_settings)
    stop_condition = _build_training_stop_condition(runtime_settings)

    env = get_environment(airsim_env_name, configuration=env_config)

    # Transfer learning: bypass Trainer and drive agent directly so the
    # transferred policy can be injected before learn() starts.
    runtime = merge_runtime_settings(
        runtime_settings if runtime_settings is not None else get_runtime_settings()
    )
    tl_cfg = dict(runtime.get("transfer_learning", {}))
    if bool(tl_cfg.get("enabled", False)):
        try:
            return _run_training_with_transfer(env, tl_cfg, stop_condition, repo_root)
        finally:
            try:
                env.gym_env.close()
            except Exception:
                pass

    # Standard path (no transfer learning)
    envs = [env]
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    print("\n" + "=" * 60)
    print("  STARTING AIRSIM RL TRAINING")
    print("=" * 60)
    print(f"  Environment: {airsim_env_name}")
    print(f"  Policy:      {PPO_POLICY_NAME}")
    print(f"  Stop rule:   {stop_condition}")
    print("  Source:      airsim/config/runtime_settings.json")
    print("=" * 60 + "\n")

    try:
        return trainer.train({airsim_env_name: stop_condition})
    finally:
        try:
            env.gym_env.close()
        except Exception:
            pass


def run_visualization(runtime_settings: dict[str, Any] | None = None):
    """Run AirSim visualization from runtime settings and return trainer instance."""
    from fw.config import PPO_POLICY_NAME
    from fw.env_factory import airsim_env_name, get as get_environment
    from fw.trainer import Trainer

    env_config = _build_visualize_env_config(runtime_settings)
    env = get_environment(airsim_env_name, configuration=env_config)
    envs = [env]
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    print("\n" + "=" * 60)
    print("  VISUALIZING TRAINED POLICY IN AIRSIM")
    print("=" * 60)
    print("  Source: airsim/config/runtime_settings.json\n")

    try:
        trainer.visualize(envs)
        return trainer
    finally:
        try:
            env.gym_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    verbose = str(os.environ.get("FW_AIRSIM_VERBOSE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    action = str(os.environ.get("FW_AIRSIM_ACTION", "train")).strip().lower()
    if action not in {"train", "visualize"}:
        action = "train"

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.getLogger("sim").setLevel(logging.INFO if verbose else logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("asyncio").addFilter(_AsyncioWin995Filter())
    multiprocessing.set_start_method("spawn", force=True)

    if action == "train":
        metrics = run_training()
        print(f"\nTraining metrics: {metrics}")
    else:
        run_visualization()
