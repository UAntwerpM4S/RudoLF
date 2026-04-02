"""Run inference for PPO policies trained directly in ``AirSimEnv``.

This runner is intentionally single-purpose:

- it loads a trained ``AirSimEnv`` policy
- it drives the live UE5/CosysAirSim ship through ``AirSimEnv.step()``
- it relies on ``AirSimEnv.reset()`` and ``AirSimEnv.close()`` for
  episode lifecycle handling

Framework-driven replay for modular framework-model visualisation lives in
``run_visualise_models.py`` and is kept separate on purpose.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from airsim.environments.airsim_env import AirSimEnv  # noqa: E402
from fw.common.airsim.tools.airsim_export import (  # noqa: E402
    AirSimSessionExporter,
    build_airsim_env_snapshot,
)
from fw.common.airsim.tools.trajectory_analysis import (  # noqa: E402
    build_run_summary_from_csv,
    format_performance_summary,
    render_side_by_side_trajectory_plot,
)
from fw.config import AIRSIM_COMPARISON_TRAJECTORY_PLOT_FILENAME  # noqa: E402
from fw.common.airsim.tools.path_utils import resolve_model_path  # noqa: E402
from fw.policies.ppo_policy import PPOPolicy  # noqa: E402


logger = logging.getLogger(__name__)

_AIRSIM_MODEL_HINTS: Tuple[str, ...] = ("airsimenv", "airsim_env", "ue5")


def _warn_if_model_name_looks_non_airsim(model_path: Path) -> None:
    """Warn when a model name does not look like an AirSim-trained policy.

    Args:
        model_path: Resolved model checkpoint path.
    """
    stem = model_path.stem.lower()
    if any(hint in stem for hint in _AIRSIM_MODEL_HINTS):
        return
    logger.warning(
        "Model '%s' does not look like an AirSimEnv-trained policy. "
        "run_inference.py now targets UE5/AirSimEnv inference only; "
        "use run_visualise_models.py for framework-model replay.",
        model_path.name,
    )


def load_policy(model_path: str, device: str = "cpu") -> PPOPolicy:
    """Load a :class:`PPOPolicy` from a saved ``.zip`` file.

    Args:
        model_path: User-provided checkpoint path.
        device: Torch device selector used during model load.

    Returns:
        PPOPolicy: Loaded policy with its network in evaluation mode.
    """
    path = resolve_model_path(model_path, _SRC_DIR)
    stem = str(path.with_suffix("")) if path.suffix == ".zip" else str(path)
    policy = PPOPolicy.load(stem, device=device)
    policy.network.eval()
    logger.info("Policy loaded from '%s' (device=%s).", path.name, device)
    return policy


def _build_inference_row(
    env: AirSimEnv,
    episode_index: int,
    step_index: int,
    reward: float,
    action,
    outcome: str,
) -> Dict[str, float | int | str]:
    """Build one inference telemetry row for CSV export.

    Args:
        env: Active AirSim environment.
        episode_index: One-based episode index.
        step_index: One-based step index within the episode.
        reward: Reward produced by the current step.
        action: Policy action applied on this step.
        outcome: Outcome label for the current step.

    Returns:
        Dict[str, float | int | str]: Flat telemetry row ready for CSV export.
    """
    snapshot = build_airsim_env_snapshot(env)
    return {
        "step": int(step_index),
        "sim_time": float(step_index * env.time_step),
        "episode": int(episode_index),
        "episode_step": int(step_index),
        "x": float(snapshot["ship_x"]),
        "y": float(snapshot["ship_y"]),
        "pos_x": float(snapshot["ship_x"]),
        "pos_y": float(snapshot["ship_y"]),
        "heading_deg": float(snapshot["heading_deg"]),
        "speed": float(snapshot["speed"]),
        "reward": float(reward),
        "steering": float(action[0]),
        "throttle": float(action[1]),
        "brake": float(env.last_controls.get("brake", 0.0)),
        "checkpoint_index": int(snapshot["checkpoint_index"]),
        "cross_track_error": float(snapshot["cross_track_error"]),
        "physics_wait_s": float(snapshot["physics_wait_s"]),
        "step_latency_s": float(snapshot["step_latency_s"]),
        "ue5_timestamp": int(snapshot["ue5_timestamp"]),
        "outcome": str(outcome),
        "model_origin": "airsim_env",
        "control_mode": "airsim_env_step",
    }


def _run_episode_airsim_env(
    env: AirSimEnv,
    policy: PPOPolicy,
    max_steps: int,
    episode_index: int = 1,
    exporter: Optional[AirSimSessionExporter] = None,
    verbose: bool = False,
) -> Tuple[List[float], List[float], float, str]:
    """Run one inference episode through ``AirSimEnv.step()``.

    This mirrors the trained control loop:

    1. ``env.reset()`` restores the ship to the episode start pose
    2. the policy predicts an action from the current observation
    3. ``env.step()`` sends the action to UE5, waits for physics, reads back
       the new state, and computes reward/termination/truncation
    4. inference stops when the episode terminates or truncates

    Args:
        env: Active AirSim environment.
        policy: Loaded PPO policy used for action selection.
        max_steps: Maximum steps allowed for the episode.
        episode_index: One-based episode index used for export metadata.
        exporter: Optional session exporter receiving per-step rows.
        verbose: Whether to emit periodic per-step debug logs.

    Returns:
        Tuple[List[float], List[float], float, str]: Trajectory x values,
        trajectory y values, total reward, and final outcome label.
    """
    obs, _ = env.reset()
    traj_x: List[float] = [float(env.ship_pos[0])]
    traj_y: List[float] = [float(env.ship_pos[1])]
    total_reward = 0.0
    step = 0
    final_outcome = "running"

    while step < max_steps:
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(policy.device)
        with torch.no_grad():
            action_t, _, _ = policy._predict(state_t, deterministic=True)
        action = action_t.detach().cpu().numpy().flatten()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step += 1

        traj_x.append(float(env.ship_pos[0]))
        traj_y.append(float(env.ship_pos[1]))

        if terminated:
            final_outcome = "terminated"
        elif truncated:
            final_outcome = "truncated"
        else:
            final_outcome = "running"

        if exporter is not None:
            exporter.log_inference_step(
                _build_inference_row(
                    env=env,
                    episode_index=episode_index,
                    step_index=step,
                    reward=float(reward),
                    action=action,
                    outcome=final_outcome,
                )
            )

        if verbose and step % 200 == 0:
            logger.info(
                "step=%d pos=(%.0f, %.0f) reward=%+.2f cumulative=%+.1f "
                "checkpoint=%d cte=%.2f",
                step,
                env.ship_pos[0],
                env.ship_pos[1],
                reward,
                total_reward,
                int(info.get("checkpoint_index", env.checkpoint_index)),
                float(info.get("cross_track_error", env.cross_error)),
            )

        if terminated or truncated:
            break

    return traj_x, traj_y, total_reward, final_outcome


def run_inference(
    model_path: str,
    ship_pos: Optional[List[float]] = None,
    target_pos: Optional[List[float]] = None,
    ip: str = "",
    port: int = 41451,
    max_steps: int = 14000,
    num_episodes: int = 3,
    device: str = "cpu",
    verbose: bool = False,
) -> None:
    """Run live inference in UE5 using the native ``AirSimEnv`` loop.

    Args:
        model_path: Path to the AirSimEnv-trained checkpoint.
        ship_pos: Optional start position override in Fossen coordinates.
        target_pos: Optional target position override in Fossen coordinates.
        ip: AirSim RPC host.
        port: AirSim RPC port.
        max_steps: Maximum steps per episode.
        num_episodes: Number of episodes to run.
        device: Torch device selector used for inference.
        verbose: Whether to emit detailed runtime logs.
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    banner = "=" * 62
    resolved_model_path = resolve_model_path(model_path, _SRC_DIR)
    _warn_if_model_name_looks_non_airsim(resolved_model_path)

    print(banner)
    print("  AIRSIMENV INFERENCE MODE")
    print(f"  Model  : {resolved_model_path.name}")
    print("  Loop   : policy -> AirSimEnv.step() -> UE5 physics -> state feedback")
    print("  Reset  : AirSimEnv.reset() per episode")
    print("  Close  : AirSimEnv.close() on shutdown")
    print("  Replay : use run_visualise_models.py for modular framework-model replay")
    print(banner)

    policy = load_policy(str(resolved_model_path), device=device)
    with AirSimSessionExporter(
        session_name=f"inference_{resolved_model_path.stem}",
        metadata={
            "script": "airsim.inference.run_inference",
            "model_path": str(resolved_model_path),
            "model_name": resolved_model_path.name,
            "environment_name": "AirSimEnv",
            "model_origin": "airsim_env",
            "configuration": {
                "ship_pos": ship_pos,
                "target_pos": target_pos,
                "ip": ip or "127.0.0.1",
                "port": port,
                "max_steps": max_steps,
                "num_episodes": num_episodes,
                "device": device,
                "simulation_speed": 1.0,
                "episode_loop": "AirSimEnv.reset -> policy -> AirSimEnv.step",
            },
        },
    ) as session_exporter:
        logger.info("Connecting to UE5 and loading world save 'simulator_1'...")
        env: Optional[AirSimEnv] = None
        comparison_plot_path = session_exporter.export_dir / AIRSIM_COMPARISON_TRAJECTORY_PLOT_FILENAME
        env = AirSimEnv(
            ship_pos=ship_pos,
            target_pos=target_pos,
            ip=ip,
            port=port,
            max_steps=max_steps,
            training_mode=False,
            use_pause_stepping=False,
            simulation_speed=1.0,
            csv_log_path=str(session_exporter.bridge_log_path),
        )
        session_exporter.update_metadata(
            {
                "resolved_model_path": str(resolved_model_path),
                "export_directory": str(session_exporter.export_dir),
                "bridge_telemetry_path": str(session_exporter.bridge_log_path),
                "control_loop": "AirSimEnv.step",
                "episode_reset": "AirSimEnv.reset",
            }
        )

        logger.info(
            "AirSimBridge connected=%s | mode=AIRSIM_ENV | episodes=%d",
            env.bridge.is_connected,
            num_episodes,
        )

        try:
            for ep in range(num_episodes):
                print(f"\nEpisode {ep + 1}/{num_episodes} [AirSimEnv]")
                traj_x, traj_y, total_reward, outcome = _run_episode_airsim_env(
                    env,
                    policy,
                    max_steps,
                    episode_index=ep + 1,
                    exporter=session_exporter,
                    verbose=verbose,
                )

                steps_run = len(traj_x) - 1
                print(
                    f"   Finished: {steps_run} steps | outcome={outcome} | "
                    f"reward={total_reward:+.1f}"
                )

                session_exporter.update_metadata(
                    {
                        "last_episode_index": ep + 1,
                        "last_outcome": outcome,
                        "last_episode_steps": steps_run,
                        "last_episode_reward": float(total_reward),
                        "last_ship_position": {
                            "x": float(traj_x[-1]) if traj_x else None,
                            "y": float(traj_y[-1]) if traj_y else None,
                        },
                    }
                )
        finally:
            if env is not None:
                env.close()
            print("\nInference complete; AirSim connection closed.")

        # Persist step rows first so summary/plot code can reuse telemetry CSVs.
        session_exporter.save()
        session_csv_path = Path(
            getattr(
                session_exporter,
                "telemetry_path",
                session_exporter.export_dir / "inference_session.csv",
            )
        )
        bridge_csv_path = Path(
            getattr(
                session_exporter,
                "bridge_log_path",
                session_exporter.export_dir / "airsim_telemetry.csv",
            )
        )

        summary = build_run_summary_from_csv(
            session_csv_path=session_csv_path,
            bridge_csv_path=bridge_csv_path,
        )
        session_exporter.update_metadata(
            {
                "performance_summary": summary.to_metadata_dict(),
            }
        )
        session_exporter.save()

        print()
        print(format_performance_summary("AIRSIM INFERENCE SUMMARY", summary))

        comparison_plot = render_side_by_side_trajectory_plot(
            session_csv_path=session_csv_path,
            bridge_csv_path=bridge_csv_path,
            output_path=comparison_plot_path,
            title=f"2D vs 3D Trajectory - {resolved_model_path.name}",
        )
        if comparison_plot is not None:
            print(f"2D vs 3D trajectory plot saved to '{comparison_plot}'.")
            session_exporter.update_metadata(
                {
                    "comparison_plot": comparison_plot.name,
                }
            )
            session_exporter.save()
        else:
            print("2D vs 3D trajectory plot skipped (missing telemetry or matplotlib).")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for AirSimEnv inference runs.

    Returns:
        argparse.ArgumentParser: Configured parser for AirSimEnv inference.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run inference for PPO models trained directly in AirSimEnv. "
            "For modular framework-model replay, use run_visualise_models instead."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the .zip model file.",
    )
    parser.add_argument(
        "--ip",
        default="",
        help="AirSim server IP address (empty string = 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=41451,
        help="AirSim RPC port.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of inference episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=14000,
        dest="max_steps",
        help="Maximum steps allowed per episode.",
    )
    parser.add_argument(
        "--ship-pos",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        dest="ship_pos",
        help="Start position in Fossen coords (east north).",
    )
    parser.add_argument(
        "--target-pos",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        dest="target_pos",
        help="Target position in Fossen coords (east north).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device for policy inference.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable detailed AirSim inference logs.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = _build_parser().parse_args()
    args.model = str(resolve_model_path(args.model, _SRC_DIR))

    run_inference(
        model_path=args.model,
        ship_pos=args.ship_pos,
        target_pos=args.target_pos,
        ip=args.ip,
        port=args.port,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        device=args.device,
        verbose=args.verbose,
    )
