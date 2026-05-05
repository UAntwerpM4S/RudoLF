"""Session-scoped logging and plotting helpers for AirSim training runs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fw.common.airsim.tools.telemetry_exporter import TelemetryExporter
from fw.common.tools.path_utils import resolve_path_relative_to_src
from fw.common.tools.training_iteration_logger import TrainingIterationLogger


def _json_safe(value: Any) -> Any:
    """Convert nested training payloads into JSON-safe structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


class AirSimTrainingSession:
    """Own the files emitted for one AirSim PPO training session."""

    _RUN_LOG_FIELDNAMES = (
        "timestamp_utc",
        "event_type",
        "env_slot",
        "env_pid",
        "episode",
        "episode_step",
        "outcome",
        "start_checkpoint_index",
        "checkpoint_index",
        "route_progress_m",
        "route_progress_ratio",
        "cross_track_error",
        "ship_x",
        "ship_y",
        "heading_deg",
        "speed",
        "reward",
        "raw_reward",
        "terminated",
        "truncated",
        "used_startup_anchor",
        "start_x",
        "start_y",
        "start_heading_deg",
        "airsim_x",
        "airsim_y",
        "airsim_z",
        "airsim_yaw_deg",
        "details_json",
    )

    def __init__(
        self,
        *,
        env_name: str,
        agent_type: str,
        device: str,
        src_dir: Path,
        model_dir: str | Path | None = None,
        log_dir: str = "logs/airsim_training_sessions",
    ) -> None:
        """Create a timestamped AirSim training session directory."""
        self._run_started = datetime.now(timezone.utc)
        timestamp = self._run_started.strftime("%Y%m%d_%H%M%S")
        root_dir = resolve_path_relative_to_src(log_dir, src_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        self.session_dir = root_dir / f"{env_name}_{agent_type}_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.run_log_path = self.session_dir / "run_log.csv"
        self.metadata_path = self.session_dir / "metadata.json"
        self.training_metrics_csv_path = self.session_dir / "training_metrics.csv"
        self.route_plot_path = self.session_dir / "route_plot.png"
        self.training_metrics_plot_path = self.session_dir / "training_metrics.png"

        self.iteration_logger = TrainingIterationLogger(
            env_name=env_name,
            agent_type=agent_type,
            device=device,
            src_dir=src_dir,
            log_dir=str(self.session_dir),
        )

        self._run_log_exporter = TelemetryExporter(
            session_type="training_run",
            model_path=str(model_dir or ""),
            environment_name=env_name,
            config_dict={
                "agent_type": agent_type,
                "device": device,
            },
            export_dir=self.session_dir,
            flush_interval=25,
            fieldnames=self._RUN_LOG_FIELDNAMES,
            csv_filename=self.run_log_path.name,
            metadata_filename=self.metadata_path.name,
        )

        self._run_events: list[dict[str, Any]] = []
        self._route_snapshot: dict[str, Any] | None = None

        self.update_metadata(
            session_directory=str(self.session_dir),
            iteration_log=str(self.iteration_logger.path),
            run_log=str(self.run_log_path),
            metadata_file=str(self.metadata_path),
            training_metrics_csv=str(self.training_metrics_csv_path),
            route_plot=str(self.route_plot_path),
            training_metrics_plot=str(self.training_metrics_plot_path),
        )

    def update_metadata(self, **metadata: Any) -> None:
        """Merge extra metadata into the session sidecar JSON."""
        self._run_log_exporter.update_metadata(metadata)

    def set_route_snapshot(self, snapshot: dict[str, Any] | None) -> None:
        """Store the latest route geometry snapshot for plotting."""
        if not snapshot:
            return
        self._route_snapshot = _json_safe(snapshot)

        route_points = np.asarray(snapshot.get("route_points", []), dtype=float)
        checkpoints = list(snapshot.get("checkpoints", []))
        summary = {
            "route_point_count": int(len(route_points)),
            "checkpoint_count": int(len(checkpoints)),
            "initial_ship_pos": _json_safe(snapshot.get("initial_ship_pos")),
            "target_pos": _json_safe(snapshot.get("target_pos")),
            "captured_startup_pose_airsim": _json_safe(
                snapshot.get("captured_startup_pose_airsim")
            ),
            "captured_startup_framework_heading_deg": _json_safe(
                snapshot.get("captured_startup_framework_heading_deg")
            ),
            "reset_mode": snapshot.get("reset_mode"),
            "save_slot": snapshot.get("save_slot"),
            "load_world_on_start": snapshot.get("load_world_on_start"),
            "capture_live_start_pose": snapshot.get("capture_live_start_pose"),
            "no_rendering_mode": snapshot.get("no_rendering_mode"),
            "simulation_speed": snapshot.get("simulation_speed"),
        }
        self.update_metadata(route_summary=summary)

    def log_iteration(self, **kwargs: Any) -> None:
        """Append one PPO iteration row to the iteration log."""
        self.iteration_logger.log_iteration(**kwargs)

    def log_run_events(self, events: list[dict[str, Any]] | None) -> None:
        """Append a batch of environment run events to the run log."""
        for event in events or []:
            normalized = dict(_json_safe(event))
            self._run_events.append(normalized)
            self._run_log_exporter.log_step(self._normalize_run_event(normalized))

    def finalize(
        self,
        *,
        training_metrics: list[dict[str, Any]],
        interrupted: bool,
        skipped_short_rollouts: int,
        last_completed_iteration: int,
        best_mean_reward: float,
        stop_rule: str,
    ) -> None:
        """Flush logs, save plots, and finalize session metadata."""
        self.iteration_logger.save()
        self._write_training_metrics_csv(training_metrics)
        self._save_training_metrics_plot(training_metrics)
        self._save_route_plot()

        self.update_metadata(
            interrupted=bool(interrupted),
            skipped_short_rollouts=int(skipped_short_rollouts),
            last_completed_iteration=int(last_completed_iteration),
            best_mean_reward=(
                None if best_mean_reward == float("-inf") else float(best_mean_reward)
            ),
            stop_rule=str(stop_rule),
            run_event_count=int(len(self._run_events)),
        )

        self._run_log_exporter.save()
        self._run_log_exporter.close()

    def _normalize_run_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Map one in-memory event payload onto the CSV schema."""
        row = {field: "" for field in self._RUN_LOG_FIELDNAMES}
        extras: dict[str, Any] = {}

        for key, value in event.items():
            if key in row and key != "details_json":
                row[key] = value
            else:
                extras[key] = value

        row["details_json"] = json.dumps(extras, sort_keys=True) if extras else ""
        return row

    def _write_training_metrics_csv(self, training_metrics: list[dict[str, Any]]) -> None:
        """Write the compact training metrics CSV alongside the plots."""
        if not training_metrics:
            return

        fieldnames = list(training_metrics[0].keys())
        with self.training_metrics_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in training_metrics:
                writer.writerow(_json_safe(row))

    @staticmethod
    def _auto_normalize_series(values: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Map one metric series to ``[-1, 1]`` while preserving NaNs."""
        series = np.asarray(values, dtype=float)
        normalized = np.full(series.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(series)
        if not np.any(finite_mask):
            return normalized, float("nan"), float("nan")

        finite_values = series[finite_mask]
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        span = float(value_max - value_min)

        if span <= 1e-12:
            normalized[finite_mask] = 0.0
        else:
            normalized[finite_mask] = (
                ((series[finite_mask] - value_min) / span) * 2.0
            ) - 1.0
        return np.clip(normalized, -1.0, 1.0), value_min, value_max

    @staticmethod
    def _rolling_average(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(array_indices, rolling_means)`` for valid window positions.

        The returned indices map directly into the original array so the caller
        can pair them with the matching x-coordinates.
        """
        if window < 2 or len(arr) < window:
            return np.array([], dtype=float), np.array([], dtype=float)
        rolled = np.array(
            [np.nanmean(arr[i : i + window]) for i in range(len(arr) - window + 1)],
            dtype=float,
        )
        indices = np.arange(window - 1, len(arr), dtype=float)
        return indices, rolled

    @staticmethod
    def _load_max_checkpoint_per_episode(
        run_log_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read ``run_log.csv`` and return per-episode max checkpoint index arrays."""
        if not run_log_path.exists():
            return np.array([], dtype=float), np.array([], dtype=float)
        episode_max: dict[int, int] = {}
        try:
            with run_log_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ep = int(row["episode"])
                        cp = int(row["checkpoint_index"])
                        if ep > 0:
                            episode_max[ep] = max(episode_max.get(ep, 0), cp)
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            return np.array([], dtype=float), np.array([], dtype=float)
        if not episode_max:
            return np.array([], dtype=float), np.array([], dtype=float)
        episodes = np.array(sorted(episode_max), dtype=float)
        checkpoints = np.array([episode_max[int(ep)] for ep in episodes], dtype=float)
        return episodes, checkpoints

    def _save_training_metrics_plot(self, training_metrics: list[dict[str, Any]]) -> None:
        """Render reward, loss, advantage, and per-episode checkpoint curves to disk."""
        if not training_metrics:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _ROLLING_WINDOW = 5

        iterations = np.arange(len(training_metrics), dtype=float)
        rewards = np.asarray([row.get("reward", np.nan) for row in training_metrics], dtype=float)
        losses = np.asarray([row.get("loss", np.nan) for row in training_metrics], dtype=float)
        advantages = np.asarray(
            [row.get("advantage", np.nan) for row in training_metrics],
            dtype=float,
        )

        normalized_reward, reward_min, reward_max = self._auto_normalize_series(rewards)
        normalized_loss, loss_min, loss_max = self._auto_normalize_series(losses)
        normalized_advantage, advantage_min, advantage_max = self._auto_normalize_series(
            advantages
        )

        episode_x, episode_max_cp = self._load_max_checkpoint_per_episode(self.run_log_path)
        has_cp_data = len(episode_x) > 0

        if has_cp_data:
            fig, all_axes = plt.subplots(4, 1, figsize=(12, 14))
            axes = all_axes[:3]
            for ax in axes[1:]:
                ax.sharex(axes[0])
            ax_cp = all_axes[3]
        else:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            ax_cp = None

        plots = (
            ("Reward", normalized_reward, reward_min, reward_max, "#1f77b4", 1.8),
            ("Loss", normalized_loss, loss_min, loss_max, "#d62728", 1.5),
            ("Advantage", normalized_advantage, advantage_min, advantage_max, "#2ca02c", 1.5),
        )

        for axis, (name, values, value_min, value_max, color, line_width) in zip(
            axes,
            plots,
            strict=False,
        ):
            axis.plot(iterations, values, color=color, linewidth=line_width, alpha=0.45)
            roll_idx, roll_y = self._rolling_average(values, _ROLLING_WINDOW)
            if len(roll_idx) > 0:
                axis.plot(
                    iterations[roll_idx.astype(int)],
                    roll_y,
                    color=color,
                    linewidth=line_width + 0.3,
                    linestyle="--",
                    label=f"avg-{_ROLLING_WINDOW}",
                )
            axis.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.55)
            axis.set_ylabel(f"{name} (norm)")
            axis.set_ylim(-1.05, 1.05)
            if np.isfinite(value_min) and np.isfinite(value_max):
                axis.text(
                    0.01,
                    0.93,
                    f"raw range: [{value_min:.4f}, {value_max:.4f}]",
                    transform=axis.transAxes,
                    fontsize=9,
                    va="top",
                )
            axis.grid(True, alpha=0.3)

        axes[0].set_title("AirSim Training Metrics (Auto-Normalized to [-1, 1])")

        if has_cp_data and ax_cp is not None:
            ax_cp.plot(episode_x, episode_max_cp, color="#ff7f0e", linewidth=1.2, alpha=0.45)
            roll_idx_ep, roll_y_ep = self._rolling_average(episode_max_cp, _ROLLING_WINDOW)
            if len(roll_idx_ep) > 0:
                ax_cp.plot(
                    episode_x[roll_idx_ep.astype(int)],
                    roll_y_ep,
                    color="#ff7f0e",
                    linewidth=1.6,
                    linestyle="--",
                )
            ax_cp.set_ylabel("Max Checkpoint")
            ax_cp.set_xlabel("Episode")
            ax_cp.set_ylim(bottom=0)
            ax_cp.grid(True, alpha=0.3)
        else:
            axes[2].set_xlabel("Iteration")

        fig.tight_layout()
        fig.savefig(self.training_metrics_plot_path, bbox_inches="tight", dpi=180)
        plt.close(fig)

    def _save_route_plot(self) -> None:
        """Render a static route/debug plot for the current training session."""
        if not self._route_snapshot:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        snapshot = self._route_snapshot
        route_points = np.asarray(snapshot.get("route_points", []), dtype=float)
        overall = np.asarray(snapshot.get("overall", []), dtype=float)
        obstacles = np.asarray(snapshot.get("obstacles", []), dtype=float)
        target_pos = np.asarray(snapshot.get("target_pos", []), dtype=float)

        fig, ax = plt.subplots(figsize=(12, 8))

        if overall.ndim == 2 and len(overall) >= 3:
            ax.add_patch(
                patches.Polygon(
                    overall,
                    closed=True,
                    edgecolor="saddlebrown",
                    facecolor="none",
                    linewidth=1.5,
                    label="Overall Area",
                )
            )

        if obstacles.ndim == 2 and len(obstacles) >= 3:
            ax.add_patch(
                patches.Polygon(
                    obstacles,
                    closed=True,
                    edgecolor="firebrick",
                    facecolor="none",
                    linewidth=1.5,
                    label="Waterway",
                )
            )

        if route_points.ndim == 2 and len(route_points) >= 2:
            ax.plot(
                route_points[:, 0],
                route_points[:, 1],
                linestyle="--",
                color="forestgreen",
                linewidth=1.5,
                label="Route",
            )

        for checkpoint in snapshot.get("checkpoints", []):
            checkpoint_x = float(checkpoint.get("x", 0.0))
            checkpoint_y = float(checkpoint.get("y", 0.0))
            checkpoint_radius = float(checkpoint.get("radius", 0.0))
            ax.add_patch(
                plt.Circle(
                    (checkpoint_x, checkpoint_y),
                    checkpoint_radius,
                    edgecolor="gray",
                    facecolor="none",
                    alpha=0.25,
                    linewidth=0.8,
                )
            )

        respawns = [
            event for event in self._run_events
            if str(event.get("event_type", "")) == "respawn"
        ]
        if respawns:
            respawn_xy = np.asarray(
                [[float(event.get("ship_x", 0.0)), float(event.get("ship_y", 0.0))] for event in respawns],
                dtype=float,
            )
            ax.scatter(
                respawn_xy[:, 0],
                respawn_xy[:, 1],
                c="royalblue",
                s=45,
                alpha=0.8,
                label="Respawns",
            )

        checkpoint_hits = [
            event for event in self._run_events
            if str(event.get("event_type", "")) == "checkpoint_passed"
        ]
        if checkpoint_hits:
            checkpoint_xy = np.asarray(
                [[float(event.get("ship_x", 0.0)), float(event.get("ship_y", 0.0))] for event in checkpoint_hits],
                dtype=float,
            )
            ax.scatter(
                checkpoint_xy[:, 0],
                checkpoint_xy[:, 1],
                c="darkorange",
                s=18,
                alpha=0.45,
                label="Checkpoint Hits",
            )

        if target_pos.shape == (2,):
            ax.scatter(
                [target_pos[0]],
                [target_pos[1]],
                c="crimson",
                s=90,
                marker="X",
                label="Target",
            )

        ax.set_title("AirSim Training Route Overview")
        ax.set_xlabel("Framework X")
        ax.set_ylabel("Framework Y")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(self.route_plot_path, bbox_inches="tight", dpi=180)
        plt.close(fig)
