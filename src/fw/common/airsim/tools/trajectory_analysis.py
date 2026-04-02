"""Shared AirSim telemetry analysis and 2D-vs-3D plotting helpers.

This module focuses on post-run analysis by reusing exported telemetry CSV
files instead of collecting duplicate runtime data.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class EpisodePerformance:
    """Aggregated metrics for one telemetry episode.

    Attributes:
        episode: One-based episode index.
        steps: Number of telemetry rows for this episode.
        outcome: Final step outcome label.
        total_reward: Sum of step rewards for the episode.
        duration_s: Simulated episode duration in seconds.
        distance_2d_m: Cumulative travelled distance in the 2D frame.
        mean_cross_track_error_m: Mean absolute cross-track error in meters.
        max_cross_track_error_m: Maximum absolute cross-track error in meters.
    """

    episode: int
    steps: int
    outcome: str
    total_reward: float
    duration_s: float
    distance_2d_m: float
    mean_cross_track_error_m: float
    max_cross_track_error_m: float


@dataclass(frozen=True)
class RunPerformanceSummary:
    """Aggregated metrics for a full telemetry session.

    Attributes:
        episodes: Per-episode metrics in chronological order.
        total_reward: Sum of rewards across all episodes.
        total_duration_s: Sum of episode durations.
        total_distance_2d_m: Sum of cumulative 2D episode distances.
        total_distance_3d_m: Cumulative 3D distance from bridge telemetry.
        mean_cross_track_error_m: Mean absolute cross-track error over all rows.
        max_cross_track_error_m: Maximum absolute cross-track error over all rows.
        bridge_samples: Number of rows loaded from bridge telemetry.
    """

    episodes: List[EpisodePerformance]
    total_reward: float
    total_duration_s: float
    total_distance_2d_m: float
    total_distance_3d_m: float
    mean_cross_track_error_m: float
    max_cross_track_error_m: float
    bridge_samples: int

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert summary dataclass content to JSON-safe metadata.

        Returns:
            Dictionary representation suitable for metadata JSON sidecars.
        """
        data = asdict(self)
        data["episodes"] = [asdict(episode) for episode in self.episodes]
        return data


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with fallback.

    Args:
        value: Source value to parse.
        default: Value returned when conversion fails.

    Returns:
        Parsed float or the provided default.
    """
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int with fallback.

    Args:
        value: Source value to parse.
        default: Value returned when conversion fails.

    Returns:
        Parsed int or the provided default.
    """
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read all CSV rows into memory.

    Args:
        path: CSV file path.

    Returns:
        List of row dictionaries. Empty list if file does not exist.
    """
    if not path.exists() or not path.is_file():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _cumulative_distance_2d(rows: Iterable[Dict[str, Any]]) -> float:
    """Compute cumulative 2D distance from x/y columns.

    Args:
        rows: Telemetry rows containing ``x`` and ``y`` fields.

    Returns:
        Cumulative Euclidean distance in meters.
    """
    distance = 0.0
    previous_xy: Optional[tuple[float, float]] = None

    for row in rows:
        x = _safe_float(row.get("x", row.get("pos_x", 0.0)))
        y = _safe_float(row.get("y", row.get("pos_y", 0.0)))
        current_xy = (x, y)
        if previous_xy is not None:
            dx = current_xy[0] - previous_xy[0]
            dy = current_xy[1] - previous_xy[1]
            distance += math.hypot(dx, dy)
        previous_xy = current_xy

    return float(distance)


def _cumulative_distance_3d(rows: Iterable[Dict[str, Any]]) -> float:
    """Compute cumulative 3D distance from bridge ``pos_*`` columns.

    Args:
        rows: Bridge telemetry rows containing ``pos_x``, ``pos_y``, and
            ``pos_z`` fields.

    Returns:
        Cumulative 3D Euclidean distance in meters.
    """
    distance = 0.0
    previous_xyz: Optional[tuple[float, float, float]] = None

    for row in rows:
        x = _safe_float(row.get("pos_x", 0.0))
        y = _safe_float(row.get("pos_y", 0.0))
        z = _safe_float(row.get("pos_z", 0.0))
        current_xyz = (x, y, z)
        if previous_xyz is not None:
            dx = current_xyz[0] - previous_xyz[0]
            dy = current_xyz[1] - previous_xyz[1]
            dz = current_xyz[2] - previous_xyz[2]
            distance += math.sqrt(dx * dx + dy * dy + dz * dz)
        previous_xyz = current_xyz

    return float(distance)


def _group_rows_by_episode(rows: List[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    """Group telemetry rows by episode index.

    Args:
        rows: Session telemetry rows.

    Returns:
        Mapping from one-based episode number to rows.
    """
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        episode = _safe_int(row.get("episode", 1), default=1)
        grouped.setdefault(episode, []).append(row)
    return grouped


def _episode_duration_s(rows: List[Dict[str, str]]) -> float:
    """Estimate episode duration from telemetry rows.

    Args:
        rows: Telemetry rows for a single episode.

    Returns:
        Simulated duration in seconds.
    """
    if not rows:
        return 0.0
    sim_times = [_safe_float(row.get("sim_time", 0.0)) for row in rows]
    duration_s = max(sim_times)
    if duration_s > 0.0:
        return float(duration_s)

    # Fallback when sim_time is unavailable: approximate by the largest
    # recorded episode step assuming dt=1.0s.
    steps = [_safe_int(row.get("episode_step", 0), default=0) for row in rows]
    return float(max(steps))


def build_run_summary_from_csv(
    session_csv_path: str | Path,
    bridge_csv_path: Optional[str | Path] = None,
) -> RunPerformanceSummary:
    """Build a full run summary from exported telemetry CSV files.

    Args:
        session_csv_path: Session telemetry CSV path from the exporter.
        bridge_csv_path: Optional AirSim bridge telemetry CSV path.

    Returns:
        Aggregated run summary with per-episode and full-session metrics.
    """
    session_rows = _read_csv_rows(Path(session_csv_path))
    bridge_rows = _read_csv_rows(Path(bridge_csv_path)) if bridge_csv_path else []

    grouped = _group_rows_by_episode(session_rows)
    episode_metrics: List[EpisodePerformance] = []

    all_cte_abs: List[float] = []
    total_reward = 0.0
    total_duration_s = 0.0
    total_distance_2d_m = 0.0

    for episode in sorted(grouped.keys()):
        rows = grouped[episode]
        rewards = [_safe_float(row.get("reward", 0.0)) for row in rows]
        cte_abs = [abs(_safe_float(row.get("cross_track_error", 0.0))) for row in rows]
        all_cte_abs.extend(cte_abs)

        episode_reward = float(sum(rewards))
        episode_duration_s = _episode_duration_s(rows)
        episode_distance_2d_m = _cumulative_distance_2d(rows)
        episode_outcome = str(rows[-1].get("outcome", "unknown")) if rows else "unknown"

        metric = EpisodePerformance(
            episode=episode,
            steps=len(rows),
            outcome=episode_outcome,
            total_reward=episode_reward,
            duration_s=episode_duration_s,
            distance_2d_m=episode_distance_2d_m,
            mean_cross_track_error_m=(sum(cte_abs) / len(cte_abs)) if cte_abs else 0.0,
            max_cross_track_error_m=max(cte_abs) if cte_abs else 0.0,
        )
        episode_metrics.append(metric)
        total_reward += metric.total_reward
        total_duration_s += metric.duration_s
        total_distance_2d_m += metric.distance_2d_m

    summary = RunPerformanceSummary(
        episodes=episode_metrics,
        total_reward=float(total_reward),
        total_duration_s=float(total_duration_s),
        total_distance_2d_m=float(total_distance_2d_m),
        total_distance_3d_m=float(_cumulative_distance_3d(bridge_rows)),
        mean_cross_track_error_m=(sum(all_cte_abs) / len(all_cte_abs)) if all_cte_abs else 0.0,
        max_cross_track_error_m=max(all_cte_abs) if all_cte_abs else 0.0,
        bridge_samples=len(bridge_rows),
    )
    return summary


def format_performance_summary(title: str, summary: RunPerformanceSummary) -> str:
    """Format a concise multi-line console summary for one run.

    Args:
        title: User-facing summary title.
        summary: Aggregated run summary.

    Returns:
        Human-readable summary string suitable for stdout.
    """
    lines = [
        "=" * 62,
        f"  {title}",
        "=" * 62,
        f"  Episodes     : {len(summary.episodes)}",
        f"  Reward total : {summary.total_reward:+.2f}",
        f"  Duration     : {summary.total_duration_s:.2f} s",
        f"  Distance 2D  : {summary.total_distance_2d_m:.2f} m",
        f"  Distance 3D  : {summary.total_distance_3d_m:.2f} m",
        f"  Mean CTE     : {summary.mean_cross_track_error_m:.3f} m",
        f"  Max CTE      : {summary.max_cross_track_error_m:.3f} m",
    ]

    for metric in summary.episodes:
        lines.append(
            "  "
            f"Ep {metric.episode}: steps={metric.steps}, outcome={metric.outcome}, "
            f"reward={metric.total_reward:+.2f}, duration={metric.duration_s:.2f}s, "
            f"distance={metric.distance_2d_m:.2f}m, "
            f"mean_cte={metric.mean_cross_track_error_m:.3f}m"
        )

    lines.append("=" * 62)
    return "\n".join(lines)


def render_side_by_side_trajectory_plot(
    session_csv_path: str | Path,
    bridge_csv_path: str | Path,
    output_path: str | Path,
    title: str,
) -> Optional[Path]:
    """Render and save a side-by-side 2D-vs-3D trajectory plot.

    The left panel shows 2D trajectory data from session telemetry
    (Fossen/local frame), while the right panel shows UE5 AirSim 3D bridge
    telemetry.

    Args:
        session_csv_path: Session telemetry CSV path.
        bridge_csv_path: Bridge telemetry CSV path.
        output_path: Output image path.
        title: Figure title.

    Returns:
        Saved output path when successful, otherwise ``None``.
    """
    session_rows = _read_csv_rows(Path(session_csv_path))
    bridge_rows = _read_csv_rows(Path(bridge_csv_path))
    if not session_rows:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    try:
        fig = plt.figure(figsize=(14, 6))
        ax_2d = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

        grouped = _group_rows_by_episode(session_rows)
        for episode in sorted(grouped.keys()):
            rows = grouped[episode]
            x = [_safe_float(row.get("x", row.get("pos_x", 0.0))) for row in rows]
            y = [_safe_float(row.get("y", row.get("pos_y", 0.0))) for row in rows]
            if x and y:
                ax_2d.plot(x, y, linewidth=1.4, label=f"Ep {episode}")

        ax_2d.set_title("2D trajectory (session telemetry)")
        ax_2d.set_xlabel("X / East (m)")
        ax_2d.set_ylabel("Y / North (m)")
        ax_2d.set_aspect("equal", adjustable="datalim")
        ax_2d.grid(True, alpha=0.3)
        if grouped:
            ax_2d.legend(loc="best", fontsize=8)

        if bridge_rows:
            bx = [_safe_float(row.get("pos_x", 0.0)) for row in bridge_rows]
            by = [_safe_float(row.get("pos_y", 0.0)) for row in bridge_rows]
            bz = [_safe_float(row.get("pos_z", 0.0)) for row in bridge_rows]
            ax_3d.plot(bx, by, bz, linewidth=1.2, color="tab:orange")
            ax_3d.scatter([bx[0]], [by[0]], [bz[0]], color="tab:green", s=25)
            ax_3d.scatter([bx[-1]], [by[-1]], [bz[-1]], color="tab:red", s=25)
        else:
            ax_3d.text2D(0.1, 0.5, "No bridge telemetry", transform=ax_3d.transAxes)

        ax_3d.set_title("3D trajectory (UE5 bridge telemetry)")
        ax_3d.set_xlabel("AirSim X (m)")
        ax_3d.set_ylabel("AirSim Y (m)")
        ax_3d.set_zlabel("AirSim Z (m)")

        fig.suptitle(title)
        fig.tight_layout()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return output
    except Exception:
        return None
