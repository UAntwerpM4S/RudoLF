"""Tests for shared AirSim telemetry analysis helpers."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[4] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fw.common.airsim.tools.trajectory_analysis import (
    build_run_summary_from_csv,
    format_performance_summary,
)


def _write_csv(path: Path, fieldnames, rows) -> None:
    """Write deterministic CSV content for tests.

    Args:
        path: Output CSV path.
        fieldnames: Ordered CSV columns.
        rows: Row dictionaries to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_run_summary_from_csv_computes_metrics(tmp_path: Path) -> None:
    """Summary should aggregate reward, duration, distance, and CTE correctly."""
    session_csv = tmp_path / "inference_session.csv"
    bridge_csv = tmp_path / "airsim_telemetry.csv"

    _write_csv(
        session_csv,
        fieldnames=[
            "step",
            "sim_time",
            "episode",
            "episode_step",
            "x",
            "y",
            "reward",
            "cross_track_error",
            "outcome",
        ],
        rows=[
            {"step": 1, "sim_time": 0.1, "episode": 1, "episode_step": 1, "x": 0.0, "y": 0.0, "reward": 1.0, "cross_track_error": 1.0, "outcome": "running"},
            {"step": 2, "sim_time": 0.2, "episode": 1, "episode_step": 2, "x": 3.0, "y": 4.0, "reward": 2.0, "cross_track_error": 2.0, "outcome": "terminated"},
            {"step": 1, "sim_time": 0.1, "episode": 2, "episode_step": 1, "x": 10.0, "y": 0.0, "reward": -1.0, "cross_track_error": 1.0, "outcome": "running"},
            {"step": 2, "sim_time": 0.2, "episode": 2, "episode_step": 2, "x": 13.0, "y": 0.0, "reward": 3.0, "cross_track_error": 3.0, "outcome": "truncated"},
        ],
    )

    _write_csv(
        bridge_csv,
        fieldnames=["pos_x", "pos_y", "pos_z"],
        rows=[
            {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0},
            {"pos_x": 0.0, "pos_y": 3.0, "pos_z": 4.0},
            {"pos_x": 0.0, "pos_y": 3.0, "pos_z": 8.0},
        ],
    )

    summary = build_run_summary_from_csv(session_csv, bridge_csv)

    assert len(summary.episodes) == 2
    assert summary.total_reward == 5.0
    assert summary.total_duration_s == 0.4
    assert summary.total_distance_2d_m == 8.0
    assert summary.total_distance_3d_m == 9.0
    assert summary.mean_cross_track_error_m == 1.75
    assert summary.max_cross_track_error_m == 3.0
    assert summary.bridge_samples == 3



def test_format_performance_summary_contains_expected_lines(tmp_path: Path) -> None:
    """Console formatter should include key BI metrics and per-episode lines."""
    session_csv = tmp_path / "visualisation_session.csv"
    _write_csv(
        session_csv,
        fieldnames=["step", "sim_time", "episode", "episode_step", "x", "y", "reward", "cross_track_error", "outcome"],
        rows=[
            {"step": 1, "sim_time": 0.1, "episode": 1, "episode_step": 1, "x": 1.0, "y": 1.0, "reward": 0.5, "cross_track_error": 0.2, "outcome": "terminated"},
        ],
    )

    summary = build_run_summary_from_csv(session_csv)
    rendered = format_performance_summary("TEST SUMMARY", summary)

    assert "TEST SUMMARY" in rendered
    assert "Reward total" in rendered
    assert "Distance 2D" in rendered
    assert "Mean CTE" in rendered
    assert "Ep 1:" in rendered
