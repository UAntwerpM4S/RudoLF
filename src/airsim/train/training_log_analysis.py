"""
Utilities for extracting training-curriculum hints from episode CSV logs.
"""

from __future__ import annotations

import csv
import statistics

from pathlib import Path
from typing import DefaultDict
from dataclasses import dataclass
from collections import Counter, defaultdict


@dataclass(frozen=True)
class TrainingLogSummary:
    """
    Compact summary of an AirSim training episode log.
    """

    path: Path
    total_rows: int
    episode_count: int
    median_max_checkpoint: int
    modal_max_checkpoint: int
    best_max_checkpoint: int
    high_speed_offtrack_ratio: float
    recommended_curriculum_target_checkpoint: int
    recommended_curriculum_window_size: int


def analyze_training_log(
    path: str | Path,
    route_total_checkpoints: int | None = None,
) -> TrainingLogSummary:
    """
    Summarize an AirSim episode log and recommend a curriculum frontier.

    The training log stores one row per environment step. For curriculum
    seeding we care primarily about the furthest checkpoint each episode
    reached and whether the policy tends to overspeed while already off track.

    Args:
        path: CSV file path.
        route_total_checkpoints: Optional upper bound used to clamp the
            curriculum recommendation to the real route length.

    Returns:
        A :class:`TrainingLogSummary` with conservative curriculum defaults.
    """

    resolved_path = Path(path)
    fallback_total = max(1, int(route_total_checkpoints or 8))
    fallback_window = min(8, fallback_total)

    if not resolved_path.exists():
        return TrainingLogSummary(
            path=resolved_path,
            total_rows=0,
            episode_count=0,
            median_max_checkpoint=0,
            modal_max_checkpoint=0,
            best_max_checkpoint=0,
            high_speed_offtrack_ratio=0.0,
            recommended_curriculum_target_checkpoint=max(1, fallback_window),
            recommended_curriculum_window_size=max(1, fallback_window),
        )

    episodes: DefaultDict[int, list[int]] = defaultdict(list)
    total_rows = 0
    high_speed_offtrack_rows = 0

    with resolved_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                episode = int(float(row["episode"]))
                checkpoint_index = int(float(row["checkpoint_index"]))
                speed = float(row["speed"])
                cross_track_error = float(row["cross_track_error"])
            except (KeyError, TypeError, ValueError):
                continue

            episodes[episode].append(checkpoint_index)
            total_rows += 1

            if speed >= 0.95 * 5.0 and cross_track_error >= 60.0:
                high_speed_offtrack_rows += 1

    if not episodes:
        return TrainingLogSummary(
            path=resolved_path,
            total_rows=0,
            episode_count=0,
            median_max_checkpoint=0,
            modal_max_checkpoint=0,
            best_max_checkpoint=0,
            high_speed_offtrack_ratio=0.0,
            recommended_curriculum_target_checkpoint=max(1, fallback_window),
            recommended_curriculum_window_size=max(1, fallback_window),
        )

    max_checkpoints = [max(checkpoints) for checkpoints in episodes.values() if checkpoints]
    median_max_checkpoint = int(round(statistics.median(max_checkpoints)))
    modal_max_checkpoint = Counter(max_checkpoints).most_common(1)[0][0]
    best_max_checkpoint = max(max_checkpoints)
    route_cap = max(1, int(route_total_checkpoints or best_max_checkpoint))

    recommended_window_size = int(max(6, min(10, median_max_checkpoint + 1)))
    recommended_curriculum_target_checkpoint = int(
        max(
            recommended_window_size,
            median_max_checkpoint + 2,
            modal_max_checkpoint + 1,
        )
    )

    recommended_window_size = min(recommended_window_size, route_cap)
    recommended_curriculum_target_checkpoint = min(
        max(1, recommended_curriculum_target_checkpoint),
        route_cap,
    )

    return TrainingLogSummary(
        path=resolved_path,
        total_rows=total_rows,
        episode_count=len(episodes),
        median_max_checkpoint=median_max_checkpoint,
        modal_max_checkpoint=modal_max_checkpoint,
        best_max_checkpoint=best_max_checkpoint,
        high_speed_offtrack_ratio=(
            float(high_speed_offtrack_rows) / float(total_rows) if total_rows > 0 else 0.0
        ),
        recommended_curriculum_target_checkpoint=recommended_curriculum_target_checkpoint,
        recommended_curriculum_window_size=max(1, recommended_window_size),
    )
