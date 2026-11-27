"""
Helpers for writing live per-iteration training metrics to CSV.
"""

from __future__ import annotations

import csv
import logging

from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone
from fw.common.tools.path_utils import resolve_path_relative_to_src

logger = logging.getLogger(__name__)

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

_MEMORY_GROWTH_ALERT_RATIO = 2.0  # alert when RSS doubles since last sample (leak indicator)


class TrainingIterationLogger:
    """
    Buffer PPO iteration rows and write them to CSV on demand.
    """

    _FIELDNAMES = (
        "run_started_utc",
        "logged_at_utc",
        "env_name",
        "agent_type",
        "device",
        "rollout_device",
        "iteration",
        "mean_reward",
        "mean_advantage",
        "loss",
        "num_episode_batches",
        "elapsed_seconds",
        "rss_mb",
    )

    def __init__(
        self,
        *,
        env_name: str,
        agent_type: str,
        device: str,
        src_dir: Path,
        log_dir: str = "logs",
        rollout_device: Optional[str] = None,
    ) -> None:
        """
        Create a new timestamped iteration log under the requested log dir.
        rollout_device is the device of the rollout forward passes (column
        "rollout_device"); None means the same as device.
        """

        self._run_started = datetime.now(timezone.utc)
        self._run_started_text = self._run_started.isoformat()
        timestamp = self._run_started.strftime("%Y%m%d_%H%M%S")
        resolved_dir = resolve_path_relative_to_src(log_dir, src_dir)
        resolved_dir.mkdir(parents=True, exist_ok=True)
        self.path = resolved_dir / f"{env_name}_{agent_type}_iterations_{timestamp}.csv"
        self._env_name = str(env_name)
        self._agent_type = str(agent_type)
        self._device = str(device)
        # Device of the rollout forward passes (PPOPolicy); the same as device without a split.
        self._rollout_device = str(device) if rollout_device is None else str(rollout_device)
        self._rows: list[dict[str, Any]] = []
        self._prev_rss_mb: float = 0.0
        self._psutil_proc = _psutil.Process() if _PSUTIL_AVAILABLE else None

    def _sample_rss_mb(self) -> float:
        """
        Return current process RSS in MiB, or 0.0 if psutil is unavailable.
        """

        if self._psutil_proc is None:
            return 0.0
        try:
            return self._psutil_proc.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def log_iteration(
        self,
        *,
        iteration: int,
        mean_reward: float,
        mean_advantage: float,
        loss: float,
        num_episode_batches: int,
        elapsed_seconds: float,
    ) -> None:
        """
        Buffer one training-iteration row; save() writes the buffer to CSV.
        """

        rss_mb = self._sample_rss_mb()
        if self._prev_rss_mb > 0.0 and rss_mb > self._prev_rss_mb * _MEMORY_GROWTH_ALERT_RATIO:
            logger.warning(
                "[Alert] Memory growth: %.1f MB \u2192 %.1f MB (+%.1f MB) at iteration %d.",
                self._prev_rss_mb, rss_mb, rss_mb - self._prev_rss_mb, iteration
            )
        self._prev_rss_mb = rss_mb

        row: dict[str, Any] = {
            "run_started_utc": self._run_started_text,
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "env_name": self._env_name,
            "agent_type": self._agent_type,
            "device": self._device,
            "rollout_device": self._rollout_device,
            "iteration": int(iteration),
            "mean_reward": float(mean_reward),
            "mean_advantage": float(mean_advantage),
            "loss": float(loss),
            "num_episode_batches": int(num_episode_batches),
            "elapsed_seconds": float(elapsed_seconds),
            "rss_mb": round(rss_mb, 2),
        }
        self._rows.append(row)

    def save(self) -> None:
        """
        Write the buffered iteration rows to disk.
        """

        if not self._rows:
            return

        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self._FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)
