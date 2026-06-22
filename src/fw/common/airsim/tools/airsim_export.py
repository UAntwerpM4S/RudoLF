"""
AirSim-specific export helpers for inference, visualization, and telemetry sessions.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .telemetry_exporter import TelemetryExporter, _DEFAULT_EXPORT_DIR
from .util import _timestamp_utc
from fw.config import (
    AIRSIM_BRIDGE_LOG_FILENAME,
    AIRSIM_METADATA_FILENAME,
    AIRSIM_SESSION_TELEMETRY_FILENAME_TEMPLATE,
)


def get_export_directory(export_dir: Optional[str | Path] = None) -> Path:
    """
    Return the export directory, creating it when needed.

    Args:
        export_dir: Optional root export directory supplied by the caller.

    Returns:
        Path: Existing or newly created export directory.
    """

    if export_dir is not None:
        path = Path(export_dir)
    else:
        path = _DEFAULT_EXPORT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(value: str) -> str:
    """
    Build a filesystem-friendly slug.

    Args:
        value: User-facing session label.

    Returns:
        str: Filesystem-safe slug suitable for directory names.
    """

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "session"


def build_airsim_env_snapshot(env) -> Dict[str, Any]:
    """
    Capture the current AirSim environment state as a serializable dictionary.

    Args:
        env: Active `AirSimEnv` instance exposing ship pose, heading,
            checkpoint, and timing state.

    Returns:
        A dictionary with scalar telemetry values that can be written directly
        to metadata or CSV exports.
    """

    snapshot = {
        "ship_x": float(env.ship_pos[0]),
        "ship_y": float(env.ship_pos[1]),
        "heading_deg": float(env.ship_heading * 180.0 / 3.141592653589793),
        "speed": float(env.ship_speed),
        "checkpoint_index": int(env.checkpoint_index),
        "cross_track_error": float(env.cross_error),
        "step_count": int(env.step_count),
        "physics_wait_s": float(env.last_physics_wait_s),
        "step_latency_s": float(env.last_step_latency_s),
        "ue5_timestamp": int(env._last_ue5_timestamp or 0),
    }
    if getattr(env, "target_pos", None) is not None:
        snapshot["target_x"] = float(env.target_pos[0])
        snapshot["target_y"] = float(env.target_pos[1])
    return snapshot


def _get_ship_scalar(value: Any, ship_index: int = 0, default: Any = 0.0) -> Any:
    """
    Return a scalar value for one ship from scalar or per-ship containers.

    Args:
        value: Scalar-like value or a per-ship container such as `list`,
            `tuple`, or `np.ndarray`.
        ship_index: Zero-based ship index to extract when `value` is a
            per-ship container.
        default: Fallback value returned when extraction is not possible.

    Returns:
        A scalar-compatible value for the requested ship.
    """

    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if len(value) > ship_index:
            return value[ship_index]
        return default
    if isinstance(value, (list, tuple)):
        if len(value) > ship_index:
            return value[ship_index]
        return default
    return value


def _get_ship_item(value: Any, ship_index: int = 0, default: Any = None) -> Any:
    """
    Return one per-ship item while tolerating flat and nested layouts.

    Args:
        value: Scalar, array, or per-ship container.
        ship_index: Preferred ship index when `value` stores per-ship items.
        default: Fallback returned when extraction is not possible.

    Returns:
        Any: Extracted ship-specific item, original flat value, or
        `default`.
    """

    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.ndim == 1:
            return value
        if len(value) > ship_index:
            return value[ship_index]
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        candidate = value[ship_index] if ship_index < len(value) else value[0]
        if isinstance(candidate, (list, tuple, np.ndarray)) or not np.isscalar(candidate):
            return candidate
        return value
    return value


def _get_ship_sim_state(env, ship_index: int = 0) -> Any:
    """
    Return one framework simulator state from common env layouts.

    Args:
        env: Active framework environment.
        ship_index: Zero-based ship index to extract when the environment
            stores per-ship simulator entries.

    Returns:
        Any: Simulator state object when available, otherwise `None`.
    """

    for attr_name in ("phys_sims", "phys_sim"):
        sim_entry = _get_ship_item(getattr(env, attr_name, None), ship_index=ship_index)
        if sim_entry is not None:
            return getattr(sim_entry, "state", sim_entry)
    return None


class AirSimSessionExporter:
    """
    Session-scoped export bundle for AirSim inference and visualization runs.

    The exporter owns one timestamped session directory and keeps telemetry,
    bridge logs, and metadata paths aligned for inference and visualization
    workflows.
    """

    def __init__(
        self,
        session_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        export_dir: Optional[str | Path] = None,
        flush_interval: int = 25,
        session_type: str = "inference",
        telemetry_filename: Optional[str] = None,
    ) -> None:
        """
        Create a dedicated export directory and file set for one session.

        Args:
            session_name: Human-readable session label used to derive the folder
                name.
            metadata: Optional initial metadata stored in `metadata.json`.
            export_dir: Optional root export directory. When omitted, the
                workspace-level `exports` folder is used.
            flush_interval: Number of queued rows to buffer before the async
                writer flushes them to disk.
            session_type: Logical session kind such as "inference" or
                "visualisation". This controls default filenames.
            telemetry_filename: Optional explicit CSV filename for the primary
                session telemetry export.
        """

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = _slugify(session_name)
        normalized_session_type = _slugify(session_type).lower()
        self.session_name = f"{slug}_{timestamp}"
        self.session_type = normalized_session_type
        self.export_dir = get_export_directory(export_dir) / self.session_name
        self.export_dir.mkdir(parents=True, exist_ok=True)

        telemetry_name = (
            telemetry_filename
            or AIRSIM_SESSION_TELEMETRY_FILENAME_TEMPLATE.format(
                session_type=self.session_type
            )
        )
        self.telemetry_path = self.export_dir / telemetry_name
        self.bridge_log_path = self.export_dir / AIRSIM_BRIDGE_LOG_FILENAME
        self.metadata_path = self.export_dir / AIRSIM_METADATA_FILENAME

        self._exporter = TelemetryExporter(
            session_type=self.session_type,
            model_path=str((metadata or {}).get("model_path", "")),
            environment_name=str((metadata or {}).get("environment_name", "")),
            config_dict=dict((metadata or {}).get("configuration", {})),
            export_dir=self.export_dir,
            flush_interval=flush_interval,
            csv_filename=self.telemetry_path.name,
            metadata_filename=self.metadata_path.name,
        )

        base_metadata = dict(metadata or {})
        base_metadata.update({
            "session_name": self.session_name,
            "session_type": self.session_type,
            "export_directory": str(self.export_dir),
            "telemetry_file": self.telemetry_path.name,
            "airsim_bridge_log": self.bridge_log_path.name,
            "created_at_utc": _timestamp_utc(),
        })
        self.update_metadata(base_metadata)

    def log_step(self, row: Dict[str, Any]) -> None:
        """
        Queue one session telemetry row.

        Args:
            row: Flat telemetry payload to append to the session CSV.
        """

        self._exporter.log_step(row)

    def log_inference_step(self, row: Dict[str, Any]) -> None:
        """
        Queue one inference telemetry row.

        Args:
            row: Inference-step telemetry payload to append to the session CSV.
        """

        self.log_step(row)

    def log_visualisation_step(self, row: Dict[str, Any]) -> None:
        """
        Queue one visualization telemetry row.

        Args:
            row: Visualisation-step telemetry payload to append to the session
                CSV.
        """

        self.log_step(row)

    def update_metadata(self, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Merge new metadata into the sidecar JSON.

        Args:
            metadata: Optional mapping of metadata fields to merge.
            **kwargs: Additional metadata fields merged after `metadata`.
        """

        self._exporter.update_metadata(metadata, **kwargs)

    def save(self) -> None:
        """
        Flush CSV rows and write metadata to disk.
        """

        self._exporter.save()

    def close(self) -> None:
        """
        Flush everything and close the underlying exporter.
        """

        self._exporter.close()

    def __enter__(self) -> "AirSimSessionExporter":
        """
        Enter runtime context for automatic resource management.

        Returns:
            AirSimSessionExporter: The active exporter instance.
        """

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """
        Ensure exporter is closed when exiting context manager.

        Args:
            exc_type: Exception type raised inside the context, if any.
            exc_value: Exception instance raised inside the context, if any.
            traceback: Traceback object for the active exception, if any.
        """

        try:
            self.save()
        finally:
            self.close()


def build_framework_visualisation_row(
    env,
    bridge,
    episode_index: int,
    step_index: int,
    reward: float,
    action,
    outcome: str,
    control_mode: str = "framework_visualisation",
    ship_index: int = 0,
    model_origin: str = "framework",
) -> Dict[str, Any]:
    """
    Build one lightweight visualization telemetry row from framework state.

    Args:
        env: Active framework environment containing ship pose and simulator
            state.
        bridge: Connected AirSim bridge providing the latest mirrored control
            state.
        episode_index: One-based episode index for the active run.
        step_index: One-based step index within the active episode.
        reward: Reward value returned for the current framework step.
        action: Policy action associated with the current step.
        outcome: Step outcome label such as "running", "terminated",
            or "truncated".
        control_mode: Export label describing how controls are being mirrored
            during visualization.
        ship_index: Zero-based ship index used when the environment stores
            per-ship state in lists or arrays.
        model_origin: High-level source label for the mirrored model, such as
            "pysim" or "fhsim".

    Returns:
        A flat CSV-friendly dictionary describing the current visualization
        step.
    """

    ship_pos = _get_ship_item(getattr(env, "ship_pos", None), ship_index=ship_index)
    if ship_pos is None or (isinstance(ship_pos, (list, tuple, np.ndarray)) and len(ship_pos) == 0):
        ship_pos = (0.0, 0.0)

    state = _get_ship_sim_state(env, ship_index=ship_index)
    if state is None:
        class _DefaultState:
            """
            Default motion state used when AirSim state is unavailable.
            """

            heading = 0.0
            u = 0.0

        state = _DefaultState()

    action_arr = np.asarray(action).reshape(-1)
    controls = getattr(bridge, "current_controls", getattr(bridge, "_controls", None))
    checkpoint_index = _get_ship_scalar(getattr(env, "checkpoint_index", 0), ship_index=ship_index, default=0)
    cross_error = _get_ship_scalar(getattr(env, "cross_error", 0.0), ship_index=ship_index, default=0.0)
    return {
        "step": int(step_index),
        "sim_time": float(step_index * env.time_step),
        "episode": int(episode_index),
        "episode_step": int(step_index),
        "x": float(ship_pos[0]),
        "y": float(ship_pos[1]),
        "pos_x": float(ship_pos[0]),
        "pos_y": float(ship_pos[1]),
        "heading_deg": float(np.rad2deg(float(getattr(state, "heading", 0.0)))),
        "speed": float(getattr(state, "u", 0.0)),
        "reward": float(reward),
        "steering": float(action_arr[0]) if action_arr.size >= 1 else 0.0,
        "throttle": float(action_arr[1]) if action_arr.size >= 2 else 0.0,
        "brake": float(getattr(controls, "brake", 0.0) if controls is not None else 0.0),
        "checkpoint_index": int(checkpoint_index),
        "cross_track_error": float(cross_error),
        "physics_wait_s": "",
        "step_latency_s": "",
        "ue5_timestamp": "",
        "outcome": str(outcome),
        "model_origin": str(model_origin),
        "control_mode": str(control_mode),
    }
