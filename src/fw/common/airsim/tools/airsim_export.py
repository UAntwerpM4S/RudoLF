"""AirSim-specific export helpers for inference, visualisation, and telemetry sessions."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from telemetry_exporter import TelemetryExporter


def get_export_directory(export_dir: Optional[str | Path] = None) -> Path:
    """Return the export directory, creating it when needed."""
    if export_dir is not None:
        path = Path(export_dir)
    else:
        path = Path(__file__).resolve().parents[4] / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp_utc() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    """Build a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._-") or "session"


def build_airsim_env_snapshot(env) -> Dict[str, Any]:
    """Capture the current AirSim environment state as a serializable dictionary.

    Args:
        env: Active ``AirSimEnv`` instance exposing ship pose, heading,
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
    """Return a scalar value for one ship from scalar or per-ship containers.

    Args:
        value: Scalar-like value or a per-ship container such as ``list``,
            ``tuple``, or ``np.ndarray``.
        ship_index: Zero-based ship index to extract when ``value`` is a
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


class AirSimSessionExporter:
    """Session-scoped export bundle for AirSim inference and visualisation runs."""

    def __init__(
        self,
        session_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        export_dir: Optional[str | Path] = None,
        flush_interval: int = 25,
        session_type: str = "inference",
        telemetry_filename: Optional[str] = None,
    ) -> None:
        """Create a dedicated export directory and file set for one session.

        Args:
            session_name: Human-readable session label used to derive the folder
                name.
            metadata: Optional initial metadata stored in ``metadata.json``.
            export_dir: Optional root export directory. When omitted, the
                workspace-level ``exports`` folder is used.
            flush_interval: Number of queued rows to buffer before the async
                writer flushes them to disk.
            session_type: Logical session kind such as ``"inference"`` or
                ``"visualisation"``. This controls default filenames.
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

        telemetry_name = telemetry_filename or f"{self.session_type}_session.csv"
        self.telemetry_path = self.export_dir / telemetry_name
        self.bridge_log_path = self.export_dir / "airsim_telemetry.csv"
        self.metadata_path = self.export_dir / "metadata.json"

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
        """Queue one session telemetry row.

        Args:
            row: Flat telemetry payload to append to the session CSV.
        """
        self._exporter.log_step(row)

    def log_inference_step(self, row: Dict[str, Any]) -> None:
        """Queue one inference telemetry row.

        Args:
            row: Inference-step telemetry payload to append to the session CSV.
        """
        self.log_step(row)

    def log_visualisation_step(self, row: Dict[str, Any]) -> None:
        """Queue one visualisation telemetry row.

        Args:
            row: Visualisation-step telemetry payload to append to the session
                CSV.
        """
        self.log_step(row)

    def update_metadata(self, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Merge new metadata into the sidecar JSON.

        Args:
            metadata: Optional mapping of metadata fields to merge.
            **kwargs: Additional metadata fields merged after ``metadata``.
        """
        self._exporter.update_metadata(metadata, **kwargs)

    def save(self) -> None:
        """Flush CSV rows and write metadata to disk."""
        self._exporter.save()

    def close(self) -> None:
        """Flush everything and close the underlying exporter."""
        self._exporter.close()


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
) -> Dict[str, Any]:
    """Build one lightweight visualisation telemetry row from framework state.

    Args:
        env: Active framework environment containing ship pose and simulator
            state.
        bridge: Connected AirSim bridge providing the latest mirrored control
            state.
        episode_index: One-based episode index for the active run.
        step_index: One-based step index within the active episode.
        reward: Reward value returned for the current framework step.
        action: Policy action associated with the current step.
        outcome: Step outcome label such as ``"running"``, ``"terminated"``,
            or ``"truncated"``.
        control_mode: Export label describing how controls are being mirrored
            during visualisation.
        ship_index: Zero-based ship index used when the environment stores
            per-ship state in lists or arrays.

    Returns:
        A flat CSV-friendly dictionary describing the current visualisation
        step.
    """
    ship_pos = env.ship_pos[0]
    state = env.phys_sims[0].state
    action_arr = np.asarray(action).reshape(-1)
    controls = getattr(bridge, "_controls", None)
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
        "heading_deg": float(np.rad2deg(float(state.heading))),
        "speed": float(state.u),
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
        "model_origin": "fossen",
        "control_mode": str(control_mode),
    }
