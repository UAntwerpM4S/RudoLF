"""AirSim connection configuration with validation.

Provides a validated ``AirSimConfig`` dataclass that centralises all
CosysAirSim connection parameters. PID / control-loop gains are
intentionally **not** stored here because they belong in the
input-mapping layer instead.

Example::

    >>> from airsim.config.airsim_config import AirSimConfig
    >>> cfg = AirSimConfig()
    >>> cfg.port
    41451
    >>> cfg = AirSimConfig(port=0)  # raises ValueError
"""

from __future__ import annotations

import json
import ipaddress
import logging
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_RUNTIME_SETTINGS_PATH = _THIS_DIR / "runtime_settings.json"
DEFAULT_AIRSIM_SAVE_SLOT = "simulator_1"
DEFAULT_AIRSIM_SIMULATION_SPEED = 8.0
# Calibrated visual gain for UE5 replay throttle animation.
# The current ship's throttle envelope is roughly 5.6, so 5.0 / 5.6 keeps the
# visible throttle close to the framework speed range during replay.
DEFAULT_THROTTLE_VISUAL_GAIN = 5.0 / 5.6


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two nested dictionaries without mutating either input.

    Args:
        base: Baseline dictionary to merge into.
        override: Override dictionary whose values win on conflicts.

    Returns:
        Dict[str, Any]: New recursively merged dictionary.
    """
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class AirSimConfig:
    """Validated, immutable configuration for the AirSim bridge.

    All values are validated in ``__post_init__``.  Invalid parameters
    raise ``ValueError`` with a clear, actionable message.

    Attributes:
        use_airsim: Enable the AirSim bridge. When ``False`` the bridge
            is never connected, regardless of other settings.
        ip: AirSim server IP address.  An empty string means
            ``127.0.0.1`` (localhost).
        port: AirSim RPC port (1–65 535).
        timeout: Connection timeout in seconds (must be > 0).
        max_retries: Maximum number of connection attempts before
            giving up (0 = try once, no retries).
        health_check_interval: Seconds between periodic health checks
            (must be > 0).
    """

    use_airsim: bool = True
    ip: str = ""
    port: int = 41451
    timeout: float = 10.0
    max_retries: int = 3
    health_check_interval: float = 30.0
    csv_log_path: str = ""

    def __post_init__(self) -> None:
        """Validate all fields after initialisation.

        Raises:
            ValueError: If any configuration field is invalid.
        """
        # ── IP ──
        if self.ip:
            try:
                ipaddress.ip_address(self.ip)
            except ValueError:
                raise ValueError(
                    f"Invalid IP address '{self.ip}'. "
                    f"Expected a valid IPv4/IPv6 address or an empty string for localhost."
                )

        # ── Port ──
        if not isinstance(self.port, int) or not (1 <= self.port <= 65_535):
            raise ValueError(
                f"Port must be an integer in range 1–65535, got {self.port!r}."
            )

        # ── Timeout ──
        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise ValueError(
                f"Timeout must be a positive number (seconds), got {self.timeout!r}."
            )

        # ── Max retries ──
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError(
                f"max_retries must be a non-negative integer, got {self.max_retries!r}."
            )

        # ── Health-check interval ──
        if (
            not isinstance(self.health_check_interval, (int, float))
            or self.health_check_interval <= 0
        ):
            raise ValueError(
                f"health_check_interval must be a positive number (seconds), "
                f"got {self.health_check_interval!r}."
            )

    # ── Convenience helpers ────────────────────────────────

    @property
    def resolved_ip(self) -> str:
        """Return the effective IP address (``127.0.0.1`` when empty).

        Returns:
            str: Resolved IP address used for network calls.
        """
        return self.ip if self.ip else "127.0.0.1"

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary of all config values.

        Returns:
            A dictionary with all configuration parameters.
        """
        return asdict(self)


def get_runtime_settings_path() -> Path:
    """Return the path to the runtime settings JSON file.

    Returns:
        Path: Runtime settings JSON path.
    """
    return _RUNTIME_SETTINGS_PATH


def get_default_runtime_settings() -> Dict[str, Any]:
    """Return the built-in runtime defaults used when JSON is absent or invalid.

    Returns:
        Dict[str, Any]: Nested runtime-settings dictionary with built-in
        defaults.
    """
    return {
        "visualization": {
            "model_visualise": False,
            "inference_visualise": False,
            "model_path": "models/PySimEnv_PPO_policy.zip",
            "inference_model_path": "models/AirSimEnv_PPO_policy.zip",
            "environment_name": "auto",
            "model_family": "auto",
            "mirror_controls_in_ue5": False,
            "animate_steering_in_ue5": True,
            "animate_throttle_in_ue5": True,
            "render_interpolation_steps": 12,
            "steering_visual_gain": 1.0,
            "throttle_visual_gain": DEFAULT_THROTTLE_VISUAL_GAIN,
            "use_kinematics_in_ue5": False,
            "min_render_hz": 90.0,
            "max_translation_step": 0.15,
            "max_yaw_step_deg": 1.5,
            "vehicle_object_name": "",
            "lock_path_in_ue5": False,
            "episodes": 1,
            "max_steps": 14000,
            "device": "cuda",
            "log_every": 50,
            "min_log_interval_s": 0.25,
            "bridge_telemetry_hz": 12.0,
            "startup_settle_s": 0.4,
            "draw_calibration_reference": True,
        },
        "training": {
            "speed_profile": "custom",
            "max_steps": 120000,
            "simulation_speed": DEFAULT_AIRSIM_SIMULATION_SPEED,
            "draw_calibration_reference": False,
            "no_rendering_mode": True,
            "load_world_on_start": True,
            "capture_live_start_pose": True,
            "use_pause_stepping": True,
            "use_sim_time_stepping": True,
            "sim_time_per_action": 0.05,
            "frames_per_step": 8,
            "time_step": 0.05,
            "physics_update_timeout": 1.2,
            "state_poll_interval": 0.003,
            "steering_dynamics_enabled": True,
            "steering_smoothing_alpha": 0.65,
            "max_steering_rate_per_second": 3.2,
        },
        "airsim": {
            **AirSimConfig().to_dict(),
            "save_slot": DEFAULT_AIRSIM_SAVE_SLOT,
            "simulation_speed": DEFAULT_AIRSIM_SIMULATION_SPEED,
        },
    }


def merge_runtime_settings(override: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge a partial runtime-settings override onto the built-in defaults.

    Args:
        override: Partial runtime-settings override dictionary.

    Returns:
        Dict[str, Any]: Fully merged runtime-settings dictionary.
    """
    if not isinstance(override, dict):
        return get_default_runtime_settings()
    return _deep_merge(get_default_runtime_settings(), override)


def get_runtime_settings() -> Dict[str, Any]:
    """Load runtime settings from JSON and merge them with defaults.

    Returns:
        Dict[str, Any]: Runtime-settings dictionary ready for consumers.
    """
    defaults = get_default_runtime_settings()

    if not _RUNTIME_SETTINGS_PATH.exists():
        return defaults

    try:
        with open(_RUNTIME_SETTINGS_PATH, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not load runtime settings from '%s'; using built-in defaults. %s",
            _RUNTIME_SETTINGS_PATH,
            exc,
        )
        return defaults
    return merge_runtime_settings(loaded)


# ── Backward-compatible accessor ──────────────────────────


def get_airsim_config() -> Dict[str, Any]:
    """Return a dictionary of AirSim configuration values.

    Backward-compatible wrapper around :class:`AirSimConfig`.

    Returns:
        A dictionary with all AirSim-specific parameters.

    Example::

        >>> cfg = get_airsim_config()
        >>> cfg["port"]
        41451
    """
    runtime = get_runtime_settings().get("airsim", {})
    valid_keys = set(AirSimConfig.__dataclass_fields__.keys())
    filtered = {key: runtime[key] for key in valid_keys if key in runtime}
    return AirSimConfig(**filtered).to_dict()
