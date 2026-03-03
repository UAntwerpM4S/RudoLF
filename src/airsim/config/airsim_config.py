"""AirSim connection configuration with validation.

Provides a validated ``AirSimConfig`` dataclass that centralises all
CosysAirSim connection parameters.  PID / control-loop gains are
intentionally **not** stored here — they belong in the future
Fossen-input-mapping layer.

Example::

    >>> from airsim.config.airsim_config import AirSimConfig
    >>> cfg = AirSimConfig()
    >>> cfg.port
    41451
    >>> cfg = AirSimConfig(port=0)  # raises ValueError
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


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

    def __post_init__(self) -> None:
        """Validate all fields after initialisation."""
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
        """Return the effective IP address (``127.0.0.1`` when empty)."""
        return self.ip if self.ip else "127.0.0.1"

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary of all config values.

        Returns:
            A dictionary with all configuration parameters.
        """
        return asdict(self)


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
    return AirSimConfig().to_dict()
