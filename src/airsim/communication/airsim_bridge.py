"""CosysAirSim bridge — robust connection manager for UE5.

This bridge manages the network connection to CosysAirSim running
inside Unreal Engine 5.  It provides:

* **UE5 auto-detection** via a lightweight TCP port probe.
* **Retry with exponential backoff** for resilient startup.
* **Periodic health-checks** to detect stale connections.
* **Pass-through control** — ``send_action()`` forwards raw
  ``[steering, throttle]`` to AirSim without any PID processing.

The actual *input mapping* (Fossen model → UE5 controls) will be
implemented in a separate layer in the future.

Coordinate frames:
    * **Ship (Fossen):** x = east, y = north, heading from east (CCW+).
    * **AirSim (NED):** x = north, y = east, yaw from north (CW+).
    * **Transform:** ``AirSim_yaw = π/2 − ship_heading``.

Example::

    >>> from airsim.config.airsim_config import AirSimConfig
    >>> bridge = AirSimBridge(AirSimConfig())
    >>> bridge.is_connected
    True
    >>> bridge.health_check()
    True
    >>> bridge.disconnect()
"""

from __future__ import annotations

import math
import socket
import time
import logging
from typing import Any, Dict, Optional

import numpy as np

from airsim.config.airsim_config import AirSimConfig

logger = logging.getLogger(__name__)


# ── Utility helpers ────────────────────────────────────────


def _normalize_angle(a: float) -> float:
    """Wrap an angle to the interval [−π, π].

    Args:
        a: Angle in radians.

    Returns:
        The equivalent angle in the range [−π, π].
    """
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quaternion(q) -> float:
    """Extract the yaw angle from an AirSim quaternion.

    Uses the standard ZYX Euler-angle decomposition to isolate
    the yaw component.

    Args:
        q: A ``cosysairsim.Quaternionr`` instance with attributes
            ``w_val``, ``x_val``, ``y_val``, and ``z_val``.

    Returns:
        Yaw angle in radians.
    """
    return math.atan2(
        2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
        1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2),
    )


def is_ue5_reachable(ip: str = "", port: int = 41451, timeout: float = 2.0) -> bool:
    """Probe the AirSim RPC port to check if UE5 is reachable.

    Performs a lightweight TCP connect — does **not** require the
    ``cosysairsim`` package.

    Args:
        ip: AirSim server IP.  Empty string = ``127.0.0.1``.
        port: AirSim RPC port.
        timeout: Socket connect timeout in seconds.

    Returns:
        ``True`` if the port is open, ``False`` otherwise.
    """
    host = ip if ip else "127.0.0.1"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


# ── AirSimBridge ───────────────────────────────────────────


class AirSimBridge:
    """Robust connection manager and pass-through adapter for CosysAirSim.

    Handles UE5 auto-detection, retry with exponential backoff,
    periodic health checks, and forwards raw control commands.

    Attributes:
        config: The :class:`AirSimConfig` used to create this bridge.
    """

    def __init__(self, config: Optional[AirSimConfig] = None) -> None:
        """Initialise the AirSim bridge and connect to the simulator.

        Connection flow:
            1. Probe the RPC port (``is_ue5_reachable``).
            2. If reachable, attempt ``CarClient.confirmConnection()``
               with retry + exponential backoff.
            3. On success, enable API control.

        If ``cosysairsim`` is not installed, or UE5 is not reachable,
        or all retries fail, the bridge stays disconnected and all
        control methods become no-ops.

        Args:
            config: Connection parameters.  Defaults to
                ``AirSimConfig()`` (localhost:41451).
        """
        self.config: AirSimConfig = config or AirSimConfig()
        self._connected: bool = False
        self._client: Any = None
        self._controls: Any = None
        self._airsim: Any = None
        self._last_health_check: float = time.monotonic()
        self._step_count: int = 0

        if not self.config.use_airsim:
            logger.info("AirSim bridge disabled via config (use_airsim=False).")
            return

        # ── Step 1: lightweight port probe ──
        if not is_ue5_reachable(
            self.config.ip, self.config.port, timeout=self.config.timeout
        ):
            logger.warning(
                "UE5 not reachable at %s:%d — bridge disabled.",
                self.config.resolved_ip,
                self.config.port,
            )
            print(
                f"[AirSimBridge] UE5 not detected at "
                f"{self.config.resolved_ip}:{self.config.port} — bridge disabled."
            )
            return

        # ── Step 2: import cosysairsim ──
        try:
            import cosysairsim as airsim

            self._airsim = airsim
        except ImportError:
            logger.warning("cosysairsim package not installed — bridge disabled.")
            print("Warning: cosysairsim not installed. AirSim bridge disabled.")
            return

        # ── Step 3: connect with retry + exponential backoff ──
        self._connect_with_retry(airsim)

    def _connect_with_retry(self, airsim) -> None:
        """Attempt to connect ``max_retries + 1`` times with backoff.

        On each failure the wait time doubles (capped at 30 s).
        If all attempts fail, ``_connected`` stays ``False`` and a
        ``ConnectionError`` is logged (but not raised, to keep the
        bridge non-fatal).

        Args:
            airsim: The imported ``cosysairsim`` module.
        """
        attempts = self.config.max_retries + 1  # first attempt + retries
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                self._client = airsim.CarClient(
                    ip=self.config.ip, port=self.config.port,
                    timeout_value=self.config.timeout,
                )
                self._client.confirmConnection()
                self._client.enableApiControl(True)
                self._controls = airsim.CarControls()
                self._connected = True
                self._last_health_check = time.monotonic()

                logger.info(
                    "Connected to Cosys-AirSim at %s:%d (attempt %d/%d)",
                    self.config.resolved_ip, self.config.port,
                    attempt + 1, attempts,
                )
                print(
                    f"[AirSimBridge] Connected to CosysAirSim at "
                    f"{self.config.resolved_ip}:{self.config.port}"
                )
                return

            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1:
                    backoff = min(2 ** attempt, 30)
                    logger.warning(
                        "Connection attempt %d/%d failed: %s — retrying in %ds…",
                        attempt + 1, attempts, exc, backoff,
                    )
                    print(
                        f"[AirSimBridge] Attempt {attempt + 1}/{attempts} failed: "
                        f"{exc} — retrying in {backoff}s…"
                    )
                    time.sleep(backoff)

        # All attempts exhausted
        logger.error(
            "Failed to connect to AirSim at %s:%d after %d attempts: %s",
            self.config.resolved_ip, self.config.port, attempts, last_error,
        )
        print(
            f"[AirSimBridge] Failed to connect after {attempts} attempts. "
            f"Bridge disabled."
        )

    # ── Properties ─────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """Whether the bridge currently has an active AirSim connection."""
        return self._connected

    @property
    def client(self) -> Any:
        """Expose the underlying CarClient for advanced environment state queries.

        Raises:
            ConnectionError: If the bridge is not currently connected to AirSim.
        """
        if not self._ensure_connected():
            raise ConnectionError("Cannot access AirSim client: bridge disconnected.")
        return self._client

    # ── Health check ───────────────────────────────────────

    def health_check(self) -> bool:
        """Verify the AirSim connection is still alive.

        Calls ``getCarState()`` as a lightweight ping.

        Returns:
            ``True`` if the connection is healthy, ``False`` otherwise.
        """
        if not self._connected:
            return False
        try:
            self._client.getCarState()
            self._last_health_check = time.monotonic()
            return True
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
            self._connected = False
            return False

    def _ensure_connected(self) -> bool:
        """Run a health check if the interval has elapsed.

        Returns:
            ``True`` if connected (or still within interval), ``False``
            if a health check ran and failed.
        """
        if not self._connected:
            return False
        elapsed = time.monotonic() - self._last_health_check
        if elapsed >= self.config.health_check_interval:
            return self.health_check()
        return True

    # ── Control methods ────────────────────────────────────

    def send_action(self, action: np.ndarray) -> None:
        """Forward raw control commands to AirSim.

        This is a **pass-through stub**.  The actual Fossen-to-UE5
        input mapping will be implemented in a separate layer.

        Args:
            action: ``[steering, throttle]`` array in [−1, 1].
        """
        if not self._ensure_connected():
            return

        try:
            action = np.asarray(action, dtype=np.float32)
            self._controls.steering = float(np.clip(action[0], -1.0, 1.0))
            self._controls.throttle = float(np.clip(action[1], 0.0, 1.0))
            self._controls.brake = 0.0
            self._controls.is_manual_gear = True
            self._controls.manual_gear = 1
            self._client.setCarControls(self._controls)
            self._step_count += 1
        except Exception as exc:
            logger.warning("AirSim send_action failed: %s", exc)

    def get_state(self) -> Dict[str, Any]:
        """Retrieve the current vehicle state from AirSim.

        Returns:
            A dictionary with ``"speed"``, ``"gear"``, and
            ``"timestamp"`` keys when connected.  Empty dict otherwise.
        """
        if not self._ensure_connected():
            return {}
        try:
            s = self._client.getCarState()
            return {"speed": s.speed, "gear": s.gear, "timestamp": s.timestamp}
        except Exception as exc:
            logger.warning("AirSim get_state failed: %s", exc)
            return {}

    def reset(self) -> None:
        """Reset the AirSim vehicle and re-enable API control.

        Does nothing if the bridge is not connected.
        """
        if not self._connected:
            return
        try:
            self._client.reset()
            time.sleep(0.5)
            self._client.enableApiControl(True)
            self._controls = self._airsim.CarControls()
            self._step_count = 0
            print("[AirSimBridge] Vehicle reset.")
        except Exception as exc:
            logger.warning("AirSim reset failed: %s", exc)

    def disconnect(self) -> None:
        """Gracefully disconnect from the AirSim simulator.

        Stops the vehicle, releases API control.  Safe to call
        multiple times.
        """
        if not self._connected:
            return
        try:
            self._controls.throttle = 0.0
            self._controls.steering = 0.0
            self._controls.brake = 0.0
            self._client.setCarControls(self._controls)
            time.sleep(0.3)
            self._client.enableApiControl(False)
            self._connected = False
            print(f"[AirSimBridge] Disconnected after {self._step_count} steps.")
        except Exception as exc:
            logger.warning("AirSim disconnect failed: %s", exc)
            self._connected = False
