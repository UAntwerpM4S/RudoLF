"""CosysAirSim bridge - robust connection manager for UE5.

This bridge manages the network connection to CosysAirSim running
inside Unreal Engine 5.  It provides:

* **UE5 auto-detection** via a lightweight TCP port probe.
* **Retry with exponential backoff** for resilient startup.
* **Periodic health-checks** to detect stale connections.
* **Pass-through control** - ``set_controls()`` forwards pre-mapped
  ``CarControls`` to AirSim without any PID processing.

Input mapping (Fossen model → UE5 controls) is handled by
:class:`~airsim.environments.airsim_env.AirSimEnv` and the
:mod:`airsim.inference.run_inference` module.  **PID control is not
used anywhere in this bridge.**

Coordinate frames:
    * **Ship (Fossen):** x = east, y = north, heading from east (CCW+).
    * **AirSim (NED):** x = north, y = east, yaw from north (CW+).
    * **Transform:** ``AirSim_yaw = π/2 - ship_heading``.

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
import csv
import os
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from airsim.config.airsim_config import AirSimConfig

logger = logging.getLogger(__name__)


# ── Utility helpers ────────────────────────────────────────


def _normalize_angle(a: float) -> float:
    """Wrap an angle to the interval [-π, π].

    Args:
        a: Angle in radians.

    Returns:
        The equivalent angle in the range [-π, π].
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

    Performs a lightweight TCP connect - does **not** require the
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
        self._sensor_buffer: deque = deque(maxlen=1000)

        if not self.config.use_airsim:
            logger.info("AirSim bridge disabled via config (use_airsim=False).")
            return

        # ── Step 1: lightweight port probe ──
        if not is_ue5_reachable(
            self.config.ip, self.config.port, timeout=self.config.timeout
        ):
            logger.warning(
                "UE5 not reachable at %s:%d - bridge disabled.",
                self.config.resolved_ip,
                self.config.port,
            )
            return

        # ── Step 2: import cosysairsim ──
        try:
            import cosysairsim as airsim

            self._airsim = airsim
        except ImportError:
            logger.warning("cosysairsim package not installed - bridge disabled.")
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
                return

            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1:
                    backoff = min(2 ** attempt, 30)
                    logger.warning(
                        "Connection attempt %d/%d failed: %s - retrying in %ds…",
                        attempt + 1, attempts, exc, backoff,
                    )
                    time.sleep(backoff)

        # All attempts exhausted
        logger.error(
            "Failed to connect to AirSim at %s:%d after %d attempts: %s",
            self.config.resolved_ip, self.config.port, attempts, last_error,
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
        If the connection is lost, it will attempt to reconnect.

        Returns:
            ``True`` if the connection is healthy or re-established,
            ``False`` otherwise.
        """
        if not self._connected:
            return False
        try:
            self._client.getCarState()
            self._last_health_check = time.monotonic()
            return True
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
            logger.info("Connection lost, attempting to reconnect...")
            self._connected = False
            
            # T-005: State cleanup and resource release
            self._client = None
            self._controls = None
            self._flush_csv()
            
            # T-005: Reconnect
            if self._airsim:
                self._connect_with_retry(self._airsim)
            
            return self._connected

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

    def set_controls(self, controls: Any) -> None:
        """Forward pre-mapped control commands directly to AirSim.

        This acts as a pure middleware pass-through. Input mapping
        must be handled by upper layers (e.g., inference module).

        Args:
            controls: A populated ``cosysairsim.CarControls`` object.
        """
        if not self._ensure_connected():
            return

        try:
            self._controls = controls
            self._client.setCarControls(self._controls)
            self._step_count += 1
        except Exception as exc:
            logger.warning("AirSim set_controls failed: %s", exc)

    def set_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        """Forcibly set the vehicle's position and orientation in UE5.
        
        Useful for visualization where another physics engine (e.g. Fossen)
        computes the trajectory and we only use UE5 for rendering.
        
        Args:
            x, y, z: Coordinates in AirSim NED frame.
            yaw: Yaw angle in radians in AirSim NED frame.
        """
        if not self._ensure_connected():
            return
            
        try:
            pose = self._airsim.Pose(
                self._airsim.Vector3r(x, y, z),
                self._airsim.Quaternionr(0, 0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
            )
            self._client.simSetVehiclePose(pose, ignore_collision=True)
        except Exception as exc:
            logger.warning("AirSim set_pose failed: %s", exc)

    def get_state(self) -> Dict[str, Any]:
        """Retrieve the full vehicle telemetry state from AirSim.

        Extracts kinematics (position, velocity, orientation) and
        validates the Euler yaw angle conversion.

        Returns:
            A dictionary containing full telemetry when connected.
            Empty dict otherwise.
        """
        if not self._ensure_connected():
            return {}
        try:
            s = self._client.getCarState()
            kinematics = s.kinematics_estimated
            pos = kinematics.position
            lin_vel = kinematics.linear_velocity
            ang_vel = kinematics.angular_velocity
            
            # T-007: Quaternion to Yaw conversion & validation
            quat = kinematics.orientation
            yaw = _yaw_from_quaternion(quat)
            yaw_norm = _normalize_angle(yaw)

            state_dict = {
                "timestamp": s.timestamp,
                "speed": s.speed,
                "gear": s.gear,
                "pos_x": pos.x_val,
                "pos_y": pos.y_val,
                "pos_z": pos.z_val,
                "vel_x": lin_vel.x_val,
                "vel_y": lin_vel.y_val,
                "vel_z": lin_vel.z_val,
                "ang_vel_x": ang_vel.x_val,
                "ang_vel_y": ang_vel.y_val,
                "ang_vel_z": ang_vel.z_val,
                "yaw": yaw_norm
            }

            # T-008: Buffer sensor data
            self._sensor_buffer.append(state_dict)

            # Flush periodically to avoid large memory buildup if not disconnecting often
            if len(self._sensor_buffer) >= self._sensor_buffer.maxlen:
                self._flush_csv()

            return state_dict
        except Exception as exc:
            logger.warning("AirSim get_state failed: %s", exc)
            return {}

    def _flush_csv(self) -> None:
        """Helper to dump buffered sensor data to a CSV file."""
        if not self.config.csv_log_path or not self._sensor_buffer:
            return

        try:
            file_exists = os.path.isfile(self.config.csv_log_path)
            with open(self.config.csv_log_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self._sensor_buffer[0].keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerows(list(self._sensor_buffer))
            self._sensor_buffer.clear()
        except Exception as exc:
            logger.warning("Failed to dump sensor buffer to CSV: %s", exc)

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
            logger.info("Vehicle reset.")
        except Exception as exc:
            logger.warning("AirSim reset failed: %s", exc)

    def disconnect(self) -> None:
        """Gracefully disconnect from the AirSim simulator.

        Stops the vehicle, releases API control, and flushes CSV buffers.
        Safe to call multiple times.
        """
        if not self._connected:
            return
            
        logger.info("Disconnecting from AirSim...")
        self._flush_csv()
        
        try:
            # Stop the vehicle safely using a fresh CarControls object
            stop_controls = self._airsim.CarControls()
            stop_controls.throttle = 0.0
            stop_controls.steering = 0.0
            stop_controls.brake = 1.0 # apply full brake on disconnect
            
            self._client.setCarControls(stop_controls)
            time.sleep(0.3)
            self._client.enableApiControl(False)
            self._connected = False
            
            logger.info(
                "Successfully disconnected from AirSim after %d steps.",
                self._step_count,
            )
        except Exception as exc:
            logger.warning("AirSim disconnect failed: %s", exc)
            self._connected = False
