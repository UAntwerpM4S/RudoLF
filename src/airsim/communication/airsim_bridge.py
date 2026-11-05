"""
Connection manager for CosysAirSim running inside Unreal Engine 5.

UE5 detection via a TCP port probe, connection retry with exponential backoff,
periodic health checks and pass-through control: set_controls() forwards
pre-mapped CarControls unchanged. Input mapping is done by AirSimEnv and
airsim.inference.run_inference.

Coordinate frames:
    * Framework: x = east, y = north, heading from east (CCW+).
    * AirSim (NED): x = north, y = east, yaw from north (CW+).
    * AirSim_yaw = pi/2 - ship_heading.

Example:

    >>> from airsim.config.airsim_config import AirSimConfig
    >>> bridge = AirSimBridge(AirSimConfig())
    >>> bridge.is_connected
    True
    >>> bridge.health_check()
    True
    >>> bridge.disconnect()
"""

from __future__ import annotations

import io
import os
import csv
import math
import time
import socket
import logging

from collections import deque
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from contextlib import redirect_stdout, redirect_stderr
from fw.common.airsim.tools.angles import (
    normalize_angle as _normalize_angle,
    yaw_from_quaternion as _yaw_from_quaternion,
)

from airsim.config.airsim_config import (
    AirSimConfig,
    DEFAULT_AIRSIM_SAVE_SLOT,
    DEFAULT_AIRSIM_SIMULATION_SPEED,
)

logger = logging.getLogger(__name__)


def _silence_library_output(callback) -> Any:
    """
    Run a callback with stdout/stderr suppressed (Cosys-AirSim prints banners/warnings).

    Args:
        callback: Zero-argument callable to execute.

    Returns:
        Any: The callback return value.
    """

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return callback()


def _roll_pitch_from_quaternion(q) -> tuple[float, float]:
    """
    Extract roll and pitch from an AirSim quaternion.

    ZYX decomposition; yaw is left to the caller so UE5's live roll/pitch can be
    kept while replacing yaw.

    Args:
        q: A `cosysairsim.Quaternionr` instance with attributes
            `w_val`, `x_val`, `y_val`, and `z_val`.

    Returns:
        A tuple `(roll, pitch)` in radians.
    """

    sinr_cosp = 2.0 * (q.w_val * q.x_val + q.y_val * q.z_val)
    cosr_cosp = 1.0 - 2.0 * (q.x_val ** 2 + q.y_val ** 2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w_val * q.y_val - q.z_val * q.x_val)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    return roll, pitch


def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """
    Create a quaternion from ZYX Euler angles.

    Args:
        roll: Roll angle in radians.
        pitch: Pitch angle in radians.
        yaw: Yaw angle in radians.

    Returns:
        A tuple `(x, y, z, w)` quaternion components.
    """

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def is_ue5_reachable(ip: str = "", port: int = 41451, timeout: float = 2.0) -> bool:
    """
    Check whether the AirSim RPC port accepts a TCP connection (no cosysairsim needed).

    Args:
        ip: AirSim server IP.  Empty string = `127.0.0.1`.
        port: AirSim RPC port.
        timeout: Socket connect timeout in seconds.

    Returns:
        `True` if the port is open, `False` otherwise.
    """

    host = ip if ip else "127.0.0.1"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


# -- AirSimBridge --


class AirSimBridge:
    """
    Connection manager and pass-through control adapter for CosysAirSim.

    Attributes:
        config: The :class:`AirSimConfig` used to create this bridge.
    """

    def __init__(self, config: Optional[AirSimConfig] = None) -> None:
        """
        Initialize the AirSim bridge and connect to the simulator.

        Probes the RPC port, then connects with retry and enables API control.
        If cosysairsim is missing, UE5 is unreachable or all retries fail, the
        bridge stays disconnected and control methods become no-ops.

        Args:
            config: Connection parameters.  Defaults to
                `AirSimConfig()` (localhost:41451).
        """

        self.config: AirSimConfig = config or AirSimConfig()
        self._connected: bool = False
        self._client: Any = None
        self._controls: Any = None
        self._airsim: Any = None
        self._last_health_check: float = time.monotonic()
        self._step_count: int = 0
        self._sensor_buffer: deque = deque(maxlen=1000)
        self._simulation_speed: float = 1.0
        self._rendering_enabled: Optional[bool] = None
        self.coord_transform = None
        self.render_z: float = 0.0

        if not self.config.use_airsim:
            logger.info("AirSim bridge disabled via config (use_airsim=False).")
            return

        # -- Step 1: lightweight port probe --
        if not is_ue5_reachable(
            self.config.ip, self.config.port, timeout=self.config.timeout
        ):
            logger.warning(
                "UE5 not reachable at %s:%d - bridge disabled.",
                self.config.resolved_ip,
                self.config.port,
            )
            return

        # -- Step 2: import cosysairsim --
        try:
            import cosysairsim as airsim

            self._airsim = airsim
        except ImportError:
            logger.warning("cosysairsim package not installed - bridge disabled.")
            return

        # -- Step 3: connect with retry + exponential backoff --
        self._connect_with_retry(airsim)

    def _connect_with_retry(self, airsim) -> None:
        """
        Attempt to connect `max_retries + 1` times with backoff.

        The wait doubles after each failure (max 30 s). If all attempts fail,
        the error is logged (not raised) and the bridge stays disconnected.

        Args:
            airsim: The imported `cosysairsim` module.
        """

        attempts = self.config.max_retries + 1  # first attempt + retries
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                self._client = airsim.CarClient(
                    ip=self.config.ip, port=self.config.port,
                    timeout_value=self.config.timeout,
                )
                _silence_library_output(self._client.confirmConnection)
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
                        "Connection attempt %d/%d failed: %s - retrying in %ds...",
                        attempt + 1, attempts, exc, backoff,
                    )
                    time.sleep(backoff)

        # All attempts exhausted
        logger.error(
            "Failed to connect to AirSim at %s:%d after %d attempts: %s",
            self.config.resolved_ip, self.config.port, attempts, last_error,
        )

    # -- Properties --

    @property
    def is_connected(self) -> bool:
        """
        Whether the bridge currently has an active AirSim connection.
        """

        return self._connected

    @property
    def client(self) -> Any:
        """
        Underlying CarClient, for advanced state queries.

        Raises:
            ConnectionError: If the bridge is not currently connected to AirSim.
        """

        if not self._ensure_connected():
            raise ConnectionError("Cannot access AirSim client: bridge disconnected.")
        return self._client

    def get_client(self) -> Any:
        """
        Method-style access to `client`.

        Returns:
            Any: Active AirSim `CarClient` instance.

        Raises:
            ConnectionError: If the bridge is currently disconnected.
        """

        return self.client

    @property
    def airsim(self) -> Any:
        """
        The imported cosysairsim module.

        Returns:
            Any: Imported `cosysairsim` module used by the bridge.

        Raises:
            ConnectionError: If the AirSim module was never initialized.
        """

        if self._airsim is None:
            raise ConnectionError("Cannot access AirSim module: bridge not initialized.")
        return self._airsim

    @property
    def current_controls(self) -> Any:
        """
        Most recently forwarded AirSim control object, or None.
        """

        return self._controls

    # -- Health check --

    def health_check(self) -> bool:
        """
        Verify the AirSim connection is still alive.

        Pings with `getCarState()`; on failure, flushes the CSV buffer and reconnects.

        Returns:
            `True` if the connection is healthy or re-established,
            `False` otherwise.
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
            
            # Release state before reconnecting.
            self._client = None
            self._controls = None
            self._rendering_enabled = None
            logger.warning(
                "Flushing CSV telemetry buffer before reconnect at %s - "
                "a gap will exist in the output file from this point.",
                datetime.now(timezone.utc).isoformat(),
            )
            self._flush_csv()
            
            if self._airsim:
                self._connect_with_retry(self._airsim)
            
            return self._connected

    def _ensure_connected(self) -> bool:
        """
        Run a health check if `health_check_interval` has elapsed.

        Returns:
            False if disconnected or the health check failed, else True.
        """

        if not self._connected:
            return False
        elapsed = time.monotonic() - self._last_health_check
        if elapsed >= self.config.health_check_interval:
            return self.health_check()
        return True

    # -- Control methods --

    def set_controls(self, controls: Any) -> None:
        """
        Forward pre-mapped controls to AirSim unchanged (mapping is done upstream).

        Args:
            controls: A populated `cosysairsim.CarControls` object.
        """

        if not self._ensure_connected():
            return

        try:
            self._controls = controls
            self._client.setCarControls(self._controls)
            self._step_count += 1
        except Exception as exc:
            logger.warning("AirSim set_controls failed: %s", exc)

    def get_car_state(self) -> Any:
        """
        Return the latest AirSim car state.

        Returns:
            Any: Current `CarState` object returned by AirSim.

        Raises:
            ConnectionError: If the bridge is disconnected.
        """

        if not self._ensure_connected():
            raise ConnectionError("Cannot read AirSim car state: bridge disconnected.")
        return self._client.getCarState()

    def get_vehicle_pose(self) -> Any:
        """
        Return the latest AirSim vehicle pose.

        Returns:
            Any: Current AirSim vehicle pose object.

        Raises:
            ConnectionError: If the bridge is disconnected.
        """

        if not self._ensure_connected():
            raise ConnectionError("Cannot read AirSim vehicle pose: bridge disconnected.")
        return self._client.simGetVehiclePose()

    def set_vehicle_pose(self, pose: Any, ignore_collision: bool = True) -> None:
        """
        Set the AirSim vehicle pose.

        Args:
            pose: AirSim `Pose` object to apply.
            ignore_collision: Whether to ignore collisions during teleport.
        """

        if not self._ensure_connected():
            return
        self._client.simSetVehiclePose(pose, ignore_collision=ignore_collision)

    def get_collision_info(self) -> Any:
        """
        Return collision information for the active vehicle.

        Returns:
            Any: AirSim collision-info payload.

        Raises:
            ConnectionError: If the bridge is disconnected.
        """

        if not self._ensure_connected():
            raise ConnectionError("Cannot read AirSim collision info: bridge disconnected.")
        return self._client.simGetCollisionInfo()

    def set_pause(self, is_paused: bool) -> None:
        """
        Pause or unpause the simulation.

        Args:
            is_paused: Desired paused state.
        """

        if not self._ensure_connected():
            return
        self._client.simPause(is_paused)

    def continue_for_frames(self, frames: int) -> None:
        """
        Advance the simulation by a fixed number of frames.

        Args:
            frames: Number of frames to advance while paused.
        """

        if not self._ensure_connected():
            return
        self._client.simContinueForFrames(frames)

    def continue_for_time(self, seconds: float) -> None:
        """
        Advance the simulation by a fixed amount of simulation time.

        Args:
            seconds: Number of simulation seconds to advance while paused.
        """

        if not self._ensure_connected():
            return
        self._client.simContinueForTime(float(seconds))

    def is_paused(self) -> bool:
        """
        Whether the UE5 simulation is paused (False when disconnected).
        """

        if not self._ensure_connected():
            return False
        return bool(self._client.simIsPause())

    def set_simulation_speed(self, simulation_speed: float) -> None:
        """
        Set the UE5 simulation speed multiplier.

        Args:
            simulation_speed: New UE5 simulation-speed multiplier.
        """

        if not self._ensure_connected():
            return
        self._simulation_speed = float(simulation_speed)
        self._client.simSetSimulationSpeed(simulation_speed)

    def set_rendering_enabled(self, is_enabled: bool, force: bool = False) -> bool:
        """
        Toggle UE5 scene rendering via the ShowFlag.Rendering console command
        (AirSim has no rendering API); physics keeps running.

        Args:
            is_enabled: True to show rendering, False to hide it.
            force: Issue the command even if the state already seems to match.

        Returns:
            True if the command was accepted or the state already matched.
        """

        if not self._ensure_connected():
            return False

        desired_state = bool(is_enabled)
        if not force and self._rendering_enabled is desired_state:
            return True

        if not hasattr(self._client, "simRunConsoleCommand"):
            logger.warning(
                "AirSim client does not support simRunConsoleCommand; cannot toggle rendering."
            )
            return False

        command = f"ShowFlag.Rendering {1 if desired_state else 0}"
        try:
            result = self._client.simRunConsoleCommand(command)
        except Exception as exc:
            logger.warning(
                "Failed to %s UE5 rendering via console command '%s': %s",
                "enable" if desired_state else "disable",
                command,
                exc,
            )
            return False

        if result is False:
            logger.warning(
                "UE5 rejected console command '%s' while toggling rendering.",
                command,
            )
            return False

        self._rendering_enabled = desired_state
        logger.info(
            "UE5 rendering %s via console command '%s'.",
            "enabled" if desired_state else "disabled",
            command,
        )
        return True

    def flush_persistent_markers(self) -> None:
        """
        Clear all persistent debug markers from UE5.
        """

        if not self._ensure_connected():
            return
        self._client.simFlushPersistentMarkers()

    def plot_line_strip(
        self,
        points: Any,
        color_rgba: list[float],
        thickness: float,
        duration: float = 1.0,
        is_persistent: bool = False,
    ) -> None:
        """
        Draw a line strip in UE5.

        Args:
            points: Sequence of AirSim `Vector3r` points.
            color_rgba: RGBA color values in the range `[0, 1]`.
            thickness: Rendered line thickness.
            duration: Seconds the line is shown in the 3D world when not
                persistent (minimap/world-map markers stay regardless).
            is_persistent: Whether the line should persist. Default False,
                as long persistent strips cost FPS.
        """

        if not self._ensure_connected():
            return
        self._client.simPlotLineStrip(
            points,
            color_rgba=color_rgba,
            thickness=thickness,
            duration=duration,
            is_persistent=is_persistent,
        )

    def plot_points(
        self,
        points: Any,
        color_rgba: list[float],
        size: float,
        duration: float = 1.0,
        is_persistent: bool = False,
    ) -> None:
        """
        Draw points in UE5.

        Args:
            points: Sequence of AirSim `Vector3r` points.
            color_rgba: RGBA color values in the range `[0, 1]`.
            size: Rendered point size.
            duration: Seconds the markers are shown in the 3D world when
                not persistent.
            is_persistent: Whether the points should persist in the scene.
        """

        if not self._ensure_connected():
            return
        self._client.simPlotPoints(
            points,
            color_rgba=color_rgba,
            size=size,
            duration=duration,
            is_persistent=is_persistent,
        )

    def get_camera_info(self, camera_name: str, vehicle_name: str = "") -> Any:
        """
        Return camera extrinsics/intrinsics from AirSim.

        Args:
            camera_name: Camera identifier configured in AirSim settings.
            vehicle_name: Optional AirSim vehicle name.

        Returns:
            Camera info object from `simGetCameraInfo`.

        Raises:
            ConnectionError: If the bridge is not connected.
        """

        if not self._ensure_connected():
            raise ConnectionError("Cannot read camera info: bridge disconnected.")
        return self._client.simGetCameraInfo(camera_name, vehicle_name=vehicle_name)

    def set_camera_pose(self, camera_name: str, pose: Any, vehicle_name: str = "") -> None:
        """
        Set a camera pose in AirSim.

        Args:
            camera_name: Camera identifier configured in AirSim settings.
            pose: `airsim.Pose` instance in AirSim world coordinates.
            vehicle_name: Optional AirSim vehicle name.
        """

        if not self._ensure_connected():
            return
        self._client.simSetCameraPose(camera_name, pose, vehicle_name=vehicle_name)

    def set_camera_fov(self, camera_name: str, fov_degrees: float, vehicle_name: str = "") -> None:
        """
        Set camera field-of-view in degrees.

        Args:
            camera_name: Camera identifier configured in AirSim settings.
            fov_degrees: Horizontal field-of-view in degrees.
            vehicle_name: Optional AirSim vehicle name.
        """

        if not self._ensure_connected():
            return
        self._client.simSetCameraFov(camera_name, float(fov_degrees), vehicle_name=vehicle_name)

    def get_camera_image(
        self,
        camera_name: str,
        image_type: Any,
        vehicle_name: str = "",
        annotation_name: str = "",
    ) -> Any:
        """
        Request one image frame from a camera.

        Args:
            camera_name: Camera identifier configured in AirSim settings.
            image_type: `airsim.ImageType` enum value.
            vehicle_name: Optional AirSim vehicle name.
            annotation_name: Optional annotation channel name.

        Returns:
            Raw return value from `simGetImage` (typically PNG bytes or base64 string).

        Raises:
            ConnectionError: If the bridge is not connected.
        """

        if not self._ensure_connected():
            raise ConnectionError("Cannot read camera image: bridge disconnected.")
        return self._client.simGetImage(
            camera_name,
            image_type,
            vehicle_name=vehicle_name,
            annotation_name=annotation_name,
        )

    def get_camera_images(self, requests: list[Any], vehicle_name: str = "") -> list[Any]:
        """
        Request a batched set of camera frames from AirSim.

        Args:
            requests: List of `airsim.ImageRequest` objects.
            vehicle_name: Optional AirSim vehicle name.

        Returns:
            List of image responses from `simGetImages`.

        Raises:
            ConnectionError: If the bridge is not connected.
        """

        if not self._ensure_connected():
            raise ConnectionError("Cannot read camera images: bridge disconnected.")
        return self._client.simGetImages(requests, vehicle_name=vehicle_name)

    def set_pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        preserve_current_z: bool = False,
        preserve_roll_pitch: bool = False,
        _current_pose: Any = None,
    ) -> None:
        """
        Teleport the vehicle (used when UE5 only renders an external trajectory).

        Args:
            x: X coordinate in AirSim NED frame.
            y: Y coordinate in AirSim NED frame.
            z: Z coordinate in AirSim NED frame.
            yaw: Yaw angle in radians in AirSim NED frame.
            preserve_current_z: Keep the current UE5 heave instead of `z`.
            preserve_roll_pitch: Keep the current UE5 roll and pitch; replace only yaw.
            _current_pose: Optional pre-fetched `simGetVehiclePose()` result, to
                save an RPC per sub-step in interpolation loops.
        """

        if not self._ensure_connected():
            return
            
        try:
            target_z = z
            quat_x, quat_y, quat_z, quat_w = _quaternion_from_euler(0.0, 0.0, yaw)
            current_pose = _current_pose
            if preserve_current_z and current_pose is None:
                current_pose = self._client.simGetVehiclePose()
            if preserve_current_z:
                target_z = float(current_pose.position.z_val)

            if preserve_roll_pitch and current_pose is None:
                current_pose = self._client.simGetVehiclePose()
            if preserve_roll_pitch:
                current_roll, current_pitch = _roll_pitch_from_quaternion(current_pose.orientation)
                quat_x, quat_y, quat_z, quat_w = _quaternion_from_euler(current_roll, current_pitch, yaw)

            pose = self._airsim.Pose(
                self._airsim.Vector3r(x, y, target_z),
                self._airsim.Quaternionr(quat_x, quat_y, quat_z, quat_w)
            )
            self._client.simSetVehiclePose(pose, ignore_collision=True)
        except Exception as exc:
            logger.warning("AirSim set_pose failed: %s", exc)

    def set_object_pose(
        self,
        object_name: str,
        x: float,
        y: float,
        z: float,
        yaw: float,
    ) -> None:
        """
        Move a named UE5 actor via simSetObjectPose, without vehicle physics (no
        buoyancy or ripples). Preferred when UE5 only renders externally simulated
        motion.

        Args:
            object_name: Exact actor name as shown in the UE5 Outliner.
            x: X coordinate in AirSim NED frame.
            y: Y coordinate in AirSim NED frame.
            z: Z coordinate in AirSim NED frame.
            yaw: Yaw angle in radians in AirSim NED frame.
        """

        if not self._ensure_connected():
            return
        try:
            quat_x, quat_y, quat_z, quat_w = _quaternion_from_euler(0.0, 0.0, yaw)
            pose = self._airsim.Pose(
                self._airsim.Vector3r(x, y, z),
                self._airsim.Quaternionr(quat_x, quat_y, quat_z, quat_w),
            )
            self._client.simSetObjectPose(object_name, pose, teleport=True)
        except Exception as exc:
            logger.warning("AirSim set_object_pose('%s') failed: %s", object_name, exc)

    def set_kinematics(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        linear_velocity_x: float = 0.0,
        linear_velocity_y: float = 0.0,
        linear_velocity_z: float = 0.0,
        angular_velocity_x: float = 0.0,
        angular_velocity_y: float = 0.0,
        angular_velocity_z: float = 0.0,
        preserve_current_z: bool = False,
        preserve_roll_pitch: bool = False,
    ) -> None:
        """
        Set vehicle pose and kinematics in UE5 for smoother visual playback.

        Args:
            x: X coordinate in AirSim NED frame.
            y: Y coordinate in AirSim NED frame.
            z: Z coordinate in AirSim NED frame.
            yaw: Yaw angle in radians in AirSim NED frame.
            linear_velocity_x: X component of linear velocity.
            linear_velocity_y: Y component of linear velocity.
            linear_velocity_z: Z component of linear velocity.
            angular_velocity_x: X component of angular velocity.
            angular_velocity_y: Y component of angular velocity.
            angular_velocity_z: Z component of angular velocity.
            preserve_current_z: Whether to preserve the current UE5 heave.
            preserve_roll_pitch: Whether to preserve current UE5 roll/pitch.
        """

        if not self._ensure_connected():
            return

        try:
            target_z = z
            quat_x, quat_y, quat_z, quat_w = _quaternion_from_euler(0.0, 0.0, yaw)
            current_pose = None
            if preserve_current_z:
                current_pose = self._client.simGetVehiclePose()
                target_z = float(current_pose.position.z_val)

            if preserve_roll_pitch:
                if current_pose is None:
                    current_pose = self._client.simGetVehiclePose()
                current_roll, current_pitch = _roll_pitch_from_quaternion(current_pose.orientation)
                quat_x, quat_y, quat_z, quat_w = _quaternion_from_euler(current_roll, current_pitch, yaw)

            if hasattr(self._client, "simSetKinematics"):
                state = self._airsim.KinematicsState()
                state.position = self._airsim.Vector3r(float(x), float(y), float(target_z))
                state.orientation = self._airsim.Quaternionr(quat_x, quat_y, quat_z, quat_w)
                state.linear_velocity = self._airsim.Vector3r(
                    float(linear_velocity_x),
                    float(linear_velocity_y),
                    float(linear_velocity_z),
                )
                state.angular_velocity = self._airsim.Vector3r(
                    float(angular_velocity_x),
                    float(angular_velocity_y),
                    float(angular_velocity_z),
                )
                state.linear_acceleration = self._airsim.Vector3r(0.0, 0.0, 0.0)
                state.angular_acceleration = self._airsim.Vector3r(0.0, 0.0, 0.0)
                self._client.simSetKinematics(state, ignore_collision=True)
                return

            self.set_pose(
                x,
                y,
                target_z,
                yaw,
                preserve_current_z=False,
                preserve_roll_pitch=False,
            )
        except Exception as exc:
            logger.warning("AirSim set_kinematics failed: %s", exc)

    def get_state(self) -> Dict[str, Any]:
        """
        Read vehicle telemetry from AirSim and buffer it for CSV logging.

        Returns:
            Telemetry dict; {} when disconnected; on a read error, a dict with
            'state_read_failed', 'error', 'step' and 'wall_time_utc'.
        """

        if not self._ensure_connected():
            return {}
        try:
            s = self._client.getCarState()
            kinematics = s.kinematics_estimated
            pos = kinematics.position
            lin_vel = kinematics.linear_velocity
            ang_vel = kinematics.angular_velocity
            lin_acc = getattr(kinematics, "linear_acceleration", None)
            
            quat = kinematics.orientation
            roll, pitch = _roll_pitch_from_quaternion(quat)
            yaw = _yaw_from_quaternion(quat)
            yaw_norm = _normalize_angle(yaw)
            controls = self._controls

            state_dict = {
                "step": self._step_count,
                "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "timestamp": s.timestamp,
                "speed": s.speed,
                "gear": s.gear,
                "pos_x": pos.x_val,
                "pos_y": pos.y_val,
                "pos_z": pos.z_val,
                "vel_x": lin_vel.x_val,
                "vel_y": lin_vel.y_val,
                "vel_z": lin_vel.z_val,
                "accel_x": float(getattr(lin_acc, "x_val", 0.0)),
                "accel_y": float(getattr(lin_acc, "y_val", 0.0)),
                "accel_z": float(getattr(lin_acc, "z_val", 0.0)),
                "ang_vel_x": ang_vel.x_val,
                "ang_vel_y": ang_vel.y_val,
                "ang_vel_z": ang_vel.z_val,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw_norm,
                "control_steering": float(getattr(controls, "steering", 0.0) if controls is not None else 0.0),
                "control_throttle": float(getattr(controls, "throttle", 0.0) if controls is not None else 0.0),
                "control_brake": float(getattr(controls, "brake", 0.0) if controls is not None else 0.0),
                "simulation_speed": self._simulation_speed,
            }

            self._sensor_buffer.append(state_dict)

            # Flush when the buffer is full.
            if len(self._sensor_buffer) >= self._sensor_buffer.maxlen:
                self._flush_csv()

            return state_dict
        except Exception as exc:
            logger.warning("AirSim get_state failed: %s", exc)
            return {
                "state_read_failed": True,
                "error": str(exc),
                "step": self._step_count,
                "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            }

    def _flush_csv(self) -> None:
        """
        Append buffered telemetry to the CSV file (no-op if disabled or empty).
        """

        if not self.config.csv_log_path or not self._sensor_buffer:
            return

        try:
            csv_dir = os.path.dirname(self.config.csv_log_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            file_exists = os.path.isfile(self.config.csv_log_path)
            with open(self.config.csv_log_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self._sensor_buffer[0].keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerows(list(self._sensor_buffer))
            self._sensor_buffer.clear()
        except Exception as exc:
            logger.warning("Failed to dump sensor buffer to CSV: %s", exc)

    def _snapshot_vehicle_position(self) -> Optional[tuple[float, float, float]]:
        """
        Return the vehicle position in AirSim coordinates, or None if unreadable.
        """

        if self._client is None:
            return None
        try:
            pose = self._client.simGetVehiclePose()
        except Exception:
            return None
        return (
            float(pose.position.x_val),
            float(pose.position.y_val),
            float(pose.position.z_val),
        )

    @staticmethod
    def _position_delta_m(
        before: Optional[tuple[float, float, float]],
        after: Optional[tuple[float, float, float]],
    ) -> Optional[float]:
        """
        Euclidean distance in m between two positions, or None if either is missing.
        """

        if before is None or after is None:
            return None
        dx = float(after[0] - before[0])
        dy = float(after[1] - before[1])
        dz = float(after[2] - before[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _reconnect_after_world_reload(self, reconnect_delay: float) -> None:
        """
        Reset client state and reconnect after a world reload (attempt).

        Args:
            reconnect_delay: Delay in seconds before attempting reconnection.

        Raises:
            ConnectionError: If reconnection fails.
        """

        time.sleep(reconnect_delay)
        self._connected = False
        self._client = None
        self._controls = None
        self._simulation_speed = 1.0
        self._rendering_enabled = None
        if self._airsim is None:
            raise ConnectionError("Cannot reconnect: cosysairsim module not available.")
        self._connect_with_retry(self._airsim)
        if not self._connected:
            raise ConnectionError("AirSimBridge failed to reconnect after level reload.")

    def _fallback_reset_after_unconfirmed_reload(self, settle_s: float = 1.0) -> None:
        """
        Reset the vehicle in-session when a save-slot reload could not be confirmed.

        Args:
            settle_s: Seconds to wait after the fallback reset.

        Raises:
            ConnectionError: If fallback reset cannot be applied.
        """

        if not self._connected or self._client is None or self._airsim is None:
            raise ConnectionError(
                "Cannot apply fallback reset: AirSim bridge is not connected."
            )

        logger.warning(
            "Falling back to an in-session vehicle reset because save-slot reload "
            "was not clearly observed."
        )
        try:
            self._client.reset()
            time.sleep(max(float(settle_s), 0.0))
            self._client.enableApiControl(True)
            self._controls = self._airsim.CarControls()
            self._step_count = 0
        except Exception as exc:
            raise ConnectionError(
                "Fallback reset failed after an unconfirmed save-slot reload."
            ) from exc

    def load_world_save(
        self,
        save_slot: str = DEFAULT_AIRSIM_SAVE_SLOT,
        reload_timeout: float = 90.0,
        poll_interval: float = 0.5,
        reconnect_delay: float = 1.5,
        max_reload_attempts: int = 2,
        min_observed_pose_shift_m: float = 1.0,
        fallback_reset_settle_s: float = 1.0,
    ) -> None:
        """
        Load a UE5 save slot and reconnect after the level reload completes.

        A reload is confirmed by the RPC port going down or by a vehicle pose
        shift of at least `min_observed_pose_shift_m`. Some sessions do not drop
        the RPC connection, so an unconfirmed reload is retried up to
        `max_reload_attempts` times, after which the vehicle is reset in-session.

        Raises:
            ConnectionError: If disconnected or reconnection/fallback fails.
            TimeoutError: If UE5 does not come back within `reload_timeout` s.
        """

        if not self._ensure_connected():
            raise ConnectionError("Cannot load world save: bridge disconnected.")

        attempts = max(1, int(max_reload_attempts))
        for attempt_index in range(attempts):
            logger.info(
                "Loading world save '%s' (attempt %d/%d)...",
                save_slot,
                attempt_index + 1,
                attempts,
            )
            position_before_reload = self._snapshot_vehicle_position()
            self._client.simSetSaveSlot(save_slot)
            self._client.simRequestLoad()

            start = time.monotonic()
            went_down = False
            while time.monotonic() - start < 10.0:
                if not is_ue5_reachable(self.config.ip, self.config.port):
                    went_down = True
                    logger.info("UE5 RPC port went offline - level is reloading...")
                    break
                time.sleep(poll_interval)

            if not went_down:
                logger.warning(
                    "UE5 RPC port never went down after save-load request on attempt %d/%d.",
                    attempt_index + 1,
                    attempts,
                )

            logger.info(
                "Waiting for UE5 to finish reloading (timeout=%ds)...",
                int(reload_timeout),
            )
            while time.monotonic() - start < reload_timeout:
                if is_ue5_reachable(self.config.ip, self.config.port):
                    break
                time.sleep(poll_interval)
            else:
                raise TimeoutError(
                    f"UE5 did not finish reloading within {reload_timeout:.0f} s."
                )

            self._reconnect_after_world_reload(reconnect_delay)
            position_after_reload = self._snapshot_vehicle_position()
            observed_shift_m = self._position_delta_m(
                position_before_reload,
                position_after_reload,
            )

            if went_down or observed_shift_m is None:
                return
            if observed_shift_m >= float(min_observed_pose_shift_m):
                logger.info(
                    "Observed %.2f m pose shift after save-load request; accepting reload.",
                    observed_shift_m,
                )
                return

            logger.warning(
                "Save-load attempt %d/%d did not clearly reload the session "
                "(RPC stayed up, pose shift %.2f m).",
                attempt_index + 1,
                attempts,
                observed_shift_m,
            )

        self._fallback_reset_after_unconfirmed_reload(
            settle_s=fallback_reset_settle_s
        )

    def prepare_model_visualisation(
        self,
        save_slot: str = DEFAULT_AIRSIM_SAVE_SLOT,
        simulation_speed: float = DEFAULT_AIRSIM_SIMULATION_SPEED,
        draw_reference: bool = True,
        reference_duration: float = -1.0,
    ):
        """
        Prepare UE5 for framework-model visualization.

        Args:
            save_slot: UE5 save slot to load before playback.
            simulation_speed: Simulation-speed multiplier to apply.
            draw_reference: Whether to draw the calibration reference overlay.
            reference_duration: Passed to the calibration draw call; unused
                for persistent markers (the default).

        Returns:
            Any: Loaded coordinate-transform object used for replay.
        """

        from fw.common.airsim.tools.calibrate_coords import load_saved_calibration_reference

        self.load_world_save(save_slot=save_slot)
        self.set_rendering_enabled(True)
        self.set_simulation_speed(simulation_speed)
        spawn_pose = self._client.simGetVehiclePose()
        self.render_z = float(spawn_pose.position.z_val)
        self.coord_transform, _ = load_saved_calibration_reference(
            self._client,
            draw_reference=draw_reference,
            duration=reference_duration,
        )

        # Park the vehicle so it does not drift (and disturb the water) before
        # the first framework pose update.
        try:
            _park = self._airsim.CarControls()
            _park.throttle = 0.0
            _park.steering = 0.0
            _park.brake = 1.0
            self._client.setCarControls(_park)
        except Exception:
            pass

        return self.coord_transform

    def redraw_calibration_reference(
        self,
        duration: float = 3600.0,
        ship_pos: tuple[float, float] | None = None,
        target_pos: tuple[float, float] | None = None,
    ) -> None:
        """
        Redraw the calibration overlay (polygons, trajectory, start/target dots).

        Reloads the saved calibration so any drift correction is re-applied,
        then renders the overlay persistently. Markers stay until
        `simFlushPersistentMarkers` is called or the level is reloaded.

        Args:
            duration: Passed to the draw call; unused for persistent markers.
            ship_pos: Optional framework start position for the blue marker
                and trajectory slicing.
            target_pos: Optional Framework target position for the red marker.
        """

        if not self._ensure_connected():
            return

        from fw.common.airsim.tools.calibrate_coords import load_saved_calibration_reference

        self.coord_transform, _ = load_saved_calibration_reference(
            self._client,
            draw_reference=True,
            ship_pos=ship_pos,
            target_pos=target_pos,
            duration=duration,
        )

    def reset(self) -> None:
        """
        Reset the AirSim vehicle and re-enable API control.

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
        """
        Gracefully disconnect from the AirSim simulator.

        Flushes the CSV buffer, brakes the vehicle, re-enables rendering and
        releases API control. Safe to call multiple times.
        """

        if not self._connected:
            return
            
        logger.info("Disconnecting from AirSim...")
        self._flush_csv()
        
        try:
            stop_controls = self._airsim.CarControls()
            stop_controls.throttle = 0.0
            stop_controls.steering = 0.0
            stop_controls.brake = 1.0
            
            self._client.setCarControls(stop_controls)
            time.sleep(0.3)
            self.set_rendering_enabled(True)
            self._client.enableApiControl(False)
            self._connected = False
            
            logger.info(
                "Successfully disconnected from AirSim after %d steps.",
                self._step_count,
            )
        except Exception as exc:
            logger.warning("AirSim disconnect failed: %s", exc)
            self._connected = False
