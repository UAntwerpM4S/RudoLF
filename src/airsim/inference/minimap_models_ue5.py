"""Model-agnostic visualisation helpers for UE5 replay workflows.

This module exposes two minimap options:
- ``VisualiseModelsMinimap``: existing matplotlib-based minimap process.
- ``UE5TopDownMinimap``: camera-driven top-down view rendered from UE5.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import numpy as np

from airsim.inference.minimap_fossen_airsim import MinimapFossenAirsim

logger = logging.getLogger(__name__)


class VisualiseModelsMinimap(MinimapFossenAirsim):
    """Backward-compatible alias for the process-based matplotlib minimap."""


class UE5TopDownMinimap:
    """Top-down UE5 camera view with optional external preview window.

    The camera is positioned above the current ship pose and pointed down,
    producing a 2D-like map view while still rendering full UE5 3D content.
    """

    def __init__(
        self,
        bridge,
        camera_name: str = "TopDownCam",
        vehicle_name: str = "",
        fov_degrees: float = 38.0,
        height_offset_m: float = 120.0,
        follow_heading: bool = False,
        enable_preview_window: bool = True,
        preview_window_name: str = "UE5 Top-Down Minimap",
        preview_width: int = 520,
        preview_height: int = 520,
        draw_trail: bool = True,
        trail_thickness: float = 8.0,
    ) -> None:
        self.bridge = bridge
        self.camera_name = camera_name
        self.vehicle_name = vehicle_name
        self.fov_degrees = float(fov_degrees)
        self.height_offset_m = float(height_offset_m)
        self.follow_heading = bool(follow_heading)
        self.enable_preview_window = bool(enable_preview_window)
        self.preview_window_name = preview_window_name
        self.preview_width = int(preview_width)
        self.preview_height = int(preview_height)
        self.draw_trail = bool(draw_trail)
        self.trail_thickness = float(trail_thickness)

        self._prev_trail_point = None
        self._cv2 = None

        if self.bridge.is_connected:
            try:
                self.bridge.set_camera_fov(
                    self.camera_name,
                    self.fov_degrees,
                    vehicle_name=self.vehicle_name,
                )
            except Exception as exc:
                logger.warning("Could not set camera FOV for '%s': %s", self.camera_name, exc)

        if self.enable_preview_window:
            try:
                import cv2  # type: ignore

                self._cv2 = cv2
                self._cv2.namedWindow(self.preview_window_name, self._cv2.WINDOW_NORMAL)
                self._cv2.resizeWindow(self.preview_window_name, self.preview_width, self.preview_height)
            except Exception as exc:
                logger.warning("OpenCV preview disabled: %s", exc)
                self._cv2 = None
                self.enable_preview_window = False

    def _quaternion_from_euler(self, roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return float(x), float(y), float(z), float(w)

    def _decode_png(self, image_payload) -> Optional[np.ndarray]:
        if image_payload is None:
            return None

        raw_bytes = None
        if isinstance(image_payload, (bytes, bytearray)):
            raw_bytes = bytes(image_payload)
        elif isinstance(image_payload, str):
            if not image_payload:
                return None
            try:
                raw_bytes = base64.b64decode(image_payload)
            except Exception:
                # Some AirSim variants return a raw byte-string via latin1 repr.
                raw_bytes = image_payload.encode("latin1", errors="ignore")

        if not raw_bytes or self._cv2 is None:
            return None

        np_data = np.frombuffer(raw_bytes, dtype=np.uint8)
        if np_data.size == 0:
            return None

        frame = self._cv2.imdecode(np_data, self._cv2.IMREAD_COLOR)
        return frame

    def _ship_pose_from_env(self, env) -> tuple[float, float, float]:
        if self.bridge.coord_transform is None:
            raise RuntimeError("UE5 minimap needs an active coordinate transform.")

        ship_pos = env.ship_pos[0]
        heading = float(env.phys_sims[0].state.heading)
        x, y, _ = self.bridge.coord_transform.fossen_to_airsim(
            float(ship_pos[0]),
            float(ship_pos[1]),
            self.bridge.render_z,
        )
        yaw = self.bridge.coord_transform.fossen_heading_to_airsim_yaw(heading)
        return x, y, yaw

    def update(self, env) -> None:
        """Update UE5 top-down camera position and optional preview image."""
        if not self.bridge.is_connected:
            return

        try:
            x, y, yaw = self._ship_pose_from_env(env)
            top_down_yaw = yaw if self.follow_heading else 0.0

            # AirSim uses NED coordinates: lower z is higher altitude.
            camera_z = float(self.bridge.render_z - self.height_offset_m)
            qx, qy, qz, qw = self._quaternion_from_euler(0.0, -np.pi / 2.0, top_down_yaw)

            pose = self.bridge._airsim.Pose(
                self.bridge._airsim.Vector3r(float(x), float(y), camera_z),
                self.bridge._airsim.Quaternionr(qx, qy, qz, qw),
            )
            self.bridge.set_camera_pose(
                self.camera_name,
                pose,
                vehicle_name=self.vehicle_name,
            )

            if self.draw_trail:
                current = self.bridge._airsim.Vector3r(float(x), float(y), float(self.bridge.render_z - 2.0))
                if self._prev_trail_point is not None:
                    self.bridge.plot_line_strip(
                        [self._prev_trail_point, current],
                        color_rgba=[0.1, 0.5, 1.0, 1.0],
                        thickness=self.trail_thickness,
                        is_persistent=True,
                    )
                self._prev_trail_point = current

            if self.enable_preview_window and self._cv2 is not None:
                payload = self.bridge.get_camera_image(
                    self.camera_name,
                    self.bridge._airsim.ImageType.Scene,
                    vehicle_name=self.vehicle_name,
                )
                frame = self._decode_png(payload)
                if frame is not None:
                    self._cv2.imshow(self.preview_window_name, frame)
                    self._cv2.waitKey(1)
        except Exception as exc:
            logger.warning("UE5 top-down minimap update failed: %s", exc)

    def close(self) -> None:
        """Dispose minimap resources."""
        self._prev_trail_point = None
        if self.enable_preview_window and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.preview_window_name)
            except Exception:
                pass
