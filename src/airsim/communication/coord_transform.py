"""
Framework <-> AirSim coordinate transform.
"""

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from fw.common.airsim.tools.angles import normalize_angle as _normalize_angle

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIBRATION_PATH = _THIS_DIR.parent / "config" / "calibration.json"


class CoordTransform:
    """
    Bidirectional coordinate transform between framework and AirSim frames.

    The transform is defined by an anchor pair
    (``framework_origin`` <-> ``airsim_origin``) plus scale, rotation, and
    optional axis mirrors. Any point is transformed by computing its delta from
    the framework origin, optionally mirroring that delta, rotating and
    scaling it, and then adding the AirSim origin.
    """

    def __init__(self, calibration: Optional[dict] = None) -> None:
        """
        Initialize the coordinate transform.

        Args:
            calibration: Dictionary containing the calibration parameters.
                Expected keys: ``"scale"``, ``"rotation"``,
                ``"framework_origin"``, ``"airsim_origin"``.
                Optional keys: ``"mirror_x"`` and ``"mirror_y"``.
                Legacy calibration files with a non-AirSim origin key are also
                accepted automatically.
        """

        if calibration is not None:
            self._scale = calibration["scale"]
            self._rotation = calibration["rotation"]
            self._mirror_x = calibration.get("mirror_x", False)
            self._mirror_y = calibration.get("mirror_y", False)
            framework_origin = calibration.get("framework_origin")
            if framework_origin is None:
                framework_origin = next(
                    (
                        value
                        for key, value in calibration.items()
                        if key.endswith("_origin") and key != "airsim_origin"
                    ),
                    [0.0, 0.0],
                )
            self._framework_origin = np.array(
                framework_origin,
                dtype=np.float64,
            )
            self._airsim_origin = np.array(calibration["airsim_origin"], dtype=np.float64)
        else:
            self._scale = 1.0
            self._rotation = -math.pi / 2.0
            self._mirror_x = False
            self._mirror_y = False
            self._framework_origin = np.array([0.0, 0.0], dtype=np.float64)
            self._airsim_origin = np.array([0.0, 0.0], dtype=np.float64)

    def framework_to_airsim(
        self, framework_x: float, framework_y: float, z: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Convert a framework position to AirSim ``(x, y, z)``.
        """

        dx = framework_x - self._framework_origin[0]
        dy = framework_y - self._framework_origin[1]

        if self._mirror_x:
            dx = -dx
        if self._mirror_y:
            dy = -dy

        cos_r = math.cos(self._rotation)
        sin_r = math.sin(self._rotation)
        ax = self._scale * (cos_r * dx - sin_r * dy) + self._airsim_origin[0]
        ay = self._scale * (sin_r * dx + cos_r * dy) + self._airsim_origin[1]
        return ax, ay, z

    def framework_to_airsim_2d(
        self, framework_x: float, framework_y: float
    ) -> Tuple[float, float]:
        """
        Convenience wrapper that returns only the 2-D AirSim position.
        """

        ax, ay, _ = self.framework_to_airsim(framework_x, framework_y)
        return ax, ay

    def framework_vector_to_airsim(
        self, framework_dx: float, framework_dy: float
    ) -> Tuple[float, float]:
        """
        Convert a framework-frame vector to an AirSim-frame vector.
        """

        dx = framework_dx
        dy = framework_dy

        if self._mirror_x:
            dx = -dx
        if self._mirror_y:
            dy = -dy

        cos_r = math.cos(self._rotation)
        sin_r = math.sin(self._rotation)
        ax = self._scale * (cos_r * dx - sin_r * dy)
        ay = self._scale * (sin_r * dx + cos_r * dy)
        return ax, ay

    def framework_velocity_to_airsim(
        self, framework_vx: float, framework_vy: float
    ) -> Tuple[float, float]:
        """
        Convert an earth-fixed framework velocity vector to AirSim.
        """

        return self.framework_vector_to_airsim(framework_vx, framework_vy)

    def airsim_to_framework(
        self, airsim_x: float, airsim_y: float
    ) -> Tuple[float, float]:
        """
        Convert an AirSim position back to the framework frame.
        """

        rx = airsim_x - self._airsim_origin[0]
        ry = airsim_y - self._airsim_origin[1]

        cos_r = math.cos(self._rotation)   # cos(-θ) = cos(θ)
        sin_r = -math.sin(self._rotation)  # sin(-θ) = -sin(θ)

        dx = (cos_r * rx - sin_r * ry) / self._scale
        dy = (sin_r * rx + cos_r * ry) / self._scale

        if self._mirror_x:
            dx = -dx
        if self._mirror_y:
            dy = -dy

        fx = dx + self._framework_origin[0]
        fy = dy + self._framework_origin[1]
        return fx, fy

    def framework_heading_to_airsim_yaw(self, heading: float) -> float:
        """
        Convert a framework heading to an AirSim yaw angle.
        """

        adjusted_heading = heading
        if self._mirror_x:
            adjusted_heading = math.pi - adjusted_heading
        if self._mirror_y:
            adjusted_heading = -adjusted_heading
        return self._rotation + adjusted_heading

    def airsim_yaw_to_framework_heading(self, yaw: float) -> float:
        """
        Convert an AirSim yaw angle back to a framework heading.
        """

        heading = yaw - self._rotation
        if self._mirror_y:
            heading = -heading
        if self._mirror_x:
            heading = math.pi - heading
        return _normalize_angle(heading)

    def framework_yaw_rate_to_airsim(self, yaw_rate: float) -> float:
        """
        Convert a framework yaw-rate to the matching AirSim yaw-rate.
        """

        sign = -1.0 if (self._mirror_x ^ self._mirror_y) else 1.0
        return sign * yaw_rate

    @classmethod
    def load(cls, path: Path = _DEFAULT_CALIBRATION_PATH) -> "CoordTransform":
        """
        Load a coordinate transform from a JSON calibration file.
        """

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(data)

    @classmethod
    def identity(cls, framework_origin=(0.0, 0.0)) -> "CoordTransform":
        """
        Create a simple axis-swap transform without full calibration.
        """

        return cls(
            {
                "scale": 1.0,
                "rotation": -math.pi / 2.0,
                "framework_origin": list(framework_origin),
                "airsim_origin": [0.0, 0.0],
            }
        )
