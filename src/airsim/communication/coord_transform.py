"""Fossen <-> AirSim coordinate transform.

Handles the mapping between Fossen (east, north) coordinates and
AirSim / UE5 coordinates, including scale, rotation, axis mirroring,
and translation offsets.  Configured via a calibration dict (usually
loaded from ``calibration.json``).
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from fw.common.airsim.tools.angles import normalize_angle as _normalize_angle

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CALIBRATION_PATH = _THIS_DIR.parent / "config" / "calibration.json"

class CoordTransform:
    """Bidirectional coordinate transform between Fossen and AirSim frames.

    The transform is defined by an anchor pair (fossen_origin <-> airsim_origin)
    plus scale, rotation, and optional axis mirrors.  Any point is transformed
    by computing its delta from the fossen_origin, optionally mirroring the
    delta, rotating + scaling it, and adding the airsim_origin.

    The mirror logic uses the simplified "delta-then-mirror" pattern::

        dx = fossen_x - fossen_origin_x
        if mirror_x:
            dx = -dx

    which is equivalent to the original ternary version but far easier to
    reason about.

    Attributes:
        _scale: Multiplicative factor between Fossen and AirSim distances.
        _rotation: Rotation angle (radians) applied after mirroring.
        _mirror_x: Whether the Fossen x-axis is flipped.
        _mirror_y: Whether the Fossen y-axis is flipped.
        _fossen_origin: 2-element anchor point in the Fossen frame.
        _airsim_origin: 2-element anchor point in the AirSim frame.
    """

    def __init__(self, calibration: Optional[dict] = None) -> None:
        """Initialise the coordinate transform.

        Args:
            calibration: Dictionary containing the calibration parameters.
                Expected keys: ``"scale"``, ``"rotation"``,
                ``"fossen_origin"``, ``"airsim_origin"``.
                Optional keys: ``"mirror_x"`` (default ``False``),
                ``"mirror_y"`` (default ``False``).
                If ``None``, an identity axis-swap transform is created
                (90° rotation, scale 1.0, no mirrors).
        """
        if calibration is not None:
            self._scale = calibration["scale"]
            self._rotation = calibration["rotation"]
            self._mirror_x = calibration.get("mirror_x", False)
            self._mirror_y = calibration.get("mirror_y", False)
            self._fossen_origin = np.array(calibration["fossen_origin"], dtype=np.float64)
            self._airsim_origin = np.array(calibration["airsim_origin"], dtype=np.float64)
        else:
            # Identity transform (Fossen x->AirSim y, Fossen y->AirSim x)
            self._scale = 1.0
            self._rotation = -math.pi / 2.0  # 90° swap
            self._mirror_x = False
            self._mirror_y = False
            self._fossen_origin = np.array([0.0, 0.0])
            self._airsim_origin = np.array([0.0, 0.0])

    # -- Forward transform --────

    def fossen_to_airsim(
        self, fossen_x: float, fossen_y: float, z: float = 0.0
    ) -> Tuple[float, float, float]:
        """Convert a Fossen (east, north) position to AirSim (x, y, z).

        Applies mirroring, rotation, scaling, and translation to map a
        point from the Fossen coordinate frame to the AirSim / UE5 frame.

        Args:
            fossen_x: East coordinate in the Fossen frame.
            fossen_y: North coordinate in the Fossen frame.
            z: Vertical coordinate (passed through unchanged).
                Defaults to ``0.0``.

        Returns:
            A tuple ``(airsim_x, airsim_y, airsim_z)``.
        """
        dx = fossen_x - self._fossen_origin[0]
        dy = fossen_y - self._fossen_origin[1]

        if self._mirror_x:
            dx = -dx
        if self._mirror_y:
            dy = -dy

        cos_r = math.cos(self._rotation)
        sin_r = math.sin(self._rotation)
        ax = self._scale * (cos_r * dx - sin_r * dy) + self._airsim_origin[0]
        ay = self._scale * (sin_r * dx + cos_r * dy) + self._airsim_origin[1]
        return ax, ay, z

    def fossen_to_airsim_2d(
        self, fossen_x: float, fossen_y: float
    ) -> Tuple[float, float]:
        """Convenience wrapper that returns only the 2-D AirSim position.

        Args:
            fossen_x: East coordinate in the Fossen frame.
            fossen_y: North coordinate in the Fossen frame.

        Returns:
            A tuple ``(airsim_x, airsim_y)``.
        """
        ax, ay, _ = self.fossen_to_airsim(fossen_x, fossen_y)
        return ax, ay

    def fossen_vector_to_airsim(
        self, fossen_dx: float, fossen_dy: float
    ) -> Tuple[float, float]:
        """Convert a Fossen-frame vector to an AirSim-frame vector.

        Unlike position transforms, vectors are rotated, mirrored, and scaled,
        but are not translated by the origin anchors.

        Args:
            fossen_dx: X component of the vector in the Fossen frame.
            fossen_dy: Y component of the vector in the Fossen frame.

        Returns:
            A tuple ``(airsim_dx, airsim_dy)``.
        """
        dx = fossen_dx
        dy = fossen_dy

        if self._mirror_x:
            dx = -dx
        if self._mirror_y:
            dy = -dy

        cos_r = math.cos(self._rotation)
        sin_r = math.sin(self._rotation)
        ax = self._scale * (cos_r * dx - sin_r * dy)
        ay = self._scale * (sin_r * dx + cos_r * dy)
        return ax, ay

    def fossen_velocity_to_airsim(
        self, fossen_vx: float, fossen_vy: float
    ) -> Tuple[float, float]:
        """Convert an earth-fixed Fossen velocity vector to AirSim.

        Args:
            fossen_vx: X velocity component in the Fossen frame.
            fossen_vy: Y velocity component in the Fossen frame.

        Returns:
            A tuple ``(airsim_vx, airsim_vy)``.
        """
        return self.fossen_vector_to_airsim(fossen_vx, fossen_vy)

    # -- Inverse transform --────

    def airsim_to_fossen(
        self, airsim_x: float, airsim_y: float
    ) -> Tuple[float, float]:
        """Convert an AirSim position back to Fossen (east, north).

        Reverses the translation, rotation, scaling, and mirroring to
        recover the original Fossen coordinates.

        Args:
            airsim_x: X coordinate in the AirSim / UE5 frame.
            airsim_y: Y coordinate in the AirSim / UE5 frame.

        Returns:
            A tuple ``(fossen_x, fossen_y)``.
        """
        rx = airsim_x - self._airsim_origin[0]
        ry = airsim_y - self._airsim_origin[1]

        cos_r = math.cos(-self._rotation)
        sin_r = math.sin(-self._rotation)

        dx = (cos_r * rx - sin_r * ry) / self._scale
        dy = (sin_r * rx + cos_r * ry) / self._scale

        # Undo mirror
        if self._mirror_x:
            dx = -dx
        if self._mirror_y:
            dy = -dy

        fx = dx + self._fossen_origin[0]
        fy = dy + self._fossen_origin[1]
        return fx, fy

    # -- Heading / yaw conversion --────

    def fossen_heading_to_airsim_yaw(self, heading: float) -> float:
        """Convert a Fossen heading to an AirSim yaw angle.

        Fossen heading is measured from east, counter-clockwise positive.
        AirSim yaw is measured from north, clockwise positive.
        This method accounts for any axis mirroring set in the calibration.

        Args:
            heading: Heading angle in radians (Fossen convention).

        Returns:
            The equivalent yaw angle in radians (AirSim convention).
        """
        h = heading
        if self._mirror_x:
            h = math.pi - h  # mirror east axis flips heading
        if self._mirror_y:
            h = -h            # mirror north axis flips heading
        return self._rotation + h

    def airsim_yaw_to_fossen_heading(self, yaw: float) -> float:
        """Convert an AirSim yaw angle to a Fossen heading.

        Reverses the rotation and mirroring applied by
        ``fossen_heading_to_airsim_yaw``.

        Args:
            yaw: Yaw angle in radians (AirSim convention).

        Returns:
            The equivalent heading in radians (Fossen convention),
            normalised to [-π, π].
        """
        h = yaw - self._rotation
        if self._mirror_y:
            h = -h
        if self._mirror_x:
            h = math.pi - h
        return _normalize_angle(h)

    def fossen_yaw_rate_to_airsim(self, yaw_rate: float) -> float:
        """Convert a Fossen yaw-rate to the matching AirSim yaw-rate.

        Args:
            yaw_rate: Yaw rate in rad/s in the Fossen frame.

        Returns:
            Yaw rate in rad/s adjusted for the AirSim frame axis convention.
        """
        sign = -1.0 if (self._mirror_x ^ self._mirror_y) else 1.0
        return sign * yaw_rate

    # -- Persistence --────

    @classmethod
    def load(cls, path: Path = _DEFAULT_CALIBRATION_PATH) -> "CoordTransform":
        """Load a coordinate transform from a JSON calibration file.

        Args:
            path: Filesystem path to the calibration JSON.
                Defaults to ``airsim/config/calibration.json``.

        Returns:
            A new ``CoordTransform`` instance configured from the file.

        Raises:
            FileNotFoundError: If the calibration file does not exist.
            KeyError: If required keys are missing from the JSON.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def identity(cls, fossen_origin=(0.0, 0.0)) -> "CoordTransform":
        """Create a simple axis-swap transform without full calibration.

        Assumes that UE5's PlayerStart is at the Fossen origin and that
        both coordinate systems use meters.  The only transformation
        applied is a 90° rotation to swap axes.

        Args:
            fossen_origin: ``(x, y)`` anchor in the Fossen frame.
                Defaults to ``(0.0, 0.0)``.

        Returns:
            A new ``CoordTransform`` with scale 1.0 and a −90° rotation.
        """
        return cls({
            "scale": 1.0,
            "rotation": -math.pi / 2.0,
            "fossen_origin": list(fossen_origin),
            "airsim_origin": [0.0, 0.0],
        })
