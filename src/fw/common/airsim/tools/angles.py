"""Shared angle and quaternion helpers for AirSim modules."""

from __future__ import annotations

import math


def normalize_angle(angle_rad: float) -> float:
    """Wrap an angle in radians to the interval [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(q) -> float:
    """Extract yaw in radians from an AirSim-style quaternion object."""
    return math.atan2(
        2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
        1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2),
    )


def wrapped_angle_delta(start: float, end: float) -> float:
    """Return the shortest signed angular delta from start to end."""
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def interpolate_angle(start: float, end: float, alpha: float) -> float:
    """Interpolate between two angles while handling wrap-around."""
    return start + alpha * wrapped_angle_delta(start, end)
