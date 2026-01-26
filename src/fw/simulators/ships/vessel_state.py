from dataclasses import dataclass


@dataclass(frozen=True)
class VesselState:
    """
    Vessel state expressed in body-fixed velocities and earth-fixed position.

    x, y      : position in world frame [m]
    psi       : heading [rad]
    u, v      : surge and sway velocity [m/s]
    r         : yaw rate [rad/s]
    """
    x: float
    y: float
    heading: float
    u: float
    v: float
    r: float
