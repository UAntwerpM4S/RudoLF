from dataclasses import dataclass


@dataclass(frozen=True)
class VesselState:
    """
    Container for the complete vessel state.

    The vessel state includes both earth-fixed positions and
    body-fixed velocities, providing all information needed for
    simulation, control, or reinforcement learning.

    Coordinate frames and conventions:
        - Earth-fixed (inertial) frame: x points east, y points north.
        - Body-fixed frame: x points forward, y points starboard.
        - Heading (yaw) is measured from the earth-fixed x-axis.

    Attributes:
        x: Vessel position along the earth-fixed x-axis [m].
        y: Vessel position along the earth-fixed y-axis [m].
        heading: Vessel heading (yaw) relative to the earth-fixed frame [rad].
        u: Surge velocity along the vessel's x-axis (forward) [m/s].
        v: Sway velocity along the vessel's y-axis (starboard) [m/s].
        r: Yaw rate (rotation about the z-axis) [rad/s].
    """

    x: float
    """
    Vessel position along the earth-fixed x-axis [m].
    """

    y: float
    """
    Vessel position along the earth-fixed y-axis [m].
    """

    heading: float
    """
    Vessel heading (yaw) relative to earth-fixed frame [rad].
    """

    u: float
    """
    Surge velocity along body-fixed x-axis (forward) [m/s].
    """

    v: float
    """
    Sway velocity along body-fixed y-axis (starboard) [m/s].
    """

    r: float
    """
    Yaw rate (rotation about body-fixed z-axis) [rad/s].
    """
