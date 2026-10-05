from dataclasses import dataclass


@dataclass()
class VesselState:
    """
    Container for the complete vessel state.

    The vessel state includes both earth-fixed positions and
    body-fixed velocities, providing all information needed for
    simulation, control, or reinforcement learning.

    Coordinate frames and conventions (framework-wide, see
    fw.dynamics.dynamics_model.DynamicsBase):
        - Earth-fixed (inertial) frame: x points east, y points north.
        - Body-fixed frame: x points forward, y points to PORT, z up
          (right-handed).
        - Heading (yaw) is measured from the earth-fixed x-axis (east),
          counter-clockwise positive.

    Attributes:
        x: Vessel position along the earth-fixed x-axis [m].
        y: Vessel position along the earth-fixed y-axis [m].
        heading: Vessel heading (yaw) relative to the earth-fixed frame [rad].
        u: Surge velocity along the vessel's x-axis (forward) [m/s].
        v: Sway velocity along the vessel's y-axis, port positive [m/s].
        r: Yaw rate about the z-axis, counter-clockwise (bow to port)
           positive, r = d(heading)/dt [rad/s].
    """

    terminated: bool
    """
    Indicates whether the ship’s episode ended due to a terminal condition,
    such as collision, goal achievement, or other criterion.
    """

    truncated: bool
    """
    Indicates whether the ship’s episode ended due to truncation, e.g., because the
    maximum allowed number of steps was reached without meeting a terminal condition.
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
    Vessel heading (yaw) relative to earth-fixed frame [rad]: 0 = east,
    counter-clockwise positive.
    """

    u: float
    """
    Surge velocity along body-fixed x-axis (forward) [m/s].
    """

    v: float
    """
    Sway velocity along body-fixed y-axis, port positive [m/s].
    """

    r: float
    """
    Yaw rate (rotation about body-fixed z-axis) [rad/s], counter-clockwise
    (bow to port) positive.
    """
