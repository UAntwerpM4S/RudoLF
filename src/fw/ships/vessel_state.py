from dataclasses import dataclass


@dataclass()
class VesselState:
    """
    Container for the complete vessel state.

    The vessel state includes both earth-fixed positions and
    body-fixed velocities, providing all information needed for
    simulation, control, or reinforcement learning.

    Coordinate frames and conventions (one convention for the whole
    framework; ShipSimulator.integrate, FH's own body frame and the
    surrogate datasets all use it):
        - Earth-fixed (inertial) frame: x points east, y points north.
        - Body-fixed frame: x points forward, y points to PORT (z up).
        - Heading (yaw) is measured from the earth-fixed x-axis,
          counter-clockwise positive.
        - Sway v is positive to PORT; yaw rate r is positive
          counter-clockwise (bow turning to port).
        - Rudder command (not part of the state): positive = the ship turns
          to PORT, for every ship and dynamics model (= azipod direction to
          starboard; for a classic rudder FhTabularModel converts to FH's
          sign, see fh_tabular_model.fh_rudder_sign).

    Attributes:
        x: Vessel position along the earth-fixed x-axis [m].
        y: Vessel position along the earth-fixed y-axis [m].
        heading: Vessel heading (yaw) relative to the earth-fixed frame [rad].
        u: Surge velocity along the vessel's x-axis (forward) [m/s].
        v: Sway velocity along the vessel's y-axis, positive to port [m/s].
        r: Yaw rate, counter-clockwise (to port) positive [rad/s].
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
    Vessel heading (yaw) relative to earth-fixed frame [rad].
    """

    u: float
    """
    Surge velocity along body-fixed x-axis (forward) [m/s].
    """

    v: float
    """
    Sway velocity along body-fixed y-axis, positive to PORT [m/s].
    """

    r: float
    """
    Yaw rate about the body-fixed z-axis (up), counter-clockwise positive [rad/s].
    """
