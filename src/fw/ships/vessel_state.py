from dataclasses import dataclass


@dataclass()
class VesselState:
    """
    Vessel state: earth-fixed position and heading, body-fixed velocities, and
    episode end flags.

    Conventions (framework-wide; also used by ShipSimulator.integrate, FH's body
    frame and the surrogate datasets):
        - Earth-fixed frame: x east, y north. Heading from the x-axis, CCW positive.
        - Body-fixed frame: x forward, y to port, z up. Sway v positive to port;
          yaw rate r CCW positive (bow turning to port).
        - Rudder command (not part of the state): positive turns the ship to port
          for every ship and model (azipod direction to starboard); FhTabularModel
          converts to FH's sign for classic rudders (fh_tabular_model.fh_rudder_sign).
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
