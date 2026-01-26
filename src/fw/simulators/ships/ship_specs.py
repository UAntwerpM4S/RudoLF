from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class ShipSpecifications:
    """
    Container for ship physical properties, actuator limits, and kinematic constraints.

    This immutable dataclass defines the vessel parameters required for
    simulation, control, and actuator modeling. It provides:

        - Physical properties (length, mass)
        - Actuator limits (rudder and thrust)
        - Jitter suppression thresholds
        - Kinematic limits for simulation constraints

    All units are expressed in SI unless otherwise specified.
    """

    # Physical properties
    length: float
    """
    Vessel length [m]. Used for calculating inertia and added-mass terms.
    """

    mass: float
    """
    Vessel mass [kg]. Used in dynamics calculations and damping scaling.
    """

    # Actuator limits
    max_rudder_angle: float
    """
    Maximum rudder deflection angle [rad]. Positive to starboard.
    """

    max_rudder_rate: float
    """
    Maximum rudder rate [rad/s]. Limits the speed at which rudder can move.
    """

    min_thrust: float
    """
    Minimum normalized thrust [-]. Used to define the forward thrust envelope.
    """

    max_thrust: float
    """
    Maximum normalized thrust [-]. Defines the upper bound of propulsion.
    """

    max_thrust_rate: float
    """
    Maximum thrust rate change [-/s]. Limits the speed of thrust adjustment.
    """

    # Jitter suppression thresholds
    rudder_jitter_threshold: float
    """
    Minimum rudder deviation [rad] to consider for actuation updates.
    Smaller deviations are ignored to prevent high-frequency oscillations.
    """

    thrust_jitter_threshold: float
    """
    Minimum thrust deviation [-] to consider for actuation updates.
    Prevents small oscillations around the current thrust value.
    """

    # Kinematic limits (simulation constraints)
    surge_limits: Tuple[float, float]
    """
    Allowed surge velocity range [m/s] for the vessel in simulation.
    """

    sway_limits: Tuple[float, float]
    """
    Allowed sway velocity range [m/s] for the vessel in simulation.
    """

    yaw_rate_limits: Tuple[float, float]
    """
    Allowed yaw rate range [rad/s] for the vessel in simulation.
    """
