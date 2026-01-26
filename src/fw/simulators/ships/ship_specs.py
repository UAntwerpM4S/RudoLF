from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class ShipSpecifications:
    # Physical properties
    length: float          # [m]
    mass: float            # [kg]

    # Actuator limits
    max_rudder_angle: float      # [rad]
    max_rudder_rate: float       # [rad/s]
    min_thrust: float            # [-]
    max_thrust: float            # [-]
    max_thrust_rate: float       # [-/s]

    # Jitter suppression
    rudder_jitter_threshold: float
    thrust_jitter_threshold: float

    # Kinematic limits (simulation constraints)
    surge_limits: Tuple[float, float]     # [m/s]
    sway_limits: Tuple[float, float]      # [m/s]
    yaw_rate_limits: Tuple[float, float]  # [rad/s]
