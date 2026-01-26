from dataclasses import dataclass


@dataclass(frozen=True)
class ActuatorState:
    """
    Actual actuator output after rate limiting.
    """
    rudder_angle: float   # [rad]
    thrust: float         # normalized [-]
