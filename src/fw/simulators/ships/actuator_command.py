from dataclasses import dataclass


@dataclass(frozen=True)
class ActuatorCommand:
    """
    Physical actuator command.
    """
    rudder_angle: float   # [rad]
    thrust: float         # normalized [-]
