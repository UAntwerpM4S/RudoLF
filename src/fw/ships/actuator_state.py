from dataclasses import dataclass


@dataclass(frozen=True)
class ActuatorState:
    """
    Container for the actual actuator outputs after applying dynamics.

    This immutable data structure represents the current state of the
    vessel actuators (rudder and thrust) after considering rate limits,
    smoothing, and saturation. It is typically produced by the
    `ActuatorModel.step()` or `ActuatorModel.apply_smoothing()` methods
    and used by the vessel dynamics or simulator.

    Attributes:
        rudder_angle: Current rudder deflection angle in radians [rad].
            Positive values correspond to starboard deflection in the
            body-fixed frame.
        thrust: Current propulsion thrust command (dimensionless or
            normalized). The interpretation (normalized vs. physical
            units) depends on the consuming dynamics model.
    """

    rudder_angle: float   # [rad]
    """
    Rudder deflection angle in radians [rad].
    """

    thrust: float         # normalized [-]
    """
    Propulsion thrust command (normalized or physical units).
    """
