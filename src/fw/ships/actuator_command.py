from dataclasses import dataclass


@dataclass(frozen=True)
class ActuatorCommand:
    """
    Immutable actuator command (rudder, thrust), produced by the action mapping
    and consumed by the dynamics model or simulator.
    """

    rudder_angle: float   # [rad]
    """
    Rudder deflection angle in radians.

    Positive = the ship turns to PORT (framework convention, see
    fw.dynamics.dynamics_model.DynamicsBase and CONVENTIONS.md).
    """

    thrust: float         # normalized [-]
    """
    Propulsion thrust command.

    The interpretation of this value (normalized vs. physical units)
    depends on the consuming dynamics model. In the current setup, it
    is assumed to be a normalized, dimensionless quantity where larger
    values correspond to greater forward thrust.
    """
