from dataclasses import dataclass


@dataclass(frozen=True)
class ActuatorCommand:
    """
    Container for physical actuator commands.

    This immutable data structure represents a single control command
    issued to the vessel actuators. It is typically produced by an
    action-mapping layer and consumed by the vessel dynamics or
    simulator.

    The class is intentionally lightweight and free of behavior; it
    serves solely as a typed and self-documenting transport object.
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
