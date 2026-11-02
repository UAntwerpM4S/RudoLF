from dataclasses import dataclass


@dataclass(frozen=True)
class ActuatorState:
    """
    Actual actuator outputs after rate limits and saturation (from ActuatorModel.step()).

    Attributes:
        rudder_angle: Rudder angle [rad]; positive turns the ship to port
            (see fw.dynamics.dynamics_model.DynamicsBase).
        thrust: Thrust command; normalized or physical depending on the
            consuming dynamics model.
    """

    rudder_angle: float   # [rad]
    """
    Rudder deflection angle in radians [rad].
    """

    thrust: float         # normalized [-]
    """
    Propulsion thrust command (normalized or physical units).
    """
