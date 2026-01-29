from typing import Tuple
from enum import auto, Enum
from abc import ABC, abstractmethod


class Model(Enum):
    NOMOTO = auto()
    FOSSEN = auto()


class DynamicsBase(ABC):
    """
    Abstract base class for vessel dynamics models.

    This interface defines the contract for continuous-time vessel dynamics
    formulated in the body-fixed reference frame. Concrete implementations
    are expected to model the surge–sway–yaw (3-DOF) dynamics of a marine
    vessel, including hydrodynamic forces, control inputs, and optionally
    environmental effects.

    The model exposes accelerations only; time integration and state
    propagation are assumed to be handled by a higher-level simulator.
    """

    @abstractmethod
    def accelerations(
        self,
        u: float,
        v: float,
        r: float,
        rudder_angle: float,
        thrust: float,
    ) -> Tuple[float, float, float]:
        """
        Compute body-fixed accelerations from the current state and inputs.

        Given the current body-fixed velocities and control inputs, this
        method computes the corresponding accelerations in surge, sway,
        and yaw. Implementations may include nonlinear hydrodynamics,
        damping, Coriolis and centripetal effects, and actuator models,
        as appropriate for the fidelity of the dynamics model.

        Args:
            u: Body-fixed surge velocity in meters per second (m/s).
            v: Body-fixed sway velocity in meters per second (m/s).
            r: Body-fixed yaw rate in radians per second (rad/s).
            rudder_angle: Rudder angle in radians, positive to starboard.
            thrust: Normalized thrust command (dimensionless), where positive
                values correspond to forward propulsion.

        Returns:
            A tuple `(du, dv, dr)` containing:
                - du: Surge acceleration in meters per second squared (m/s²).
                - dv: Sway acceleration in meters per second squared (m/s²).
                - dr: Yaw rate in radians per second squared (rad/s²).
        """
        pass
