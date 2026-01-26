from typing import Tuple
from abc import ABC, abstractmethod


class DynamicsModel(ABC):
    """Abstract base class for vessel dynamics models."""

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
        Calculate body-fixed accelerations.

        Args:
            u: Body-fixed surge velocity [m/s]
            v: Body-fixed sway velocity [m/s]
            r: Body-fixed yaw rate [rad/s]
            rudder_angle: Rudder angle [rad], positive to starboard
            thrust: Normalized thrust [-], positive forward

        Returns:
            Tuple of (du, dv, dr) accelerations [m/s2, m/s2, rad/s2]
        """
        pass
