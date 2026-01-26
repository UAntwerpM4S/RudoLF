from typing import Tuple
from abc import ABC, abstractmethod


class DynamicsModel(ABC):
    """
    Body-fixed 3DOF vessel dynamics.

    Assumptions:
    - Body-fixed reference at center of gravity
    - x forward, y starboard, z down
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
        pass
