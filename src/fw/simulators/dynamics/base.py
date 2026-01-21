from typing import Tuple
from abc import ABC, abstractmethod


class DynamicsModel(ABC):
    @abstractmethod
    def calculate_accelerations(self, u: float, v: float, r: float,
                                rudder_angle: float, thrust: float) -> Tuple[float, float, float]:
        pass
