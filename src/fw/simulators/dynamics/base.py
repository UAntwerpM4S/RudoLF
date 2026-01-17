import numpy as np

from typing import Tuple
from abc import ABC, abstractmethod


class DynamicsModel(ABC):
    @abstractmethod
    def calculate_accelerations(self, u: float, v: float, r: float,
                                rudder_angle: float, thrust: float) -> Tuple[float, float, float]:
        pass


    @abstractmethod
    def integrate(self, state: np.ndarray, accelerations: Tuple[float, float, float],
                  dt: float) -> np.ndarray:
        pass
