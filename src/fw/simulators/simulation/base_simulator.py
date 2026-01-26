import numpy as np

from abc import ABC, abstractmethod
from fw.simulators.ships.vessel_state import VesselState
from fw.simulators.ships.actuator_model import ActuatorModel
from fw.simulators.ships.ship_specs import ShipSpecifications


class BaseSimulator(ABC):
    """
    Unified simulator interface for RL and control loops.
    """


    def __init__(self, specs: ShipSpecifications, dt: float):
        self.dt = dt
        self._specs = specs
        self._actuators = ActuatorModel(specs)
        self._state = VesselState(x=0.0, y=0.0, heading=0.0, u=0.0, v=0.0, r=0.0)

    @property
    def state(self) -> VesselState:
        return self._state

    @property
    def actuators(self) -> ActuatorModel:
        return self._actuators

    def reset(self, state: VesselState) -> None:
        self._state = state
        self._actuators.reset()

    @abstractmethod
    def step(self, action: np.ndarray, enable_smoothing: bool) -> VesselState:
        """
        Step simulator by one timestep using normalized action [-1, 1].
        Returns updated VesselState.
        """
        pass
