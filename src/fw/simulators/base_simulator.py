import numpy as np

from abc import ABC, abstractmethod
from fw.ships.vessel_state import VesselState
from fw.ships.actuator_model import ActuatorModel
from fw.ships.ship_specs import ShipSpecifications


class BaseSimulator(ABC):
    """
    Abstract base class defining a unified simulator interface for vessels.

    This interface is intended for both control loops and reinforcement
    learning environments. It standardizes the interaction with:

        - Vessel state representation (VesselState)
        - Actuator dynamics (ActuatorModel)
        - Time stepping with fixed simulation interval

    Subclasses must implement the `step` method to define the vessel
    dynamics and environmental effects.
    """

    def __init__(self, specs: ShipSpecifications, dt: float):
        """
        Initialize the simulator with vessel specifications and time step.

        Args:
            specs: ShipSpecifications object defining vessel dimensions,
                actuator limits, and kinematic constraints.
            dt: Simulation time step [s]. Must be positive.

        Raises:
            ValueError: If dt <= 0.
        """

        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")

        self.dt = dt
        self._specs = specs
        self._actuators = ActuatorModel(specs)
        # Initial state
        self._state = VesselState(x=0.0, y=0.0, heading=0.0, u=0.0, v=0.0, r=0.0)

    @property
    def state(self) -> VesselState:
        """
        Get the current vessel state.

        Returns:
            VesselState: Current vessel positions (x, y, heading) in
            earth-fixed frame and body-fixed velocities (u, v, r).
        """

        return self._state

    @property
    def actuators(self) -> ActuatorModel:
        """
        Get the actuator model.

        Returns:
            ActuatorModel: Current actuator dynamics instance, used
            to apply rate limiting, smoothing, and saturation.
        """

        return self._actuators

    def reset(self, state: VesselState) -> None:
        """
        Reset the simulator to a specific vessel state.

        This also resets the actuator model to neutral positions.

        Args:
            state: VesselState object defining the initial state
                (position, heading, velocities).
        """

        self._state = state
        self._actuators.reset()

    @abstractmethod
    def step(self, action: np.ndarray, enable_smoothing: bool = True) -> None:
        """
        Advance the simulation by one time step using a normalized action.

        Subclasses must implement this method to update the vessel
        state according to the chosen dynamics model, environmental
        effects, and actuator commands.

        Args:
            action: Normalized action array [rudder, thrust] in the
                range [-1, 1].
            enable_smoothing: Enable actuator rate limiting and smoothing
                to prevent unrealistic instantaneous changes.
        """
        pass
