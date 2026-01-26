import copy
import numpy as np

from fw.simulators.ships.vessel_state import VesselState
from fw.simulators.ships.ship_specs import ShipSpecifications
from fw.simulators.simulation.base_simulator import BaseSimulator


class FhSimAdapter(BaseSimulator):
    """
    FH simulator adapter implementing the unified interface.

    Handles coordinate transformations between framework and FH Sim conventions.
    """

    def __init__(self, specs: ShipSpecifications, fh_engine, dt: float):
        """
        Initialize FH Sim adapter.

        Args:
            specs: Ship specifications
            fh_engine: FH Simulator engine instance
            dt: Time step [s]
        """
        super().__init__(specs, dt)

        self._engine = fh_engine

    def __deepcopy__(self, memo):
        """Create deep copy excluding FH engine."""
        new_instance = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            if key == '_engine':
                setattr(new_instance, key, value)   # Retain the original reference
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))  # Deep copy all other attributes

        return new_instance

    def step(self, action: np.ndarray, enable_smoothing: bool = True) -> VesselState:
        """Step FH Simulator forward one time step."""
        # Apply smoothing to actions
        actuator = self._actuators.apply_smoothing(action, enable_smoothing)

        # Update FH engine controls
        # Update rudder controls based on the turning action.
        for rudder in self._engine.ship_interface.getRudderControls():
            rudder.setControlValue(float(-actuator.rudder_angle))   # this is to compensate for opposite behavior
                                                                    # of the Python environment
                                                                    # in Python: -1 is turn right ; 1 is turn left
                                                                    # FH sim: -1 is turn left ; 1 is turn right
        # Update propeller controls based on the thrust action.
        for propeller in self._engine.ship_interface.getPropellerControls():
            propeller.setEngineLeverValue(float(actuator.thrust))

        # Simulate FH physics
        self._engine.math_model.simulateSeconds(self.dt)

        # Fetch updated state
        new_ship_pos = self._engine.ship_interface.getShipPosition()
        velocity_over_ground = self._engine.ship_interface.getShipVelocityOverGround()
        heading = np.radians(self._engine.ship_interface.getShipHeading())
        yaw_rate = np.radians(self._engine.ship_interface.getShipYawRate())

        self._state = VesselState(
            x=new_ship_pos.x,
            y=new_ship_pos.y,
            heading=(heading + np.pi) % (2.0 * np.pi) - np.pi,
            u=velocity_over_ground.x,
            v=velocity_over_ground.y,
            r=yaw_rate,
        )

        return self._state
