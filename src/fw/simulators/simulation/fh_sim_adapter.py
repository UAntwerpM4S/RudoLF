import copy
import numpy as np

from fw.simulators.ships.vessel_state import VesselState
from fw.simulators.ships.ship_specs import ShipSpecifications
from fw.simulators.simulation.base_simulator import BaseSimulator


class FhSimAdapter(BaseSimulator):
    """
    Adapter for FH Simulator integrating with the unified simulator interface.

    This class wraps the FH Simulator engine to provide a consistent
    interface compatible with the framework's BaseSimulator. It handles:

        - Conversion between framework and FH Simulator coordinate and
          control conventions
        - Actuator smoothing and rate limiting
        - State extraction in the framework's VesselState format

    FH-specific notes:
        - Rudder direction is inverted relative to the framework convention:
          framework: -1 = turn right, 1 = turn left
          FH Sim: -1 = turn left, 1 = turn right
    """

    def __init__(self, specs: ShipSpecifications, fh_engine, dt: float):
        """
        Initialize the FH Simulator adapter.

        Args:
            specs: ShipSpecifications object defining vessel and actuator limits.
            fh_engine: Instance of the FH Simulator engine.
            dt: Simulation time step [s].

        Raises:
            ValueError: If dt <= 0 (handled by BaseSimulator).
        """

        super().__init__(specs, dt)

        self._engine = fh_engine

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the adapter, excluding the FH engine reference.

        The FH engine instance is shared; all other attributes are deep-copied.

        Args:
            memo: Dictionary used by copy.deepcopy to avoid duplicate copies.

        Returns:
            FhSimAdapter: Deep-copied instance with shared FH engine.
        """

        new_instance = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            if key == '_engine':
                setattr(new_instance, key, value)   # Retain the original reference
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))  # Deep copy all other attributes

        return new_instance

    def step(self, action: np.ndarray, enable_smoothing: bool = True) -> VesselState:
        """
        Advance the FH Simulator by one time step using a normalized action.

        Args:
            action: Normalized action array [rudder, thrust] in [-1, 1].
            enable_smoothing: Enable actuator smoothing and rate limiting.

        Returns:
            VesselState: Updated vessel state in the framework format, with:
                - x, y: Earth-fixed positions [m]
                - heading: Earth-fixed heading [rad], wrapped to [-π, π]
                - u, v: Body-fixed surge and sway velocities [m/s]
                - r: Yaw rate [rad/s]

        Notes:
            - Rudder commands are inverted to match FH Sim conventions.
            - Thrust commands are applied directly to propeller controls.
            - Actuator smoothing and rate limiting are applied via
              ActuatorModel.
            - Coordinates and velocities are extracted from FH engine outputs.
        """

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
