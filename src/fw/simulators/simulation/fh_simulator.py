import numpy as np

from fw.simulators.ships.ship import Ship


class FhSimulator:
    def __init__(self, ship: Ship, dynamics, initial_ship_pos, initial_ship_heading, dt: float, wind: bool, current: bool):
        self.dt = dt
        self.ship = ship
        self.wind = wind
        self.current = current
        self.dynamics = dynamics
        self._state = np.array([initial_ship_pos[0], initial_ship_pos[1], initial_ship_heading, 0.0, 0.0, 0.0], dtype=np.float32)

        self._initialize_control_parameters()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["dynamics"] = None   # drop non-picklable engine
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dynamics = None

    @property
    def position(self):
        return self._state[:2]


    @property
    def heading(self):
        return self._state[2]


    @property
    def surge(self):
        return self._state[3]


    @property
    def sway(self):
        return self._state[4]


    @property
    def yaw_rate(self):
        return self._state[5]


    def _initialize_control_parameters(self) -> None:
        """Initialize environmental effect defaults."""

        pass


    def step(self, action: np.ndarray, enable_smoothing: bool):
        """
        Update ship state based on actions and environmental effects.

        Args:
            action: array of [rudder, thrust] commands
            enable_smoothing: enable/disable smoothing
        """

        actual_rudder, actual_thrust = self.ship.apply_smoothing(action, enable_smoothing)

        # Save the current ship position as the previous position for tracking.
        # Update rudder controls based on the turning action.
        for rudder in self.dynamics.ship_interface.getRudderControls():
            rudder.setControlValue(float(-1.0*actual_rudder)) # this is to compensate for opposite behaviour of the Python environment
                                                        # in Python: -1 is turn right ; 1 is turn left
                                                        # FH sim: -1 is turn left ; 1 is turn right
        # Update propeller controls based on the thrust action.
        for propeller in self.dynamics.ship_interface.getPropellerControls():
            propeller.setEngineLeverValue(float(actual_thrust))

        # Simulate the ship's dynamics for a fixed period.
        self.dynamics.math_model.simulateSeconds(self.dt)

        # Retrieve the updated ship position from the ship interface.
        new_ship_pos = self.dynamics.ship_interface.getShipPosition()
        velocity_over_ground = self.dynamics.ship_interface.getShipVelocityOverGround()

        self._state[0] = new_ship_pos.x
        self._state[1] = new_ship_pos.y
        self._state[2] = (np.radians(self.dynamics.ship_interface.getShipHeading()) + np.pi) % (2.0 * np.pi) - np.pi
        self._state[3] = velocity_over_ground.x
        self._state[4] = velocity_over_ground.y
        self._state[5] = np.radians(self.dynamics.ship_interface.getShipYawRate())
