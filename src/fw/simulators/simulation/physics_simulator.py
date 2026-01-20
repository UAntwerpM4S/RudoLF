import numpy as np

from fw.simulators.ships.ship import Ship
from fw.simulators.dynamics.base import DynamicsModel


class PhysicsSimulator:
    def __init__(self, ship: Ship, dynamics: DynamicsModel, dt: float, wind: bool, current: bool):
        self.dt = dt
        self.ship = ship
        self.wind = wind
        self.current = current
        self.dynamics = dynamics

        self._initialize_control_parameters()


    def _initialize_control_parameters(self) -> None:
        """Initialize environmental effect defaults."""

        self.radians_current = np.radians(180.0)
        self.current_direction = np.array([np.cos(self.radians_current), np.sin(self.radians_current)],
                                          dtype=np.float32)
        self.current_strength = 0.35

        self.radians_wind = np.radians(90.0)
        self.wind_direction = np.array([np.cos(self.radians_wind), np.sin(self.radians_wind)], dtype=np.float32)
        self.wind_strength = 0.35


    def step(self, state: np.ndarray, action: np.ndarray, enable_smoothing: bool) -> np.ndarray:
        """
        Update ship state based on actions and environmental effects.

        Args:
            state: current state of ship
            action: array of [rudder, thrust] commands
            enable_smoothing: enable/disable smoothing
        """

        # Convert control inputs to physical values
        # Rudder: -1 to 1 maps to -60° to 60° (typical ship rudder limits)
        target_rudder = action[0] * self.ship.specifications.max_rudder_angle   # rudder angle in radians
        # Thrust: 0 to 1 maps to 0 to full ahead
        target_thrust = (action[1] if enable_smoothing else abs(action[1])) * self.ship.specifications.max_thrust

        actual_rudder, actual_thrust = self.ship.apply_control([target_rudder, target_thrust], self.dt, enable_smoothing)

        # Environmental effects relative to ship heading
        wind_effect = np.zeros(2, dtype=np.float32)
        if self.wind:
            relative_wind_angle = self.radians_wind - state[2]
            wind_effect = np.array([
                self.wind_strength * np.cos(relative_wind_angle),
                self.wind_strength * np.sin(relative_wind_angle)
            ], dtype=np.float32)

        current_effect = np.zeros(2, dtype=np.float32)
        if self.current:
            relative_current_angle = self.radians_current - state[2]
            current_effect = np.array([
                self.current_strength * np.cos(relative_current_angle),
                self.current_strength * np.sin(relative_current_angle)
            ], dtype=np.float32)

        du, dv, dr = self.dynamics.calculate_accelerations(state[3], state[4], state[5], actual_rudder, actual_thrust)

        # Add environmental effects as additional accelerations
        du += wind_effect[0] + current_effect[0]
        dv += wind_effect[1] + current_effect[1]

        return np.array(self.dynamics.integrate(state, [du, dv, dr], self.dt), dtype=np.float32)
