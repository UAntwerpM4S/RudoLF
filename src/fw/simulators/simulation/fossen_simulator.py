import numpy as np

from fw.simulators.ships.ship import Ship


class FossenSimulator:
    def __init__(self, ship: Ship, dynamics, initial_ship_pos, initial_ship_heading, dt: float, wind: bool, current: bool):
        self.dt = dt
        self.ship = ship
        self.wind = wind
        self.current = current
        self.dynamics = dynamics
        self._state = np.array([initial_ship_pos[0], initial_ship_pos[1], initial_ship_heading, 0.0, 0.0, 0.0], dtype=np.float32)

        self._initialize_control_parameters()


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

        self.radians_current = np.radians(180.0)
        self.current_direction = np.array([np.cos(self.radians_current), np.sin(self.radians_current)],
                                          dtype=np.float32)
        self.current_strength = 0.35

        self.radians_wind = np.radians(90.0)
        self.wind_direction = np.array([np.cos(self.radians_wind), np.sin(self.radians_wind)], dtype=np.float32)
        self.wind_strength = 0.35


    def step(self, action: np.ndarray, enable_smoothing: bool):
        """
        Update ship state based on actions and environmental effects.

        Args:
            action: array of [rudder, thrust] commands
            enable_smoothing: enable/disable smoothing
        """

        x, y, psi, u, v, r = self._state

        # Convert control inputs to physical values
        # Rudder: -1 to 1 maps to -60° to 60° (typical ship rudder limits)
        target_rudder = action[0] * self.ship.specifications.max_rudder_angle   # rudder angle in radians
        # Thrust: -1 to 1 maps to min_thrust to max_thrust
        target_thrust = (self.ship.specifications.min_thrust + 0.5 * (action[1] + 1.0) *
                         (self.ship.specifications.max_thrust - self.ship.specifications.min_thrust))

        actual_rudder, actual_thrust = self.ship.apply_control([target_rudder, target_thrust], self.dt, enable_smoothing)

        # Environmental effects relative to ship heading
        wind_effect = np.zeros(2, dtype=np.float32)
        if self.wind:
            relative_wind_angle = self.radians_wind - psi
            wind_effect = np.array([
                self.wind_strength * np.cos(relative_wind_angle),
                self.wind_strength * np.sin(relative_wind_angle)
            ], dtype=np.float32)

        current_effect = np.zeros(2, dtype=np.float32)
        if self.current:
            relative_current_angle = self.radians_current - psi
            current_effect = np.array([
                self.current_strength * np.cos(relative_current_angle),
                self.current_strength * np.sin(relative_current_angle)
            ], dtype=np.float32)

        du, dv, dr = self.dynamics.calculate_accelerations(u, v, r, actual_rudder, actual_thrust)

        # Add environmental effects as additional accelerations
        du += wind_effect[0] + current_effect[0]
        dv += wind_effect[1] + current_effect[1]

        # Surge integration (semi-implicit for damping)
        surge_damping_factor = abs(self.dynamics.X_u / self.dynamics.m11)
        new_u = (u + du * self.dt) / (1.0 + surge_damping_factor * self.dt)

        # Sway integration (semi-implicit for damping)
        sway_damping_factor = abs(self.dynamics.Y_v / self.dynamics.m22)
        new_v = (v + dv * self.dt) / (1.0 + sway_damping_factor * self.dt)

        # Yaw integration (semi-implicit for damping)
        yaw_damping_factor = abs(self.dynamics.N_r / self.dynamics.m33)
        new_r = (r + dr * self.dt) / (1.0 + yaw_damping_factor * self.dt)

        # Apply velocity limits
        new_u = np.clip(new_u, self.ship.specifications.min_surge_velocity, self.ship.specifications.max_surge_velocity)
        new_v = np.clip(new_v, self.ship.specifications.min_sway_velocity, self.ship.specifications.max_sway_velocity)
        new_r = np.clip(new_r, self.ship.specifications.min_yaw_rate, self.ship.specifications.max_yaw_rate)

        # Position integration in world coordinates
        # Use midpoint heading for better accuracy
        psi_mid = psi + 0.5 * new_r * self.dt
        cos_psi_mid = np.cos(psi_mid)
        sin_psi_mid = np.sin(psi_mid)

        # Earth-fixed velocity components
        dx = new_u * cos_psi_mid - new_v * sin_psi_mid
        dy = new_u * sin_psi_mid + new_v * cos_psi_mid

        # Update state
        self._state[0] = x + dx * self.dt
        self._state[1] = y + dy * self.dt
        self._state[2] = (psi + new_r * self.dt + np.pi) % (2.0 * np.pi) - np.pi
        self._state[3] = new_u
        self._state[4] = new_v
        self._state[5] = new_r
