import numpy as np

from typing import Tuple
from fw.simulators.dynamics.base import DynamicsModel
from fw.simulators.ships.ship import ShipSpecifications


# Model constants
RUDDER_SPEED_FACTOR_DENOMINATOR = 3.0
MIN_RUDDER_EFFECTIVENESS = 0.2


class Fossen3DOF(DynamicsModel):
    def __init__(self, ship_spec: ShipSpecifications):
        self.ship_spec: ShipSpecifications = ship_spec
        self._initialize_coefficients()


    def _initialize_coefficients(self):
        """
        Initialize with ship parameters
        """

        # Ship parameters
        length = self.ship_spec.length  # Length (m)
        mass = self.ship_spec.mass  # Mass (kg)

        # Added mass coefficients
        self.X_udot = -0.05 * mass  # Added mass
        self.Y_vdot = -0.5 * mass  # Added mass
        self.N_rdot = -0.05 * mass * length ** 2.0  # Added inertia

        # Mass matrix components
        self.m11 = mass - self.X_udot
        self.m22 = mass - self.Y_vdot
        self.m33 = (mass * length ** 2.0 / 12.0) - self.N_rdot

        # Hydrodynamic damping coefficients
        self.X_u = -0.002 * mass  # Surge damping
        self.Y_v = -0.02 * mass  # Sway damping
        self.N_r = -0.001 * mass * length ** 2.0    # Yaw damping

        # Nonlinear (quadratic) damping coefficients
        self.X_uu = -0.0005 * mass  # Quadratic damping
        self.Y_vv = -0.005 * mass  # Quadratic damping
        self.N_rr = -0.0005 * mass * length ** 2.0  # Quadratic damping

        # Rudder coefficients
        self.Y_rudder = 0.1 * mass  # Sway force from rudder
        self.N_rudder = 0.001 * mass * length  # Yaw moment from rudder

        # Propeller/thrust coefficient
        self.X_thrust = 0.05 * mass  # Surge force from thrust

        # Cross-flow drag coefficients
        self.Y_uv = -0.005 * mass
        self.N_uv = -0.0005 * mass * length


    def calculate_accelerations(self, u: float, v: float, r: float,
                                rudder_angle: float, thrust: float) -> Tuple[float, float, float]:
        """
        Calculate accelerations - with speed-dependent rudder effectiveness
        """

        # --- CORIOLIS-CENTRIPETAL MATRIX ---
        coriolis_surge = self.m22 * v * r
        coriolis_sway = -self.m11 * u * r
        coriolis_yaw = (self.m22 - self.m11) * u * v

        # --- HYDRODYNAMIC DAMPING FORCES ---
        d_surge = self.X_u * u + self.X_uu * u * abs(u)
        d_sway = (self.Y_v * v +
                  self.Y_vv * v * abs(v) +
                  self.Y_uv * u * v)
        d_yaw = (self.N_r * r +
                 self.N_rr * r * abs(r) +
                 self.N_uv * u * v)

        # --- CONTROL FORCES ---
        # Speed-dependent rudder effectiveness
        # At low speeds, rudder is less effective
        speed_factor = max(u / RUDDER_SPEED_FACTOR_DENOMINATOR, MIN_RUDDER_EFFECTIVENESS)

        # Rudder effectiveness
        f_rudder_sway = self.Y_rudder * rudder_angle * speed_factor

        # IMPORTANT: Add direct sway force from rudder (helps initial turn)
        # Ships create sideways force when rudder is applied
        m_rudder_yaw = self.N_rudder * rudder_angle * speed_factor

        # Thrust force
        f_thrust = self.X_thrust * thrust

        # --- COMBINE ALL FORCES/MOMENTS ---
        total_surge_force = (f_thrust + d_surge + coriolis_surge)
        total_sway_force = (f_rudder_sway + d_sway + coriolis_sway)
        total_yaw_moment = (m_rudder_yaw + d_yaw + coriolis_yaw)

        # --- CALCULATE ACCELERATIONS ---
        du = total_surge_force / self.m11
        dv = total_sway_force / self.m22
        dr = total_yaw_moment / self.m33

        return du, dv, dr


    def integrate(self, state: np.ndarray, accelerations: Tuple[float, float, float],
                  dt: float) -> np.ndarray:
        x, y, psi, u, v, r = state
        du, dv, dr = accelerations

        # Surge integration (semi-implicit for damping)
        surge_damping_factor = abs(self.X_u / self.m11)
        new_u = (u + du * dt) / (1.0 + surge_damping_factor * dt)

        # Sway integration (semi-implicit for damping)
        sway_damping_factor = abs(self.Y_v / self.m22)
        new_v = (v + dv * dt) / (1.0 + sway_damping_factor * dt)

        # Yaw integration (semi-implicit for damping)
        yaw_damping_factor = abs(self.N_r / self.m33)
        new_r = (r + dr * dt) / (1.0 + yaw_damping_factor * dt)

        # Apply realistic limits
        state[3] = np.clip(new_u, self.ship_spec.min_surge_velocity, self.ship_spec.max_surge_velocity)
        state[4] = np.clip(new_v, self.ship_spec.min_sway_velocity, self.ship_spec.max_sway_velocity)
        state[5] = np.clip(new_r, self.ship_spec.min_yaw_rate, self.ship_spec.max_yaw_rate)

        # Position integration in world coordinates
        # Use midpoint heading for better accuracy
        psi_mid = psi + 0.5 * state[5] * dt
        cos_psi_mid = np.cos(psi_mid)
        sin_psi_mid = np.sin(psi_mid)

        # Earth-fixed velocity components
        dx = state[3] * cos_psi_mid - state[4] * sin_psi_mid
        dy = state[3] * sin_psi_mid + state[4] * cos_psi_mid

        # Update position and heading
        state[0] = x + dx * dt
        state[1] = y + dy * dt
        state[2] = (psi + state[5] * dt + np.pi) % (2.0 * np.pi) - np.pi
