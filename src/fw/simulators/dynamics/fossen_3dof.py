from typing import Tuple
from fw.simulators.ships.ship_specs import ShipSpecifications
from fw.simulators.dynamics.dynamics_model import DynamicsModel


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


    def accelerations(self, u: float, v: float, r: float,
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
