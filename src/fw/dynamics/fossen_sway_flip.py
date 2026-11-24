from typing import Tuple
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase


# Model constants
RUDDER_SPEED_FACTOR_DENOMINATOR = 3.0
MIN_RUDDER_EFFECTIVENESS = 0.2


class FossenSwayFlip(DynamicsBase):
    """
    Simplified 3-DOF (surge, sway, yaw) vessel dynamics after Fossen, T. I.
    (2011), Handbook of Marine Craft Hydrodynamics and Motion Control, Wiley.

    Includes added mass, Coriolis-centripetal terms, linear and quadratic
    damping, and simplified rudder and thrust forces. Meant for simulation and
    control design, not high-fidelity prediction.

    Frame and assumptions:
        - Body frame at the CoG, taken at midships (x_g = 0) as in the FH
          identification (FH ship data list x_g = 0.5 - 2.24 m; not modeled).
        - Framework axes (CONVENTIONS.md): x forward, y to port, yaw CCW positive.
        - No wind or current forces.
        - Coefficients are stored as ratios (divided by m, m*L or m*L^2, see
          ShipSpecifications) and scaled back with mass and length. Hull
          coefficients of MYZAKO_29/37 and TRIPOLI_38/45 come from the FH
          "Fossen model" identification; all other values are heuristic.
        - I_z comes from ShipSpecifications.I_z, else the slender-rod value
          m * L^2 / 12 (as used for the FH presets).
        - Equations follow Fossen_3DOF_Model.docx; the yaw Coriolis sign
          differs from Fossen (2011), see _compute_accelerations.

    Known limitation: the turning direction is correct (positive rudder turns
    to port), but the sway during a turn starts toward the inside of the turn
    instead of the outside. The turn is driven mainly by that sway through the
    u*v coupling terms, not by N_rudder, so flipping Y_rudder alone reverses
    the turn; fixing it needs a re-tune of N_rudder / N_uv (e.g. against FH).
    The sway is small (|v| < ~0.2 m/s, drift < 2 deg).
    """

    def __init__(self, ship_spec: ShipSpecifications):
        """
        Initialize the Fossen 3-DOF dynamics model.

        Args:
            ship_spec: Vessel properties and coefficient ratios.
        """

        self.ship_spec: ShipSpecifications = ship_spec
        self._initialize_coefficients()


    def _initialize_coefficients(self):
        """
        Scale the coefficient ratios to dimensional mass, damping, rudder and
        thrust coefficients.
        """

        # Ship parameters
        length = self.ship_spec.length  # Length (m)
        mass = self.ship_spec.mass  # Mass (kg)

        # Added mass coefficients
        self.X_udot = self.ship_spec.X_udot_coeff * mass  # Added mass
        self.Y_vdot = self.ship_spec.Y_vdot_coeff * mass  # Added mass
        self.N_rdot = self.ship_spec.N_rdot_coeff * mass * length ** 2.0  # Added inertia

        # Identified I_z if available, else the slender-rod value (overestimates
        # RIZZ of the Myzako and Tripoli hulls by ~29-38 %).
        i_z = self.ship_spec.I_z
        if i_z is None:
            i_z = mass * length ** 2.0 / 12.0

        # Mass matrix components
        self.m11 = mass - self.X_udot
        self.m22 = mass - self.Y_vdot
        self.m33 = i_z - self.N_rdot

        # Hydrodynamic damping coefficients
        self.X_u = self.ship_spec.X_u_coeff * mass  # Surge damping
        self.Y_v = self.ship_spec.Y_v_coeff * mass  # Sway damping
        self.N_r = self.ship_spec.N_r_coeff * mass * length ** 2.0    # Yaw damping

        # Nonlinear (quadratic) damping coefficients
        self.X_uu = self.ship_spec.X_uu_coeff * mass  # Quadratic damping
        self.Y_vv = self.ship_spec.Y_vv_coeff * mass  # Quadratic damping
        self.N_rr = self.ship_spec.N_rr_coeff * mass * length ** 2.0  # Quadratic damping

        # Rudder coefficients
        self.Y_rudder = self.ship_spec.Y_rudder_coeff * mass  # Sway force from rudder
        self.N_rudder = self.ship_spec.N_rudder_coeff * mass * length  # Yaw moment from rudder

        # Propeller/thrust coefficient
        self.X_thrust = self.ship_spec.X_thrust_coeff * mass  # Surge force from thrust

        # Cross-flow drag coefficients
        self.Y_uv = self.ship_spec.Y_uv_coeff * mass
        self.N_uv = self.ship_spec.N_uv_coeff * mass * length

    def _compute_accelerations(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
    ) -> Tuple[float, float, float]:
        """
        Compute body-fixed accelerations from state and control inputs.

        Args:
            state:        Current vessel state.
            rudder_angle: Rudder command [rad]; positive = the ship turns to PORT
                (framework convention, see DynamicsBase).
            thrust:       Normalized thrust command (dimensionless), positive forward.

        Returns:
            Tuple (du, dv, dr): surge, sway, and yaw accelerations [m/s², m/s², rad/s²].
        """

        # Notation follows Fossen_3DOF_Model.docx:
        #     M * nu_dot = tau - C(nu) nu - D(nu)
        # coriolis_* = the components of -C(nu) nu, d_* = the components of
        # -D(nu) = [X_D, Y_D, N_D], and the control terms below form tau.

        # --- CORIOLIS-CENTRIPETAL TERMS: -C(nu) nu ---
        # Fossen (2011) has -(m22 - m11) * u * v (Munk moment). The + sign is
        # deliberate: FH identified the Myzako/Tripoli coefficients with these
        # equations. Do not flip it without re-identifying the coefficients.
        coriolis_surge = self.m22 * state.v * state.r
        coriolis_sway = -self.m11 * state.u * state.r
        coriolis_yaw = (self.m22 - self.m11) * state.u * state.v

        # --- HYDRODYNAMIC DAMPING FORCES: -D(nu) = [X_D, Y_D, N_D] ---
        # X_D, Y_D, N_D are the damping forces acting on the ship.
        d_surge = self.X_u * state.u + self.X_uu * state.u * abs(state.u)
        d_sway = (self.Y_v * state.v +
                  self.Y_vv * state.v * abs(state.v) +
                  self.Y_uv * state.u * state.v)
        d_yaw = (self.N_r * state.r +
                 self.N_rr * state.r * abs(state.r) +
                 self.N_uv * state.u * state.v)

        # --- CONTROL FORCES: tau = [X_T, Y_R, N_R] ---
        # Speed-dependent rudder effectiveness kappa(u) = max(u / U_ref, kappa_min).
        speed_factor = max(state.u / RUDDER_SPEED_FACTOR_DENOMINATOR, MIN_RUDDER_EFFECTIVENESS)

        # Rudder lift (sway), scaled by the same speed_factor.
        f_rudder_sway = self.Y_rudder * rudder_angle * speed_factor

        m_rudder_yaw = self.N_rudder * rudder_angle * speed_factor

        f_thrust = self.X_thrust * thrust

        # --- CALCULATE ACCELERATIONS ---
        du = (f_thrust + d_surge + coriolis_surge) / self.m11
        dv = (f_rudder_sway + d_sway + coriolis_sway) / self.m22
        dr = (m_rudder_yaw + d_yaw + coriolis_yaw) / self.m33

        return du, dv, dr
