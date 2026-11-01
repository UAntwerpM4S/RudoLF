from typing import Tuple
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase


# Model constants
RUDDER_SPEED_FACTOR_DENOMINATOR = 3.0
MIN_RUDDER_EFFECTIVENESS = 0.2


class Fossen(DynamicsBase):
    """
    Three-degrees-of-freedom vessel dynamics model based on Fossen.

    This class implements a simplified 3-DOF horizontal-plane vessel
    dynamics model (surge, sway, yaw) inspired by the formulations in:

        Fossen, T. I. (2011). *Handbook of Marine Craft Hydrodynamics and
        Motion Control*. Wiley.

    The model captures added mass effects, Coriolis and centripetal terms,
    linear and nonlinear hydrodynamic damping, and simplified control
    forces from rudder and propeller thrust. It is intended for simulation
    and control design rather than high-fidelity prediction.

    Coordinate frame and assumptions:
        - Body-fixed reference frame located at the vessel center of gravity.
        - Axes: the framework's (fw.ships.vessel_state, CONVENTIONS.md):
          x forward, y to PORT, yaw counter-clockwise positive.
        - Motion is restricted to the horizontal plane (3-DOF).
        - Environmental forces (wind, current) are not included.
        - Hydrodynamic coefficients are stored per ship as ratios (dimensional
          value divided by m, m*L or m*L^2, see ShipSpecifications) and scaled
          back with the vessel's mass and length. For Myzako (MYZAKO_29/37) and Tripoli
          (TRIPOLI_38/45) the hull coefficients come from the FH "Fossen
          model" identification; other presets, and the rudder and thrust
          coefficients of all presets, are heuristic.
        - The reference point is the CoG, taken at midships (x_g = 0), as in
          the FH identification. The FH ship data list x_g = 0.5 - 2.24 m;
          this offset is not modelled.
        - Rigid-body yaw inertia I_z comes from ShipSpecifications.I_z when
          given, else from the slender-rod approximation m * L^2 / 12. The
          FH presets use the slender-rod value, as in the FH identification.
        - Equations and notation follow Fossen_3DOF_Model.docx. The yaw
          Coriolis term (m22 - m11) * u * v has the opposite sign to the Munk
          moment in Fossen (2011); see _compute_accelerations.

    Known limitation (toy model): the turning DIRECTION follows the framework
    convention (positive rudder -> turn to port), but the sway during a turn
    does not follow real ship behaviour: it starts toward the inside of the
    turn (port for a port turn) instead of the outside. The turn is driven
    mainly by that sway force through the u*v coupling terms (N_uv,
    (m22 - m11) * u * v), not by N_rudder, so flipping Y_rudder alone reverses
    the turning direction (checked 2026-10-19); correcting it needs a re-tune
    of N_rudder / N_uv, e.g. against FH. The sway concerned is small
    (|v| < ~0.2 m/s, drift < 2 deg).
    """

    def __init__(self, ship_spec: ShipSpecifications):
        """
        Initialize the Fossen 3-DOF dynamics model.

        Args:
            ship_spec: Container holding the vessel's physical properties
                (e.g. mass, length) used to parameterize the dynamics model.
        """

        self.ship_spec: ShipSpecifications = ship_spec
        self._initialize_coefficients()


    def _initialize_coefficients(self):
        """
        Initialize mass, damping, and control coefficients.

        Derives all hydrodynamic coefficients from the vessel's physical
        properties. The resulting parameters define:
            - Rigid-body and added-mass terms
            - Linear and quadratic damping
            - Rudder-induced forces and moments
            - Propeller thrust effectiveness

        This method is intended to be called once during initialization.
        """

        # Ship parameters
        length = self.ship_spec.length  # Length (m)
        mass = self.ship_spec.mass  # Mass (kg)

        # Added mass coefficients
        self.X_udot = self.ship_spec.X_udot_coeff * mass  # Added mass
        self.Y_vdot = self.ship_spec.Y_vdot_coeff * mass  # Added mass
        self.N_rdot = self.ship_spec.N_rdot_coeff * mass * length ** 2.0  # Added inertia

        # Rigid-body yaw inertia: identified value when available, otherwise
        # the slender-rod approximation (overestimates the RIZZ of the Myzako and Tripoli hulls by ~29-38 %).
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
        # NOTE: Fossen (2011) gives -(m22 - m11) * u * v for the yaw term
        # (destabilising Munk moment). The + sign below is deliberate: FH
        # identified the MYZAKO_29/37 and TRIPOLI_38/45 coefficients with
        # exactly these equations, so the fit absorbs it. Do not flip the
        # sign without re-identifying the coefficients.
        coriolis_surge = self.m22 * state.v * state.r
        coriolis_sway = -self.m11 * state.u * state.r
        coriolis_yaw = (self.m22 - self.m11) * state.u * state.v

        # --- HYDRODYNAMIC DAMPING FORCES: -D(nu) = [X_D, Y_D, N_D] ---
        # X_D, Y_D, N_D are the damping forces as they act on the ship; the
        # minus sign in D(nu) = -[X_D, Y_D, N_D] follows from the equation form.
        d_surge = self.X_u * state.u + self.X_uu * state.u * abs(state.u)
        d_sway = (self.Y_v * state.v +
                  self.Y_vv * state.v * abs(state.v) +
                  self.Y_uv * state.u * state.v)
        d_yaw = (self.N_r * state.r +
                 self.N_rr * state.r * abs(state.r) +
                 self.N_uv * state.u * state.v)

        # --- CONTROL FORCES: tau = [X_T, Y_R, N_R] ---
        # Speed-dependent rudder effectiveness kappa(u) = max(u / U_ref, kappa_min)
        # At low speeds, rudder is less effective
        speed_factor = max(state.u / RUDDER_SPEED_FACTOR_DENOMINATOR, MIN_RUDDER_EFFECTIVENESS)

        # Direct sway force from the rudder (rudder lift). Note that it is
        # scaled by the same speed_factor, so it does not add extra authority
        # at low speed.
        f_rudder_sway = self.Y_rudder * rudder_angle * speed_factor

        # Yaw moment from the rudder
        m_rudder_yaw = self.N_rudder * rudder_angle * speed_factor

        # Thrust force
        f_thrust = self.X_thrust * thrust

        # --- CALCULATE ACCELERATIONS ---
        du = (f_thrust + d_surge + coriolis_surge) / self.m11
        dv = (f_rudder_sway + d_sway + coriolis_sway) / self.m22
        dr = (m_rudder_yaw + d_yaw + coriolis_yaw) / self.m33

        return du, dv, dr
