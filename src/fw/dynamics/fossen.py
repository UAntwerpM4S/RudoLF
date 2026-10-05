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
        - Axes: x forward, y to PORT, z up;
          r counter-clockwise positive; positive rudder turns to starboard (r < 0).
        - Motion is restricted to the horizontal plane (3-DOF).
        - Environmental forces (wind, current) are not included.
        - Coefficients are heuristic and scale with vessel mass and length.
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

        # Mass matrix components
        self.m11 = mass - self.X_udot
        self.m22 = mass - self.Y_vdot
        self.m33 = (mass * length ** 2.0 / 12.0) - self.N_rdot

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
            rudder_angle: Rudder deflection angle [rad], positive to starboard.
            thrust:       Normalized thrust command (dimensionless), positive forward.

        Returns:
            Tuple (du, dv, dr) — surge, sway, and yaw accelerations [m/s², m/s², rad/s²].
        """

        # --- CORIOLIS-CENTRIPETAL MATRIX ---
        coriolis_surge = self.m22 * state.v * state.r
        coriolis_sway = -self.m11 * state.u * state.r
        coriolis_yaw = (self.m22 - self.m11) * state.u * state.v

        # --- HYDRODYNAMIC DAMPING FORCES ---
        d_surge = self.X_u * state.u + self.X_uu * state.u * abs(state.u)
        d_sway = (self.Y_v * state.v +
                  self.Y_vv * state.v * abs(state.v) +
                  self.Y_uv * state.u * state.v)
        d_yaw = (self.N_r * state.r +
                 self.N_rr * state.r * abs(state.r) +
                 self.N_uv * state.u * state.v)

        # --- CONTROL FORCES ---
        # The rudder coefficients (Y_rudder_coeff, N_rudder_coeff > 0 by
        # default) were written for "positive rudder turns to port". The
        # framework convention is the opposite (positive = starboard, r < 0),
        # so the rudder input is mirrored. A starboard command now produces
        # exactly the mirror image of what a port command produced before.
        delta = -rudder_angle

        # Speed-dependent rudder effectiveness
        # At low speeds, rudder is less effective
        speed_factor = max(state.u / RUDDER_SPEED_FACTOR_DENOMINATOR, MIN_RUDDER_EFFECTIVENESS)

        # Rudder effectiveness
        f_rudder_sway = self.Y_rudder * delta * speed_factor

        # IMPORTANT: Add direct sway force from rudder (helps initial turn)
        # Ships create sideways force when rudder is applied
        m_rudder_yaw = self.N_rudder * delta * speed_factor

        # Thrust force
        f_thrust = self.X_thrust * thrust

        # --- CALCULATE ACCELERATIONS ---
        du = (f_thrust + d_surge + coriolis_surge) / self.m11
        dv = (f_rudder_sway + d_sway + coriolis_sway) / self.m22
        dr = (m_rudder_yaw + d_yaw + coriolis_yaw) / self.m33

        return du, dv, dr
