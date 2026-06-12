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
        - Axes: x forward, y starboard, z downward.
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
        self.Y_rudder = 0.1 * mass  # Sway force from rudder
        self.N_rudder = 0.001 * mass * length  # Yaw moment from rudder

        # Propeller/thrust coefficient
        self.X_thrust = 0.05 * mass  # Surge force from thrust

        # Cross-flow drag coefficients
        self.Y_uv = self.ship_spec.Y_uv_coeff * mass
        self.N_uv = self.ship_spec.N_uv_coeff * mass * length


    def accelerations(self, state: VesselState,
                      rudder_angle: float, thrust: float) -> Tuple[float, float, float]:
        """
        Compute body-fixed accelerations from state and control inputs.

        Computes the surge, sway, and yaw accelerations using a nonlinear
        3-DOF maneuvering model. Rudder effectiveness is scaled by forward
        speed to reflect reduced control authority at low speeds, with a
        lower bound to preserve basic maneuverability.

        Args:
            state: The current vessel state.
            rudder_angle: Rudder deflection angle in radians, positive to
                starboard.
            thrust: Normalized thrust command (dimensionless), positive in
                the forward direction.

        Returns:
            A tuple `(du, dv, dr)` where:
                - du: Surge acceleration (m/s²).
                - dv: Sway acceleration (m/s²).
                - dr: Yaw acceleration (rad/s²).

        Notes:
            - Coriolis and centripetal effects are included explicitly.
            - Rudder forces contribute both sway force and yaw moment.
            - The model assumes thrust acts purely in surge.
            - This formulation is suitable for control and reinforcement
              learning, but not for detailed seakeeping analysis.
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
        # Speed-dependent rudder effectiveness
        # At low speeds, rudder is less effective
        speed_factor = max(state.u / RUDDER_SPEED_FACTOR_DENOMINATOR, MIN_RUDDER_EFFECTIVENESS)

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
