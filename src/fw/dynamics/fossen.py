from typing import Tuple
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
        length = 108 # self.ship_spec.length  # Length (m)
        mass = 4.19e+06 # self.ship_spec.mass  # Mass (kg)

        # Added mass coefficients
        self.X_udot = -0.265 * mass  # Added mass
        self.Y_vdot = -0.86 * mass  # Added mass
        self.N_rdot = -0.047 * mass * length ** 2.0  # Added inertia

        # Mass matrix components
        self.m11 = mass - self.X_udot
        self.m22 = mass - self.Y_vdot
        self.m33 = (mass * length ** 2.0 / 12.0) - self.N_rdot

        # Hydrodynamic damping coefficients
        self.X_u = 6.712e-18 * mass  # Surge damping
        self.Y_v = -0.0018 * mass  # Sway damping
        self.N_r = 0.000806 * mass * length ** 2.0    # Yaw damping

        # Nonlinear (quadratic) damping coefficients
        self.X_uu = -0.0015 * mass  # Quadratic damping
        self.Y_vv = -0.0566 * mass  # Quadratic damping
        self.N_rr = -0.0362 * mass * length ** 2.0  # Quadratic damping

        # Rudder coefficients
        self.Y_rudder = 0.1 * mass  # Sway force from rudder
        self.N_rudder = 0.001 * mass * length  # Yaw moment from rudder

        # Propeller/thrust coefficient
        self.X_thrust = 0.05 * mass  # Surge force from thrust

        # Cross-flow drag coefficients
        self.Y_uv = -0.00467 * mass
        self.N_uv = -0.00007 * mass * length


    def accelerations(self, u: float, v: float, r: float,
                      rudder_angle: float, thrust: float) -> Tuple[float, float, float]:
        """
        Compute body-fixed accelerations from state and control inputs.

        Computes the surge, sway, and yaw accelerations using a nonlinear
        3-DOF maneuvering model. Rudder effectiveness is scaled by forward
        speed to reflect reduced control authority at low speeds, with a
        lower bound to preserve basic maneuverability.

        Args:
            u: Body-fixed surge velocity in meters per second (m/s).
            v: Body-fixed sway velocity in meters per second (m/s).
            r: Body-fixed yaw rate in radians per second (rad/s).
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
