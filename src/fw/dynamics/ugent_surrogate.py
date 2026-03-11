import numpy as np

from typing import Tuple
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase


class UGentSurrogate(DynamicsBase):
    """
    Three-degrees-of-freedom vessel dynamics model based on UGent's research.

    This class implements a simplified 3-DOF horizontal-plane vessel
    dynamics model (surge, sway, yaw).

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
        Initialize the UGentSurrogate 3-DOF dynamics model.

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
        self.mass = 6161870.27  # kg
        self.length = 110  # m

        self.X_u = 1995.729394734604
        self.X_uabsu = 2678.2760354288225
        self.X_udot = -1148854.9267941066
        self.X_uabsv = -38200.00534358345
        self.Y_vdot = -3321096.011059891
        self.Y_uv = -85501.698921936
        self.Y_vabsv = -139770.31875187394
        self.N_rdot = -1611696438.7519011
        self.N_ur = -1.484041721506744e-11
        self.N_rabsr = -3720289534.3422813


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

        n = thrust * 320
        n_rps = n / 60

        I_zz = 4651416900.21  # kgm2
        x_G = 1.1874  # m
        D_prop = 1.6573  # m
        T = 2 * ((0.6723 - 0.7968 * (u * 0.7 / (n_rps * D_prop))) * 1025 * n_rps ** 2 * D_prop ** 4)  # N

        du = (T * np.cos(rudder_angle) - self.X_u * u - self.X_uabsu * u * abs(u) + self.mass * v * np.deg2rad(
            r) + self.X_uabsv * u * abs(v)) / (self.mass - self.X_udot)
        dv = (-self.mass * u * r + self.Y_uv * u * v + self.Y_vabsv * v * abs(v) + T * np.sin(rudder_angle)) / (
                    self.mass - self.Y_vdot)
        dr = (-self.mass * x_G * u * r + self.N_ur * u * r + self.N_rabsr * r * np.abs(
            r) - T * (self.length / 2) * np.sin(rudder_angle)) / (I_zz - self.N_rdot)

        return du, dv, dr
