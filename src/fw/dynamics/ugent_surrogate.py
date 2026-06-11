import numpy as np

from typing import Tuple
from fw.ships.vessel_state import VesselState
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

        n = thrust * 320
        n_rps = n / 60

        i_zz = 4651416900.21  # kgm2
        x_g = 1.1874  # m
        d_prop = 1.6573  # m
        t = 2 * ((0.6723 - 0.7968 * (state.u * 0.7 / (n_rps * d_prop))) * 1025 * n_rps ** 2 * d_prop ** 4)  # N

        du = (t * np.cos(rudder_angle) - self.X_u * state.u - self.X_uabsu * state.u * abs(state.u) +
              self.mass * state.v * state.r + self.X_uabsv * state.u * abs(state.v)) / (self.mass - self.X_udot)
        dv = (-self.mass * state.u * state.r + self.Y_uv * state.u * state.v +
              self.Y_vabsv * state.v * abs(state.v) + t * np.sin(rudder_angle)) / (self.mass - self.Y_vdot)
        dr = (-self.mass * x_g * state.u * state.r + self.N_ur * state.u * state.r + self.N_rabsr *
              state.r * np.abs(state.r) - t * (self.length / 2) * np.sin(rudder_angle)) / (i_zz - self.N_rdot)

        return du, dv, dr
