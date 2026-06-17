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

        assert thrust >= 0.0, (
            f"UGentSurrogate expects a non-negative thrust value, got {thrust}."
        )

        n = thrust * 320
        n_rps = n / 60

        i_zz = 4651416900.21  # kgm2
        x_g = 1.1874  # m
        d_prop = 1.6573  # m

        if n_rps < 1e-6:
            t = 0.0
        else:
            t = 2 * ((0.6723 - 0.7968 * (state.u * 0.7 / (n_rps * d_prop))) * 1025 * n_rps ** 2 * d_prop ** 4)  # N

        du = (t * np.cos(rudder_angle) - self.X_u * state.u - self.X_uabsu * state.u * abs(state.u) +
              self.mass * state.v * state.r + self.X_uabsv * state.u * abs(state.v)) / (self.mass - self.X_udot)
        dv = (-self.mass * state.u * state.r + self.Y_uv * state.u * state.v +
              self.Y_vabsv * state.v * abs(state.v) + t * np.sin(rudder_angle)) / (self.mass - self.Y_vdot)
        dr = (-self.mass * x_g * state.u * state.r + self.N_ur * state.u * state.r + self.N_rabsr *
              state.r * np.abs(state.r) - t * (self.length / 2) * np.sin(rudder_angle)) / (i_zz - self.N_rdot)

        return du, dv, dr
