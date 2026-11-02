import numpy as np

from typing import Tuple
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase


class Nomoto(DynamicsBase):
    """
    Nomoto-inspired, low-order 3-DOF (u, v, r) maneuvering model for control
    design, RL and fast-time simulation; not for high-fidelity hydrodynamics.

    Extends the first-order Nomoto yaw equation with heuristic surge, sway and
    yaw coupling terms. Assumptions:
        - Body-fixed frame at the CG, framework axes (x forward, y to port,
          yaw counter-clockwise positive; see CONVENTIONS.md)
        - No wind, waves or current
        - Heuristic coefficients, identical for every ship (ship_spec is ignored)

    Known limitation: the turn direction follows the framework convention, but
    sway goes toward the inside of the turn (k_v > 0) instead of the outside.
    The effect is small (drift ~0.5 deg, ~4 % turning-rate change), so it is
    left as is.
    """

    def __init__(self, ship_spec: ShipSpecifications):
        """
        Initialize the Nomoto maneuvering model.

        Args:
            ship_spec: Vessel specifications (stored, not used by the dynamics).
        """

        self.ship_spec: ShipSpecifications = ship_spec
        self._initialize_coefficients()

    def _initialize_coefficients(self):
        """
        Set heuristic coefficients giving qualitatively reasonable behavior.
        """

        # Linear damping and control effectiveness coefficients
        self.xu = -0.02      # Surge damping
        self.yv = -0.4       # Sway damping
        self.yv_r = -0.09    # Sway-yaw coupling
        self.nr = -0.36      # Yaw-rate damping

        # Characteristic length scale (used for cross-coupling)
        self.l = 50.0

        # Control gains
        self.k_t = 0.05      # Thrust effectiveness
        self.k_r = 0.039     # Rudder-to-yaw effectiveness
        self.k_v = 0.03      # Rudder-to-sway effectiveness


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

        du = self.k_t * (thrust * 60.0) + self.xu * state.u
        dv = self.k_v * np.sin(rudder_angle) + self.yv * state.v
        dr = (self.k_r * rudder_angle + self.nr * state.r + self.yv_r * state.v +
              (state.v * state.u) / self.l)

        return du, dv, dr
