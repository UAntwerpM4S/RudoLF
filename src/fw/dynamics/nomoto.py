import numpy as np

from typing import Tuple
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase


class Nomoto(DynamicsBase):
    """
    Nomoto-inspired low-order vessel maneuvering model.

    This class implements a simplified, control-oriented horizontal-plane
    maneuvering model with three degrees of freedom (3-DOF):
    surge (u), sway (v), and yaw (r).

    The formulation is *inspired by* Nomoto-type models but extends beyond
    the classical first-order Nomoto yaw equation by explicitly modeling
    surge, sway, and yaw dynamics with heuristic coupling terms.

    The model is intended for:
        - Control design
        - Reinforcement learning
        - Fast-time simulation

    It is **not** intended for high-fidelity hydrodynamic prediction or
    seakeeping analysis.

    Coordinate frame and assumptions:
        - Body-fixed reference frame located at the vessel center of gravity (CG)
        - Axes: x forward, y starboard, z downward (right-handed)
        - Motion restricted to the horizontal plane (3-DOF)
        - No environmental disturbances (wind, waves, current)
        - Hydrodynamic coefficients are heuristic and weakly scaled
          to vessel dimensions rather than identified from experiments
    """

    def __init__(self, ship_spec: ShipSpecifications):
        """
        Initialize the Nomoto maneuvering model.

        Args:
            ship_spec: Vessel physical specifications used to parameterize
                the dynamics model (e.g., mass, length, actuator limits).
        """

        self.ship_spec: ShipSpecifications = ship_spec
        self._initialize_coefficients()

    def _initialize_coefficients(self):
        """
        Initialize hydrodynamic and control coefficients.

        Coefficients are intentionally simple and heuristic, chosen to yield
        qualitatively reasonable vessel behavior rather than quantitatively
        accurate hydrodynamics.
        """

        # Linear damping and control effectiveness coefficients
        self.xu = -0.02      # Surge damping
        self.yv = -0.4       # Sway damping
        self.yv_r = -0.09    # Sway–yaw coupling
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
            rudder_angle: Rudder deflection angle [rad], positive to starboard.
            thrust:       Normalized thrust command (dimensionless), positive forward.

        Returns:
            Tuple (du, dv, dr) — surge, sway, and yaw accelerations [m/s², m/s², rad/s²].
        """

        # Simplified dynamics
        du = self.k_t * (thrust * 60.0) + self.xu * state.u
        dv = self.k_v * np.sin(rudder_angle) + self.yv * state.v
        dr = (self.k_r * rudder_angle + self.nr * state.r + self.yv_r * state.v +
              (state.v * state.u) / self.l)

        return du, dv, dr
