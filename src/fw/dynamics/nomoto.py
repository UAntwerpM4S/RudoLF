import numpy as np

from typing import Tuple
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase


YAW_RATE_DAMPING = 0.1


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

        Notes:
            - All hydrodynamic and control coefficients are derived or
              initialized during construction.
            - The model state itself is not stored internally; only
              accelerations are computed from inputs.
        """
        self.ship_spec: ShipSpecifications = ship_spec
        self._initialize_coefficients()


    def _initialize_coefficients(self):
        """
        Initialize hydrodynamic and control coefficients.

        This method defines all coefficients used in the maneuvering model,
        including:
            - Linear velocity damping terms
            - Yaw-rate damping
            - Rudder-induced sway force and yaw moment
            - Thrust effectiveness in surge

        The coefficients are intentionally simple and heuristic, chosen
        to yield qualitatively reasonable vessel behavior rather than
        quantitatively accurate hydrodynamics.

        Notes:
            - This method is intended to be called once during initialization.
            - Coefficients are currently hard-coded and do not yet scale
              explicitly with vessel mass or length.
        """

        # Linear damping and control effectiveness coefficients
        self.xu = -0.02      # Surge damping
        self.yv = -0.4       # Sway damping
        self.yv_r = -0.09    # Sway–yaw coupling
        self.nr = -0.26      # Yaw-rate damping

        # Characteristic length scale (used for cross-coupling)
        self.l = 50.0

        # Control gains
        self.k_t = 0.05      # Thrust effectiveness
        self.k_r = 0.039     # Rudder-to-yaw effectiveness
        self.k_v = 0.03      # Rudder-to-sway effectiveness


    def accelerations(
        self,
        u: float,
        v: float,
        r: float,
        rudder_angle: float,
        thrust: float
    ) -> Tuple[float, float, float]:
        """
        Compute body-fixed accelerations from state and control inputs.

        Computes surge, sway, and yaw accelerations using a nonlinear,
        low-order maneuvering model with heuristic coupling terms.

        Rudder-induced forces generate both sway acceleration and yaw
        acceleration, while thrust acts purely in surge.

        Args:
            u: Body-fixed surge velocity [m/s].
            v: Body-fixed sway velocity [m/s].
            r: Body-fixed yaw rate [rad/s].
            rudder_angle: Rudder deflection angle [rad], positive to starboard.
            thrust: Normalized thrust command (dimensionless),
                positive in the forward direction.

        Returns:
            Tuple of accelerations `(du, dv, dr)`:
                - du: Surge acceleration [m/s²]
                - dv: Sway acceleration [m/s²]
                - dr: Yaw acceleration [rad/s²]

        Modeling notes:
            - Linear damping is applied independently in surge, sway, and yaw.
            - Rudder forces are modeled as instantaneous functions of rudder
              angle (no actuator dynamics).
            - A simple velocity cross-coupling term `(u * v) / L` is included
              in yaw to emulate centripetal effects.
            - Additional linear yaw-rate damping is applied to stabilize
              rotational dynamics.

        Limitations:
            - No explicit mass or added-mass matrix formulation
            - No nonlinear damping or saturation effects
            - No environmental forces or moments
        """

        # Simplified dynamics
        du = self.k_t * (thrust * 60.0) + self.xu * u
        dv = self.k_v * np.sin(rudder_angle) + self.yv * v
        dr = self.k_r * rudder_angle + self.nr * r + self.yv_r * v + (v * u) / self.l - YAW_RATE_DAMPING * r

        return du, dv, dr
