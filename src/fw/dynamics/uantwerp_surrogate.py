import numpy as np

from typing import Tuple, Optional
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.surrogate.inference_surrogate import SurrogatePredictor


class UAntwerpSurrogate(DynamicsBase):
    """
    Surrogate-model-backed vessel dynamics for 3-DOF maneuvering.

    Wraps a trained MLP surrogate (SurrogatePredictor) that predicts
    one-step body-frame velocity deltas given the current kinematic state
    and normalized control inputs.

    The surrogate predicts:
        (delta_surge, delta_sway, delta_yaw_rate)

    World-frame integration (dx, dy, dpsi) is handled externally by
    ShipSimulator.integrate(), which applies midpoint heading, velocity
    clipping, and heading wrapping.  This class must NOT attempt to
    replicate that logic.

    Output of accelerations() matches the DynamicsBase / ShipSimulator
    contract:
        (du, dv, dr)
    where each value is a one-step body-frame delta to be added to the
    current velocity state by the integrator.

    Coordinate frame and assumptions:
        - Body-fixed reference frame at the vessel center of gravity (CG)
        - Axes: x forward, y starboard, z downward (right-handed)
        - Motion restricted to the horizontal plane (3-DOF)
        - No environmental disturbances (wind, waves, current)
    """

    def __init__(self, ship_spec: ShipSpecifications):
        """
        Initialize the surrogate dynamics model.

        Args:
            ship_spec: Vessel physical specifications.  Used to inverse-map
                physical actuator values back to the normalized [-1, 1]
                action space that the surrogate was trained on.
        """
        self.ship_spec: ShipSpecifications = ship_spec
        self.surrogate = SurrogatePredictor("checkpoints/best_surrogate.pth")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_normalized_rudder(self, rudder_angle: float) -> float:
        """Inverse of ActionMapper: physical rudder [rad] -> normalized [-1, 1]."""
        return rudder_angle / self.ship_spec.max_rudder_angle

    def _to_normalized_thrust(self, thrust: float) -> float:
        """Inverse of ActionMapper: physical thrust -> normalized [-1, 1]."""
        span = self.ship_spec.max_thrust - self.ship_spec.min_thrust
        return 2.0 * (thrust - self.ship_spec.min_thrust) / span - 1.0

    # ------------------------------------------------------------------
    # DynamicsBase interface
    # ------------------------------------------------------------------

    def accelerations(
        self,
        u: float,
        v: float,
        r: float,
        rudder_angle: float,
        thrust: float,
        heading: Optional[float] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Predict one-step body-frame velocity deltas using the trained surrogate.

        The surrogate expects a 10-D input in the same feature order used
        during training (see sim_fh_dataset_generator.py):
            [x, y, elapsed_t, heading, surge, sway, yaw_rate,
             rudder_normalized, thrust_normalized, dt]

        It returns:
            (delta_surge, delta_sway, delta_yaw_rate)

        World-frame position and heading updates are computed by
        ShipSimulator.integrate() and must not be duplicated here.

        Args:
            u:            Body-fixed surge velocity [m/s].
            v:            Body-fixed sway velocity [m/s].
            r:            Body-fixed yaw rate [rad/s].
            rudder_angle: Physical rudder deflection [rad] (inverse-mapped
                          to normalized [-1, 1] internally).
            thrust:       Physical thrust command (inverse-mapped to
                          normalized [-1, 1] internally).
            heading:      Current vessel heading [rad].  Required.
            x:            World-frame x position [m].  Required.
            y:            World-frame y position [m].  Required.

        Returns:
            Tuple (du, dv, dr) — one-step body-frame velocity deltas to be
            added directly to the current state by ShipSimulator.integrate().

        Raises:
            ValueError: If any required positional/context argument is None.
        """
        missing = [
            name for name, val in (
                ("heading", heading), ("x", x), ("y", y),
            )
            if val is None
        ]
        if missing:
            raise ValueError(
                f"UAntwerpSurrogate.accelerations() requires: {missing}."
            )

        # rudder_norm = self._to_normalized_rudder(rudder_angle)
        # thrust_norm = self._to_normalized_thrust(thrust)

        # Feature order must match training data (10-D):
        # [x, y, elapsed_t, heading, surge, sway, yaw_rate,
        #  rudder_normalized, thrust_normalized, dt]
        x_raw = np.array(
            [x, y, heading, u, v, r, rudder_angle, thrust],
            dtype=np.float32,
        )

        # print(f"x_raw: {x_raw}")

        # surrogate predicts (delta_surge, delta_sway, delta_yaw_rate)
        du, dv, dr = self.surrogate.predict_single(x_raw)

        return float(du), float(dv), float(dr)
