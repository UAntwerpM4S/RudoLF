import numpy as np

from typing import Tuple
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.surrogate.inference_surrogate import SurrogatePredictor

# Time step the surrogate was trained on [s]
_SURROGATE_DT = 0.1


class UAntwerpSurrogate(DynamicsBase):
    """
    Surrogate-model-backed vessel dynamics for 3-DOF maneuvering.

    Wraps a trained MLP surrogate (SurrogatePredictor) that predicts
    one-step body-frame velocity deltas given the current kinematic state
    and normalized control inputs.

    The surrogate predicts absolute velocity deltas:
        (delta_u, delta_v, delta_r)

    These deltas are divided by _SURROGATE_DT (0.1 s) to yield accelerations,
    which are then integrated over dt by DynamicsBase.step(). This is
    mathematically correct only when dt == _SURROGATE_DT; step() enforces
    this with an assertion.

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
            ship_spec: Vessel physical specifications. Used to inverse-map
                physical actuator values back to the normalized [-1, 1]
                action space that the surrogate was trained on.
        """

        self.ship_spec: ShipSpecifications = ship_spec
        self.surrogate = SurrogatePredictor("checkpoints/best_surrogate.pth")

    def step(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        assert dt == _SURROGATE_DT, (
            f"UAntwerpSurrogate requires dt == {_SURROGATE_DT} s (surrogate training step), got {dt}."
        )
        return super().step(state, rudder_angle, thrust, dt)

    def _compute_accelerations(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
    ) -> Tuple[float, float, float]:
        """
        Predict updated body-fixed velocities using the trained surrogate.

        The surrogate expects an 8-D input in the feature order used during
        training (see sim_fh_dataset_generator.py):
            [x, y, heading, surge, sway, yaw_rate, rudder_normalized, thrust_normalized]

        It returns absolute velocity deltas (delta_u, delta_v, delta_r) for
        a 0.1 s step, which are divided by _SURROGATE_DT to yield accelerations
        and integrated over dt by DynamicsBase.step().

        Damping and velocity clipping are applied by ShipSimulator after
        this call and must not be duplicated here.

        Args:
            state:        Current vessel state.
            rudder_angle: Normalized rudder command (dimensionless) in [-1, 1],
                          passed directly as a surrogate input feature.
            thrust:       Normalized thrust command (dimensionless) in [-1, 1],
                          passed directly as a surrogate input feature.

        Returns:
            Tuple (du, dv, dr) — body-fixed accelerations [m/s², m/s², rad/s²],
            valid only for dt == _SURROGATE_DT (0.1 s).

        Raises:
            ValueError: If required state fields (x, y, heading) are None.
        """

        missing = [
            name for name, val in (
                ("heading", state.heading), ("x", state.x), ("y", state.y),
            )
            if val is None
        ]
        if missing:
            raise ValueError(
                f"UAntwerpSurrogate.step() requires state fields: {missing}."
            )

        x_raw = np.array(
            [state.x, state.y, state.heading, state.u, state.v, state.r, rudder_angle, thrust],
            dtype=np.float32,
        )

        delta_u, delta_v, delta_r = self.surrogate.predict_single(x_raw)
        return float(delta_u/_SURROGATE_DT), float(delta_v/_SURROGATE_DT), float(delta_r/_SURROGATE_DT)
