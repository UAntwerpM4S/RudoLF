import numpy as np

from pathlib import Path
from typing import Tuple, Optional
from fw.dynamics.ukc_lookup import UKCLookup
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.surrogate.inference_surrogate import SurrogatePredictor

# Time step the surrogate was trained on [s]
_SURROGATE_DT = 0.1
WATER_DEPTH_M = 18.5
FH_MYZAKO_MAX_RUDDER_ANGLE = 75.0
SHIP_DATA_FILE = Path(__file__).with_name("ship_dynamics_dataset.csv")

# Ray-cast proximity sensor constants
RAY_MAX_DISTANCE = 200.0                                    # metres; beyond this the feature is 0
RAY_ANGLES_DEG = np.arange(0, 360, 15)                     # 24 rays: 0°, 15°, 30°, …, 345°
NBR_RAY_FEATURES = len(RAY_ANGLES_DEG)                      # 24
_RAY_ANGLES_RAD = np.radians(RAY_ANGLES_DEG)               # precomputed; shape (24,)
_RAY_SIN = np.sin(_RAY_ANGLES_RAD)                         # x-components of unit direction vectors
_RAY_COS = np.cos(_RAY_ANGLES_RAD)                         # y-components of unit direction vectors
# Local-frame end-point offsets (rotated by heading at call time); shape (24, 2)
_RAY_OFFSETS = np.stack([_RAY_SIN, _RAY_COS], axis=1) * RAY_MAX_DISTANCE


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
        self.ukc_lookup = UKCLookup(str(SHIP_DATA_FILE)) if SHIP_DATA_FILE.exists() else None

        if self.ukc_lookup is None:
            print(
                f"Ship data file {SHIP_DATA_FILE} not found, "
                f"falling back to default UKC value of {WATER_DEPTH_M}"
            )

        # Imported locally to avoid a circular import (py_sim_env imports this module).
        # A single instance is created here and reused for ray-cast geometry lookups.
        from fw.environments.py_sim_env import PySimEnv
        self._env = PySimEnv(render_mode=None, time_step=_SURROGATE_DT, wind=False, current=False)

    @staticmethod
    def compute_ray_features(env: "PySimEnv", ship_pos: np.ndarray, heading_rad: float) -> np.ndarray:
        """Cast 24 rays around the ship and return normalized proximity features.

        Each ray is fired from *ship_pos* in the direction
        (ship heading + ray_angle_deg) expressed in the world frame.
        The ray is clipped against the river-polygon boundary (or at
        RAY_MAX_DISTANCE, whichever comes first).

        Mapping:  distance d  →  feature = (RAY_MAX_DISTANCE - d) / RAY_MAX_DISTANCE
            d = 0 m   -> feature = 1.0   (obstacle right here)
            d = 200 m -> feature = 0.0   (no obstacle within range)

        The function tries a vectorized batch call first
        (`env.ray_cast_distances_batch`), which avoids Python-loop overhead when
        the environment exposes it.  If that method is absent it falls back to 24
        sequential `env.ray_cast_distance` calls.

        End-point directions are computed from module-level precomputed sin/cos
        tables (_RAY_SIN, _RAY_COS) rotated by *heading_rad*, so no trig is
        evaluated at call time.

        Args:
            env:         PySimEnv instance (provides polygon_shape and ray-cast method).
            ship_pos:    [x, y] world-frame ship position (m).
            heading_rad: Current ship heading in radians (world frame, 0 = North/+Y).

        Returns:
            features: float32 array of shape (NBR_RAY_FEATURES,) = (24,).
        """
        # Rotate the precomputed local-frame offsets into world frame using heading.
        # Convention: dx = sin(angle), dy = cos(angle) (North-referenced bearing).
        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        rotated = np.empty_like(_RAY_OFFSETS)
        rotated[:, 0] = _RAY_OFFSETS[:, 0] * cos_h + _RAY_OFFSETS[:, 1] * sin_h
        rotated[:, 1] = -_RAY_OFFSETS[:, 0] * sin_h + _RAY_OFFSETS[:, 1] * cos_h
        end_points = ship_pos + rotated  # (24, 2)

        origin = (float(ship_pos[0]), float(ship_pos[1]))

        # -- Batch path (preferred) -----------------------------------------------
        if hasattr(env, "ray_cast_distances_batch"):
            # Expected signature:
            #   env.ray_cast_distances_batch(origin, end_points, max_distance)
            #   -> array-like of shape (24,) with distances in [0, RAY_MAX_DISTANCE]
            distances = np.asarray(
                env.ray_cast_distances_batch(origin, end_points, RAY_MAX_DISTANCE),
                dtype=np.float32,
            )
            np.clip(distances, 0.0, RAY_MAX_DISTANCE, out=distances)

        # -- Sequential fallback --------------------------------------------------
        else:
            distances = np.empty(NBR_RAY_FEATURES, dtype=np.float32)
            for i in range(NBR_RAY_FEATURES):
                d = env.ray_cast_distance(
                    origin,
                    (float(end_points[i, 0]), float(end_points[i, 1])),
                    max_distance=RAY_MAX_DISTANCE,
                )
                distances[i] = min(max(float(d), 0.0), RAY_MAX_DISTANCE)

        return (RAY_MAX_DISTANCE - distances) / RAY_MAX_DISTANCE

    def get_ukc(self, pos: Optional[np.ndarray]):
        """
        Query the current under-keel clearance.

        Args:
            pos (np.array, optional): The position for which
                to query the keel clearance.

        Returns:
            float: Keel clearance in metres.
        """

        return self.ukc_lookup.get_ukc(pos[0], pos[1])

    def step(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
        dt: float,
        enable_rudder_angle_scaling: Optional[bool] = False,
    ) -> Tuple[float, float, float]:
        """
        Advance body-fixed velocities by dt seconds via the surrogate.

        Delegates to DynamicsBase.step() after asserting that dt matches the
        surrogate's training step length.  ShipSimulator must therefore use a
        dt that is an exact integer multiple of _SURROGATE_DT (0.1 s); any
        fractional remainder sub-step produced by ShipSimulator.step() will
        trip this assertion.

        Args:
            state:                       Current vessel state.
            rudder_angle:                Rudder deflection angle [rad], positive
                                         to starboard.
            thrust:                      Normalized thrust command in [-1, 1].
            dt:                          Integration step size [s].  Must equal
                                         _SURROGATE_DT exactly.
            enable_rudder_angle_scaling: Passed through to super().step() for
                                         interface consistency; unused here.

        Returns:
            Tuple (u, v, r) — updated body-fixed velocities.

        Raises:
            AssertionError: If dt != _SURROGATE_DT.
        """

        assert np.isclose(dt, _SURROGATE_DT), (
            f"UAntwerpSurrogate requires dt == {_SURROGATE_DT} s (surrogate training step), got {dt}."
            f"Ensure ShipSimulator.dt is an exact integer multiple of {_SURROGATE_DT} s "
            f"so no fractional remainder sub-step is produced."
        )
        return super().step(state, rudder_angle, thrust, dt, enable_rudder_angle_scaling)

    def _compute_accelerations(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
    ) -> Tuple[float, float, float]:
        """
        Predict body-fixed velocity deltas using the trained MLP surrogate.

        The surrogate expects a 33-D input vector in the feature order used
        during training (see sim_fh_dataset_generator.py):

            [x, y, heading, surge, sway, yaw_rate,
             rudder_normalized, thrust_normalized,
             ray_0, ray_1, …, ray_23]          # 24 proximity features

        `rudder_normalized` is obtained by converting `rudder_angle` from
        radians to degrees and dividing by FH_MYZAKO_MAX_RUDDER_ANGLE (75°),
        producing a value in [-1, 1] that matches the training data scale.

        The surrogate returns absolute velocity deltas (delta_u, delta_v,
        delta_r) for a 0.1 s step. These are divided by _SURROGATE_DT to
        yield accelerations, which DynamicsBase.step() then integrates over dt.

        Damping and velocity clipping are applied by ShipSimulator after
        this call; they must not be duplicated here.

        Args:
            state:        Current vessel state.
            rudder_angle: Rudder deflection angle [rad], positive to starboard
                          (body-frame convention, identical to every other
                          DynamicsBase implementation).  Converted internally to
                          a normalized [-1, 1] surrogate feature.
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

        ray_features = self.compute_ray_features(
            self._env,
            np.array([state.x, state.y]),
            float(state.heading),
        )

        rudder = np.degrees(rudder_angle) / FH_MYZAKO_MAX_RUDDER_ANGLE
        ukc = self.ukc_lookup.get_ukc(state.x, state.y) if self.ukc_lookup else WATER_DEPTH_M

        x_raw = np.array(
            [state.x, state.y, state.heading, state.u, state.v, state.r, ukc, rudder, thrust, *ray_features],
            dtype=np.float32,
        )

        delta_u, delta_v, delta_r = self.surrogate.predict_single(x_raw)
        return float(delta_u/_SURROGATE_DT), float(delta_v/_SURROGATE_DT), float(delta_r/_SURROGATE_DT)
