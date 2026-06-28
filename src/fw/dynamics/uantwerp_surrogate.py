import numpy as np

from pathlib import Path
from typing import Tuple, Optional, Dict
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.dynamics.wcu_lookup import WcuLookup, EnvironmentData
from fw.surrogate.inference_surrogate import SurrogatePredictor

# Time step the surrogate was trained on [s]
_SURROGATE_DT = 0.1
WATER_DEPTH_M = 18.5
FH_MYZAKO_MAX_RUDDER_ANGLE = 75.0
SHIP_DATA_FILE = Path(__file__).with_name("ship_dynamics_dataset.csv")

# Fallback environmental conditions used when no lookup CSV is available.
# Values are dataset-representative defaults; replace if the deployment
# environment differs significantly.
_FALLBACK_ENV = EnvironmentData(
    ukc=WATER_DEPTH_M,
    wind_speed=0.0,
    wind_direction=0.0,
    current_speed=0.0,
    current_direction=0.0,
)


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
        - Wind and current are included as surrogate inputs (looked up from
          the training dataset via nearest-neighbor UKCLookup); when no
          CSV is available, zero wind / zero current fallbacks are used.
    """

    def __init__(self,
                 ship_spec: ShipSpecifications,
                 ua_surrogate_config: Optional[Dict]=None):
        """
        Initialize the surrogate dynamics model.

        Args:
            ship_spec: Vessel physical specifications. Used to inverse-map
                physical actuator values back to the normalized [-1, 1]
                action space that the surrogate was trained on.
            ua_surrogate_config: Configuration for the UA surrogate model.
        """

        self.ship_spec: ShipSpecifications = ship_spec
        self.wcu_enabled = ua_surrogate_config.get("wcu_enabled", False)
        self.surrogate = SurrogatePredictor(ua_surrogate_config.get("file_name", "checkpoints/best_surrogate_clean.pth"))

        expects_wcu = self.surrogate.input_dim >= 9
        if expects_wcu != self.wcu_enabled:
            raise RuntimeError(
                f"Incorrect surrogate model file is used. "
                f"Wind, current and UKC {'are' if self.wcu_enabled else 'are not'} enabled, "
                f"but the surrogate model expects {self.surrogate.input_dim} inputs."
            )

        self.wcu_lookup = WcuLookup(str(SHIP_DATA_FILE)) if self.wcu_enabled and SHIP_DATA_FILE.exists() else None
        self.default_ukc = WATER_DEPTH_M if self.wcu_enabled else 0.0
        self.ukc = self.default_ukc

        if self.wcu_enabled and self.wcu_lookup is None:
            print(
                f"Ship data file {SHIP_DATA_FILE} not found, "
                f"falling back to default UKC value of {WATER_DEPTH_M}"
            )

    def get_ukc(self, pos: Optional[np.ndarray]):
        """
        Query the current under-keel clearance.

        Args:
            pos (np.array, optional): The position for which
                to query the keel clearance.

        Returns:
            float: Keel clearance in metres.
        """

        return self.ukc

    def get_env(self, pos: Optional[np.ndarray]) -> EnvironmentData:
        """
        Query environmental conditions (UKC, wind, current) at *pos*.

        Uses nearest-neighbor lookup against the training dataset when
        available; falls back to `_FALLBACK_ENV` (zero wind/current,
        `WATER_DEPTH_M` UKC) otherwise.

        Args:
            pos: [x, y] world-frame position (m).

        Returns:
            EnvironmentData with ukc, wind_speed, wind_direction,
            current_speed, current_direction fields.
        """
        if self.wcu_lookup is None:
            # Return a fresh instance instead of mutating the shared
            # module-level _FALLBACK_ENV singleton.
            return EnvironmentData(
                ukc=self.default_ukc,
                wind_speed=_FALLBACK_ENV.wind_speed,
                wind_direction=_FALLBACK_ENV.wind_direction,
                current_speed=_FALLBACK_ENV.current_speed,
                current_direction=_FALLBACK_ENV.current_direction,
            )

        env = self.wcu_lookup.get_env(pos[0], pos[1])
        self.ukc = env.ukc if env.ukc is not None else self.default_ukc

        return env

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

    def _build_input_vector(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
    ) -> np.ndarray:
        """
        Assemble the 13-D surrogate input vector matching the training
        dataset column layout (see sim_fh_dataset_generator.py):

            [x, y, heading, surge, sway, yaw_rate,
             ukc, wind_speed, wind_direction, current_speed, current_direction,
             rudder_action, thrust_action]

        Environmental fields come from get_env(); rudder_angle [rad] is
        converted to the normalized [-1, 1] rudder_action feature by
        dividing degrees by FH_MYZAKO_MAX_RUDDER_ANGLE (75°).
        """
        env = self.get_env(np.array([state.x, state.y]))
        rudder = np.degrees(rudder_angle) / FH_MYZAKO_MAX_RUDDER_ANGLE

        if self.wcu_enabled:
            return np.array(
                [
                    state.x, state.y, state.heading,
                    state.u, state.v, state.r,
                    env.ukc, env.wind_speed, env.wind_direction,
                    env.current_speed, env.current_direction,
                    rudder, thrust,
                ],
                dtype=np.float32,
            )
        else:
            return np.array(
                [
                    state.x, state.y, state.heading,
                    state.u, state.v, state.r,
                    rudder, thrust,
                ],
                dtype=np.float32,
            )

    def _compute_accelerations(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
    ) -> Tuple[float, float, float]:
        """
        Predict body-fixed velocity deltas using the trained MLP surrogate.

        The surrogate expects a 13-D input vector matching the column layout
        of the training dataset (see sim_fh_dataset_generator.py):

            [x, y, heading, surge, sway, yaw_rate,
             ukc, wind_speed, wind_direction, current_speed, current_direction,
             rudder_action, thrust_action]

        Environmental fields (UKC, wind, current) are obtained via
        `get_env()`, which performs a nearest-neighbor lookup against the
        training CSV; `_FALLBACK_ENV` is used when no CSV is present.

        `rudder_action` is obtained by converting `rudder_angle` from
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

        x_raw = self._build_input_vector(state, rudder_angle, thrust)
        delta_u, delta_v, delta_r = self.surrogate.predict_single(x_raw)
        return float(delta_u / _SURROGATE_DT), float(delta_v / _SURROGATE_DT), float(delta_r / _SURROGATE_DT)
