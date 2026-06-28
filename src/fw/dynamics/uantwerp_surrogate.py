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
    Surrogate-model-based 3-DOF vessel dynamics.

    Uses a trained neural-network surrogate to predict body-frame surge,
    sway, and yaw accelerations from the current vessel state, actuator
    commands, and optionally environmental conditions (UKC, wind, and
    current).

    The surrogate operates in a body-fixed reference frame with motion
    restricted to the horizontal plane. It is trained for a fixed time step
    of `_SURROGATE_DT`, which is enforced by `step()`.
    """

    def __init__(self,
                 ship_spec: ShipSpecifications,
                 ua_surrogate_config: Optional[Dict]=None):
        """
        Initialize the surrogate dynamics model.

        Args:
            ship_spec:           Vessel specifications used to normalize
                                 actuator inputs for the surrogate model.
            ua_surrogate_config: Configuration dictionary for the surrogate
                                 model and optional environmental inputs.
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
        Return the current under-keel clearance.

        The position argument is accepted for interface compatibility but is
        currently ignored.

        Args:
            pos: Vessel position.

        Returns:
            Current under-keel clearance in metres.
        """

        return self.ukc

    def get_env(self, pos: Optional[np.ndarray]) -> EnvironmentData:
        """
        Return environmental conditions at the specified position.

        Uses nearest-neighbor lookup against the training dataset when
        available. Otherwise, returns default environmental conditions with
        constant UKC and zero wind and current.

        Args:
            pos: World-frame position `[x, y]` in metres.

        Returns:
            Environmental conditions at the requested position.
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
        Advance the vessel dynamics by one surrogate time step.

        The surrogate is trained for a fixed integration interval of
        `_SURROGATE_DT`; therefore, `dt` must match this value exactly.

        Args:
            state:                       Current vessel state.
            rudder_angle:                Rudder deflection angle [rad], positive to starboard.
            thrust:                      Normalized thrust command in `[-1, 1]`.
            dt:                          Integration time step [s].
            enable_rudder_angle_scaling: Passed to the base implementation for interface compatibility.
                                         Ignored by this class.

        Returns:
            Tuple `(u, v, r)` containing the updated body-frame velocities.

        Raises:
            AssertionError: If `dt` differs from `_SURROGATE_DT`.
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
        Assemble the surrogate input vector.

        The input consists of the vessel state, normalized control inputs, and,
        when WCU support is enabled, environmental conditions (UKC, wind, and
        current) obtained from `get_env()`.

        Args:
            state:        Current vessel state.
            rudder_angle: Rudder deflection angle [rad].
            thrust:       Normalized thrust command in `[-1, 1]`.

        Returns:
            Input vector expected by the trained surrogate model.
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
        Compute body-frame accelerations using the trained surrogate.

        The surrogate input vector is assembled by `_build_input_vector()`. The
        surrogate predicts body-frame velocity changes over one training time
        step (`_SURROGATE_DT`), which are converted to accelerations by dividing
        by that interval.

        Args:
            state:        Current vessel state.
            rudder_angle: Rudder deflection angle [rad], positive to starboard.
            thrust:       Normalized thrust command in `[-1, 1]`.

        Returns:
            Tuple `(du, dv, dr)` containing the predicted body-frame
            accelerations.

        Raises:
            ValueError: If `state.heading`, `state.x`, or `state.y` is `None`.
        """

        missing = [
            name for name, val in (
                ("heading", state.heading), ("x", state.x), ("y", state.y),
            )
            if val is None
        ]
        if missing:
            raise ValueError(
                f"UAntwerpSurrogate requires state fields {missing} to evaluate the surrogate."
            )

        x_raw = self._build_input_vector(state, rudder_angle, thrust)
        delta_u, delta_v, delta_r = self.surrogate.predict_single(x_raw)
        return float(delta_u / _SURROGATE_DT), float(delta_v / _SURROGATE_DT), float(delta_r / _SURROGATE_DT)
