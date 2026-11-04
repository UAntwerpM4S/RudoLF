import numpy as np

from enum import auto, Enum
from typing import Tuple, Optional
from fw.ships.vessel_state import VesselState


class Model(Enum):
    """
    Enumeration of supported vessel-dynamics model families.
    """

    NOMOTO = auto()
    FOSSEN = auto()
    UGENT_SURROGATE = auto()
    UANTWERP = auto()
    FH_TABULAR = auto()
    UGENT_B01_SURROGATE = auto()


class DynamicsBase:
    """
    Abstract base class for 3-DOF (surge, sway, yaw) vessel dynamics in the body frame.

    `step()` advances the body-fixed velocities (u, v, r) by one time step;
    world-frame position and heading are integrated by ShipSimulator.integrate(),
    which also applies velocity clipping.

    Implementation families:
    - Physics-based (Fossen, Nomoto, UGentSurrogate, UGentB01Surrogate): implement
      `_compute_accelerations` and use the default Euler `step()`.
    - Engine-backed (FhTabularModel): override `step()` and return the engine's
      (u, v, r) after dt directly.
    - Surrogate (UAntwerpSurrogate): converts a predicted velocity delta to an
      acceleration (delta / dt) and uses the default `step()`; only valid when dt
      equals the training step, which the subclass enforces.

    Wind and current exist only in models that simulate them themselves
    (`models_own_wind_current`: FhTabularModel, UAntwerpSurrogate full_env).
    `observed_wind_current()` gives them to the RL observation (zeros for all
    other models); `reset_episode()` clears per-episode model state.
    """

    models_own_wind_current: bool = False
    """
    True if this model simulates wind and current itself (default False: no
    wind or current at all). Subclasses override it as a class attribute or, if
    it depends on the instance, as a read-only property.
    """

    def observed_wind_current(self, state: VesselState) -> np.ndarray:
        """
        Wind and current at `state` as the ship observes them, in the
        "body_components" convention of the surrogate dataset
        (fw.surrogate.schema.env_body_components).

        Returns:
            float64 array [wind_from_ahead, wind_from_stbd, current_to_ahead,
            current_to_stbd] in m/s; zeros for models without wind and current.
        """

        return np.zeros(4, dtype=np.float64)

    def reset_episode(self) -> None:
        """
        Clear per-episode state (called by ShipSimulator.reset); no-op by default.
        """

        return None

    def _compute_accelerations(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
    ) -> Tuple[float, float, float]:
        """
        Compute body-fixed accelerations from state and control inputs.

        Args:
            state:        Current vessel state (v port positive, r counter-clockwise
                          positive, see fw.ships.vessel_state).
            rudder_angle: Rudder command [rad]; positive = the ship turns to PORT
                          (framework convention, see fw.ships.vessel_state).
            thrust:       Normalized thrust command (dimensionless), positive forward.

        Returns:
            Tuple (du, dv, dr): surge, sway, and yaw accelerations [m/s², m/s², rad/s²].
        """

        raise NotImplementedError

    def get_ukc(self, pos: Optional[np.ndarray]):
        """
        Query the current under-keel clearance.

        Args:
            pos (np.array, optional): Position to query.

        Returns:
            float: Keel clearance in metres.
        """

        pass

    def step(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
        dt: float,
        enable_rudder_angle_scaling: Optional[bool] = False,
    ) -> Tuple[float, float, float]:
        """
        Advance body-fixed velocities by dt seconds using Euler integration.

        Velocity clipping is done by ShipSimulator; do not duplicate it here.

        Args:
            state:                       Current vessel state.
            rudder_angle:                Rudder command [rad]; positive = the ship turns
                                         to PORT (see fw.ships.vessel_state).
            thrust:                      Normalized thrust command (dimensionless),
                                         positive forward.
            dt:                          Integration time step [s].
            enable_rudder_angle_scaling: Rescale the rudder command to the target
                                         engine's maximum rudder angle (sim-to-sim
                                         transfer). Only used by FhTabularModel.

        Returns:
            Tuple (u, v, r): un-clipped updated body-fixed velocities.
        """

        du, dv, dr = self._compute_accelerations(state, rudder_angle, thrust)
        return state.u + du * dt, state.v + dv * dt, state.r + dr * dt
