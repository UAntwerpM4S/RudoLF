import math
import numpy as np

from enum import auto, Enum
from typing import Dict, Tuple, Optional
from fw.ships.vessel_state import VesselState


_STEADY_THRUST_ITERATIONS = 60


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

    def episode_conditions(self) -> Dict[str, object]:
        """
        Environmental conditions of the current episode for the episode log
        (e.g. drawn wind and current); empty by default.
        """

        return {}

    def reset_episode(self, rng: Optional[np.random.Generator] = None) -> None:
        """
        Clear per-episode state (called by ShipSimulator.reset); `rng` serves
        models that draw per-episode values. No-op by default.
        """

        return None

    supports_steady_thrust: bool = True
    """
    False if steady_thrust() must not be derived from _compute_accelerations
    (UAntwerpSurrogate); steady_thrust() then returns None. Models that do not
    implement _compute_accelerations (FhTabularModel) also return None.
    """

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

    def _surge_acceleration(self, state: VesselState, thrust: float) -> float:
        """
        Surge acceleration [m/s²] at *state* with the rudder at zero; used by
        steady_thrust(). Subclasses may override it, e.g. to leave out a warning
        that _compute_accelerations logs.
        """

        return float(self._compute_accelerations(state, 0.0, thrust)[0])

    @property
    def derives_steady_thrust(self) -> bool:
        """
        True if steady_thrust() can solve for a thrust: supports_steady_thrust
        and the model implements _compute_accelerations.
        """

        return (self.supports_steady_thrust
                and type(self)._compute_accelerations is not DynamicsBase._compute_accelerations)

    def steady_thrust(self, state: VesselState, min_thrust: float, max_thrust: float) -> Optional[float]:
        """
        Thrust in [min_thrust, max_thrust] that holds the surge speed state.u:
        zero surge acceleration with the rudder at zero, no sway and no yaw rate,
        at the position and heading of *state*. Calm water: EnvironmentModel wind
        and current are not included. Solved by bisection between min_thrust and
        max_thrust: unique if the surge acceleration increases with thrust.

        Args:
            state: Start state; only x, y, heading and u are used.
            min_thrust: Lower thrust limit.
            max_thrust: Upper thrust limit.

        Returns:
            The thrust, or None if not derives_steady_thrust or no thrust in
            [min_thrust, max_thrust] holds state.u.
        """

        if not self.derives_steady_thrust:
            return None

        straight = VesselState(terminated=False, truncated=False, x=state.x, y=state.y,
                               heading=state.heading, u=state.u, v=0.0, r=0.0)
        low, high = float(min_thrust), float(max_thrust)
        du_low = self._surge_acceleration(straight, low)
        du_high = self._surge_acceleration(straight, high)
        if not (math.isfinite(du_low) and math.isfinite(du_high)) or du_low > 0.0 or du_high < 0.0:
            return None

        for _ in range(_STEADY_THRUST_ITERATIONS):
            mid = 0.5 * (low + high)
            if self._surge_acceleration(straight, mid) < 0.0:
                low = mid
            else:
                high = mid
        return min(max(0.5 * (low + high), float(min_thrust)), float(max_thrust))

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
