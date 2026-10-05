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


class DynamicsBase:
    """
    Abstract base class for vessel dynamics models.

    This interface defines the contract for vessel dynamics formulated in
    the body-fixed reference frame. Concrete implementations model the
    surge–sway–yaw (3-DOF) dynamics of a marine vessel.

    The model exposes a single `step()` method that advances the body-fixed
    velocities (u, v, r) by one time step. World-frame position and heading
    integration is handled externally by ShipSimulator.integrate().

    Two families of implementations exist:

    - Physics-based models (Fossen, Nomoto, UGentSurrogate):
        Compute accelerations (du, dv, dr) from equations of motion and
        return (u + du*dt, v + dv*dt, r + dr*dt) via this class's default
        `step()` implementation. Velocity clipping is applied by
        ShipSimulator after the call.

    - Engine-backed models (FhTabularModel):
        Advance an internal engine by dt seconds and return the resulting
        (u, v, r) directly, bypassing this class's `step()` entirely (the
        subclass overrides `step()` and never calls `_compute_accelerations`).

    - Surrogate models (UAntwerpSurrogate):
        Predict an absolute velocity delta over a fixed training time step
        and convert it to an equivalent acceleration (delta / dt), then
        still go through this class's default Euler-integration `step()`.
        This only reproduces the surrogate's intended behavior when dt
        equals the surrogate's training step size, which the subclass
        enforces explicitly.

    `handles_environment`: True for models whose own dynamics already include
    wind/current effects (engine-backed and surrogate models). ShipSimulator
    then does NOT add EnvironmentModel accelerations on top. Physics-based
    models keep the default False.
    """

    handles_environment: bool = False

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

        raise NotImplementedError

    def get_ukc(self, pos: Optional[np.ndarray]):
        """
        Query the current under-keel clearance.

        Args:
            pos (np.array, optional): The position for which
                to query the keel clearance.

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

        Velocity clipping is applied by ShipSimulator after this call and
        must not be duplicated here.

        Args:
            state:                       Current vessel state.
            rudder_angle:                Rudder deflection angle [rad], positive to starboard.
            thrust:                      Normalized thrust command (dimensionless),
                                         positive forward.
            dt:                          Integration time step [s].
            enable_rudder_angle_scaling: When True, rescales the rudder command to
                                         compensate for a difference between this
                                         framework's maximum rudder angle and a
                                         target engine's maximum rudder angle (used
                                         for sim-to-sim transfer). Currently only
                                         consumed by FhTabularModel; other models
                                         accept but ignore this flag.

        Returns:
            Tuple (u, v, r) — un-clipped updated body-fixed velocities.
        """

        du, dv, dr = self._compute_accelerations(state, rudder_angle, thrust)
        return state.u + du * dt, state.v + dv * dt, state.r + dr * dt
