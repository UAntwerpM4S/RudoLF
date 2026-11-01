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
    Abstract base class for vessel dynamics models.

    This interface defines the contract for vessel dynamics formulated in
    the body-fixed reference frame. Concrete implementations model the
    surge–sway–yaw (3-DOF) dynamics of a marine vessel.

    The model exposes a single `step()` method that advances the body-fixed
    velocities (u, v, r) by one time step. World-frame position and heading
    integration is handled externally by ShipSimulator.integrate().

    Two families of implementations exist:

    - Physics-based models (Fossen, Nomoto, UGentSurrogate, UGentB01Surrogate):
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

    Wind and current (`models_own_wind_current`):
        Some models already include wind and current in their own dynamics
        (FhTabularModel; UAntwerpSurrogate with a full_env checkpoint). For
        them ShipSimulator must NOT add EnvironmentModel's accelerations as
        well, or the effect is applied twice. All other models (Fossen,
        Nomoto, UGentSurrogate, UGentB01Surrogate, UAntwerpSurrogate
        open_water/confined_ukc) get their wind/current from EnvironmentModel
        via ShipSimulator, except UGentB01Surrogate (see below).

    EnvironmentModel support (`supports_environment_model`):
        EnvironmentModel adds fixed full-scale accelerations. A model for which
        they make no sense (UGentB01Surrogate, model scale: they are about 9x
        its maximum thrust acceleration) sets this to False, and ShipSimulator
        then refuses wind=True / current=True instead of adding them.
    """

    models_own_wind_current: bool = False
    """
    True if this model includes wind and current in its own dynamics, so
    ShipSimulator must skip the EnvironmentModel accelerations for it.
    Default False: ShipSimulator adds them. Subclasses override it as a class
    attribute or, if it depends on the instance, as a read-only property.
    """

    supports_environment_model: bool = True
    """
    False if EnvironmentModel's (full-scale) wind/current accelerations must
    never be added to this model; ShipSimulator then raises ValueError when
    wind or current is enabled. Default True.
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
            rudder_angle:                Rudder command [rad]; positive = the ship turns
                                         to PORT (see fw.ships.vessel_state).
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
            Tuple (u, v, r): un-clipped updated body-fixed velocities.
        """

        du, dv, dr = self._compute_accelerations(state, rudder_angle, thrust)
        return state.u + du * dt, state.v + dv * dt, state.r + dr * dt
