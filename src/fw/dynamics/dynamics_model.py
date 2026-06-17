from typing import Tuple
from enum import auto, Enum
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
        return (u + du*dt, v + dv*dt, r + dr*dt). Damping and clipping
        are applied by ShipSimulator after the call.

    - Black-box / engine-backed models (FhTabularModel, UAntwerpSurrogate):
        Advance an internal engine or surrogate by dt seconds and return
        the resulting (u, v, r) directly. No re-integration is performed.
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
            state:        Current vessel state.
            rudder_angle: Rudder deflection angle [rad], positive to starboard.
            thrust:       Normalized thrust command (dimensionless), positive forward.

        Returns:
            Tuple (du, dv, dr) — surge, sway, and yaw accelerations [m/s², m/s², rad/s²].
        """

        raise NotImplementedError

    def step(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        """
        Advance body-fixed velocities by dt seconds using Euler integration.

        Damping and velocity clipping are applied by ShipSimulator after
        this call and must not be duplicated here.

        Args:
            state:        Current vessel state.
            rudder_angle: Rudder deflection angle [rad], positive to starboard.
            thrust:       Normalized thrust command (dimensionless), positive forward.
            dt:           Integration time step [s].

        Returns:
            Tuple (u, v, r) — un-damped, un-clipped updated body-fixed velocities.
        """

        du, dv, dr = self._compute_accelerations(state, rudder_angle, thrust)
        return state.u + du * dt, state.v + dv * dt, state.r + dr * dt
