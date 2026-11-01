import numpy as np

from typing import Optional
from fw.ships.vessel_state import VesselState
from fw.ships.action_mapper import ActionMapper
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.simulators.base_simulator import BaseSimulator
from fw.dynamics.fh_tabular_model import FhTabularModel
from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate
from fw.dynamics.environmental_model import EnvironmentModel
from fw.surrogate.schema import SURROGATE_DT


class ShipSimulator(BaseSimulator):
    """
    Ship simulator implementing the unified interface.

    This simulator can integrate:

        - Several dynamics models: Fossen 3-DOF, Nomoto, UGentSurrogate, UGentB01Surrogate,
          UAntwerpSurrogate, FhTabularModel.
        - Actuator dynamics and rate limiting.
        - Environmental forces (wind and current) for models that do not
          model them themselves.
        - A unified interface compatible with RL or control loops.

    The simulator computes the vessel state in earth-fixed coordinates
    using body-fixed velocities and yaw rate. Integration uses a
    midpoint approximation for improved accuracy.

    Dynamics model contract
    -----------------------
    All DynamicsBase implementations expose a single step() method:

        u, v, r = dynamics.step(state, rudder_angle, thrust, dt)

    Physics-based models (Fossen, Nomoto, UGentSurrogate, UGentB01Surrogate)
    and the UAntwerpSurrogate return un-clipped (u, v, r) from Euler integration of
    their accelerations; FhTabularModel returns (u, v, r) read from the FH
    engine. ShipSimulator applies velocity clipping to all of them.

    Wind and current
    ----------------
    EnvironmentModel's accelerations are added after dynamics.step() only
    when dynamics.models_own_wind_current is False:

        models_own_wind_current = True  -> skipped (the model already has them):
            FhTabularModel, UAntwerpSurrogate with a full_env checkpoint.
        models_own_wind_current = False -> added:
            Fossen, Nomoto, UGentSurrogate, UGentB01Surrogate, UAntwerpSurrogate open_water /
            confined_ukc.

    The `wind` / `current` switches only control EnvironmentModel, so they
    have no effect on models with models_own_wind_current = True: FH takes
    wind and current from its exercise, a full_env surrogate from its
    lookup CSV (by design).

    Sub-stepping and dt constraints
    --------------------------------
    step() always uses a 0.1 s physics sub-step.  If the environment dt is
    not an exact multiple of 0.1 s a fractional remainder sub-step is issued.
    UAntwerpSurrogate only accepts dt == 0.1 s, so for that model a dt that is
    not a multiple is refused in __init__ (ValueError).  Use an environment dt
    that is an exact integer multiple of 0.1 s when using UAntwerpSurrogate
    (e.g. 0.1, 0.5, 1.0, 2.0).
    """

    # Physics integration sub-step [s] (= surrogate training step).
    PHYSICS_DT: float = SURROGATE_DT


    def __init__(self, specs: ShipSpecifications, dynamics: DynamicsBase, dt: float, wind: bool = False,
                 current: bool = False):
        """
        Initialize the ship simulator.

        Args:
            specs:             ShipSpecifications object defining vessel dimensions,
                               actuator limits, and kinematic constraints.
            dynamics:          DynamicsBase instance implementing the step() interface.
            dt:                Simulation time step [s]. Must be positive.
            wind:              Enable EnvironmentModel wind if True. No effect when
                               dynamics.models_own_wind_current is True.
            current:           Enable EnvironmentModel current if True. No effect
                               when dynamics.models_own_wind_current is True.

        Raises:
            ValueError: wind or current enabled for a model with
                supports_environment_model = False (UGentB01Surrogate).
        """

        super().__init__(specs, dt)

        # Physics sub-stepping grid, computed and validated once here (not on
        # the first step(), deep inside a training run). The sub-step equals
        # the surrogate training step (single source of truth: schema).
        n_full = int(round(dt / self.PHYSICS_DT))
        if n_full >= 1 and np.isclose(n_full * self.PHYSICS_DT, dt, rtol=0.0, atol=1e-9):
            dt_rem = 0.0
        else:
            n_full = int(np.floor(dt / self.PHYSICS_DT + 1e-9))
            dt_rem = dt - n_full * self.PHYSICS_DT
            if isinstance(dynamics, UAntwerpSurrogate):
                raise ValueError(
                    f"time_step={dt} s must be a positive integer multiple of {self.PHYSICS_DT} s "
                    f"for {type(dynamics).__name__} (the surrogate's training step)."
                )
        self._n_sub_steps = n_full
        self._dt_remainder = dt_rem

        # Decided once here (fail fast): a non-bool flag (e.g. a forgotten
        # property returning None) must not silently pick a branch per step.
        own_wc = getattr(dynamics, "models_own_wind_current", None)
        if not isinstance(own_wc, bool):
            raise TypeError(
                f"{type(dynamics).__name__}.models_own_wind_current must be a bool, got {own_wc!r}."
            )
        self._add_environment = not own_wc

        supports_env = getattr(dynamics, "supports_environment_model", None)
        if not isinstance(supports_env, bool):
            raise TypeError(
                f"{type(dynamics).__name__}.supports_environment_model must be a bool, got {supports_env!r}."
            )
        if (wind or current) and self._add_environment and not supports_env:
            raise ValueError(
                f"wind={wind}/current={current}: {type(dynamics).__name__} does not support "
                f"EnvironmentModel's wind/current (fixed full-scale accelerations). Disable wind and current."
            )

        self._dynamics = dynamics
        self._mapper = ActionMapper()
        self._env = EnvironmentModel(wind, current)

    @property
    def environment(self) -> EnvironmentModel:
        """
        Get the environmental model.

        Returns:
            EnvironmentModel: Contains wind and current settings and
            provides accelerations in body-fixed frame.
        """

        return self._env

    def integrate(self, actuator,
                  time_step,
                  enable_rudder_angle_scaling: Optional[bool] = False) -> None:
        """
        Advance the vessel state by a single physics integration step.

        Calls dynamics.step() to obtain updated body-fixed velocities (u, v, r),
        then applies environmental forces (unless the model has its own, see
        the class docstring), velocity clipping, and midpoint position
        integration.

        Args:
            actuator:                    ActuatorState containing the current rudder_angle
                                         and thrust command, already updated for the step.
            time_step:                   Physics integration step size in seconds.
            enable_rudder_angle_scaling: Enable rudder angle scaling.

        Notes:
            - EnvironmentModel accelerations are added to (u, v) before
              clipping, unless dynamics.models_own_wind_current is True.
            - Velocity clipping is applied for all model types.
            - Earth-fixed position is updated using midpoint heading integration.
            - Heading is wrapped to [-π, π].
        """

        state = self._state

        # Advance dynamics, returns updated (u, v, r)
        u, v, r = self._dynamics.step(state, actuator.rudder_angle, actuator.thrust,
                                      time_step, enable_rudder_angle_scaling)

        # Add environmental accelerations, except for models that already
        # include wind and current themselves (avoids applying them twice).
        if self._add_environment:
            ax_env, ay_env = self._env.accelerations(state.heading)
            u += ax_env * time_step
            v += ay_env * time_step

        # Apply velocity limits (all model types)
        u = np.clip(u, *self._specs.surge_limits)
        v = np.clip(v, *self._specs.sway_limits)
        r = np.clip(r, *self._specs.yaw_rate_limits)

        # Midpoint heading for improved position integration accuracy
        psi_mid = state.heading + 0.5 * r * time_step
        dx = u * np.cos(psi_mid) - v * np.sin(psi_mid)
        dy = u * np.sin(psi_mid) + v * np.cos(psi_mid)

        self._state = VesselState(
            terminated=state.terminated,
            truncated=state.truncated,
            x=state.x + dx * time_step,
            y=state.y + dy * time_step,
            heading=(state.heading + r * time_step + np.pi) % (2.0 * np.pi) - np.pi,
            u=u,
            v=v,
            r=r,
        )

    def step(self, action: np.ndarray, enable_smoothing: bool = False, enable_rudder_angle_scaling: bool = False) -> None:
        """
        Advance the ship simulator by one environment time step.

        The step method maps a normalized RL action to physical actuator
        commands, applies actuator dynamics with optional smoothing, and
        integrates the vessel state over the environment time step using
        repeated calls to `integrate`.

        Args:
            action:                      Normalized action array [rudder, thrust] in [-1, 1].
            enable_smoothing:            Enable actuator rate limiting and smoothing.
            enable_rudder_angle_scaling: Enable rudder angle scaling.

        Raises:
            ValueError: If `action` does not have shape (2,).

        Notes:
            - The action is mapped to physical actuator commands via ActionMapper.
            - For physics-based models, ActuatorModel.step() is called once per sub-step.
            - Partial physics steps are used if the environment time step is not
              an integer multiple of the dynamics time step.
        """

        # Validate action
        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")

        dyn_model_time_step = self.PHYSICS_DT
        n_full = self._n_sub_steps
        dt_rem = self._dt_remainder

        # Map normalized actions to physical actuators
        command = self._mapper.map(action, self._specs)

        for i in range(n_full):
            actuator = self._actuators.step(command, dyn_model_time_step, enable_smoothing)
            self.integrate(actuator, dyn_model_time_step, enable_rudder_angle_scaling)

        if dt_rem > 0.0:
            actuator = self._actuators.step(command, dt_rem, enable_smoothing)
            self.integrate(actuator, dt_rem, enable_rudder_angle_scaling)

        if isinstance(self._dynamics, FhTabularModel):
            fh_pos = self._dynamics._engine.ship_interface.getShipPosition()
            fh_hdg = np.radians(self._dynamics._engine.ship_interface.getShipHeading())

            s = self._state
            self._state = VesselState(
                terminated=s.terminated,
                truncated=s.truncated,
                x=fh_pos.x,
                y=fh_pos.y,
                heading=(fh_hdg + np.pi) % (2.0 * np.pi) - np.pi,
                u=s.u,
                v=s.v,
                r=s.r,
            )
