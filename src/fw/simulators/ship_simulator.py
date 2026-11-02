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
    Ship simulator: actuator dynamics plus a dynamics model (Fossen, Nomoto,
    UGentSurrogate, UGentB01Surrogate, UAntwerpSurrogate or FhTabularModel),
    integrated into an earth-fixed state with midpoint heading integration.

    Every model implements u, v, r = dynamics.step(state, rudder, thrust, dt)
    and returns unclipped velocities (FH reads them from its engine);
    ShipSimulator clips them for all models.

    Wind and current: EnvironmentModel accelerations are added only when
    dynamics.models_own_wind_current is False (Fossen, Nomoto, UGentSurrogate,
    UAntwerpSurrogate open_water/confined_ukc). FhTabularModel and full_env
    UAntwerpSurrogate model them themselves, so the wind/current switches have
    no effect on them. Models with supports_environment_model = False
    (UGentB01Surrogate, model scale) refuse wind/current with a ValueError.

    Sub-stepping: step() integrates in PHYSICS_DT (0.1 s) sub-steps plus a
    remainder sub-step if dt is not a multiple. UAntwerpSurrogate only runs at
    0.1 s, so for it a dt that is not an integer multiple of 0.1 s raises a
    ValueError in __init__.
    """

    # Physics integration sub-step [s], equal to the surrogate training step.
    PHYSICS_DT: float = SURROGATE_DT


    def __init__(self, specs: ShipSpecifications, dynamics: DynamicsBase, dt: float, wind: bool = False,
                 current: bool = False):
        """
        Args:
            specs:    Vessel dimensions, actuator limits and kinematic constraints.
            dynamics: DynamicsBase implementing step().
            dt:       Environment time step [s], positive.
            wind:     Enable EnvironmentModel wind (ignored if the model has its own).
            current:  Enable EnvironmentModel current (ignored if the model has its own).

        Raises:
            ValueError: dt not a multiple of PHYSICS_DT for UAntwerpSurrogate, or
                wind/current enabled for a model with supports_environment_model = False.
            TypeError: the model's wind/current flags are not bools.
        """

        super().__init__(specs, dt)

        # Sub-step grid, validated once here rather than deep inside a training run.
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

        # Fail fast on non-bool flags (e.g. a property returning None).
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
        EnvironmentModel holding wind/current settings (body-fixed accelerations).
        """

        return self._env

    def integrate(self, actuator,
                  time_step,
                  enable_rudder_angle_scaling: Optional[bool] = False) -> None:
        """
        Advance the state by one physics sub-step: dynamics.step(), environment
        accelerations (see class docstring), velocity clipping, then midpoint
        position integration; heading wrapped to [-pi, pi).

        Args:
            actuator:                    ActuatorState (rudder_angle, thrust) for this sub-step.
            time_step:                   Sub-step size [s].
            enable_rudder_angle_scaling: Passed on to dynamics.step().
        """

        state = self._state

        u, v, r = self._dynamics.step(state, actuator.rudder_angle, actuator.thrust,
                                      time_step, enable_rudder_angle_scaling)

        # Skipped for models that include wind/current themselves.
        if self._add_environment:
            ax_env, ay_env = self._env.accelerations(state.heading)
            u += ax_env * time_step
            v += ay_env * time_step

        u = np.clip(u, *self._specs.surge_limits)
        v = np.clip(v, *self._specs.sway_limits)
        r = np.clip(r, *self._specs.yaw_rate_limits)

        # Midpoint heading for better position accuracy.
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
        Advance one environment time step: map the normalized action to actuator
        commands (ActionMapper), then per sub-step run the actuator model and
        integrate(). For FhTabularModel, position and heading are then taken from
        the FH engine.

        Args:
            action:                      Normalized [rudder, thrust] in [-1, 1].
            enable_smoothing:            Enable actuator rate limiting and smoothing.
            enable_rudder_angle_scaling: Passed on to dynamics.step().

        Raises:
            ValueError: If action does not have shape (2,).
        """

        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")

        dyn_model_time_step = self.PHYSICS_DT
        n_full = self._n_sub_steps
        dt_rem = self._dt_remainder

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
