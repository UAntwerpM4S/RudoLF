import numpy as np

from typing import Optional, Tuple
from fw.surrogate.schema import SURROGATE_DT
from fw.ships.vessel_state import VesselState
from fw.ships.action_mapper import ActionMapper
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.simulators.base_simulator import BaseSimulator
from fw.dynamics.fh_tabular_model import FhTabularModel
from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate


class ShipSimulator(BaseSimulator):
    """
    Ship simulator: actuator dynamics plus a dynamics model (Fossen, Nomoto,
    UGentSurrogate, UGentB01Surrogate, UAntwerpSurrogate or FhTabularModel),
    integrated into an earth-fixed state with midpoint heading integration.

    Every model implements u, v, r = dynamics.step(state, rudder, thrust, dt)
    and returns unclipped velocities (FH reads them from its engine);
    ShipSimulator clips them for all models.

    Wind and current come only from models that simulate them themselves
    (dynamics.models_own_wind_current: FhTabularModel, UAntwerpSurrogate
    full_env); nothing is added here.

    Sub-stepping: step() integrates in PHYSICS_DT (0.1 s) sub-steps plus a
    remainder sub-step if dt is not a multiple. UAntwerpSurrogate only runs at
    0.1 s, so for it a dt that is not an integer multiple of 0.1 s raises a
    ValueError in __init__.
    """

    # Physics integration sub-step [s], equal to the surrogate training step.
    PHYSICS_DT: float = SURROGATE_DT


    def __init__(self, specs: ShipSpecifications, dynamics: DynamicsBase, dt: float,
                 surge_limits: Optional[Tuple[float, float]] = None):
        """
        Args:
            specs:        Vessel dimensions, actuator limits and kinematic constraints
                          (kept by reference: profiles patch it in place).
            dynamics:     DynamicsBase implementing step().
            dt:           Environment time step [s], positive.
            surge_limits: (low, high) [m/s] replacing specs.surge_limits in
                          the clip (e.g. over-ground surge with a current);
                          None uses the specs.

        Raises:
            ValueError: dt not a multiple of PHYSICS_DT for UAntwerpSurrogate, or
                surge_limits not (low, high) with low < high.
        """

        super().__init__(specs, dt)
        if surge_limits is not None:
            low, high = (float(x) for x in surge_limits)
            if not low < high:
                raise ValueError(f"surge_limits={surge_limits} must be (low, high) with low < high.")
            surge_limits = (low, high)
        self._surge_limits = surge_limits

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

        self._dynamics = dynamics
        self._mapper = ActionMapper()

    def reset(self, state: VesselState, rng: Optional[np.random.Generator] = None,
              initial_thrust: Optional[float] = None) -> None:
        """
        Reset to `state` with the actuators at `initial_thrust` (see
        BaseSimulator.reset) and start a new episode of the dynamics model
        (DynamicsBase.reset_episode, with `rng`).
        """

        super().reset(state, initial_thrust)
        self._dynamics.reset_episode(rng)

    def integrate(self, actuator,
                  time_step,
                  enable_rudder_angle_scaling: Optional[bool] = False) -> None:
        """
        Advance the state by one physics sub-step: dynamics.step(), velocity
        clipping, then midpoint position integration; heading wrapped to [-pi, pi).

        Args:
            actuator:                    ActuatorState (rudder_angle, thrust) for this sub-step.
            time_step:                   Sub-step size [s].
            enable_rudder_angle_scaling: Passed on to dynamics.step().
        """

        state = self._state

        u, v, r = self._dynamics.step(state, actuator.rudder_angle, actuator.thrust,
                                      time_step, enable_rudder_angle_scaling)

        u = np.clip(u, *(self._specs.surge_limits if self._surge_limits is None else self._surge_limits))
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
            fh_x, fh_y, fh_hdg, *_ = self._dynamics.read_state()

            s = self._state
            self._state = VesselState(
                terminated=s.terminated,
                truncated=s.truncated,
                x=fh_x,
                y=fh_y,
                heading=fh_hdg,
                u=s.u,
                v=s.v,
                r=s.r,
            )
