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


class ShipSimulator(BaseSimulator):
    """
    Ship simulator implementing the unified interface.

    This simulator can integrate:

        - Several dynamics models: Fossen 3-DOF, Nomoto, UGentSurrogate,
          UAntwerpSurrogate, FhTabularModel.
        - Actuator dynamics and rate limiting.
        - Environmental forces (wind and current) for physics-based models.
        - A unified interface compatible with RL or control loops.

    The simulator computes the vessel state in earth-fixed coordinates
    using body-fixed velocities and yaw rate. Integration uses a
    midpoint approximation for improved accuracy.

    Dynamics model contract
    -----------------------
    All DynamicsBase implementations expose a single step() method:

        u, v, r = dynamics.step(state, rudder_angle, thrust, dt)

    Two families exist:

    Physics-based models (Fossen, Nomoto, UGentSurrogate):
        Return un-clipped (u, v, r) computed via Euler integration of
        their equations of motion. ShipSimulator applies velocity clipping,
        and environmental forces after the call.

    Engine-backed models (FhTabularModel, UAntwerpSurrogate):
        Return (u, v, r) directly from an internal engine or surrogate after
        advancing by dt. No re-integration is performed by ShipSimulator;
        clipping is still applied for consistency.
        Environmental forces are NOT added for these models, as their engine
        already accounts for relevant dynamics.

    Sub-stepping and dt constraints
    --------------------------------
    step() always uses a 0.1 s physics sub-step.  If the environment dt is
    not an exact multiple of 0.1 s a fractional remainder sub-step is issued.
    UAntwerpSurrogate asserts dt == 0.1 s on every call, so it will crash on
    a remainder sub-step.  Use an environment dt that is an exact integer
    multiple of 0.1 s when using UAntwerpSurrogate (e.g. 0.1, 0.5, 1.0, 2.0).
    """

    def __init__(self, specs: ShipSpecifications, dynamics: DynamicsBase, dt: float, wind: bool = False,
                 current: bool = False):
        """
        Initialize the ship simulator.

        Args:
            specs:             ShipSpecifications object defining vessel dimensions,
                               actuator limits, and kinematic constraints.
            dynamics:          DynamicsBase instance implementing the step() interface.
            dt:                Simulation time step [s]. Must be positive.
            wind:              Enable wind influence if True.
            current:           Enable water current influence if True.
        """

        super().__init__(specs, dt)

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
        then applies environmental forces (for physics-based models), velocity
        clipping, and midpoint position integration.

        Args:
            actuator:                    ActuatorState containing the current rudder_angle
                                         and thrust command, already updated for the step.
            time_step:                   Physics integration step size in seconds.
            enable_rudder_angle_scaling: Enable rudder angle scaling.

        Notes:
            - For physics-based models, environmental accelerations are added
              to (u, v) before clipping.
            - For engine-backed models (FhTabularModel, UAntwerpSurrogate),
              environmental forces are skipped; the engine handles them internally.
            - Velocity clipping is applied for all model types.
            - Earth-fixed position is updated using midpoint heading integration.
            - Heading is wrapped to [-π, π].
        """

        state = self._state

        # Advance dynamics — returns updated (u, v, r)
        u, v, r = self._dynamics.step(state, actuator.rudder_angle, actuator.thrust,
                                      time_step, enable_rudder_angle_scaling)

        # Add environmental accelerations, except for FhTabularModel and UAntwerpSurrogate,
        # as they already add environmental effects internally.
        if not isinstance(self._dynamics, (FhTabularModel, UAntwerpSurrogate)):
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
            x=np.float32(state.x + dx * time_step),
            y=np.float32(state.y + dy * time_step),
            heading=np.float32((state.heading + r * time_step + np.pi) % (2.0 * np.pi) - np.pi),
            u=np.float32(u),
            v=np.float32(v),
            r=np.float32(r),
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

        dyn_model_time_step = 0.1
        assert dyn_model_time_step <= self.dt
        n_full = int(self.dt / dyn_model_time_step)
        dt_rem = self.dt - n_full * dyn_model_time_step

        # guard against floating noise
        if dt_rem < 1e-15:
            dt_rem = 0.0

        # Map normalized actions to physical actuators
        command = self._mapper.map(action, self._specs)

        for i in range(n_full):
            actuator = self._actuators.step(command, dyn_model_time_step, enable_smoothing)
            self.integrate(actuator, dyn_model_time_step, enable_rudder_angle_scaling)

        if dt_rem > 1e-12:
            actuator = self._actuators.step(command, dt_rem, enable_smoothing)
            self.integrate(actuator, dt_rem, enable_rudder_angle_scaling)

        s = self._state
        if isinstance(self._dynamics, FhTabularModel):
            fh_pos = self._dynamics._engine.ship_interface.getShipPosition()
            fh_hdg = np.radians(self._dynamics._engine.ship_interface.getShipHeading())

            self._state = VesselState(
                terminated=s.terminated,
                truncated=s.truncated,
                x=np.float32(fh_pos.x),
                y=np.float32(fh_pos.y),
                heading=np.float32((fh_hdg + np.pi) % (2.0 * np.pi) - np.pi),
                u=np.float32(s.u),
                v=np.float32(s.v),
                r=np.float32(s.r),
            )
