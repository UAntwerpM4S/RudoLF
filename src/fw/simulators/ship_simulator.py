import copy
import numpy as np

from fw.ships.vessel_state import VesselState
from fw.ships.action_mapper import ActionMapper
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.simulators.base_simulator import BaseSimulator
from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate
from fw.dynamics.environmental_model import EnvironmentModel


class ShipSimulator(BaseSimulator):
    """
    Ship simulator implementing the unified interface.

    This simulator can integrate:

        - Several models like the Fossen 3DOF vessel dynamics model, Nomoto, MMG, etc.
        - Actuator dynamics and rate limiting
        - Environmental forces (wind and current)
        - Unified interface compatible with RL or control loops

    The simulator computes the vessel state in earth-fixed coordinates
    using body-fixed velocities and yaw rate. Integration uses a
    midpoint approximation for improved accuracy.
    """

    def __init__(self, specs: ShipSpecifications, dynamics: DynamicsBase, dt: float, wind: bool = False,
                 current: bool = False, numerical_damping: bool = False):
        """
        Initialize the ship simulator.

        Args:
            specs: ShipSpecifications object defining vessel dimensions,
                actuator limits, and kinematic constraints.
            dynamics: DynamicsModel instance implementing 3DOF accelerations.
            dt: Simulation time step [s]. Must be positive.
            wind: Enable wind influence if True.
            current: Enable water current influence if True.
        """

        super().__init__(specs, dt)

        self._dynamics = dynamics
        self._mapper = ActionMapper()
        self.numerical_damping = numerical_damping
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

    def integrate(self, actuator, time_step, surge_damping_factor, sway_damping_factor, yaw_damping_factor) -> None:
        """
        Advance the vessel state by a single physics integration step.

        Performs one substep of the continuous-time vessel dynamics using a
        zero-order hold actuator input. The provided actuator state is assumed
        to remain constant over the given time_step.

        Args:
            actuator: ActuatorState containing the current rudder_angle [rad]
                and thrust command [-], already updated for the environment step.
            time_step: Physics integration step size in seconds.
            surge_damping_factor: Linear damping coefficient applied to surge
                velocity (u).
            sway_damping_factor: Linear damping coefficient applied to sway
                velocity (v).
            yaw_damping_factor: Linear damping coefficient applied to yaw
                rate (r).

        Notes:
            - The actuator state is not modified in this method.
            - Vessel accelerations are computed using the DynamicsModel.
            - Environmental accelerations (e.g., wind and current) are added
              in the body-fixed frame.
            - Linear damping is applied implicitly in surge, sway, and yaw.
            - Body-fixed velocities (u, v, r) are clipped to limits defined
              in ShipSpecifications.
            - Earth-fixed position is updated using midpoint integration
              for improved numerical accuracy.
            - Heading is wrapped to the interval [-π, π].
        """

        state = copy.deepcopy(self._state)

        # Compute accelerations from dynamics
        du, dv, dr = self._dynamics.accelerations(state, actuator.rudder_angle, actuator.thrust)

        # Add environmental accelerations
        ax_env, ay_env = self._env.accelerations(state.heading)
        du += ax_env
        dv += ay_env

        # Apply velocity limits
        u = np.clip((state.u + du * time_step) / (1.0 + surge_damping_factor * time_step),
                    *self._specs.surge_limits)
        v = np.clip((state.v + dv * time_step) / (1.0 + sway_damping_factor * time_step),
                    *self._specs.sway_limits)
        r = np.clip((state.r + dr * time_step) / (1.0 + yaw_damping_factor * time_step),
                    *self._specs.yaw_rate_limits)

        # Midpoint heading for better integration
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
            action: Normalized action array [rudder, thrust] in [-1, 1].
            enable_smoothing: Enable actuator rate limiting and smoothing.
            enable_rudder_angle_scaling: Accepted for API consistency with
                :class:`~fw.simulators.fh_sim_wrapper.FhSimWrapper` but has
                no effect here. ShipSimulator uses its own
                :class:`~fw.ships.ship_specs.ShipSpecifications` directly, so
                there is no angle gap to compensate for.

        Raises:
            ValueError: If `action` does not have shape (2,).

        Notes:
            - The action is mapped to physical actuator commands via
              ActionMapper.
            - Actuator dynamics (smoothing/rate limiting) are applied before
              integration.
            - Vessel accelerations are computed using the DynamicsModel.
            - Environmental accelerations from wind and current are added
              in the body-fixed frame.
            - Linear damping is applied in surge, sway, and yaw according
              to the damping factors derived from the dynamics model.
            - Velocities are clipped to simulation limits specified in
              ShipSpecifications.
            - Midpoint integration is used to update earth-fixed positions
              and heading.
            - Heading is wrapped to [-π, π] after each integration step.
            - Partial physics steps are used if the environment time step
              is not an integer multiple of the dynamics time step.
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

        is_ua_surrogate = isinstance(self._dynamics, UAntwerpSurrogate)
        self._actuators.pass_through = is_ua_surrogate

        # Map normalized actions to physical actuators
        command = self._mapper.map(action, self._specs) if not is_ua_surrogate else action

        # Surge damping
        if self.numerical_damping and hasattr(self._dynamics, "X_u") and hasattr(self._dynamics, "m11"):
            surge_damping_factor = abs(self._dynamics.X_u / self._dynamics.m11)
        else:
            surge_damping_factor = 0.0
        # Sway damping
        if self.numerical_damping and hasattr(self._dynamics, "Y_v") and hasattr(self._dynamics, "m22"):
            sway_damping_factor = abs(self._dynamics.Y_v / self._dynamics.m22)
        else:
            sway_damping_factor = 0.0
        # Yaw damping
        if self.numerical_damping and hasattr(self._dynamics, "N_r") and hasattr(self._dynamics, "m33"):
            yaw_damping_factor = abs(self._dynamics.N_r / self._dynamics.m33)
        else:
            yaw_damping_factor = 0.0

        for i in range(n_full):
            actuator = self._actuators.step(command, dyn_model_time_step, enable_smoothing)
            self.integrate(actuator, dyn_model_time_step, surge_damping_factor, sway_damping_factor, yaw_damping_factor)

        if dt_rem > 1e-12:
            actuator = self._actuators.step(command, dt_rem, enable_smoothing)
            self.integrate(actuator, dt_rem, surge_damping_factor, sway_damping_factor, yaw_damping_factor)

        s = self._state
        self._state = VesselState(
            terminated=s.terminated,
            truncated=s.truncated,
            x=np.float32(s.x),
            y=np.float32(s.y),
            heading=np.float32(s.heading),
            u=np.float32(s.u),
            v=np.float32(s.v),
            r=np.float32(s.r),
        )
