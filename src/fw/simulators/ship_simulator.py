import numpy as np

from fw.ships.vessel_state import VesselState
from fw.ships.action_mapper import ActionMapper
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.simulators.base_simulator import BaseSimulator
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

    def integrate(self, command, time_step, surge_damping_factor, sway_damping_factor, yaw_damping_factor, enable_smoothing: bool = True) -> None:
        """
        Advance the ship state by a single physics integration step.

        Applies actuator commands, computes vessel accelerations, adds
        environmental effects, applies damping, clips velocities to
        limits, and updates the earth-fixed position using midpoint
        integration.

        Args:
            command: ActuatorCommand object containing the current
                rudder_angle and thrust setpoints.
            time_step: Time step (seconds) over which to integrate the dynamics.
            surge_damping_factor: Linear damping factor for surge velocity.
            sway_damping_factor: Linear damping factor for sway velocity.
            yaw_damping_factor: Linear damping factor for yaw rate.
            enable_smoothing: If True, actuator outputs are rate-limited
                and smoothed according to the ActuatorModel.

        Notes:
            - Rudder and thrust are processed through the ActuatorModel
              before computing dynamics.
            - Vessel accelerations are computed via the DynamicsModel.
            - Environmental accelerations (wind, current) are added
              in the body-fixed frame.
            - Velocity limits from ShipSpecifications are enforced.
            - Midpoint integration improves numerical accuracy for position
              and heading updates.
            - Heading is wrapped to [-π, π].
        """

        s = self._state
        actuator = self._actuators.step(command, time_step, enable_smoothing)

        # Compute accelerations from dynamics
        du, dv, dr = self._dynamics.accelerations(s.u, s.v, s.r, actuator.rudder_angle, actuator.thrust)

        # Add environmental accelerations
        ax_env, ay_env = self._env.accelerations(s.heading)
        du += ax_env
        dv += ay_env

        # Apply velocity limits
        u = np.clip((s.u + du * time_step) / (1.0 + surge_damping_factor * time_step),
                    *self._specs.surge_limits)
        v = np.clip((s.v + dv * time_step) / (1.0 + sway_damping_factor * time_step),
                    *self._specs.sway_limits)
        r = np.clip((s.r + dr * time_step) / (1.0 + yaw_damping_factor * time_step),
                    *self._specs.yaw_rate_limits)

        # Midpoint heading for better integration
        psi_mid = s.heading + 0.5 * r * time_step
        dx = u * np.cos(psi_mid) - v * np.sin(psi_mid)
        dy = u * np.sin(psi_mid) + v * np.cos(psi_mid)

        self._state = VesselState(
            terminated=s.terminated,
            truncated=s.truncated,
            x=s.x + dx * time_step,
            y=s.y + dy * time_step,
            heading=(s.heading + r * time_step + np.pi) % (2.0 * np.pi) - np.pi,
            u=u,
            v=v,
            r=r,
        )

    def step(self, action: np.ndarray, enable_smoothing: bool = True) -> None:
        """
        Advance the ship simulator by one environment time step.

        The step method maps a normalized RL action to physical actuator
        commands, applies actuator dynamics with optional smoothing, and
        integrates the vessel state over the environment time step using
        repeated calls to `integrate`.

        Args:
            action: Normalized action array [rudder, thrust] in [-1, 1].
            enable_smoothing: Enable actuator rate limiting and smoothing.

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

        # Map normalized actions to physical actuators
        command = self._mapper.map(action, self._specs)

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
            self.integrate(command, dyn_model_time_step, surge_damping_factor, sway_damping_factor, yaw_damping_factor, enable_smoothing)

        if dt_rem > 1e-12:
            self.integrate(command, dt_rem, surge_damping_factor, sway_damping_factor, yaw_damping_factor, enable_smoothing)

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
