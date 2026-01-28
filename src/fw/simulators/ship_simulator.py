import numpy as np

from fw.ships.vessel_state import VesselState
from fw.ships.action_mapper import ActionMapper
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsModel
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

    def __init__(self, specs: ShipSpecifications, dynamics: DynamicsModel, dt: float, wind: bool = False,
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

    def step(self, action: np.ndarray, enable_smoothing: bool = True) -> None:
        """
        Advance the ship simulator by one time step.

        Updates `VesselState` after applying dynamics,
        actuator commands, and environmental effects.

        Args:
            action: Normalized action array [rudder, thrust] in [-1, 1].
            enable_smoothing: Enable actuator rate limiting and smoothing.

        Raises:
            ValueError: If action does not have shape (2,).

        Notes:
            - Normalized actions are mapped to physical actuators using
              ActionMapper.
            - Actuator dynamics are applied using ActuatorModel with optional
              smoothing.
            - Vessel accelerations are computed using the provided DynamicsModel.
            - Environmental accelerations from wind and current are added.
            - Surge, sway, and yaw velocities are clipped to simulation limits.
            - Midpoint integration is used to update earth-fixed positions.
            - Heading is wrapped to [-π, π].
        """

        s = self._state

        # Validate action
        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")

        # Map normalized actions to physical actuators
        command = self._mapper.map(action, self._specs)
        actuator = self._actuators.step(command, self.dt, enable_smoothing)

        # Compute accelerations from dynamics
        du, dv, dr = self._dynamics.accelerations(s.u, s.v, s.r, actuator.rudder_angle, actuator.thrust)

        # Add environmental accelerations
        ax_env, ay_env = self._env.accelerations(s.heading)
        du += ax_env
        dv += ay_env

        # Surge damping
        surge_damping_factor = abs(self._dynamics.X_u / self._dynamics.m11) if self.numerical_damping else 0.0
        # Sway damping
        sway_damping_factor = abs(self._dynamics.Y_v / self._dynamics.m22) if self.numerical_damping else 0.0
        # Yaw damping
        yaw_damping_factor = abs(self._dynamics.N_r / self._dynamics.m33) if self.numerical_damping else 0.0

        # Apply velocity limits
        u = np.clip((s.u + du * self.dt) / (1.0 + surge_damping_factor * self.dt), *self._specs.surge_limits)
        v = np.clip((s.v + dv * self.dt) / (1.0 + sway_damping_factor * self.dt), *self._specs.sway_limits)
        r = np.clip((s.r + dr * self.dt) / (1.0 + yaw_damping_factor * self.dt), *self._specs.yaw_rate_limits)

        # Midpoint heading for better integration
        psi_mid = s.heading + 0.5 * r * self.dt
        dx = u * np.cos(psi_mid) - v * np.sin(psi_mid)
        dy = u * np.sin(psi_mid) + v * np.cos(psi_mid)

        self._state = VesselState(
            x=np.float32(s.x + dx * self.dt),
            y=np.float32(s.y + dy * self.dt),
            heading=np.float32((s.heading + r * self.dt + np.pi) % (2.0 * np.pi) - np.pi),
            u=np.float32(u),
            v=np.float32(v),
            r=np.float32(r),
        )
