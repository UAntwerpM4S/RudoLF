import numpy as np

from fw.simulators.ships.vessel_state import VesselState
from fw.simulators.ships.action_mapper import ActionMapper
from fw.simulators.ships.ship_specs import ShipSpecifications
from fw.simulators.dynamics.dynamics_model import DynamicsModel
from fw.simulators.simulation.base_simulator import BaseSimulator
from fw.simulators.dynamics.environmental_model import EnvironmentModel


class FossenSimulator(BaseSimulator):
    """
    Clean Fossen 3DOF simulator implementing the unified interface.
    """

    def __init__(self, specs: ShipSpecifications, dynamics: DynamicsModel, dt: float, wind: bool = False,
                 current: bool = False):
        super().__init__(specs, dt)

        self._dynamics = dynamics
        self._mapper = ActionMapper()
        self._env = EnvironmentModel(wind, current)

    @property
    def environment(self) -> EnvironmentModel:
        return self._env

    def step(self, action: np.ndarray, enable_smoothing: bool) -> VesselState:
        s = self._state

        # Map normalized actions to physical actuators
        command = self._mapper.map(action, self._specs)
        actuator = self._actuators.step(command, self.dt, enable_smoothing)

        # Compute accelerations from dynamics
        du, dv, dr = self._dynamics.accelerations(s.u, s.v, s.r, actuator.rudder_angle, actuator.thrust)

        # Add environmental accelerations
        ax_env, ay_env = self._env.accelerations(s.heading)
        du += ax_env
        dv += ay_env

        # Apply velocity limits
        u = np.clip(s.u + du * self.dt, *self._specs.surge_limits)
        v = np.clip(s.v + dv * self.dt, *self._specs.sway_limits)
        r = np.clip(s.r + dr * self.dt, *self._specs.yaw_rate_limits)

        # Midpoint heading for better integration
        psi_mid = s.heading + 0.5 * r * self.dt
        dx = u * np.cos(psi_mid) - v * np.sin(psi_mid)
        dy = u * np.sin(psi_mid) + v * np.cos(psi_mid)

        self._state = VesselState(
            x=s.x + dx * self.dt,
            y=s.y + dy * self.dt,
            heading=(s.heading + r * self.dt + np.pi) % (2.0 * np.pi) - np.pi,
            u=u,
            v=v,
            r=r,
        )

        return self._state
