import numpy as np

from fw.simulators.ships.actuator_state import ActuatorState
from fw.simulators.ships.ship_specs import ShipSpecifications
from fw.simulators.ships.actuator_command import ActuatorCommand


class ActuatorModel:
    def __init__(self, specifications: ShipSpecifications):
        self.specifications: ShipSpecifications = specifications

        # Internal actuator state
        self._rudder = 0.0
        self._thrust = 0.0

        self._initialized = False

    @property
    def performed_action(self):
        return np.array([self._rudder, self._thrust], dtype=np.float32)

    def reset(self):
        self._rudder = 0.0
        self._thrust = 0.0

        self._initialized = False

    def apply_smoothing(self, action: np.ndarray, enable: bool, alpha: float = 0.3) -> ActuatorState:
        """
        Apply action smoothing.

        Args:
            action: Array of [rudder, thrust] commands
            enable: Enable/disable smoothing
            alpha: Smoothing factor

        Returns:
            ActuatorState: Array containing updated rudder and thrust
        """

        # Smooth action application
        if not self._initialized or not enable:
            self._rudder = action[0]
            self._thrust = abs(action[1])
        else:
            self._rudder = alpha * action[0] + (1 - alpha) * self._rudder
            self._thrust = alpha * abs(action[1]) + (1 - alpha) * self._thrust

        self._initialized = True

        return ActuatorState(self._rudder, self._thrust)


    def step(self, action: ActuatorCommand, dt: float, enable_smoothing: bool) -> ActuatorState:
        """
        Apply rate-limited and saturated control.

        Args:
            action: Array of [rudder, thrust] commands
            dt: Time step in seconds
            enable_smoothing: Enable/disable smoothing

        Returns:
            ActuatorState: Array containing updated rudder and thrust
        """

        assert dt > 0.0

        # Ensure numeric stability
        target_rudder, target_thrust = action.rudder_angle, action.thrust

        if enable_smoothing:
            # Rate limits (per-second -> per-step)
            max_rudder_step = self.specifications.max_rudder_rate * dt
            max_thrust_step = self.specifications.max_thrust_rate * dt

            # Rudder dynamics
            rudder_error = target_rudder - self._rudder
            # Deadband on change to prevent jitter
            if abs(rudder_error) < self.specifications.rudder_jitter_threshold:
                rudder_step = 0.0
            else:
                rudder_step = np.clip(rudder_error, -max_rudder_step, max_rudder_step)
            self._rudder += rudder_step

            # Rudder safety clamp
            self._rudder = np.clip(self._rudder, -self.specifications.max_rudder_angle, self.specifications.max_rudder_angle)

            # Thrust dynamics
            thrust_error = target_thrust - self._thrust
            if abs(thrust_error) < self.specifications.thrust_jitter_threshold:
                thrust_step = 0.0
            else:
                thrust_step = np.clip(thrust_error, -max_thrust_step, max_thrust_step)
            self._thrust += thrust_step

            # Thrust safety clamp
            self._thrust = np.clip(self._thrust, self.specifications.min_thrust, self.specifications.max_thrust)
        else:
            self._rudder = np.clip(target_rudder, -self.specifications.max_rudder_angle, self.specifications.max_rudder_angle)
            self._thrust = np.clip(target_thrust, self.specifications.min_thrust, self.specifications.max_thrust)

        return ActuatorState(self._rudder, self._thrust)
