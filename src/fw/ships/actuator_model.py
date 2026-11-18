import numpy as np

from typing import Optional

from fw.ships.actuator_state import ActuatorState
from fw.ships.ship_specs import ShipSpecifications
from fw.ships.actuator_command import ActuatorCommand


class ActuatorModel:
    """
    Rudder and thrust actuator response to commands.

    With `enable_smoothing`, changes are rate limited and errors below a jitter
    threshold are ignored; without it, actuators jump to the command. Both are
    clipped to physical limits. Despite its name, the toggle does rate
    limiting only; no low-pass filter is applied.

    Attributes:
        _rudder: Current rudder angle [rad].
        _thrust: Current thrust command (normalized or physical units).
    """

    def __init__(self, specifications: ShipSpecifications):
        """
        Initialize the actuator model with vessel specifications.

        Args:
            specifications: ShipSpecifications object containing actuator
                limits (max/min angles, max rates, jitter thresholds).
        """

        self.specifications: ShipSpecifications = specifications

        self._rudder = 0.0  # Current rudder angle [rad]
        self._thrust = specifications.min_thrust  # Current thrust [-]

    @property
    def current_actuator_state(self):
        """
        Current actuator state as float32 array [rudder_angle, thrust].
        """

        return np.array([self._rudder, self._thrust], dtype=np.float32)

    def reset(self, thrust: Optional[float] = None) -> None:
        """
        Reset the rudder to zero and the thrust to its episode start value.

        Args:
            thrust: Start thrust within [min_thrust, max_thrust]; None = min_thrust
                (the lowest value the actuator can hold).

        Raises:
            ValueError: If thrust is outside [min_thrust, max_thrust] (or NaN).
        """

        spec = self.specifications
        if thrust is None:
            thrust = spec.min_thrust
        # "not (a <= x <= b)" also rejects NaN
        if not spec.min_thrust <= thrust <= spec.max_thrust:
            raise ValueError(f"Start thrust {thrust} is outside [{spec.min_thrust}, {spec.max_thrust}].")

        self._rudder = 0.0
        self._thrust = float(thrust)

    def step(self, action: ActuatorCommand, dt: float, enable_smoothing: bool) -> ActuatorState:
        """
        Move rudder and thrust toward the command for one time step.

        Args:
            action: Desired ActuatorCommand containing rudder_angle [rad]
                and thrust [-].
            dt: Time step in seconds.
            enable_smoothing: Enable rate limiting and jitter suppression
                (clipping to limits always applies).

        Returns:
            ActuatorState with the resulting actuator positions.

        Raises:
            ValueError: If dt is not positive.
        """

        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")

        target_rudder, target_thrust = action.rudder_angle, action.thrust

        if enable_smoothing:
            # Rate limits (per-second -> per-step)
            max_rudder_step = self.specifications.max_rudder_rate * dt
            max_thrust_step = self.specifications.max_thrust_rate * dt

            # Rudder dynamics with jitter suppression
            rudder_error = target_rudder - self._rudder
            if abs(rudder_error) < self.specifications.rudder_jitter_threshold:
                rudder_step = 0.0
            else:
                rudder_step = np.clip(rudder_error, -max_rudder_step, max_rudder_step)
            self._rudder += rudder_step

            # Rudder saturation
            self._rudder = np.clip(self._rudder, -self.specifications.max_rudder_angle, self.specifications.max_rudder_angle)

            # Thrust dynamics with jitter suppression
            thrust_error = target_thrust - self._thrust
            if abs(thrust_error) < self.specifications.thrust_jitter_threshold:
                thrust_step = 0.0
            else:
                thrust_step = np.clip(thrust_error, -max_thrust_step, max_thrust_step)
            self._thrust += thrust_step

            # Thrust saturation
            self._thrust = np.clip(self._thrust, self.specifications.min_thrust, self.specifications.max_thrust)
        else:
            self._rudder = np.clip(target_rudder, -self.specifications.max_rudder_angle, self.specifications.max_rudder_angle)
            self._thrust = np.clip(target_thrust, self.specifications.min_thrust, self.specifications.max_thrust)

        return ActuatorState(self._rudder, self._thrust)
