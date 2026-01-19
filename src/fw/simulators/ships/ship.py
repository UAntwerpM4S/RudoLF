import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class ShipSpecifications:
    """Ship physical specifications"""
    length: float       # meters
    mass: float     # kg
    max_thrust: float       # normalized
    max_rudder_angle: float     # radians
    min_surge_velocity: float
    max_surge_velocity: float
    min_sway_velocity: float
    max_sway_velocity: float
    min_yaw_rate: float
    max_yaw_rate: float
    max_rudder_rate: float
    max_thrust_rate: float
    rudder_jitter_threshold: float


class Ship:
    def __init__(self, specifications: ShipSpecifications):
        self.specifications: ShipSpecifications = specifications

        # Internal actuator state
        self._rudder = 0.0
        self._thrust = 0.0

    @property
    def performed_action(self):
        return np.array([self._rudder, self._thrust], dtype=np.float32)


    def apply_control(self, action: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply rate-limited and saturated control.

        Args:
            action: Array of [rudder, thrust] commands
            dt: Time step in seconds

        Returns:
            np.ndarray: Array containing updated rudder and thrust
        """

        # Ensure numeric stability
        action = np.asarray(action, dtype=np.float32)
        target_rudder, target_thrust = action[0], action[1]

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

        # Thrust dynamics
        thrust_error = target_thrust - self._thrust
        thrust_step = np.clip(thrust_error, -max_thrust_step, max_thrust_step)
        self._thrust += thrust_step

        return np.array([self._rudder, self._thrust], dtype=np.float32)
