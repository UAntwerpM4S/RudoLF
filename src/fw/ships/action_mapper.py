import numpy as np

from fw.ships.ship_specs import ShipSpecifications
from fw.ships.actuator_command import ActuatorCommand


class ActionMapper:
    """
    Maps normalized actions [rudder, thrust], both in [-1, 1], to actuator commands
    within the ship's limits.
    """

    @staticmethod
    def map(action: np.ndarray, specs: ShipSpecifications) -> ActuatorCommand:
        """
        Convert a normalized action vector into actuator commands.

        Rudder maps linearly to [-max_rudder_angle, max_rudder_angle] (rad);
        thrust maps linearly to [min_thrust, max_thrust]. No clipping is done.

        Args:
            action: Array of shape (2,) with values in [-1, 1].
            specs: Ship specifications with the actuator limits.

        Returns:
            ActuatorCommand with rudder_angle [rad] and thrust.

        Raises:
            ValueError: If `action` does not have shape (2,).
        """

        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")

        rudder = action[0] * specs.max_rudder_angle
        thrust = (specs.min_thrust + 0.5 * (action[1] + 1.0) * (specs.max_thrust - specs.min_thrust))

        return ActuatorCommand(rudder_angle=rudder, thrust=thrust)

    @staticmethod
    def map_thrust(thrust_action: float, specs: ShipSpecifications) -> float:
        """
        Thrust for a normalized thrust action, as in map(), but in double
        precision and clipped to [min_thrust, max_thrust] (so rounding cannot
        leave the limits).

        Args:
            thrust_action: Thrust action in [-1, 1].
            specs: Ship specifications with the thrust limits.
        """

        thrust = specs.min_thrust + 0.5 * (float(thrust_action) + 1.0) * (specs.max_thrust - specs.min_thrust)
        return float(min(max(thrust, specs.min_thrust), specs.max_thrust))
