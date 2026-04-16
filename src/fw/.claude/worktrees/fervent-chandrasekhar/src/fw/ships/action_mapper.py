import numpy as np

from fw.ships.ship_specs import ShipSpecifications
from fw.ships.actuator_command import ActuatorCommand


class ActionMapper:
    """
    Maps normalized control actions to physical actuator commands.

    This utility class converts dimensionless control inputs—typically
    originating from a controller or reinforcement-learning policy—into
    physically meaningful actuator commands for a vessel. The mapping
    enforces actuator limits defined in the ship specifications.

    The expected action space is two-dimensional:
        - action[0]: Rudder command, normalized to the interval [-1, 1].
        - action[1]: Thrust command, normalized to the interval [-1, 1].
    """

    @staticmethod
    def map(action: np.ndarray, specs: ShipSpecifications) -> ActuatorCommand:
        """
        Convert a normalized action vector into actuator commands.

        The rudder command is mapped linearly and symmetrically from the
        normalized interval [-1, 1] to the physical rudder angle limits.
        The thrust command is mapped linearly from [-1, 1] to the interval
        [min_thrust, max_thrust] defined in the ship specifications.

        Args:
            action: Normalized action vector of shape `(2,)` with values
                in the range [-1, 1].
            specs: Ship specification object containing actuator limits
                such as maximum rudder angle and thrust bounds.

        Returns:
            An `ActuatorCommand` instance containing:
                - rudder_angle: Physical rudder angle in radians.
                - thrust: Physical or normalized thrust command, scaled to
                  the vessel's actuator limits.

        Raises:
            ValueError: If `action` does not have shape `(2,)`.

        Notes:
            - No clipping is performed; callers are expected to ensure that
              the action values lie within [-1, 1].
            - The thrust mapping allows asymmetric thrust limits by
              respecting `min_thrust` and `max_thrust`.
        """

        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")

        rudder = action[0] * specs.max_rudder_angle
        thrust = (specs.min_thrust + 0.5 * (action[1] + 1.0) * (specs.max_thrust - specs.min_thrust))

        return ActuatorCommand(rudder_angle=rudder, thrust=thrust)
