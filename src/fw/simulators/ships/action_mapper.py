import numpy as np

from fw.simulators.ships.ship_specs import ShipSpecifications
from fw.simulators.ships.actuator_command import ActuatorCommand


class ActionMapper:
    """
    Maps normalized actions [-1, 1] to physical actuator commands.
    """

    @staticmethod
    def map(action: np.ndarray, specs: ShipSpecifications) -> ActuatorCommand:
        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")

        rudder = action[0] * specs.max_rudder_angle
        thrust = (specs.min_thrust + 0.5 * (action[1] + 1.0) * (specs.max_thrust - specs.min_thrust))

        return ActuatorCommand(rudder_angle=rudder, thrust=thrust)
