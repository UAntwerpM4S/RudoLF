import numpy as np

from fw.simulators.ships.ship_specs import ShipSpecifications


class MyzakoShip:
    """
    Minimal ship model with predefined specifications for simulation.

    This class defines a simple, self-contained ship model suitable
    for use in simulators and control experiments. It provides only
    the vessel specifications and does not include dynamics, actuators,
    or environmental effects.

    Attributes:
        specifications: ShipSpecifications object containing physical
            properties, actuator limits, and operational constraints.
    """

    def __init__(self):
        """
        Initialize the Myzako ship with default parameters.

        The default values are intended for simulation and control
        testing purposes:

            - Length: 110 m
            - Mass: 3.86e6 kg
            - Maximum rudder angle: 60 deg
            - Maximum rudder rate: 1 rad/s
            - Thrust: min 0.1, max 1.0 (normalized)
            - Maximum thrust rate: 0.5 /s
            - Rudder jitter threshold: 0.01 rad
            - Thrust jitter threshold: 0.005
            - Surge velocity limits: 0 – 5 m/s
            - Sway velocity limits: -2 – 2 m/s
            - Yaw rate limits: -0.5 – 0.5 rad/s
        """

        self.specifications = ShipSpecifications(
            length=110.0,
            mass=3.86e6,
            max_rudder_angle=np.radians(60.0),
            max_rudder_rate=1.0,
            min_thrust=0.1,
            max_thrust=1.0,
            max_thrust_rate=0.5,
            rudder_jitter_threshold=0.01,
            thrust_jitter_threshold=0.005,
            surge_limits=(0.0, 5.0),
            sway_limits=(-2.0, 2.0),
            yaw_rate_limits=(-0.5, 0.5),
        )

def create_myzako() -> MyzakoShip:
    """
    Factory function to create a MyzakoShip instance.

    Returns:
        MyzakoShip: A new ship object with default specifications
        suitable for use in simulators and experiments.

    Example:
        >>> ship = create_myzako()
        >>> ship.specifications.length
        110.0
    """

    return MyzakoShip()
