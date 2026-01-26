import numpy as np

from fw.simulators.ships.ship_specs import ShipSpecifications


class MyzakoShip:
    """Minimal ship model compatible with simulators."""

    def __init__(self):
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
    return MyzakoShip()
