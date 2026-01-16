import numpy as np

from fw.simulators.ships.ship import Ship, ShipSpecifications


def create_myzako() -> Ship:
    return Ship(
        specifications=ShipSpecifications(
            length=110.0,
            mass=3.86e6,
            max_thrust=1.0,
            max_rudder_angle=np.radians(60.0),
            min_surge_velocity=0.0,
            max_surge_velocity=5.0,
            min_sway_velocity=-2.0,
            max_sway_velocity=2.0,
            min_yaw_rate=-0.5,
            max_yaw_rate=0.5,
            max_rudder_rate=1.0,
            max_thrust_rate=0.5,
            rudder_jitter_threshold=0.01,
        ),
    )
