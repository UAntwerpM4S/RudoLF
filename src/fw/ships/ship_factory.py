import numpy as np

from enum import auto, Enum
from fw.ships.ship_specs import ShipSpecifications


class ShipModel(Enum):
    """
    Enumeration of supported ship specification presets.
    """

    MYZAKO_DEFAULT = auto()
    MYZAKO_29 = auto()
    MYZAKO_37 = auto()
    TRIPOLI = auto()


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

    def __init__(self, ship_model: ShipModel):
        """
        Initialize the ship model with parameters specific to *ship_model*.

        Hydrodynamic coefficients (mass, length, added-mass, damping) differ
        per variant; the actuator and kinematic limits below are shared across
        all variants:

            - Maximum rudder angle : 46.5 deg  (≈ 0.812 rad)
            - Maximum rudder rate  : 6.0 deg/s (≈ 0.105 rad/s)
            - Rudder jitter threshold : 0.6 deg
            - Thrust               : min 0.1, max 1.0 (normalized)
            - Maximum thrust rate  : 0.07 /s
            - Thrust jitter threshold : 0.005
            - Surge velocity limits: 0 – 5 m/s
            - Sway velocity limits : -2 – 2 m/s
            - Yaw rate limits      : -0.5 – 0.5 rad/s

        Per-variant hull parameters:

            MYZAKO_DEFAULT / MYZAKO_29 / MYZAKO_37:
                Length 108–110 m, mass 3.13–4.19 × 10⁶ kg, identified
                hydrodynamic coefficients from model-test data.

            TRIPOLI:
                Length 110 m, mass 6.16 × 10⁶ kg, heuristic coefficients.
        """

        default_actuator_envelope = dict(
            max_rudder_angle=np.radians(46.5),
            max_rudder_rate=np.radians(6.0),
            min_thrust=0.1,
            max_thrust=1.0,
            max_thrust_rate=0.07,
            rudder_jitter_threshold=np.radians(0.6),
            thrust_jitter_threshold=0.005,
            surge_limits=(0.0, 5.0),
            sway_limits=(-2.0, 2.0),
            yaw_rate_limits=(-0.5, 0.5),
        )

        if ship_model == ShipModel.MYZAKO_29:
            mass = 3.13e6
            length = 108.0
            X_udot_coeff = -0.255
            Y_vdot_coeff = -0.845
            N_rdot_coeff = -0.047
            X_u_coeff = 4.995e-19
            Y_v_coeff = -0.0024
            N_r_coeff = 0.000875
            X_uu_coeff = -0.0023
            Y_vv_coeff = -0.0729
            N_rr_coeff = -0.0392
            Y_uv_coeff = -0.00426
            N_uv_coeff = -0.00007
            actuator_envelope = dict(default_actuator_envelope)
        elif ship_model == ShipModel.MYZAKO_37:
            mass = 4.19e6
            length = 108.0
            X_udot_coeff = -0.265
            Y_vdot_coeff = -0.86
            N_rdot_coeff = -0.047
            X_u_coeff = 6.712e-18
            Y_v_coeff = -0.0018
            N_r_coeff = 0.000806
            X_uu_coeff = -0.0015
            Y_vv_coeff = -0.0566
            N_rr_coeff = -0.0362
            Y_uv_coeff = -0.00467
            N_uv_coeff = -0.00007
            actuator_envelope = dict(default_actuator_envelope)
        elif ship_model == ShipModel.TRIPOLI:
            mass = 6161870.27   # 7.16e6
            length = 110.0  # 109.0
            X_udot_coeff = -0.05
            Y_vdot_coeff = -0.5
            N_rdot_coeff = -0.05
            X_u_coeff = -0.002
            Y_v_coeff = -0.02
            N_r_coeff = -0.001
            X_uu_coeff = -0.0005
            Y_vv_coeff = -0.005
            N_rr_coeff = -0.0005
            Y_uv_coeff = -0.005
            N_uv_coeff = -0.0005
            actuator_envelope = dict(default_actuator_envelope)
        else:
            mass = 3.86e6
            length = 110.0
            X_udot_coeff = -0.05
            Y_vdot_coeff = -0.5
            N_rdot_coeff = -0.05
            X_u_coeff = -0.002
            Y_v_coeff = -0.02
            N_r_coeff = -0.001
            X_uu_coeff = -0.0005
            Y_vv_coeff = -0.005
            N_rr_coeff = -0.0005
            Y_uv_coeff = -0.005
            N_uv_coeff = -0.0005
            actuator_envelope = dict(default_actuator_envelope)

        self.specifications = ShipSpecifications(
            length=length,
            mass=mass,
            X_udot_coeff=X_udot_coeff,
            Y_vdot_coeff=Y_vdot_coeff,
            N_rdot_coeff=N_rdot_coeff,
            X_u_coeff=X_u_coeff,
            Y_v_coeff=Y_v_coeff,
            N_r_coeff=N_r_coeff,
            X_uu_coeff=X_uu_coeff,
            Y_vv_coeff=Y_vv_coeff,
            N_rr_coeff=N_rr_coeff,
            Y_uv_coeff=Y_uv_coeff,
            N_uv_coeff=N_uv_coeff,
            **actuator_envelope,
        )

def create_ship(ship_model: ShipModel = ShipModel.MYZAKO_DEFAULT) -> MyzakoShip:
    """
    Factory function to create a MyzakoShip instance.

    Args:
        ship_model (ShipModel, optional): Ship model to use. Defaults to MYZAKO_DEFAULT.

    Returns:
        MyzakoShip: A new ship object with default specifications
        suitable for use in simulators and experiments.

    Example:
        >>> ship = create_ship(ShipModel.MYZAKO_DEFAULT)
        >>> ship.specifications.length
        110.0
    """

    if not isinstance(ship_model, ShipModel):
        raise ValueError(f"Ship model '{ship_model}' not supported yet.")
    return MyzakoShip(ship_model)
