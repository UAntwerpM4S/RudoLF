import copy
import logging
import numpy as np

from fw.ships.ship_factory import ShipModel
from fw.ships.vessel_state import VesselState
from fw.ships.actuator_state import ActuatorState
from fw.ships.ship_specs import ShipSpecifications
from fw.simulators.base_simulator import BaseSimulator
from fw.dynamics.fh_tabular_model import fh_rudder_sign

logger = logging.getLogger(__name__)


class FhSimWrapper(BaseSimulator):
    """
    BaseSimulator wrapper around the FH Simulator (deprecated; use ShipSimulator).

    Conventions handled here:
        - Rudder sign: framework +1 turns to port for every ship. Tripoli azipod
          setDesiredDirection(+deg) also turns to port (sent as is); Myzako rudder
          setControlValue(+) turns to starboard (negated). Both via fh_rudder_sign(),
          measured with check_rudder_sign.py.
        - Units: max_rudder_angle is in radians, getMaximumAngle() in degrees.
        - Velocities: getShipVelocityOverGround() is already body-fixed despite
          its name (as found for FhTabularModel), so it is used as (u, v) without
          rotation; its y is positive to port, like the framework.
    """

    def __init__(self, specs: ShipSpecifications, fh_engine, dt: float):
        """
        Args:
            specs: Vessel and actuator limits.
            fh_engine: FhSimBridge instance.
            dt: Time step [s], must be positive.

        Raises:
            ValueError: If dt <= 0 (from BaseSimulator).
        """

        logger.warning(
            "FhSimWrapper is deprecated and will be removed in a future version; "
            "use ShipSimulator instead."
        )

        super().__init__(specs, dt)

        self._engine = fh_engine


    def __deepcopy__(self, memo):
        """
        Deep copy that shares _engine (non-picklable native handles).
        """

        new_instance = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            if key == '_engine':
                setattr(new_instance, key, value)
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))

        return new_instance


    def step(self, action: np.ndarray, enable_smoothing: bool = False, enable_rudder_angle_scaling: bool = False) -> None:
        """
        Send a normalized action to FH, advance dt seconds and read back the state.

        The action is sent without smoothing or rate limiting: FH's own actuator
        model handles the physical rate limits.

        Args:
            action: [rudder, thrust] in [-1, 1].
            enable_smoothing: Currently ignored.
            enable_rudder_angle_scaling: Myzako only. If True, scale the rudder
                command by framework_max_deg / getMaximumAngle(); if False, send
                the policy output unscaled (correct when both share the same
                limit). The Tripoli azipod direction is always the policy output
                times the framework maximum (as in FhTabularModel).
        """

        # Map thrust from [-1, 1] to [min_thrust, max_thrust]
        thrust = (self._specs.min_thrust + 0.5 * (action[1] + 1.0) *
                  (self._specs.max_thrust - self._specs.min_thrust))
        actuator = ActuatorState(action[0], thrust)

        sign = fh_rudder_sign(self._engine.ship_model)   # framework to FH command sign

        if self._engine.ship_model == ShipModel.TRIPOLI:
            # [-1, 1] spans the ship's own rudder limit (35 deg for the Tripoli),
            # as FhTabularModel sends to the azipods.
            framework_max_deg = np.rad2deg(self._specs.max_rudder_angle)
            for rudder in self._engine.ship_interface.getAzipodControls():
                rudder.setDesiredDirection(float(sign * actuator.rudder_angle * framework_max_deg))
                rudder.setEngineLeverValue(float(actuator.thrust))
        else:
            # Rudder (policy output in [-1, 1])
            for rudder in self._engine.ship_interface.getRudderControls():
                if enable_rudder_angle_scaling:
                    fh_max_deg = rudder.getMaximumAngle()   # [deg]
                    framework_max_deg = np.rad2deg(self._specs.max_rudder_angle)
                    val = (actuator.rudder_angle * framework_max_deg) / fh_max_deg
                else:
                    val = actuator.rudder_angle
                rudder.setControlValue(float(sign * val))   # Myzako: sign = -1

            # Propeller
            for propeller in self._engine.ship_interface.getPropellerControls():
                propeller.engage()
                propeller.setEngineLeverValue(float(actuator.thrust))

        self._engine.math_model.simulateSeconds(self.dt)

        new_ship_pos          = self._engine.ship_interface.getShipPosition()
        velocity_over_ground  = self._engine.ship_interface.getShipVelocityOverGround()
        heading_deg           = self._engine.ship_interface.getShipHeading()
        yaw_rate_deg_per_s    = self._engine.ship_interface.getShipYawRate()

        heading_rad = float(np.radians(heading_deg))
        yaw_rate    = float(np.radians(yaw_rate_deg_per_s))

        self._state = VesselState(
            terminated=False,
            truncated=False,
            x=new_ship_pos.x,
            y=new_ship_pos.y,
            heading=(heading_rad + np.pi) % (2.0 * np.pi) - np.pi,
            u=velocity_over_ground.x,
            v=velocity_over_ground.y,   # port positive, as the framework
            r=yaw_rate,
        )
