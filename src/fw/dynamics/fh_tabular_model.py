import numpy as np

from typing import Tuple, Optional
from fw.ships.ship_factory import ShipModel
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase


class FhTabularModel(DynamicsBase):
    """
    FH-engine-backed vessel dynamics model.

    Wraps the FH math model (FastShip / FhSimBridge engine) as a DynamicsBase
    implementation. On each call to step(), the FH engine is advanced by dt
    seconds and the resulting body-fixed velocities (u, v, r) are read back
    directly — no velocity-delta differencing and no re-integration.

    This avoids the numerical drift that arises when accelerations are
    reconstructed by finite-differencing velocity snapshots and then
    re-integrated by ShipSimulator.integrate().

    Coordinate convention:
        u  =  velocity_over_ground.x          (surge, forward positive)
        v  = -velocity_over_ground.y          (sway,  starboard positive)
        r  =  radians(getShipYawRate())       (yaw rate, bow-to-starboard positive)

    Note: getShipVelocityOverGround() returns velocities already expressed
    in the body-fixed frame despite its name (confirmed empirically — the
    x-component aligns with surge regardless of heading).
    """

    def __init__(self, ship_spec: ShipSpecifications, fh_engine):
        """
        Initialize the FH tabular dynamics model.

        Args:
            ship_spec:  Vessel physical specifications, used to scale
                        rudder commands to the FH engine's range.
            fh_engine:  Initialized FH engine object exposing
                        ship_interface and math_model.
        """

        self.ship_spec: ShipSpecifications = ship_spec
        self._engine = fh_engine

    def step(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
        dt: float,
        enable_rudder_angle_scaling: Optional[bool] = False,
    ) -> Tuple[float, float, float]:
        """
        Advance the FH engine by dt seconds and return updated (u, v, r).

        Args:
            state:        Current vessel state (used for interface consistency;
                          the FH engine maintains its own internal state).
            rudder_angle: Rudder deflection angle [rad], positive to starboard.
                          Converted internally to the FH engine's normalized
                          [-1, 1] range via np.degrees(rudder_angle) / fh_max_deg.
            thrust:       Normalized thrust command (dimensionless) in [-1, 1],
                          passed directly to setEngineLeverValue().
            dt:           Simulation step size in seconds. Passed directly to
                          simulateSeconds().
            enable_rudder_angle_scaling: Enable rudder angle scaling.

        Returns:
            Tuple (u, v, r) read directly from the FH engine after advancing
            by dt seconds:
                u = velocity_over_ground.x          [m/s]
                v = -velocity_over_ground.y         [m/s]
                r = radians(getShipYawRate())       [rad/s]
        """

        # --- Set controls ---
        if self._engine.ship_model == ShipModel.TRIPOLI:
            for rudder in self._engine.ship_interface.getAzipodControls():
                # rudder_angle is normalized [-1, 1]; scale to azipod range [-90, 90] deg.
                val_deg = np.degrees(rudder_angle)
                rudder.setDesiredDirection(float(-val_deg))
                rudder.setEngineLeverValue(float(thrust))
        else:
            for rudder in self._engine.ship_interface.getRudderControls():
                if enable_rudder_angle_scaling:
                    framework_max_deg = np.rad2deg(self.ship_spec.max_rudder_angle)
                                                            # convert to degrees as max_rudder_angle
                                                            # is stored in radians in ShipSpecifications
                    val = np.degrees(rudder_angle) / framework_max_deg
                else:
                    val = np.degrees(rudder_angle) / rudder.getMaximumAngle()
                rudder.setControlValue(float(-val))   # negated: framework -> FH sign convention

            for propeller in self._engine.ship_interface.getPropellerControls():
                propeller.engage()
                propeller.setEngineLeverValue(float(thrust))

        # --- Advance FH physics ---
        self._engine.math_model.simulateSeconds(dt)

        # --- Read updated state directly from FH engine ---
        vel = self._engine.ship_interface.getShipVelocityOverGround()
        yaw_rate = np.radians(self._engine.ship_interface.getShipYawRate())

        return float(vel.x), float(-vel.y), float(yaw_rate)
