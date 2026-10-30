import numpy as np

from typing import Tuple, Optional
from fw.ships.ship_factory import ShipModel
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase


def fh_rudder_sign(ship_model: ShipModel) -> float:
    """
    Factor from the framework's rudder sign to FH's command sign.

    Framework convention (every ship, every dynamics model): a positive rudder
    command turns the ship to PORT. FH, measured with check_rudder_sign.py
    (2026-10-19):
      - azipod (Tripoli), setDesiredDirection(+deg): the ship turns to port  -> +1
      - classic rudder (Myzako), setControlValue(+):  the ship turns to starboard -> -1
    Used by FhTabularModel, FhSimWrapper, FhSession and the Myzako dataset
    generators, so the conversion lives in one place.

    Raises:
        ValueError: For a ship without a known sign: measure it first (see
            CONVENTIONS.md, "Adding a ship") and add it to _FH_RUDDER_SIGNS.
    """

    try:
        return _FH_RUDDER_SIGNS[ship_model]
    except KeyError:
        raise ValueError(
            f"No FH rudder sign known for {ship_model!r}; measure it with check_rudder_sign.py "
            f"and add it to fh_tabular_model._FH_RUDDER_SIGNS."
        ) from None


# Measured for TRIPOLI and MYZAKO_37; the other Myzako presets use the same
# FH rudder control (setControlValue) and are assumed to share its sign.
_FH_RUDDER_SIGNS = {
    ShipModel.TRIPOLI: 1.0,
    ShipModel.MYZAKO_DEFAULT: -1.0,
    ShipModel.MYZAKO_29: -1.0,
    ShipModel.MYZAKO_37: -1.0,
}


class FhTabularModel(DynamicsBase):
    """
    FH-engine-backed vessel dynamics model.

    Wraps the FH math model (FastShip / FhSimBridge engine) as a DynamicsBase
    implementation. On each call to step(), the FH engine is advanced by dt
    seconds and the resulting body-fixed velocities (u, v, r) are read back
    directly: no velocity-delta differencing and no re-integration.

    This avoids the numerical drift that arises when accelerations are
    reconstructed by finite-differencing velocity snapshots and then
    re-integrated by ShipSimulator.integrate().

    Coordinate convention:
        u  =  velocity_over_ground.x          (surge, forward positive)
        v  = -velocity_over_ground.y          (sway,  starboard positive)
        r  =  radians(getShipYawRate())       (yaw rate, bow-to-starboard positive)

    Note: getShipVelocityOverGround() returns velocities already expressed
    in the body-fixed frame despite its name (confirmed empirically: the
    x-component aligns with surge regardless of heading).

    Wind and current come from the FH exercise itself, so ShipSimulator must
    not add EnvironmentModel's accelerations (models_own_wind_current = True).
    """

    # FH simulates the exercise's own wind and current.
    models_own_wind_current: bool = True

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
            rudder_angle: Rudder command [rad]; positive = the ship turns to
                          PORT (the framework convention, see fh_rudder_sign),
                          NOT a normalized [-1, 1] value. Converted to FH's
                          sign with fh_rudder_sign(ship_model).
                          - Conventional rudder ships: converted internally to
                            the FH engine's normalized [-1, 1] range via
                            np.degrees(rudder_angle) / fh_max_deg.
                          - TRIPOLI (azipod-controlled): converted to plain
                            degrees via np.degrees(rudder_angle) and passed
                            directly to setDesiredDirection(), which expects
                            an absolute angle in degrees (not normalized).
            thrust:       Normalized thrust command (dimensionless) in [-1, 1],
                          passed directly to setEngineLeverValue().
            dt:           Simulation step size in seconds. Passed directly to
                          simulateSeconds().
            enable_rudder_angle_scaling: When True (conventional-rudder ships
                          only), scales the rudder command using this
                          framework's max_rudder_angle instead of the FH
                          rudder object's own getMaximumAngle(), to correct
                          for a mismatch between the two. Ignored for TRIPOLI.

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
                # rudder_angle is in radians; convert to degrees (FH sign +1 for azipods).
                val_deg = fh_rudder_sign(self._engine.ship_model) * np.degrees(rudder_angle)
                rudder.setDesiredDirection(float(val_deg))
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
                # FH's positive rudder control turns the ship to starboard: convert.
                rudder.setControlValue(float(fh_rudder_sign(self._engine.ship_model) * val))

            for propeller in self._engine.ship_interface.getPropellerControls():
                propeller.engage()
                propeller.setEngineLeverValue(float(thrust))

        # --- Advance FH physics ---
        self._engine.math_model.simulateSeconds(dt)

        # --- Read updated state directly from FH engine ---
        vel = self._engine.ship_interface.getShipVelocityOverGround()
        yaw_rate = np.radians(self._engine.ship_interface.getShipYawRate())

        return float(vel.x), float(-vel.y), float(yaw_rate)
