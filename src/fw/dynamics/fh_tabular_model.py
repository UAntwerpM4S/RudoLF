import numpy as np

from typing import Tuple, Optional
from fw.ships.ship_factory import ShipModel, require_fh_supported
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.dynamics.dynamics_model import DynamicsBase
from fw.simulators.fh_interface_utils import fh_heading_to_env_rad, fh_wind_current_state
from fw.surrogate.schema import env_body_components


def fh_rudder_sign(ship_model: ShipModel) -> float:
    """
    Factor from the framework's rudder sign (positive = turn to port) to FH's
    command sign, as measured with check_rudder_sign.py:
      - azipod (Tripoli), setDesiredDirection(+deg) turns to port: +1
      - classic rudder (Myzako), setControlValue(+) turns to starboard: -1
    Shared by FhTabularModel, FhSimWrapper, FhSession and the Myzako dataset generators.

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
    DynamicsBase wrapper around the FH math model (FastShip / FhSimBridge engine).

    step() advances the FH engine by dt and reads the body-fixed velocities
    (u, v, r) back directly, avoiding the drift of finite-differencing velocities
    and re-integrating them.

    Coordinate convention:
        u  =  velocity_over_ground.x          (surge, forward positive)
        v  =  velocity_over_ground.y          (sway, port positive -- FH's body
                                               frame is the framework's, see
                                               fw.ships.vessel_state)
        r  =  radians(getShipYawRate())       (yaw rate, counter-clockwise =
                                               bow-to-port positive)

    Despite its name, getShipVelocityOverGround() returns body-fixed velocities
    (checked empirically: x aligns with surge regardless of heading).

    FH simulates the exercise's own wind and current (models_own_wind_current);
    observed_wind_current reads them from FH as the dataset generator does.
    """

    models_own_wind_current: bool = True

    def __init__(self, ship_spec: ShipSpecifications, fh_engine):
        """
        Initialize the FH tabular dynamics model.

        Args:
            ship_spec:  Vessel specifications (max_rudder_angle for rudder scaling).
            fh_engine:  Initialized FH engine exposing ship_interface and math_model.
        """

        # getattr: also covers engines other than FhSimBridge (e.g. SyncSimBridge).
        engine_ship_model = getattr(fh_engine, "ship_model", None)
        if engine_ship_model is not None:
            require_fh_supported(engine_ship_model, "FhTabularModel")

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
            state:        Unused (FH keeps its own state); kept for the interface.
            rudder_angle: Rudder command [rad], positive = turn to port (not
                          normalized); sign converted with fh_rudder_sign().
                          Rudder ships: normalized to [-1, 1] by the max angle.
                          TRIPOLI (azipods): passed in degrees to setDesiredDirection().
            thrust:       Normalized thrust in [-1, 1], passed to setEngineLeverValue().
            dt:           Step size [s], passed to simulateSeconds().
            enable_rudder_angle_scaling: Rudder ships only: normalize by the
                          framework's max_rudder_angle instead of the FH rudder's
                          getMaximumAngle() (corrects a mismatch). Ignored for TRIPOLI.

        Returns:
            (u [m/s], v [m/s], r [rad/s]) read from the FH engine after the step.
        """

        # --- Set controls ---
        if self._engine.ship_model == ShipModel.TRIPOLI:
            for rudder in self._engine.ship_interface.getAzipodControls():
                # FH sign is +1 for azipods.
                val_deg = fh_rudder_sign(self._engine.ship_model) * np.degrees(rudder_angle)
                rudder.setDesiredDirection(float(val_deg))
                rudder.setEngineLeverValue(float(thrust))
        else:
            for rudder in self._engine.ship_interface.getRudderControls():
                if enable_rudder_angle_scaling:
                    framework_max_deg = np.rad2deg(self.ship_spec.max_rudder_angle)
                    val = np.degrees(rudder_angle) / framework_max_deg
                else:
                    val = np.degrees(rudder_angle) / rudder.getMaximumAngle()
                rudder.setControlValue(float(fh_rudder_sign(self._engine.ship_model) * val))

            for propeller in self._engine.ship_interface.getPropellerControls():
                propeller.engage()
                propeller.setEngineLeverValue(float(thrust))

        # --- Advance FH physics ---
        self._engine.math_model.simulateSeconds(dt)

        # --- Read updated state directly from FH engine ---
        vel = self._engine.ship_interface.getShipVelocityOverGround()
        yaw_rate = np.radians(self._engine.ship_interface.getShipYawRate())

        return float(vel.x), float(vel.y), float(yaw_rate)

    def read_state(self) -> Tuple[float, float, float, float, float, float]:
        """
        FH's ship state in framework conventions (the single place it is read
        for ShipSimulator.step and PySimEnv's episode start).

        Returns:
            (x [m], y [m], heading [rad, 0 = East, CCW], u [m/s], v [m/s, port
            positive], r [rad/s, CCW positive]).
        """

        ship_interface = self._engine.ship_interface
        pos = ship_interface.getShipPosition()
        vel = ship_interface.getShipVelocityOverGround()
        return (float(pos.x), float(pos.y), fh_heading_to_env_rad(ship_interface.getShipHeading()),
                float(vel.x), float(vel.y), float(np.radians(ship_interface.getShipYawRate())))

    def observed_wind_current(self, state: VesselState) -> np.ndarray:
        """
        FH's wind and current at its own ship (see DynamicsBase): WindState and
        over-ground minus through-water velocity, as in the dataset generator.

        Args:
            state: Unused (FH keeps its own state); kept for the interface.
        """

        ship_interface = self._engine.ship_interface
        heading = fh_heading_to_env_rad(ship_interface.getShipHeading())
        return np.array(env_body_components(*fh_wind_current_state(ship_interface), heading), dtype=np.float64)
