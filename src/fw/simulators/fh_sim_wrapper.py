import copy
import logging
import numpy as np

from fw.ships.ship_factory import ShipModel
from fw.ships.vessel_state import VesselState
from fw.ships.actuator_state import ActuatorState
from fw.ships.ship_specs import ShipSpecifications
from fw.simulators.base_simulator import BaseSimulator


class FhSimWrapper(BaseSimulator):
    """
    Wrapper for the FH Simulator integrating with the unified BaseSimulator interface.

    Handles all coordinate-frame and sign-convention conversions between the
    framework and the FH Simulator black box:

    Rudder sign convention:
        Framework:  -1 = turn right (starboard),  +1 = turn left (port)
        FH Sim:     -1 = turn left (port),        +1 = turn right (starboard)
        -> rudder command is negated before being sent to the FH engine.

    Rudder angle units:
        The framework stores max_rudder_angle in radians; getMaximumAngle()
        returns degrees. The conversion factor np.rad2deg(max_rudder_angle)
        normalizes the policy output [-1, 1] to the FH engine's degree scale.
        If the SimAPI ever changes getMaximumAngle() to return radians, this
        line must be updated.

    Velocity frame:
        getShipVelocityOverGround() returns velocities in the earth-fixed
        (NED/ENU) frame. VesselState expects body-fixed surge (u) and sway (v).
        The transformation is:
            u =  vx * cos(ψ) + vy * sin(ψ)
            v = -vx * sin(ψ) + vy * cos(ψ)
        where ψ is the vessel heading [rad] and (vx, vy) are earth-fixed
        velocity components. Without this rotation the observation vector
        contains earth-fixed velocities mislabeled as body-fixed, which
        becomes significant in the presence of wind or current.

    Sway sign:
        The FH engine returns y-velocity with positive = port (opposite to the
        body-fixed convention where positive = starboard), so vy is negated.
        This is consistent with the earth ->body rotation above.
    """

    def __init__(self, specs: ShipSpecifications, fh_engine, dt: float):
        """
        Initialize the FH Simulator wrapper.

        .. deprecated::
            FhSimWrapper is deprecated; use ShipSimulator instead.

        Args:
            specs: ShipSpecifications defining vessel and actuator limits.
            fh_engine: FhSimBridge instance (the FH engine handle).
            dt: Simulation time step [s]. Must be positive.

        Raises:
            ValueError: If dt <= 0 (enforced by BaseSimulator).
        """

        logging.warning(
            "FhSimWrapper is deprecated and will be removed in a future version; "
            "use ShipSimulator instead."
        )

        super().__init__(specs, dt)

        self._engine = fh_engine


    def __deepcopy__(self, memo):
        """
        Deep copy this wrapper, sharing the FH engine reference.

        The FH engine (_engine) holds non-picklable native library handles
        and is shared between copies. All other attributes are deep-copied.

        Args:
            memo: Memoization dict used by copy.deepcopy.

        Returns:
            FhSimWrapper: Deep copy with shared _engine reference.
        """

        new_instance = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            if key == '_engine':
                setattr(new_instance, key, value)   # Retain the original reference
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))  # Deep copy all other attributes

        return new_instance


    def step(self, action: np.ndarray, enable_smoothing: bool = False, enable_rudder_angle_scaling: bool = False) -> None:
        """
        Advance the FH Simulator by one time step using a normalized action.

        Workflow:
            1. Apply exponential smoothing to the raw policy action.
            2. Convert the smoothed action to FH engine control values and
               send them to the rudder and propeller interfaces.
            3. Advance the physics by dt seconds.
            4. Read the updated state from the FH engine.
            5. Transform earth-fixed velocities to body-fixed frame.
            6. Store the result as a VesselState.

        Args:
            action: Normalized action array [rudder, thrust] in [-1, 1].
            enable_smoothing: If True, apply exponential smoothing to the
                action before forwarding to the FH engine. If False, the
                raw action is sent directly (used during visualization to
                ensure full thrust).
            enable_rudder_angle_scaling: If True, rescale the rudder action
                to compensate for the difference between the framework's
                maximum rudder angle (from ShipSpecifications, used during
                training) and the FH engine's own rudder limit (from
                getMaximumAngle()). If False, the framework angle is used
                directly as the FH engine target, which is correct when both
                simulators share the same rudder limit.

        Note on actuator model:
            apply_smoothing() (EMA only) is used rather than ActuatorModel.step()
            (full rate-limiting + saturation) because the FH Simulator's own
            actuator model handles physical rate limits internally. Applying
            our rate limiter on top would double-constrain the actuator and
            produce overly conservative maneuvers. If the FH engine does not
            have internal actuator dynamics, switch to self._actuators.step().
        """

        # Step 1: smooth the policy action
        thrust = (self._specs.min_thrust + 0.5 * (action[1] + 1.0) *
                  (self._specs.max_thrust - self._specs.min_thrust))
        actuator = ActuatorState(action[0], thrust)

        if self._engine.ship_model == ShipModel.TRIPOLI:
            for rudder in self._engine.ship_interface.getAzipodControls():
                fh_max_deg = 90.0       # [deg] - FH engine limit
                rudder.setDesiredDirection(float(-actuator.rudder_angle*fh_max_deg))
                rudder.setEngineLeverValue(float(actuator.thrust))
        else:
            # Step 2a: rudder
            # actuator.rudder_angle is the (potentially smoothed) policy output in [-1, 1].
            for rudder in self._engine.ship_interface.getRudderControls():
                if enable_rudder_angle_scaling:
                    # Rescale the rudder angle relative to FH simulator's rudder angle range.
                    # Value mapping [-1, 1] -> [-max_deg, +max_deg] in the FH degree scale.
                    fh_max_deg = rudder.getMaximumAngle()   # FH engine's rudder limit in degrees
                    framework_max_deg = np.rad2deg(self._specs.max_rudder_angle)
                                                            # convert to degrees as max_rudder_angle
                                                            # is stored in radians in ShipSpecifications
                    val = (actuator.rudder_angle * framework_max_deg) / fh_max_deg
                else:
                    # No rescaling: treat the policy output as already in the FH range.
                    val = actuator.rudder_angle
                rudder.setControlValue(float(-val))         # negated: framework -> FH sign flip

            # Step 2b: propeller
            for propeller in self._engine.ship_interface.getPropellerControls():
                propeller.engage()
                propeller.setEngineLeverValue(float(actuator.thrust))

        # Step 3: advance FH physics
        self._engine.math_model.simulateSeconds(self.dt)

        # Step 4: read updated state from FH engine
        new_ship_pos          = self._engine.ship_interface.getShipPosition()
        velocity_over_ground  = self._engine.ship_interface.getShipVelocityOverGround()
        heading_deg           = self._engine.ship_interface.getShipHeading()
        yaw_rate_deg_per_s    = self._engine.ship_interface.getShipYawRate()

        heading_rad = float(np.radians(heading_deg))
        yaw_rate    = float(np.radians(yaw_rate_deg_per_s))

        # Step 5: store VesselState
        self._state = VesselState(
            terminated=False,
            truncated=False,
            x=new_ship_pos.x,
            y=new_ship_pos.y,
            heading=(heading_rad + np.pi) % (2.0 * np.pi) - np.pi,
            u=velocity_over_ground.x,
            v=-velocity_over_ground.y,
            r=yaw_rate,
        )
