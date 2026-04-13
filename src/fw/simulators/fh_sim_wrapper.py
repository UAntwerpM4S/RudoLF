import copy
import numpy as np

from fw.ships.vessel_state import VesselState
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

        Args:
            specs: ShipSpecifications defining vessel and actuator limits.
            fh_engine: FhSimBridge instance (the FH engine handle).
            dt: Simulation time step [s]. Must be positive.

        Raises:
            ValueError: If dt <= 0 (enforced by BaseSimulator).
        """

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


    def step(self, action: np.ndarray, enable_smoothing: bool = True) -> None:
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

        Note on actuator model:
            apply_smoothing() (EMA only) is used rather than ActuatorModel.step()
            (full rate-limiting + saturation) because the FH Simulator's own
            actuator model handles physical rate limits internally. Applying
            our rate limiter on top would double-constrain the actuator and
            produce overly conservative maneuvers. If the FH engine does not
            have internal actuator dynamics, switch to self._actuators.step().
        """

        # Step 1: smooth the policy action
        actuator = self._actuators.apply_smoothing(action, enable_smoothing)

        # Step 2a: rudder
        # actuator.rudder_angle is the smoothed policy output in [-1, 1].
        # getMaximumAngle() returns the FH engine's rudder limit in degrees.
        # max_rudder_angle is stored in radians in ShipSpecifications.
        # -> val maps [-1, 1] -> [-max_deg, +max_deg] in the FH degree scale.
        # -> negated to compensate for the opposite sign convention (see class docstring).
        for rudder in self._engine.ship_interface.getRudderControls():
            # NOTE: getMaximumAngle() is assumed to return degrees. If the SimAPI
            # changes this to radians, remove the np.rad2deg() conversion.
            fh_max_deg = rudder.getMaximumAngle()   # [deg] - FH engine limit
            framework_max_deg = np.rad2deg(self._specs.max_rudder_angle)  # [deg]
            val = (actuator.rudder_angle * framework_max_deg) / fh_max_deg
            rudder.setControlValue(float(-val))     # negated: framework -> FH sign flip

        # Step 2b: propeller
        for propeller in self._engine.ship_interface.getPropellerControls():
            # During visualization (enable_smoothing=False) use full thrust so
            # the ship maintains steerage way at the correct speed.
            thrust_cmd = float(actuator.thrust) if enable_smoothing else 1.0
            propeller.setEngineLeverValue(thrust_cmd)

        # Step 3: advance FH physics
        self._engine.math_model.simulateSeconds(self.dt)

        # Step 4: read updated state from FH engine
        new_ship_pos          = self._engine.ship_interface.getShipPosition()
        velocity_over_ground  = self._engine.ship_interface.getShipVelocityOverGround()
        heading_deg           = self._engine.ship_interface.getShipHeading()
        yaw_rate_deg_per_s    = self._engine.ship_interface.getShipYawRate()

        heading_rad = np.radians(heading_deg)
        yaw_rate    = np.radians(yaw_rate_deg_per_s)

        # Step 5: store VesselState
        self._state = VesselState(
            terminated=False,
            truncated=False,
            x=np.float32(new_ship_pos.x),
            y=np.float32(new_ship_pos.y),
            heading=np.float32((heading_rad + np.pi) % (2.0 * np.pi) - np.pi),
            u=np.float32(velocity_over_ground.x),
            v=np.float32(-velocity_over_ground.y),
            r=np.float32(yaw_rate),
        )
