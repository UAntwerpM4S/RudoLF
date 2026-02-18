"""CosysAirSim bridge.

The UE5 water physics differ from the Fossen model used in matplotlib.
So instead of mapping RL *inputs* 1-to-1, this bridge maps *outputs*:

- **Speed**: matched to the Fossen simulation's surge speed.
- **Heading**: a P-controller steers the AirSim vehicle to track the
  Fossen-computed heading each step. Because heading + speed together
  define the trajectory, the AirSim vehicle should follow the same
  route as the matplotlib ship.

How it works:
    Each step:
        1. Read the ship's heading from ``VesselState.heading`` (Fossen output).
        2. Convert to AirSim yaw frame.
        3. Read the AirSim vehicle's actual yaw from its pose quaternion.
        4. Compute heading error and steer proportionally.

Coordinate frames:
    (source: Fossen / Unreal Engine water physics calculation)

    - Ship:   x = east,  y = north, heading from east (CCW positive).
    - AirSim: x = north, y = east (NED), yaw from north (CW positive).
    - Transform: ``AirSim_yaw = π/2 - ship_heading``.

Example:
    >>> bridge = AirSimBridge()
    >>> bridge.send_action(
    ...     action,
    ...     ship_speed=env.phys_sim.state.u,
    ...     ship_heading=env.phys_sim.state.heading,
    ... )
    >>> bridge.disconnect()
"""

import math
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _normalize_angle(a: float) -> float:
    """Wrap an angle to the interval [-π, π].

    Args:
        a: Angle in radians.

    Returns:
        The equivalent angle wrapped into [-π, π].
    """
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quaternion(q) -> float:
    """Extract the yaw angle from an AirSim quaternion.

    Args:
        q: An AirSim ``Quaternionr`` object with attributes
            ``w_val``, ``x_val``, ``y_val``, and ``z_val``.

    Returns:
        The yaw angle in radians.
    """
    return math.atan2(
        2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
        1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2),
    )


class AirSimBridge:
    """Heading-tracking adapter for CosysAirSim.

    Maps Fossen simulation outputs (heading and speed) to AirSim vehicle
    controls using PID controllers for both steering and throttle.

    Attributes:
        _connected: Whether the bridge has an active AirSim connection.
    """

    def __init__(self, ip: str = "", port: int = 41451,
                 heading_kp: float = 5.0, heading_ki: float = 0.5,
                 heading_kd: float = 1.0, heading_ff: float = 0.5,
                 speed_scale: float = 1.0, lat_gain: float = 0.5,
                 throttle_kp: float = 1.0, throttle_ki: float = 0.05,
                 dist_gain: float = 0.1, dist_ki: float = 0.0,
                 log_telemetry: bool = True):
        """Initialise the AirSim bridge and connect to the simulator.

        Args:
            ip: AirSim server address. Defaults to ``""`` (localhost).
            port: AirSim RPC port. Defaults to ``41451``.
            heading_kp: Proportional gain for heading PID. Higher values
                give a faster response to turns. Defaults to ``5.0``.
            heading_ki: Integral gain for heading PID. Eliminates
                persistent heading drift. Defaults to ``0.5``.
            heading_kd: Derivative gain for heading PID. Defaults to
                ``1.0``.
            heading_ff: Feedforward gain. Steers based on the ship's
                yaw rate to anticipate turns. Defaults to ``0.5``.
            speed_scale: Multiplier applied to the ship speed (e.g.
                when units differ). Defaults to ``1.0``.
            lat_gain: Gain for cross-track (lateral) error correction.
                Defaults to ``0.5``.
            throttle_kp: Proportional gain for the throttle PI
                controller. Defaults to ``1.0``.
            throttle_ki: Integral gain for the throttle PI controller.
                Defaults to ``0.05``.
            dist_gain: Proportional gain for cumulative-distance error
                correction. Defaults to ``0.1``.
            dist_ki: Integral gain for cumulative-distance error
                correction. Defaults to ``0.0``.
            log_telemetry: If ``True``, writes per-step debug data to
                ``airsim_telemetry.csv``. Defaults to ``True``.
        """
        self._connected = False
        self._client = None
        self._controls = None
        self._heading_kp = heading_kp
        self._heading_ki = heading_ki
        self._heading_kd = heading_kd # Derivative term
        self._heading_ff = heading_ff
        self._lat_gain = lat_gain     # Lateral error gain
        self._speed_scale = speed_scale
        self._throttle_kp = throttle_kp
        self._throttle_ki = throttle_ki
        self._dist_gain = dist_gain
        self._dist_ki = dist_ki
        self._heading_integral = 0.0  # accumulated heading error
        self._throttle_integral = 0.0 # accumulated speed error
        self._dist_integral = 0.0     # accumulated distance error
        self._heading_offset = None   # calibrated on first step
        
        # Origin recording for absolute distance tracking
        self._origin_ship_pos = None  # (x, y)
        self._origin_car_pos = None   # (x, y)
        # Cumulative Path Integration variables
        self._last_ship_pos = None    # (x, y) previous step
        self._last_car_pos = None     # (x, y) previous step
        self._total_ship_dist = 0.0
        self._total_car_dist = 0.0
        
        self._step_count = 0

        # Telemetry logging
        self._log_telemetry = log_telemetry
        self._csv_file = None
        self._csv_writer = None

        try:
            import cosysairsim as airsim

            self._airsim = airsim
            self._client = airsim.CarClient(ip=ip, port=port)
            self._client.confirmConnection()
            self._client.enableApiControl(True)
            self._controls = airsim.CarControls()
            self._connected = True
            logger.info("Connected to Cosys-AirSim at %s:%d", ip, port)
            print(f"Connected to Cosys-AirSim at {ip}:{port}!")
            print(f"  Mode: PI+FF heading (Kp={heading_kp}, Ki={heading_ki}) | PI throttle (Kp={throttle_kp}, Ki={throttle_ki})")
            print(f"  Speed scale: {speed_scale}")

            if self._log_telemetry:
                import csv
                self._csv_file = open("airsim_telemetry.csv", "w", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow([
                    "step", "ship_speed", "car_speed", "target_speed",
                    "dist_error", "heading_error_deg", "ship_heading_deg", "car_yaw_deg",
                    "steer", "throttle", "ship_yaw_rate"
                ])
                print("  Telemetry logging to 'airsim_telemetry.csv'")

        except ImportError:
            logger.warning("cosysairsim package not installed — bridge disabled.")
            print("Warning: cosysairsim not installed. AirSim bridge disabled.")

        except Exception as exc:
            logger.warning("Could not connect to AirSim at %s:%d: %s", ip, port, exc)
            print(f"Warning: Could not connect to AirSim at {ip}:{port}: {exc}")
            print("AirSim bridge disabled — matplotlib visualization will still run.")


    @property
    def is_connected(self) -> bool:
        """Whether the bridge currently has an active AirSim connection."""
        return self._connected


    def send_action(
        self,
        action: np.ndarray,
        ship_speed: float = 0.0,
        ship_heading: float = 0.0,
        ship_yaw_rate: float = 0.0,
        ship_state=None,
    ) -> None:
        """Steer the AirSim vehicle to match the Fossen ship's heading and speed.

        Computes steering via a PID + feedforward + cross-track controller
        and throttle via a PI controller with distance-error correction,
        then sends the resulting ``CarControls`` to AirSim.

        Args:
            action: ``[rudder, thrust]`` array. Kept for interface
                compatibility; not used directly for steering.
            ship_speed: Ship's current surge speed from Fossen in m/s.
                Defaults to ``0.0``.
            ship_heading: Ship's current heading from Fossen in
                radians (measured from east, CCW positive).
                Defaults to ``0.0``.
            ship_yaw_rate: Ship's current yaw rate in rad/s.
                Defaults to ``0.0``.
            ship_state: Optional ``VesselState`` object providing
                ``x`` and ``y`` position attributes for cross-track
                and distance-error calculations.
        """
        if not self._connected:
            return

        try:
            self._step_count += 1

            # ── Read AirSim state ──
            car_state = self._client.getCarState()
            airsim_speed = car_state.speed

            pose = self._client.simGetVehiclePose()
            airsim_yaw = _yaw_from_quaternion(pose.orientation)

            # ── STEERING: track Fossen heading ──
            # Convert ship heading (from east) to AirSim yaw (from north)
            target_yaw = math.pi / 2.0 - ship_heading

            # Calibrate offset on first step (so initial mismatch = 0)
            if self._heading_offset is None:
                self._heading_offset = _normalize_angle(target_yaw - airsim_yaw)
                print(f"  Heading offset calibrated: {math.degrees(self._heading_offset):+.1f}°")

            # Heading error minus initial offset
            heading_error = _normalize_angle(target_yaw - airsim_yaw - self._heading_offset)

            # PI-controller
            # Accumulate integral (with anti-windup clamp)
            self._heading_integral += heading_error
            self._heading_integral = float(np.clip(
                self._heading_integral, -0.5 / max(self._heading_ki, 1e-9),
                                          0.5 / max(self._heading_ki, 1e-9),
            ))

            # Calculation of Derivative (D) term
            if not hasattr(self, '_prev_heading_error'): self._prev_heading_error = heading_error
            heading_derivative = heading_error - self._prev_heading_error
            self._prev_heading_error = heading_error
            
            # Calculation of Cross-Track Error (Lateral Error)
            # Use RELATIVE positions from start to align coordinate frames
            if self._origin_car_pos is None or self._origin_ship_pos is None:
                dx, dy = 0.0, 0.0 # Should not happen if initialized correctly above
            else:
                # Car displacement from its start
                car_rel_x = pose.position.x_val - self._origin_car_pos[0]
                car_rel_y = pose.position.y_val - self._origin_car_pos[1]
                
                # Ship displacement from its start
                ship_rel_x = float(ship_state.x) - self._origin_ship_pos[0]
                ship_rel_y = float(ship_state.y) - self._origin_ship_pos[1]
                
                # Error vector in the aligned relative frame
                dx = car_rel_x - ship_rel_x
                dy = car_rel_y - ship_rel_y
            
            path_yaw = target_yaw
            # Normal vector to the right
            nx = -math.sin(path_yaw)
            ny = math.cos(path_yaw)
            
            # Use the ALIGNED relative displacement vector (dx, dy) calculated above
            cross_track_error = dx * nx + dy * ny
            
            steering_lat = -1.0 * self._lat_gain * cross_track_error

            # Combine Control Outputs
            steering =  (self._heading_kp * heading_error +    # P-term
                         self._heading_ki * self._heading_integral + # I-term
                         self._heading_kd * heading_derivative + # D-term
                         steering_lat)                         # Lateral Correction
            
            steering -= self._heading_ff * ship_yaw_rate # FF-term (subtracted?)
            
            # DEBUG: Print components to identify saturation
            """
            if self._step_count % 50 == 0:
                print(f"[SteerDebug] P: {self._heading_kp * heading_error:.3f} | "
                      f"I: {self._heading_ki * self._heading_integral:.3f} | "
                      f"D: {self._heading_kd * heading_derivative:.3f} | "
                      f"Lat(XTE): {steering_lat:.3f} (err={cross_track_error:.3f}) | "
                      f"FF: {-self._heading_ff * ship_yaw_rate:.3f}")
            """

            steering = float(np.clip(steering, -1.0, 1.0))
            self._controls.steering = steering

            # ── THROTTLE: track absolute distance ──
            # Use cumulative distance (path integral) properly
            if self._origin_ship_pos is None and ship_state is not None:
                self._origin_ship_pos = (float(ship_state.x), float(ship_state.y))
                self._last_ship_pos = (float(ship_state.x), float(ship_state.y))
                self._last_car_pos = (pose.position.x_val, pose.position.y_val)
                self._total_ship_dist = 0.0
                self._total_car_dist = 0.0
                print(f"  Origin captured for distance tracking.")

            dist_error = 0.0
            if self._origin_ship_pos is not None and ship_state is not None:
                # Integrate Ship Distance
                curr_ship_x = float(ship_state.x)
                curr_ship_y = float(ship_state.y)
                ds_s = math.sqrt((curr_ship_x - self._last_ship_pos[0])**2 + 
                                 (curr_ship_y - self._last_ship_pos[1])**2)
                self._total_ship_dist += ds_s
                self._last_ship_pos = (curr_ship_x, curr_ship_y)

                # Integrate Car Distance
                curr_car_x = pose.position.x_val
                curr_car_y = pose.position.y_val
                ds_c = math.sqrt((curr_car_x - self._last_car_pos[0])**2 + 
                                 (curr_car_y - self._last_car_pos[1])**2)
                self._total_car_dist += ds_c
                self._last_car_pos = (curr_car_x, curr_car_y)

                # Calculate Error: Positive = car is behind
                dist_error = (self._total_ship_dist * self._speed_scale) - self._total_car_dist

            # Target speed = ship_speed + PI correction for distance
            # Integral term absorbs the drift caused by "slow python loop" vs "real-time airsim"
            self._dist_integral += dist_error
            # Clamp integral loosely to prevent runaway windup
            self._dist_integral = float(np.clip(self._dist_integral, -50.0, 50.0))
            
            scaled_ship_speed = ship_speed * self._speed_scale
            target_speed = (scaled_ship_speed 
                            + self._dist_gain * dist_error
                            + self._dist_ki * self._dist_integral)
            
            # Cornering logic: slow down if heading error is large
            # If error > 10 degrees, reduce target speed
            abs_head_err = abs(heading_error)
            if abs_head_err > math.radians(10.0):
                 # Reduce speed proportional to error severity
                 slow_down = (abs_head_err - math.radians(10.0)) * 3.0
                 target_speed -= slow_down

            # Ensure target speed doesn't go below 0 unless really needed
            target_speed = max(0.0, target_speed)

            # PI-controller for throttle
            # Error = target - actual
            speed_error = target_speed - airsim_speed
            
            # Anti-windup: stop integrating if throttle is saturated and error is same sign
            self._throttle_integral += speed_error
            # Relaxed clamp to allow full throttle range from I-term
            # If throttle_ki is 0.1, integral needs to reach ~10.0 to give 1.0 throttle
            # So clamp to +/- 20.0 to be safe
            self._throttle_integral = float(np.clip(
                self._throttle_integral, -20.0, 20.0
            ))

            throttle_cmd = (self._throttle_kp * speed_error + 
                            self._throttle_ki * self._throttle_integral)
            
            # If target is 0, force 0 (unless we need to brake)
            if target_speed < 0.1 and airsim_speed < 0.1:
                throttle_cmd = 0.0
                self._throttle_integral = 0.0

            self._controls.throttle = float(np.clip(throttle_cmd, 0.0, 1.0))
            
            # Simple braking if overshooting significantly
            if airsim_speed > target_speed + 1.0:
                 self._controls.brake = 0.5
            else:
                 self._controls.brake = 0.0

            # No braking — PPO never brakes
            self._controls.brake = 0.0
            self._controls.is_manual_gear = True
            self._controls.manual_gear = 1

            self._client.setCarControls(self._controls)

            # Log to CSV
            if self._csv_writer:
                self._csv_writer.writerow([
                    self._step_count,
                    f"{ship_speed:.4f}", f"{airsim_speed:.4f}", f"{target_speed:.4f}",
                    f"{dist_error:.4f}",
                    f"{math.degrees(heading_error):.4f}",
                    f"{math.degrees(ship_heading):.4f}",
                    f"{math.degrees(airsim_yaw):.4f}",
                    f"{steering:.4f}", f"{self._controls.throttle:.4f}",
                    f"{ship_yaw_rate:.4f}"
                ])
                self._csv_file.flush()

            # Telemetry every 200 steps
            if self._step_count % 200 == 1:
                print(
                    f"  [AirSim] step={self._step_count:>5} | "
                    f"ship_spd={ship_speed:>5.2f} | "
                    f"car_spd={airsim_speed:>5.2f} | "
                    f"dist_err={dist_error:>+5.1f}m | "
                    f"head_err={math.degrees(heading_error):>+5.1f}° | "
                    f"steer={steering:>+.3f} | "
                    f"thr={self._controls.throttle:.3f}"
                )

        except Exception as exc:
            logger.warning("AirSim send_action failed: %s", exc)

    def get_state(self) -> dict:
        """Retrieve the current vehicle state from AirSim.

        Returns:
            A dictionary with keys ``"speed"``, ``"gear"``, and
            ``"timestamp"``, or an empty dictionary if the bridge
            is not connected or the query fails.
        """
        if not self._connected:
            return {}
        try:
            s = self._client.getCarState()
            return {"speed": s.speed, "gear": s.gear, "timestamp": s.timestamp}
        except Exception as exc:
            logger.warning("AirSim get_state failed: %s", exc)
            return {}

    def reset(self) -> None:
        """Reset the AirSim vehicle and all internal controller state.

        Re-enables API control, zeroes PID integrals, and clears the
        heading-offset calibration so it is recalculated on the next
        ``send_action`` call.
        """
        if not self._connected:
            return
        try:
            self._client.reset()
            time.sleep(0.5)
            self._client.enableApiControl(True)
            self._controls = self._airsim.CarControls()
            self._heading_offset = None  # recalibrate on next step
            self._heading_offset = None  # recalibrate on next step
            self._heading_integral = 0.0
            self._throttle_integral = 0.0
            self._dist_integral = 0.0
            self._origin_ship_pos = None
            self._origin_car_pos = None
            self._step_count = 0
            print("\n--- AirSim vehicle reset ---")
        except Exception as exc:
            logger.warning("AirSim reset failed: %s", exc)

    def disconnect(self) -> None:
        """Gracefully disconnect from the AirSim simulator.

        Stops the vehicle by zeroing throttle, steering, and brake,
        closes any open telemetry CSV file, and releases API control.
        """
        if not self._connected:
            return
        try:
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
                print("  Telemetry log closed.")

            self._controls.throttle = 0.0
            self._controls.steering = 0.0
            self._controls.brake = 0.0
            self._client.setCarControls(self._controls)
            time.sleep(0.3)
            self._client.enableApiControl(False)
            self._connected = False
            print(f"\nAirSim released after {self._step_count} steps.")
        except Exception as exc:
            logger.warning("AirSim disconnect failed: %s", exc)
            self._connected = False
