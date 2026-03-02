"""CosysAirSim bridge — heading-tracking adapter for UE5.

The UE5 water-surface physics differ from the Fossen mathematical model
used in the matplotlib simulator.  Rather than mapping RL *inputs*
1-to-1, this bridge maps *outputs*:

* **Speed** is matched to the Fossen simulation's surge speed via a PI
  controller with cumulative-distance correction.
* **Heading** is tracked by a PID + feedforward + cross-track-error
  controller that steers the AirSim vehicle to follow the
  Fossen-computed heading at every step.

Workflow per step:
    1. Read the ship's heading from ``VesselState.heading``.
    2. Convert to AirSim yaw frame (``AirSim_yaw = π/2 − heading``).
    3. Read the AirSim vehicle's actual yaw from its pose quaternion.
    4. Compute heading error and derive steering + throttle commands.

Coordinate frames:
    * **Ship (Fossen):** x = east, y = north, heading from east (CCW+).
    * **AirSim (NED):** x = north, y = east, yaw from north (CW+).
    * **Transform:** ``AirSim_yaw = π/2 − ship_heading``.

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
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Resolve the telemetry output directory.
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _normalize_angle(a: float) -> float:
    """Wrap an angle to the interval [−π, π].

    Args:
        a: Angle in radians.

    Returns:
        The equivalent angle in the range [−π, π].
    """
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quaternion(q) -> float:
    """Extract the yaw angle from an AirSim quaternion.

    Uses the standard ZYX Euler-angle decomposition to isolate
    the yaw component.

    Args:
        q: A ``cosysairsim.Quaternionr`` instance with attributes
            ``w_val``, ``x_val``, ``y_val``, and ``z_val``.

    Returns:
        Yaw angle in radians.
    """
    return math.atan2(
        2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
        1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2),
    )


class AirSimBridge:
    """Heading-tracking adapter between the Fossen sim and CosysAirSim.

    Maps Fossen simulation outputs (heading and speed) to AirSim vehicle
    ``CarControls`` using PID controllers for both steering and throttle.

    Attributes:
        _connected: ``True`` while the bridge holds an active AirSim
            connection; ``False`` otherwise (including when
            ``cosysairsim`` is not installed).
    """

    def __init__(self, ip: str = "", port: int = 41451,
                 heading_kp: float = 5.0, heading_ki: float = 0.5,
                 heading_kd: float = 1.0, heading_ff: float = 0.5,
                 speed_scale: float = 1.0, lat_gain: float = 0.5,
                 throttle_kp: float = 1.0, throttle_ki: float = 0.05,
                 dist_gain: float = 0.1, dist_ki: float = 0.0,
                 log_telemetry: bool = True):
        """Initialise the AirSim bridge and connect to the simulator.

        If ``cosysairsim`` is not installed or the connection fails, the
        bridge starts in a disconnected state and all control methods
        become no-ops.

        Args:
            ip: AirSim server IP address.  An empty string (the default)
                connects to ``localhost``.
            port: AirSim RPC port.
            heading_kp: Proportional gain for the heading PID controller.
            heading_ki: Integral gain for the heading PID controller.
            heading_kd: Derivative gain for the heading PID controller.
            heading_ff: Feedforward gain — steers proactively based on
                the ship's yaw rate to anticipate turns.
            speed_scale: Multiplicative factor applied to the Fossen
                ship speed before sending it to AirSim.
            lat_gain: Gain for cross-track (lateral) error correction.
            throttle_kp: Proportional gain for the throttle PI
                controller.
            throttle_ki: Integral gain for the throttle PI controller.
            dist_gain: Proportional gain for cumulative-distance error
                correction.
            dist_ki: Integral gain for cumulative-distance error
                correction.
            log_telemetry: If ``True``, writes per-step debug data to
                ``airsim_telemetry.csv`` inside ``airsim/config/``.
        """
        self._connected = False
        self._client = None
        self._controls = None
        self._heading_kp = heading_kp
        self._heading_ki = heading_ki
        self._heading_kd = heading_kd
        self._heading_ff = heading_ff
        self._lat_gain = lat_gain
        self._speed_scale = speed_scale
        self._throttle_kp = throttle_kp
        self._throttle_ki = throttle_ki
        self._dist_gain = dist_gain
        self._dist_ki = dist_ki
        self._heading_integral = 0.0
        self._throttle_integral = 0.0
        self._dist_integral = 0.0
        self._heading_offset = None
        self._prev_heading_error = 0.0

        # Origin recording for absolute distance tracking
        self._origin_ship_pos = None
        self._origin_car_pos = None
        # Cumulative Path Integration variables
        self._last_ship_pos = None
        self._last_car_pos = None
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
                _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                self._csv_file = open(_CONFIG_DIR / "airsim_telemetry.csv", "w", newline="")
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
        """Whether the bridge currently has an active AirSim connection.

        Returns:
            ``True`` if connected, ``False`` otherwise.
        """
        return self._connected

    def send_action(
        self,
        action: np.ndarray,
        ship_speed: float = 0.0,
        ship_heading: float = 0.0,
        ship_yaw_rate: float = 0.0,
        ship_state=None,
    ) -> None:
        """Steer the AirSim vehicle to match the Fossen ship.

        Computes steering via a PID + feedforward + cross-track-error
        controller and throttle via a PI controller with cumulative
        distance-error correction, then sends the resulting
        ``CarControls`` to AirSim.

        Args:
            action: ``[rudder, thrust]`` array from the RL agent.  Kept
                for interface compatibility; the bridge derives its own
                steering and throttle from the Fossen outputs.
            ship_speed: Surge speed (m/s) from the Fossen model.
            ship_heading: Heading (rad) from the Fossen model, measured
                from east, counter-clockwise positive.
            ship_yaw_rate: Yaw rate (rad/s) from the Fossen model.
            ship_state: Optional ``VesselState`` with ``.x`` and ``.y``
                attributes for cross-track and distance calculations.
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
            target_yaw = math.pi / 2.0 - ship_heading

            # Calibrate offset on first step
            if self._heading_offset is None:
                self._heading_offset = _normalize_angle(target_yaw - airsim_yaw)
                print(f"  Heading offset calibrated: {math.degrees(self._heading_offset):+.1f}°")

            # Heading error minus initial offset
            heading_error = _normalize_angle(target_yaw - airsim_yaw - self._heading_offset)

            # PI-controller — accumulate integral (with anti-windup clamp)
            self._heading_integral += heading_error
            self._heading_integral = float(np.clip(
                self._heading_integral, -0.5 / max(self._heading_ki, 1e-9),
                                          0.5 / max(self._heading_ki, 1e-9),
            ))

            # Derivative (D) term
            heading_derivative = heading_error - self._prev_heading_error
            self._prev_heading_error = heading_error

            # Cross-Track Error (Lateral Error)
            if self._origin_car_pos is None or self._origin_ship_pos is None:
                dx, dy = 0.0, 0.0
            else:
                car_rel_x = pose.position.x_val - self._origin_car_pos[0]
                car_rel_y = pose.position.y_val - self._origin_car_pos[1]
                ship_rel_x = float(ship_state.x) - self._origin_ship_pos[0]
                ship_rel_y = float(ship_state.y) - self._origin_ship_pos[1]
                dx = car_rel_x - ship_rel_x
                dy = car_rel_y - ship_rel_y

            path_yaw = target_yaw
            nx = -math.sin(path_yaw)
            ny = math.cos(path_yaw)
            cross_track_error = dx * nx + dy * ny

            steering_lat = -1.0 * self._lat_gain * cross_track_error

            # Combine Control Outputs
            steering = (self._heading_kp * heading_error +
                        self._heading_ki * self._heading_integral +
                        self._heading_kd * heading_derivative +
                        steering_lat)

            steering -= self._heading_ff * ship_yaw_rate

            steering = float(np.clip(steering, -1.0, 1.0))
            self._controls.steering = steering

            # ── THROTTLE: track absolute distance ──
            if self._origin_ship_pos is None and ship_state is not None:
                self._origin_ship_pos = (float(ship_state.x), float(ship_state.y))
                self._last_ship_pos = (float(ship_state.x), float(ship_state.y))
                self._last_car_pos = (pose.position.x_val, pose.position.y_val)
                self._total_ship_dist = 0.0
                self._total_car_dist = 0.0
                print("  Origin captured for distance tracking.")

            dist_error = 0.0
            if self._origin_ship_pos is not None and ship_state is not None:
                curr_ship_x = float(ship_state.x)
                curr_ship_y = float(ship_state.y)
                ds_s = math.sqrt((curr_ship_x - self._last_ship_pos[0])**2 +
                                 (curr_ship_y - self._last_ship_pos[1])**2)
                self._total_ship_dist += ds_s
                self._last_ship_pos = (curr_ship_x, curr_ship_y)

                curr_car_x = pose.position.x_val
                curr_car_y = pose.position.y_val
                ds_c = math.sqrt((curr_car_x - self._last_car_pos[0])**2 +
                                 (curr_car_y - self._last_car_pos[1])**2)
                self._total_car_dist += ds_c
                self._last_car_pos = (curr_car_x, curr_car_y)

                dist_error = (self._total_ship_dist * self._speed_scale) - self._total_car_dist

            self._dist_integral += dist_error
            self._dist_integral = float(np.clip(self._dist_integral, -50.0, 50.0))

            scaled_ship_speed = ship_speed * self._speed_scale
            target_speed = (scaled_ship_speed
                            + self._dist_gain * dist_error
                            + self._dist_ki * self._dist_integral)

            # Cornering logic: slow down if heading error is large
            abs_head_err = abs(heading_error)
            if abs_head_err > math.radians(10.0):
                slow_down = (abs_head_err - math.radians(10.0)) * 3.0
                target_speed -= slow_down

            target_speed = max(0.0, target_speed)

            # PI-controller for throttle
            speed_error = target_speed - airsim_speed
            self._throttle_integral += speed_error
            self._throttle_integral = float(np.clip(
                self._throttle_integral, -20.0, 20.0
            ))

            throttle_cmd = (self._throttle_kp * speed_error +
                            self._throttle_ki * self._throttle_integral)

            if target_speed < 0.1 and airsim_speed < 0.1:
                throttle_cmd = 0.0
                self._throttle_integral = 0.0

            self._controls.throttle = float(np.clip(throttle_cmd, 0.0, 1.0))
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
            A dictionary with the following keys when connected:

            * ``"speed"`` — current vehicle speed in m/s.
            * ``"gear"`` — current gear (integer).
            * ``"timestamp"`` — AirSim timestamp.

            Returns an empty dictionary if the bridge is not connected
            or if the query fails.
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

        Performs a full AirSim vehicle reset, re-enables API control,
        zeroes all PID integrals, and clears the heading-offset
        calibration so it is recalculated on the next ``send_action``
        call.  Does nothing if the bridge is not connected.
        """
        if not self._connected:
            return
        try:
            self._client.reset()
            time.sleep(0.5)
            self._client.enableApiControl(True)
            self._controls = self._airsim.CarControls()
            self._heading_offset = None
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

        Stops the vehicle by zeroing throttle, steering, and brake;
        closes any open telemetry CSV file; and releases API control.
        Safe to call multiple times — subsequent calls are no-ops.
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
