"""
Classical and nonlinear path-following simulation.

This script is intentionally kept separate from the RL training pipeline so it
can be used as a classical-control baseline.  It uses the same vessel simulator
and the same Western Scheldt map CSV files as the main framework visualization.

Main processing flow:
    1. Read the framework settings from fw/configuration/config.py.
    2. Load the configured route and map contours from fw/environments/*.csv.
    3. Optionally simplify the dense imported route into a piecewise-linear
       reference path.
    4. Use the selected guidance law to compute a desired course/heading from
       vessel position and cross-track error.
    5. Use the selected heading controller to compute the physical rudder
       command.  Supported controllers include PID/PD, feedback linearization,
       Norrbin sliding mode, and backstepping dynamic SMC.
    6. Use a surge-speed PID controller to compute the thrust command.
    7. Advance the configured vessel/dynamics model through ShipSimulator.
    8. Post-process the run into map figures, controller-performance figures,
       raw CSV data, scalar metrics, and an animated GIF.

Typical command:
    python -m fw.control.path_following
"""

import argparse
import csv
import json
import os
import webbrowser
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from fw.common.default_positions import DEFAULT_FRAMEWORK_SHIP_POS, DEFAULT_FRAMEWORK_TARGET_POS
from fw.configuration.config import get_config_parameters
from fw.dynamics.dynamics_model import DynamicsBase, Model
from fw.dynamics.fossen import Fossen
from fw.dynamics.nomoto import Nomoto
from fw.dynamics.ugent_surrogate import UGentSurrogate
from fw.ships.ship_factory import ShipModel, create_ship
from fw.ships.vessel_state import VesselState
from fw.simulators.ship_simulator import ShipSimulator


Point = Tuple[float, float]
MIN_GRID_POS = -11700
MAX_GRID_POS = 14500
DEFAULT_ENV_DIR = Path(__file__).resolve().parents[1] / "environments"

CSV_FIELD_UNITS = {
    "time": "s",
    "x": "m",
    "y": "m",
    "heading": "rad",
    "path_heading": "rad",
    "guidance_desired_heading": "rad",
    "controller_desired_heading": "rad",
    "desired_heading": "rad",
    "heading_error": "rad",
    "heading_error_deg": "deg",
    "cross_track_error": "m",
    "cross_track_error_abs": "m",
    "rudder_angle": "rad",
    "raw_rudder_angle": "rad",
    "rudder_feedback_angle": "rad",
    "rudder_feedforward_angle": "rad",
    "rudder_action": "normalized",
    "rudder_rate": "rad/s",
    "raw_rudder_rate": "rad/s",
    "rudder_rate_deg_s": "deg/s",
    "raw_rudder_rate_deg_s": "deg/s",
    "target_speed": "m/s",
    "speed_error": "m/s",
    "thrust": "normalized",
    "thrust_action": "normalized",
    "thrust_rate": "1/s",
    "u": "m/s",
    "v": "m/s",
    "r": "rad/s",
    "los_x": "m",
    "los_y": "m",
    "fl_virtual_acceleration": "rad/s^2",
    "fl_sliding_variable": "rad/s",
    "fl_m_hat": "model",
    "fl_d1_hat": "model",
    "fl_d3_hat": "model",
    "controller_desired_heading_rate": "rad/s",
    "controller_desired_heading_acceleration": "rad/s^2",
    "yaw_rate_error": "rad/s",
    "paper_error": "rad",
    "paper_error_rate": "rad/s",
    "sliding_surface_s": "rad/s",
    "sat_s_over_phi": "dimensionless",
    "boundary_layer_phi": "rad/s",
    "sideslip_angle_beta": "rad",
    "sideslip_compensation_angle": "rad",
    "virtual_course": "rad",
    "raw_controller_desired_heading": "rad",
    "filtered_controller_desired_heading": "rad",
    "cross_track_term": "dimensionless",
    "cross_track_term_rate": "1/s",
    "paper_error_integral": "rad*s",
    "direct_heading_error_integral": "rad*s",
    "heading_integral": "rad*s",
    "heading_error_derivative": "rad/s",
    "norrbin_model_rudder_component": "rad",
    "damping_compensation_component": "rad",
    "direct_heading_rudder_component": "rad",
    "integral_rudder_component": "rad",
    "cross_track_rudder_component": "rad",
    "surge_pid_integral": "m",
    "surge_pid_derivative": "m/s^2",
    "surge_pid_feedback": "normalized",
    "surge_pid_feedforward": "normalized",
}


@dataclass
class PIDController:
    """
    PID controller for heading control.

    The controller input is heading error [rad], and the output is a rudder
    angle command [rad].
    """

    kp: float
    ki: float
    kd: float
    output_limit: float
    integral_limit: float = 2.0

    def __post_init__(self) -> None:
        # The integral and previous-error terms are controller memory.
        # They are initialized outside the dataclass fields because they are
        # runtime state, not controller tuning parameters.
        self._integral = 0.0
        self._previous_error = 0.0

    def reset(self) -> None:
        # Resetting the controller removes any accumulated integral action and
        # derivative history.  This should be called before starting a new run
        # if the same PIDController instance is reused.
        self._integral = 0.0
        self._previous_error = 0.0

    def step(self, error: float, dt: float) -> float:
        # Integral action compensates steady-state heading bias.  The clamp is
        # a basic anti-windup mechanism: without it, the integral term can grow
        # while the rudder is saturated and later cause large overshoot.
        self._integral = np.clip(
            self._integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )

        # The derivative term damps fast heading-error changes.  It is computed
        # from the error difference between two consecutive control samples.
        derivative = (error - self._previous_error) / dt
        self._previous_error = error

        # PID output is a desired physical rudder angle [rad].  The simulator
        # has a finite rudder range, so the command is saturated before it is
        # converted to the normalized action expected by ShipSimulator.
        command = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(command, -self.output_limit, self.output_limit))


@dataclass
class SurgeSpeedPIDController:
    """
    PID controller for surge-speed control.

    The heading loop and the speed loop are intentionally separated:
        - Heading PID controls rudder angle.
        - Surge PID controls the propulsive thrust command.

    The controller input is surge-speed error [m/s]:
        e_u = target_speed - current_surge_speed

    The controller output is a physical normalized thrust command in
    [min_thrust, max_thrust].  A small feed-forward term is added before the
    PID correction.  When this script is run with the simplified Nomoto model,
    the surge equation is approximately:
        du/dt = 3.0 * thrust - 0.02 * u

    At steady state du/dt = 0, so thrust ~= 0.02 * target_speed / 3.0.  This
    This model-based feed-forward gives the PID loop a sensible operating
    point for Nomoto.  When config.py selects another dynamics model, such as
    Fossen, the feed-forward is only a small initial bias and the PI feedback
    learns the thrust level needed to hold the target speed.
    """

    kp: float
    ki: float
    kd: float
    target_speed: float
    min_thrust: float
    max_thrust: float
    integral_limit: float = 20.0
    feedforward_thrust: Optional[float] = None

    def __post_init__(self) -> None:
        self._integral = 0.0
        self._previous_error = 0.0
        self.last_integral = 0.0
        self.last_derivative = 0.0
        self.last_feedback = 0.0
        self.last_feedforward = 0.0

        if self.feedforward_thrust is None:
            # Feed-forward based on the nominal Nomoto surge model:
            # du = 3 * thrust - 0.02 * u.  The required steady thrust for that
            # model is therefore 0.02 * target_speed / 3.  For other configured
            # dynamics models this remains a small bias only; the integral term
            # provides the model-specific steady thrust correction.
            self.feedforward_thrust = 0.02 * self.target_speed / 3.0

    def step(self, current_speed: float, dt: float) -> Tuple[float, float]:
        """
        Compute the thrust command for the current surge speed.

        Returns:
            Tuple (thrust_command, speed_error), where thrust_command is the
            physical normalized thrust sent through ShipSimulator and
            speed_error is saved for diagnostics and plotting.
        """

        speed_error = self.target_speed - current_speed

        # Integral action removes residual speed bias caused by model mismatch
        # or thrust saturation.  Anti-windup is important because the thrust
        # command cannot go below min_thrust or above max_thrust.
        self._integral = np.clip(
            self._integral + speed_error * dt,
            -self.integral_limit,
            self.integral_limit,
        )

        # Derivative action reacts to changes in speed error.  The default gain
        # is zero because the simplified surge dynamics are already smooth, but
        # the term is available for tuning experiments.
        derivative = (speed_error - self._previous_error) / dt
        self._previous_error = speed_error

        feedback = self.kp * speed_error + self.ki * self._integral + self.kd * derivative
        thrust_command = self.feedforward_thrust + feedback
        thrust_command = float(np.clip(thrust_command, self.min_thrust, self.max_thrust))

        self.last_integral = float(self._integral)
        self.last_derivative = float(derivative)
        self.last_feedback = float(feedback)
        self.last_feedforward = float(self.feedforward_thrust)

        return thrust_command, float(speed_error)


@dataclass
class LOSGuidance:
    """
    Classical line-of-sight guidance over piecewise-linear waypoints.

    Cross-track error is computed in the current segment frame:
        e_y = -sin(chi_p) * (x - x_k) + cos(chi_p) * (y - y_k)

    The desired heading is:
        psi_d = chi_p + atan2(-e_y, lookahead_distance)
    """

    waypoints: Sequence[Point]
    lookahead_distance: float
    switch_distance: float = 200.0

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("At least two waypoints are required.")
        if self.lookahead_distance <= 0.0:
            raise ValueError("lookahead_distance must be positive.")
        if self.switch_distance < 0.0:
            raise ValueError("switch_distance must be non-negative.")
        self._segment_index = 0

    @property
    def segment_index(self) -> int:
        return self._segment_index

    @property
    def finished(self) -> bool:
        return self._segment_index >= len(self.waypoints) - 2

    def reached_goal(self, position: Point) -> bool:
        # The run is considered complete only on the last path segment and
        # within the configured switching radius around the final waypoint.
        # This prevents the simulation from continuing along the extension of
        # the final path segment after the target has effectively been reached.
        goal = np.array(self.waypoints[-1], dtype=float)
        pos = np.array(position, dtype=float)
        return self.finished and float(np.linalg.norm(pos - goal)) <= self.switch_distance

    def step(self, position: Point) -> Tuple[float, float, Point]:
        # p0 and p1 are the endpoints of the currently active path segment.
        # The vessel should converge to this segment before advancing to the
        # next segment.
        p0 = np.array(self.waypoints[self._segment_index], dtype=float)
        p1 = np.array(self.waypoints[self._segment_index + 1], dtype=float)
        pos = np.array(position, dtype=float)

        # Build a local path frame.  "unit" points along the segment from p0 to
        # p1.  Projection of the vessel position onto this unit vector gives
        # the along-track progress along the segment.
        segment = p1 - p0
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-9:
            raise ValueError("Consecutive waypoints must not be identical.")

        unit = segment / segment_length
        along_track = float(np.dot(pos - p0, unit))
        distance_to_next = float(np.linalg.norm(pos - p1))

        # Waypoint switching:
        # - If the vessel projection has passed p1, move to the next segment.
        # - Also switch slightly early if the vessel is close to p1.
        #
        # The first condition is important on dense real-map paths.  A vessel
        # can pass a waypoint without entering a small circle around it; if we
        # only used distance_to_next, the controller could stay locked to an
        # old segment and follow its infinite extension.
        if (
            self._segment_index < len(self.waypoints) - 2
            and (
                along_track >= segment_length
                or (
                    along_track >= segment_length - self.switch_distance
                    and distance_to_next <= self.switch_distance
                )
            )
        ):
            self._segment_index += 1
            return self.step(position)

        # chi_p is the path-tangential heading of the active segment.
        path_heading = float(np.arctan2(segment[1], segment[0]))

        # Signed cross-track error in the path frame.  Positive/negative sign
        # depends on which side of the path the vessel is on.  The sign is used
        # by LOS guidance to steer back toward the segment instead of merely
        # reporting a distance magnitude.
        cross_track_error = float(
            -np.sin(path_heading) * (position[0] - p0[0])
            + np.cos(path_heading) * (position[1] - p0[1])
        )

        # Classical LOS law:
        # The vessel does not aim directly at the closest point on the path.
        # Instead, it aims at a virtual point ahead of that projection.  The
        # lookahead distance controls aggressiveness:
        # - smaller lookahead: tighter tracking, more rudder activity;
        # - larger lookahead: smoother motion, more corner cutting.
        desired_heading = wrap_angle(
            path_heading + np.arctan2(-cross_track_error, self.lookahead_distance)
        )

        # los_point is the orthogonal projection of the vessel onto the current
        # segment, clamped to the finite segment endpoints.  It is saved for
        # diagnostics and can be plotted later if needed.
        los_point = tuple((p0 + np.clip(along_track, 0.0, segment_length) * unit).tolist())
        return desired_heading, cross_track_error, los_point



@dataclass
class VectorFieldGuidance:
    """
    Time-varying vector-field guidance for piecewise-linear paths.

    For the active path segment, the desired course/heading is computed as:

        chi_d = gamma_p - sgn(y_e) * atan((|y_e| / Delta) ** theta)

    where `gamma_p` is the segment tangent angle, `y_e` is signed cross-track
    error, `Delta` controls the vector-field transition width, and `theta`
    controls the convergence-rate shape.  The configurable theta model is:

        theta = theta_base + theta_gain * |y_e| / theta_error_scale

    with an upper cap for numerical robustness on full-scale routes.
    """

    waypoints: Sequence[Point]
    delta: float
    switch_distance: float = 15.0
    theta_base: float = 1.0
    theta_gain: float = 0.4
    theta_error_scale: Optional[float] = None
    theta_max: float = 6.0

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("At least two waypoints are required.")
        if self.delta <= 0.0:
            raise ValueError("delta must be positive.")
        self._segment_index = 0
        if self.theta_error_scale is None:
            self.theta_error_scale = self.delta
        self.theta_error_scale = max(float(self.theta_error_scale), 1e-9)

    @property
    def segment_index(self) -> int:
        return self._segment_index

    @property
    def finished(self) -> bool:
        return self._segment_index >= len(self.waypoints) - 2

    def reached_goal(self, position: Point) -> bool:
        goal = np.array(self.waypoints[-1], dtype=float)
        pos = np.array(position, dtype=float)
        return self.finished and float(np.linalg.norm(pos - goal)) <= self.switch_distance

    def step(self, position: Point) -> Tuple[float, float, Point]:
        p0 = np.array(self.waypoints[self._segment_index], dtype=float)
        p1 = np.array(self.waypoints[self._segment_index + 1], dtype=float)
        pos = np.array(position, dtype=float)

        segment = p1 - p0
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-9:
            raise ValueError("Consecutive waypoints must not be identical.")

        unit = segment / segment_length
        along_track = float(np.dot(pos - p0, unit))
        distance_to_next = float(np.linalg.norm(pos - p1))

        if (
            self._segment_index < len(self.waypoints) - 2
            and (
                along_track >= segment_length
                or (
                    along_track >= segment_length - self.switch_distance
                    and distance_to_next <= self.switch_distance
                )
            )
        ):
            self._segment_index += 1
            return self.step(position)

        path_heading = float(np.arctan2(segment[1], segment[0]))
        cross_track_error = float(
            -np.sin(path_heading) * (position[0] - p0[0])
            + np.cos(path_heading) * (position[1] - p0[1])
        )
        abs_error = abs(cross_track_error)
        theta = self.theta_base + self.theta_gain * abs_error / self.theta_error_scale
        theta = float(np.clip(theta, self.theta_base, self.theta_max))
        vector_turn_angle = np.arctan((abs_error / self.delta) ** theta)
        desired_heading = wrap_angle(path_heading - np.sign(cross_track_error) * vector_turn_angle)

        projection = p0 + np.clip(along_track, 0.0, segment_length) * unit
        return desired_heading, cross_track_error, tuple(projection.tolist())


@dataclass
class MapData:
    """
    Static map assets loaded from the same CSV files used by PySimEnv.

    Keeping the map data in a small dataclass makes the post-processing code
    explicit: every visualization receives the river outline, waterway outline,
    and the nominal path from one object instead of reading files again.
    """

    overall: np.ndarray
    obstacles: np.ndarray
    reference_path: np.ndarray


@dataclass(frozen=True)
class MapViewport:
    """
    Shared square x/y limits for all trajectory-style figures.

    Using one viewport object avoids a common visualization problem: one plot
    zooms to the reference path, another zooms to the vessel path, and the user
    cannot visually compare geometric deviations.  The viewport is forced to a
    square data range so all figures preserve a true 1:1 x/y scale.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class ClassicalControlConfig:
    """
    Resolved configuration used by this standalone LOS/PID simulation.

    The fields mirror the parts of `CONFIG_PARAMETERS` that `main.py` forwards
    to PySimEnv: map files, route endpoints, ship model, dynamics model, and
    environmental toggles.  Keeping them in one dataclass makes it explicit
    which parts of the framework configuration are consumed by this script.
    """

    ship_model: ShipModel
    dynamics_model: Model
    start_pos: np.ndarray
    target_pos: np.ndarray
    overall_contour: str
    depth_contour: str
    trajectory: str
    wind: bool
    current: bool
    profile: str


@dataclass(frozen=True)
class PathFollowingAlgorithmConfig:
    """
    Controller and guidance selection for the standalone path-following run.

    The values come from config.py and can be overridden from the command line.
    Keeping them in one object makes the run archive explicit about which
    nonlinear controller and guidance law produced the trajectory.
    """

    controller: str = "pid"
    guidance: str = "los"
    pid: dict = None
    vector_field: dict = None
    feedback_linearization: dict = None
    sliding_mode: dict = None
    backstepping_dynamic_smc: dict = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "controller", self.controller.lower())
        object.__setattr__(self, "guidance", self.guidance.lower())
        if self.pid is None:
            object.__setattr__(
                self,
                "pid",
                {
                    "heading_kp": 0.9,
                    "heading_ki": 0.0,
                    "heading_kd": 0.0,
                    "yaw_rate_damping": 10.0,
                    "nomoto_feedforward_enabled": True,
                    "nomoto_feedforward_gain": 0.05,
                    "desired_yaw_time_constant": 10.0,
                    "desired_heading_rate_limit_deg_s": 5.0,
                    "heading_reference_filter_alpha": 0.6,
                    "integral_limit": 1.0,
                },
            )
        if self.vector_field is None:
            object.__setattr__(
                self,
                "vector_field",
                {
                    "delta": 150.0,
                    "theta_base": 0.8,
                    "theta_gain": 0.16,
                    "theta_error_scale": 150.0,
                    "theta_max": 2.0,
                },
            )
        if self.feedback_linearization is None:
            object.__setattr__(
                self,
                "feedback_linearization",
                {
                    "mode": "norrbin",
                    "lambda": 0.04,
                    "heading_kp": None,
                    "heading_kd": None,
                    "use_integral_action": False,
                    "integral_limit": 1.0,
                    "desired_heading_rate_limit_deg_s": 5.0,
                    "heading_reference_filter_alpha": 0.50,
                    "heading_error_deadband_deg": 0.05,
                    "yaw_rate_error_deadband_deg_s": 0.010,
                    "use_model_estimates": True,
                    "m_hat": None,
                    "d1_hat": None,
                    "d3_hat": 0.0,
                    "min_abs_m_hat": 1e-3,
                    "max_abs_m_hat": 50000.0,
                    "max_abs_d1_hat": 50000.0,
                    "max_abs_d3_hat": 50000.0,
                    "adaptive_enabled": False,
                    "gamma_m": 0.0,
                    "gamma_d1": 0.0,
                    "gamma_d3": 0.0,
                    "min_adaptation_speed": 0.5,
                    "model_gain_epsilon": 1e-6,
                    "model_probe_rudder_deg": 1.0,
                },
            )
        if self.sliding_mode is None:
            object.__setattr__(
                self,
                "sliding_mode",
                {
                    "lambda": 0.30,
                    "eta": 0.00008,
                    "boundary_layer": 0.02,
                    "heading_error_acceleration_gain": 0.00018,
                    "large_error_boost_gain": 0.00012,
                    "large_error_boost_width_deg": 15.0,
                    "direct_heading_rudder_gain_deg": 12.0,
                    "direct_heading_rudder_width_deg": 10.0,
                    "direct_heading_integral_gain_deg_per_rad_s": 0.45,
                    "direct_heading_integral_limit_rad_s": 20.0,
                    "direct_heading_integral_leak_time_s": 60.0,
                    "integral_gain": 0.00004,
                    "integral_limit": 0.20,
                    "desired_heading_rate_limit_deg_s": 6.0,
                    "heading_reference_filter_alpha": 0.45,
                    "desired_acceleration_limit_deg_s2": 0.35,
                    "use_raw_heading_for_error": False,
                    "use_model_estimates": True,
                    "update_model_estimates_each_step": True,
                    "m_hat": None,
                    "d1_hat": None,
                    "d3_hat": 0.0,
                    "min_abs_m_hat": 1e-3,
                    "max_abs_m_hat": 50000.0,
                    "max_abs_d1_hat": 50000.0,
                    "max_abs_d3_hat": 50000.0,
                    "model_gain_epsilon": 1e-6,
                    "model_probe_rudder_deg": 1.0,
                },
            )
        if self.backstepping_dynamic_smc is None:
            object.__setattr__(
                self,
                "backstepping_dynamic_smc",
                {
                    "lambda": 0.30,
                    "eta": 0.00010,
                    "boundary_layer": 0.025,
                    "use_path_tangent_input": False,
                    "virtual_course_from_cross_track": False,
                    "virtual_lookahead_m": 150.0,
                    "cross_track_gain": 0.0,
                    "cross_track_scale_m": 80.0,
                    "cross_track_term_rate_limit": 0.01,
                    "cross_track_rudder_gain_deg": 0.0,
                    "sideslip_compensation_enabled": True,
                    "sideslip_compensation_gain": 0.4,
                    "sideslip_min_speed": 0.5,
                    "sideslip_limit_deg": 12.0,
                    "heading_error_acceleration_gain": 0.00018,
                    "large_error_boost_gain": 0.00008,
                    "large_error_boost_width_deg": 15.0,
                    "direct_heading_rudder_gain_deg": 9.0,
                    "direct_heading_rudder_width_deg": 10.0,
                    "direct_heading_integral_gain_deg_per_rad_s": 0.35,
                    "direct_heading_integral_limit_rad_s": 15.0,
                    "direct_heading_integral_leak_time_s": 45.0,
                    "integral_gain": 0.00003,
                    "integral_limit": 0.20,
                    "desired_heading_rate_limit_deg_s": 6.0,
                    "heading_reference_filter_alpha": 0.15,
                    "desired_acceleration_limit_deg_s2": 0.03,
                    "use_raw_heading_for_error": True,
                    "use_model_estimates": True,
                    "update_model_estimates_each_step": True,
                    "use_damping_compensation": True,
                    "m_hat": None,
                    "d1_hat": None,
                    "d3_hat": 0.0,
                    "min_abs_m_hat": 1e-3,
                    "max_abs_m_hat": 50000.0,
                    "max_abs_d1_hat": 50000.0,
                    "max_abs_d3_hat": 50000.0,
                    "model_gain_epsilon": 1e-6,
                    "model_probe_rudder_deg": 1.0,
                },
            )


@dataclass(frozen=True)
class PathSimplificationConfig:
    """
    Route simplification settings applied immediately after importing CSV path.

    The simplified route is still a piecewise-linear path.  The tolerance is
    the maximum perpendicular distance, in metres, that removed points may
    deviate from the retained straight-line segments.
    """

    enabled: bool = True
    tolerance_m: float = 50.0


@dataclass(frozen=True)
class GuidanceParameterConfig:
    """
    Guidance-law geometry and switching settings from config.py.
    """

    lookahead_distance: Optional[float] = None
    lookahead_ship_length_multiplier: float = 3.0
    switch_distance: float = 35.0


@dataclass(frozen=True)
class SurgeSpeedControlConfig:
    """
    Surge-speed PID settings from config.py.
    """

    target_speed: float = 4.0
    kp: float = 0.8
    ki: float = 0.01
    kd: float = 0.0
    integral_limit: float = 20.0
    feedforward_thrust: Optional[float] = None
    adaptive_turn_slowdown_enabled: bool = False
    min_turn_speed: float = 3.3
    turn_slowdown_heading_error_deg: float = 15.0
    turn_slowdown_max_reduction: float = 0.20


@dataclass(frozen=True)
class ControllerOutputLimitConfig:
    """
    Physical output limits applied before commands enter ShipSimulator.
    """

    rudder_angle_limit_deg: Optional[float] = None
    min_thrust: float = 0.0
    max_thrust: Optional[float] = None


@dataclass(frozen=True)
class RudderResponseConfig:
    """
    First-order rudder actuator response used to smooth controller commands.

    The heading controller still computes the mathematically desired rudder
    angle.  This response model converts that raw command into a realizable
    actuator command before the simulator sees it.
    """

    enabled: bool = True
    time_constant_s: float = 0.15
    small_signal_time_constant_s: float = 1.0
    large_signal_threshold_deg: float = 1.5
    max_rate_deg_s: Optional[float] = None
    deadband_deg: float = 0.02
    neutral_deadband_deg: float = 0.20


def wrap_angle(angle: float) -> float:
    """
    Wrap an angle to [-pi, pi].

    This is important for heading control because a raw subtraction such as
    desired_heading - current_heading can jump by almost 2*pi at the wrap point.
    """

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def unwrap_angle_series(angles: Iterable[float]) -> np.ndarray:
    """
    Convert a wrapped angle history into a continuous degree series.

    Vessel heading and LOS desired heading are stored in the standard
    [-pi, pi] range.  That representation is correct for control, but a plot
    can show artificial jumps at +/-180 deg.  Unwrapping is only used for
    visualization; heading error metrics continue to use wrapped angle errors.
    """

    return np.degrees(np.unwrap(np.array(list(angles), dtype=float)))


def continuous_heading_tracking_series(samples: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build visually consistent heading and desired-heading histories.

    Heading control uses wrapped error:
        heading_error = wrap(desired_heading - heading)

    For plotting, the actual heading is unwrapped first.  The desired heading
    curve is then reconstructed as:
        desired_heading_continuous = heading_continuous + heading_error

    This preserves the exact controller error on the plot and avoids a common
    display artifact where independently unwrapped actual/desired headings
    appear separated by a false 360 deg offset.
    """

    heading_continuous = unwrap_angle_series(row["heading"] for row in samples)
    heading_error_deg = np.degrees([row["heading_error"] for row in samples])
    desired_heading_continuous = heading_continuous + heading_error_deg
    return heading_continuous, desired_heading_continuous


def thrust_to_normalized(thrust: float, min_thrust: float, max_thrust: float) -> float:
    """
    Convert a physical normalized thrust command into the simulator action range.

    ShipSimulator expects actions in [-1, 1].  ActionMapper then maps that value
    back into [min_thrust, max_thrust], so this function is the inverse mapping.
    """

    thrust = float(np.clip(thrust, min_thrust, max_thrust))
    return 2.0 * (thrust - min_thrust) / (max_thrust - min_thrust) - 1.0


@dataclass(frozen=True)
class FrameworkControllerConfig:
    """
    LOS/PID controller settings used when this module is called through main.py.

    The standalone script has CLI arguments.  The framework runner does not use
    argparse because main.py is driven by config.py, so this small dataclass
    gives the main entrypoint an explicit, serializable controller setup.
    """

    lookahead_distance: Optional[float] = None
    heading_kp: float = 1.6
    heading_ki: float = 0.0
    heading_kd: float = 20.0
    heading_yaw_rate_gain: float = 8.0
    heading_reference_alpha: float = 0.8
    max_heading_reference_rate_deg_s: float = 15.0
    surge_kp: float = 0.8
    surge_ki: float = 0.01
    surge_kd: float = 0.0
    target_speed: float = 4.0
    switch_distance: float = 35.0


class FrameworkPathFollowingAgent:
    """
    Adapter that lets the classical LOS/PID controller behave like an RL agent.

    The framework visualization path is built around `EpisodeCollector`, which
    expects an object with `predict(state, deterministic=True)`.  This adapter
    implements that method by reading the live PySimEnv state, computing a LOS
    desired heading, closing the heading loop with PD/PID rudder control, and
    closing the surge-speed loop with PID thrust control.

    The returned action has the same normalized shape as the PPO agents:
        np.array([rudder_action, thrust_action]) in [-1, 1].

    That compatibility is what allows main.py to reuse the standard
    MetricsComputer, MatplotlibRenderer, and AnimationRenderer without a second
    path-following-specific post-processing pipeline.
    """

    def __init__(
        self,
        env,
        algorithm_config: PathFollowingAlgorithmConfig,
        guidance_config: GuidanceParameterConfig,
        surge_config: SurgeSpeedControlConfig,
        output_limit_config: ControllerOutputLimitConfig,
        rudder_response_config: RudderResponseConfig,
        model_dir: Path,
    ) -> None:
        self.env = env
        self.algorithm_config = algorithm_config
        self.guidance_config = guidance_config
        self.surge_config = surge_config
        self.output_limit_config = output_limit_config
        self.rudder_response_config = rudder_response_config
        self.model_dir = str(model_dir)
        self._lookahead_distance: Optional[float] = None
        self._heading_pid: Optional[NomotoFeedforwardPIDHeadingController] = None
        self._feedback_linearization: Optional[FeedbackLinearizationHeadingController] = None
        self._sliding_mode: Optional[SlidingModeHeadingController] = None
        self._backstepping_dynamic_smc: Optional[BacksteppingDynamicSMCController] = None
        self._surge_controller: Optional[SurgeSpeedPIDController] = None
        self._rudder_response: Optional[RudderResponseFilter] = None
        self._controller_step_index: Optional[int] = None
        self.controller_diagnostics: list[dict] = []

    def set_policy_eval(self) -> None:
        """
        Match the RL agent interface.

        PPO agents switch neural networks into evaluation mode here.  The
        classical controller has no train/eval state, so this is intentionally
        a no-op.
        """

    def get_env(self):
        """
        Return the environment controlled by this adapter.

        AgentManager uses this method on RL agents; providing it here keeps the
        adapter compatible with either direct EpisodeCollector use or a future
        AgentManager-based integration.
        """

        return self.env

    def predict(self, state, deterministic: bool = True):
        """
        Compute one normalized environment action from the current env state.

        Args:
            state: Observation returned by the environment.  The classical
                controller does not need the normalized observation vector
                because it reads physical state directly from env.phys_sim.
            deterministic: Present for API compatibility with PPO agents.

        Returns:
            Tuple `(action, None, None)`, matching the RL agent convention used
            by EpisodeCollector.  The second and third values would normally be
            log-probability and value estimate.
        """

        del state, deterministic

        if self._needs_controller_reset():
            self._reset_controllers_from_env()

        specs = self.env.ship.specifications
        sim_state = self.env.phys_sim.state
        dt = float(getattr(self.env, "time_step", 1.0))

        raw_desired_heading, cross_track_error, path_segment_heading = (
            self._compute_environment_synchronized_guidance()
        )
        desired_heading = raw_desired_heading
        heading_error = wrap_angle(desired_heading - sim_state.heading)

        # Store the LOS diagnostics on the env before the step.  PySimEnv will
        # overwrite these after env.step(), but setting them here makes the
        # values available to live renderers and keeps the initial diagnostic
        # state meaningful.
        self.env.heading_error = heading_error
        self.env.cross_error = abs(cross_track_error)

        if self.surge_config.adaptive_turn_slowdown_enabled:
            heading_error_scale = max(np.deg2rad(self.surge_config.turn_slowdown_heading_error_deg), 1e-6)
            slowdown = self.surge_config.turn_slowdown_max_reduction * np.tanh(
                abs(heading_error) / heading_error_scale
            )
            target_speed = self.surge_config.target_speed * (1.0 - slowdown)
            target_speed = max(self.surge_config.min_turn_speed, target_speed)
            target_speed = min(self.surge_config.target_speed, target_speed)
            self._surge_controller.target_speed = float(target_speed)
        else:
            self._surge_controller.target_speed = float(self.surge_config.target_speed)

        thrust_command, speed_error = self._surge_controller.step(float(sim_state.u), dt)
        rudder_angle, desired_heading, heading_error, virtual_acceleration = self._compute_rudder_command(
            state=sim_state,
            raw_desired_heading=raw_desired_heading,
            path_segment_heading=path_segment_heading,
            cross_track_error=cross_track_error,
            thrust=thrust_command,
            dt=dt,
        )
        raw_rudder_angle = rudder_angle
        rudder_angle = self._rudder_response.step(rudder_angle, dt)

        rudder_action = float(np.clip(rudder_angle / specs.max_rudder_angle, -1.0, 1.0))
        thrust_action = thrust_to_normalized(thrust_command, specs.min_thrust, specs.max_thrust)
        sliding_variable = 0.0
        active_controller = self.algorithm_config.controller
        if active_controller == "feedback_linearization":
            sliding_variable = self._feedback_linearization.last_sliding_variable
        elif active_controller == "sliding_mode":
            sliding_variable = self._sliding_mode.last_sliding_variable
        elif active_controller == "backstepping_dynamic_smc":
            sliding_variable = self._backstepping_dynamic_smc.last_sliding_variable

        controller_step = int(self._controller_step_index or 0)
        self.controller_diagnostics.append(
            {
                "step": controller_step,
                "time_s": controller_step * dt,
                "x_m": float(sim_state.x),
                "y_m": float(sim_state.y),
                "checkpoint_index": int(getattr(self.env, "checkpoint_index", 0)),
                "cross_track_error_m": float(cross_track_error),
                "raw_desired_heading_rad": float(raw_desired_heading),
                "filtered_desired_heading_rad": float(desired_heading),
                "actual_heading_rad": float(sim_state.heading),
                "controller_heading_error_rad": float(heading_error),
                "yaw_rate_rad_s": float(getattr(sim_state, "r", 0.0)),
                "raw_rudder_angle_rad": float(raw_rudder_angle),
                "rudder_angle_rad": float(rudder_angle),
                "rudder_action": rudder_action,
                "surge_velocity_mps": float(sim_state.u),
                "surge_speed_error_mps": float(speed_error),
                "thrust_command": float(thrust_command),
                "thrust_action": float(thrust_action),
                "virtual_yaw_acceleration_rad_s2": float(virtual_acceleration),
                "sliding_variable": float(sliding_variable),
                "controller": active_controller,
                "guidance": self.algorithm_config.guidance,
            }
        )
        self._controller_step_index = controller_step + 1
        action = np.array([rudder_action, thrust_action], dtype=np.float32)
        return action, None, None

    def _needs_controller_reset(self) -> bool:
        """
        Detect episode resets performed by EpisodeCollector.

        EpisodeCollector calls env.reset() before each episode.  Resetting
        controllers here clears integrators and reinitializes LOS segment
        progress so the second episode never inherits controller memory from
        the first.
        """

        env_step_count = int(getattr(self.env, "step_count", 0))
        checkpoint_index = int(getattr(self.env, "checkpoint_index", 1))
        episode_restarted = (
            self._controller_step_index is not None
            and self._controller_step_index > 0
            and env_step_count == 0
            and checkpoint_index <= 1
        )
        return (
            self._lookahead_distance is None
            or self._heading_pid is None
            or self._feedback_linearization is None
            or self._sliding_mode is None
            or self._backstepping_dynamic_smc is None
            or self._surge_controller is None
            or self._rudder_response is None
            or self._controller_step_index is None
            or episode_restarted
        )

    def _reset_controllers_from_env(self) -> None:
        """
        Build LOS/PID state from the current configured env.

        The controller deliberately does not keep its own waypoint index in
        framework mode.  PySimEnv already advances `checkpoint_index` when the
        ship reaches or passes checkpoints; reading that index every step keeps
        the controller, reward function, metrics, and map renderer aligned.
        """

        if not hasattr(self.env, "checkpoints") or len(self.env.checkpoints) < 2:
            raise ValueError("Framework LOS/PID runner requires an environment with at least two checkpoints.")

        _apply_framework_output_limits(self.env, self.output_limit_config)
        specs = self.env.ship.specifications
        lookahead = self.guidance_config.lookahead_distance
        if lookahead is None:
            lookahead = self.guidance_config.lookahead_ship_length_multiplier * float(specs.length)

        self._lookahead_distance = float(lookahead)
        dynamics = getattr(self.env, "dynamics", None)
        if dynamics is None:
            dynamics = getattr(getattr(self.env, "phys_sim", None), "_dynamics", None)
        if dynamics is None:
            raise ValueError("Framework path-following runner requires an environment dynamics model.")

        pid_params = self.algorithm_config.pid
        self._heading_pid = NomotoFeedforwardPIDHeadingController(
            specs=specs,
            kp=float(pid_params.get("heading_kp", 0.9)),
            ki=float(pid_params.get("heading_ki", 0.0)),
            kd=float(pid_params.get("heading_kd", 0.0)),
            yaw_rate_damping=float(pid_params.get("yaw_rate_damping", 10.0)),
            feedforward_enabled=bool(pid_params.get("nomoto_feedforward_enabled", True)),
            feedforward_gain=float(pid_params.get("nomoto_feedforward_gain", 0.05)),
            desired_yaw_time_constant=float(pid_params.get("desired_yaw_time_constant", 10.0)),
            desired_heading_rate_limit_deg_s=float(pid_params.get("desired_heading_rate_limit_deg_s", 5.0)),
            heading_reference_filter_alpha=float(pid_params.get("heading_reference_filter_alpha", 0.6)),
            integral_limit=float(pid_params.get("integral_limit", 1.0)),
        )
        fl_params = self.algorithm_config.feedback_linearization
        self._feedback_linearization = FeedbackLinearizationHeadingController(
            dynamics=dynamics,
            max_rudder_angle=specs.max_rudder_angle,
            ship_length=specs.length,
            params=fl_params,
            kp=fl_params.get("heading_kp", None),
            kd=fl_params.get("heading_kd", None),
            gain_epsilon=float(fl_params.get("model_gain_epsilon", 1e-6)),
        )
        smc_params = self.algorithm_config.sliding_mode
        self._sliding_mode = SlidingModeHeadingController(
            dynamics=dynamics,
            max_rudder_angle=specs.max_rudder_angle,
            ship_length=specs.length,
            params=smc_params,
            lambda_gain=float(smc_params.get("lambda", 0.12)),
            eta=float(smc_params.get("eta", 0.00002)),
            boundary_layer=float(smc_params.get("boundary_layer", 0.03)),
        )
        backstepping_params = self.algorithm_config.backstepping_dynamic_smc
        self._backstepping_dynamic_smc = BacksteppingDynamicSMCController(
            dynamics=dynamics,
            max_rudder_angle=specs.max_rudder_angle,
            ship_length=specs.length,
            params=backstepping_params,
            lambda_gain=float(backstepping_params.get("lambda", 0.30)),
            eta=float(backstepping_params.get("eta", 0.00010)),
            boundary_layer=float(backstepping_params.get("boundary_layer", 0.025)),
        )
        self._surge_controller = SurgeSpeedPIDController(
            kp=float(self.surge_config.kp),
            ki=float(self.surge_config.ki),
            kd=float(self.surge_config.kd),
            target_speed=float(self.surge_config.target_speed),
            min_thrust=float(specs.min_thrust),
            max_thrust=float(specs.max_thrust),
            integral_limit=float(self.surge_config.integral_limit),
            feedforward_thrust=self.surge_config.feedforward_thrust,
        )
        self._rudder_response = RudderResponseFilter(
            config=self.rudder_response_config,
            max_rudder_angle=specs.max_rudder_angle,
            default_max_rudder_rate=specs.max_rudder_rate,
        )
        self._controller_step_index = 0

    def _compute_rudder_command(
        self,
        state: VesselState,
        raw_desired_heading: float,
        path_segment_heading: float,
        cross_track_error: float,
        thrust: float,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        """
        Dispatch the configured heading controller in framework-rollout mode.
        """

        if self.algorithm_config.controller == "pid":
            rudder_angle, _, _, desired_heading, heading_error = self._heading_pid.step(
                state,
                raw_desired_heading,
                dt,
            )
            return rudder_angle, desired_heading, heading_error, 0.0
        if self.algorithm_config.controller == "feedback_linearization":
            return self._feedback_linearization.step(state, raw_desired_heading, thrust, dt)
        if self.algorithm_config.controller == "sliding_mode":
            return self._sliding_mode.step(state, raw_desired_heading, thrust, dt)
        if self.algorithm_config.controller == "backstepping_dynamic_smc":
            if self._backstepping_dynamic_smc.use_path_tangent_input:
                controller_course = path_segment_heading
            else:
                controller_course = raw_desired_heading
            return self._backstepping_dynamic_smc.step(
                state,
                controller_course,
                cross_track_error,
                thrust,
                dt,
            )
        raise ValueError(
            f"Unsupported path_following_controller={self.algorithm_config.controller!r}. "
            "Use 'pid', 'feedback_linearization', 'sliding_mode', or 'backstepping_dynamic_smc'."
        )

    def _compute_environment_synchronized_guidance(self) -> Tuple[float, float, float]:
        """
        Compute guidance on the same active path segment as PySimEnv.

        The framework renderer and reward logic use PySimEnv's checkpoint
        progression, so the controller reads that same active segment instead of
        maintaining a separate guidance object.  This keeps path switching,
        metrics, and visualization aligned while still using the guidance law
        selected in config.py.

        Returns `(desired_heading, signed_cross_track_error, path_heading)`.
        """

        checkpoints = self.env.checkpoints
        current_index = int(np.clip(getattr(self.env, "checkpoint_index", 1), 1, len(checkpoints) - 1))
        previous_index = max(0, current_index - 1)

        p0 = np.asarray(checkpoints[previous_index]["pos"], dtype=float)
        p1 = np.asarray(checkpoints[current_index]["pos"], dtype=float)
        pos = np.asarray([self.env.phys_sim.state.x, self.env.phys_sim.state.y], dtype=float)

        segment = p1 - p0
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-9:
            # Degenerate checkpoint spacing should not happen after the
            # framework path pre-processing, but pointing directly at the
            # current checkpoint is a stable fallback.
            direct = p1 - pos
            heading = float(np.arctan2(direct[1], direct[0]))
            return heading, 0.0, heading

        path_heading = float(np.arctan2(segment[1], segment[0]))
        cross_track_error = float(
            -np.sin(path_heading) * (pos[0] - p0[0])
            + np.cos(path_heading) * (pos[1] - p0[1])
        )

        if self.algorithm_config.guidance in {"los", "pure_pursuit"}:
            desired_heading = wrap_angle(
                path_heading + np.arctan2(-cross_track_error, self._lookahead_distance)
            )
        elif self.algorithm_config.guidance == "vector_field":
            vf_params = self.algorithm_config.vector_field
            delta = vf_params.get("delta", None)
            if delta is None:
                delta = self._lookahead_distance
            theta_error_scale = vf_params.get("theta_error_scale", None)
            if theta_error_scale is None:
                theta_error_scale = delta
            theta_error_scale = max(float(theta_error_scale), 1e-9)
            abs_error = abs(cross_track_error)
            theta = float(vf_params.get("theta_base", 1.0)) + float(vf_params.get("theta_gain", 0.2)) * (
                abs_error / theta_error_scale
            )
            theta = float(np.clip(theta, float(vf_params.get("theta_base", 1.0)), float(vf_params.get("theta_max", 2.5))))
            turn_angle = np.arctan((abs_error / max(float(delta), 1e-9)) ** theta)
            desired_heading = wrap_angle(path_heading - np.sign(cross_track_error) * turn_angle)
        else:
            raise ValueError(
                f"Unsupported path_following_guidance={self.algorithm_config.guidance!r}. "
                "Use 'los', 'pure_pursuit', or 'vector_field'."
            )
        return desired_heading, cross_track_error, path_heading


class FeedbackLinearizationHeadingController:
    """
    Feedback-linearizing heading controller for ship steering.

    The default implementation follows the structure used by Fossen and
    Paulsen for Norrbin-type steering dynamics.  Their starting model is

        m * psi_ddot + d1 * psi_dot + d3 * psi_dot^3 = delta

    where `psi` is heading, `psi_dot` is yaw rate `r`, and `delta` is rudder
    angle.  The feedback-linearizing control law directly cancels the modeled
    yaw-rate damping terms:

        delta = m_hat * a_psi + d1_hat * r + d3_hat * r^3

    The virtual input `a_psi` is a desired heading acceleration.  With
    pole-placement gains it creates a second-order heading-error equation:

        e_ddot + Kd * e_dot + Kp * e = 0,

    with paper notation `e = psi - psi_d`.

    A legacy `numerical_model_inverse` mode is kept for comparison.  That mode
    performs the previous finite-difference inversion of the active simulator:

        r_dot = f(x) + g(x) * delta.
    """

    def __init__(
        self,
        dynamics: DynamicsBase,
        max_rudder_angle: float,
        ship_length: float,
        params: dict,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        gain_epsilon: float = 1e-6,
    ) -> None:
        self.dynamics = dynamics
        self.max_rudder_angle = float(max_rudder_angle)
        self.ship_length = max(float(ship_length), 1e-6)
        self.params = dict(params or {})
        self.mode = str(self.params.get("mode", "norrbin")).lower()
        self.lambda_gain = float(self.params.get("lambda", 0.04))

        configured_kp = self.params.get("heading_kp", kp)
        configured_kd = self.params.get("heading_kd", kd)
        if configured_kp is None and self.lambda_gain > 0.0:
            configured_kp = self.lambda_gain ** 2.0
        if configured_kd is None and self.lambda_gain > 0.0:
            configured_kd = 2.0 * self.lambda_gain
        self.kp = float(0.8 if configured_kp is None else configured_kp)
        self.kd = float(1.8 if configured_kd is None else configured_kd)

        self.use_integral_action = bool(self.params.get("use_integral_action", False))
        self.integral_limit = float(self.params.get("integral_limit", 1.0))
        self.desired_heading_rate_limit = np.deg2rad(
            float(self.params.get("desired_heading_rate_limit_deg_s", 5.0))
        )
        self.heading_reference_filter_alpha = float(
            np.clip(self.params.get("heading_reference_filter_alpha", 0.50), 0.0, 1.0)
        )
        self.heading_error_deadband = np.deg2rad(float(self.params.get("heading_error_deadband_deg", 0.05)))
        self.yaw_rate_error_deadband = np.deg2rad(float(self.params.get("yaw_rate_error_deadband_deg_s", 0.010)))

        self.use_model_estimates = bool(self.params.get("use_model_estimates", True))
        self.m_hat = self._optional_float(self.params.get("m_hat", None))
        self.d1_hat = self._optional_float(self.params.get("d1_hat", None))
        self.d3_hat = float(self.params.get("d3_hat", 0.0) or 0.0)
        self.min_abs_m_hat = float(self.params.get("min_abs_m_hat", 1e-3))
        self.max_abs_m_hat = float(self.params.get("max_abs_m_hat", 50000.0))
        self.max_abs_d1_hat = float(self.params.get("max_abs_d1_hat", 50000.0))
        self.max_abs_d3_hat = float(self.params.get("max_abs_d3_hat", 50000.0))

        self.adaptive_enabled = bool(self.params.get("adaptive_enabled", False))
        self.gamma_m = float(self.params.get("gamma_m", 0.0))
        self.gamma_d1 = float(self.params.get("gamma_d1", 0.0))
        self.gamma_d3 = float(self.params.get("gamma_d3", 0.0))
        self.min_adaptation_speed = float(self.params.get("min_adaptation_speed", 0.5))

        self.gain_epsilon = float(self.params.get("model_gain_epsilon", gain_epsilon))
        self.model_probe_rudder = np.deg2rad(float(self.params.get("model_probe_rudder_deg", 1.0)))

        self._filtered_desired_heading: Optional[float] = None
        self._desired_heading_rate = 0.0
        self._integral_heading_error = 0.0
        self._model_estimates_initialized = self.m_hat is not None and self.d1_hat is not None
        self.last_virtual_acceleration = 0.0
        self.last_sliding_variable = 0.0
        self.last_raw_desired_heading = 0.0
        self.last_filtered_desired_heading = 0.0
        self.last_desired_heading_rate = 0.0
        self.last_desired_heading_acceleration = 0.0
        self.last_yaw_rate_error = 0.0
        self.last_paper_error = 0.0
        self.last_paper_error_rate = 0.0
        self.last_sat_s_over_phi = 0.0
        self.last_norrbin_model_rudder_component = 0.0
        self.last_damping_compensation_component = 0.0
        self.last_direct_heading_rudder_component = 0.0
        self.last_integral_rudder_component = 0.0
        self.last_cross_track_rudder_component = 0.0

    @staticmethod
    def _optional_float(value: object) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _state_with_yaw_rate(state: VesselState, yaw_rate: float) -> VesselState:
        return VesselState(
            terminated=state.terminated,
            truncated=state.truncated,
            x=state.x,
            y=state.y,
            heading=state.heading,
            u=state.u,
            v=state.v,
            r=float(yaw_rate),
        )

    def _yaw_acceleration(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
        dt: float,
    ) -> float:
        _, _, next_yaw_rate = self.dynamics.step(state, rudder_angle, thrust, dt)
        return float((next_yaw_rate - state.r) / dt)

    def _clip_parameter_estimates(self) -> None:
        m_hat = float(1.0 if self.m_hat is None else self.m_hat)
        if abs(m_hat) < self.min_abs_m_hat:
            m_hat = np.copysign(self.min_abs_m_hat, m_hat if m_hat != 0.0 else 1.0)
        self.m_hat = float(np.clip(m_hat, -self.max_abs_m_hat, self.max_abs_m_hat))
        self.d1_hat = float(
            np.clip(0.0 if self.d1_hat is None else self.d1_hat, -self.max_abs_d1_hat, self.max_abs_d1_hat)
        )
        self.d3_hat = float(np.clip(self.d3_hat, -self.max_abs_d3_hat, self.max_abs_d3_hat))

    def _initialize_norrbin_estimates(self, state: VesselState, thrust: float, dt: float) -> None:
        """
        Initialize `(m_hat, d1_hat, d3_hat)` from the selected simulator model.

        The paper treats these as steering-model parameters.  This framework
        may run Nomoto, Fossen, or surrogate dynamics, so we estimate a local
        Norrbin approximation around the initial operating point:

            r_dot ~= (1 / m_hat) * delta - (d1_hat / m_hat) * r.

        The cubic damping term is left at its configured value because a
        reliable `d3` estimate would require multiple yaw-rate excitation
        points, which is not available before the run starts.
        """

        if self._model_estimates_initialized or not self.use_model_estimates:
            self._clip_parameter_estimates()
            self._model_estimates_initialized = True
            return

        delta_probe = max(self.model_probe_rudder, np.deg2rad(0.25))
        delta_probe = min(delta_probe, self.max_rudder_angle)
        r_dot_zero = self._yaw_acceleration(state, 0.0, thrust, dt)
        r_dot_plus = self._yaw_acceleration(state, delta_probe, thrust, dt)
        rudder_gain = (r_dot_plus - r_dot_zero) / delta_probe

        if abs(rudder_gain) >= self.gain_epsilon:
            if self.m_hat is None:
                self.m_hat = 1.0 / rudder_gain

            if self.d1_hat is None:
                yaw_rate_probe = max(np.deg2rad(0.5), 0.1 * max(abs(state.r), np.deg2rad(1.0)))
                perturbed_state = self._state_with_yaw_rate(state, state.r + yaw_rate_probe)
                r_dot_perturbed = self._yaw_acceleration(perturbed_state, 0.0, thrust, dt)
                yaw_damping_slope = (r_dot_perturbed - r_dot_zero) / yaw_rate_probe
                self.d1_hat = -yaw_damping_slope / rudder_gain

        if self.m_hat is None:
            self.m_hat = 1.0
        if self.d1_hat is None:
            self.d1_hat = 0.0

        self._clip_parameter_estimates()
        self._model_estimates_initialized = True

    def _smooth_desired_heading(self, raw_desired_heading: float, dt: float) -> Tuple[float, float, float]:
        previous_heading = self._filtered_desired_heading
        previous_rate = self._desired_heading_rate
        if previous_heading is None:
            filtered_heading = float(raw_desired_heading)
            desired_rate = 0.0
            desired_acceleration = 0.0
        else:
            raw_delta = wrap_angle(raw_desired_heading - previous_heading)
            limited_delta = float(
                np.clip(
                    raw_delta,
                    -self.desired_heading_rate_limit * dt,
                    self.desired_heading_rate_limit * dt,
                )
            )
            filtered_heading = wrap_angle(
                previous_heading + self.heading_reference_filter_alpha * limited_delta
            )
            desired_rate = wrap_angle(filtered_heading - previous_heading) / dt
            desired_acceleration = (desired_rate - previous_rate) / dt

        self._filtered_desired_heading = filtered_heading
        self._desired_heading_rate = desired_rate
        self.last_raw_desired_heading = float(raw_desired_heading)
        self.last_filtered_desired_heading = float(filtered_heading)
        self.last_desired_heading_rate = float(desired_rate)
        self.last_desired_heading_acceleration = float(desired_acceleration)
        return filtered_heading, desired_rate, desired_acceleration

    def _step_numerical_model_inverse(
        self,
        state: VesselState,
        heading_error: float,
        thrust: float,
        dt: float,
    ) -> float:
        delta_probe = max(0.1 * self.max_rudder_angle, self.model_probe_rudder)
        delta_probe = min(delta_probe, self.max_rudder_angle)

        r_dot_zero = self._yaw_acceleration(state, 0.0, thrust, dt)
        r_dot_plus = self._yaw_acceleration(state, delta_probe, thrust, dt)
        rudder_gain = (r_dot_plus - r_dot_zero) / delta_probe

        desired_yaw_acceleration = self.kp * heading_error - self.kd * state.r
        if abs(rudder_gain) < self.gain_epsilon:
            command = 0.0
        else:
            command = (desired_yaw_acceleration - r_dot_zero) / rudder_gain

        return float(np.clip(command, -self.max_rudder_angle, self.max_rudder_angle))

    def _adapt_parameters(
        self,
        virtual_acceleration: float,
        yaw_rate: float,
        paper_sliding_variable: float,
        surge_speed: float,
        dt: float,
    ) -> None:
        if not self.adaptive_enabled:
            return
        adaptation_speed = max(abs(surge_speed), self.min_adaptation_speed)
        speed_over_length = adaptation_speed / self.ship_length
        length_over_speed = self.ship_length / adaptation_speed

        self.m_hat = float(self.m_hat - self.gamma_m * virtual_acceleration * paper_sliding_variable * dt)
        self.d1_hat = float(self.d1_hat - self.gamma_d1 * speed_over_length * yaw_rate * paper_sliding_variable * dt)
        self.d3_hat = float(self.d3_hat - self.gamma_d3 * length_over_speed * yaw_rate ** 3.0 * paper_sliding_variable * dt)
        self._clip_parameter_estimates()

    def step(
        self,
        state: VesselState,
        raw_desired_heading: float,
        thrust: float,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute rudder command.

        Returns:
            Tuple `(rudder_angle, filtered_desired_heading, heading_error,
            virtual_heading_acceleration)`.
        """

        filtered_heading, desired_rate, desired_acceleration = self._smooth_desired_heading(
            raw_desired_heading,
            dt,
        )
        heading_error = wrap_angle(filtered_heading - state.heading)
        if abs(heading_error) < self.heading_error_deadband:
            heading_error = 0.0

        if self.mode == "numerical_model_inverse":
            rudder_angle = self._step_numerical_model_inverse(state, heading_error, thrust, dt)
            self.last_virtual_acceleration = self.kp * heading_error - self.kd * state.r
            self.last_sliding_variable = 0.0
            self.last_yaw_rate_error = float(state.r - desired_rate)
            self.last_paper_error = float(-heading_error)
            self.last_paper_error_rate = float(state.r - desired_rate)
            self.last_norrbin_model_rudder_component = float(rudder_angle)
            self.last_damping_compensation_component = 0.0
            self.last_direct_heading_rudder_component = 0.0
            self.last_integral_rudder_component = 0.0
            self.last_cross_track_rudder_component = 0.0
            return rudder_angle, filtered_heading, heading_error, self.last_virtual_acceleration

        if self.mode != "norrbin":
            raise ValueError(
                f"Unsupported feedback_linearization mode={self.mode!r}. "
                "Use 'norrbin' or 'numerical_model_inverse'."
            )

        self._initialize_norrbin_estimates(state, thrust, dt)

        desired_yaw_rate_error = desired_rate - state.r
        if abs(desired_yaw_rate_error) < self.yaw_rate_error_deadband:
            desired_yaw_rate_error = 0.0
        self._integral_heading_error = float(
            np.clip(
                self._integral_heading_error + heading_error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )

        if self.use_integral_action and self.lambda_gain > 0.0:
            virtual_acceleration = (
                desired_acceleration
                + 3.0 * self.lambda_gain * desired_yaw_rate_error
                + 3.0 * self.lambda_gain ** 2.0 * heading_error
                + self.lambda_gain ** 3.0 * self._integral_heading_error
            )
            paper_error = -heading_error
            paper_error_rate = state.r - desired_rate
            paper_error_integral = -self._integral_heading_error
            paper_sliding_variable = (
                paper_error_rate
                + 2.0 * self.lambda_gain * paper_error
                + self.lambda_gain ** 2.0 * paper_error_integral
            )
        else:
            virtual_acceleration = (
                desired_acceleration
                + self.kd * desired_yaw_rate_error
                + self.kp * heading_error
            )
            paper_error = -heading_error
            paper_error_rate = state.r - desired_rate
            paper_sliding_variable = paper_error_rate + self.lambda_gain * paper_error

        rudder_angle = (
            float(self.m_hat) * virtual_acceleration
            + float(self.d1_hat) * state.r
            + float(self.d3_hat) * state.r ** 3.0
        )
        rudder_angle = float(np.clip(rudder_angle, -self.max_rudder_angle, self.max_rudder_angle))

        self.last_virtual_acceleration = float(virtual_acceleration)
        self.last_sliding_variable = float(paper_sliding_variable)
        self.last_yaw_rate_error = float(state.r - desired_rate)
        self.last_paper_error = float(paper_error)
        self.last_paper_error_rate = float(paper_error_rate)
        self.last_sat_s_over_phi = 0.0
        self.last_norrbin_model_rudder_component = float(rudder_angle)
        self.last_damping_compensation_component = float(self.d1_hat) * state.r
        self.last_direct_heading_rudder_component = 0.0
        self.last_integral_rudder_component = 0.0
        self.last_cross_track_rudder_component = 0.0
        self._adapt_parameters(
            virtual_acceleration=virtual_acceleration,
            yaw_rate=state.r,
            paper_sliding_variable=paper_sliding_variable,
            surge_speed=state.u,
            dt=dt,
        )

        return rudder_angle, filtered_heading, heading_error, self.last_virtual_acceleration


class RudderResponseFilter:
    """
    Low-pass rudder response model for suppressing high-frequency steering.

    The controller output is a desired rudder angle.  Real steering gear cannot
    jump instantly to every small command, and allowing such jumps in simulation
    often creates visually noisy rudder histories.  This filter approximates a
    command-dependent first-order actuator:

        tau * delta_dot + delta = delta_cmd

    Small command errors use `small_signal_time_constant_s` to suppress chatter.
    Large command errors use `time_constant_s` so real turns remain responsive.
    The filtered command is then rate-limited and clipped to the physical rudder
    envelope before being converted to the simulator action range.
    """

    def __init__(
        self,
        config: RudderResponseConfig,
        max_rudder_angle: float,
        default_max_rudder_rate: float,
    ) -> None:
        self.enabled = bool(config.enabled)
        self.time_constant_s = max(float(config.time_constant_s), 0.0)
        self.small_signal_time_constant_s = max(float(config.small_signal_time_constant_s), self.time_constant_s)
        self.large_signal_threshold = np.deg2rad(float(config.large_signal_threshold_deg))
        configured_rate = config.max_rate_deg_s
        if configured_rate is None:
            self.max_rudder_rate = float(default_max_rudder_rate)
        else:
            self.max_rudder_rate = np.deg2rad(float(configured_rate))
        self.deadband = np.deg2rad(float(config.deadband_deg))
        self.neutral_deadband = np.deg2rad(float(config.neutral_deadband_deg))
        self.max_rudder_angle = float(max_rudder_angle)
        self._rudder_angle = 0.0

    def reset(self, initial_rudder_angle: float = 0.0) -> None:
        self._rudder_angle = float(
            np.clip(initial_rudder_angle, -self.max_rudder_angle, self.max_rudder_angle)
        )

    def step(self, raw_rudder_angle: float, dt: float) -> float:
        raw_rudder_angle = float(
            np.clip(raw_rudder_angle, -self.max_rudder_angle, self.max_rudder_angle)
        )
        if not self.enabled:
            self._rudder_angle = raw_rudder_angle
            return self._rudder_angle

        command_error = raw_rudder_angle - self._rudder_angle
        active_time_constant = (
            self.small_signal_time_constant_s
            if abs(command_error) < self.large_signal_threshold
            else self.time_constant_s
        )

        if active_time_constant <= 1e-9:
            first_order_target = raw_rudder_angle
        else:
            alpha = dt / (active_time_constant + dt)
            first_order_target = self._rudder_angle + alpha * (raw_rudder_angle - self._rudder_angle)

        if (
            self.neutral_deadband > 0.0
            and abs(raw_rudder_angle) < self.neutral_deadband
            and abs(first_order_target) < self.neutral_deadband
        ):
            self._rudder_angle = 0.0
            return self._rudder_angle

        rudder_error = first_order_target - self._rudder_angle
        if abs(rudder_error) < self.deadband:
            rudder_step = 0.0
        else:
            max_step = self.max_rudder_rate * dt
            rudder_step = float(np.clip(rudder_error, -max_step, max_step))

        self._rudder_angle = float(
            np.clip(
                self._rudder_angle + rudder_step,
                -self.max_rudder_angle,
                self.max_rudder_angle,
            )
        )
        return self._rudder_angle


class NomotoFeedforwardPIDHeadingController:
    """
    Heading controller for the LOS+PID baseline with Nomoto feedforward.

    The feedback part avoids differentiating heading error aggressively:

        delta_fb = Kp * e_psi + Ki * integral(e_psi) + Kd * d(e_psi)/dt - Kr * r

    The feedforward part uses a Nomoto yaw model to predict the rudder angle
    needed to create the yaw acceleration required by the desired heading-rate
    command.  The feedforward term is blended with `nomoto_feedforward_gain`,
    so it can help turns without dominating the stabilizing feedback.
    """

    def __init__(
        self,
        specs,
        kp: float,
        ki: float,
        kd: float,
        yaw_rate_damping: float,
        feedforward_enabled: bool,
        feedforward_gain: float,
        desired_yaw_time_constant: float,
        desired_heading_rate_limit_deg_s: float,
        heading_reference_filter_alpha: float,
        integral_limit: float,
    ) -> None:
        self.specs = specs
        self.max_rudder_angle = float(specs.max_rudder_angle)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.yaw_rate_damping = float(yaw_rate_damping)
        self.feedforward_enabled = bool(feedforward_enabled)
        self.feedforward_gain = float(feedforward_gain)
        self.desired_yaw_time_constant = max(float(desired_yaw_time_constant), 1e-6)
        self.desired_heading_rate_limit = np.deg2rad(float(desired_heading_rate_limit_deg_s))
        self.heading_reference_filter_alpha = float(np.clip(heading_reference_filter_alpha, 0.0, 1.0))
        self.integral_limit = float(integral_limit)
        self.nomoto = Nomoto(specs)
        self._integral = 0.0
        self._previous_error = 0.0
        self._filtered_desired_heading: Optional[float] = None
        self.last_desired_heading_rate = 0.0
        self.last_desired_heading_acceleration = 0.0
        self.last_yaw_rate_error = 0.0
        self.last_heading_integral = 0.0
        self.last_heading_error_derivative = 0.0
        self.last_raw_desired_heading = 0.0
        self.last_filtered_desired_heading = 0.0

    def step(
        self,
        state: VesselState,
        raw_desired_heading: float,
        dt: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute rudder angle and expose feedback/feedforward components.

        Returns:
            Tuple `(rudder_angle, feedback_angle, feedforward_angle,
            filtered_desired_heading, heading_error)`.
        """

        previous_filtered_heading = self._filtered_desired_heading
        if previous_filtered_heading is None:
            filtered_desired_heading = raw_desired_heading
            desired_heading_rate = 0.0
            desired_heading_acceleration = 0.0
        else:
            raw_delta = wrap_angle(raw_desired_heading - previous_filtered_heading)
            limited_delta = float(
                np.clip(
                    raw_delta,
                    -self.desired_heading_rate_limit * dt,
                    self.desired_heading_rate_limit * dt,
                )
            )
            filtered_desired_heading = wrap_angle(
                previous_filtered_heading
                + self.heading_reference_filter_alpha * limited_delta
            )
            desired_heading_rate = wrap_angle(filtered_desired_heading - previous_filtered_heading) / dt
            desired_heading_acceleration = 0.0

        self._filtered_desired_heading = filtered_desired_heading
        heading_error = wrap_angle(filtered_desired_heading - state.heading)

        self._integral = float(
            np.clip(
                self._integral + heading_error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        error_derivative = (heading_error - self._previous_error) / dt
        self._previous_error = heading_error

        feedback = (
            self.kp * heading_error
            + self.ki * self._integral
            + self.kd * error_derivative
            - self.yaw_rate_damping * state.r
        )

        feedforward = 0.0
        if self.feedforward_enabled:
            desired_yaw_acceleration = (desired_heading_rate - state.r) / self.desired_yaw_time_constant
            known_nomoto_yaw = (
                self.nomoto.nr * state.r
                + self.nomoto.yv_r * state.v
                + (state.v * state.u) / self.nomoto.l
            )
            feedforward = (desired_yaw_acceleration - known_nomoto_yaw) / self.nomoto.k_r

        rudder_angle = feedback + self.feedforward_gain * feedforward
        rudder_angle = float(np.clip(rudder_angle, -self.max_rudder_angle, self.max_rudder_angle))
        self.last_raw_desired_heading = float(raw_desired_heading)
        self.last_filtered_desired_heading = float(filtered_desired_heading)
        self.last_desired_heading_rate = float(desired_heading_rate)
        self.last_desired_heading_acceleration = float(desired_heading_acceleration)
        self.last_yaw_rate_error = float(state.r - desired_heading_rate)
        self.last_heading_integral = float(self._integral)
        self.last_heading_error_derivative = float(error_derivative)
        return (
            rudder_angle,
            float(feedback),
            float(feedforward),
            float(filtered_desired_heading),
            float(heading_error),
        )


class SlidingModeHeadingController:
    """
    Sliding-mode heading controller based on the second-order Norrbin model.

    The steering model is written as:

        m * psi_ddot + d1 * psi_dot + d3 * psi_dot^3 = delta

    where `psi_dot = r`.  The sliding surface uses tracking error in the
    paper's sign convention:

        e = psi - psi_d
        s = e_dot + lambda * e

    The implemented sliding surface also supports a small integral term:

        s = e_dot + lambda * e + ki * integral(e)

    The integral channel is useful when the closed-loop heading response shows
    a steady residual bias.  It is clipped with an anti-windup limit because
    the Norrbin inverse multiplies yaw-acceleration demand by `m_hat`; even a
    small persistent error can otherwise become a large rudder command.

    A small direct heading-error acceleration term can also be enabled.  This
    compensates for cases where the reduced Norrbin model is too weak compared
    with the full simulation model and the vessel otherwise leaves a large
    heading error while moving between waypoints.  An optional bounded
    `tanh()` boost adds extra authority only for medium/large heading errors;
    near zero heading error it fades out smoothly, so it does not behave like a
    constant rudder bias or create unnecessary small-error chatter.

    A separate bounded direct-rudder feedback can be added after model
    inversion.  This is deliberately outside the Norrbin inverse because the
    identified `d1_hat*r` term can cancel much of the acceleration-based
    command while the vessel is still carrying a visible heading error. A
    leaky integral path can add a slow rudder bias when the heading error keeps
    the same sign over a segment; the leak makes that bias decay automatically
    after the path geometry or error sign changes.

    The virtual yaw acceleration is:

        psi_ddot_cmd = psi_d_ddot - lambda*e_dot - kp_a*e
                        - k_boost*tanh(e / e_boost)
                        - ki*e - eta*sat(s / phi)

    and the rudder command is obtained by Norrbin model inversion:

        delta = m_hat*psi_ddot_cmd + d1_hat*r + d3_hat*r^3.
    """

    def __init__(
        self,
        dynamics: DynamicsBase,
        max_rudder_angle: float,
        ship_length: float,
        params: dict,
        lambda_gain: float,
        eta: float,
        boundary_layer: float,
    ) -> None:
        self.dynamics = dynamics
        self.max_rudder_angle = float(max_rudder_angle)
        self.ship_length = max(float(ship_length), 1e-6)
        self.params = dict(params or {})
        self.lambda_gain = float(lambda_gain)
        self.eta = float(eta)
        self.boundary_layer = max(float(boundary_layer), 1e-6)
        self.heading_error_acceleration_gain = max(
            float(self.params.get("heading_error_acceleration_gain", 0.0)),
            0.0,
        )
        self.large_error_boost_gain = max(
            float(self.params.get("large_error_boost_gain", 0.0)),
            0.0,
        )
        self.large_error_boost_width = max(
            np.deg2rad(float(self.params.get("large_error_boost_width_deg", 15.0))),
            1e-6,
        )
        self.direct_heading_rudder_gain = max(
            np.deg2rad(float(self.params.get("direct_heading_rudder_gain_deg", 0.0))),
            0.0,
        )
        self.direct_heading_rudder_width = max(
            np.deg2rad(float(self.params.get("direct_heading_rudder_width_deg", 12.0))),
            1e-6,
        )
        self.direct_heading_integral_gain = max(
            np.deg2rad(float(self.params.get("direct_heading_integral_gain_deg_per_rad_s", 0.0))),
            0.0,
        )
        self.direct_heading_integral_limit = max(
            float(self.params.get("direct_heading_integral_limit_rad_s", 20.0)),
            0.0,
        )
        self.direct_heading_integral_leak_time = max(
            float(self.params.get("direct_heading_integral_leak_time_s", 60.0)),
            1e-6,
        )
        self.integral_gain = max(float(self.params.get("integral_gain", 0.0)), 0.0)
        self.integral_limit = max(float(self.params.get("integral_limit", 0.20)), 0.0)
        self.desired_heading_rate_limit = np.deg2rad(
            float(self.params.get("desired_heading_rate_limit_deg_s", 5.0))
        )
        self.heading_reference_filter_alpha = float(
            np.clip(self.params.get("heading_reference_filter_alpha", 0.5), 0.0, 1.0)
        )
        self.desired_acceleration_limit = np.deg2rad(
            float(self.params.get("desired_acceleration_limit_deg_s2", 0.4))
        )
        self.use_raw_heading_for_error = bool(self.params.get("use_raw_heading_for_error", False))

        self.use_model_estimates = bool(self.params.get("use_model_estimates", True))
        self.update_model_estimates_each_step = bool(
            self.params.get("update_model_estimates_each_step", False)
        )
        self._manual_m_hat = self.params.get("m_hat", None) is not None
        self._manual_d1_hat = self.params.get("d1_hat", None) is not None
        self.m_hat = FeedbackLinearizationHeadingController._optional_float(self.params.get("m_hat", None))
        self.d1_hat = FeedbackLinearizationHeadingController._optional_float(self.params.get("d1_hat", None))
        self.d3_hat = float(self.params.get("d3_hat", 0.0) or 0.0)
        self.min_abs_m_hat = float(self.params.get("min_abs_m_hat", 1e-3))
        self.max_abs_m_hat = float(self.params.get("max_abs_m_hat", 50000.0))
        self.max_abs_d1_hat = float(self.params.get("max_abs_d1_hat", 50000.0))
        self.max_abs_d3_hat = float(self.params.get("max_abs_d3_hat", 50000.0))
        self.gain_epsilon = float(self.params.get("model_gain_epsilon", 1e-6))
        self.model_probe_rudder = np.deg2rad(float(self.params.get("model_probe_rudder_deg", 1.0)))

        self._filtered_desired_heading: Optional[float] = None
        self._desired_heading_rate = 0.0
        self._paper_error_integral = 0.0
        self._direct_heading_error_integral = 0.0
        self._model_estimates_initialized = self.m_hat is not None and self.d1_hat is not None
        self.last_sliding_variable = 0.0
        self.last_virtual_acceleration = 0.0
        self.last_raw_desired_heading = 0.0
        self.last_filtered_desired_heading = 0.0
        self.last_desired_heading_rate = 0.0
        self.last_desired_heading_acceleration = 0.0
        self.last_yaw_rate_error = 0.0
        self.last_paper_error = 0.0
        self.last_paper_error_rate = 0.0
        self.last_sat_s_over_phi = 0.0
        self.last_cross_track_term = 0.0
        self.last_cross_track_term_rate = 0.0
        self.last_sideslip_angle_beta = 0.0
        self.last_sideslip_compensation_angle = 0.0
        self.last_virtual_course = 0.0
        self.last_norrbin_model_rudder_component = 0.0
        self.last_damping_compensation_component = 0.0
        self.last_direct_heading_rudder_component = 0.0
        self.last_integral_rudder_component = 0.0
        self.last_cross_track_rudder_component = 0.0

    @staticmethod
    def _state_with_yaw_rate(state: VesselState, yaw_rate: float) -> VesselState:
        return VesselState(
            terminated=state.terminated,
            truncated=state.truncated,
            x=state.x,
            y=state.y,
            heading=state.heading,
            u=state.u,
            v=state.v,
            r=float(yaw_rate),
        )

    def _yaw_acceleration(self, state: VesselState, rudder_angle: float, thrust: float, dt: float) -> float:
        _, _, next_yaw_rate = self.dynamics.step(state, rudder_angle, thrust, dt)
        return float((next_yaw_rate - state.r) / dt)

    def _clip_parameter_estimates(self) -> None:
        m_hat = float(1.0 if self.m_hat is None else self.m_hat)
        if abs(m_hat) < self.min_abs_m_hat:
            m_hat = np.copysign(self.min_abs_m_hat, m_hat if m_hat != 0.0 else 1.0)
        self.m_hat = float(np.clip(m_hat, -self.max_abs_m_hat, self.max_abs_m_hat))
        self.d1_hat = float(
            np.clip(0.0 if self.d1_hat is None else self.d1_hat, -self.max_abs_d1_hat, self.max_abs_d1_hat)
        )
        self.d3_hat = float(np.clip(self.d3_hat, -self.max_abs_d3_hat, self.max_abs_d3_hat))

    def _initialize_norrbin_estimates(self, state: VesselState, thrust: float, dt: float) -> None:
        if (
            self._model_estimates_initialized
            and not self.update_model_estimates_each_step
        ) or not self.use_model_estimates:
            self._clip_parameter_estimates()
            self._model_estimates_initialized = True
            return

        delta_probe = max(self.model_probe_rudder, np.deg2rad(0.25))
        delta_probe = min(delta_probe, self.max_rudder_angle)
        r_dot_zero = self._yaw_acceleration(state, 0.0, thrust, dt)
        r_dot_plus = self._yaw_acceleration(state, delta_probe, thrust, dt)
        rudder_gain = (r_dot_plus - r_dot_zero) / delta_probe

        if abs(rudder_gain) >= self.gain_epsilon:
            if self.m_hat is None or (self.update_model_estimates_each_step and not self._manual_m_hat):
                self.m_hat = 1.0 / rudder_gain
            if self.d1_hat is None or (self.update_model_estimates_each_step and not self._manual_d1_hat):
                yaw_rate_probe = max(np.deg2rad(0.5), 0.1 * max(abs(state.r), np.deg2rad(1.0)))
                perturbed_state = self._state_with_yaw_rate(state, state.r + yaw_rate_probe)
                r_dot_perturbed = self._yaw_acceleration(perturbed_state, 0.0, thrust, dt)
                yaw_damping_slope = (r_dot_perturbed - r_dot_zero) / yaw_rate_probe
                self.d1_hat = -yaw_damping_slope / rudder_gain

        if self.m_hat is None:
            self.m_hat = 1.0
        if self.d1_hat is None:
            self.d1_hat = 0.0
        self._clip_parameter_estimates()
        self._model_estimates_initialized = True

    def _smooth_desired_heading(self, raw_desired_heading: float, dt: float) -> Tuple[float, float, float]:
        """
        Smooth the heading reference used for rate/acceleration feed-forward.

        The returned heading is normally the filtered heading.  When
        `use_raw_heading_for_error` is enabled, the returned heading is the raw
        guidance heading, while the returned desired rate and acceleration still
        come from the filtered reference.  This removes most reference-angle
        delay without differentiating waypoint-switching jumps directly.
        """

        previous_heading = self._filtered_desired_heading
        previous_rate = self._desired_heading_rate
        if previous_heading is None:
            filtered_heading = float(raw_desired_heading)
            desired_rate = 0.0
            desired_acceleration = 0.0
        else:
            raw_delta = wrap_angle(raw_desired_heading - previous_heading)
            limited_delta = float(
                np.clip(
                    raw_delta,
                    -self.desired_heading_rate_limit * dt,
                    self.desired_heading_rate_limit * dt,
                )
            )
            filtered_heading = wrap_angle(
                previous_heading + self.heading_reference_filter_alpha * limited_delta
            )
            desired_rate = wrap_angle(filtered_heading - previous_heading) / dt
            desired_acceleration = float(
                np.clip(
                    (desired_rate - previous_rate) / dt,
                    -self.desired_acceleration_limit,
                    self.desired_acceleration_limit,
                )
            )

        self._filtered_desired_heading = filtered_heading
        self._desired_heading_rate = desired_rate
        self.last_raw_desired_heading = float(raw_desired_heading)
        self.last_filtered_desired_heading = float(filtered_heading)
        self.last_desired_heading_rate = float(desired_rate)
        self.last_desired_heading_acceleration = float(desired_acceleration)
        if self.use_raw_heading_for_error:
            return float(raw_desired_heading), desired_rate, desired_acceleration
        return filtered_heading, desired_rate, desired_acceleration

    def _update_heading_integrators(self, heading_error: float, paper_error: float, dt: float) -> None:
        """
        Update the two slow bias-correction channels used by SMC variants.

        `paper_error` follows the steering-paper sign convention
        (psi - psi_d), while `heading_error` follows the simulation diagnostic
        convention (psi_d - psi).  The sliding surface uses `paper_error`; the
        direct rudder bias uses `heading_error`, because a positive diagnostic
        heading error should command a positive steering correction.
        """

        if self.integral_gain > 0.0 and self.integral_limit > 0.0:
            self._paper_error_integral = float(
                np.clip(
                    self._paper_error_integral + paper_error * dt,
                    -self.integral_limit,
                    self.integral_limit,
                )
            )
        else:
            self._paper_error_integral = 0.0

        if self.direct_heading_integral_gain > 0.0 and self.direct_heading_integral_limit > 0.0:
            leak = self._direct_heading_error_integral / self.direct_heading_integral_leak_time
            self._direct_heading_error_integral = float(
                np.clip(
                    self._direct_heading_error_integral + (heading_error - leak) * dt,
                    -self.direct_heading_integral_limit,
                    self.direct_heading_integral_limit,
                )
            )
        else:
            self._direct_heading_error_integral = 0.0

    def _compute_sliding_variable(
        self,
        paper_error: float,
        paper_error_rate: float,
        cross_track_term: float = 0.0,
        cross_track_gain: Optional[float] = None,
    ) -> float:
        """
        Compute the scalar sliding surface shared by heading SMC variants.

        The optional cross-track term is only active in the backstepping
        controller.  Keeping it here avoids repeating the same heading-error
        and integral terms in every SMC implementation.
        """

        if cross_track_gain is None:
            cross_track_gain = 0.0
        return float(
            paper_error_rate
            + self.lambda_gain * paper_error
            + cross_track_gain * cross_track_term
            + self.integral_gain * self._paper_error_integral
        )

    def _compute_virtual_yaw_acceleration(
        self,
        desired_acceleration: float,
        paper_error: float,
        paper_error_rate: float,
        sliding_variable: float,
        cross_track_term_rate: float = 0.0,
        cross_track_gain: Optional[float] = None,
    ) -> float:
        """
        Build the desired yaw acceleration before model inversion.

        The first two terms track the smoothed desired heading reference.  The
        bounded tanh terms make large heading errors produce a clear correction
        without introducing discontinuous chattering near zero error.
        """

        if cross_track_gain is None:
            cross_track_gain = 0.0
        sat_s = float(np.clip(sliding_variable / self.boundary_layer, -1.0, 1.0))
        self.last_sat_s_over_phi = sat_s
        return float(
            desired_acceleration
            - self.lambda_gain * paper_error_rate
            - cross_track_gain * cross_track_term_rate
            - self.heading_error_acceleration_gain * paper_error
            - self.large_error_boost_gain * np.tanh(paper_error / self.large_error_boost_width)
            - self.integral_gain * paper_error
            - self.eta * sat_s
        )

    def _norrbin_inverse_rudder(
        self,
        state: VesselState,
        virtual_acceleration: float,
        heading_error: float,
        include_damping_compensation: bool = True,
        cross_track_term: float = 0.0,
        cross_track_rudder_gain: float = 0.0,
    ) -> float:
        """
        Convert a desired yaw acceleration into a physical rudder angle [rad].

        The Norrbin inverse supplies the model-based part of the command.  The
        additional direct heading and cross-track feedback terms are bounded
        corrections that improve practical tracking when the estimated Norrbin
        parameters do not perfectly match the selected simulator model.
        """

        norrbin_model_component = (
            float(self.m_hat) * virtual_acceleration
            + float(self.d3_hat) * state.r ** 3.0
        )
        damping_component = 0.0
        if include_damping_compensation:
            damping_component = float(self.d1_hat) * state.r

        direct_heading_component = self.direct_heading_rudder_gain * np.tanh(
            heading_error / self.direct_heading_rudder_width
        )
        integral_component = self.direct_heading_integral_gain * self._direct_heading_error_integral

        # Positive cross-track error means the vessel is on the positive side
        # of the path-frame normal, so the steering correction must point back
        # toward the active segment.
        cross_track_component = -cross_track_rudder_gain * cross_track_term
        rudder_angle = (
            norrbin_model_component
            + damping_component
            + direct_heading_component
            + integral_component
            + cross_track_component
        )
        self.last_norrbin_model_rudder_component = float(norrbin_model_component)
        self.last_damping_compensation_component = float(damping_component)
        self.last_direct_heading_rudder_component = float(direct_heading_component)
        self.last_integral_rudder_component = float(integral_component)
        self.last_cross_track_rudder_component = float(cross_track_component)
        return float(np.clip(rudder_angle, -self.max_rudder_angle, self.max_rudder_angle))

    def step(
        self,
        state: VesselState,
        raw_desired_heading: float,
        thrust: float,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        filtered_heading, desired_rate, desired_acceleration = self._smooth_desired_heading(
            raw_desired_heading,
            dt,
        )
        heading_error = wrap_angle(filtered_heading - state.heading)
        paper_error = -heading_error
        paper_error_rate = state.r - desired_rate
        self._update_heading_integrators(heading_error, paper_error, dt)
        sliding_variable = self._compute_sliding_variable(paper_error, paper_error_rate)

        self._initialize_norrbin_estimates(state, thrust, dt)

        virtual_acceleration = self._compute_virtual_yaw_acceleration(
            desired_acceleration,
            paper_error,
            paper_error_rate,
            sliding_variable,
        )
        rudder_angle = self._norrbin_inverse_rudder(
            state,
            virtual_acceleration,
            heading_error,
            include_damping_compensation=True,
        )

        self.last_sliding_variable = float(sliding_variable)
        self.last_virtual_acceleration = float(virtual_acceleration)
        self.last_yaw_rate_error = float(state.r - desired_rate)
        self.last_paper_error = float(paper_error)
        self.last_paper_error_rate = float(paper_error_rate)
        self.last_cross_track_term = 0.0
        self.last_cross_track_term_rate = 0.0
        self.last_sideslip_angle_beta = 0.0
        self.last_sideslip_compensation_angle = 0.0
        self.last_virtual_course = float(raw_desired_heading)
        return rudder_angle, filtered_heading, heading_error, self.last_virtual_acceleration


class BacksteppingDynamicSMCController(SlidingModeHeadingController):
    """
    Backstepping-style dynamic sliding-mode controller for path following.

    Unlike `SlidingModeHeadingController`, which only tracks the heading
    produced by the guidance law, this controller feeds the cross-track error
    into the sliding surface:

        s = e_dot + lambda*e + k_y*tanh(e_y / y_scale)

    where `e = psi - psi_d` follows the paper sign convention and `e_y` is the
    signed path-frame cross-track error.  Depending on config.py, the input
    course can be either the active guidance heading or the geometric path
    tangent.  When `virtual_course_from_cross_track` is enabled, the outer
    backstepping step converts cross-track error into a virtual course before
    optional sideslip compensation:

        chi_v = chi_p + atan2(-e_y, Delta_b)
        psi_d = chi_v - beta, beta = atan2(v, u)

    This keeps the controller closer to a path-following backstepping design:
    the inner yaw controller is not only chasing heading, it also reacts
    directly to persistent path error.
    """

    def __init__(
        self,
        dynamics: DynamicsBase,
        max_rudder_angle: float,
        ship_length: float,
        params: dict,
        lambda_gain: float,
        eta: float,
        boundary_layer: float,
    ) -> None:
        super().__init__(
            dynamics=dynamics,
            max_rudder_angle=max_rudder_angle,
            ship_length=ship_length,
            params=params,
            lambda_gain=lambda_gain,
            eta=eta,
            boundary_layer=boundary_layer,
        )
        self.use_path_tangent_input = bool(self.params.get("use_path_tangent_input", False))
        self.virtual_course_from_cross_track = bool(
            self.params.get("virtual_course_from_cross_track", False)
        )
        self.virtual_lookahead = max(float(self.params.get("virtual_lookahead_m", ship_length)), 1e-6)
        self.cross_track_gain = max(float(self.params.get("cross_track_gain", 0.0)), 0.0)
        self.cross_track_scale = max(float(self.params.get("cross_track_scale_m", ship_length)), 1e-6)
        self.cross_track_term_rate_limit = max(
            float(self.params.get("cross_track_term_rate_limit", 0.02)),
            0.0,
        )
        self.cross_track_rudder_gain = max(
            np.deg2rad(float(self.params.get("cross_track_rudder_gain_deg", 0.0))),
            0.0,
        )
        self.use_damping_compensation = bool(self.params.get("use_damping_compensation", False))
        self.sideslip_compensation_enabled = bool(
            self.params.get("sideslip_compensation_enabled", True)
        )
        self.sideslip_compensation_gain = float(
            np.clip(self.params.get("sideslip_compensation_gain", 1.0), 0.0, 1.0)
        )
        self.sideslip_min_speed = max(float(self.params.get("sideslip_min_speed", 0.5)), 1e-6)
        self.sideslip_limit = np.deg2rad(float(self.params.get("sideslip_limit_deg", 12.0)))
        self._previous_cross_track_term = 0.0
        self._cross_track_term_initialized = False

    def _sideslip_angle(self, state: VesselState) -> float:
        if not self.sideslip_compensation_enabled:
            return 0.0
        beta = np.arctan2(state.v, max(abs(state.u), self.sideslip_min_speed))
        beta = self.sideslip_compensation_gain * beta
        return float(np.clip(beta, -self.sideslip_limit, self.sideslip_limit))

    def step(
        self,
        state: VesselState,
        raw_desired_course: float,
        cross_track_error: float,
        thrust: float,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        beta = self._sideslip_angle(state)
        if self.virtual_course_from_cross_track:
            virtual_course = wrap_angle(
                raw_desired_course + np.arctan2(-cross_track_error, self.virtual_lookahead)
            )
        else:
            virtual_course = float(raw_desired_course)
        raw_desired_heading = wrap_angle(virtual_course - beta)
        filtered_heading, desired_rate, desired_acceleration = self._smooth_desired_heading(
            raw_desired_heading,
            dt,
        )

        heading_error = wrap_angle(filtered_heading - state.heading)
        paper_error = -heading_error
        paper_error_rate = state.r - desired_rate
        cross_track_term = float(np.tanh(cross_track_error / self.cross_track_scale))
        if not self._cross_track_term_initialized:
            cross_track_term_rate = 0.0
            self._cross_track_term_initialized = True
        else:
            cross_track_term_rate = (cross_track_term - self._previous_cross_track_term) / dt
            if self.cross_track_term_rate_limit > 0.0:
                cross_track_term_rate = float(
                    np.clip(
                        cross_track_term_rate,
                        -self.cross_track_term_rate_limit,
                        self.cross_track_term_rate_limit,
                    )
        )
        self._previous_cross_track_term = cross_track_term

        self._update_heading_integrators(heading_error, paper_error, dt)
        sliding_variable = self._compute_sliding_variable(
            paper_error,
            paper_error_rate,
            cross_track_term=cross_track_term,
            cross_track_gain=self.cross_track_gain,
        )

        self._initialize_norrbin_estimates(state, thrust, dt)

        virtual_acceleration = self._compute_virtual_yaw_acceleration(
            desired_acceleration,
            paper_error,
            paper_error_rate,
            sliding_variable,
            cross_track_term_rate=cross_track_term_rate,
            cross_track_gain=self.cross_track_gain,
        )
        rudder_angle = self._norrbin_inverse_rudder(
            state,
            virtual_acceleration,
            heading_error,
            include_damping_compensation=self.use_damping_compensation,
            cross_track_term=cross_track_term,
            cross_track_rudder_gain=self.cross_track_rudder_gain,
        )

        self.last_sliding_variable = float(sliding_variable)
        self.last_virtual_acceleration = float(virtual_acceleration)
        self.last_yaw_rate_error = float(state.r - desired_rate)
        self.last_paper_error = float(paper_error)
        self.last_paper_error_rate = float(paper_error_rate)
        self.last_cross_track_term = float(cross_track_term)
        self.last_cross_track_term_rate = float(cross_track_term_rate)
        self.last_sideslip_angle_beta = float(beta)
        self.last_sideslip_compensation_angle = float(beta)
        self.last_virtual_course = float(virtual_course)
        return rudder_angle, filtered_heading, heading_error, self.last_virtual_acceleration


def _coerce_ship_model(value) -> ShipModel:
    """
    Convert a config value into a ShipModel enum.

    `config.py` normally stores enum members directly, but accepting strings is
    useful for experiments or external configuration layers.
    """

    if isinstance(value, ShipModel):
        return value
    if isinstance(value, str):
        return ShipModel[value]
    raise ValueError(f"Unsupported ship_model value: {value!r}")


def _coerce_dynamics_model(value) -> Model:
    """
    Convert a config value into a dynamics Model enum.
    """

    if isinstance(value, Model):
        return value
    if isinstance(value, str):
        return Model[value]
    raise ValueError(f"Unsupported dynamics_model value: {value!r}")


def load_classical_control_config(framework_config: dict) -> ClassicalControlConfig:
    """
    Extract the framework configuration used by the classical-control baseline.

    This follows the same source of truth as `main.py`: callers pass in the
    result of `get_config_parameters()`, which already includes profile
    overrides.  For example, if the selected profile changes `ship_model`, this
    function sees the changed value.
    """

    return ClassicalControlConfig(
        ship_model=_coerce_ship_model(framework_config.get("ship_model", ShipModel.MYZAKO_DEFAULT)),
        dynamics_model=_coerce_dynamics_model(framework_config.get("dynamics_model", Model.NOMOTO)),
        start_pos=np.array(framework_config.get("start_pos", DEFAULT_FRAMEWORK_SHIP_POS), dtype=float),
        target_pos=np.array(framework_config.get("target_pos", DEFAULT_FRAMEWORK_TARGET_POS), dtype=float),
        overall_contour=str(framework_config.get("overall_contour", "env_Sche_no_scale.csv")),
        depth_contour=str(framework_config.get("depth_contour", "env_Sche_250cm_no_scale.csv")),
        trajectory=str(framework_config.get("trajectory", "trajectory_points_no_scale.csv")),
        wind=bool(framework_config.get("wind", False)),
        current=bool(framework_config.get("current", False)),
        profile=str(framework_config.get("profile", "default_profile")),
    )


def load_path_following_algorithm_config(framework_config: dict) -> PathFollowingAlgorithmConfig:
    """
    Read controller/guidance choices from config.py.

    Missing keys intentionally fall back to the original LOS + PID behavior, so
    existing config.py files continue to run unchanged.
    """

    params = framework_config.get("path_following_controller_parameters", {}) or {}
    guidance_params = framework_config.get("path_following_guidance_parameters", {}) or {}
    return PathFollowingAlgorithmConfig(
        controller=str(framework_config.get("path_following_controller", "pid")),
        guidance=str(framework_config.get("path_following_guidance", "los")),
        pid=dict(params.get("pid", {}) or {}),
        vector_field=dict(
            guidance_params.get("vector_field", params.get("vector_field", {})) or {}
        ),
        feedback_linearization=dict(params.get("feedback_linearization", {}) or {}),
        sliding_mode=dict(params.get("sliding_mode", {}) or {}),
        backstepping_dynamic_smc=dict(params.get("backstepping_dynamic_smc", {}) or {}),
    )


def load_path_simplification_config(framework_config: dict) -> PathSimplificationConfig:
    """
    Read route simplification settings from config.py.
    """

    return PathSimplificationConfig(
        enabled=bool(framework_config.get("path_following_simplify_path", True)),
        tolerance_m=float(framework_config.get("path_following_simplification_tolerance_m", 50.0)),
    )


def load_guidance_parameter_config(framework_config: dict, guidance_name: str) -> GuidanceParameterConfig:
    """
    Read geometry/switching parameters for the selected guidance law.
    """

    params = framework_config.get("path_following_guidance_parameters", {}) or {}
    selected = dict(params.get(guidance_name, {}) or {})
    if guidance_name == "vector_field":
        selected.setdefault("switch_distance", 35.0)
        selected.setdefault("lookahead_distance", None)
        selected.setdefault("lookahead_ship_length_multiplier", 3.0)
    return GuidanceParameterConfig(
        lookahead_distance=selected.get("lookahead_distance", None),
        lookahead_ship_length_multiplier=float(selected.get("lookahead_ship_length_multiplier", 3.0)),
        switch_distance=float(selected.get("switch_distance", 35.0)),
    )


def load_surge_speed_control_config(framework_config: dict) -> SurgeSpeedControlConfig:
    """
    Read surge-speed PID settings from config.py.
    """

    params = framework_config.get("path_following_surge_speed_control", {}) or {}
    return SurgeSpeedControlConfig(
        target_speed=float(params.get("target_speed", 4.0)),
        kp=float(params.get("kp", 0.8)),
        ki=float(params.get("ki", 0.01)),
        kd=float(params.get("kd", 0.0)),
        integral_limit=float(params.get("integral_limit", 20.0)),
        feedforward_thrust=params.get("feedforward_thrust", None),
        adaptive_turn_slowdown_enabled=bool(params.get("adaptive_turn_slowdown_enabled", False)),
        min_turn_speed=float(params.get("min_turn_speed", 3.3)),
        turn_slowdown_heading_error_deg=float(params.get("turn_slowdown_heading_error_deg", 15.0)),
        turn_slowdown_max_reduction=float(params.get("turn_slowdown_max_reduction", 0.20)),
    )


def load_controller_output_limit_config(framework_config: dict) -> ControllerOutputLimitConfig:
    """
    Read controller output limits from config.py.
    """

    params = framework_config.get("path_following_output_limits", {}) or {}
    return ControllerOutputLimitConfig(
        rudder_angle_limit_deg=params.get("rudder_angle_limit_deg", None),
        min_thrust=float(params.get("min_thrust", 0.0)),
        max_thrust=params.get("max_thrust", None),
    )


def load_rudder_response_config(framework_config: dict) -> RudderResponseConfig:
    """
    Read rudder actuator-response filter settings from config.py.
    """

    params = framework_config.get("path_following_rudder_response", {}) or {}
    return RudderResponseConfig(
        enabled=bool(params.get("enabled", True)),
        time_constant_s=float(params.get("time_constant_s", 0.15)),
        small_signal_time_constant_s=float(params.get("small_signal_time_constant_s", 1.0)),
        large_signal_threshold_deg=float(params.get("large_signal_threshold_deg", 1.5)),
        max_rate_deg_s=params.get("max_rate_deg_s", None),
        deadband_deg=float(params.get("deadband_deg", 0.02)),
        neutral_deadband_deg=float(params.get("neutral_deadband_deg", 0.20)),
    )


def build_dynamics_model(model: Model, specs) -> DynamicsBase:
    """
    Create the configured pure-Python dynamics model.

    The standalone LOS/PID baseline is intentionally limited to dynamics models
    that can run directly inside this Python process.  Engine-backed models
    such as FH_TABULAR need additional bridge setup and are better handled by
    the main framework entry points.
    """

    if model == Model.NOMOTO:
        return Nomoto(specs)
    if model == Model.FOSSEN:
        return Fossen(specs)
    if model == Model.UGENT_SURROGATE:
        return UGentSurrogate(specs)

    raise NotImplementedError(
        f"Classical LOS/PID baseline does not support dynamics_model={model.name}. "
        "Use Model.NOMOTO, Model.FOSSEN, or Model.UGENT_SURROGATE for this script."
    )


def load_xy_csv(path: Path) -> np.ndarray:
    """
    Load a two-column x/y CSV file as a float array.

    The environment map files follow the convention used by PySimEnv:
    one header row followed by x,y coordinate rows.
    """

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{path} must contain x,y columns.")
    return data.astype(float)


def load_map_data(
    env_dir: Path = DEFAULT_ENV_DIR,
    overall_file: str = "env_Sche_no_scale.csv",
    obstacles_file: str = "env_Sche_250cm_no_scale.csv",
    path_file: str = "trajectory_points_no_scale.csv",
) -> MapData:
    """
    Load the map contours and predefined path used by the main PySimEnv setup.
    """

    return MapData(
        overall=load_xy_csv(env_dir / overall_file),
        obstacles=load_xy_csv(env_dir / obstacles_file),
        reference_path=load_xy_csv(env_dir / path_file),
    )


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """
    Distance from one point to a finite line segment.

    This is the geometric primitive used by Ramer-Douglas-Peucker path
    simplification.  Degenerate zero-length segments fall back to point-point
    distance so duplicate path points do not crash simplification.
    """

    segment = end - start
    segment_length_sq = float(np.dot(segment, segment))
    if segment_length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))

    projection = float(np.dot(point - start, segment) / segment_length_sq)
    projection = float(np.clip(projection, 0.0, 1.0))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def simplify_path_rdp(points: np.ndarray, tolerance_m: float) -> np.ndarray:
    """
    Simplify a polyline with the Ramer-Douglas-Peucker algorithm.

    The first and last points are always preserved.  Intermediate points are
    recursively removed when the original path between two retained endpoints
    stays within `tolerance_m` of the straight segment connecting them.
    """

    points = np.asarray(points, dtype=float)
    if points.shape[0] <= 2 or tolerance_m <= 0.0:
        return points.copy()

    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, points.shape[0] - 1)]

    while stack:
        start_idx, end_idx = stack.pop()
        if end_idx <= start_idx + 1:
            continue

        start = points[start_idx]
        end = points[end_idx]
        max_distance = -1.0
        max_index = start_idx

        for idx in range(start_idx + 1, end_idx):
            distance = _point_to_segment_distance(points[idx], start, end)
            if distance > max_distance:
                max_distance = distance
                max_index = idx

        if max_distance > tolerance_m:
            keep[max_index] = True
            stack.append((start_idx, max_index))
            stack.append((max_index, end_idx))

    return points[keep]


def remove_consecutive_duplicate_points(points: np.ndarray, min_distance: float = 1e-6) -> np.ndarray:
    """
    Remove consecutive duplicate or near-duplicate path points.

    Duplicate points create zero-length path segments, which can make guidance
    switch rapidly or raise a degenerate-segment error.
    """

    points = np.asarray(points, dtype=float)
    if points.shape[0] <= 1:
        return points.copy()

    kept = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - kept[-1])) > min_distance:
            kept.append(point)
    return np.asarray(kept, dtype=float)


def path_to_waypoints(
    reference_path: np.ndarray,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    simplification_config: Optional[PathSimplificationConfig] = None,
) -> List[Point]:
    """
    Convert the map trajectory into LOS waypoints.

    This mirrors the route reduction used by PySimEnv: find the closest point
    to the framework start position, find the closest point to the target, and
    keep the path segment between them.  The final waypoint is forced to the
    configured target, while the first waypoint is the actual initial vessel
    position.  This avoids artificial backtracking at the beginning of the run.
    """

    start = np.array(start_pos, dtype=float)
    target = np.array(target_pos, dtype=float)

    start_index = int(np.argmin(np.linalg.norm(reference_path - start, axis=1)))
    target_index = int(np.argmin(np.linalg.norm(reference_path - target, axis=1)))
    if start_index == target_index:
        raise ValueError("Start and target resolve to the same reference-path point.")

    if target_index > start_index:
        reduced_path = reference_path[start_index + 1: target_index + 1].copy()
    else:
        reduced_path = reference_path[target_index:start_index][::-1].copy()

    if len(reduced_path) == 0:
        raise ValueError("Reduced reference path is empty.")

    reduced_path[-1] = target
    path = np.vstack([start, reduced_path])
    path = remove_consecutive_duplicate_points(path)

    if simplification_config is not None and simplification_config.enabled:
        path = simplify_path_rdp(path, simplification_config.tolerance_m)
        path[0] = start
        path[-1] = target

    return [(float(x), float(y)) for x, y in path]


def initial_heading_from_path(waypoints: Sequence[Point]) -> float:
    """
    Initialize the vessel heading along the first path segment.

    Starting aligned with the predefined route avoids an artificial large
    transient that would otherwise dominate the early path-following metrics.
    """

    if len(waypoints) < 2:
        return 0.0
    p0 = np.array(waypoints[0], dtype=float)
    p1 = np.array(waypoints[1], dtype=float)
    delta = p1 - p0
    if np.linalg.norm(delta) <= 1e-9:
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def active_segment_heading(waypoints: Sequence[Point], segment_index: int) -> float:
    """
    Return the tangent heading of the currently active waypoint segment.

    Backstepping path-following control needs the path tangent itself, not the
    LOS/vector-field heading that already contains cross-track correction.
    """

    if len(waypoints) < 2:
        return 0.0
    index = int(np.clip(segment_index, 0, len(waypoints) - 2))
    p0 = np.array(waypoints[index], dtype=float)
    p1 = np.array(waypoints[index + 1], dtype=float)
    delta = p1 - p0
    if np.linalg.norm(delta) <= 1e-9:
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def collect_controller_diagnostics(controller: object) -> dict:
    """
    Read optional controller-internal diagnostics in a uniform schema.

    Different heading controllers expose different internal states.  This helper
    keeps `run_simulation()` simple by returning the same dictionary keys for
    PID, feedback linearization, sliding mode, and backstepping SMC.  Missing
    quantities are recorded as None in the exported CSV.
    """

    return {
        "raw_controller_desired_heading": getattr(controller, "last_raw_desired_heading", None),
        "filtered_controller_desired_heading": getattr(controller, "last_filtered_desired_heading", None),
        "controller_desired_heading_rate": getattr(controller, "last_desired_heading_rate", None),
        "controller_desired_heading_acceleration": getattr(
            controller,
            "last_desired_heading_acceleration",
            None,
        ),
        "yaw_rate_error": getattr(controller, "last_yaw_rate_error", None),
        "paper_error": getattr(controller, "last_paper_error", None),
        "paper_error_rate": getattr(controller, "last_paper_error_rate", None),
        "sliding_surface_s": getattr(controller, "last_sliding_variable", None),
        "sat_s_over_phi": getattr(controller, "last_sat_s_over_phi", None),
        "boundary_layer_phi": getattr(controller, "boundary_layer", None),
        "sideslip_angle_beta": getattr(controller, "last_sideslip_angle_beta", None),
        "sideslip_compensation_angle": getattr(controller, "last_sideslip_compensation_angle", None),
        "virtual_course": getattr(controller, "last_virtual_course", None),
        "cross_track_term": getattr(controller, "last_cross_track_term", None),
        "cross_track_term_rate": getattr(controller, "last_cross_track_term_rate", None),
        "paper_error_integral": getattr(controller, "_paper_error_integral", None),
        "direct_heading_error_integral": getattr(controller, "_direct_heading_error_integral", None),
        "heading_integral": getattr(controller, "last_heading_integral", None),
        "heading_error_derivative": getattr(controller, "last_heading_error_derivative", None),
        "norrbin_model_rudder_component": getattr(controller, "last_norrbin_model_rudder_component", None),
        "damping_compensation_component": getattr(controller, "last_damping_compensation_component", None),
        "direct_heading_rudder_component": getattr(controller, "last_direct_heading_rudder_component", None),
        "integral_rudder_component": getattr(controller, "last_integral_rudder_component", None),
        "cross_track_rudder_component": getattr(controller, "last_cross_track_rudder_component", None),
    }


def run_simulation(
    control_config: ClassicalControlConfig,
    algorithm_config: PathFollowingAlgorithmConfig,
    surge_config: SurgeSpeedControlConfig,
    output_limit_config: ControllerOutputLimitConfig,
    rudder_response_config: RudderResponseConfig,
    waypoints: Sequence[Point],
    duration: float,
    dt: float,
    lookahead_distance: float,
    switch_distance: float,
    heading_pid_gains: Tuple[float, float, float],
) -> List[dict]:
    """
    Run the closed-loop guidance and heading-control simulation.

    Control architecture:
        1. The selected guidance law computes desired heading and cross-track
           error.
        2. The selected heading controller computes rudder command.
        3. SurgeSpeedPIDController converts speed error into a thrust command.
        4. ShipSimulator advances the configured dynamics model using
           rudder/thrust.

    The returned list contains a full time series of navigation, control, and
    vessel-state variables.  It is intentionally plain dictionaries so that the
    same data can be exported directly to CSV without an extra dependency.
    """

    ship = create_ship(control_config.ship_model)
    specs = ship.specifications

    # Apply controller output limits before the simulator is constructed so the
    # actuator mapping and controller saturation use the same physical ranges.
    if output_limit_config.rudder_angle_limit_deg is not None:
        specs.max_rudder_angle = np.deg2rad(float(output_limit_config.rudder_angle_limit_deg))
    specs.min_thrust = float(output_limit_config.min_thrust)
    if output_limit_config.max_thrust is not None:
        specs.max_thrust = float(output_limit_config.max_thrust)

    dynamics = build_dynamics_model(control_config.dynamics_model, specs)
    simulator = ShipSimulator(
        specs,
        dynamics,
        dt=dt,
        wind=control_config.wind,
        current=control_config.current,
    )
    simulator.reset(
        VesselState(
            terminated=False,
            truncated=False,
            x=waypoints[0][0],
            y=waypoints[0][1],
            heading=initial_heading_from_path(waypoints),
            u=0.0,
            v=0.0,
            r=0.0,
        )
    )

    if algorithm_config.guidance == "los":
        guidance = LOSGuidance(
            waypoints,
            lookahead_distance=lookahead_distance,
            switch_distance=switch_distance,
        )
    elif algorithm_config.guidance == "pure_pursuit":
        guidance = PurePursuitGuidance(
            waypoints,
            lookahead_distance=lookahead_distance,
            switch_distance=switch_distance,
        )
    elif algorithm_config.guidance == "vector_field":
        vf_params = algorithm_config.vector_field
        vf_delta = vf_params.get("delta", None)
        if vf_delta is None:
            vf_delta = lookahead_distance
        theta_error_scale = vf_params.get("theta_error_scale", None)
        guidance = VectorFieldGuidance(
            waypoints,
            delta=float(vf_delta),
            switch_distance=switch_distance,
            theta_base=float(vf_params.get("theta_base", 1.0)),
            theta_gain=float(vf_params.get("theta_gain", 0.2)),
            theta_error_scale=None if theta_error_scale is None else float(theta_error_scale),
            theta_max=float(vf_params.get("theta_max", 2.5)),
        )
    else:
        raise ValueError(
            f"Unsupported path_following_guidance={algorithm_config.guidance!r}. "
            "Use 'los', 'pure_pursuit', or 'vector_field'."
        )
    pid_params = algorithm_config.pid
    heading_pid = NomotoFeedforwardPIDHeadingController(
        specs=specs,
        kp=heading_pid_gains[0],
        ki=heading_pid_gains[1],
        kd=heading_pid_gains[2],
        yaw_rate_damping=float(pid_params.get("yaw_rate_damping", 10.0)),
        feedforward_enabled=bool(pid_params.get("nomoto_feedforward_enabled", True)),
        feedforward_gain=float(pid_params.get("nomoto_feedforward_gain", 0.05)),
        desired_yaw_time_constant=float(pid_params.get("desired_yaw_time_constant", 10.0)),
        desired_heading_rate_limit_deg_s=float(pid_params.get("desired_heading_rate_limit_deg_s", 5.0)),
        heading_reference_filter_alpha=float(pid_params.get("heading_reference_filter_alpha", 0.6)),
        integral_limit=float(pid_params.get("integral_limit", 1.0)),
    )
    surge_pid = SurgeSpeedPIDController(
        kp=surge_config.kp,
        ki=surge_config.ki,
        kd=surge_config.kd,
        target_speed=surge_config.target_speed,
        min_thrust=specs.min_thrust,
        max_thrust=specs.max_thrust,
        integral_limit=surge_config.integral_limit,
        feedforward_thrust=surge_config.feedforward_thrust,
    )
    fl_params = algorithm_config.feedback_linearization
    feedback_linearization = FeedbackLinearizationHeadingController(
        dynamics=dynamics,
        max_rudder_angle=specs.max_rudder_angle,
        ship_length=specs.length,
        params=fl_params,
        kp=fl_params.get("heading_kp", None),
        kd=fl_params.get("heading_kd", None),
        gain_epsilon=float(fl_params.get("model_gain_epsilon", 1e-6)),
    )
    smc_params = algorithm_config.sliding_mode
    sliding_mode = SlidingModeHeadingController(
        dynamics=dynamics,
        max_rudder_angle=specs.max_rudder_angle,
        ship_length=specs.length,
        params=smc_params,
        lambda_gain=float(smc_params.get("lambda", 0.12)),
        eta=float(smc_params.get("eta", 0.00002)),
        boundary_layer=float(smc_params.get("boundary_layer", 0.03)),
    )
    backstepping_params = algorithm_config.backstepping_dynamic_smc
    backstepping_dynamic_smc = BacksteppingDynamicSMCController(
        dynamics=dynamics,
        max_rudder_angle=specs.max_rudder_angle,
        ship_length=specs.length,
        params=backstepping_params,
        lambda_gain=float(backstepping_params.get("lambda", 0.30)),
        eta=float(backstepping_params.get("eta", 0.00010)),
        boundary_layer=float(backstepping_params.get("boundary_layer", 0.025)),
    )
    rudder_response = RudderResponseFilter(
        config=rudder_response_config,
        max_rudder_angle=specs.max_rudder_angle,
        default_max_rudder_rate=specs.max_rudder_rate,
    )

    samples: List[dict] = []
    n_steps = int(duration / dt)
    previous_rudder_angle: Optional[float] = None
    previous_raw_rudder_angle: Optional[float] = None
    previous_thrust: Optional[float] = None

    for step_idx in range(n_steps + 1):
        state = simulator.state

        # Guidance layer: convert vessel position relative to the active path
        # segment into desired heading and signed cross-track error.
        desired_heading, cross_track_error, los_point = guidance.step((state.x, state.y))
        guidance_desired_heading = float(desired_heading)
        path_heading = active_segment_heading(waypoints, guidance.segment_index)
        heading_error = wrap_angle(desired_heading - state.heading)

        # Speed layer: a second PID autopilot regulates surge velocity by
        # commanding thrust.  The resulting thrust command is converted into the
        # normalized action range used by ShipSimulator.
        if surge_config.adaptive_turn_slowdown_enabled:
            heading_error_scale = max(np.deg2rad(surge_config.turn_slowdown_heading_error_deg), 1e-6)
            slowdown = surge_config.turn_slowdown_max_reduction * np.tanh(
                abs(heading_error) / heading_error_scale
            )
            dynamic_target_speed = surge_config.target_speed * (1.0 - slowdown)
            dynamic_target_speed = max(surge_config.min_turn_speed, dynamic_target_speed)
            dynamic_target_speed = min(surge_config.target_speed, dynamic_target_speed)
            surge_pid.target_speed = float(dynamic_target_speed)
        else:
            surge_pid.target_speed = float(surge_config.target_speed)
        thrust, speed_error = surge_pid.step(state.u, dt)

        # Heading control layer. PID is the original baseline. Feedback
        # linearization locally inverts the configured simulation model's yaw
        # acceleration dynamics. Sliding mode uses the Nomoto yaw equation.
        if algorithm_config.controller == "pid":
            (
                rudder_angle,
                rudder_feedback,
                rudder_feedforward,
                desired_heading,
                heading_error,
            ) = heading_pid.step(
                state,
                desired_heading,
                dt,
            )
            fl_virtual_acceleration = 0.0
            active_heading_controller = heading_pid
        elif algorithm_config.controller == "feedback_linearization":
            (
                rudder_angle,
                desired_heading,
                heading_error,
                fl_virtual_acceleration,
            ) = feedback_linearization.step(state, desired_heading, thrust, dt)
            rudder_feedback = rudder_angle
            rudder_feedforward = 0.0
            active_heading_controller = feedback_linearization
        elif algorithm_config.controller == "sliding_mode":
            (
                rudder_angle,
                desired_heading,
                heading_error,
                fl_virtual_acceleration,
            ) = sliding_mode.step(state, desired_heading, thrust, dt)
            rudder_feedback = rudder_angle
            rudder_feedforward = 0.0
            diagnostic_sliding_variable = sliding_mode.last_sliding_variable
            diagnostic_m_hat = sliding_mode.m_hat
            diagnostic_d1_hat = sliding_mode.d1_hat
            diagnostic_d3_hat = sliding_mode.d3_hat
            active_heading_controller = sliding_mode
        elif algorithm_config.controller == "backstepping_dynamic_smc":
            if backstepping_dynamic_smc.use_path_tangent_input:
                controller_course = active_segment_heading(waypoints, guidance.segment_index)
            else:
                controller_course = desired_heading
            (
                rudder_angle,
                desired_heading,
                heading_error,
                fl_virtual_acceleration,
            ) = backstepping_dynamic_smc.step(state, controller_course, cross_track_error, thrust, dt)
            rudder_feedback = rudder_angle
            rudder_feedforward = 0.0
            diagnostic_sliding_variable = backstepping_dynamic_smc.last_sliding_variable
            diagnostic_m_hat = backstepping_dynamic_smc.m_hat
            diagnostic_d1_hat = backstepping_dynamic_smc.d1_hat
            diagnostic_d3_hat = backstepping_dynamic_smc.d3_hat
            active_heading_controller = backstepping_dynamic_smc
        else:
            raise ValueError(
                f"Unsupported path_following_controller={algorithm_config.controller!r}. "
                "Use 'pid', 'feedback_linearization', 'sliding_mode', or 'backstepping_dynamic_smc'."
            )
        if algorithm_config.controller == "pid":
            fl_virtual_acceleration = 0.0
            diagnostic_sliding_variable = 0.0
            diagnostic_m_hat = None
            diagnostic_d1_hat = None
            diagnostic_d3_hat = None
        elif algorithm_config.controller == "feedback_linearization":
            diagnostic_sliding_variable = feedback_linearization.last_sliding_variable
            diagnostic_m_hat = feedback_linearization.m_hat
            diagnostic_d1_hat = feedback_linearization.d1_hat
            diagnostic_d3_hat = feedback_linearization.d3_hat

        raw_rudder_angle = float(rudder_angle)
        rudder_angle = rudder_response.step(raw_rudder_angle, dt)
        if previous_rudder_angle is None:
            rudder_rate = 0.0
            raw_rudder_rate = 0.0
            thrust_rate = 0.0
        else:
            rudder_rate = (rudder_angle - previous_rudder_angle) / dt
            raw_rudder_rate = (raw_rudder_angle - previous_raw_rudder_angle) / dt
            thrust_rate = (thrust - previous_thrust) / dt
        rudder_action = float(np.clip(rudder_angle / specs.max_rudder_angle, -1.0, 1.0))
        thrust_action = thrust_to_normalized(thrust, specs.min_thrust, specs.max_thrust)
        controller_diagnostics = collect_controller_diagnostics(active_heading_controller)

        samples.append(
            {
                "time": step_idx * dt,
                "controller": algorithm_config.controller,
                "guidance": algorithm_config.guidance,
                "x": state.x,
                "y": state.y,
                "heading": state.heading,
                "path_heading": path_heading,
                "guidance_desired_heading": guidance_desired_heading,
                "controller_desired_heading": desired_heading,
                "desired_heading": desired_heading,
                "heading_error": heading_error,
                "heading_error_deg": float(np.degrees(heading_error)),
                "cross_track_error": cross_track_error,
                "cross_track_error_abs": abs(cross_track_error),
                "rudder_angle": rudder_angle,
                "raw_rudder_angle": raw_rudder_angle,
                "rudder_feedback_angle": rudder_feedback,
                "rudder_feedforward_angle": rudder_feedforward,
                "rudder_action": rudder_action,
                "rudder_rate": rudder_rate,
                "raw_rudder_rate": raw_rudder_rate,
                "rudder_rate_deg_s": float(np.degrees(rudder_rate)),
                "raw_rudder_rate_deg_s": float(np.degrees(raw_rudder_rate)),
                "fl_virtual_acceleration": fl_virtual_acceleration,
                "fl_sliding_variable": diagnostic_sliding_variable,
                "fl_m_hat": diagnostic_m_hat,
                "fl_d1_hat": diagnostic_d1_hat,
                "fl_d3_hat": diagnostic_d3_hat,
                "target_speed": surge_pid.target_speed,
                "speed_error": speed_error,
                "thrust": thrust,
                "thrust_action": thrust_action,
                "thrust_rate": thrust_rate,
                "surge_pid_integral": surge_pid.last_integral,
                "surge_pid_derivative": surge_pid.last_derivative,
                "surge_pid_feedback": surge_pid.last_feedback,
                "surge_pid_feedforward": surge_pid.last_feedforward,
                "u": state.u,
                "v": state.v,
                "r": state.r,
                "segment_index": guidance.segment_index,
                "los_x": los_point[0],
                "los_y": los_point[1],
                **controller_diagnostics,
            }
        )
        previous_rudder_angle = rudder_angle
        previous_raw_rudder_angle = raw_rudder_angle
        previous_thrust = thrust

        # Stop once the final waypoint is reached.  This keeps the post-run
        # metrics focused on path following, not on sailing beyond the target.
        if guidance.reached_goal((state.x, state.y)):
            break

        simulator.step(np.array([rudder_action, thrust_action], dtype=np.float32))

    return samples


def csv_header_with_unit(field_name: str) -> str:
    """
    Format exported CSV headers as `name [unit]` when a unit is defined.

    Internal sample dictionaries keep plain variable names because plotting and
    metric functions use those keys directly.  The unit annotation is applied
    only at the file boundary, so the code remains stable while the raw exported
    data is self-describing.
    """

    unit = CSV_FIELD_UNITS.get(field_name)
    if not unit:
        return field_name
    return f"{field_name} [{unit}]"


def write_csv(samples: Iterable[dict], path: Path) -> None:
    """
    Write the complete time-series simulation data to CSV.

    This file includes trajectory, guidance errors, control commands, and vessel
    state variables, so it is the primary data export for offline analysis.
    """

    rows = list(samples)
    if not rows:
        raise ValueError("No simulation samples to write.")

    fieldnames = list(rows[0].keys())
    export_fieldnames = [csv_header_with_unit(field) for field in fieldnames]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=export_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    csv_header_with_unit(field): row.get(field, "")
                    for field in fieldnames
                }
            )


def write_trajectory_csv(samples: Sequence[dict], path: Path) -> None:
    """
    Write a compact x/y trajectory CSV for quick plotting or GIS-style import.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        fieldnames = ["time", "x", "y"]
        export_fieldnames = [csv_header_with_unit(field) for field in fieldnames]
        writer = csv.DictWriter(csv_file, fieldnames=export_fieldnames)
        writer.writeheader()
        for row in samples:
            writer.writerow(
                {
                    csv_header_with_unit("time"): row["time"],
                    csv_header_with_unit("x"): row["x"],
                    csv_header_with_unit("y"): row["y"],
                }
            )


def compute_performance_metrics(
    samples: Sequence[dict],
    waypoints: Sequence[Point],
    control_config: ClassicalControlConfig,
) -> dict:
    """
    Compute scalar path-following performance metrics.

    The metrics are deliberately simple and interpretable:
        - signed final errors show terminal behavior,
        - mean/RMS/max absolute errors summarize tracking quality,
        - rudder statistics summarize control effort,
        - final target distance reports whether the route was completed.
    """

    if not samples:
        raise ValueError("No simulation samples available for metrics.")

    # Convert list-of-dict samples into NumPy arrays so vectorized statistics
    # can be computed consistently and without manual loops.
    times = np.array([row["time"] for row in samples], dtype=float)
    cross_track = np.array([row["cross_track_error"] for row in samples], dtype=float)
    heading_error = np.array([row["heading_error"] for row in samples], dtype=float)
    rudder = np.array([row["rudder_angle"] for row in samples], dtype=float)
    raw_rudder = np.array([row.get("raw_rudder_angle", row["rudder_angle"]) for row in samples], dtype=float)
    rudder_action = np.array([row["rudder_action"] for row in samples], dtype=float)
    speed = np.array([row["u"] for row in samples], dtype=float)
    target_speed = np.array([row["target_speed"] for row in samples], dtype=float)
    speed_error = np.array([row["speed_error"] for row in samples], dtype=float)
    thrust = np.array([row["thrust"] for row in samples], dtype=float)

    # The last sample is the terminal state used for final-position and final
    # error reporting.
    final = samples[-1]
    target = np.array(waypoints[-1], dtype=float)
    final_pos = np.array([final["x"], final["y"]], dtype=float)

    # RMS penalizes occasional large tracking errors more strongly than the
    # mean absolute value.  Keeping both metrics makes it easier to distinguish
    # between a controller with small persistent bias and one with rare spikes.
    dt_series = np.diff(times)
    valid_dt = dt_series > 1e-12
    rudder_rate = np.diff(rudder)[valid_dt] / dt_series[valid_dt] if np.any(valid_dt) else np.array([0.0])
    raw_rudder_rate = (
        np.diff(raw_rudder)[valid_dt] / dt_series[valid_dt] if np.any(valid_dt) else np.array([0.0])
    )
    return {
        "profile": control_config.profile,
        "ship_model": control_config.ship_model.name,
        "dynamics_model": control_config.dynamics_model.name,
        "wind_enabled": control_config.wind,
        "current_enabled": control_config.current,
        "duration_s": float(times[-1] - times[0]),
        "num_samples": int(len(samples)),
        "final_x_m": float(final["x"]),
        "final_y_m": float(final["y"]),
        "target_x_m": float(target[0]),
        "target_y_m": float(target[1]),
        "final_target_distance_m": float(np.linalg.norm(final_pos - target)),
        "final_cross_track_error_m": float(final["cross_track_error"]),
        "mean_abs_cross_track_error_m": float(np.mean(np.abs(cross_track))),
        "rms_cross_track_error_m": float(np.sqrt(np.mean(cross_track ** 2))),
        "max_abs_cross_track_error_m": float(np.max(np.abs(cross_track))),
        "final_heading_error_deg": float(np.degrees(final["heading_error"])),
        "mean_abs_heading_error_deg": float(np.degrees(np.mean(np.abs(heading_error)))),
        "rms_heading_error_deg": float(np.degrees(np.sqrt(np.mean(heading_error ** 2)))),
        "max_abs_heading_error_deg": float(np.degrees(np.max(np.abs(heading_error)))),
        "mean_abs_rudder_deg": float(np.degrees(np.mean(np.abs(rudder)))),
        "max_abs_rudder_deg": float(np.degrees(np.max(np.abs(rudder)))),
        "mean_abs_raw_rudder_deg": float(np.degrees(np.mean(np.abs(raw_rudder)))),
        "max_abs_raw_rudder_deg": float(np.degrees(np.max(np.abs(raw_rudder)))),
        "mean_abs_rudder_rate_deg_s": float(np.degrees(np.mean(np.abs(rudder_rate)))),
        "max_abs_rudder_rate_deg_s": float(np.degrees(np.max(np.abs(rudder_rate)))),
        "mean_abs_raw_rudder_rate_deg_s": float(np.degrees(np.mean(np.abs(raw_rudder_rate)))),
        "max_abs_raw_rudder_rate_deg_s": float(np.degrees(np.max(np.abs(raw_rudder_rate)))),
        "rudder_saturation_ratio": float(np.mean(np.abs(rudder_action) >= 0.999)),
        "max_abs_rudder_action": float(np.max(np.abs(rudder_action))),
        "target_surge_velocity_mps": float(target_speed[-1]),
        "final_surge_velocity_mps": float(speed[-1]),
        "final_speed_error_mps": float(speed_error[-1]),
        "mean_abs_speed_error_mps": float(np.mean(np.abs(speed_error))),
        "rms_speed_error_mps": float(np.sqrt(np.mean(speed_error ** 2))),
        "max_abs_speed_error_mps": float(np.max(np.abs(speed_error))),
        "mean_surge_velocity_mps": float(np.mean(speed)),
        "max_surge_velocity_mps": float(np.max(speed)),
        "mean_thrust_command": float(np.mean(thrust)),
        "max_thrust_command": float(np.max(thrust)),
    }


def write_metrics_summary(metrics: dict, csv_path: Path, json_path: Path) -> None:
    """
    Save scalar performance metrics in both machine-readable CSV and JSON.
    """

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})

    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_resolved_config(
    framework_config: dict,
    control_config: ClassicalControlConfig,
    controller_config: dict,
    path: Path,
) -> None:
    """
    Save the effective configuration used for this run.

    This mirrors the traceability provided by `main.py` run bundling: the output
    directory contains both the framework config values read from config.py and
    the classical-controller parameters supplied through this script.
    """

    payload = {
        "framework_config": framework_config,
        "classical_control_config": {
            "ship_model": control_config.ship_model.name,
            "dynamics_model": control_config.dynamics_model.name,
            "start_pos": control_config.start_pos.tolist(),
            "target_pos": control_config.target_pos.tolist(),
            "overall_contour": control_config.overall_contour,
            "depth_contour": control_config.depth_contour,
            "trajectory": control_config.trajectory,
            "wind": control_config.wind,
            "current": control_config.current,
            "profile": control_config.profile,
        },
        "controller_config": controller_config,
    }

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_control_parameters(controller_config: dict, path: Path) -> None:
    """
    Save the controller and guidance parameters in a compact JSON file.

    `resolved_config.json` intentionally stores the broader framework state.
    This companion file is focused on experiment reproducibility for the
    path-following controller: selected controller, selected guidance law,
    guidance parameters, heading-controller gains, surge controller gains,
    output limits, and rudder-response settings.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(controller_config, indent=2, default=str), encoding="utf-8")


def create_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"los_pid_path_following_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _json_safe(value):
    """
    Convert framework metrics/config objects into JSON-safe primitives.

    Some framework metric dictionaries use tuple keys for confusion matrices.
    JSON requires string keys, so this helper recursively stringifies keys and
    converts NumPy scalar/array values into native Python objects.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_framework_episode_data(run_data, csv_path: Path) -> None:
    """
    Save EpisodeCollector output as a flat CSV time series.

    This file is the data equivalent of `image_trajectory.png` and
    `image_metrics.png`: every row corresponds to one recorded timestep from
    the same rollout used by the main framework post-processing.
    """

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ship_id",
        "episode",
        "step",
        "x_m",
        "y_m",
        "cross_track_error_m",
        "heading_error_rad",
        "surge_velocity_mps",
        "rudder_action",
        "rudder_change",
        "thrust_action",
        "thrust_change",
        "ukc_m",
        "episode_reward",
        "collision",
        "collision_type",
        "collision_side",
        "colreg_violations",
    ]

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for ship_id, episodes in run_data.ships.items():
            for episode in episodes:
                length = int(episode.path.shape[0])
                for step in range(length):
                    writer.writerow(
                        {
                            "ship_id": ship_id,
                            "episode": episode.episode,
                            "step": step,
                            "x_m": float(episode.path[step, 0]),
                            "y_m": float(episode.path[step, 1]),
                            "cross_track_error_m": _series_value(episode.cross_errors, step),
                            "heading_error_rad": _series_value(episode.heading_errors, step),
                            "surge_velocity_mps": _series_value(episode.velocities, step),
                            "rudder_action": _series_value(episode.rudder_actions, step),
                            "rudder_change": _series_value(episode.rudder_changes, step),
                            "thrust_action": _series_value(episode.thrust_actions, step),
                            "thrust_change": _series_value(episode.thrust_changes, step),
                            "ukc_m": _series_value(episode.ukc_measurements, step),
                            "episode_reward": float(episode.reward),
                            "collision": bool(episode.collision),
                            "collision_type": episode.collision_type or "",
                            "collision_side": episode.collision_side or "",
                            "colreg_violations": int(episode.colreg_violations),
                        }
                    )


def _series_value(series, index: int) -> float:
    """
    Read an optional episode time-series value without assuming equal lengths.

    EpisodeCollector records several signals from different environment
    attributes.  Most arrays align with the path, but some optional arrays can
    be shorter in edge cases.  Empty cells keep the CSV rectangular without
    inventing fake values.
    """

    array = np.asarray(series)
    if array.size == 0 or index >= array.shape[0]:
        return ""
    return float(np.ravel(array[index])[0])


def write_framework_metrics(metrics: dict, csv_path: Path, json_path: Path) -> None:
    """
    Save MetricsComputer output in CSV and JSON formats.

    The CSV is convenient for quick inspection; the JSON preserves the complete
    nested metric structure for later automated comparison.
    """

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["ship_id", "metric", "value"])
        writer.writeheader()
        for ship_id, ship_metrics in metrics.items():
            for key, value in asdict(ship_metrics).items():
                writer.writerow({"ship_id": ship_id, "metric": key, "value": _json_safe(value)})

    json_payload = {
        ship_id: _json_safe(asdict(ship_metrics))
        for ship_id, ship_metrics in metrics.items()
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def write_controller_diagnostics(diagnostics: Sequence[dict], csv_path: Path) -> None:
    """
    Save the controller-internal signals used for heading-control tuning.

    These signals are deliberately separate from EpisodeCollector output.
    EpisodeCollector records what the framework environment reports after
    `env.step()`, while this CSV records exactly what the LOS/PID controller
    used before sending the action.
    """

    if not diagnostics:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(diagnostics[0].keys())
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)


def plot_controller_heading_diagnostics(diagnostics: Sequence[dict], output_path: Path) -> None:
    """
    Plot controller-side heading tracking performance.

    The standard framework metrics figure plots environment-reported heading
    error.  This figure answers a more specific controller-design question:
    did the vessel track the desired heading that this LOS/PID controller
    actually commanded?
    """

    if not diagnostics:
        return

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_s = np.array([row["time_s"] for row in diagnostics], dtype=float)
    raw_desired = unwrap_angle_series(row["raw_desired_heading_rad"] for row in diagnostics)
    filtered_desired = unwrap_angle_series(row["filtered_desired_heading_rad"] for row in diagnostics)
    actual_heading = unwrap_angle_series(row["actual_heading_rad"] for row in diagnostics)
    heading_error_deg = np.degrees([row["controller_heading_error_rad"] for row in diagnostics])
    yaw_rate_deg_s = np.degrees([row["yaw_rate_rad_s"] for row in diagnostics])
    rudder_deg = np.degrees([row["rudder_angle_rad"] for row in diagnostics])
    cross_track_error = np.array([row["cross_track_error_m"] for row in diagnostics], dtype=float)
    speed = np.array([row["surge_velocity_mps"] for row in diagnostics], dtype=float)
    target_speed = float(diagnostics[0]["surge_velocity_mps"] + diagnostics[0]["surge_speed_error_mps"])

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(time_s, raw_desired, label="Raw LOS desired heading", alpha=0.55)
    axes[0].plot(time_s, filtered_desired, label="Filtered desired heading", linewidth=1.8)
    axes[0].plot(time_s, actual_heading, label="Actual heading", linewidth=1.2)
    axes[0].set_ylabel("Heading [deg]")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, heading_error_deg, color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Heading error [deg]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_s, rudder_deg, label="Rudder angle", color="tab:purple")
    axes[2].plot(time_s, yaw_rate_deg_s, label="Yaw rate", color="tab:orange", alpha=0.8)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Rudder [deg] / yaw rate [deg/s]")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_s, cross_track_error, color="tab:green")
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_ylabel("Signed CTE [m]")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(time_s, speed, color="tab:blue")
    axes[4].axhline(target_speed, color="black", linestyle="--", linewidth=0.8, label="Target speed")
    axes[4].set_ylabel("Surge speed [m/s]")
    axes[4].set_xlabel("Time [s]")
    axes[4].legend(loc="best")
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_framework_index(output_dir: Path) -> None:
    """
    Create a small local index for the framework-generated artifacts.

    main.py's visualize path writes images directly.  The index is added here
    only to make the path-following baseline archive easier to browse.
    """

    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LOS/PID Framework Post-processing</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; }
    img { display: block; max-width: 100%; margin: 18px 0 32px; border: 1px solid #ddd; }
    a { color: #0645ad; }
  </style>
</head>
<body>
  <h1>LOS/PID Framework Post-processing</h1>
  <p><a href="framework_episode_data.csv">framework_episode_data.csv</a></p>
  <p><a href="framework_metrics.csv">framework_metrics.csv</a></p>
  <p><a href="framework_metrics.json">framework_metrics.json</a></p>
  <p><a href="controller_heading_diagnostics.csv">controller_heading_diagnostics.csv</a></p>
  <p><a href="resolved_config.json">resolved_config.json</a></p>
  <h2>Trajectory</h2>
  <img src="image_trajectory.png" alt="Trajectory figure">
  <h2>Metrics</h2>
  <img src="image_metrics.png" alt="Metrics figure">
  <h2>Controller Heading Diagnostics</h2>
  <img src="controller_heading_performance.png" alt="Controller heading diagnostics">
  <h2>Animation</h2>
  <img src="live_animation.gif" alt="Trajectory animation">
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _selected_framework_environment_name(framework_config: dict) -> str:
    """
    Pick the environment that should run the classical baseline.

    Path following needs a PySim-compatible physical simulator with
    `phys_sim`, `ship`, and `checkpoints`.  If config.py lists several
    environments, the first non-AirSim environment is used because AirSim
    visualization depends on an external UE process and is not needed for this
    controller baseline.
    """

    from fw.configuration.env_factory import airsim_env_name, py_sim_env_name, resolve_environment_names

    configured = resolve_environment_names(framework_config.get("env_type", py_sim_env_name))
    for name in configured:
        if name != airsim_env_name:
            return name
    raise ValueError("path_following requires a non-AirSim environment in config.py env_type.")


def _apply_framework_output_limits(env, output_limit_config: ControllerOutputLimitConfig) -> None:
    """
    Apply path-following actuator limits to the live framework environment.

    The standalone runner edits the ship specifications before constructing its
    ShipSimulator.  The framework runner receives an already-created PySimEnv, so
    the same limits are applied directly to the environment ship specification.
    PySimEnv and ShipSimulator share this specification object for normal PySim
    runs, which keeps the controller normalization and simulator saturation
    consistent.
    """

    specs = getattr(getattr(env, "ship", None), "specifications", None)
    if specs is None:
        return

    if output_limit_config.rudder_angle_limit_deg is not None:
        specs.max_rudder_angle = np.deg2rad(float(output_limit_config.rudder_angle_limit_deg))
    specs.min_thrust = float(output_limit_config.min_thrust)
    if output_limit_config.max_thrust is not None:
        specs.max_thrust = float(output_limit_config.max_thrust)


def run_with_framework_postprocessing(
    framework_config: Optional[dict] = None,
    controller_config: Optional[FrameworkControllerConfig] = None,
    output_dir: Optional[Path] = None,
    show_plots: bool = True,
    save_animation: bool = True,
) -> Path:
    """
    Run path-following control through the same post-processing path as main.py visualize.

    Processing sequence:
        1. Read the already-profiled framework config.
        2. Build the configured PySim-compatible environment through
           env_factory, exactly as main.py does.
        3. Wrap the selected path-following controller as `FrameworkPathFollowingAgent`.
        4. Collect one rollout with `EpisodeCollector`.
        5. Reuse `CollisionAnnotator`, `MetricsComputer`,
           `MatplotlibRenderer`, and `AnimationRenderer`.
        6. Save framework-compatible plots plus CSV/JSON data.

    Returns:
        The timestamped output directory containing all generated artifacts.
    """

    if not show_plots:
        os.environ.setdefault("MPLBACKEND", "Agg")

    from fw.configuration.config_profile import ProfileRegistry
    from fw.configuration.env_factory import build_default_environment, initialize_all_environments
    from fw.configuration.profile_applier import SimulatorApplier
    from fw.postprocessing.animation_renderer import AnimationRenderer
    from fw.postprocessing.episode_collector import EpisodeCollector
    from fw.postprocessing.matplotlib_renderer import MatplotlibRenderer
    from fw.postprocessing.metrics_computer import MetricsComputer
    from fw.postprocessing.ship_episode_data import CollisionAnnotator

    framework_config = framework_config or get_config_parameters()
    del controller_config
    control_config = load_classical_control_config(framework_config)
    algorithm_config = load_path_following_algorithm_config(framework_config)
    guidance_config = load_guidance_parameter_config(framework_config, algorithm_config.guidance)
    surge_config = load_surge_speed_control_config(framework_config)
    output_limit_config = load_controller_output_limit_config(framework_config)
    rudder_response_config = load_rudder_response_config(framework_config)
    output_dir = output_dir or create_output_dir(Path("run_archives"))
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_matplotlib(output_dir / "image.png")

    initialize_all_environments()
    env_name = _selected_framework_environment_name(framework_config)
    env_wrapper = build_default_environment(env_name, framework_config)
    env = env_wrapper.gym_env

    profile = ProfileRegistry.get(framework_config.get("profile", "default_profile"))
    simulator = getattr(env, "phys_sim", None) or getattr(env, "_phys_sim", None)
    if simulator is not None:
        SimulatorApplier.apply(profile, simulator)
    _apply_framework_output_limits(env, output_limit_config)

    agent = FrameworkPathFollowingAgent(
        env=env,
        algorithm_config=algorithm_config,
        guidance_config=guidance_config,
        surge_config=surge_config,
        output_limit_config=output_limit_config,
        rudder_response_config=rudder_response_config,
        model_dir=output_dir,
    )

    collector = EpisodeCollector(agent, env, profile)
    run_data = collector.collect(num_episodes=1, live_animation=False)
    CollisionAnnotator().annotate(run_data)
    metrics = MetricsComputer.compute(run_data)

    if save_animation:
        AnimationRenderer.render_gif(
            run_data.ships,
            env,
            save_path=str(output_dir / "live_animation.gif"),
        )

    MatplotlibRenderer.render(
        run_data.ships,
        metrics,
        env,
        save_path=str(output_dir / "image.png"),
        show_plots=show_plots,
    )

    write_framework_episode_data(run_data, output_dir / "framework_episode_data.csv")
    write_framework_metrics(metrics, output_dir / "framework_metrics.csv", output_dir / "framework_metrics.json")
    write_resolved_config(
        framework_config=framework_config,
        control_config=control_config,
        controller_config={
            "runner": "framework_postprocessing",
            "environment_name": env_name,
            "output_dir": str(output_dir),
            "controller": algorithm_config.controller,
            "guidance": algorithm_config.guidance,
            "guidance_parameters": asdict(guidance_config),
            "pid": algorithm_config.pid,
            "vector_field": algorithm_config.vector_field,
            "feedback_linearization": algorithm_config.feedback_linearization,
            "sliding_mode": algorithm_config.sliding_mode,
            "backstepping_dynamic_smc": algorithm_config.backstepping_dynamic_smc,
            "surge_pid": asdict(surge_config),
            "output_limits": asdict(output_limit_config),
            "rudder_response": asdict(rudder_response_config),
        },
        path=output_dir / "resolved_config.json",
    )

    return output_dir


def _configure_matplotlib(output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))


def _draw_map_background(ax, map_data: MapData, show_boundary: bool = True) -> None:
    import matplotlib.patches as patches
    from shapely import MultiPolygon, Polygon

    if show_boundary:
        # This rectangular boundary is the same fixed simulation boundary used
        # by AgentManagerSimple.render.  It gives context for the coordinate
        # range even when the river geometry occupies only part of the axes.
        ax.plot(
            [MIN_GRID_POS, MAX_GRID_POS, MAX_GRID_POS, MIN_GRID_POS, MIN_GRID_POS],
            [MIN_GRID_POS, MIN_GRID_POS, MAX_GRID_POS, MAX_GRID_POS, MIN_GRID_POS],
            "k-",
            linewidth=1.0,
            label="Boundary",
        )

    # Low water level area:
    # The existing framework visualizes the region between the wider Western
    # Scheldt contour and the navigable waterway contour as a hatched area.
    # This is computed geometrically as overall - obstacles, then rendered as
    # one or more polygon patches.  The label is only attached to the first
    # patch so the legend stays readable when the difference is a MultiPolygon.
    low_water_geometry = Polygon(map_data.overall).difference(Polygon(map_data.obstacles))
    if isinstance(low_water_geometry, MultiPolygon):
        low_water_polygons = list(low_water_geometry.geoms)
    elif low_water_geometry.is_empty:
        low_water_polygons = []
    else:
        low_water_polygons = [low_water_geometry]

    for idx, polygon in enumerate(low_water_polygons):
        ax.add_patch(
            patches.Polygon(
                np.array(polygon.exterior.coords),
                closed=True,
                facecolor="none",
                edgecolor="gray",
                hatch="///",
                linewidth=0.0,
                alpha=0.3,
                label="Low water level area" if idx == 0 else "",
            )
        )

    # "obstacles" is the navigable waterway boundary in the existing PySimEnv
    # naming.  It is drawn in red to match AgentManagerSimple.render.
    ax.add_patch(
        patches.Polygon(
            map_data.obstacles,
            closed=True,
            edgecolor="red",
            facecolor="none",
            linewidth=1.5,
            label="Waterway",
        )
    )

    # "overall" is the wider Western Scheldt outline.  Drawing it separately
    # helps distinguish the navigable channel from the surrounding map contour.
    ax.add_patch(
        patches.Polygon(
            map_data.overall,
            closed=True,
            edgecolor="brown",
            facecolor="none",
            linewidth=1.5,
            label="Western Scheldt",
        )
    )


def make_square_viewport(paths: Sequence[np.ndarray], margin_ratio: float = 0.08) -> MapViewport:
    """
    Compute one square viewport for map/path visualizations.

    The same x/y limits are reused by all trajectory figures.  The larger of
    the x-span and y-span defines the square side length, so a metre in x and a
    metre in y occupy the same visual distance.
    """

    points = [np.asarray(path, dtype=float) for path in paths if np.asarray(path).size]
    if not points:
        return MapViewport(MIN_GRID_POS, MAX_GRID_POS, MIN_GRID_POS, MAX_GRID_POS)

    merged = np.vstack(points)
    x_span = max(float(np.ptp(merged[:, 0])), 1.0)
    y_span = max(float(np.ptp(merged[:, 1])), 1.0)
    square_side = max(x_span, y_span)
    margin = max(150.0, margin_ratio * square_side)

    x_center = 0.5 * (float(merged[:, 0].min()) + float(merged[:, 0].max()))
    y_center = 0.5 * (float(merged[:, 1].min()) + float(merged[:, 1].max()))
    half_side = 0.5 * square_side + margin

    return MapViewport(
        x_min=x_center - half_side,
        x_max=x_center + half_side,
        y_min=y_center - half_side,
        y_max=y_center + half_side,
    )


def apply_map_viewport(ax, viewport: MapViewport) -> None:
    """
    Apply shared 1:1 x/y limits to a matplotlib axes.
    """

    ax.set_xlim(viewport.x_min, viewport.x_max)
    ax.set_ylim(viewport.y_min, viewport.y_max)
    ax.set_aspect("equal", adjustable="box")


def plot_predefined_path_on_map(
    map_data: MapData,
    waypoints: Sequence[Point],
    viewport: MapViewport,
    path: Path,
    original_waypoints: Optional[Sequence[Point]] = None,
) -> None:
    """
    Save a map-only diagnostic showing the route before simulation.

    This plot answers a simple setup question: "What path will the controller
    try to follow on the map?"  It is generated before plotting the vessel
    response so that route definition and controller performance can be checked
    independently.
    """

    _configure_matplotlib(path)

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    waypoint_arr = np.array(waypoints)

    fig, ax = plt.subplots(figsize=(12, 8))
    _draw_map_background(ax, map_data)
    if original_waypoints is not None:
        original_arr = np.array(original_waypoints)
        ax.plot(
            original_arr[:, 0],
            original_arr[:, 1],
            color="0.55",
            linewidth=1.0,
            alpha=0.55,
            label="Original imported path",
        )
    ax.plot(
        waypoint_arr[:, 0],
        waypoint_arr[:, 1],
        "g--",
        linewidth=2.0,
        alpha=0.85,
        label="Controller path",
    )
    ax.scatter(
        waypoint_arr[:, 0],
        waypoint_arr[:, 1],
        c="tab:orange",
        edgecolors="black",
        linewidths=0.6,
        s=42,
        label="Simplified waypoints",
        zorder=5,
    )
    for idx, (x_pos, y_pos) in enumerate(waypoint_arr):
        ax.annotate(
            str(idx),
            (x_pos, y_pos),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            color="black",
            zorder=6,
        )
    ax.scatter([waypoint_arr[0, 0]], [waypoint_arr[0, 1]], c="blue", s=80, label="Start", zorder=4)
    ax.scatter([waypoint_arr[-1, 0]], [waypoint_arr[-1, 1]], c="red", s=80, label="Target", zorder=4)
    apply_map_viewport(ax, viewport)
    ax.set_title("Predefined Path on Map")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_trajectory_comparison_on_map(
    samples: Sequence[dict],
    map_data: MapData,
    waypoints: Sequence[Point],
    viewport: MapViewport,
    path: Path,
) -> None:
    """
    Save the main spatial comparison plot.

    The predefined path is drawn on top of the vessel trajectory so small
    deviations remain visible.  This is useful when the controller tracks well
    and the two lines nearly overlap.
    """

    _configure_matplotlib(path)

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    waypoint_arr = np.array(waypoints)
    actual_path = np.array([[row["x"], row["y"]] for row in samples])

    fig, ax = plt.subplots(figsize=(12, 8))
    _draw_map_background(ax, map_data)
    ax.plot(
        actual_path[:, 0],
        actual_path[:, 1],
        color="tab:blue",
        linewidth=2.0,
        alpha=0.85,
        label="LOS+PID vessel path",
    )
    ax.plot(
        waypoint_arr[:, 0],
        waypoint_arr[:, 1],
        "g--",
        linewidth=2.6,
        alpha=0.95,
        label="Predefined path",
        zorder=4,
    )
    ax.scatter([waypoint_arr[0, 0]], [waypoint_arr[0, 1]], c="blue", s=80, label="Start", zorder=4)
    ax.scatter([waypoint_arr[-1, 0]], [waypoint_arr[-1, 1]], c="red", s=80, label="Target", zorder=4)
    ax.scatter([actual_path[-1, 0]], [actual_path[-1, 1]], c="black", s=55, label="Final vessel pos", zorder=4)
    apply_map_viewport(ax, viewport)
    ax.set_title("Path Following Comparison on Map")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_results(
    samples: Sequence[dict],
    waypoints: Sequence[Point],
    metrics: dict,
    viewport: MapViewport,
    path: Path,
) -> None:
    """
    Save a control-performance summary figure.

    This figure complements the map plots.  It focuses on controller behavior:
    path-following geometry, cross-track error, heading error, and rudder effort.
    """

    _configure_matplotlib(path)

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.array([row["time"] for row in samples])
    xs = np.array([row["x"] for row in samples])
    ys = np.array([row["y"] for row in samples])
    cross_track_errors = np.array([row["cross_track_error"] for row in samples])
    heading_errors = np.degrees([row["heading_error"] for row in samples])
    rudder_angles = np.degrees([row["rudder_angle"] for row in samples])
    raw_rudder_angles = np.degrees([row.get("raw_rudder_angle", row["rudder_angle"]) for row in samples])
    speeds = np.array([row["u"] for row in samples])
    target_speeds = np.array([row["target_speed"] for row in samples])
    speed_errors = np.array([row["speed_error"] for row in samples])
    thrust_commands = np.array([row["thrust"] for row in samples])
    headings, desired_headings = continuous_heading_tracking_series(samples)
    guidance_desired_headings = unwrap_angle_series(
        row.get("guidance_desired_heading", row["desired_heading"]) for row in samples
    )
    controller_name = str(samples[0].get("controller", "pid")).replace("_", " ")
    guidance_name = str(samples[0].get("guidance", "los")).replace("_", " ")

    waypoint_x = [point[0] for point in waypoints]
    waypoint_y = [point[1] for point in waypoints]

    figure, axes = plt.subplots(4, 2, figsize=(13, 14), constrained_layout=True)

    axes[0, 0].plot(waypoint_x, waypoint_y, "k--", linewidth=1.5, label="reference path")
    axes[0, 0].plot(xs, ys, "tab:blue", linewidth=1.8, label="vessel trajectory")
    axes[0, 0].scatter(waypoint_x, waypoint_y, c="tab:red", s=28, zorder=3, label="waypoints")
    apply_map_viewport(axes[0, 0], viewport)
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_title(f"{guidance_name.upper()} path following")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(times, cross_track_errors, "tab:orange")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("cross-track error [m]")
    axes[0, 1].set_title("Cross-track error")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(times, headings, label="heading")
    axes[1, 0].plot(times, guidance_desired_headings, "--", label="desired heading")
    axes[1, 0].plot(times, desired_headings, ":", alpha=0.65, label="controller target")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("heading [deg]")
    axes[1, 0].set_title("Heading tracking")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(times, heading_errors, "tab:red")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("heading error [deg]")
    axes[1, 1].set_title("Heading error")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(times, rudder_angles, "tab:green", label="filtered rudder")
    axes[2, 0].plot(times, raw_rudder_angles, color="tab:gray", linestyle="--", alpha=0.65, label="raw command")
    axes[2, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("rudder angle [deg]")
    axes[2, 0].set_title(f"{controller_name.title()} rudder command")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    metrics_text = "\n".join(
        [
            f"RMS cross-track error: {metrics['rms_cross_track_error_m']:.2f} m",
            f"Mean abs cross-track error: {metrics['mean_abs_cross_track_error_m']:.2f} m",
            f"Max abs cross-track error: {metrics['max_abs_cross_track_error_m']:.2f} m",
            f"RMS heading error: {metrics['rms_heading_error_deg']:.2f} deg",
            f"Mean abs heading error: {metrics['mean_abs_heading_error_deg']:.2f} deg",
            f"Target surge speed: {metrics['target_surge_velocity_mps']:.2f} m/s",
            f"Final surge speed: {metrics['final_surge_velocity_mps']:.2f} m/s",
            f"RMS speed error: {metrics['rms_speed_error_mps']:.2f} m/s",
            f"Mean abs speed error: {metrics['mean_abs_speed_error_mps']:.2f} m/s",
            f"Final target distance: {metrics['final_target_distance_m']:.2f} m",
            f"Max abs rudder: {metrics['max_abs_rudder_deg']:.2f} deg",
            f"Rudder saturation: {100.0 * metrics['rudder_saturation_ratio']:.2f} %",
            f"Max thrust command: {metrics['max_thrust_command']:.3f}",
        ]
    )
    axes[2, 1].plot(times, speeds, label="surge speed")
    axes[2, 1].plot(times, target_speeds, "--", label="target speed")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("speed [m/s]")
    axes[2, 1].set_title("Surge speed tracking")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    thrust_axis = axes[3, 0]
    speed_error_axis = thrust_axis.twinx()
    thrust_line = thrust_axis.plot(times, thrust_commands, "tab:purple", label="thrust command")
    speed_error_line = speed_error_axis.plot(times, speed_errors, "tab:brown", alpha=0.65, label="speed error")
    speed_error_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    thrust_axis.set_xlabel("time [s]")
    thrust_axis.set_ylabel("thrust command [-]", color="tab:purple")
    speed_error_axis.set_ylabel("speed error [m/s]", color="tab:brown")
    thrust_axis.tick_params(axis="y", labelcolor="tab:purple")
    speed_error_axis.tick_params(axis="y", labelcolor="tab:brown")
    thrust_axis.set_title("Surge PID command and error")
    thrust_axis.grid(True, alpha=0.3)
    lines = thrust_line + speed_error_line
    thrust_axis.legend(lines, [line.get_label() for line in lines], loc="upper right")

    axes[3, 1].axis("off")
    axes[3, 1].text(
        0.02,
        0.95,
        metrics_text,
        transform=axes[3, 1].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="0.75", boxstyle="round,pad=0.5"),
    )
    axes[3, 1].set_title("Path-following performance")

    figure.savefig(path, dpi=160)
    plt.close(figure)


def render_interactive_html(
    samples: Sequence[dict],
    map_data: MapData,
    waypoints: Sequence[Point],
    html_path: Path,
    static_plot_path: Path,
    predefined_map_path: Path,
    comparison_map_path: Path,
    gif_path: Optional[Path],
    csv_path: Path,
    trajectory_csv_path: Path,
    metrics_csv_path: Path,
    metrics_json_path: Path,
    resolved_config_path: Path,
    metrics: dict,
) -> None:
    """
    Build the per-run HTML index file.

    If Plotly is installed, the file becomes an interactive dashboard.  If not,
    it still works as a simple static report that links together all generated
    PNG/GIF/CSV/JSON outputs.
    """

    html_path.parent.mkdir(parents=True, exist_ok=True)
    final = samples[-1]
    summary = (
        f"Final: x={final['x']:.1f} m, y={final['y']:.1f} m, "
        f"cross-track error={final['cross_track_error']:.2f} m, "
        f"heading error={np.degrees(final['heading_error']):.2f} deg, "
        f"RMS CTE={metrics['rms_cross_track_error_m']:.2f} m, "
        f"RMS speed error={metrics['rms_speed_error_mps']:.2f} m/s"
    )

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        gif_section = []
        if gif_path is not None and gif_path.exists():
            gif_section = [
                "<h2>Trajectory Animation</h2>",
                f"<img src='{gif_path.name}' style='max-width:100%;height:auto;'>",
            ]

        html_path.write_text(
            "\n".join(
                [
                    "<!doctype html>",
                    "<html><head><meta charset='utf-8'><title>LOS PID Path Following</title></head>",
                    "<body style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px;'>",
                    "<h1>LOS PID Path Following</h1>",
                    f"<p>{summary}</p>",
                    "<h2>Predefined Path on Map</h2>",
                    f"<img src='{predefined_map_path.name}' style='max-width:100%;height:auto;'>",
                    "<h2>Trajectory Comparison on Map</h2>",
                    f"<img src='{comparison_map_path.name}' style='max-width:100%;height:auto;'>",
                    "<h2>Control Summary</h2>",
                    f"<img src='{static_plot_path.name}' style='max-width:100%;height:auto;'>",
                    *gif_section,
                    "<h2>Raw Data</h2>",
                    f"<p><a href='{csv_path.name}'>simulation_data.csv</a></p>",
                    f"<p><a href='{trajectory_csv_path.name}'>trajectory_path.csv</a></p>",
                    f"<p><a href='{metrics_csv_path.name}'>metrics_summary.csv</a></p>",
                    f"<p><a href='{metrics_json_path.name}'>metrics_summary.json</a></p>",
                    f"<p><a href='{resolved_config_path.name}'>resolved_config.json</a></p>",
                    "<p><a href='control_parameters.json'>control_parameters.json</a></p>",
                    "</body></html>",
                ]
            ),
            encoding="utf-8",
        )
        return

    times = np.array([row["time"] for row in samples])
    xs = np.array([row["x"] for row in samples])
    ys = np.array([row["y"] for row in samples])
    cross_track_errors = np.array([row["cross_track_error"] for row in samples])
    heading_errors = np.degrees([row["heading_error"] for row in samples])
    rudder_angles = np.degrees([row["rudder_angle"] for row in samples])
    raw_rudder_angles = np.degrees([row.get("raw_rudder_angle", row["rudder_angle"]) for row in samples])
    speeds = np.array([row["u"] for row in samples])
    target_speeds = np.array([row["target_speed"] for row in samples])
    speed_errors = np.array([row["speed_error"] for row in samples])
    thrust_commands = np.array([row["thrust"] for row in samples])
    headings, desired_headings = continuous_heading_tracking_series(samples)
    guidance_desired_headings = unwrap_angle_series(
        row.get("guidance_desired_heading", row["desired_heading"]) for row in samples
    )
    waypoint_x = [point[0] for point in waypoints]
    waypoint_y = [point[1] for point in waypoints]

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scatter", "colspan": 2}, None],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=[
            "Trajectory",
            "Cross-track error",
            "Heading tracking",
            "Rudder command",
            "Surge velocity",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=map_data.overall[:, 0],
            y=map_data.overall[:, 1],
            mode="lines",
            name="Western Scheldt",
            line=dict(color="brown", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=map_data.obstacles[:, 0],
            y=map_data.obstacles[:, 1],
            mode="lines",
            name="Waterway",
            line=dict(color="red", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=waypoint_x,
            y=waypoint_y,
            mode="lines+markers",
            name="reference path",
            line=dict(color="black", dash="dash", width=2),
            marker=dict(color="crimson", size=8),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="vessel trajectory",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<extra>trajectory</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=cross_track_errors, mode="lines", name="cross-track error [m]"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=headings, mode="lines", name="heading [deg]"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=guidance_desired_headings,
            mode="lines",
            name="desired heading [deg]",
            line=dict(dash="dash"),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=desired_headings,
            mode="lines",
            name="controller target [deg]",
            line=dict(dash="dot"),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=times, y=rudder_angles, mode="lines", name="filtered rudder [deg]"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=raw_rudder_angles,
            mode="lines",
            name="raw rudder command [deg]",
            line=dict(color="gray", dash="dash"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=heading_errors, mode="lines", name="heading error [deg]"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=speeds, mode="lines", name="surge velocity [m/s]"),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=times, y=target_speeds, mode="lines", name="target speed [m/s]", line=dict(dash="dash")),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=times, y=thrust_commands, mode="lines", name="thrust command [-]"),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=times, y=speed_errors, mode="lines", name="speed error [m/s]"),
        row=3,
        col=2,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_layout(
        title=f"LOS Guidance + PID Path Following<br><sup>{summary}</sup>",
        height=1050,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)


def render_animation_gif(
    samples: Sequence[dict],
    map_data: MapData,
    waypoints: Sequence[Point],
    viewport: MapViewport,
    gif_path: Path,
    max_frames: int = 160,
) -> None:
    """
    Render a lightweight GIF animation of the closed-loop trajectory.

    Only a fixed number of frames is sampled from the full trajectory.  This
    keeps GIF generation fast and prevents the output file from becoming too
    large for long simulations.
    """

    _configure_matplotlib(gif_path)

    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    gif_path.parent.mkdir(parents=True, exist_ok=True)

    path = np.array([[row["x"], row["y"]] for row in samples])
    waypoint_arr = np.array(waypoints)
    waypoint_x = waypoint_arr[:, 0]
    waypoint_y = waypoint_arr[:, 1]
    frame_indices = np.linspace(0, len(path) - 1, min(max_frames, len(path))).astype(int)
    frames: list[Image.Image] = []

    fig = Figure(figsize=(8, 6), dpi=120)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    for idx in frame_indices:
        ax.clear()
        visible = path[: idx + 1]
        _draw_map_background(ax, map_data, show_boundary=False)
        ax.plot(visible[:, 0], visible[:, 1], color="tab:blue", linewidth=2, alpha=0.85, label="vessel path")
        ax.plot(waypoint_x, waypoint_y, "g--", linewidth=1.8, label="predefined path", zorder=4)
        ax.scatter([visible[-1, 0]], [visible[-1, 1]], c="tab:blue", s=55, zorder=4)
        ax.scatter([waypoint_x[0]], [waypoint_y[0]], c="blue", s=50, zorder=4)
        ax.scatter([waypoint_x[-1]], [waypoint_y[-1]], c="red", s=50, zorder=4)
        apply_map_viewport(ax, viewport)
        ax.set_title(f"LOS PID path following, t={samples[idx]['time']:.1f}s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=60,
        loop=0,
        optimize=True,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options for the standalone classical-control run.

    Defaults are chosen for the Western Scheldt map and the bundled predefined
    route.  They can be overridden to test LOS/PID sensitivity without editing
    source code.
    """

    parser = argparse.ArgumentParser(
        description="Simple vessel path-following simulation with LOS guidance and PID heading control."
    )
    parser.add_argument("--duration", type=float, default=3900.0, help="Simulation duration [s].")
    parser.add_argument("--dt", type=float, default=0.5, help="Simulation time step [s].")
    parser.add_argument(
        "--lookahead",
        type=float,
        default=None,
        help="Optional LOS lookahead distance [m]. Defaults to three times the configured ship length.",
    )
    parser.add_argument("--switch-distance", type=float, default=None, help="Override config.py waypoint switching distance [m].")
    parser.add_argument("--target-speed", type=float, default=None, help="Override config.py target surge speed [m/s].")
    parser.add_argument(
        "--controller",
        choices=("pid", "feedback_linearization", "sliding_mode", "backstepping_dynamic_smc"),
        default=None,
        help="Override config.py path_following_controller.",
    )
    parser.add_argument(
        "--guidance",
        choices=("los", "pure_pursuit", "vector_field"),
        default=None,
        help="Override config.py path_following_guidance.",
    )
    parser.add_argument(
        "--path-simplification-tolerance",
        type=float,
        default=None,
        help="Override config.py path_following_simplification_tolerance_m [m].",
    )
    parser.add_argument(
        "--no-path-simplification",
        action="store_true",
        help="Disable route simplification and use every imported path point.",
    )
    # The heading controller is intentionally used as a PD controller by
    # default.  The LOS law already removes geometric cross-track error, and an
    # integral term in the heading loop can wind up during turns or while the
    # rudder is saturated.  A parameter sweep on the current config.py baseline
    # showed kp=1.1, kd=8.0 gives much lower heading and cross-track errors than
    # the previous kp=1.8, kd=4.0 setting.  After changing the LOS lookahead to
    # three ship lengths, the selected kp=1.5, kd=20.0 is a better compromise:
    # it keeps desired-heading tracking smooth while reducing the cross-track
    # error that appears when the long lookahead makes the vessel cut corners.
    # The LOS lookahead distance is resolved later from the configured ship
    # length, so changing ship_model in config.py also changes the default
    # guidance geometry.
    parser.add_argument("--kp", type=float, default=None, help="Override config.py PID heading proportional gain.")
    parser.add_argument("--ki", type=float, default=None, help="Override config.py PID heading integral gain.")
    parser.add_argument("--kd", type=float, default=None, help="Override config.py PID heading derivative gain.")
    # These default surge gains are tuned for the current config.py baseline
    # (ShipModel.MYZAKO_37 with Model.FOSSEN).  They remain command-line
    # options because other ship/dynamics combinations can require different
    # thrust-loop tuning.
    parser.add_argument("--surge-kp", type=float, default=None, help="Override config.py surge-speed PID proportional gain.")
    parser.add_argument("--surge-ki", type=float, default=None, help="Override config.py surge-speed PID integral gain.")
    parser.add_argument("--surge-kd", type=float, default=None, help="Override config.py surge-speed PID derivative gain.")
    parser.add_argument("--rudder-limit-deg", type=float, default=None, help="Override config.py rudder output limit [deg].")
    parser.add_argument("--disable-rudder-response", action="store_true", help="Disable config.py rudder response filter.")
    parser.add_argument("--rudder-response-tau", type=float, default=None, help="Override rudder response time constant [s].")
    parser.add_argument("--rudder-response-small-tau", type=float, default=None, help="Override small-signal rudder response time constant [s].")
    parser.add_argument("--rudder-response-threshold-deg", type=float, default=None, help="Override small/large rudder command threshold [deg].")
    parser.add_argument("--rudder-response-rate-deg-s", type=float, default=None, help="Override rudder response rate limit [deg/s].")
    parser.add_argument("--rudder-response-deadband-deg", type=float, default=None, help="Override rudder response deadband [deg].")
    parser.add_argument("--rudder-neutral-deadband-deg", type=float, default=None, help="Override rudder neutral snap deadband [deg].")
    parser.add_argument("--min-thrust", type=float, default=None, help="Override config.py minimum thrust command.")
    parser.add_argument("--max-thrust", type=float, default=None, help="Override config.py maximum thrust command.")
    parser.add_argument("--env-dir", type=Path, default=DEFAULT_ENV_DIR)
    parser.add_argument("--overall-contour", default=None, help="Optional override for config.py overall_contour.")
    parser.add_argument("--depth-contour", default=None, help="Optional override for config.py depth_contour.")
    parser.add_argument("--trajectory", default=None, help="Optional override for config.py trajectory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV/PNG/HTML/GIF outputs.")
    parser.add_argument("--archive-root", type=Path, default=Path("run_archives"))
    parser.add_argument("--csv", type=Path, default=None, help="Optional explicit complete simulation-data CSV path.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional explicit PNG output path.")
    parser.add_argument("--html", type=Path, default=None, help="Optional explicit interactive HTML output path.")
    parser.add_argument("--gif", type=Path, default=None, help="Optional explicit GIF output path.")
    parser.add_argument("--no-gif", action="store_true", help="Skip animated GIF generation.")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML dashboard in the default browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # main.py obtains the framework setup through get_config_parameters()
    # instead of reading CONFIG_PARAMETERS directly.  That distinction matters
    # because get_config_parameters() applies the selected profile first.  This
    # standalone controller follows the same rule, so changing profile,
    # ship_model, dynamics_model, start_pos, target_pos, wind/current, or map
    # filenames in config.py changes this LOS/PID run as well.
    framework_config = get_config_parameters()
    control_config = load_classical_control_config(framework_config)
    algorithm_config = load_path_following_algorithm_config(framework_config)
    simplification_config = load_path_simplification_config(framework_config)
    guidance_parameter_config = load_guidance_parameter_config(
        framework_config,
        algorithm_config.guidance,
    )
    surge_config = load_surge_speed_control_config(framework_config)
    output_limit_config = load_controller_output_limit_config(framework_config)
    rudder_response_config = load_rudder_response_config(framework_config)

    if args.controller is not None:
        algorithm_config = replace(algorithm_config, controller=args.controller)
    if args.guidance is not None:
        algorithm_config = replace(algorithm_config, guidance=args.guidance)
        guidance_parameter_config = load_guidance_parameter_config(
            framework_config,
            algorithm_config.guidance,
        )
    if args.no_path_simplification:
        simplification_config = replace(simplification_config, enabled=False)
    if args.path_simplification_tolerance is not None:
        simplification_config = replace(
            simplification_config,
            tolerance_m=args.path_simplification_tolerance,
        )
    pid_params = algorithm_config.pid
    heading_kp = float(args.kp if args.kp is not None else pid_params.get("heading_kp", 0.9))
    heading_ki = float(args.ki if args.ki is not None else pid_params.get("heading_ki", 0.0))
    heading_kd = float(args.kd if args.kd is not None else pid_params.get("heading_kd", 0.0))
    if args.target_speed is not None:
        surge_config = replace(surge_config, target_speed=args.target_speed)
    if args.surge_kp is not None:
        surge_config = replace(surge_config, kp=args.surge_kp)
    if args.surge_ki is not None:
        surge_config = replace(surge_config, ki=args.surge_ki)
    if args.surge_kd is not None:
        surge_config = replace(surge_config, kd=args.surge_kd)
    if args.rudder_limit_deg is not None:
        output_limit_config = replace(output_limit_config, rudder_angle_limit_deg=args.rudder_limit_deg)
    if args.disable_rudder_response:
        rudder_response_config = replace(rudder_response_config, enabled=False)
    if args.rudder_response_tau is not None:
        rudder_response_config = replace(rudder_response_config, time_constant_s=args.rudder_response_tau)
    if args.rudder_response_small_tau is not None:
        rudder_response_config = replace(
            rudder_response_config,
            small_signal_time_constant_s=args.rudder_response_small_tau,
        )
    if args.rudder_response_threshold_deg is not None:
        rudder_response_config = replace(
            rudder_response_config,
            large_signal_threshold_deg=args.rudder_response_threshold_deg,
        )
    if args.rudder_response_rate_deg_s is not None:
        rudder_response_config = replace(rudder_response_config, max_rate_deg_s=args.rudder_response_rate_deg_s)
    if args.rudder_response_deadband_deg is not None:
        rudder_response_config = replace(rudder_response_config, deadband_deg=args.rudder_response_deadband_deg)
    if args.rudder_neutral_deadband_deg is not None:
        rudder_response_config = replace(rudder_response_config, neutral_deadband_deg=args.rudder_neutral_deadband_deg)
    if args.min_thrust is not None:
        output_limit_config = replace(output_limit_config, min_thrust=args.min_thrust)
    if args.max_thrust is not None:
        output_limit_config = replace(output_limit_config, max_thrust=args.max_thrust)

    # CLI map-file options are explicit overrides.  If they are omitted, the
    # same filenames selected by config.py/profile application are used.
    if args.overall_contour is not None:
        control_config = replace(control_config, overall_contour=args.overall_contour)
    if args.depth_contour is not None:
        control_config = replace(control_config, depth_contour=args.depth_contour)
    if args.trajectory is not None:
        control_config = replace(control_config, trajectory=args.trajectory)

    configured_ship_length = create_ship(control_config.ship_model).specifications.length
    configured_lookahead = guidance_parameter_config.lookahead_distance
    if configured_lookahead is None:
        configured_lookahead = guidance_parameter_config.lookahead_ship_length_multiplier * configured_ship_length
    effective_lookahead = args.lookahead if args.lookahead is not None else float(configured_lookahead)
    effective_switch_distance = (
        args.switch_distance
        if args.switch_distance is not None
        else guidance_parameter_config.switch_distance
    )

    # Each run receives its own timestamped output directory, matching the
    # "post-process after run" style used by the main framework.  This avoids
    # overwriting previous experiments and makes it easy to compare controller
    # parameter sweeps.
    output_dir = args.output_dir or create_output_dir(args.archive_root)

    # File layout:
    # - simulation_data.csv: complete time series for analysis,
    # - trajectory_path.csv: compact x/y path only,
    # - metrics_summary.*: scalar performance indicators,
    # - PNG/GIF/HTML: visual post-processing artifacts.
    csv_path = args.csv or output_dir / "simulation_data.csv"
    trajectory_csv_path = output_dir / "trajectory_path.csv"
    metrics_csv_path = output_dir / "metrics_summary.csv"
    metrics_json_path = output_dir / "metrics_summary.json"
    resolved_config_path = output_dir / "resolved_config.json"
    control_parameters_path = output_dir / "control_parameters.json"
    plot_path = args.plot or output_dir / "summary.png"
    predefined_map_path = output_dir / "map_predefined_path.png"
    comparison_map_path = output_dir / "trajectory_comparison.png"
    html_path = args.html or output_dir / "index.html"
    gif_path = args.gif or output_dir / "trajectory.gif"

    # Load the same static map assets that PySimEnv uses for visualization.
    # This is why the generated plots show the Western Scheldt and waterway
    # contours instead of a blank local x/y plot.
    map_data = load_map_data(
        env_dir=args.env_dir,
        overall_file=control_config.overall_contour,
        obstacles_file=control_config.depth_contour,
        path_file=control_config.trajectory,
    )

    # Convert the raw map trajectory into the actual waypoint sequence used by
    # guidance.  The helper trims the route to the configured start/target and
    # optionally simplifies dense CSV points into a smaller piecewise-linear
    # route that remains close to the original path.
    unsimplified_waypoints = path_to_waypoints(
        map_data.reference_path,
        start_pos=control_config.start_pos,
        target_pos=control_config.target_pos,
        simplification_config=PathSimplificationConfig(enabled=False),
    )
    waypoints = path_to_waypoints(
        map_data.reference_path,
        start_pos=control_config.start_pos,
        target_pos=control_config.target_pos,
        simplification_config=simplification_config,
    )

    # Run the closed-loop controller.  At this point no plots are generated; we
    # first collect a clean numerical time series and then post-process it.
    samples = run_simulation(
        control_config=control_config,
        algorithm_config=algorithm_config,
        surge_config=surge_config,
        output_limit_config=output_limit_config,
        rudder_response_config=rudder_response_config,
        waypoints=waypoints,
        duration=args.duration,
        dt=args.dt,
        lookahead_distance=effective_lookahead,
        switch_distance=effective_switch_distance,
        heading_pid_gains=(heading_kp, heading_ki, heading_kd),
    )

    # Scalar metrics are computed once and reused by the CSV/JSON exports, the
    # summary plot, and the HTML report header.
    metrics = compute_performance_metrics(samples, waypoints, control_config)
    waypoint_arr = np.array(waypoints, dtype=float)
    actual_path_arr = np.array([[row["x"], row["y"]] for row in samples], dtype=float)

    # All trajectory/map figures use the same square viewport.  The complete
    # Western Scheldt outline, the waterway outline, the reference path, and the
    # actual path are all included so the map is not clipped in any trajectory
    # visualization.
    unsimplified_waypoint_arr = np.array(unsimplified_waypoints, dtype=float)
    viewport = make_square_viewport(
        [map_data.overall, map_data.obstacles, unsimplified_waypoint_arr, waypoint_arr, actual_path_arr]
    )

    # Persist numerical outputs before plotting.  This makes the run useful
    # even if a later optional renderer fails on a local graphics dependency.
    write_csv(samples, csv_path)
    write_trajectory_csv(samples, trajectory_csv_path)
    write_metrics_summary(metrics, metrics_csv_path, metrics_json_path)
    control_parameters = {
        "duration_s": args.duration,
        "dt_s": args.dt,
        "controller": algorithm_config.controller,
        "guidance": algorithm_config.guidance,
        "lookahead_m": effective_lookahead,
        "lookahead_source": "cli" if args.lookahead is not None else "config",
        "configured_ship_length_m": configured_ship_length,
        "guidance_parameters": {
            "los": {
                "lookahead_distance_m": guidance_parameter_config.lookahead_distance,
                "lookahead_ship_length_multiplier": guidance_parameter_config.lookahead_ship_length_multiplier,
                "switch_distance_m": effective_switch_distance,
            },
            "vector_field": algorithm_config.vector_field,
        },
        "path_simplification": {
            "enabled": simplification_config.enabled,
            "tolerance_m": simplification_config.tolerance_m,
            "raw_waypoint_count": len(unsimplified_waypoints),
            "simplified_waypoint_count": len(waypoints),
        },
        "heading_pid": {
            "kp": heading_kp,
            "ki": heading_ki,
            "kd": heading_kd,
            "yaw_rate_damping": pid_params.get("yaw_rate_damping", 10.0),
            "nomoto_feedforward_enabled": pid_params.get("nomoto_feedforward_enabled", True),
            "nomoto_feedforward_gain": pid_params.get("nomoto_feedforward_gain", 0.05),
            "desired_yaw_time_constant_s": pid_params.get("desired_yaw_time_constant", 10.0),
            "desired_heading_rate_limit_deg_s": pid_params.get("desired_heading_rate_limit_deg_s", 5.0),
            "heading_reference_filter_alpha": pid_params.get("heading_reference_filter_alpha", 0.6),
            "integral_limit_rad_s": pid_params.get("integral_limit", 1.0),
        },
        "feedback_linearization": algorithm_config.feedback_linearization,
        "sliding_mode": algorithm_config.sliding_mode,
        "backstepping_dynamic_smc": algorithm_config.backstepping_dynamic_smc,
        "surge_pid": {
            "target_speed_mps": surge_config.target_speed,
            "kp": surge_config.kp,
            "ki": surge_config.ki,
            "kd": surge_config.kd,
            "integral_limit": surge_config.integral_limit,
            "feedforward_thrust": surge_config.feedforward_thrust,
            "adaptive_turn_slowdown_enabled": surge_config.adaptive_turn_slowdown_enabled,
            "min_turn_speed_mps": surge_config.min_turn_speed,
            "turn_slowdown_heading_error_deg": surge_config.turn_slowdown_heading_error_deg,
            "turn_slowdown_max_reduction": surge_config.turn_slowdown_max_reduction,
        },
        "output_limits": {
            "rudder_angle_limit_deg": output_limit_config.rudder_angle_limit_deg,
            "min_thrust": output_limit_config.min_thrust,
            "max_thrust": output_limit_config.max_thrust,
        },
        "rudder_response": asdict(rudder_response_config),
    }
    write_control_parameters(control_parameters, control_parameters_path)
    write_resolved_config(
        framework_config,
        control_config,
        {
            "duration": args.duration,
            "dt": args.dt,
            "lookahead": effective_lookahead,
            "lookahead_source": "cli" if args.lookahead is not None else "config",
            "configured_ship_length_m": configured_ship_length,
            "guidance_parameters": {
                "lookahead_distance": guidance_parameter_config.lookahead_distance,
                "lookahead_ship_length_multiplier": guidance_parameter_config.lookahead_ship_length_multiplier,
                "switch_distance": effective_switch_distance,
            },
            "controller": algorithm_config.controller,
            "guidance": algorithm_config.guidance,
            "vector_field": algorithm_config.vector_field,
            "path_simplification": {
                "enabled": simplification_config.enabled,
                "tolerance_m": simplification_config.tolerance_m,
                "raw_waypoint_count": len(unsimplified_waypoints),
                "simplified_waypoint_count": len(waypoints),
            },
            "heading_pid": {
                "kp": heading_kp,
                "ki": heading_ki,
                "kd": heading_kd,
                "yaw_rate_damping": pid_params.get("yaw_rate_damping", 10.0),
                "nomoto_feedforward_enabled": pid_params.get("nomoto_feedforward_enabled", True),
                "nomoto_feedforward_gain": pid_params.get("nomoto_feedforward_gain", 0.05),
                "desired_yaw_time_constant": pid_params.get("desired_yaw_time_constant", 10.0),
                "desired_heading_rate_limit_deg_s": pid_params.get("desired_heading_rate_limit_deg_s", 5.0),
                "heading_reference_filter_alpha": pid_params.get("heading_reference_filter_alpha", 0.6),
                "integral_limit": pid_params.get("integral_limit", 1.0),
            },
            "feedback_linearization": algorithm_config.feedback_linearization,
            "sliding_mode": algorithm_config.sliding_mode,
            "backstepping_dynamic_smc": algorithm_config.backstepping_dynamic_smc,
            "surge_pid": {
                "target_speed": surge_config.target_speed,
                "kp": surge_config.kp,
                "ki": surge_config.ki,
                "kd": surge_config.kd,
                "integral_limit": surge_config.integral_limit,
                "feedforward_thrust": surge_config.feedforward_thrust,
            },
            "output_limits": {
                "rudder_angle_limit_deg": output_limit_config.rudder_angle_limit_deg,
                "min_thrust": output_limit_config.min_thrust,
                "max_thrust": output_limit_config.max_thrust,
            },
            "rudder_response": asdict(rudder_response_config),
        },
        resolved_config_path,
    )

    # Spatial post-processing:
    # 1. show the intended route on the map,
    # 2. compare the actual closed-loop vessel trajectory to that route.
    plot_predefined_path_on_map(
        map_data,
        waypoints,
        viewport,
        predefined_map_path,
        original_waypoints=unsimplified_waypoints,
    )
    plot_trajectory_comparison_on_map(samples, map_data, waypoints, viewport, comparison_map_path)

    # Control post-processing: time histories and scalar tracking metrics.
    plot_results(samples, waypoints, metrics, viewport, plot_path)

    # The animation is optional because it is slower than static plots and can
    # be skipped during parameter sweeps with --no-gif.
    if not args.no_gif:
        render_animation_gif(samples, map_data, waypoints, viewport, gif_path)

    # The HTML file acts as a single entry point for reviewing a run.  It either
    # contains a Plotly dashboard or a static report with embedded PNG/GIF files.
    render_interactive_html(
        samples,
        map_data,
        waypoints,
        html_path,
        plot_path,
        predefined_map_path,
        comparison_map_path,
        None if args.no_gif else gif_path,
        csv_path,
        trajectory_csv_path,
        metrics_csv_path,
        metrics_json_path,
        resolved_config_path,
        metrics,
    )

    if args.open:
        # Opening a browser is opt-in.  The default behavior is file generation
        # only, which works in headless or remote execution environments.
        webbrowser.open(html_path.resolve().as_uri())

    final = samples[-1]
    print(f"Saved run outputs: {output_dir.resolve()}")
    print(f"Interactive dashboard: {html_path.resolve()}")
    print(f"Predefined path map: {predefined_map_path.resolve()}")
    print(f"Trajectory comparison map: {comparison_map_path.resolve()}")
    print(f"Static summary plot: {plot_path.resolve()}")
    if not args.no_gif:
        print(f"Trajectory animation: {gif_path.resolve()}")
    print(f"Simulation data CSV: {csv_path.resolve()}")
    print(f"Trajectory path CSV: {trajectory_csv_path.resolve()}")
    print(f"Resolved config JSON: {resolved_config_path.resolve()}")
    print(f"Control parameters JSON: {control_parameters_path.resolve()}")
    print(f"Metrics summary CSV: {metrics_csv_path.resolve()}")
    print(f"Metrics summary JSON: {metrics_json_path.resolve()}")
    print(
        "Final state: "
        f"x={final['x']:.1f} m, y={final['y']:.1f} m, "
        f"cross_track_error={final['cross_track_error']:.2f} m, "
        f"heading_error={np.degrees(final['heading_error']):.2f} deg"
    )


if __name__ == "__main__":
    main()
