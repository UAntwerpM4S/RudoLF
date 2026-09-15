"""
LOS guidance + PID path-following simulation.

This script is intentionally kept separate from the RL training pipeline so it
can be used as a classical-control baseline.  It uses the same vessel simulator
and the same Western Scheldt map CSV files as the main framework visualization.

Main processing flow:
    1. Read the framework settings from fw/configuration/config.py.
    2. Load the configured route and map contours from fw/environments/*.csv.
    3. Use LOS guidance to compute a desired heading from cross-track error.
    4. Use a PID heading controller to compute the rudder command.
    5. Use a surge-speed PID controller to compute the thrust command.
    6. Advance the configured vessel/dynamics model through ShipSimulator.
    7. Post-process the run into map figures, controller-performance figures,
       raw CSV data, scalar metrics, and an animated GIF.

Typical command:
    python -m fw.control.path_following_los_pid
"""

import argparse
import csv
import json
import os
import webbrowser
from dataclasses import dataclass, replace
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
    That model-based feed-forward gives the PID loop a sensible operating point
    for Nomoto.  When config.py selects another dynamics model, such as Fossen,
    the feed-forward is only a small initial bias and the PI feedback learns
    the thrust level needed to hold the target speed.
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
    switch_distance: float = 15.0

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


def path_to_waypoints(
    reference_path: np.ndarray,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
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


def run_simulation(
    control_config: ClassicalControlConfig,
    waypoints: Sequence[Point],
    duration: float,
    dt: float,
    lookahead_distance: float,
    switch_distance: float,
    target_speed: float,
    heading_pid_gains: Tuple[float, float, float],
    surge_pid_gains: Tuple[float, float, float],
) -> List[dict]:
    """
    Run the closed-loop LOS guidance and PID heading-control simulation.

    Control architecture:
        1. LOSGuidance projects the vessel position onto the active path
           segment and computes cross-track error.
        2. LOSGuidance converts cross-track error into a desired heading.
        3. PIDController converts heading error into a rudder command.
        4. SurgeSpeedPIDController converts speed error into a thrust command.
        5. ShipSimulator advances the configured dynamics model using
           rudder/thrust.

    The returned list contains a full time series of navigation, control, and
    vessel-state variables.  It is intentionally plain dictionaries so that the
    same data can be exported directly to CSV without an extra dependency.
    """

    ship = create_ship(control_config.ship_model)
    specs = ship.specifications

    # The default ship specification enforces a minimum forward thrust of 0.1.
    # With the simplified Nomoto surge equation, that minimum thrust would drive
    # the vessel to the surge velocity limit and make a 4 m/s closed-loop speed
    # target impossible.  For this classical-control speed experiment we allow
    # the propulsive command to reduce to zero, which gives the surge PID enough
    # authority to regulate below the simulator's 5 m/s speed cap.
    specs.min_thrust = 0.0

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

    guidance = LOSGuidance(
        waypoints,
        lookahead_distance=lookahead_distance,
        switch_distance=switch_distance,
    )
    heading_pid = PIDController(
        kp=heading_pid_gains[0],
        ki=heading_pid_gains[1],
        kd=heading_pid_gains[2],
        output_limit=specs.max_rudder_angle,
    )
    surge_pid = SurgeSpeedPIDController(
        kp=surge_pid_gains[0],
        ki=surge_pid_gains[1],
        kd=surge_pid_gains[2],
        target_speed=target_speed,
        min_thrust=specs.min_thrust,
        max_thrust=specs.max_thrust,
    )

    samples: List[dict] = []
    n_steps = int(duration / dt)

    for step_idx in range(n_steps + 1):
        state = simulator.state

        # Guidance layer: convert vessel position relative to the active path
        # segment into desired heading and signed cross-track error.
        desired_heading, cross_track_error, los_point = guidance.step((state.x, state.y))
        heading_error = wrap_angle(desired_heading - state.heading)

        # Control layer: a PID heading autopilot produces the rudder angle.
        # Positive/negative rudder commands are then normalized for ShipSimulator.
        rudder_angle = heading_pid.step(heading_error, dt)
        rudder_action = float(np.clip(rudder_angle / specs.max_rudder_angle, -1.0, 1.0))

        # Speed layer: a second PID autopilot regulates surge velocity by
        # commanding thrust.  The resulting thrust command is converted into the
        # normalized action range used by ShipSimulator.
        thrust, speed_error = surge_pid.step(state.u, dt)
        thrust_action = thrust_to_normalized(thrust, specs.min_thrust, specs.max_thrust)

        samples.append(
            {
                "time": step_idx * dt,
                "x": state.x,
                "y": state.y,
                "heading": state.heading,
                "desired_heading": desired_heading,
                "heading_error": heading_error,
                "cross_track_error": cross_track_error,
                "rudder_angle": rudder_angle,
                "rudder_action": rudder_action,
                "target_speed": target_speed,
                "speed_error": speed_error,
                "thrust": thrust,
                "thrust_action": thrust_action,
                "u": state.u,
                "v": state.v,
                "r": state.r,
                "segment_index": guidance.segment_index,
                "los_x": los_point[0],
                "los_y": los_point[1],
            }
        )

        # Stop once the final waypoint is reached.  This keeps the post-run
        # metrics focused on path following, not on sailing beyond the target.
        if guidance.reached_goal((state.x, state.y)):
            break

        simulator.step(np.array([rudder_action, thrust_action], dtype=np.float32))

    return samples


def write_csv(samples: Iterable[dict], path: Path) -> None:
    """
    Write the complete time-series simulation data to CSV.

    This file includes trajectory, guidance errors, control commands, and vessel
    state variables, so it is the primary data export for offline analysis.
    """

    rows = list(samples)
    if not rows:
        raise ValueError("No simulation samples to write.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_trajectory_csv(samples: Sequence[dict], path: Path) -> None:
    """
    Write a compact x/y trajectory CSV for quick plotting or GIS-style import.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["time", "x", "y"])
        writer.writeheader()
        for row in samples:
            writer.writerow({"time": row["time"], "x": row["x"], "y": row["y"]})


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


def create_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"los_pid_path_following_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    ax.plot(
        waypoint_arr[:, 0],
        waypoint_arr[:, 1],
        "g--",
        linewidth=2.0,
        alpha=0.85,
        label="Predefined path",
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
    speeds = np.array([row["u"] for row in samples])
    target_speeds = np.array([row["target_speed"] for row in samples])
    speed_errors = np.array([row["speed_error"] for row in samples])
    thrust_commands = np.array([row["thrust"] for row in samples])
    headings, desired_headings = continuous_heading_tracking_series(samples)

    waypoint_x = [point[0] for point in waypoints]
    waypoint_y = [point[1] for point in waypoints]

    figure, axes = plt.subplots(4, 2, figsize=(13, 14), constrained_layout=True)

    axes[0, 0].plot(waypoint_x, waypoint_y, "k--", linewidth=1.5, label="reference path")
    axes[0, 0].plot(xs, ys, "tab:blue", linewidth=1.8, label="vessel trajectory")
    axes[0, 0].scatter(waypoint_x, waypoint_y, c="tab:red", s=28, zorder=3, label="waypoints")
    apply_map_viewport(axes[0, 0], viewport)
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].set_title("LOS path following")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(times, cross_track_errors, "tab:orange")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("cross-track error [m]")
    axes[0, 1].set_title("Cross-track error")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(times, headings, label="heading")
    axes[1, 0].plot(times, desired_headings, "--", label="desired heading")
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

    axes[2, 0].plot(times, rudder_angles, "tab:green")
    axes[2, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("rudder angle [deg]")
    axes[2, 0].set_title("PID rudder command")
    axes[2, 0].grid(True, alpha=0.3)

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
    speeds = np.array([row["u"] for row in samples])
    target_speeds = np.array([row["target_speed"] for row in samples])
    speed_errors = np.array([row["speed_error"] for row in samples])
    thrust_commands = np.array([row["thrust"] for row in samples])
    headings, desired_headings = continuous_heading_tracking_series(samples)
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
            y=desired_headings,
            mode="lines",
            name="desired heading [deg]",
            line=dict(dash="dash"),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=times, y=rudder_angles, mode="lines", name="rudder [deg]"),
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
    parser.add_argument("--switch-distance", type=float, default=35.0, help="Waypoint switching distance [m].")
    parser.add_argument("--target-speed", type=float, default=4.0, help="Target surge speed [m/s].")
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
    parser.add_argument("--kp", type=float, default=1.5, help="Heading PID/PD proportional gain.")
    parser.add_argument("--ki", type=float, default=0.0, help="Heading PID integral gain; default 0.0 gives PD control.")
    parser.add_argument("--kd", type=float, default=20.0, help="Heading PID/PD derivative gain.")
    # These default surge gains are tuned for the current config.py baseline
    # (ShipModel.MYZAKO_37 with Model.FOSSEN).  They remain command-line
    # options because other ship/dynamics combinations can require different
    # thrust-loop tuning.
    parser.add_argument("--surge-kp", type=float, default=0.8, help="Surge-speed PID proportional gain.")
    parser.add_argument("--surge-ki", type=float, default=0.01, help="Surge-speed PID integral gain.")
    parser.add_argument("--surge-kd", type=float, default=0.0, help="Surge-speed PID derivative gain.")
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

    # CLI map-file options are explicit overrides.  If they are omitted, the
    # same filenames selected by config.py/profile application are used.
    if args.overall_contour is not None:
        control_config = replace(control_config, overall_contour=args.overall_contour)
    if args.depth_contour is not None:
        control_config = replace(control_config, depth_contour=args.depth_contour)
    if args.trajectory is not None:
        control_config = replace(control_config, trajectory=args.trajectory)

    configured_ship_length = create_ship(control_config.ship_model).specifications.length
    effective_lookahead = args.lookahead if args.lookahead is not None else 3.0 * configured_ship_length

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
    # LOS guidance.  The helper trims the route to the configured start/target.
    waypoints = path_to_waypoints(
        map_data.reference_path,
        start_pos=control_config.start_pos,
        target_pos=control_config.target_pos,
    )

    # Run the closed-loop controller.  At this point no plots are generated; we
    # first collect a clean numerical time series and then post-process it.
    samples = run_simulation(
        control_config=control_config,
        waypoints=waypoints,
        duration=args.duration,
        dt=args.dt,
        lookahead_distance=effective_lookahead,
        switch_distance=args.switch_distance,
        target_speed=args.target_speed,
        heading_pid_gains=(args.kp, args.ki, args.kd),
        surge_pid_gains=(args.surge_kp, args.surge_ki, args.surge_kd),
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
    viewport = make_square_viewport(
        [map_data.overall, map_data.obstacles, waypoint_arr, actual_path_arr]
    )

    # Persist numerical outputs before plotting.  This makes the run useful
    # even if a later optional renderer fails on a local graphics dependency.
    write_csv(samples, csv_path)
    write_trajectory_csv(samples, trajectory_csv_path)
    write_metrics_summary(metrics, metrics_csv_path, metrics_json_path)
    write_resolved_config(
        framework_config,
        control_config,
        {
            "duration": args.duration,
            "dt": args.dt,
            "lookahead": effective_lookahead,
            "lookahead_source": "cli" if args.lookahead is not None else "3x_configured_ship_length",
            "configured_ship_length_m": configured_ship_length,
            "switch_distance": args.switch_distance,
            "target_speed": args.target_speed,
            "heading_pid": {"kp": args.kp, "ki": args.ki, "kd": args.kd},
            "surge_pid": {
                "kp": args.surge_kp,
                "ki": args.surge_ki,
                "kd": args.surge_kd,
            },
        },
        resolved_config_path,
    )

    # Spatial post-processing:
    # 1. show the intended route on the map,
    # 2. compare the actual closed-loop vessel trajectory to that route.
    plot_predefined_path_on_map(map_data, waypoints, viewport, predefined_map_path)
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
