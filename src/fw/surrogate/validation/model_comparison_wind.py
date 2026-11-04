"""
Wind counterpart of model_comparison.generate_trajectories(): compares all
dynamics models against FH (ground truth) for a ship sailing straight (rudder
0, fixed thrust) in open water, over a grid of wind speed x relative wind
direction.

Models:
    FH          - simexe engine with the wind field applied.
    FH_nowind   - same exercise with wind-field Scale = 0: how straight FH
                  sails without wind (run once, drawn in every subplot).
    UGent, Fossen, UA_clean, UA_confined - wind-agnostic models.
    UA_full     - the only non-FH model with wind inputs. Each step it gets
                  FH's world-frame true wind seen from its own state (heading,
                  surge, sway), so a diverging UA_full is not steered by FH's
                  apparent wind. UKC from the lookup CSV at its own position
                  (UkcSource); current from FH (off). Injected via
                  _inject_live_environment.

A no-wind baseline run (Scale = 0) for all models gives FH_nowind and
UA_full_nowind; the summary's paired columns compare each model's wind effect
(model in wind minus the same model in the baseline) with FH's. Every run
also writes an env_*.csv with FH's true/apparent wind, UKC, sway and yaw rate
and the inputs each UA_* model got.

Non-FH models run in their own ShipSimulator, which adds no wind or current:
wind-agnostic models stay wind-blind and UA_full gets wind exactly once.

Test layout: one open-water start pose; initial surge WIND_TRAJECTORY_SPEED,
thrust = min(1, v/6), rudder 0 (as model_comparison.py); per wind speed one
PNG with a subplot per direction in REL_WIND_DIRS_DEG plus a lateral
deviation PNG; every model runs as many steps as FH needs for
TARGET_PATH_LENGTH (capped at MAX_STEPS).

Wind conventions (measured with diagnose_fh_wind_standalone.py):
    FH's WindState is the apparent wind, direction relative to the bow,
    positive = from starboard (clockwise). Only with this convention does the
    earth-fixed true wind stay constant in turning generator episodes. Grid
    speeds and directions are true wind: apparent = true + (u, -v) in (ahead,
    starboard) components, since sway v is positive to port; hence
    TRUE_WIND_SWAY_SIGN = -1. UA_full gets apparent wind in the convention of
    its training data.

Setting a target wind in FH:
    FH has no wind speed/direction setter; the exercise's wind field is
    rotated/scaled via WindFieldTransform, passed as
    FhSession(wind_field=(rotation, scale)) before reset(), so each probe and
    run is its own FH session. WindState lags after reset(), so every session
    first runs WIND_WARMUP_TIME and all models start from FH's state after
    that. calibrate_wind_field() derives the true wind per unit Scale and the
    rotation sign, and checks linearity, zero wind at Scale = 0 and
    steadiness. Labels and CSVs use the wind FH actually delivered.

Open-water and domain checks (diagnostic; trajectories are not cut short):
    - start pose: all ray features 0, else RuntimeError (REQUIRE_OPEN_WATER_START);
    - every OPEN_WATER_CHECK_EVERY steps each model's pose is checked the same
      way; the first hit per model goes to the summary CSV;
    - FH's state against the generator's stop limits (_training_domain_issue);
    - FH's current below MAX_CURRENT_SPEED after warm-up and every step,
      else WindTestSetupError.

Validated against FH for calibration and the full grid; the no-wind
baseline, paired metrics and own-position UKC were only run with stubbed fw
modules.

Usage:
    from fw.surrogate.validation.model_comparison_wind import generate_wind_trajectories
    generate_wind_trajectories(
        start_x=..., start_y=..., start_heading_deg=...,
        exercise_path=Path(".../DDShip_Tripoli_scenario_windonly.tab"),
    )
start_heading_deg uses the convention model_comparison.py passes to FhSession
(degrees of PySimEnv.ship_heading).
"""

import csv
import logging
import sys
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fw.ships.ship_factory import create_ship
from fw.ships.vessel_state import VesselState
from fw.environments.py_sim_env import PySimEnv
from fw.environments.ray_features import RAY_ANGLES_DEG, compute_ray_features, ray_world_angles
from fw.ships.actuator_command import ActuatorCommand
from fw.surrogate.sim_fh_dataset_generator import _cutoff_geometry
from fw.surrogate.validation.model_comparison import (
    FhSession,
    OUTPUT_DIR,
    RUDDER_ANGLE_RAD,
    SHIP_MODEL,
    SURROGATE_DT,
    TRAJECTORY_COLORS,
    TRAJECTORY_LINESTYLES,
    _inject_live_environment,
    _training_domain_issue,
    SurgeClipCounter,
    build_reference_models,
    build_ua_models,
    make_model_simulator,
    plt,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Test parameters
# --------------------------------------------------------------------------

WIND_SPEEDS = (5.0, 10.0, 15.0, 20.0)                 # target true wind speed [m/s], one PNG each
REL_WIND_DIRS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)  # target relative direction (FH frame), one subplot each

WIND_TRAJECTORY_SPEED = 5                             # initial surge speed [m/s] (the ship-bank test uses its own TRAJECTORY_SPEED)
TARGET_PATH_LENGTH = 500.0                            # [m] stop once FH's own path length reaches this
MAX_STEPS = 8000                                      # safety cap (8000 * 0.1 s = 800 s)

CAL_STEADY_CHECK_TIME = 5.0        # [s] first calibration probe is read again after this long
WIND_WARMUP_TIME = 1.0             # [s] FH time after reset() before WindState shows the transform

# apparent = true + (u, TRUE_WIND_SWAY_SIGN * v) in (ahead, starboard) components;
# v is port positive. None = sway not corrected. The calibration prints an estimate.
TRUE_WIND_SWAY_SIGN: Optional[float] = -1.0

# As sim_fh_dataset_generator, where the wind-field transform works; FhSession never rewinds.
FH_REWIND_ENABLED = False

# Wind-only (no current) Tripoli exercise; or pass exercise_path to generate_wind_trajectories().
WIND_EXERCISE_PATH: Optional[Path] = None

REQUIRE_OPEN_WATER_START = True    # RuntimeError if any ray hits at the start pose
OPEN_WATER_CHECK_EVERY = 10        # check every model's rays every N steps (1 s)
MAX_CURRENT_SPEED = 0.05           # [m/s] FH current above this -> RuntimeError (current should be off)

# Calibration tolerances (warnings only)
CAL_MIN_BASE_SPEED = 0.1           # [m/s] below this at Scale=1 the exercise effectively has no wind -> error
CAL_ROTATION_TOL_DEG = 5.0         # Rotation=90 must change Direction by 90 +- this
CAL_LINEARITY_TOL = 0.05           # relative tolerance for speed(Scale=2) == 2 * speed(Scale=1)
CAL_ZERO_WIND_TOL = 0.2            # [m/s] Scale=0 must give (near) zero TRUE wind
CAL_STEADY_TOL_DEG = 2.0           # repeated probe: direction change tolerance
CAL_STEADY_TOL_SPEED = 0.05        # repeated probe: relative speed change tolerance
RUN_DIR_TOL_DEG = 5.0              # per-run actual vs. target relative direction tolerance
RUN_SPEED_TOL = 0.05               # per-run actual vs. target wind speed (relative)
RUN_ABORT_DIR_DEG = 15.0           # per-run true wind further off its target than this -> WindTestSetupError
RUN_ABORT_SPEED_REL = 0.2          # (the transform did not produce the requested wind)

# Upper wind speed of the UA_full training data (e.g. p95 of column 7); None disables the flag.
UA_FULL_MAX_TRAINED_WIND_SPEED: Optional[float] = None

WIND_OUTPUT_DIR = OUTPUT_DIR / "wind_trajectories"

# UKC for UA_confined/UA_full at their own position (UkcSource): None = each
# model's preset lookup CSV; a path = that CSV for all of them.
UKC_LOOKUP_CSV: Optional[str] = None
CLEAN_OUTPUT_DIR = True            # remove this script's previous outputs at the start of a run
ONESTEP_CHECK = True               # teacher-forced one-step check of every model at FH's states
UKC_LOOKUP_MAX_DISTANCE: Optional[float] = 50.0   # [m] UKC_LOOKUP_CSV only: farther -> miss

COLORS = {**TRAJECTORY_COLORS, "FH_nowind": "#6b6b6b", "UA_full_nowind": "#7fbf7f"}
LINESTYLES = {**TRAJECTORY_LINESTYLES, "FH_nowind": (0, (8, 3)), "UA_full_nowind": (0, (8, 3))}
MODEL_ORDER = ["FH", "FH_nowind", "UGent", "Fossen", "UA_clean", "UA_confined", "UA_full", "UA_full_nowind"]

class WindTestSetupError(RuntimeError):
    """
    Fail-fast setup error (current present, start not in open water, no wind
    field); applies to every run, so the driver re-raises instead of skipping.
    """

_RAY_IDX_BOW = int(np.flatnonzero(np.isclose(np.asarray(RAY_ANGLES_DEG, dtype=float), 0.0))[0])
_RAY_IDX_STBD = int(np.flatnonzero(np.isclose(np.asarray(RAY_ANGLES_DEG, dtype=float), 90.0))[0])


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _wrap180(angle_deg: float) -> float:
    """
    Wrap to (-180, 180].
    """

    a = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def _wrap360(angle_deg: float) -> float:
    """
    Wrap an angle in degrees to [0, 360).
    """

    return float(angle_deg) % 360.0


@dataclass
class TrackPoint:
    """
    One point of a model's path.
    """

    t: float
    x: float
    y: float
    heading: float   # rad, PySimEnv convention
    u: float         # surge [m/s]


def _bow_and_starboard(heading_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    World-frame bow (ray_0) and starboard (ray_90) unit vectors, from the shared ray module.
    """

    angles = ray_world_angles(float(heading_rad))
    bow = np.array([np.cos(angles[_RAY_IDX_BOW]), np.sin(angles[_RAY_IDX_BOW])])
    stbd = np.array([np.cos(angles[_RAY_IDX_STBD]), np.sin(angles[_RAY_IDX_STBD])])
    return bow, stbd


def track_metrics(start: TrackPoint, point: TrackPoint) -> Dict[str, float]:
    """
    Position/heading of `point` relative to the start pose's straight track.

    along: distance along the initial heading [m]; lateral: off the initial
    track, starboard positive [m]; turn: heading change, starboard positive
    [deg]; u: surge [m/s].
    """

    bow0, stbd0 = _bow_and_starboard(start.heading)
    d = np.array([point.x - start.x, point.y - start.y])
    bow1, _ = _bow_and_starboard(point.heading)
    return {
        "along": float(d @ bow0),
        "lateral": float(d @ stbd0),
        "turn": float(np.degrees(np.arctan2(bow1 @ stbd0, bow1 @ bow0))),
        "u": float(point.u),
    }


def _rays_hit(ray_env: PySimEnv, x: float, y: float, heading_rad: float) -> bool:
    """
    True if any ray at this pose sees a bank within RAY_MAX_DISTANCE (any feature > 1e-6).
    """

    features = compute_ray_features(ray_env, (float(x), float(y)), float(heading_rad))
    return bool(np.any(np.asarray(features) > 1e-6))


def _fh_point(fh: FhSession, t: float) -> TrackPoint:
    """
    FH's current state as a TrackPoint at time `t`.
    """

    x, y, h, u, _v, _r = fh.state()
    return TrackPoint(t=t, x=x, y=y, heading=h, u=u)


def _check_current(fh: FhSession, where: str) -> Tuple[float, float]:
    """
    FH's current (speed, direction); raises WindTestSetupError above MAX_CURRENT_SPEED.
    """

    speed, direction = fh.get_current()
    if speed > MAX_CURRENT_SPEED:
        raise WindTestSetupError(
            f"{where}: FH reports a current of {speed:.3f} m/s (> MAX_CURRENT_SPEED="
            f"{MAX_CURRENT_SPEED}) -- the wind-only exercise should have current "
            f"disabled; check the .tab file."
        )
    return speed, direction


# --------------------------------------------------------------------------
# Wind-field calibration
# --------------------------------------------------------------------------
# Measured: true wind is linear in Scale (~1.11 m/s per unit at the test pose)
# and Rotation=90 turns it by -90 deg; WindState shows the requested transform
# by t = 1 s after reset(). True wind = apparent - (u, k*v), k = TRUE_WIND_SWAY_SIGN.

def _vec(speed: float, dir_deg: float) -> np.ndarray:
    """
    2-D vector of length `speed` at angle `dir_deg` [deg] (cos, sin components).
    """

    return float(speed) * np.array([np.cos(np.radians(dir_deg)), np.sin(np.radians(dir_deg))])


def _speed_dir(vec: np.ndarray) -> Tuple[float, float]:
    """
    (length, angle [deg, 0-360)) of a 2-D vector; inverse of _vec.
    """

    return float(np.hypot(vec[0], vec[1])), _wrap360(np.degrees(np.arctan2(vec[1], vec[0])))


def _true_wind(fh: FhSession) -> Tuple[float, float, float, float]:
    """
    (true speed, true dir, apparent speed, apparent dir); directions [deg] relative to the bow.
    """

    app_speed, app_dir = fh.get_wind_raw()
    _, _, _, u, v, _ = fh.state()
    lateral = 0.0 if TRUE_WIND_SWAY_SIGN is None else TRUE_WIND_SWAY_SIGN * v
    true_speed, true_dir = _speed_dir(_vec(app_speed, app_dir) - np.array([u, lateral]))
    return true_speed, true_dir, app_speed, app_dir


def _speed_dir_only(t: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    (true speed, true dir) of a _true_wind tuple.
    """

    return t[0], t[1]


def _sway_sign() -> float:
    """
    TRUE_WIND_SWAY_SIGN as a float; 0.0 when it is None (sway not corrected).
    """

    return 0.0 if TRUE_WIND_SWAY_SIGN is None else float(TRUE_WIND_SWAY_SIGN)


def _true_wind_world(fh: FhSession) -> np.ndarray:
    """
    FH's true wind as a world-frame 'coming from' vector [m/s].
    """

    app_speed, app_dir = fh.get_wind_raw()
    _, _, heading, u, v, _ = fh.state()
    t = _vec(app_speed, app_dir) - np.array([u, _sway_sign() * v])
    bow, stbd = _bow_and_starboard(heading)
    return t[0] * bow + t[1] * stbd


def _apparent_wind_for(state, w_world: np.ndarray) -> Tuple[float, float]:
    """
    Apparent wind (speed [m/s], direction [rad], as FhSession.get_wind()) for a
    ship in `state` given the world-frame true wind (inverse of _true_wind_world).
    """

    bow, stbd = _bow_and_starboard(float(state.heading))
    rel = np.array([w_world @ bow, w_world @ stbd]) + np.array([float(state.u), _sway_sign() * float(state.v)])
    speed, dir_deg = _speed_dir(rel)
    return speed, float(PySimEnv.normalize_angle(np.radians(dir_deg)))


@dataclass
class WindCalibration:
    """
    Linear mapping from a target true wind to the FH wind-field transform, from calibrate_wind_field.
    """

    base_speed: float       # true wind speed at Scale=1, Rotation=0 [m/s]
    base_dir_deg: float     # its direction relative to the bow [deg, WindState convention]
    rotation_sign: float    # +1: increasing Rotation increases that direction; -1: decreases it
    summary: str = ""       # one-line description for the report

    def transform_for(self, wind_speed: float, rel_dir_deg: float) -> Tuple[float, float]:
        """
        (Rotation [deg], Scale) for a true wind of `wind_speed` at `rel_dir_deg` from the bow.
        """

        rotation = _wrap360(self.rotation_sign * _wrap180(rel_dir_deg - self.base_dir_deg))
        scale = float(wind_speed) / self.base_speed
        return rotation, scale


def derive_calibration(t_0: Tuple[float, float], t_90: Tuple[float, float],
                       t_scale2: Tuple[float, float], t_zero: Tuple[float, float],
                       t_repeat: Tuple[float, float], heading_change_deg: float = 0.0) -> WindCalibration:
    """
    Calibration from five probes, each a true wind (speed, rel_dir_deg) after
    WIND_WARMUP_TIME. Raises if the transform has no effect; warns on broken assumptions.
    """

    s0, d0 = t_0
    delta = _wrap180(t_90[1] - d0)

    def same(a, b):
        """
        True if two (speed, dir) winds agree within the CAL_STEADY tolerances.
        """

        return abs(a[0] - b[0]) <= CAL_STEADY_TOL_SPEED * max(a[0], 1e-6) and abs(_wrap180(a[1] - b[1])) <= CAL_STEADY_TOL_DEG
    if same(t_0, t_90) and same(t_0, t_scale2) and same(t_0, t_zero):
        raise WindTestSetupError(
            f"calibrate_wind_field: the wind-field transform has NO effect -- Rotation/Scale 0/1, 90/1, "
            f"0/2 and 0/0 all give the same true wind ({t_0[0]:.3f} m/s @ {t_0[1]:.1f} deg)."
        )
    if s0 < CAL_MIN_BASE_SPEED:
        raise WindTestSetupError(
            f"calibrate_wind_field: true wind at Scale=1 is only {s0:.3f} m/s (< {CAL_MIN_BASE_SPEED}) "
            f"-- the transform has no effect or the exercise has no wind field (Scale=2 gave "
            f"{t_scale2[0]:.3f} m/s)."
        )
    if abs(abs(delta) - 90.0) > CAL_ROTATION_TOL_DEG:
        logger.warning(
            "calibrate_wind_field: Rotation=90 changed the true wind direction by %.2f deg "
            "(expected +-90); per-run actual directions will deviate from the targets.", delta,
        )
    sign = 1.0 if delta >= 0.0 else -1.0
    if abs(t_scale2[0] - 2.0 * s0) > CAL_LINEARITY_TOL * 2.0 * s0:
        logger.warning(
            "calibrate_wind_field: true wind at Scale=2 is %.3f m/s, expected %.3f (2 x Scale=1); "
            "per-run actual speeds will deviate from the targets.", t_scale2[0], 2.0 * s0,
        )
    if t_zero[0] > CAL_ZERO_WIND_TOL:
        logger.warning(
            "calibrate_wind_field: Scale=0 still leaves %.3f m/s of true wind -- FH_nowind is not "
            "a clean no-wind reference.", t_zero[0],
        )
    # Heading is CCW and WindState angles CW, so a steady earth-fixed wind
    # shifts the bow-relative direction by +heading_change.
    d_rel = _wrap180(t_repeat[1] - d0)
    dir_dev = abs(_wrap180(d_rel - heading_change_deg))
    if dir_dev > CAL_STEADY_TOL_DEG or abs(t_repeat[0] - s0) > CAL_STEADY_TOL_SPEED * s0:
        logger.warning(
            "calibrate_wind_field: true wind at Scale=1 changed from (%.3f m/s, %.2f deg) to "
            "(%.3f m/s, %.2f deg) after %g s (ship heading changed %.2f deg meanwhile) -- the field is "
            "not steady (or not uniform); see the logged min/max per run.",
            s0, d0, t_repeat[0], t_repeat[1], CAL_STEADY_CHECK_TIME, heading_change_deg,
        )
    return WindCalibration(base_speed=float(s0), base_dir_deg=float(d0), rotation_sign=sign)


def _wind_session(start_x, start_y, start_heading_deg, v_init, thrust, ship_spec, exercise_path,
                  rotation: float, scale: float, where: str) -> FhSession:
    """
    FH session with the wind-field transform, advanced WIND_WARMUP_TIME.
    Raises if WindState is still unchanged since reset().
    """

    fh = FhSession(x=start_x, y=start_y, heading_deg=start_heading_deg, v_init=v_init,
                   thrust=thrust, ship_spec=ship_spec, exercise_path=exercise_path,
                   wind_field=(rotation, scale), rewind_enabled=FH_REWIND_ENABLED)
    try:
        initial = fh.get_wind_raw()
        fh.advance(WIND_WARMUP_TIME)
        after = fh.get_wind_raw()
        # A stale WindState is bit-identical, so exact equality is the test. A
        # live one ignoring the transform is caught by the calibration and RUN_ABORT_* checks.
        if initial == after:
            raise WindTestSetupError(
                f"{where}: WindState is unchanged after WIND_WARMUP_TIME={WIND_WARMUP_TIME} s "
                f"({after}) -- it still shows the wind from right after reset(). Increase "
                f"WIND_WARMUP_TIME."
            )
        _check_current(fh, where)
    except Exception:
        fh.close()
        raise
    return fh


def calibrate_wind_field(start_x: float, start_y: float, start_heading_deg: float,
                         v_init: float, thrust: float, ship_spec,
                         exercise_path: Path) -> WindCalibration:
    """
    Probe the true wind at the start pose for Rotation/Scale 0/1, 90/1, 0/2 and
    0/0, each in its own FH session; 0/1 is re-read after CAL_STEADY_CHECK_TIME.
    """

    common = (start_x, start_y, start_heading_deg, v_init, thrust, ship_spec, exercise_path)

    fh = _wind_session(*common, 0.0, 1.0, "calibrate_wind_field R0/S1")
    try:
        print(f"calibrate_wind_field: FH loaded exercise {fh._bridge.exercise_path} "
              f"(rewind_enabled={fh._bridge.rewind_enabled})", flush=True)
        t0 = _true_wind(fh)
        _, _, h0, u0, v0, _ = fh.state()
        fh.advance(CAL_STEADY_CHECK_TIME)
        trep = _true_wind(fh)
        _, _, h1, u1, v1, _ = fh.state()
        dpsi = _wrap180(np.degrees(h1 - h0))
        # Sway check: in a steady field the apparent lateral wind changes by +-dv.
        a0, a1 = _vec(t0[2], t0[3]), _vec(trep[2], trep[3])
        d_lat, dv = float(a1[1] - a0[1]), float(v1 - v0)
        ratio = d_lat / dv if abs(dv) > 1e-3 else float("nan")
        print(f"calibrate_wind_field: sway check over {CAL_STEADY_CHECK_TIME:g} s: u {u0:.3f}->{u1:.3f}, "
              f"v {v0:+.4f}->{v1:+.4f} m/s (port +); apparent lateral component changed {d_lat:+.4f} m/s; "
              f"ratio d_lateral/dv = {ratio:+.2f} (about +1 or -1 -> sway explains the drift, sign = "
              f"TRUE_WIND_SWAY_SIGN; far from +-1 -> the field itself varies). "
              f"TRUE_WIND_SWAY_SIGN currently {TRUE_WIND_SWAY_SIGN}.", flush=True)
    finally:
        fh.close()

    def probe(rotation: float, scale: float):
        """
        True wind (as _true_wind) of a fresh session with the given wind-field transform.
        """

        fh_probe = _wind_session(*common, rotation, scale, f"calibrate_wind_field R{rotation:g}/S{scale:g}")
        try:
            return _true_wind(fh_probe)
        finally:
            fh_probe.close()

    t90, t2, tz = probe(90.0, 1.0), probe(0.0, 2.0), probe(0.0, 0.0)
    cal = derive_calibration(_speed_dir_only(t0), _speed_dir_only(t90), _speed_dir_only(t2),
                             _speed_dir_only(tz), _speed_dir_only(trep), heading_change_deg=dpsi)
    cal.summary = (f"{cal.base_speed:.3f} m/s true wind per unit Scale, from {cal.base_dir_deg:.1f} deg rel. bow at "
                   f"Rotation 0; rotation sign {cal.rotation_sign:+.0f}; Scale 0 leaves {tz[0]:.3f} m/s")

    def fmt(t):
        """
        Text of a _true_wind tuple for the calibration log line.
        """

        return f"true {t[0]:.3f}@{t[1]:.1f} (apparent {t[2]:.3f}@{t[3]:.1f})"

    print(
        f"calibrate_wind_field: base true wind {cal.base_speed:.3f} m/s per unit Scale from "
        f"{cal.base_dir_deg:.1f} deg rel. bow, rotation sign {cal.rotation_sign:+.0f}; "
        f"R0/S1 {fmt(t0)} | R90/S1 {fmt(t90)} | R0/S2 {fmt(t2)} | R0/S0 {fmt(tz)} | "
        f"R0/S1 after {CAL_STEADY_CHECK_TIME:g} s {fmt(trep)} (heading change {dpsi:+.2f} deg)",
        flush=True,
    )
    return cal


# --------------------------------------------------------------------------
# UKC at each model's own position
# --------------------------------------------------------------------------

class UkcSource:
    """
    UKC for the UA_* models at their own (x, y), so a model on a different
    path than FH is not steered by FH's UKC.

    Source: the shared UKC_LOOKUP_CSV if set (rows farther than
    UKC_LOOKUP_MAX_DISTANCE are misses: FH's UKC is used and counted), else each
    model's own wcu_lookup (nearest row, no distance limit). Models without a
    UKC input or without a lookup get FH's UKC.
    """

    def __init__(self, ua_models: dict):
        """
        Pick a lookup per UA model with a UKC input (shared CSV, else its own).
        """

        self._shared = None
        if UKC_LOOKUP_CSV is not None:
            from fw.dynamics.wcu_lookup import WcuLookup
            self._shared = WcuLookup(str(UKC_LOOKUP_CSV))
        self._lookups = {}
        for label, model in ua_models.items():
            scenario = getattr(model, "scenario", None)
            if scenario is None or not scenario.includes_ukc:
                continue
            lookup = self._shared if self._shared is not None else getattr(model, "wcu_lookup", None)
            if lookup is not None:
                self._lookups[label] = lookup
        self._models = ua_models

    def uses_ukc(self, label: str) -> bool:
        """
        True if model `label` has a UKC lookup.
        """

        return label in self._lookups

    def ukc(self, label: str, x: float, y: float) -> Optional[float]:
        """
        UKC [m] for `label` at (x, y); None without a lookup or on a shared-lookup miss.
        """

        lookup = self._lookups.get(label)
        if lookup is None:
            return None
        max_distance = UKC_LOOKUP_MAX_DISTANCE if lookup is self._shared else None
        return lookup.get_ukc(float(x), float(y), max_distance=max_distance)

    def validate(self, x: float, y: float) -> None:
        """
        Fail fast if a UKC-using model gets no UKC at the start position.
        """

        for label, model in self._models.items():
            scenario = getattr(model, "scenario", None)
            if not self.uses_ukc(label):
                if scenario is not None and scenario.includes_ukc:
                    print(f"UkcSource: {label} has no lookup CSV -- it gets FH's UKC.", flush=True)
                continue
            try:
                value = self.ukc(label, x, y)
            except Exception as exc:
                raise WindTestSetupError(
                    f"UkcSource: {label}'s UKC lookup failed at the start position ({x:.1f}, {y:.1f}): {exc!r}."
                ) from exc
            if value is None:
                raise WindTestSetupError(
                    f"UkcSource: {label}'s UKC lookup found no row within {UKC_LOOKUP_MAX_DISTANCE} m of the "
                    f"start position ({x:.1f}, {y:.1f}); set UKC_LOOKUP_CSV to a CSV that covers this area."
                )
            print(f"UkcSource: {label} UKC at start = {value:.3f} m "
                  f"({'UKC_LOOKUP_CSV' if self._shared is not None else 'own lookup CSV'})", flush=True)


# --------------------------------------------------------------------------
# One run (one wind setting, or the no-wind baseline): FH + all non-FH models
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Result of one wind-test run (or the no-wind baseline): FH and every model on one start pose.
    """

    run_id: str
    target_wind_speed: float          # 0 for the no-wind baseline
    target_rel_dir_deg: float         # nan for the no-wind baseline
    actual_wind_speed: float          # true wind at the start state
    actual_rel_dir_deg: float         # true wind direction rel. bow at the start state
    apparent_wind_speed: float        # FH WindState at the start state
    apparent_rel_dir_deg: float
    wind_speed_min: float             # true wind along FH's path
    wind_speed_max: float
    paths: Dict[str, List[TrackPoint]]
    env_log: List[dict]               # per step: FH wind/UKC/sway/yaw + per-UA inputs
    ukc_misses: Dict[str, int]        # UA model -> steps where its own-position UKC lookup found nothing
    first_bank_hit: Dict[str, Optional[int]]   # model -> first step with a ray hit (None: never)
    domain_exit: Optional[Tuple[int, str]]     # FH: first step outside the training domain
    hit_max_steps: bool
    is_baseline: bool = False         # the no-wind (Scale=0) reference run
    windstate_updates: int = 0        # FH WindState changes seen during the run (1 Hz)
    onestep: Dict[str, dict] = field(default_factory=dict)   # model -> teacher-forced one-step stats


def run_case(
        run_id: str, rotation: float, scale: float, target_speed: float, target_rel_deg: float,
        start_x: float, start_y: float, start_heading_deg: float,
        v_init: float, thrust: float, ship_spec, exercise_path: Path,
        ray_env: PySimEnv, cutoff_geom, non_fh_models: dict, ua_models: dict, ukc_source: UkcSource,
        is_baseline: bool = False, tf_models: Optional[dict] = None,
) -> RunResult:
    """
    Run FH with the given wind-field transform and every other model from the same
    start state until FH has sailed TARGET_PATH_LENGTH (at most MAX_STEPS).

    UA_* models get apparent wind (sampled on FH's 1 Hz WindState updates) and UKC
    for their own state; `tf_models` also get a teacher-forced one-step check.

    Raises:
        WindTestSetupError: If the actual true wind is far from the target (not for the
            baseline), FH reports a current above MAX_CURRENT_SPEED, or the start pose is not
            in open water while REQUIRE_OPEN_WATER_START is set.
    """

    fh = _wind_session(start_x, start_y, start_heading_deg, v_init, thrust, ship_spec,
                       exercise_path, rotation, scale, run_id)
    try:
        ws, rel_actual, app_ws, app_wd = _true_wind(fh)
        if not is_baseline:
            if (abs(_wrap180(rel_actual - target_rel_deg)) > RUN_ABORT_DIR_DEG
                    or abs(ws - target_speed) > RUN_ABORT_SPEED_REL * target_speed):
                raise WindTestSetupError(
                    f"{run_id}: true wind {ws:.2f} m/s @ {rel_actual:.1f} deg is far from the target "
                    f"{target_speed:g} m/s @ {target_rel_deg:g} deg -- the wind-field transform did not "
                    f"take effect for this run (Rotation={rotation:.1f}, Scale={scale:.3f})."
                )
            if abs(_wrap180(rel_actual - target_rel_deg)) > RUN_DIR_TOL_DEG:
                logger.warning("%s: actual true wind direction %.2f deg vs target %.2f deg",
                               run_id, rel_actual, target_rel_deg)
            if abs(ws - target_speed) > RUN_SPEED_TOL * target_speed:
                logger.warning("%s: actual true wind speed %.3f m/s vs target %.3f m/s", run_id, ws, target_speed)
        if UA_FULL_MAX_TRAINED_WIND_SPEED is not None and app_ws > UA_FULL_MAX_TRAINED_WIND_SPEED:
            print(f"{run_id}: apparent wind {app_ws:.2f} m/s is above UA_FULL_MAX_TRAINED_WIND_SPEED="
                  f"{UA_FULL_MAX_TRAINED_WIND_SPEED} -- UA_full is extrapolating.", flush=True)

        start = _fh_point(fh, 0.0)
        if _rays_hit(ray_env, start.x, start.y, start.heading):
            msg = (f"{run_id}: start pose ({start.x:.2f}, {start.y:.2f}) is not in open water -- "
                   f"at least one ray hits a bank within RAY_MAX_DISTANCE.")
            if REQUIRE_OPEN_WATER_START:
                raise WindTestSetupError(msg)
            print(msg, flush=True)

        init_state = dict(x=start.x, y=start.y, heading=start.heading, u=start.u)
        _, _, _, _, v0, r0 = fh.state()
        actuator = ActuatorCommand(rudder_angle=RUDDER_ANGLE_RAD, thrust=thrust)

        sims = {}
        for label, model in non_fh_models.items():
            sim = make_model_simulator(ship_spec, model)
            sim.reset(VesselState(terminated=False, truncated=False, v=v0, r=r0, **init_state))
            sims[label] = sim
        surge_clip = SurgeClipCounter(ship_spec, sims)

        paths = {"FH": [start]}
        paths.update({label: [TrackPoint(0.0, start.x, start.y, start.heading, start.u)] for label in sims})
        first_hit: Dict[str, Optional[int]] = {label: None for label in paths}
        ukc_misses = {label: 0 for label in ua_models}
        env_log: List[dict] = []
        domain_exit = None
        wind_speeds = [ws]
        fh_length = 0.0
        step_i = 0
        last_raw = None
        held_true = None
        held_apparent: Dict[str, Tuple[float, float]] = {}
        n_updates = 0
        tf_pred: Dict[str, list] = {label: [] for label in (tf_models or {})}
        tf_fh: list = []
        tf_pre: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # FH (u, v, r) before the step

        while fh_length < TARGET_PATH_LENGTH and step_i < MAX_STEPS:
            fh_ukc = fh.get_ukc()
            fh_cs, fh_cd = fh.get_current()
            if fh_cs > MAX_CURRENT_SPEED:
                raise WindTestSetupError(
                    f"{run_id}: FH current {fh_cs:.3f} m/s at step {step_i} (> MAX_CURRENT_SPEED="
                    f"{MAX_CURRENT_SPEED}) -- the wind-only exercise should have current disabled."
                )
            # WindState updates at 1 Hz: on a change, recompute the true wind (FH's
            # state) and each UA model's apparent wind (its own state) and hold
            # them, the same timing as in UA_full's training data.
            raw = fh.get_wind_raw()
            windstate_update = raw != last_raw
            if windstate_update:
                last_raw = raw
                held_true = _true_wind(fh)
                w_world = _true_wind_world(fh)
                for label in ua_models:
                    held_apparent[label] = _apparent_wind_for(sims[label].state, w_world)
                n_updates += 1
            t_ws, t_wd, a_ws, a_wd = held_true
            _, _, _, _, fh_v, fh_r = fh.state()
            wind_speeds.append(t_ws)
            cur = paths["FH"][-1]
            if domain_exit is None:
                issue = _training_domain_issue(ray_env, cutoff_geom, cur.x, cur.y, cur.heading, fh_ukc)
                if issue is not None:
                    domain_exit = (step_i, issue)

            log = dict(step=step_i, t=step_i * SURROGATE_DT, windstate_update=int(windstate_update),
                       fh_true_wind_speed=t_ws, fh_true_wind_dir_deg=t_wd,
                       fh_apparent_wind_speed=a_ws, fh_apparent_wind_dir_deg=a_wd, fh_ukc=fh_ukc,
                       fh_sway=fh_v, fh_yaw_rate=fh_r)

            # Teacher-forced one-step check: each model gets FH's state and
            # environment (as a training row) and its accelerations are compared
            # with FH's change over this step.
            if tf_models:
                _, _, fh_h, fh_u, _, _ = fh.state()
                fs = VesselState(terminated=False, truncated=False, x=cur.x, y=cur.y, heading=fh_h,
                                 u=fh_u, v=fh_v, r=fh_r)
                fw_s, fw_d = fh.get_wind()
                for label, model in tf_models.items():
                    if label in ua_models:
                        _inject_live_environment(model, fh_ukc, fw_s, fw_d, fh_cs, fh_cd, ray_env)
                    tf_pred[label].append(tuple(float(a) for a in model._compute_accelerations(fs, RUDDER_ANGLE_RAD, thrust)))
                tf_pre = (fh_u, fh_v, fh_r)

            # Held apparent wind and own-position UKC per UA model (only UA_full
            # uses wind, UA_confined only UKC).
            for label, model in ua_models.items():
                st = sims[label].state
                own_ws, own_wd = held_apparent[label]
                own_ukc = ukc_source.ukc(label, st.x, st.y) if ukc_source.uses_ukc(label) else None
                if own_ukc is None:
                    if ukc_source.uses_ukc(label):
                        ukc_misses[label] += 1
                    own_ukc = fh_ukc
                _inject_live_environment(model, own_ukc, own_ws, own_wd, fh_cs, fh_cd, ray_env)
                log[f"{label}_apparent_wind_speed"] = own_ws
                log[f"{label}_apparent_wind_dir_deg"] = float(np.degrees(own_wd))
                log[f"{label}_ukc"] = own_ukc
            env_log.append(log)

            t = (step_i + 1) * SURROGATE_DT
            for label, sim in sims.items():
                sim.integrate(actuator, SURROGATE_DT)
                s = sim.state
                paths[label].append(TrackPoint(t, float(s.x), float(s.y), float(s.heading), float(s.u)))
                surge_clip.update(label, float(s.u))

            fh.advance(SURROGATE_DT)
            if tf_models:
                _, _, _, u1, v1, r1 = fh.state()
                tf_fh.append(((u1 - tf_pre[0]) / SURROGATE_DT, (v1 - tf_pre[1]) / SURROGATE_DT,
                              (r1 - tf_pre[2]) / SURROGATE_DT))
            p = _fh_point(fh, t)
            fh_length += float(np.hypot(p.x - cur.x, p.y - cur.y))
            paths["FH"].append(p)
            step_i += 1

            if step_i % OPEN_WATER_CHECK_EVERY == 0:
                for label, pts in paths.items():
                    if first_hit[label] is None and _rays_hit(ray_env, pts[-1].x, pts[-1].y, pts[-1].heading):
                        first_hit[label] = step_i
    finally:
        fh.close()
    surge_clip.report(run_id)

    return RunResult(
        run_id=run_id, target_wind_speed=target_speed, target_rel_dir_deg=target_rel_deg,
        actual_wind_speed=ws, actual_rel_dir_deg=rel_actual,
        apparent_wind_speed=app_ws, apparent_rel_dir_deg=app_wd,
        wind_speed_min=float(min(wind_speeds)), wind_speed_max=float(max(wind_speeds)),
        paths=paths, env_log=env_log, ukc_misses=ukc_misses,
        first_bank_hit=first_hit, domain_exit=domain_exit, hit_max_steps=step_i >= MAX_STEPS,
        is_baseline=is_baseline, windstate_updates=n_updates,
        onestep=_onestep_stats(tf_pred, tf_fh),
    )


def _all_paths(result: RunResult, base: Optional[RunResult]) -> Dict[str, List[TrackPoint]]:
    """
    The run's paths plus FH_nowind and UA_full_nowind from the baseline (wind runs only).
    """

    paths = dict(result.paths)
    if base is not None and not result.is_baseline:
        paths["FH_nowind"] = base.paths["FH"]
        if "UA_full" in base.paths:
            paths["UA_full_nowind"] = base.paths["UA_full"]
    return paths


def _onestep_stats(tf_pred: Dict[str, list], tf_fh: list) -> Dict[str, dict]:
    """
    Per model: bias and RMS of the one-step acceleration error vs FH, plus FH's
    own RMS as scale (du, dv [m/s^2], dr [rad/s^2]; T [s]).
    """

    out = {}
    if not tf_fh:
        return out
    fh_a = np.asarray(tf_fh, dtype=float)
    for label, preds in tf_pred.items():
        if len(preds) != len(fh_a):
            continue
        err = np.asarray(preds, dtype=float) - fh_a
        out[label] = dict(n=len(err), T=len(err) * SURROGATE_DT,
                          bias=err.mean(axis=0).tolist(), rms=np.sqrt((err ** 2).mean(axis=0)).tolist(),
                          fh_rms=np.sqrt((fh_a ** 2).mean(axis=0)).tolist())
    return out


ONESTEP_HEADER = ["target_wind_speed", "target_rel_dir_deg", "model", "n_steps", "duration_s",
                  "bias_du", "bias_dv", "bias_dr", "rms_err_du", "rms_err_dv", "rms_err_dr",
                  "fh_rms_du", "fh_rms_dv", "fh_rms_dr"]


def onestep_rows(result: "RunResult") -> List[list]:
    """
    Rows for the one-step CSV (ONESTEP_HEADER), one per model with teacher-forced statistics.
    """

    rows = []
    for label in MODEL_ORDER:
        st = result.onestep.get(label)
        if not st:
            continue
        rows.append([result.target_wind_speed, result.target_rel_dir_deg, label, st["n"], st["T"],
                     *st["bias"], *st["rms"], *st["fh_rms"]])
    return rows


# --------------------------------------------------------------------------
# Output: CSVs
# --------------------------------------------------------------------------

SUMMARY_HEADER = [
    "target_wind_speed", "target_rel_dir_deg", "actual_wind_speed", "actual_rel_dir_deg",
    "apparent_wind_speed_start", "apparent_rel_dir_deg_start",
    "true_wind_speed_min_along_path", "true_wind_speed_max_along_path", "model", "n_steps", "duration_s",
    "final_along", "final_lateral", "final_turn_deg", "final_u",
    "lateral_error_vs_fh", "turn_error_vs_fh_deg", "u_error_vs_fh", "final_pos_error_vs_fh",
    "rms_lateral_error_vs_fh",
    # paired wind effect: (model in wind - same model in the no-wind baseline) vs. the same for FH
    "effect_n_steps", "wind_effect_lateral", "wind_effect_turn_deg",
    "fh_wind_effect_lateral", "fh_wind_effect_turn_deg",
    "wind_effect_lateral_error", "wind_effect_turn_error_deg",
    "ukc_lookup_misses", "first_bank_hit_step", "fh_domain_exit_step", "fh_domain_exit_reason",
]


def _metrics_at(pts: List[TrackPoint], n: int) -> Dict[str, float]:
    """
    track_metrics of point `n` relative to the path's start.
    """

    return track_metrics(pts[0], pts[n])


def summary_rows(result: RunResult, base: Optional[RunResult]) -> List[list]:
    """
    One row per model; errors are vs FH of the same run over the common steps.

    Wind runs: the paired wind-effect columns compare (model - model in
    baseline) with (FH - FH in baseline) at the shortest of the four paths. In
    wind runs the FH_nowind / UA_full_nowind *_error_vs_fh columns therefore
    show FH's wind effect, not a model error. Baseline rows (target wind 0)
    show each model's no-wind error vs FH_nowind.
    """

    rows = []
    fh_path = result.paths["FH"]
    fh_metrics = [track_metrics(fh_path[0], p) for p in fh_path]
    all_paths = _all_paths(result, base)
    for label in MODEL_ORDER:
        pts = all_paths.get(label)
        if not pts:
            continue
        n = min(len(pts), len(fh_path))
        m = [track_metrics(pts[0], p) for p in pts[:n]]
        fm = fh_metrics[:n]
        final, ffinal = m[-1], fm[-1]
        lat_err = np.array([a["lateral"] - b["lateral"] for a, b in zip(m, fm)])
        pos_err = float(np.hypot(pts[n - 1].x - fh_path[n - 1].x, pts[n - 1].y - fh_path[n - 1].y))

        effect = [""] * 7
        if base is not None and not result.is_baseline and label in result.paths and label in base.paths:
            b_pts, b_fh = base.paths[label], base.paths["FH"]
            ne = min(len(pts), len(b_pts), len(fh_path), len(b_fh)) - 1
            if ne > 0:
                e_lat = _metrics_at(pts, ne)["lateral"] - _metrics_at(b_pts, ne)["lateral"]
                e_turn = _wrap180(_metrics_at(pts, ne)["turn"] - _metrics_at(b_pts, ne)["turn"])
                f_lat = _metrics_at(fh_path, ne)["lateral"] - _metrics_at(b_fh, ne)["lateral"]
                f_turn = _wrap180(_metrics_at(fh_path, ne)["turn"] - _metrics_at(b_fh, ne)["turn"])
                effect = [ne, e_lat, e_turn, f_lat, f_turn, e_lat - f_lat, _wrap180(e_turn - f_turn)]

        rows.append([
            result.target_wind_speed, result.target_rel_dir_deg,
            result.actual_wind_speed, result.actual_rel_dir_deg,
            result.apparent_wind_speed, result.apparent_rel_dir_deg,
            result.wind_speed_min, result.wind_speed_max,
            label, n - 1, pts[n - 1].t,
            final["along"], final["lateral"], final["turn"], final["u"],
            final["lateral"] - ffinal["lateral"], _wrap180(final["turn"] - ffinal["turn"]),
            final["u"] - ffinal["u"], pos_err, float(np.sqrt(np.mean(lat_err ** 2))),
            *effect,
            result.ukc_misses.get(label, ""),
            result.first_bank_hit.get(label, ""),
            result.domain_exit[0] if result.domain_exit else "",
            result.domain_exit[1] if result.domain_exit else "",
        ])
    return rows


def write_trajectory_csv(path: Path, result: RunResult, base: Optional[RunResult]) -> None:
    """
    Write every model's path (including the no-wind reference lines) with its track metrics.
    """

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "step", "t", "x", "y", "heading", "u", "along", "lateral", "turn_deg"])
        for label, pts in ((lb, _all_paths(result, base).get(lb)) for lb in MODEL_ORDER):
            if not pts:
                continue
            for i, p in enumerate(pts):
                m = track_metrics(pts[0], p)
                writer.writerow([label, i, p.t, p.x, p.y, p.heading, p.u, m["along"], m["lateral"], m["turn"]])


def write_env_csv(path: Path, result: RunResult) -> None:
    """
    Write the per-step environment log (FH's wind/UKC/sway/yaw rate, each UA model's inputs).
    """

    if not result.env_log:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.env_log[0].keys()))
        writer.writeheader()
        writer.writerows(result.env_log)


# --------------------------------------------------------------------------
# Output: plots
# --------------------------------------------------------------------------

def _save(fig, png_path: Path) -> None:
    """
    Save the figure as a PNG and close it; a failed save is printed, not raised.
    """

    try:
        fig.savefig(str(png_path), dpi=150)   # str(): Windows/Pillow Path quirk, see model_comparison.py
        print(f"Wrote {png_path}", flush=True)
    except OSError as exc:
        print(f"Failed to save {png_path} ({exc}); the CSVs are unaffected.", flush=True)
    finally:
        plt.close(fig)


def _grid(n: int):
    """
    Figure with at most 4 columns of subplots for `n` panels: (fig, flat array of axes).
    """

    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(4.6 * ncols, 11.0), 4.8 * nrows), squeeze=False)
    return fig, axes.flatten()


def _legend_and_layout(fig, ax0, title: str) -> None:
    """
    Shared legend below the panels (from `ax0`), figure title and layout.
    """

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.subplots_adjust(hspace=0.4)


def _subplot_title(result: RunResult) -> str:
    """
    Panel title: target and actual relative wind direction and speed, or 'no wind' for the baseline.
    """

    if result.is_baseline:
        return f"no wind (Scale=0)\ntrue wind {result.actual_wind_speed:.2f} m/s"
    return (f"rel. wind {result.target_rel_dir_deg:g}° (actual {result.actual_rel_dir_deg:.0f}°)\n"
            f"{result.actual_wind_speed:.1f} m/s")


def _line_kwargs(label: str) -> dict:
    """
    Plot style for a model label (FH drawn thicker and on top).
    """

    return dict(color=COLORS.get(label, "#000000"), linestyle=LINESTYLES.get(label, "-"),
                linewidth=2.0 if label == "FH" else 1.4, label=label, zorder=3 if label == "FH" else 2)


def plot_trajectories_for_speed(results: List[RunResult], base: Optional[RunResult],
                                ray_env: Optional[PySimEnv], png_path: Path, title: str) -> None:
    """
    One panel per run with every model's x/y path (and the river boundary if `ray_env`).
    """

    if plt is None:
        print("plot: matplotlib not importable -- PNG skipped, CSVs written.", flush=True)
        return
    fig, axes = _grid(len(results))
    boundary_xy = np.array(ray_env.polygon_shape.boundary.coords) if ray_env is not None else None
    for ax, res in zip(axes, results):
        all_paths = _all_paths(res, base)
        if boundary_xy is not None:
            ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="#9a9a9a", linewidth=1.0, zorder=1)
        xs_all, ys_all = [], []
        for label in MODEL_ORDER:
            pts = all_paths.get(label)
            if not pts:
                continue
            xs, ys = [p.x for p in pts], [p.y for p in pts]
            xs_all += xs
            ys_all += ys
            ax.plot(xs, ys, **_line_kwargs(label))
        ax.scatter([res.paths["FH"][0].x], [res.paths["FH"][0].y], color="#000000", s=20, zorder=4)
        pad = 20.0
        ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
        ax.set_ylim(min(ys_all) - pad, max(ys_all) + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(_subplot_title(res), fontsize=9)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    for ax in axes[len(results):]:
        ax.axis("off")
    _legend_and_layout(fig, axes[0], title)
    _save(fig, png_path)


def plot_lateral_for_speed(results: List[RunResult], base: Optional[RunResult],
                           png_path: Path, title: str) -> None:
    """
    One panel per run with every model's lateral deviation from the initial track over time.
    """

    if plt is None:
        return
    fig, axes = _grid(len(results))
    for ax, res in zip(axes, results):
        all_paths = _all_paths(res, base)
        for label in MODEL_ORDER:
            pts = all_paths.get(label)
            if not pts:
                continue
            lat = [track_metrics(pts[0], p)["lateral"] for p in pts]
            ax.plot([p.t for p in pts], lat, **_line_kwargs(label))
        ax.axhline(0.0, color="#9a9a9a", linewidth=0.8)
        lo, hi = ax.get_ylim()   # minimum +-1 m span, so pure head/tail wind doesn't autoscale onto noise
        ax.set_ylim(min(lo, -1.0), max(hi, 1.0))
        ax.set_title(_subplot_title(res), fontsize=9)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("lateral deviation [m] (stbd +)")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    for ax in axes[len(results):]:
        ax.axis("off")
    _legend_and_layout(fig, axes[0], title)
    _save(fig, png_path)


# --------------------------------------------------------------------------
# Report (self-contained HTML, built from the output folder)
# --------------------------------------------------------------------------

REPORT_NAME = "wind_comparison_report.html"
ONESTEP_NAME = "onestep_summary.csv"
RUN_META_NAME = "run_meta.json"
WIND_AGNOSTIC = ["UGent", "Fossen", "UA_clean", "UA_confined"]
REPORT_MODELS = ["UGent", "Fossen", "UA_clean", "UA_confined", "UA_full"]

_REPORT_CSS = """
:root { --surface:#fcfcfb; --text:#0b0b0b; --text2:#52514e; --rule:#e2e1dc; --accent:#2a78d6; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--text);
       font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 36px 0 8px; padding-top: 12px; border-top: 1px solid var(--rule); }
h3 { font-size: 16px; margin: 20px 0 6px; }
p, li { color: var(--text); max-width: 80ch; }
.sub { color: var(--text2); margin: 0 0 20px; }
.note { color: var(--text2); font-size: 13px; }
table { border-collapse: collapse; margin: 8px 0 16px; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 8px; text-align: right; }
td { white-space: nowrap; }
th { color: var(--text2); font-weight: 600; background: #f3f2ef; white-space: normal; max-width: 120px;
     vertical-align: bottom; }
td:first-child, th:first-child { text-align: left; }
.wrap { overflow-x: auto; }
figure { margin: 12px 0 24px; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--rule); }
figcaption { color: var(--text2); font-size: 13px; margin-top: 4px; max-width: 90ch; }
.kpis { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0 8px; }
.kpi { border: 1px solid var(--rule); border-radius: 6px; padding: 10px 14px; min-width: 200px; background: #fff; }
.kpi .v { font-size: 22px; font-weight: 650; }
.kpi .l { color: var(--text2); font-size: 13px; }
.warn { color: #8a5a00; }
@media print { h2 { break-before: auto; } figure, table { break-inside: avoid; } }
"""


def _fig_to_data_uri(fig) -> str:
    """
    The figure as a PNG data URI (the figure is closed).
    """

    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _png_to_data_uri(path: Path) -> Optional[str]:
    """
    The PNG file at `path` as a data URI, or None if it does not exist.
    """

    import base64
    if not path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _table_html(df, float_fmt: str = "{:.2f}", int_cols=(), col_fmts=None) -> str:
    """
    DataFrame as an HTML table; int_cols without decimals, col_fmts overrides float_fmt per column.
    """

    col_fmts = col_fmts or {}
    import html
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for col, v in zip(df.columns, r.values):
            if isinstance(v, (float, np.floating, int, np.integer)) and not isinstance(v, bool):
                v = float(v)
                if np.isnan(v):
                    cells.append("")
                elif not np.isfinite(v):
                    cells.append("n/a")
                elif col in int_cols:
                    cells.append(f"{v:.0f}")
                else:
                    cells.append(col_fmts.get(col, float_fmt).format(v))
            else:
                cells.append(html.escape(str(v)))
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _effect_plot(d, value_col: str, fh_col: str, ylabel: str, title: str) -> str:
    """
    One panel per wind speed: wind effect vs relative direction, FH and every model.
    """

    speeds = sorted(d["target_wind_speed"].unique())
    fig, axes = plt.subplots(1, len(speeds), figsize=(4.2 * len(speeds), 3.8), squeeze=False, sharey=True)
    for ax, ws in zip(axes[0], speeds):
        sub = d[d["target_wind_speed"] == ws]
        fh = sub[sub["model"] == "FH"].sort_values("target_rel_dir_deg")
        ax.plot(fh["target_rel_dir_deg"], fh[fh_col], color=COLORS["FH"], lw=2.6, marker="o", ms=5, label="FH (reference)",
                zorder=4)
        for m in REPORT_MODELS:
            mm = sub[sub["model"] == m].sort_values("target_rel_dir_deg")
            if mm.empty:
                continue
            ax.plot(mm["target_rel_dir_deg"], mm[value_col], color=COLORS.get(m, "#000"), lw=1.6,
                    linestyle=LINESTYLES.get(m, "-"), marker="o", ms=3.5, label=m, zorder=3)
        ax.axhline(0, color="#9a9a9a", lw=0.8, zorder=1)
        ax.set_title(f"true wind {ws:g} m/s", fontsize=11)
        ax.set_xticks(sorted(sub["target_rel_dir_deg"].unique()))
        ax.set_xlabel("relative wind direction [deg]\n(0 = ahead, 90 = from starboard)", fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=8)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    axes[0][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _fig_to_data_uri(fig)


def _error_bar_plot(err_table, title: str, ylabel: str) -> str:
    """
    Grouped bars: mean |wind-effect error| per model, per wind speed (one axis).
    """

    speeds = [c for c in err_table.columns if c != "model"]
    models = list(err_table["model"])
    x = np.arange(len(speeds))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, m in enumerate(models):
        vals = err_table[err_table["model"] == m][speeds].to_numpy(dtype=float)[0]
        ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, width=w * 0.9, color=COLORS.get(m, "#000"), label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in speeds])
    ax.set_xlabel("target true wind speed [m/s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", lw=0.3, alpha=0.5)
    ax.legend(ncol=len(models), frameon=False, fontsize=9, loc="upper left")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def _onestep_section(out_dir: Path, d) -> str:
    """
    Report section for the teacher-forced one-step check (empty if not run).
    """

    import pandas as pd
    path = out_dir / ONESTEP_NAME
    if not path.is_file():
        return ""
    o = pd.read_csv(path)
    if o.empty:
        return ""
    u0 = float(WIND_TRAJECTORY_SPEED)
    t_run = o["duration_s"]
    # Errors a constant bias would cause over the run (rough). Biases are port
    # positive, the rollout errors starboard positive, hence the minus signs (a
    # pure bias then lies on Figure 4's diagonal).
    o["implied_turn_deg"] = -np.degrees(0.5 * o["bias_dr"] * t_run ** 2)
    o["implied_lateral_m"] = -(0.5 * o["bias_dv"] * t_run ** 2 + u0 * o["bias_dr"] * t_run ** 3 / 6.0)
    key = ["target_wind_speed", "target_rel_dir_deg", "model"]
    roll = d[key + ["turn_error_vs_fh_deg", "lateral_error_vs_fh"]]
    o["target_rel_dir_deg"] = o["target_rel_dir_deg"].astype(float)
    m = o.merge(roll, on=key, how="left")

    def group(row):
        """
        Group of a one-step row: 'no wind', 'from starboard', 'from port' or 'head / tail'.
        """

        if row["target_wind_speed"] == 0:
            return "no wind"
        rd = row["target_rel_dir_deg"]
        return "from starboard" if 0 < rd < 180 else ("from port" if rd > 180 else "head / tail")
    m["group"] = m.apply(group, axis=1)

    rows = []
    for model in [x for x in MODEL_ORDER if x in set(m["model"])]:
        for g in ("no wind", "from port", "from starboard", "head / tail"):
            sub = m[(m["model"] == model) & (m["group"] == g)]
            if sub.empty:
                continue
            act = float(sub["turn_error_vs_fh_deg"].abs().mean())
            imp = float(sub["implied_turn_deg"].abs().mean())
            rows.append({
                "model": model, "wind": g,
                "RMS dv error / RMS FH dv [%]": 100.0 * float((sub["rms_err_dv"] / sub["fh_rms_dv"]).mean()),
                "RMS dr error / RMS FH dr [%]": 100.0 * float((sub["rms_err_dr"] / sub["fh_rms_dr"]).mean()),
                "mean dr bias [deg/s^2]": float(np.degrees(sub["bias_dr"]).mean()),
                "heading error implied by bias [deg]": imp,
                "actual rollout heading error [deg]": act,
                "lateral error implied by bias [m]": float(sub["implied_lateral_m"].abs().mean()),
                "actual rollout lateral error [m]": float(sub["lateral_error_vs_fh"].abs().mean()),
            })
    tbl = pd.DataFrame(rows)

    fig_uri = ""
    if plt is not None:
        models = [x for x in REPORT_MODELS if x in set(m["model"])]
        fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.6), squeeze=False)
        for ax, model in zip(axes[0], models):
            sub = m[m["model"] == model]
            for g, mk in (("no wind", "s"), ("from port", "o"), ("from starboard", "^"), ("head / tail", "D")):
                gg = sub[sub["group"] == g]
                ax.scatter(gg["turn_error_vs_fh_deg"], gg["implied_turn_deg"], s=28, marker=mk,
                           color=COLORS.get(model, "#000"), edgecolor="white", linewidth=0.6, label=g)
            lim = max(5.0, float(np.nanmax(np.abs(np.r_[sub["turn_error_vs_fh_deg"], sub["implied_turn_deg"]]))) * 1.1)
            ax.plot([-lim, lim], [-lim, lim], color="#9a9a9a", lw=0.8)
            ax.axhline(0, color="#d0d0d0", lw=0.6)
            ax.axvline(0, color="#d0d0d0", lw=0.6)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_title(model, fontsize=11)
            ax.set_xlabel("actual rollout heading error [deg]", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, lw=0.3, alpha=0.5)
        axes[0][0].set_ylabel("heading error implied by\none-step bias [deg]", fontsize=9)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], linestyle="none", marker=mk, markersize=7, color="#6b6b6b", markeredgecolor="white")
             for mk in ("s", "o", "^", "D")]
        l = ["no wind", "from port", "from starboard", "head / tail"]
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=9)
        fig.suptitle("Figure 4 - One-step (teacher-forced) bias vs actual rollout error, per run", fontsize=12)
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        fig_uri = _fig_to_data_uri(fig)

    return f"""
<h2>4b. One-step (teacher-forced) check</h2>
<p>At every FH step each model is given exactly FH's state and FH's own environment (wind sensor, UKC,
current, bank rays) - the content of a training row - and its predicted accelerations are compared with
FH's actual change over that 0.1 s step. Separate model instances are used, so this does not affect the
rollouts. A constant one-step bias b in yaw acceleration alone produces a heading error of about
&frac12;&middot;b&middot;T&sup2; over a run of T seconds (and a lateral error of about
&frac12;&middot;b<sub>v</sub>&middot;T&sup2; + u&middot;b&middot;T&sup3;/6).</p>
<ul>
<li>If the heading error implied by the bias is close to the actual rollout error (points near the diagonal
in Figure 4), the rollout error comes from the one-step prediction itself (data / model).</li>
<li>If the actual rollout error is much larger than the bias implies (points far from the diagonal), small
one-step errors are being amplified along the rollout (compounding / stability).</li>
</ul>
{_table_html(tbl, "{:.1f}", col_fmts={"mean dr bias [deg/s^2]": "{:.4f}"}) if not tbl.empty else "<p>No one-step data.</p>"}
{f'<figure><img src="{fig_uri}" alt="Figure 4"><figcaption>Figure 4 - Each point is one run; diagonal = bias fully explains the rollout heading error. FH accelerations are finite differences over one 0.1 s step, the models predict instantaneous accelerations.</figcaption></figure>' if fig_uri else ""}
<p class="note">The bias-implied numbers are a rough indicator: the bias is measured along FH's path,
while a rollout drifts away from it; bias that changes sign during a run partly cancels.</p>
"""


def write_report(out_dir: Path = None) -> Path:
    """
    Build wind_comparison_report.html from a finished run's wind_summary.csv,
    run_meta.json and PNGs (nothing is re-simulated).
    """

    import html
    import json
    import pandas as pd

    out_dir = Path(out_dir) if out_dir is not None else WIND_OUTPUT_DIR
    if plt is None:
        raise RuntimeError("write_report needs matplotlib.")
    d = pd.read_csv(out_dir / "wind_summary.csv")
    meta = json.loads((out_dir / RUN_META_NAME).read_text()) if (out_dir / RUN_META_NAME).is_file() else {}

    base = d[d["target_wind_speed"] == 0].copy()
    wind = d[d["target_wind_speed"] > 0].copy()
    models_present = [m for m in REPORT_MODELS if m in set(wind["model"])]
    speeds = sorted(wind["target_wind_speed"].unique())

    # --- key tables ----------------------------------------------------------
    def mean_abs(col):
        """
        Mean |col| per model, per target wind speed and over all speeds, as a DataFrame.
        """

        rows = []
        for m in models_present:
            sub = wind[wind["model"] == m]
            r: Dict[str, object] = {"model": m}
            for speed in speeds:
                r[f"{speed:g}"] = float(sub[sub["target_wind_speed"] == speed][col].abs().mean())
            r["all"] = float(sub[col].abs().mean())
            rows.append(r)
        return pd.DataFrame(rows)

    err_lat = mean_abs("wind_effect_lateral_error")
    err_turn = mean_abs("wind_effect_turn_error_deg")
    rms_abs = mean_abs("rms_lateral_error_vs_fh")

    fh = wind[wind["model"] == "FH"]
    fh_eff_max = float(fh["fh_wind_effect_lateral"].abs().max()) if not fh.empty else float("nan")
    fh_turn_max = float(fh["fh_wind_effect_turn_deg"].abs().max()) if not fh.empty else float("nan")
    ua = wind[wind["model"] == "UA_full"]

    def side_of(rel_dir):
        """
        Side group of a relative wind direction [deg].
        """

        if 0 < rel_dir < 180:
            return "from starboard (45-135 deg)"
        return "from port (225-315 deg)" if rel_dir > 180 else "head / tail wind (0, 180 deg)"
    side_rows = []
    for side in ("from port (225-315 deg)", "from starboard (45-135 deg)", "head / tail wind (0, 180 deg)"):
        fs = fh[fh["target_rel_dir_deg"].map(side_of) == side]
        us = ua[ua["target_rel_dir_deg"].map(side_of) == side]
        ag = wind[wind["model"].isin(WIND_AGNOSTIC) & (wind["target_rel_dir_deg"].map(side_of) == side)]
        side_rows.append({
            "relative wind": side,
            "FH mean |lateral effect| [m]": float(fs["fh_wind_effect_lateral"].abs().mean()),
            "UA_full mean |lateral error| [m]": float(us["wind_effect_lateral_error"].abs().mean()),
            "no-wind models mean |lateral error| [m]": float(ag["wind_effect_lateral_error"].abs().mean()),
            "FH mean |heading effect| [deg]": float(fs["fh_wind_effect_turn_deg"].abs().mean()),
            "UA_full mean |heading error| [deg]": float(us["wind_effect_turn_error_deg"].abs().mean()),
            "no-wind models mean |heading error| [deg]": float(ag["wind_effect_turn_error_deg"].abs().mean()),
        })
    side_tbl = pd.DataFrame(side_rows)
    ag_all = wind[wind["model"].isin(WIND_AGNOSTIC)]
    ag_max_eff = float(ag_all["wind_effect_lateral"].abs().max()) if not ag_all.empty else float("nan")
    ag_max_model = (ag_all.loc[ag_all["wind_effect_lateral"].abs().idxmax(), "model"] if not ag_all.empty else "n/a")
    ag_lat = float(ag_all["wind_effect_lateral_error"].abs().mean()) if not ag_all.empty else float("nan")
    ag_turn = float(ag_all["wind_effect_turn_error_deg"].abs().mean()) if not ag_all.empty else float("nan")
    ua_lat = float(ua["wind_effect_lateral_error"].abs().mean()) if not ua.empty else float("nan")
    ua_turn = float(ua["wind_effect_turn_error_deg"].abs().mean()) if not ua.empty else float("nan")

    # --- per-run quality table ------------------------------------------------
    q = d[d["model"] == "FH"][["target_wind_speed", "target_rel_dir_deg", "actual_wind_speed", "actual_rel_dir_deg",
                               "true_wind_speed_min_along_path", "true_wind_speed_max_along_path", "n_steps",
                               "fh_domain_exit_step"]].copy()
    # The no-wind run's direction is NaN (NaN != NaN): use a sentinel key.
    _NO_DIR = -999.0
    keyed = d.assign(_dir=d["target_rel_dir_deg"].fillna(_NO_DIR))
    hits = keyed.groupby(["target_wind_speed", "_dir"])["first_bank_hit_step"].apply(
        lambda s: int(s.notna().sum()))
    misses = keyed.groupby(["target_wind_speed", "_dir"])["ukc_lookup_misses"].apply(
        lambda s: int(np.nansum(s.to_numpy(dtype=float))))
    q_dirs = q["target_rel_dir_deg"].fillna(_NO_DIR)
    q["bank_hits"] = [hits.get((a, b), 0) for a, b in zip(q["target_wind_speed"], q_dirs)]
    q["ukc_misses"] = [misses.get((a, b), 0) for a, b in zip(q["target_wind_speed"], q_dirs)]
    q = q.rename(columns={
        "target_wind_speed": "target wind [m/s]", "target_rel_dir_deg": "target dir [deg]",
        "actual_wind_speed": "true wind at start [m/s]", "actual_rel_dir_deg": "true dir at start [deg]",
        "true_wind_speed_min_along_path": "true wind min on path", "true_wind_speed_max_along_path": "true wind max on path",
        "n_steps": "steps", "fh_domain_exit_step": "FH left training domain at step",
        "bank_hits": "models with bank in ray range", "ukc_misses": "UKC lookup misses"})
    updates = meta.get("windstate_updates", {})

    # --- per-direction detail (FH vs UA_full) -----------------------------------
    det_rows = []
    for ws in speeds:
        for rd in sorted(wind["target_rel_dir_deg"].unique()):
            f = fh[(fh["target_wind_speed"] == ws) & (fh["target_rel_dir_deg"] == rd)]
            u = ua[(ua["target_wind_speed"] == ws) & (ua["target_rel_dir_deg"] == rd)]
            if f.empty:
                continue
            row = {"wind [m/s]": ws, "dir [deg]": rd,
                   "FH lateral effect [m]": float(f["fh_wind_effect_lateral"].iloc[0]),
                   "FH turn effect [deg]": float(f["fh_wind_effect_turn_deg"].iloc[0])}
            if not u.empty:
                row.update({"UA_full lateral effect [m]": float(u["wind_effect_lateral"].iloc[0]),
                            "UA_full turn effect [deg]": float(u["wind_effect_turn_deg"].iloc[0]),
                            "UA_full lateral error [m]": float(u["wind_effect_lateral_error"].iloc[0]),
                            "UA_full turn error [deg]": float(u["wind_effect_turn_error_deg"].iloc[0])})
            det_rows.append(row)
    detail = pd.DataFrame(det_rows)

    # --- no-wind baseline table -------------------------------------------------
    bt = base[base["model"].isin(["FH"] + models_present)][["model", "final_lateral", "final_turn_deg", "final_u",
                                                            "lateral_error_vs_fh", "turn_error_vs_fh_deg",
                                                            "rms_lateral_error_vs_fh"]].copy()
    bt.columns = ["model", "final lateral [m]", "final turn [deg]", "final surge [m/s]", "lateral error vs FH [m]",
                  "turn error vs FH [deg]", "RMS lateral error vs FH [m]"]

    # --- charts -------------------------------------------------------------------
    fig_lat = _effect_plot(wind, "wind_effect_lateral", "fh_wind_effect_lateral", "wind effect on lateral position [m]\n(stbd +)",
                           "Figure 1 - Wind effect on lateral drift: each model (in wind minus its own no-wind run) vs FH")
    fig_turn = _effect_plot(wind, "wind_effect_turn_deg", "fh_wind_effect_turn_deg", "wind effect on heading [deg]\n(stbd +)",
                            "Figure 2 - Wind effect on heading change: each model vs FH")
    fig_err = _error_bar_plot(err_lat.drop(columns=["all"]), "Figure 3 - Mean absolute error of the lateral wind effect",
                              "mean |error| [m]")

    def kpi(v, caption, fmt="{:.1f}"):
        """
        HTML KPI tile for value `v`, formatted with `fmt` ("n/a" for None/NaN).
        """

        return f'<div class="kpi"><div class="v">{"n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)}</div><div class="l">{html.escape(caption)}</div></div>'

    setup_rows = [
        ("Ship model", meta.get("ship_model", "")),
        ("FH exercise", meta.get("exercise", "")),
        ("Start pose (x, y, heading)", meta.get("start_pose", "")),
        ("Initial surge / thrust / rudder", meta.get("speed_thrust_rudder", "")),
        ("Path length per run", meta.get("path_length", "")),
        ("Wind grid", meta.get("grid", "")),
        ("Wind calibration", meta.get("calibration", "")),
        ("FH settings", meta.get("fh_settings", "")),
        ("UKC for UA models", meta.get("ukc_source", "")),
        ("Run started / finished", f'{meta.get("started", "")} / {meta.get("finished", "")}'),
    ]
    setup_html = "<table><tbody>" + "".join(
        f"<tr><th>{html.escape(k)}</th><td style='text-align:left;white-space:normal'>{html.escape(str(v))}</td></tr>"
        for k, v in setup_rows) + "</tbody></table>"

    warnings_list = meta.get("warnings", [])
    warn_html = ("<ul>" + "".join(f'<li class="warn">{html.escape(w)}</li>' for w in warnings_list) + "</ul>"
                 if warnings_list else "<p>No warnings were logged during the run.</p>")

    appendix = []
    for ws in [0.0] + list(speeds):
        for kind, label in (("trajectories", "Trajectories"), ("lateral", "Lateral deviation over time")):
            uri = _png_to_data_uri(out_dir / f"wind_{kind}_{ws:g}ms.png")
            if uri:
                appendix.append(f'<figure><img src="{uri}" alt="{label}, {ws:g} m/s"><figcaption>{label} - '
                                f'{"no wind (baseline)" if ws == 0 else f"target true wind {ws:g} m/s"}. FH = ground truth'
                                f'{"" if ws == 0 else "; FH_nowind / UA_full_nowind = the no-wind baseline paths"}.'
                                f'</figcaption></figure>')

    fhw = wind[wind["model"] == "FH"]
    var_lo = float((fhw["true_wind_speed_min_along_path"] / fhw["target_wind_speed"]).min() * 100.0) if not fhw.empty else float("nan")
    var_hi = float((fhw["true_wind_speed_max_along_path"] / fhw["target_wind_speed"]).max() * 100.0) if not fhw.empty else float("nan")
    med_steps = wind["effect_n_steps"].median() if "effect_n_steps" in wind else float("nan")
    run_seconds = 0.0 if med_steps != med_steps else float(med_steps) * SURROGATE_DT
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wind Model Comparison</title><style>{_REPORT_CSS}</style></head>
<body><main>
<h1>Wind response of ship dynamics models compared with FH</h1>
<p class="sub">{html.escape(str(meta.get("ship_model", "")))} in open water, rudder 0, fixed thrust &middot; generated {html.escape(str(meta.get("finished", "")))} by model_comparison_wind.py</p>

<h2>1. Summary</h2>
<div class="kpis">
{kpi(fh_eff_max, "largest FH wind effect on lateral position [m]")}
{kpi(fh_turn_max, "largest FH wind effect on heading [deg]")}
{kpi(ua_lat, "UA_full mean |lateral wind-effect error| [m]")}
{kpi(ag_lat, "no-wind models mean |lateral wind-effect error| [m]")}
{kpi(ua_turn, "UA_full mean |heading wind-effect error| [deg]")}
{kpi(ag_turn, "no-wind models mean |heading wind-effect error| [deg]")}
</div>
<ul>
<li>FH (ground truth): the wind changes its position by up to {fh_eff_max:.0f} m sideways and its heading by up to {fh_turn_max:.0f}&deg; within one run (about {run_seconds:.0f} s), compared with the same run without wind.</li>
<li>UGent, Fossen, UA_clean and UA_confined have no wind inputs. Their wind effect stays small (largest: {ag_max_eff:.0f} m, {html.escape(str(ag_max_model))}, caused by the slightly different start state after the warm-up in wind), so their error is close to FH's full wind effect. They are the "no wind response" reference: {ag_lat:.1f} m lateral and {ag_turn:.1f}&deg; heading mean absolute error.</li>
<li>UA_full is the only wind-aware model: {ua_lat:.1f} m lateral and {ua_turn:.1f}&deg; heading mean absolute error of the wind effect. Its errors depend strongly on the side the wind comes from (Table 1).</li>
</ul>

<h3>Table 1 - Wind-effect errors by side of the wind (all wind speeds)</h3>
{_table_html(side_tbl, "{:.1f}")}
<h3>Table 2a - Mean absolute error of the lateral wind effect [m], per target true wind speed [m/s]</h3>
{_table_html(err_lat, "{:.1f}")}
<h3>Table 2b - Mean absolute error of the heading wind effect [deg], per target true wind speed [m/s]</h3>
{_table_html(err_turn, "{:.1f}")}
<p class="note">Wind effect = (model in wind) - (same model in the no-wind run), at the same time step; error = model's wind effect - FH's wind effect. This separates a model's response to wind from its no-wind (hydrodynamic) error, which is reported separately in section 5.</p>

<h2>2. Test setup</h2>
{setup_html}

<h2>3. Method</h2>
<ul>
<li><b>Ground truth</b>: the FH simulator. Every run is a fresh FH session; the wind-field transform (rotation, scale) is set before FH's first time step, then FH sails 1 s with rudder 0 so the wind is fully established. All models start from FH's state at that moment.</li>
<li><b>Models</b>: UGent, Fossen, UA_clean, UA_confined (no wind inputs) and UA_full (wind inputs), each integrated with the framework's ShipSimulator with its own environment model switched off, so no wind is added on top of what a model computes itself.</li>
<li><b>No model is steered by FH</b>: each UA model gets the apparent wind for its own heading, surge and sway (from FH's true wind), and UKC from its own WcuLookup data at its own position. FH's wind sensor updates at 1 Hz; the models' wind inputs are updated at the same instants and held in between, matching the training data.</li>
<li><b>Wind grid</b>: true wind = FH's apparent wind minus the ship's own motion; direction relative to the bow, 0&deg; = from ahead, 90&deg; = from starboard. A calibration at the start pose maps the target wind to FH's wind-field transform; the wind actually delivered is logged per run (section 6).</li>
<li><b>No-wind baseline</b>: one run with the wind field switched off (scale 0) for all models. Comparing each model with its own baseline isolates the wind effect.</li>
<li><b>Open water</b>: the start pose has no bank within ray range; every model's path is checked every second (section 6).</li>
</ul>

<h2>4. Results</h2>
<figure><img src="{fig_lat}" alt="Figure 1"><figcaption>Figure 1 - Lateral wind effect after the common run time, per relative wind direction. FH is the reference (thick blue line); lines close to zero are models that do not respond to wind.</figcaption></figure>
<figure><img src="{fig_turn}" alt="Figure 2"><figcaption>Figure 2 - Heading wind effect (turn to starboard positive), same layout.</figcaption></figure>
<figure><img src="{fig_err}" alt="Figure 3"><figcaption>Figure 3 - Mean absolute error of the lateral wind effect over all directions, per wind speed (values in Table 2a).</figcaption></figure>
<h3>Table 3 - Per-direction detail: FH and UA_full</h3>
{_table_html(detail, "{:.1f}", int_cols=("wind [m/s]", "dir [deg]"))}
<h3>Table 4 - Mean RMS lateral deviation from FH over the whole path [m] (absolute, includes no-wind error)</h3>
{_table_html(rms_abs, "{:.1f}")}

{_onestep_section(out_dir, d)}

<h2>5. No-wind baseline</h2>
<p>Without wind, at rudder 0, FH itself does not sail perfectly straight. Each model's deviation from FH here is its no-wind (hydrodynamic) error, independent of wind.</p>
{_table_html(bt, "{:.2f}")}

<h2>6. Data quality per run</h2>
{_table_html(q, "{:.2f}", int_cols=("target wind [m/s]", "target dir [deg]", "steps", "FH left training domain at step",
                                     "models with bank in ray range", "UKC lookup misses"))}
<p class="note">FH WindState updates per run: {html.escape(", ".join(f"{k}: {v}" for k, v in updates.items())) if updates else "n/a"}.</p>
<h3>Warnings logged during the run</h3>
{warn_html}

<h2>7. Limitations</h2>
<ul>
<li>The exercise's wind field is not uniform: along FH's path the true wind ranges from {var_lo:.0f} % to {var_hi:.0f} % of the target (largest spread over all runs; section 6). The wind speed labels are the target at the start pose.</li>
<li>FH's wind sensor updates at 1 Hz; the true-wind reconstruction uses the ship state at the update step (error at most one 0.1 s step).</li>
<li>FH's WindState convention (positive angle = wind from starboard) was determined from the dataset generator's turning episodes (earth-fixed true wind constant within an episode); the sway term of the apparent-wind conversion follows from it.</li>
<li>UKC for UA_confined and UA_full comes from each model's own training CSV (nearest sample), not from FH; the two CSVs can give different values at the same position.</li>
<li>UA_full was trained on data from a generator version that recorded the first ~1 s of every episode with the exercise's default wind (since fixed); with the current dataset's mean episode length (~70 s) that is about 1-2 % of its training rows.</li>
<li>One start pose, one ship speed (rudder 0, fixed thrust): results describe straight-line sailing in open water only.</li>
</ul>

<h2>Appendix - trajectory and lateral-deviation plots</h2>
{"".join(appendix) if appendix else "<p>No plots found.</p>"}
</main></body></html>
"""
    out = out_dir / REPORT_NAME
    out.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def diagnose_wind_transform(start_x: float, start_y: float, start_heading_deg: float,
                            exercise_path: Path, speed: float = WIND_TRAJECTORY_SPEED,
                            rewind_enabled: Optional[bool] = FH_REWIND_ENABLED) -> None:
    """
    Diagnostic: per (Rotation, Scale), print the transform FH reports back and
    the apparent wind right after reset() and after WIND_WARMUP_TIME.
    """

    ship_spec = create_ship(SHIP_MODEL).specifications
    thrust = float(min(1.0, speed / 6.0))
    print(f"diagnose_wind_transform: {exercise_path} (rewind_enabled={rewind_enabled})", flush=True)
    for rotation, scale in [(0.0, 1.0), (90.0, 1.0), (0.0, 2.0), (0.0, 0.0)]:
        fh = FhSession(x=float(start_x), y=float(start_y), heading_deg=float(start_heading_deg),
                       v_init=float(speed), thrust=thrust, ship_spec=ship_spec,
                       exercise_path=Path(exercise_path), wind_field=(rotation, scale),
                       rewind_enabled=rewind_enabled)
        try:
            print(f"  FH loaded exercise: {fh._bridge.exercise_path}", flush=True)
            control = fh._bridge.math_model.getEnvironmentInterface().getWindFieldControl()
            if control is None:
                readback = "no wind-field control"
            else:
                tr = control.getFieldTransform()
                readback = f"Rotation={tr.Rotation}, Scale={tr.Scale}, Offset={tr.Offset}"
            ws0, wd0 = fh.get_wind_raw()
            fh.advance(WIND_WARMUP_TIME)
            ws, wd = fh.get_wind_raw()
            print(f"  requested R={rotation:g}, S={scale:g} -> read-back [{readback}] -> apparent wind "
                  f"t0 {ws0:.3f}@{wd0:.2f}, after {WIND_WARMUP_TIME:g} s {ws:.3f}@{wd:.2f}", flush=True)
        finally:
            fh.close()


def _report(res: RunResult) -> None:
    """
    Print a one-run summary: wind, FH's final offset and turn, and any warnings.
    """

    fh_final = track_metrics(res.paths["FH"][0], res.paths["FH"][-1])
    print(f"{res.run_id}: true wind {res.actual_wind_speed:.2f} m/s @ rel {res.actual_rel_dir_deg:.1f} deg "
          f"(apparent {res.apparent_wind_speed:.2f} @ {res.apparent_rel_dir_deg:.1f}; true along path "
          f"{res.wind_speed_min:.2f}-{res.wind_speed_max:.2f}), {len(res.paths['FH']) - 1} steps; "
          f"FH final lateral {fh_final['lateral']:+.2f} m (stbd +), turn {fh_final['turn']:+.1f} deg (stbd +)",
          flush=True)
    if res.hit_max_steps:
        print(f"{res.run_id}: hit MAX_STEPS before {TARGET_PATH_LENGTH} m.", flush=True)
    if res.domain_exit:
        print(f"{res.run_id}: FH left the training domain at step {res.domain_exit[0]} "
              f"('{res.domain_exit[1]}') -- UA_* after that is extrapolation.", flush=True)
    hits = {k: v for k, v in res.first_bank_hit.items() if v is not None}
    if hits:
        print(f"{res.run_id}: banks within ray range (first step per model): {hits} -- "
              f"no longer pure open water after that.", flush=True)
    misses = {k: v for k, v in res.ukc_misses.items() if v}
    if misses:
        print(f"{res.run_id}: own-position UKC lookup found nothing (FH's UKC used instead) for "
              f"{misses} steps.", flush=True)


def generate_wind_trajectories(
        start_x: float,
        start_y: float,
        start_heading_deg: float,
        exercise_path: Optional[Path] = None,
        wind_speeds: Sequence[float] = WIND_SPEEDS,
        rel_dirs_deg: Sequence[float] = REL_WIND_DIRS_DEG,
        speed: float = WIND_TRAJECTORY_SPEED,
) -> None:
    """
    Calibrate the wind field, run the no-wind baseline and every (speed, direction)
    case, and write CSVs, plots, run_meta.json and the report to WIND_OUTPUT_DIR.

    Args:
        start_x: Start x [m].
        start_y: Start y [m].
        start_heading_deg: Start heading [deg].
        exercise_path: Wind-only .tab file; None = WIND_EXERCISE_PATH.
        wind_speeds: Target true wind speeds [m/s].
        rel_dirs_deg: Target wind directions relative to the bow [deg].
        speed: Initial ship speed [m/s]; the thrust is min(1, speed / 6).

    Raises:
        ValueError: If no exercise path is given or configured.
    """

    exercise_path = exercise_path if exercise_path is not None else WIND_EXERCISE_PATH
    if exercise_path is None:
        raise ValueError("generate_wind_trajectories: pass exercise_path (or set WIND_EXERCISE_PATH) "
                         "to the wind-only Tripoli .tab file.")
    exercise_path = Path(exercise_path)
    WIND_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if CLEAN_OUTPUT_DIR:
        _clean_output_dir(WIND_OUTPUT_DIR)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    warn_handler = _ListHandler()
    logger.addHandler(warn_handler)
    try:
        _generate(start_x, start_y, start_heading_deg, exercise_path, wind_speeds, rel_dirs_deg, speed,
                  started, warn_handler)
    finally:
        logger.removeHandler(warn_handler)


class _ListHandler(logging.Handler):
    """
    Collects this module's warnings for the report.
    """

    def __init__(self):
        """
        Collect messages of level WARNING and above.
        """

        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record):
        """
        Store the message and echo it to stderr.
        """

        msg = record.getMessage()
        self.messages.append(msg)
        # This handler disables the last-resort console output, so echo it.
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def _clean_output_dir(out_dir: Path) -> None:
    """
    Remove this script's outputs of a previous run, so the report never mixes runs.
    """

    patterns = ("trajectory_*.csv", "env_*.csv", "wind_trajectories_*ms.png", "wind_lateral_*ms.png",
                "wind_summary.csv", ONESTEP_NAME, RUN_META_NAME, REPORT_NAME)
    n = 0
    for pat in patterns:
        for f in out_dir.glob(pat):
            f.unlink()
            n += 1
    if n:
        print(f"Removed {n} output file(s) of a previous run from {out_dir}", flush=True)


def _generate(start_x, start_y, start_heading_deg, exercise_path, wind_speeds, rel_dirs_deg, speed,
              started, warn_handler) -> None:
    """
    Body of generate_wind_trajectories; `warn_handler` collects warnings for run_meta.json.
    """

    import json

    ship_spec = create_ship(SHIP_MODEL).specifications
    thrust = float(min(1.0, speed / 6.0))

    ray_env = PySimEnv(render_mode=None, time_step=SURROGATE_DT, wind=False, current=False,
                       start_pos=np.array([start_x, start_y], dtype=np.float32), ship_model=SHIP_MODEL)
    cutoff_geom = _cutoff_geometry(ray_env)

    # Built once and reused: their only per-episode state (UA smoothing, off here) is
    # cleared by ShipSimulator.reset.
    ua_models = build_ua_models(ship_spec)          # the UA_* slots SHIP_MODEL has a preset for
    non_fh_models = {**build_reference_models(ship_spec, ray_env), **ua_models}
    ukc_source = UkcSource(ua_models)
    ukc_source.validate(start_x, start_y)

    # Separate instances for the one-step check, so it cannot affect the rollouts.
    tf_models = None
    if ONESTEP_CHECK:
        tf_models = {**build_reference_models(ship_spec, ray_env), **build_ua_models(ship_spec)}

    common = dict(start_x=float(start_x), start_y=float(start_y), start_heading_deg=float(start_heading_deg),
                  v_init=float(speed), thrust=thrust, ship_spec=ship_spec, exercise_path=exercise_path)
    run_kwargs = dict(ray_env=ray_env, cutoff_geom=cutoff_geom, non_fh_models=non_fh_models,
                      ua_models=ua_models, ukc_source=ukc_source, tf_models=tf_models, **common)

    cal = calibrate_wind_field(**common)
    updates: Dict[str, int] = {}

    # No-wind baseline (Scale=0) for all models, for the paired wind-effect metrics.
    base = run_case("nowind", 0.0, 0.0, 0.0, float("nan"), is_baseline=True, **run_kwargs)
    _report(base)
    updates[base.run_id] = base.windstate_updates

    summary_path = WIND_OUTPUT_DIR / "wind_summary.csv"
    onestep_file = open(WIND_OUTPUT_DIR / ONESTEP_NAME, "w", newline="")
    onestep_writer = csv.writer(onestep_file)
    onestep_writer.writerow(ONESTEP_HEADER)
    onestep_writer.writerows(onestep_rows(base))
    with open(summary_path, "w", newline="") as summary_file, onestep_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(SUMMARY_HEADER)

        write_trajectory_csv(WIND_OUTPUT_DIR / f"trajectory_nowind_v{speed:g}.csv", base, None)
        write_env_csv(WIND_OUTPUT_DIR / f"env_nowind_v{speed:g}.csv", base)
        for row in summary_rows(base, None):
            summary_writer.writerow(row)
        summary_file.flush()
        plot_trajectories_for_speed([base], None, ray_env, WIND_OUTPUT_DIR / "wind_trajectories_0ms.png",
                                    f"Trajectories vs. FH -- no wind, v0={speed:g} m/s, rudder 0")
        plot_lateral_for_speed([base], None, WIND_OUTPUT_DIR / "wind_lateral_0ms.png",
                               "Lateral deviation from initial track -- no wind")

        for ws in wind_speeds:
            results = []
            for rel in rel_dirs_deg:
                run_id = f"wind{ws:g}_rel{rel:g}"
                rotation, scale = cal.transform_for(float(ws), float(rel))
                try:
                    res = run_case(run_id, rotation, scale, float(ws), float(rel), **run_kwargs)
                except WindTestSetupError:
                    raise
                except RuntimeError as exc:
                    # Same per-point resilience as generate_trajectories (e.g. FH init failure).
                    print(f"SKIPPED {run_id}: {exc}", flush=True)
                    continue
                _report(res)
                updates[res.run_id] = res.windstate_updates
                write_trajectory_csv(WIND_OUTPUT_DIR / f"trajectory_{run_id}_v{speed:g}.csv", res, base)
                write_env_csv(WIND_OUTPUT_DIR / f"env_{run_id}_v{speed:g}.csv", res)
                for row in summary_rows(res, base):
                    summary_writer.writerow(row)
                onestep_writer.writerows(onestep_rows(res))
                summary_file.flush()
                onestep_file.flush()
                results.append(res)

            if results:
                plot_trajectories_for_speed(
                    results, base, ray_env, WIND_OUTPUT_DIR / f"wind_trajectories_{ws:g}ms.png",
                    f"Trajectories vs. FH -- open water, target wind {ws:g} m/s, v0={speed:g} m/s, rudder 0")
                plot_lateral_for_speed(
                    results, base, WIND_OUTPUT_DIR / f"wind_lateral_{ws:g}ms.png",
                    f"Lateral deviation from initial track -- target wind {ws:g} m/s")
    print(f"Wrote {summary_path}", flush=True)

    meta = {
        "ship_model": getattr(SHIP_MODEL, "name", str(SHIP_MODEL)),
        "exercise": str(exercise_path),
        "start_pose": f"({start_x:.1f}, {start_y:.1f}) m, heading {start_heading_deg:.1f} deg",
        "speed_thrust_rudder": f"{speed:g} m/s / {min(1.0, speed / 6.0):.3f} (normalized) / 0 deg",
        "path_length": f"until FH has sailed {TARGET_PATH_LENGTH:g} m (max {MAX_STEPS} steps of {SURROGATE_DT:g} s); "
                       f"all models run the same number of steps",
        "grid": f"true wind {', '.join(f'{w:g}' for w in wind_speeds)} m/s x relative direction "
                f"{', '.join(f'{r:g}' for r in rel_dirs_deg)} deg, plus a no-wind run",
        "calibration": cal.summary,
        "fh_settings": f"rewind {'on' if FH_REWIND_ENABLED else 'off'}, {WIND_WARMUP_TIME:g} s warm-up after reset, "
                       f"sway sign {TRUE_WIND_SWAY_SIGN}",
        "ukc_source": (f"explicit CSV {UKC_LOOKUP_CSV} (max {UKC_LOOKUP_MAX_DISTANCE} m)" if UKC_LOOKUP_CSV
                       else "each UA model's own lookup CSV, at the model's own position"),
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "windstate_updates": updates,
        "warnings": list(dict.fromkeys(warn_handler.messages)),
    }
    (WIND_OUTPUT_DIR / RUN_META_NAME).write_text(json.dumps(meta, indent=2))
    try:
        write_report(WIND_OUTPUT_DIR)
    except Exception as exc:  # the CSVs/PNGs are complete; the report can be rebuilt with --report-only
        print(f"Report generation failed ({exc!r}); rebuild with: python model_comparison_wind.py --report-only",
              flush=True)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_report(WIND_OUTPUT_DIR)          # rebuild the report from an existing output folder
    else:
        # Wind-only (no current) exercise of SHIP_MODEL, e.g.
        # DDShip_Tripoli_scenario_windnocurrent.tab (see sim_fh_dataset_generator.SHIP_PROFILES).
        from fw.surrogate.sim_fh_dataset_generator import exercise_paths
        _, _wind_only_exercise = exercise_paths(SHIP_MODEL, "full")
        generate_wind_trajectories(
            start_x=-1559.0,
            start_y=11990.0,
            start_heading_deg=-45.0,   # same convention as model_comparison.py's heading_deg
            exercise_path=_wind_only_exercise,
            wind_speeds=WIND_SPEEDS,
            rel_dirs_deg=REL_WIND_DIRS_DEG,
        )
