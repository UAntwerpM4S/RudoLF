"""
Current counterpart of model_comparison_wind.py: compares all dynamics models
against FH (ground truth) for a ship sailing straight (rudder 0, fixed thrust)
in open water, over a grid of current level x direction relative to the ship.
Wind is off.

Models:
    FH: simexe, current exercise, wind-field Scale 0
    FH_nocurrent: same run (timeStart, heading) on the no-current exercise,
        the paired baseline of every run
    UGent, Fossen, UA_clean, UA_confined: no current input
    UA_full: the only non-FH model with current inputs
    UA_full_nocurrent: UA_full in the baseline run

Setting the current: FH has no setter; the <timeStart> of the area's current
and tide XMLs shifts the tidal phase. It is written before each FH session and
the originals are restored at the end (also on errors or Ctrl+C; after a hard
kill, from CURRENT_OUTPUT_DIR/timestart_backup/ at the next start). The
baseline uses the same timeStart, so the same water level.

Calibration: the current at the start pose is scanned over
CURRENT_TIME_START_RANGE; one timeStart per level is picked
(CURRENT_LEVEL_FRACTIONS of the strongest, or CURRENT_SPEED_TARGETS). The
field cannot be rotated, so the start heading sets the relative direction:
heading = (current toward) + 180 deg + relative direction.

Relative direction: where the current comes from, relative to the bow,
clockwise (0 = head current, 90 = from starboard, 180 = following).

UA_* inputs every step: FH's live UKC, FH's current at FH's position, and the
apparent wind for the model's own state, held between FH's 1 Hz WindState
updates. Non-FH models start from FH's state after warm-up; those velocities
over ground already contain the current, so models without current input
still show a small "current effect". Runs near a bank are flagged, not skipped.

Usage (from src):
    python fw/surrogate/validation/model_comparison_current.py               # full run
    python fw/surrogate/validation/model_comparison_current.py --calibrate-only
    python fw/surrogate/validation/model_comparison_current.py --report-only # rebuild the report
"""

import sys
import csv
import json
import time
import logging
import numpy as np

from pathlib import Path
from dataclasses import dataclass, field
from fw.ships.ship_factory import create_ship
from fw.ships.vessel_state import VesselState
from fw.environments.py_sim_env import PySimEnv
from fw.ships.actuator_command import ActuatorCommand
from typing import Dict, List, Optional, Sequence, Tuple
from fw.surrogate.sim_fh_dataset_generator import (
    CURRENT_TIME_START_RANGE,
    TIME_START_XML_PATHS,
    _TIME_START_PATTERN,
    _cutoff_geometry,
    randomize_current_time_start,
)
from fw.surrogate.validation.model_comparison import (
    FhSession,
    OUTPUT_DIR,
    RUDDER_ANGLE_RAD,
    SHIP_MODEL,
    SURROGATE_DT,
    _inject_live_environment,
    _training_domain_issue,
    SurgeClipCounter,
    build_reference_models,
    build_ua_models,
    make_model_simulator,
    plt,
)
# Shared helpers of the wind test (geometry, wind reconstruction, one-step
# statistics, report building blocks).
from fw.surrogate.validation.model_comparison_wind import (
    COLORS as _WIND_COLORS,
    FH_REWIND_ENABLED,
    LINESTYLES as _WIND_LINESTYLES,
    OPEN_WATER_CHECK_EVERY,
    REQUIRE_OPEN_WATER_START,
    TrackPoint,
    WIND_WARMUP_TIME,
    _ListHandler,
    _REPORT_CSS,
    _apparent_wind_for,
    _fh_point,
    _fig_to_data_uri,
    _onestep_stats,
    _rays_hit,
    _save,
    _table_html,
    _true_wind,
    _true_wind_world,
    _wrap180,
    _wrap360,
    track_metrics,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Test parameters
# --------------------------------------------------------------------------

# Exercises (wind-field Scale 0 in both). Can also be passed to generate_current_trajectories().
CURRENT_EXERCISE_PATH: Optional[Path] = None      # e.g. .../DDShip_Tripoli_scenario_windcurrent.tab
NOCURRENT_EXERCISE_PATH: Optional[Path] = None    # e.g. .../DDShip_Tripoli_scenario_windnocurrent.tab

REL_CURRENT_DIRS_DEG = (0, 45, 90, 135, 180, 225, 270, 315)   # current FROM, rel. to bow, clockwise
CURRENT_LEVEL_FRACTIONS = (0.5, 1.0)        # of the strongest current found in the calibration scan
CURRENT_SPEED_TARGETS: Optional[List[float]] = None   # [m/s]; if set, used instead of the fractions

CURRENT_TRAJECTORY_SPEED = 5                # initial surge speed [m/s]; thrust = min(1, v/6)
TARGET_PATH_LENGTH = 300.0                  # [m] stop once FH's own path over ground reaches this
MAX_STEPS = 8000                            # safety cap (800 s)

CAL_TIME_STEP_S = 1800                      # calibration scan step over CURRENT_TIME_START_RANGE [s]
CAL_MIN_CURRENT = 0.1                       # [m/s] strongest current found below this -> error
CAL_MIN_SPREAD = 0.1                        # [m/s] (max - min) current over the scan below this -> error
RUN_DIR_TOL_DEG = 10.0                      # actual vs target relative direction (warning)
RUN_SPEED_TOL_REL = 0.15                    # actual vs calibrated current speed (warning)
MAX_BASELINE_CURRENT = 0.05                 # [m/s] baseline (no-current exercise) above this -> error
MAX_RESIDUAL_WIND = 0.5                     # [m/s] true wind with Scale 0 above this -> warning (logged)

CURRENT_OUTPUT_DIR = OUTPUT_DIR / "current_trajectories"
CLEAN_OUTPUT_DIR = True
ONESTEP_CHECK = True
# FH RMS below which a one-step error ratio is not shown (signal ~0, e.g. dv/dr in head/following current)
ONESTEP_MIN_FH_RMS = {"du": 1e-4, "dv": 1e-4, "dr": 1e-6}

REPORT_NAME = "current_comparison_report.html"
SUMMARY_NAME = "current_summary.csv"
ONESTEP_NAME = "onestep_summary.csv"
CALIBRATION_NAME = "current_calibration.csv"
RUN_META_NAME = "run_meta.json"

NO_CURRENT_MODELS = ["UGent", "Fossen", "UA_clean", "UA_confined"]
REPORT_MODELS = ["UGent", "Fossen", "UA_clean", "UA_confined", "UA_full"]
MODEL_ORDER = ["FH", "FH_nocurrent", "UGent", "Fossen", "UA_clean", "UA_confined", "UA_full", "UA_full_nocurrent"]
COLORS = {**_WIND_COLORS, "FH_nocurrent": _WIND_COLORS.get("FH_nowind", "#6b6b6b"),
          "UA_full_nocurrent": _WIND_COLORS.get("UA_full_nowind", "#7fbf7f")}
LINESTYLES = {**_WIND_LINESTYLES, "FH_nocurrent": (0, (8, 3)), "UA_full_nocurrent": (0, (8, 3))}


class CurrentTestSetupError(RuntimeError):
    """Fail-fast setup problem that applies to every run (re-raised, not skipped)."""


# --------------------------------------------------------------------------
# timeStart in the current/tide XMLs
# --------------------------------------------------------------------------

TIMESTART_BACKUP_DIR_NAME = "timestart_backup"   # in CURRENT_OUTPUT_DIR; exists only while a run is active


def _backup_dir() -> Path:
    """
    Folder holding the <timeStart> XML backups while a run is active.
    """

    return CURRENT_OUTPUT_DIR / TIMESTART_BACKUP_DIR_NAME


def snapshot_xmls() -> Dict[Path, bytes]:
    """
    Original raw bytes of every TIME_START_XML_PATHS file, also backed up (with
    an index) to CURRENT_OUTPUT_DIR/timestart_backup/ for recovery after a hard kill.
    """

    original = {}
    for p in TIME_START_XML_PATHS:
        p = Path(p)
        raw = p.read_bytes()
        if _TIME_START_PATTERN.search(raw) is None:
            raise CurrentTestSetupError(f'"<timeStart>...</timeStart>" not found in {p}')
        original[p] = raw
    bdir = _backup_dir()
    bdir.mkdir(parents=True, exist_ok=True)
    index = {}
    for i, (p, raw) in enumerate(original.items()):
        (bdir / f"{i}_{p.name}").write_bytes(raw)
        index[f"{i}_{p.name}"] = str(p)
    (bdir / "index.json").write_text(json.dumps(index, indent=2))
    return original


def _time_start_text(raw: bytes) -> str:
    """
    The <timeStart> value in the raw XML bytes, or "?" if not found.
    """

    m = _TIME_START_PATTERN.search(raw)
    return m.group(2).decode("ascii", errors="replace").strip() if m else "?"


def _write_bytes_atomic(path: Path, raw: bytes) -> None:
    """
    Write via temp file + os.replace, retrying on transient Windows locks (as the generator).
    """

    import os
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        tmp.write_bytes(raw)
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.5)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def restore_xmls(original: Dict[Path, bytes]) -> List[str]:
    """
    Write the original bytes back, each file independently. Returns error
    messages (empty = all restored); the backup folder is removed only then.
    """

    errors = []
    for path, raw in original.items():
        try:
            _write_bytes_atomic(path, raw)
        except Exception as exc:
            errors.append(f"{path}: {exc!r}")
    if errors:
        print("!! Could not restore <timeStart> in: " + "; ".join(errors)
              + f" -- the original files are in {_backup_dir()}", file=sys.stderr, flush=True)
    else:
        print("Restored the original current/tide XMLs (<timeStart> "
              + ", ".join(f"{p.name}={_time_start_text(r)}" for p, r in original.items()) + ")", flush=True)
        import shutil
        shutil.rmtree(_backup_dir(), ignore_errors=True)
    return errors


def recover_stale_backup() -> None:
    """
    If a previous run was killed before it could restore the XMLs, restore them now.
    """

    bdir = _backup_dir()
    index_path = bdir / "index.json"
    if not index_path.is_file():
        return
    index = json.loads(index_path.read_text())
    original = {Path(target): (bdir / name).read_bytes() for name, target in index.items()}
    print(f"Found {bdir} from a previous run that did not finish cleanly -- restoring the XMLs from it.",
          flush=True)
    if restore_xmls(original):
        raise CurrentTestSetupError(f"Could not restore the XMLs from {bdir}; restore them by hand first.")


def set_time_start(value: int) -> None:
    """
    Write value into all TIME_START_XML_PATHS (generator's writer). Only call while
    no FH session runs: simexe reads these files itself.
    """

    randomize_current_time_start(TIME_START_XML_PATHS, time_range=(int(value), int(value)))


# --------------------------------------------------------------------------
# FH sessions and current geometry
# --------------------------------------------------------------------------

def _session(x, y, heading_deg, v_init, thrust, ship_spec, exercise_path, where: str) -> FhSession:
    """
    FH session with wind Scale 0, advanced WIND_WARMUP_TIME (rudder 0, fixed thrust).
    """

    try:
        fh = FhSession(x=x, y=y, heading_deg=heading_deg, v_init=v_init, thrust=thrust, ship_spec=ship_spec,
                       exercise_path=exercise_path, wind_field=(0.0, 0.0), rewind_enabled=FH_REWIND_ENABLED)
    except Exception as exc:
        raise RuntimeError(f"{where}: FH session could not be started ({exc})") from exc
    try:
        fh.advance(WIND_WARMUP_TIME)
    except Exception:
        fh.close()
        raise
    return fh


def _loaded_exercise(fh: FhSession) -> str:
    """
    The exercise FH actually loaded (FhSimBridge.exercise_path), not the scenario
    name the bridge derives from the ship.
    """

    bridge = getattr(fh, "_bridge", None)
    return str(getattr(bridge, "exercise_path", "?"))


def current_toward_deg(fh: FhSession) -> Tuple[float, float]:
    """
    (speed [m/s], direction the current flows TOWARD [deg, env frame: 0 = East, CCW]).
    """

    cs, cd = fh.get_current()
    return float(cs), _wrap360(np.degrees(cd))


def rel_from_dir_deg(heading_rad: float, toward_deg: float) -> float:
    """
    Direction the current comes FROM, relative to the bow, clockwise [deg].
    """

    return _wrap360(np.degrees(heading_rad) - (toward_deg + 180.0))


def heading_for(toward_deg: float, rel_from_deg: float) -> float:
    """
    Start heading [deg, env frame] that puts a current flowing toward toward_deg at rel_from_deg.
    """

    return _wrap180(toward_deg + 180.0 + rel_from_deg)


# --------------------------------------------------------------------------
# Calibration: current at the start pose vs. timeStart
# --------------------------------------------------------------------------

@dataclass
class CurrentLevel:
    """
    One calibrated current level: the timeStart that gives it and the current at the start pose.
    """

    name: str                 # e.g. "c1.12" (current speed at the start pose)
    time_start: int
    speed: float              # current at the start pose [m/s]
    toward_deg: float         # direction it flows toward [deg, env frame]


def calibrate_current(start_x, start_y, start_heading_deg, v_init, thrust, ship_spec,
                      exercise_path: Path) -> Tuple[List[CurrentLevel], List[dict]]:
    """
    Scan timeStart over CURRENT_TIME_START_RANGE and pick the test levels.
    """

    lo, hi = CURRENT_TIME_START_RANGE
    scan = []
    for ts in range(int(lo), int(hi) + 1, int(CAL_TIME_STEP_S)):
        set_time_start(ts)
        fh = _session(start_x, start_y, start_heading_deg, v_init, thrust, ship_spec, exercise_path,
                      f"calibrate_current ts={ts}")
        try:
            if not scan:
                print(f"calibrate_current: FH loaded exercise {_loaded_exercise(fh)}", flush=True)
            cs, cd = current_toward_deg(fh)
            ukc = fh.get_ukc()
        finally:
            fh.close()
        scan.append(dict(time_start=ts, current_speed=cs, current_toward_deg=cd, ukc=ukc))
        print(f"calibrate_current: timeStart {ts:+7d} s -> current {cs:.3f} m/s toward {cd:6.1f} deg, "
              f"UKC {ukc:.2f} m", flush=True)

    vmax = max(s["current_speed"] for s in scan)
    vmin = min(s["current_speed"] for s in scan)
    if vmax - vmin < CAL_MIN_SPREAD:
        raise CurrentTestSetupError(
            f"calibrate_current: the current at the start pose hardly changes with timeStart "
            f"({vmin:.3f}-{vmax:.3f} m/s over {len(scan)} values) -- simexe may not re-read the XMLs per "
            f"session, or the exercise does not use them.")
    if vmax < CAL_MIN_CURRENT:
        raise CurrentTestSetupError(
            f"calibrate_current: the strongest current at the start pose is {vmax:.3f} m/s "
            f"(< {CAL_MIN_CURRENT}) -- wrong exercise (no current) or timeStart has no effect.")
    targets = (list(CURRENT_SPEED_TARGETS) if CURRENT_SPEED_TARGETS
               else [f * vmax for f in CURRENT_LEVEL_FRACTIONS])
    if len(targets) > len(scan):
        raise CurrentTestSetupError(f"calibrate_current: {len(targets)} levels requested but only {len(scan)} "
                                    f"timeStarts scanned; decrease CAL_TIME_STEP_S.")
    levels, used = [], set()
    for tgt in targets:
        best = min((s for s in scan if s["time_start"] not in used), key=lambda s: abs(s["current_speed"] - tgt))
        used.add(best["time_start"])
        if abs(best["current_speed"] - tgt) > 0.15 * max(tgt, 1e-6):
            logger.warning("calibrate_current: target %.2f m/s -> closest scanned current %.3f m/s "
                           "(timeStart %d); decrease CAL_TIME_STEP_S for a closer match.",
                           tgt, best["current_speed"], best["time_start"])
        levels.append(CurrentLevel(name=f"c{best['current_speed']:.2f}", time_start=int(best["time_start"]),
                                   speed=float(best["current_speed"]), toward_deg=float(best["current_toward_deg"])))
    levels.sort(key=lambda lv: lv.speed)
    names = [lv.name for lv in levels]
    if len(set(names)) != len(names):
        raise CurrentTestSetupError(f"calibrate_current: two levels have the same current speed ({names}); "
                                    f"change CURRENT_LEVEL_FRACTIONS / CURRENT_SPEED_TARGETS.")
    print("calibrate_current: levels " + "; ".join(
        f"{lv.name}: timeStart {lv.time_start} s, {lv.speed:.3f} m/s toward {lv.toward_deg:.1f} deg" for lv in levels),
        flush=True)
    return levels, scan


# --------------------------------------------------------------------------
# One run: FH (+ all models) for one level, one heading, one exercise
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    One current-test run (FH and every model, one level, one relative direction).
    The baseline ("_nocurrent") run has with_current=False.
    """

    run_id: str
    level: str
    time_start: int
    target_rel_dir_deg: float
    heading_deg: float
    with_current: bool
    actual_current_speed: float
    actual_rel_dir_deg: float
    current_speed_min: float
    current_speed_max: float
    residual_wind_max: float
    paths: Dict[str, List[TrackPoint]]
    env_log: List[dict]
    first_bank_hit: Dict[str, Optional[int]]
    domain_exit: Optional[Tuple[int, str]]
    hit_max_steps: bool
    onestep: Dict[str, dict] = field(default_factory=dict)


def run_case(run_id: str, level: CurrentLevel, rel_deg: float, heading_deg: float, with_current: bool,
             start_x, start_y, v_init, thrust, ship_spec, exercise_path: Path,
             ray_env: PySimEnv, cutoff_geom, non_fh_models: dict, ua_models: dict,
             tf_models: Optional[dict] = None, n_steps: Optional[int] = None) -> RunResult:
    """
    Run one case. n_steps: fixed step count (the baseline matches its current run);
    None = until FH has sailed TARGET_PATH_LENGTH.
    """

    set_time_start(level.time_start)
    fh = _session(start_x, start_y, heading_deg, v_init, thrust, ship_spec, exercise_path, run_id)
    try:
        print(f"{run_id}: FH loaded exercise {_loaded_exercise(fh)} (timeStart {level.time_start} s)", flush=True)
        cs0, cd0 = current_toward_deg(fh)
        start = _fh_point(fh, 0.0)
        rel_actual = rel_from_dir_deg(start.heading, cd0)
        if with_current:
            if abs(_wrap180(rel_actual - rel_deg)) > RUN_DIR_TOL_DEG:
                logger.warning("%s: actual relative current direction %.1f deg vs target %.1f deg",
                               run_id, rel_actual, rel_deg)
            if abs(cs0 - level.speed) > RUN_SPEED_TOL_REL * level.speed:
                logger.warning("%s: current %.3f m/s at the start vs %.3f m/s in the calibration",
                               run_id, cs0, level.speed)
        elif cs0 > MAX_BASELINE_CURRENT:
            raise CurrentTestSetupError(
                f"{run_id}: the no-current exercise has a current of {cs0:.3f} m/s (> {MAX_BASELINE_CURRENT}).")
        if _rays_hit(ray_env, start.x, start.y, start.heading):
            msg = f"{run_id}: start pose ({start.x:.2f}, {start.y:.2f}) is not in open water."
            if REQUIRE_OPEN_WATER_START:
                raise CurrentTestSetupError(msg)
            print(msg, flush=True)

        _, _, _, _, v0, r0 = fh.state()
        actuator = ActuatorCommand(rudder_angle=RUDDER_ANGLE_RAD, thrust=thrust)
        sims = {}
        for label, model in non_fh_models.items():
            sim = make_model_simulator(ship_spec, model)
            sim.reset(VesselState(terminated=False, truncated=False, x=start.x, y=start.y,
                                  heading=start.heading, u=start.u, v=v0, r=r0))
            sims[label] = sim
        surge_clip = SurgeClipCounter(ship_spec, sims)

        paths = {"FH": [start]}
        paths.update({label: [TrackPoint(0.0, start.x, start.y, start.heading, start.u)] for label in sims})
        first_hit: Dict[str, Optional[int]] = {label: None for label in paths}
        env_log: List[dict] = []
        domain_exit = None
        speeds, residual = [cs0], 0.0
        fh_length, step_i = 0.0, 0
        last_raw, held_true, held_apparent = None, None, {}
        tf_pred: Dict[str, list] = {label: [] for label in (tf_models or {})}
        tf_fh: list = []
        tf_pre: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # FH (u, v, r) before the step

        def keep_going():
            """
            True until n_steps, or until FH has sailed TARGET_PATH_LENGTH (at most MAX_STEPS).
            """

            if n_steps is not None:
                return step_i < n_steps
            return fh_length < TARGET_PATH_LENGTH and step_i < MAX_STEPS

        while keep_going():
            fh_ukc = fh.get_ukc()
            fh_cs, fh_cd = fh.get_current()
            if not with_current and fh_cs > MAX_BASELINE_CURRENT:
                raise CurrentTestSetupError(f"{run_id}: no-current exercise shows {fh_cs:.3f} m/s at step {step_i}.")
            speeds.append(fh_cs)
            raw = fh.get_wind_raw()
            if raw != last_raw:     # FH's WindState updates at 1 Hz: sample and hold, as in the wind test
                last_raw = raw
                held_true = _true_wind(fh)
                residual = max(residual, float(held_true[0]))
                w_world = _true_wind_world(fh)
                for label in ua_models:
                    held_apparent[label] = _apparent_wind_for(sims[label].state, w_world)
            _, _, fh_h, fh_u, fh_v, fh_r = fh.state()
            cur = paths["FH"][-1]
            if domain_exit is None:
                issue = _training_domain_issue(ray_env, cutoff_geom, cur.x, cur.y, cur.heading, fh_ukc)
                if issue is not None:
                    domain_exit = (step_i, issue)
            log = dict(step=step_i, t=step_i * SURROGATE_DT, fh_current_speed=fh_cs,
                       fh_current_rel_from_deg=rel_from_dir_deg(fh_h, _wrap360(np.degrees(fh_cd))),
                       fh_current_toward_deg=_wrap360(np.degrees(fh_cd)), fh_ukc=fh_ukc,
                       fh_true_wind_speed=held_true[0], fh_sway=fh_v, fh_yaw_rate=fh_r)

            if tf_models:   # teacher-forced one-step check at FH's state and environment
                fs = VesselState(terminated=False, truncated=False, x=cur.x, y=cur.y, heading=fh_h,
                                 u=fh_u, v=fh_v, r=fh_r)
                fw_s, fw_d = fh.get_wind()
                for label, model in tf_models.items():
                    if label in ua_models:
                        _inject_live_environment(model, fh_ukc, fw_s, fw_d, fh_cs, fh_cd, ray_env)
                    tf_pred[label].append(tuple(float(a) for a in
                                                model._compute_accelerations(fs, RUDDER_ANGLE_RAD, thrust)))
                tf_pre = (fh_u, fh_v, fh_r)

            # UA models: FH's live UKC and current, apparent wind for their own state.
            for label, model in ua_models.items():
                own_ws, own_wd = held_apparent[label]
                _inject_live_environment(model, fh_ukc, own_ws, own_wd, fh_cs, fh_cd, ray_env)
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

    if residual > MAX_RESIDUAL_WIND:
        logger.warning("%s: true wind up to %.2f m/s with wind-field Scale 0 (FH's WindState vs its motion over "
                       "ground in current); UA_full gets the same apparent wind FH reports.", run_id, residual)
    return RunResult(run_id=run_id, level=level.name, time_start=level.time_start, target_rel_dir_deg=float(rel_deg),
                     heading_deg=float(heading_deg), with_current=with_current, actual_current_speed=cs0,
                     actual_rel_dir_deg=rel_actual, current_speed_min=float(min(speeds)),
                     current_speed_max=float(max(speeds)), residual_wind_max=residual, paths=paths,
                     env_log=env_log, first_bank_hit=first_hit, domain_exit=domain_exit,
                     hit_max_steps=(n_steps is None and step_i >= MAX_STEPS),
                     onestep=_onestep_stats(tf_pred, tf_fh))


def _all_paths(res: RunResult, base: Optional[RunResult]) -> Dict[str, List[TrackPoint]]:
    """
    The run's own paths plus FH_nocurrent and UA_full_nocurrent from the baseline run, if given.
    """

    paths = dict(res.paths)
    if base is not None:
        paths["FH_nocurrent"] = base.paths["FH"]
        if "UA_full" in base.paths:
            paths["UA_full_nocurrent"] = base.paths["UA_full"]
    return paths


# --------------------------------------------------------------------------
# CSV output
# --------------------------------------------------------------------------

SUMMARY_HEADER = [
    "level", "time_start", "target_rel_dir_deg", "heading_deg", "actual_current_speed", "actual_rel_dir_deg",
    "current_speed_min_along_path", "current_speed_max_along_path", "model", "n_steps", "duration_s",
    "final_along", "final_lateral", "final_turn_deg", "final_u",
    "along_error_vs_fh", "lateral_error_vs_fh", "turn_error_vs_fh_deg", "final_pos_error_vs_fh",
    "rms_lateral_error_vs_fh",
    # paired current effect: (model in current - same model in its no-current run) vs. the same for FH
    "effect_n_steps", "current_effect_along", "current_effect_lateral", "current_effect_turn_deg",
    "fh_current_effect_along", "fh_current_effect_lateral", "fh_current_effect_turn_deg",
    "current_effect_along_error", "current_effect_lateral_error", "current_effect_turn_error_deg",
    "first_bank_hit_step", "fh_domain_exit_step", "fh_domain_exit_reason",
]


def summary_rows(res: RunResult, base: RunResult) -> List[list]:
    """
    current_summary.csv rows, one per model: final track metrics, errors versus FH,
    and the paired current effect (run minus baseline) versus FH's.
    """

    rows = []
    fh_path = res.paths["FH"]
    allp = _all_paths(res, base)
    for label in MODEL_ORDER:
        pts = allp.get(label)
        if not pts:
            continue
        n = min(len(pts), len(fh_path))
        m = [track_metrics(pts[0], p) for p in pts[:n]]
        fm = [track_metrics(fh_path[0], p) for p in fh_path[:n]]
        final, ffinal = m[-1], fm[-1]
        lat_err = np.array([a["lateral"] - b["lateral"] for a, b in zip(m, fm)])
        pos_err = float(np.hypot(pts[n - 1].x - fh_path[n - 1].x, pts[n - 1].y - fh_path[n - 1].y))
        effect = [""] * 10
        if label in res.paths and label in base.paths:
            b_pts, b_fh = base.paths[label], base.paths["FH"]
            ne = min(len(pts), len(b_pts), len(fh_path), len(b_fh)) - 1
            if ne > 0:
                me, mb = track_metrics(pts[0], pts[ne]), track_metrics(b_pts[0], b_pts[ne])
                fe, fb = track_metrics(fh_path[0], fh_path[ne]), track_metrics(b_fh[0], b_fh[ne])
                e = (me["along"] - mb["along"], me["lateral"] - mb["lateral"], _wrap180(me["turn"] - mb["turn"]))
                f = (fe["along"] - fb["along"], fe["lateral"] - fb["lateral"], _wrap180(fe["turn"] - fb["turn"]))
                effect = [ne, *e, *f, e[0] - f[0], e[1] - f[1], _wrap180(e[2] - f[2])]
        rows.append([
            res.level, res.time_start, res.target_rel_dir_deg, res.heading_deg, res.actual_current_speed,
            res.actual_rel_dir_deg, res.current_speed_min, res.current_speed_max, label, n - 1, pts[n - 1].t,
            final["along"], final["lateral"], final["turn"], final["u"],
            final["along"] - ffinal["along"], final["lateral"] - ffinal["lateral"],
            _wrap180(final["turn"] - ffinal["turn"]), pos_err, float(np.sqrt(np.mean(lat_err ** 2))),
            *effect,
            res.first_bank_hit.get(label, ""),
            res.domain_exit[0] if res.domain_exit else "", res.domain_exit[1] if res.domain_exit else "",
        ])
    return rows


ONESTEP_HEADER = ["level", "target_rel_dir_deg", "with_current", "model", "n_steps", "duration_s",
                  "bias_du", "bias_dv", "bias_dr", "rms_err_du", "rms_err_dv", "rms_err_dr",
                  "fh_rms_du", "fh_rms_dv", "fh_rms_dr"]


def onestep_rows(res: RunResult) -> List[list]:
    """
    Rows for the one-step CSV (ONESTEP_HEADER), one per model with teacher-forced statistics.
    """

    return [[res.level, res.target_rel_dir_deg, int(res.with_current), label, st["n"], st["T"],
             *st["bias"], *st["rms"], *st["fh_rms"]]
            for label in MODEL_ORDER if (st := res.onestep.get(label))]


def write_trajectory_csv(path: Path, res: RunResult, base: RunResult) -> None:
    """
    Write every model's path (including the no-current reference lines) with its track metrics.
    """

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "t", "x", "y", "heading", "u", "along", "lateral", "turn_deg"])
        allp = _all_paths(res, base)
        for label in MODEL_ORDER:
            pts = allp.get(label)
            for i, p in enumerate(pts or []):
                m = track_metrics(pts[0], p)
                w.writerow([label, i, p.t, p.x, p.y, p.heading, p.u, m["along"], m["lateral"], m["turn"]])


def write_env_csv(path: Path, res: RunResult) -> None:
    """
    Write the run's per-step environment log; nothing is written if the log is empty.
    """

    if res.env_log:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res.env_log[0].keys()))
            w.writeheader()
            w.writerows(res.env_log)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _dir_label(rel: float) -> str:
    """
    Readable name for a relative current direction [deg] (e.g. "from starboard").
    """

    names = {0: "head current", 90: "from starboard", 180: "following", 270: "from port"}
    return names.get(int(round(rel)) % 360, f"from {rel:g}°")


def plot_paths_for_level(results: List[RunResult], bases: Dict[str, RunResult], png_path: Path, title: str,
                         models=("FH", "FH_nocurrent", "UGent", "Fossen", "UA_clean", "UA_confined",
                                 "UA_full", "UA_full_nocurrent")) -> None:
    """
    Ship-fixed view per direction (axes fixed to the start heading, bow up at start).
    """

    if plt is None:
        return
    n = len(results)
    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, res in zip(axes, results):
        allp = _all_paths(res, bases.get(res.run_id))
        for label in models:
            pts = allp.get(label)
            if not pts:
                continue
            m = [track_metrics(pts[0], p) for p in pts]
            ax.plot([q["lateral"] for q in m], [q["along"] for q in m], color=COLORS.get(label, "#000"),
                    linestyle=LINESTYLES.get(label, "-"), lw=2.2 if label == "FH" else 1.4, label=label,
                    zorder=3 if label == "FH" else 2)
        ax.plot(0, 0, "o", color="#000", ms=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{_dir_label(res.target_rel_dir_deg)} ({res.target_rel_dir_deg:g}°)\n"
                     f"{res.actual_current_speed:.2f} m/s", fontsize=9)
        ax.set_xlabel("sideways [m] (stbd +)", fontsize=8)
        ax.set_ylabel("ahead [m]", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.3, alpha=0.5)
    for ax in axes[n:]:
        ax.axis("off")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, png_path)


# Slide figures (same style as the wind slides).
_INK, _INK2, _MUTED, _GRID, _BLUE, _GRAY = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#2a78d6", "#b4b3ad"


def _slide_style():
    """
    Set matplotlib rcParams for the slide figures.
    """

    plt.rcParams.update({"font.size": 13, "axes.edgecolor": _MUTED, "axes.labelcolor": _INK2, "xtick.color": _INK2,
                         "ytick.color": _INK2, "axes.titlesize": 15, "axes.titleweight": "bold",
                         "axes.titlecolor": _INK, "axes.spines.top": False, "axes.spines.right": False})


def slide_response_per_direction(d, level: str, png_path: Path) -> None:
    """
    FH vs UA_full current effect per relative direction, for one level (3 panels).
    """

    if plt is None:
        return
    import matplotlib as mpl
    # noinspection PyTypeChecker
    with mpl.rc_context():
        _slide_style()
        cur = d[d["level"] == level]
        fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.2))
        for ax, (col, fcol, unit, title) in zip(axes, (
                ("current_effect_along", "fh_current_effect_along", "m", "Distance ahead"),
                ("current_effect_lateral", "fh_current_effect_lateral", "m", "Sideways (stbd +)"),
                ("current_effect_turn_deg", "fh_current_effect_turn_deg", "deg", "Heading change (stbd +)"))):
            fh = cur[cur["model"] == "FH"].sort_values("target_rel_dir_deg")
            ua = cur[cur["model"] == "UA_full"].sort_values("target_rel_dir_deg")
            xs = list(fh["target_rel_dir_deg"]) + [360]
            ax.axhline(0, color=_MUTED, lw=1)
            ax.plot(xs, list(fh[fcol]) + [fh[fcol].iloc[0]], color=_INK, lw=2.5, marker="o", ms=7,
                    label="FH simulator (ground truth)")
            if not ua.empty:
                ax.plot(xs, list(ua[col]) + [ua[col].iloc[0]], color=_BLUE, lw=2, marker="o", ms=7, mfc="white",
                        mew=2, label="UA_full")
            ax.set_xticks([0, 90, 180, 270, 360], ["0\nhead", "90\nstbd", "180\nfollow", "270\nport", "360"])
            ax.set_xlabel("Current from, rel. to bow [deg]")
            ax.set_ylabel(f"[{unit}]")
            ax.set_title(title, loc="left")
            ax.yaxis.grid(True, color=_GRID, lw=1)
            ax.set_axisbelow(True)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=12)
        spd = float(cur[cur["model"] == "FH"]["actual_current_speed"].mean())
        fig.suptitle(f"Effect of a {spd:.1f} m/s current on the ship's path, per current direction "
                     f"(run with current minus the same run without)", x=0.01, ha="left", fontsize=13, color=_INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.subplots_adjust(bottom=0.24)
        fig.savefig(str(png_path), dpi=200)
        plt.close(fig)
        print(f"Wrote {png_path}", flush=True)


def slide_example_paths(results: Dict[Tuple[str, float], RunResult], bases: Dict[str, RunResult], level: str,
                        png_path: Path, rels=(0.0, 90.0, 180.0)) -> None:
    """
    FH vs UA_full paths (ship-fixed view) for a few directions at one level.
    """

    if plt is None:
        return
    import matplotlib as mpl
    sel = [results[(level, r)] for r in rels if (level, r) in results]
    if not sel:
        return
    # noinspection PyTypeChecker
    with mpl.rc_context():
        _slide_style()
        fig, axes = plt.subplots(1, len(sel), figsize=(13.33, 6.2), sharey=True, squeeze=False)
        for ax, res in zip(axes[0], sel):
            allp = _all_paths(res, bases.get(res.run_id))
            for label, color, lw, ls, name in (("FH_nocurrent", _GRAY, 1.8, "--", "FH without current"),
                                               ("FH", _INK, 2.6, "-", "FH simulator (ground truth)"),
                                               ("UA_full", _BLUE, 2.4, "-", "UA_full surrogate")):
                pts = allp.get(label)
                if not pts:
                    continue
                m = [track_metrics(pts[0], p) for p in pts]
                ax.plot([q["lateral"] for q in m], [q["along"] for q in m], color=color, lw=lw, ls=ls, label=name)
                if label != "FH_nocurrent":
                    ax.plot(m[-1]["lateral"], m[-1]["along"], "o", color=color, ms=8, mec="white", mew=2)
            ax.plot(0, 0, "o", color=_INK2, ms=7)
            f_end, u_end = res.paths["FH"][-1], res.paths["UA_full"][-1] if "UA_full" in res.paths else None
            if u_end is not None:
                ax.text(0.98, 0.98, f"end-point gap {np.hypot(f_end.x - u_end.x, f_end.y - u_end.y):.0f} m",
                        transform=ax.transAxes, ha="right", va="top", fontsize=11, color=_INK2)
            ax.set_title(f"{_dir_label(res.target_rel_dir_deg).capitalize()} ({res.target_rel_dir_deg:g}°)", loc="left")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, color=_GRID, lw=1)
            ax.set_xlabel("Sideways [m]  (starboard +)")
        axes[0][0].set_ylabel("Ahead [m]")
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=12)
        spd = sel[0].actual_current_speed
        fig.suptitle(f"Example paths in a {spd:.1f} m/s current (axes fixed to the start heading, bow up at start)",
                     x=0.01, ha="left", fontsize=13, color=_INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(str(png_path), dpi=200)
        plt.close(fig)
        print(f"Wrote {png_path}", flush=True)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _effect_plot(d, col: str, fcol: str, ylabel: str, title: str) -> str:
    """
    One panel per current level: current effect col per model versus FH's fcol, over
    the relative direction. Returns a PNG data URI.
    """

    levels = sorted(d["level"].unique(), key=lambda s: float(s[1:]))
    fig, axes = plt.subplots(1, len(levels), figsize=(4.6 * len(levels), 3.8), squeeze=False, sharey=True)
    for ax, lv in zip(axes[0], levels):
        sub = d[d["level"] == lv]
        fh = sub[sub["model"] == "FH"].sort_values("target_rel_dir_deg")
        ax.plot(fh["target_rel_dir_deg"], fh[fcol], color=COLORS["FH"], lw=2.6, marker="o", ms=5,
                label="FH (reference)", zorder=4)
        for m in REPORT_MODELS:
            mm = sub[sub["model"] == m].sort_values("target_rel_dir_deg")
            if not mm.empty:
                ax.plot(mm["target_rel_dir_deg"], mm[col], color=COLORS.get(m, "#000"), lw=1.6,
                        linestyle=LINESTYLES.get(m, "-"), marker="o", ms=3.5, label=m, zorder=3)
        ax.axhline(0, color="#9a9a9a", lw=0.8)
        ax.set_title(f"current {float(lv[1:]):.2f} m/s", fontsize=11)
        ax.set_xticks(sorted(sub["target_rel_dir_deg"].unique()))
        ax.set_xlabel("current from, rel. to bow [deg]\n(0 = head, 90 = from stbd, 180 = following)", fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel(ylabel)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _fig_to_data_uri(fig)


def write_report(out_dir: Optional[Path] = None) -> Path:
    """
    Build the HTML report from the CSVs and run_meta.json (no re-simulation).
    """

    import base64
    import html
    import pandas as pd

    out_dir = Path(out_dir) if out_dir is not None else CURRENT_OUTPUT_DIR
    if plt is None:
        raise RuntimeError("write_report needs matplotlib.")
    d = pd.read_csv(out_dir / SUMMARY_NAME)
    meta = json.loads((out_dir / RUN_META_NAME).read_text()) if (out_dir / RUN_META_NAME).is_file() else {}
    cal = pd.read_csv(out_dir / CALIBRATION_NAME) if (out_dir / CALIBRATION_NAME).is_file() else None
    levels = sorted(d["level"].unique(), key=lambda name: float(name[1:]))
    models_present = [m for m in REPORT_MODELS if m in set(d["model"])]

    def mean_abs(col):
        """
        Mean |col| per model, per current level and over all levels, as a DataFrame.
        """

        out_rows = []
        for m in models_present:
            sub = d[d["model"] == m]
            r: Dict[str, object] = {"model": m}
            for level in levels:
                r[f"{float(level[1:]):.2f} m/s"] = float(sub[sub["level"] == level][col].abs().mean())
            r["all"] = float(sub[col].abs().mean())
            out_rows.append(r)
        return pd.DataFrame(out_rows)

    t_along = mean_abs("current_effect_along_error")
    t_lat = mean_abs("current_effect_lateral_error")
    t_turn = mean_abs("current_effect_turn_error_deg")
    t_rms = mean_abs("rms_lateral_error_vs_fh")

    fh = d[d["model"] == "FH"]
    ua = d[d["model"] == "UA_full"]
    ag = d[d["model"].isin(NO_CURRENT_MODELS)]

    def m_abs(df, col):
        """
        Mean |col| of df, NaN for an empty frame.
        """

        return float(df[col].abs().mean()) if not df.empty else float("nan")

    fh_pos_eff = np.hypot(fh["fh_current_effect_along"], fh["fh_current_effect_lateral"])
    kpis = [
        (float(fh_pos_eff.max()) if not fh.empty else float("nan"), "largest FH current effect on position [m]"),
        (float(fh["fh_current_effect_turn_deg"].abs().max()) if not fh.empty else float("nan"),
         "largest FH current effect on heading [deg]"),
        (float(np.hypot(ua["current_effect_along_error"], ua["current_effect_lateral_error"]).mean())
         if not ua.empty else float("nan"), "UA_full mean position error of the current effect [m]"),
        (float(np.hypot(ag["current_effect_along_error"], ag["current_effect_lateral_error"]).mean())
         if not ag.empty else float("nan"), "no-current models mean position error [m]"),
        (m_abs(ua, "current_effect_turn_error_deg"), "UA_full mean |heading-effect error| [deg]"),
        (m_abs(ag, "current_effect_turn_error_deg"), "no-current models mean |heading-effect error| [deg]"),
    ]

    det = []
    for lv in levels:
        for rd in sorted(d["target_rel_dir_deg"].unique()):
            f = fh[(fh["level"] == lv) & (fh["target_rel_dir_deg"] == rd)]
            u = ua[(ua["level"] == lv) & (ua["target_rel_dir_deg"] == rd)]
            if f.empty:
                continue
            row = {"current [m/s]": float(f["actual_current_speed"].iloc[0]), "from [deg]": rd,
                   "FH ahead [m]": float(f["fh_current_effect_along"].iloc[0]),
                   "FH sideways [m]": float(f["fh_current_effect_lateral"].iloc[0]),
                   "FH turn [deg]": float(f["fh_current_effect_turn_deg"].iloc[0])}
            if not u.empty:
                row.update({"UA_full ahead [m]": float(u["current_effect_along"].iloc[0]),
                            "UA_full sideways [m]": float(u["current_effect_lateral"].iloc[0]),
                            "UA_full turn [deg]": float(u["current_effect_turn_deg"].iloc[0])})
            det.append(row)
    detail = pd.DataFrame(det)

    q = fh[["level", "target_rel_dir_deg", "time_start", "heading_deg", "actual_current_speed", "actual_rel_dir_deg",
            "current_speed_min_along_path", "current_speed_max_along_path", "n_steps", "fh_domain_exit_step"]].copy()
    hits = d.groupby(["level", "target_rel_dir_deg"])["first_bank_hit_step"].apply(lambda hit_steps: int(hit_steps.notna().sum()))
    q["models with bank in ray range"] = [hits.get((a, b), 0) for a, b in zip(q["level"], q["target_rel_dir_deg"])]
    q.columns = ["level", "target from [deg]", "timeStart [s]", "start heading [deg]", "current at start [m/s]",
                 "actual from [deg]", "current min on path", "current max on path", "steps",
                 "FH left training domain at step", "models with bank in ray range"]

    # one-step (current runs only), grouped by side
    os_html = ""
    if (out_dir / ONESTEP_NAME).is_file():
        o = pd.read_csv(out_dir / ONESTEP_NAME)
        o = o[o["with_current"] == 1]
        if not o.empty:

            def grp(rel_dir):
                """
                Side group of a relative current direction [deg] for the one-step table.
                """

                if 0 < rel_dir < 180:
                    return "from starboard"
                return "from port" if rel_dir > 180 else "head / following"

            o["side"] = o["target_rel_dir_deg"].map(grp)
            rows = []
            for m in [x for x in MODEL_ORDER if x in set(o["model"])]:
                for g in ("head / following", "from starboard", "from port"):
                    s = o[(o["model"] == m) & (o["side"] == g)]
                    if s.empty:
                        continue

                    def pct(k):   # runs where FH's own signal is ~0 are left out of the ratio
                        """
                        Mean RMS one-step error of k as % of FH's RMS (runs with FH RMS > ONESTEP_MIN_FH_RMS[k]).
                        """

                        ok = s[f"fh_rms_{k}"] > ONESTEP_MIN_FH_RMS[k]
                        return 100 * float((s.loc[ok, f"rms_err_{k}"] / s.loc[ok, f"fh_rms_{k}"]).mean()) if ok.any() \
                            else float("nan")
                    rows.append({"model": m, "current": g,
                                 "RMS du error / RMS FH du [%]": pct("du"),
                                 "RMS dv error / RMS FH dv [%]": pct("dv"),
                                 "RMS dr error / RMS FH dr [%]": pct("dr"),
                                 "RMS du error [m/s^2]": float(s["rms_err_du"].mean()),
                                 "RMS dv error [m/s^2]": float(s["rms_err_dv"].mean()),
                                 "RMS dr error [deg/s^2]": float(np.degrees(s["rms_err_dr"]).mean())})
            os_html = ("<h2>5. One-step (teacher-forced) check</h2><p>At every FH step each model gets exactly FH's "
                       "state and environment (current, UKC, wind sensor, rays) and its predicted accelerations are "
                       "compared with FH's change over that 0.1 s step (runs with current only). Low one-step errors "
                       "with large path errors mean that small errors accumulate along the path.</p>"
                       + _table_html(pd.DataFrame(rows), "{:.1f}",
                                     col_fmts={"RMS du error [m/s^2]": "{:.4f}", "RMS dv error [m/s^2]": "{:.4f}",
                                               "RMS dr error [deg/s^2]": "{:.4f}"}))

    fig_along = _effect_plot(d, "current_effect_along", "fh_current_effect_along", "current effect ahead [m]",
                             "Figure 1 - Current effect on the distance sailed ahead (model vs FH)")
    fig_lat = _effect_plot(d, "current_effect_lateral", "fh_current_effect_lateral",
                           "current effect sideways [m]\n(stbd +)", "Figure 2 - Current effect on sideways position")
    fig_turn = _effect_plot(d, "current_effect_turn_deg", "fh_current_effect_turn_deg",
                            "current effect on heading [deg]\n(stbd +)", "Figure 3 - Current effect on heading")

    def kpi(v, label):
        """
        HTML KPI tile for v ("n/a" for None/NaN).
        """

        txt = "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"
        return f'<div class="kpi"><div class="v">{txt}</div><div class="l">{html.escape(label)}</div></div>'

    def img(path: Path, alt: str) -> str:
        """
        HTML figure embedding the PNG at path as a data URI ("" if missing).
        """

        if not path.is_file():
            return ""
        uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<figure><img src="{uri}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'

    setup = [("Ship model", meta.get("ship_model", "")), ("Current exercise", meta.get("exercise", "")),
             ("No-current exercise", meta.get("exercise_nocurrent", "")),
             ("Start position", meta.get("start_pos", "")),
             ("Initial surge / thrust / rudder", meta.get("speed_thrust_rudder", "")),
             ("Path length per run", meta.get("path_length", "")), ("Grid", meta.get("grid", "")),
             ("Inputs to the UA models", meta.get("ua_inputs", "")),
             ("Run started / finished", f'{meta.get("started", "")} / {meta.get("finished", "")}')]
    setup_html = "<table><tbody>" + "".join(
        f"<tr><th>{html.escape(k)}</th><td style='text-align:left;white-space:normal'>{html.escape(str(v))}</td></tr>"
        for k, v in setup) + "</tbody></table>"
    warns = meta.get("warnings", [])
    warn_html = ("<ul>" + "".join(f'<li class="warn">{html.escape(w)}</li>' for w in warns) + "</ul>"
                 if warns else "<p>No warnings were logged during the run.</p>")
    cal_html = _table_html(cal, "{:.3f}", int_cols=("time_start",)) if cal is not None else "<p>n/a</p>"
    appendix = "".join(img(out_dir / f"current_paths_{lv}.png", f"Paths per direction, current {float(lv[1:]):.2f} m/s "
                                                                 f"(ship-fixed view, bow up at start)") for lv in levels)
    slides = (img(out_dir / "slide_current_response_per_direction.png", "Slide figure - response per direction")
              + img(out_dir / "slide_current_example_paths.png", "Slide figure - example paths"))

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Current Model Comparison</title><style>{_REPORT_CSS}</style></head>
<body><main>
<h1>Current response of ship dynamics models compared with FH</h1>
<p class="sub">{html.escape(str(meta.get("ship_model", "")))} in open water, rudder 0, fixed thrust, no wind &middot; generated {html.escape(str(meta.get("finished", "")))} by model_comparison_current.py</p>
<h2>1. Summary</h2>
<div class="kpis">{"".join(kpi(v, l) for v, l in kpis)}</div>
<p>Current effect = (model in current) - (the same model in the same run without current: same start, heading and water level), at the same time step. Error = model's current effect - FH's current effect. UGent, Fossen, UA_clean and UA_confined have no current input; they are the "no current response" reference. Like UA_full they start from FH's state with velocities over ground (as in the training data), which in a current already include the current; their small "current effect" is their response to that different start state. UA_full is the only current-aware model.</p>
<h3>Table 1a - Mean |error| of the current effect on the distance ahead [m]</h3>{_table_html(t_along, "{:.1f}")}
<h3>Table 1b - Mean |error| of the current effect sideways [m]</h3>{_table_html(t_lat, "{:.1f}")}
<h3>Table 1c - Mean |error| of the current effect on heading [deg]</h3>{_table_html(t_turn, "{:.1f}")}
<h2>2. Test setup</h2>{setup_html}
<h3>Calibration scan (current at the start position vs. timeStart)</h3>{cal_html}
<h2>3. Method</h2><ul>
<li><b>Ground truth</b>: the FH simulator. The current is set by the tidal phase (&lt;timeStart&gt; in the current and tide XML files, as in the dataset generator); its direction relative to the ship by the start heading. Wind-field scale 0.</li>
<li><b>Baseline per run</b>: the same start, heading and timeStart on the no-current exercise (same water level, no current). Every model is run in both.</li>
<li><b>UA model inputs</b>: FH's live UKC and FH's current at FH's position (FH has no current query at other positions); the apparent wind for each model's own motion, as in the wind test.</li>
<li><b>Direction convention</b>: the direction the current comes from, relative to the bow, clockwise: 0&deg; head current, 90&deg; from starboard, 180&deg; following, 270&deg; from port.</li></ul>
<h2>4. Results</h2>
<figure><img src="{fig_along}" alt="Figure 1"></figure>
<figure><img src="{fig_lat}" alt="Figure 2"></figure>
<figure><img src="{fig_turn}" alt="Figure 3"></figure>
<h3>Table 2 - Per-direction detail: FH and UA_full current effect</h3>{_table_html(detail, "{:.1f}", int_cols=("from [deg]",), col_fmts={"current [m/s]": "{:.2f}"})}
<h3>Table 3 - Mean RMS sideways deviation from FH over the whole path [m] (includes no-current error)</h3>{_table_html(t_rms, "{:.1f}")}
{slides}
{os_html}
<h2>6. Data quality per run</h2>{_table_html(q, "{:.2f}", int_cols=("target from [deg]", "timeStart [s]", "steps", "FH left training domain at step", "models with bank in ray range"))}
<h3>Warnings logged during the run</h3>{warn_html}
<h2>7. Limitations</h2><ul>
<li>The UA models get FH's current at FH's position; a model that drifts away from FH's path still gets the current at FH's position.</li>
<li>The current field is not uniform: along FH's path the current ranges between the min/max per run in section 6.</li>
<li>Changing timeStart also changes the water level (UKC); the baseline uses the same timeStart, so the water level is the same in both runs of a pair.</li>
<li>One start position, one ship speed (rudder 0, fixed thrust): straight-line sailing in open water only. Headings across the river may come near a bank within the path length (section 6).</li></ul>
<h2>Appendix - paths per direction</h2>{appendix or "<p>No plots found.</p>"}
</main></body></html>
"""
    out = out_dir / REPORT_NAME
    out.write_text(doc, encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _clean_output_dir(out_dir: Path) -> None:
    """
    Remove this script's outputs of a previous run, so the report never mixes runs.
    """

    patterns = ("trajectory_*.csv", "env_*.csv", "current_paths_*.png", "slide_current_*.png", SUMMARY_NAME,
                ONESTEP_NAME, CALIBRATION_NAME, RUN_META_NAME, REPORT_NAME)
    n = 0
    for pat in patterns:
        for f in out_dir.glob(pat):
            f.unlink()
            n += 1
    if n:
        print(f"Removed {n} output file(s) of a previous run from {out_dir}", flush=True)


def _check_fh_directions(results: List[RunResult], bases: Dict[str, RunResult]) -> None:
    """
    Check the direction convention on FH's paths (head current reduces the distance
    ahead; current from starboard sets to port, from port to starboard). Warns only.
    """

    expect = {0.0: ("along", -1), 90.0: ("lateral", -1), 180.0: ("along", +1), 270.0: ("lateral", +1)}
    for res in results:
        key = expect.get(res.target_rel_dir_deg)
        base = bases.get(res.run_id)
        if key is None or base is None:
            continue
        n = min(len(res.paths["FH"]), len(base.paths["FH"])) - 1
        eff = (track_metrics(res.paths["FH"][0], res.paths["FH"][n])[key[0]]
               - track_metrics(base.paths["FH"][0], base.paths["FH"][n])[key[0]])
        if eff * key[1] <= 0:
            logger.warning("%s: FH's current effect %s = %+.1f m has the wrong sign for a current from %g deg "
                           "-- check the direction convention (current measurement / heading_for).",
                           res.run_id, key[0], eff, res.target_rel_dir_deg)


def _print_run(res: RunResult) -> None:
    """
    Print one run's current, FH's final position and heading change, and any domain/bank/MAX_STEPS issue.
    """

    fh_final = track_metrics(res.paths["FH"][0], res.paths["FH"][-1])
    print(f"{res.run_id}: current {res.actual_current_speed:.2f} m/s from {res.actual_rel_dir_deg:.1f} deg rel. bow "
          f"(along path {res.current_speed_min:.2f}-{res.current_speed_max:.2f}), heading {res.heading_deg:.1f} deg, "
          f"{len(res.paths['FH']) - 1} steps; FH final ahead {fh_final['along']:.1f} m, sideways "
          f"{fh_final['lateral']:+.1f} m, turn {fh_final['turn']:+.1f} deg", flush=True)
    if res.domain_exit:
        print(f"{res.run_id}: FH left the training domain at step {res.domain_exit[0]} ('{res.domain_exit[1]}').",
              flush=True)
    hits = {k: v for k, v in res.first_bank_hit.items() if v is not None}
    if hits:
        print(f"{res.run_id}: banks within ray range (first step per model): {hits}", flush=True)
    if res.hit_max_steps:
        print(f"{res.run_id}: hit MAX_STEPS before {TARGET_PATH_LENGTH:g} m.", flush=True)


def generate_current_trajectories(start_x: float, start_y: float, start_heading_deg: float,
                                  exercise_path: Optional[Path] = None,
                                  exercise_path_nocurrent: Optional[Path] = None,
                                  rel_dirs_deg: Sequence[float] = REL_CURRENT_DIRS_DEG, speed: float = CURRENT_TRAJECTORY_SPEED,
                                  calibrate_only: bool = False) -> None:
    """
    Run the current test. start_heading_deg is used only for calibration; every run
    sets its own heading.
    """

    exercise_path = Path(exercise_path or CURRENT_EXERCISE_PATH or "")
    exercise_path_nocurrent = Path(exercise_path_nocurrent or NOCURRENT_EXERCISE_PATH or "")
    for p, name in ((exercise_path, "exercise_path"), (exercise_path_nocurrent, "exercise_path_nocurrent")):
        if not p.is_file():
            raise ValueError(f"generate_current_trajectories: {name} '{p}' is not a file.")
    CURRENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recover_stale_backup()
    if CLEAN_OUTPUT_DIR and not calibrate_only:
        _clean_output_dir(CURRENT_OUTPUT_DIR)
    original = snapshot_xmls()
    print("Original <timeStart>: " + ", ".join(f"{p.name}={_time_start_text(r)}" for p, r in original.items())
          + f" (backup in {_backup_dir()})", flush=True)
    warn = _ListHandler()
    logger.addHandler(warn)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        _generate(start_x, start_y, start_heading_deg, exercise_path, exercise_path_nocurrent, rel_dirs_deg,
                  speed, started, warn, calibrate_only)
    except BaseException:
        logger.removeHandler(warn)
        restore_xmls(original)      # errors are printed; the original exception is what is raised
        raise
    logger.removeHandler(warn)
    if restore_xmls(original):
        raise RuntimeError(f"The run finished, but the XMLs could not be restored; originals in {_backup_dir()}.")


def _generate(start_x, start_y, start_heading_deg, exercise_path, exercise_path_nocurrent, rel_dirs_deg, speed,
              started, warn, calibrate_only) -> None:
    """
    Calibrate the levels, run every (level, direction) case with its baseline, and
    write CSVs, plots, run_meta.json and the report (calibrate_only: stop after the
    calibration CSV).
    """

    import pandas as pd

    ship_spec = create_ship(SHIP_MODEL).specifications
    thrust = float(min(1.0, speed / 6.0))
    common = dict(v_init=float(speed), thrust=thrust, ship_spec=ship_spec)

    levels, scan = calibrate_current(float(start_x), float(start_y), float(start_heading_deg),
                                     exercise_path=exercise_path, **common)
    with open(CURRENT_OUTPUT_DIR / CALIBRATION_NAME, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scan[0].keys()))
        w.writeheader()
        w.writerows(scan)
    if calibrate_only:
        print(f"Calibration written to {CURRENT_OUTPUT_DIR / CALIBRATION_NAME}", flush=True)
        return

    ray_env = PySimEnv(render_mode=None, time_step=SURROGATE_DT, wind=False, current=False,
                       start_pos=np.array([start_x, start_y], dtype=np.float32), ship_model=SHIP_MODEL)
    cutoff_geom = _cutoff_geometry(ray_env)

    # One UAntwerpSurrogate per preset (model_comparison.UA_PRESETS_BY_SHIP); UGent only for Tripoli.
    ua_models = build_ua_models(ship_spec)
    non_fh = {**build_reference_models(ship_spec, ray_env), **ua_models}
    tf_models = None
    if ONESTEP_CHECK:
        tf_models = {**build_reference_models(ship_spec, ray_env), **build_ua_models(ship_spec)}
    run_kwargs = dict(start_x=float(start_x), start_y=float(start_y), ray_env=ray_env, cutoff_geom=cutoff_geom,
                      non_fh_models=non_fh, ua_models=ua_models, tf_models=tf_models, **common)

    results: Dict[Tuple[str, float], RunResult] = {}
    bases: Dict[str, RunResult] = {}
    with open(CURRENT_OUTPUT_DIR / SUMMARY_NAME, "w", newline="") as sf, \
            open(CURRENT_OUTPUT_DIR / ONESTEP_NAME, "w", newline="") as of:
        sw, ow = csv.writer(sf), csv.writer(of)
        sw.writerow(SUMMARY_HEADER)
        ow.writerow(ONESTEP_HEADER)
        for lv in levels:
            lv_results = []
            for rel in rel_dirs_deg:
                rel = float(rel)
                heading = heading_for(lv.toward_deg, rel)
                run_id = f"{lv.name}_rel{rel:g}"
                try:
                    res = run_case(run_id, lv, rel, heading, True, exercise_path=exercise_path, **run_kwargs)
                    base = run_case(run_id + "_nocurrent", lv, rel, heading, False,
                                    exercise_path=exercise_path_nocurrent, n_steps=len(res.paths["FH"]) - 1,
                                    **run_kwargs)
                except CurrentTestSetupError:
                    raise
                except Exception as exc:     # also pythonnet/.NET exceptions (e.g. grounding in a cross-river run)
                    print(f"SKIPPED {run_id}: {type(exc).__name__}: {exc}", flush=True)
                    logger.warning("SKIPPED %s: %s: %s", run_id, type(exc).__name__, exc)
                    continue
                _print_run(res)
                _print_run(base)
                results[(lv.name, rel)] = res
                bases[res.run_id] = base
                lv_results.append(res)
                write_trajectory_csv(CURRENT_OUTPUT_DIR / f"trajectory_{run_id}.csv", res, base)
                write_env_csv(CURRENT_OUTPUT_DIR / f"env_{run_id}.csv", res)
                write_env_csv(CURRENT_OUTPUT_DIR / f"env_{run_id}_nocurrent.csv", base)
                sw.writerows(summary_rows(res, base))
                ow.writerows(onestep_rows(res))
                ow.writerows(onestep_rows(base))
                sf.flush()
                of.flush()
            if lv_results:
                _check_fh_directions(lv_results, bases)
                plot_paths_for_level(lv_results, bases, CURRENT_OUTPUT_DIR / f"current_paths_{lv.name}.png",
                                     f"Paths vs. FH -- current {lv.speed:.2f} m/s (timeStart {lv.time_start} s), "
                                     f"v0={speed:g} m/s, rudder 0, no wind")
    print(f"Wrote {CURRENT_OUTPUT_DIR / SUMMARY_NAME}", flush=True)

    if results:
        top = [lv.name for lv in levels if any(k[0] == lv.name for k in results)][-1]   # strongest level with results
        try:
            d = pd.read_csv(CURRENT_OUTPUT_DIR / SUMMARY_NAME)
            slide_response_per_direction(d, top, CURRENT_OUTPUT_DIR / "slide_current_response_per_direction.png")
            slide_example_paths(results, bases, top, CURRENT_OUTPUT_DIR / "slide_current_example_paths.png")
        except Exception as exc:
            logger.warning("Slide figures failed (%r); the CSVs and the report are not affected.", exc)

    meta = {
        "ship_model": getattr(SHIP_MODEL, "name", str(SHIP_MODEL)),
        "exercise": str(exercise_path), "exercise_nocurrent": str(exercise_path_nocurrent),
        "start_pos": f"({start_x:.1f}, {start_y:.1f}) m; heading per run = current direction + 180 + relative direction",
        "speed_thrust_rudder": f"{speed:g} m/s / {thrust:.3f} (normalized) / 0 deg",
        "path_length": f"until FH has sailed {TARGET_PATH_LENGTH:g} m over ground (max {MAX_STEPS} steps of "
                       f"{SURROGATE_DT:g} s); the no-current run and all models run the same number of steps",
        "grid": "levels " + ", ".join(f"{lv.speed:.2f} m/s (timeStart {lv.time_start} s)" for lv in levels)
                + " x current from " + ", ".join(f"{r:g}" for r in rel_dirs_deg) + " deg rel. bow",
        "ua_inputs": "FH's live UKC; FH's current at FH's position; apparent wind for the model's own motion "
                     "(wind-field scale 0)",
        "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warnings": list(dict.fromkeys(warn.messages)),
    }
    (CURRENT_OUTPUT_DIR / RUN_META_NAME).write_text(json.dumps(meta, indent=2))
    try:
        write_report(CURRENT_OUTPUT_DIR)
    except Exception as exc:
        print(f"Report generation failed ({exc!r}); rebuild with --report-only", flush=True)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_report(CURRENT_OUTPUT_DIR)
    else:
        # SHIP_MODEL's full exercises with and without current
        # (sim_fh_dataset_generator.SHIP_PROFILES).
        from fw.surrogate.sim_fh_dataset_generator import exercise_paths
        _current_exercise, _nocurrent_exercise = exercise_paths(SHIP_MODEL, "full")
        generate_current_trajectories(
            start_x=-1559.0, start_y=11990.0, start_heading_deg=-45.0,
            exercise_path=_current_exercise,
            exercise_path_nocurrent=_nocurrent_exercise,
            calibrate_only="--calibrate-only" in sys.argv,
        )
