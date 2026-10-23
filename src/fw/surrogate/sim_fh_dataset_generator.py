"""
FH dataset generator for the surrogate models -- every ship, every configuration.

Simulates episodes in FH (simexe) and writes the supervised dataset (one row
per time step: state + action + rays, then the one-step deltas and the
episode_id; layout in load_supervised_dataset).

Ships (SHIP_PROFILES, `ship_model=`):
  TRIPOLI    azipods: setDesiredDirection(+-RUDDER deg) + setEngineLeverValue;
             rudder scale 35 deg (fixed).
  MYZAKO_37  classic rudder + propeller: rudder setControlValue(+-1, FH's own
             maximum) and propeller engage() + setEngineLeverValue; rudder
             scale = FH's own maximum (getMaximumAngle), probed once before
             the run (probe_fh_rudder_max_deg) and checked in every episode.
             It must equal the ship's max_rudder_angle in ship_factory.
  The framework rudder command (positive = turn to PORT) is converted to FH's
  sign with fh_tabular_model.fh_rudder_sign, for the initial rudder angle and
  for every step.

Configurations (CONFIGURATIONS, `configuration=`):
  full       ship-bank + UKC + wind + current: random wind-field transform
             per episode (with WIND_WARMUP_TIME), a random <timeStart> (tidal
             offset) per chunk; 37 input columns (scenario full_env).
  confined   ship-bank + UKC only: the nowindnocurrent exercise, no wind
             field; FH reads the static tide (STATIC_TIDE_XML_PATH), the
             tide/current XMLs with <timeStart> are neither read nor written;
             33 input columns (scenario confined_ukc).

"Straight" episodes (STRAIGHT_*) exist for both ships and both
configurations; only the full configuration has a no-current exercise for
STRAIGHT_NO_CURRENT_SHARE of them.
"""

import gc
import os
import re
import sys
import json
import time
import shutil
import hashlib
import tempfile
import logging
import traceback
import numpy as np
import multiprocessing
import signal

import matplotlib

# Non-interactive backend by default. The default (TkAgg on Windows) creates Tk
# objects (images, variables) that Python's GC may later destroy from one of the
# ThreadPoolExecutor threads, which makes Tcl abort the whole process
# ("Tcl_AsyncDelete: async handler deleted by the wrong thread", exit code
# -2147483645). Only plot_xy_points_and_trajectories(show=True) switches to Tk,
# from the main thread while no worker threads are running.
matplotlib.use("Agg")

from pathlib import Path
from matplotlib import pyplot as plt
from multiprocessing.pool import Pool
from dataclasses import dataclass, field
from shapely.geometry import LineString, MultiLineString, Point
from fw.dynamics.fh_tabular_model import fh_rudder_sign
from fw.environments.py_sim_env import PySimEnv
from fw.environments.ray_features import (
    RAY_ANGLES_DEG,
    RAY_LAYOUT_VERSION,
    RAY_MAX_DISTANCE,
    NBR_RAY_FEATURES,
    compute_ray_features,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from fw.ships.ship_factory import ShipModel, create_ship
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

DEFAULT_MATH_MODEL_FREQUENCY = 10.0  # Hz
# Simulated before the first recorded state (whole math-model steps only).
INITIALIZATION_TIME = 1.0 / DEFAULT_MATH_MODEL_FREQUENCY  # s
# FH's WindState lags behind the wind-field transform: right after the
# settle above it still reports the exercise's default transform (measured
# 2026-10-13: 11.174 m/s @ 63.29 deg for every episode), and only about 1 s
# later the episode's own transform. Without this warm-up the first rows of
# every episode carried that default wind. Simulated after the settle, before
# the first recorded state (whole math-model steps only). Full configuration only.
WIND_WARMUP_TIME = 1.0  # s

# Episode stop limits (see stop_reason). Rows recorded before a limit is hit are kept.
MIN_KEEL_CLEARANCE = 1.0    # m
MIN_BANK_DISTANCE = 10.0    # m, nearest bank from the ship center (via the rays)
# Map cut-off edges are not real banks, but the rays may see them as banks.
# Start points closer than this to a cut-off edge are resampled, and an episode
# stops ("near_cutoff") when the ship center gets this close. Equal to the ray
# range on purpose: a cut-off edge can then never appear in the ray features of
# a recorded row. 0 disables.
CUTOFF_CLEARANCE = RAY_MAX_DISTANCE  # m

# Start points: this fraction starts BANK_START_BAND m from the nearest real
# bank, the rest uniformly in the river.
BANK_START_FRACTION = 0.2
BANK_START_BAND = (20.0, 135.0)  # m, from the ship center

# Episode initial conditions and controls (all part of the run signature).
# The rudder scale (action[0] = +-1 <-> +-rudder_max_deg) is per ship: SHIP_PROFILES.
_PROPELLER_RPS_SCALE = 5.5            # action[1] = +-1 -> initial propeller rps
INIT_SURGE_RANGE = (0.0, 6.0)         # m/s
# thrust_action range of the random episodes (uniform). The default keeps the
# earlier [-1, 1]; the RL framework maps its action to [min_thrust, max_thrust]
# (ship_factory: 0.1-1.0) before it reaches FH or a surrogate, so a narrower
# range such as (0, 1) spends no data on negative thrust the surrogate never gets.
THRUST_ACTION_RANGE = (-1.0, 1.0)
INIT_SWAY_RANGE = (-2.0, 2.0)         # m/s
INIT_YAW_RATE_RANGE = (-1.5, 1.5)     # FH units (deg/s)
WIND_SCALE_SIGMA = 7.8                # |N(0, sigma)|: about 80% of the wind scales between 0 and 10
WIND_SCALE_MAX = 25.0                 # clip of the wind-field scale factor

# "Straight" episodes: steady straight sailing in open water (rudder ~0,
# constant forward thrust, near-steady initial speed, heading along the
# longest free stretch of water). The random episodes above almost never
# contain this operating point (measured on the tripoli_wind_current
# dataset: 0.03 % of the rows at rudder ~0 / thrust ~0.83 in open water, none
# at low wind), so a surrogate trained on them has to extrapolate there.
# Taken from the uniform starts; the rest of the episodes are unchanged.
STRAIGHT_FRACTION = 0.25              # of ALL episodes (0 disables)
STRAIGHT_RUDDER_MAX_DEG = 2.0         # rudder uniform in +-this ...
STRAIGHT_RUDDER_ZERO_SHARE = 0.5      # ... except this share, which gets exactly 0
STRAIGHT_THRUST_RANGE = (0.2, 1.0)    # constant thrust action, uniform
# Initial surge = <ship's straight_surge_per_thrust> * thrust * U(STRAIGHT_SURGE_SPREAD),
# clipped to INIT_SURGE_RANGE (straight_surge_per_thrust: see SHIP_PROFILES).
# Only the start is affected: the ship settles to FH's own steady speed during
# the episode.
STRAIGHT_SURGE_SPREAD = (0.8, 1.1)
STRAIGHT_INIT_SWAY_RANGE = (-0.1, 0.1)       # m/s
STRAIGHT_INIT_YAW_RATE_RANGE = (-0.1, 0.1)   # FH units (deg/s)
STRAIGHT_CALM_SHARE = 0.4             # share of straight episodes with wind Scale = 0
                                      # (with current: current-only data for a current test)
STRAIGHT_NO_CURRENT_SHARE = 0.4       # share of straight episodes run with the ship's no-current
                                      # exercise (full configuration only, see SHIP_PROFILES)
STRAIGHT_HEADING_STEP_DEG = 5.0       # candidate headings for the free-stretch search
STRAIGHT_LOOKAHEAD = 600.0            # m; search range for free water ahead (see straight_lookahead)
# Heading of a straight episode: uniformly random among the headings with
# enough free water for THIS episode, i.e. (initial surge + this margin for a
# following current) * horizon * 1.1 -- not the longest stretch. Slow episodes
# can then also cross the river, so the current (which runs mainly along the
# river) is also met from the side, not only from ahead or astern. 0 in the
# confined configuration (no current).
STRAIGHT_CURRENT_MARGIN = 1.5         # m/s

# "Bank-straight" episodes: steady straight sailing ALONGSIDE a real bank
# (rudder ~0, constant forward thrust, near-steady initial speed, heading
# parallel to the nearest real bank). The random near-bank starts above have
# random headings/actions and the straight episodes avoid the banks, so the
# operating point of a ship following a bank (the model_comparison ship-bank
# test) is almost absent from the data (measured on myzako_confined: < 0.1 %
# of the rows at rudder ~0 within 20 m of a bank). Rudder, thrust, initial
# speed/sway/yaw rate and the calm-wind share: as the straight episodes
# (STRAIGHT_*). 0 disables (the default: earlier runs are reproduced exactly).
BANK_STRAIGHT_FRACTION = 0.0          # of ALL episodes
BANK_STRAIGHT_BAND = (11.0, 40.0)     # m, from the ship center to the nearest real bank
BANK_STRAIGHT_HEADING_JITTER_DEG = 3.0  # heading = bank tangent + U(-this, +this)

# Actuator settling before the first recorded row: the command is applied
# (unrecorded steps of dt) until every steering actuator is at its commanded
# angle, at most ACTUATOR_SETTLE_MAX_SECONDS (else the episode FAILS). Needed
# for the azipods: FH ignores setInitialRudderAngle for them, they start at 0
# deg and turn at about 36 deg/s (probe_actuator_lag), so without it the first
# up to ~1 s of an episode had a thrust direction the recorded action did not
# describe. A rudder that starts at its commanded angle needs 0 steps.
ACTUATOR_SETTLE_MAX_SECONDS = 3.0
ACTUATOR_SETTLE_TOL_DEG = 0.1

TIMEOUT_SECONDS = 60.0

# NOTE: change the constants in this file by editing them here. Assigning them
# at runtime (module.CONSTANT = ...) does not reach the simulation workers:
# on Windows each pool worker re-imports this module with the file's values.

CHECKPOINT_EVERY = 1000   # episodes per checkpoint shard
_CHECKPOINT_VERSION = 2   # bump when the data or the run signature changes
                          # (2: actuator settling, no initial reverse rps without reverse thrust)

CACHE_CLEAR_INTERVAL = 100        # simulations between C:\cache cleanups
# Crash dumps of failed simulator processes. ALL contents are deleted every
# CACHE_CLEAR_INTERVAL simulations: this folder must be used by FH/simexe only.
CACHE_FOLDER = Path(r"C:\cache")

try:
    simexe_path = os.environ['SIMEXE_REPO_FOLDER']
except KeyError:
    simexe_path = ""

# All simexe inputs are derived from SIMEXE_REPO_FOLDER, so the files written
# here are always the ones simexe reads.
AREA_INPUT_DIR = Path(simexe_path) / "database" / "areas" / "ScheldeSaeftinge_23_002" / "invoer"
EXERCISE_DIR = AREA_INPUT_DIR / "tra"

# Current and tide XMLs read by every simexe process. Both get the same random
# <timeStart> (tidal offset) once per chunk in the full configuration; see
# randomize_current_time_start.
CURRENT_XML_PATH = AREA_INPUT_DIR / "sne" / "Duplex_005_-01h00HWdyn_HwPros20190324_1704_3Dt.current.v30.xml"
TIDE_XML_PATH = AREA_INPUT_DIR / "tide" / "Duplex_005_-01h00HWdyn_HwPros20190324_1704_2Dt.tide.v30.xml"
TIME_START_XML_PATHS = (CURRENT_XML_PATH, TIDE_XML_PATH)
# Static tide read by the exercises WITHOUT current (*nowindnocurrent*, used by the
# confined configuration, and *windnocurrent*, used for the straight_nocur
# episodes), as their math_model.log shows; they do not read the two XMLs above.
# Never written by this module; fingerprinted in the run signature when used.
STATIC_TIDE_XML_PATH = AREA_INPUT_DIR / "tide" / "Duplex_005_-01h00HWstat_HwPros20190324_1704_2Dt.tide.v30.xml"
CURRENT_TIME_START_RANGE = (-21840, 24960)  # s; about -6.1 h ... +6.9 h
# Rays: see fw.environments.ray_features (shared with the surrogate);
# ray_0 = ahead, clockwise, origin = ship center.
# Row width: see nbr_x_features (37 full, 33 confined).
NBR_Y_FEATURES = 3                           # delta_surge, delta_sway, delta_yaw_rate

# FH heading convention: "math" (0 = East, counter-clockwise; verified for the
# Tripoli with check_fh_conventions.py) or "compass" (0 = North, clockwise).
FH_HEADING_CONVENTION = "math"



# ----------------------------------------------------------------------
# Ships and configurations
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ShipProfile:
    """
    Ship-specific settings of the generator.

    Attributes:
        rudder_max_deg: rudder_action = +-1 <-> +-this many degrees (initial
            rudder angle; azipod direction); stored as the dataset's rudder
            scale. None = FH's own rudder maximum (getMaximumAngle), read from
            FH (see resolve_rudder_max_deg). Either way it must equal the ship's
            max_rudder_angle in fw.ships.ship_factory.
        steering: "azipod" (getAzipodControls: setDesiredDirection in degrees
            and setEngineLeverValue) or "rudder" (getRudderControls:
            setControlValue normalized to FH's own maximum; getPropellerControls:
            engage() and setEngineLeverValue).
        exercises: configuration -> (exercise, same exercise without current
            or None), file names in EXERCISE_DIR.
        straight_surge_per_thrust: Initial surge of a straight episode per
            unit thrust action [m/s] (see STRAIGHT_SURGE_SPREAD).
        reverse_thrust: Whether FH turns the propeller(s) backwards for a
            negative engine lever. False = FH runs them down to 0 rps instead
            (measured for the Tripoli azipods with probe_actuator_lag: from the
            initial -rps to 0 at about 1 rps/s, from any speed), so a negative
            thrust_action means "no thrust" and the initial rps is set to 0.
    """

    rudder_max_deg: Optional[float]
    steering: str
    exercises: Dict[str, Tuple[str, Optional[str]]]
    straight_surge_per_thrust: float
    reverse_thrust: bool


SHIP_PROFILES: Dict[ShipModel, ShipProfile] = {
    ShipModel.TRIPOLI: ShipProfile(
        rudder_max_deg=35.0,
        steering="azipod",
        exercises={
            # The no-current exercise has tide / water levels still on; the
            # wind test model_comparison_wind uses it too.
            "full": ("DDShip_Tripoli_scenario_windcurrent.tab", "DDShip_Tripoli_scenario_windnocurrent.tab"),
            "confined": ("DDShip_Tripoli_scenario_nowindnocurrent.tab", None),
        },
        # The approximation model_comparison uses (thrust = speed / 6; FH holds
        # ~5.0 m/s at thrust 0.83).
        straight_surge_per_thrust=6.0,
        reverse_thrust=False,   # FH azipods: negative lever -> 0 rps (probe_actuator_lag)
    ),
    ShipModel.MYZAKO_37: ShipProfile(
        # setControlValue(+-1) moves the rudder to +-FH's own maximum, so that is
        # the scale (ship_factory: 75 deg; a difference is refused, see
        # check_ship_against_factory).
        rudder_max_deg=None,
        steering="rudder",
        exercises={
            "full": ("DDShip_scenario3_windcurrent_myzako_37.tab", "DDShip_scenario2_windnocurrent_myzako_37.tab"),
            "confined": ("DDShip_scenario1_nowindnocurrent_myzako_37.tab", None),
        },
        straight_surge_per_thrust=6.0,   # Tripoli value, not measured for the Myzako
        reverse_thrust=True,    # propeller runs backwards (probe_actuator_lag: -4.47 rps at lever -0.8)
    ),
}


@dataclass(frozen=True)
class ConfigurationProfile:
    """
    Settings of one environment configuration.

    Attributes:
        wind_current: Random wind-field transform per episode (and
            WIND_WARMUP_TIME); wind and current are stored in every row (a
            state of 11 values instead of 7).
        randomize_time_start: A new random <timeStart> in TIME_START_XML_PATHS
            per chunk (the XMLs must exist); otherwise they are not touched.
    """

    wind_current: bool
    randomize_time_start: bool


CONFIGURATIONS: Dict[str, ConfigurationProfile] = {
    "full": ConfigurationProfile(wind_current=True, randomize_time_start=True),       # ship-bank + UKC + wind + current
    "confined": ConfigurationProfile(wind_current=False, randomize_time_start=False),  # ship-bank + UKC
}
DEFAULT_SHIP_MODEL = ShipModel.TRIPOLI
DEFAULT_CONFIGURATION = "full"


def _as_ship_model(ship_model: Union[ShipModel, str]) -> ShipModel:
    """ShipModel from a ShipModel or its name; ValueError for a ship without a profile."""

    if isinstance(ship_model, str):
        try:
            ship_model = ShipModel[ship_model]
        except KeyError:
            raise ValueError(f"Unknown ship model {ship_model!r}.") from None
    if ship_model not in SHIP_PROFILES:
        raise ValueError(f"No generator profile for {ship_model!r}; supported: "
                         f"{', '.join(m.name for m in SHIP_PROFILES)} (add one to SHIP_PROFILES).")
    return ship_model


def ship_profile(ship_model: Union[ShipModel, str]) -> ShipProfile:
    """Generator settings of `ship_model` (ShipModel or name)."""

    return SHIP_PROFILES[_as_ship_model(ship_model)]


def configuration_profile(configuration: str) -> ConfigurationProfile:
    """Settings of `configuration` ("full" / "confined"); ValueError for an unknown name."""

    try:
        return CONFIGURATIONS[configuration]
    except KeyError:
        raise ValueError(f"Unknown configuration {configuration!r}; choose from "
                         f"{sorted(CONFIGURATIONS)}.") from None


def ship_rudder_max_deg(ship_model: Union[ShipModel, str]) -> Optional[float]:
    """
    Fixed rudder scale of `ship_model` (rudder_action = +-1 <-> +-this many
    degrees), or None if it is FH's own maximum (see resolve_rudder_max_deg).
    """

    value = ship_profile(ship_model).rudder_max_deg
    return None if value is None else float(value)


def nbr_x_features(configuration: str) -> int:
    """Input columns of a row: state (11 full, 7 confined) + action (2) + rays."""

    n_state = 11 if configuration_profile(configuration).wind_current else 7
    return n_state + 2 + NBR_RAY_FEATURES


def exercise_paths(ship_model: Union[ShipModel, str], configuration: str) -> Tuple[Path, Optional[Path]]:
    """(exercise, no-current exercise or None) of a ship and configuration, in EXERCISE_DIR."""

    configuration_profile(configuration)  # validates the name
    main, no_current = ship_profile(ship_model).exercises[configuration]
    return EXERCISE_DIR / main, (EXERCISE_DIR / no_current if no_current else None)


def straight_current_margin(configuration: str) -> float:
    """STRAIGHT_CURRENT_MARGIN with current (full configuration), 0 without."""

    return STRAIGHT_CURRENT_MARGIN if configuration_profile(configuration).wind_current else 0.0


def check_ship_against_factory(ship_model: Union[ShipModel, str],
                               rudder_max_deg: Optional[float] = None) -> None:
    """
    Raise ValueError if the rudder scale differs from the ship's
    max_rudder_angle in fw.ships.ship_factory (the dataset's rudder scale
    would then not match the ship the surrogate is used with).

    Args:
        ship_model: Ship (ShipModel or name).
        rudder_max_deg: The scale to check; None = the profile's fixed value
            (ValueError for a ship whose scale comes from FH: resolve it first
            with resolve_rudder_max_deg).
    """

    ship_model = _as_ship_model(ship_model)
    value = rudder_max_deg if rudder_max_deg is not None else ship_rudder_max_deg(ship_model)
    if value is None:
        raise ValueError(f"{ship_model.name}: the rudder scale is FH's getMaximumAngle(); "
                         f"resolve it first (resolve_rudder_max_deg).")
    factory_deg = float(np.degrees(create_ship(ship_model).specifications.max_rudder_angle))
    if not np.isclose(factory_deg, float(value)):
        source = ("FH's getMaximumAngle()" if ship_rudder_max_deg(ship_model) is None
                  else f"SHIP_PROFILES[{ship_model.name}].rudder_max_deg")
        raise ValueError(
            f"{ship_model.name}: rudder scale {float(value):g} deg ({source}), but ship_factory gives "
            f"{factory_deg:g} deg. The dataset would not match the framework's ship: align "
            f"fw.ships.ship_factory (max_rudder_angle of this ship) with {float(value):g} deg first.")


SIM_AVAILABLE = False
Config = Exercise = MathModel = SimException = None

try:
    from sim import Config, Exercise, MathModel, SimException

    SIM_AVAILABLE = True
except ImportError:
    pass  # imported in run_simulation after extending sys.path


def clear_cache_folder(cache_path: Path = CACHE_FOLDER) -> None:
    """
    Empty *cache_path* (keep the folder itself), so crash dumps of failed
    simulator processes don't pile up. Locked items are skipped with a warning.
    """

    if not cache_path.exists():
        return

    deleted, skipped = 0, 0
    for item in cache_path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            deleted += 1
        except (PermissionError, OSError) as exc:
            print(f"Warning: Could not remove cache item {item}: {exc}")
            skipped += 1

    print(f"Cache folder '{cache_path}' cleared - {deleted} item(s) removed"
          + (f", {skipped} skipped (locked)" if skipped else "."))


_TIME_START_PATTERN = re.compile(rb"(<timeStart>)(.*?)(</timeStart>)", re.DOTALL)


def randomize_current_time_start(
        xml_paths: Union[Path, str, Iterable[Union[Path, str]]] = TIME_START_XML_PATHS,
        time_range: Tuple[int, int] = CURRENT_TIME_START_RANGE,
        rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Write one new random <timeStart> (tidal offset) into all *xml_paths*, so
    the current field and the water levels match.

    Call once per chunk, only while no simulation is running: simexe reads
    these shared files itself. Only the value between the tags is replaced
    (bytes otherwise untouched), all files are validated before any is
    written, and each file is replaced atomically (temp file + os.replace).
    The files are not replaced as one transaction: if the second replacement
    fails, the files differ until the next successful write, but the exception
    aborts the run before another simulation is started.

    Args:
        xml_paths: One path or several; default TIME_START_XML_PATHS.
        time_range: Inclusive (low, high) range in seconds.
        rng: Optional numpy Generator (for reproducibility).

    Returns:
        The value written.

    Raises:
        FileNotFoundError: A path does not exist.
        ValueError: No paths, or a file without <timeStart>.
    """

    if isinstance(xml_paths, (str, Path)):
        xml_paths = (xml_paths,)
    paths = [Path(p) for p in xml_paths]
    if not paths:
        raise ValueError("xml_paths is empty: nothing to randomize.")

    rng = rng or np.random.default_rng()
    low, high = time_range
    new_time_start = int(rng.integers(low, high, endpoint=True))
    replacement = str(new_time_start).encode("ascii")

    # Validate all files first (binary mode: line endings stay untouched).
    updated = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Environment-input XML not found: {path}. This file is read "
                "directly by simexe (not passed as a parameter), so it must "
                "exist at this fixed path before any simulation is started."
            )
        new_raw, n_subs = _TIME_START_PATTERN.subn(
            lambda m: m.group(1) + replacement + m.group(3),
            path.read_bytes(), count=1,
        )
        if n_subs == 0:
            raise ValueError(f'"<timeStart>...</timeStart>" not found in {path}')
        updated.append(new_raw)

    for path, new_raw in zip(paths, updated):
        tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
        try:
            tmp_path.write_bytes(new_raw)
            # Retry briefly: on Windows a transient lock (antivirus, indexer) makes os.replace fail.
            for attempt in range(10):
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.5)
        except BaseException:
            tmp_path.unlink(missing_ok=True)  # never leave a stray temp file
            raise

    return new_time_start


def _file_fingerprint(path: Union[Path, str]) -> str:
    """
    Short SHA-256 of a simulator input file, with the <timeStart> value masked
    (it is rewritten by this module), for the run signature.
    """

    raw = _TIME_START_PATTERN.sub(rb"\1*\3", Path(path).read_bytes())
    return hashlib.sha256(raw).hexdigest()[:16]


def _describe_cutoffs(env: "PySimEnv") -> Tuple[Optional[MultiLineString], int, float, Optional[str]]:
    """
    Map cut-off edges (not real banks) from PySimEnv.bank_segments().

    Returns:
        (geometry, n_segments, total_length_m, problem): `problem` is None when
        at least one cut-off segment was found, otherwise a description of why
        there is no geometry (method missing, method raised, nothing found).
    """

    if not hasattr(env, "bank_segments"):
        return None, 0, 0.0, "PySimEnv has no bank_segments() method"
    try:
        _, cutoff = env.bank_segments()
    except Exception as exc:
        return None, 0, 0.0, f"PySimEnv.bank_segments() raised {type(exc).__name__}: {exc}"
    lines = [LineString(np.asarray(seg, dtype=np.float64)) for seg in cutoff if len(seg) >= 2]
    if not lines:
        return None, 0, 0.0, (f"PySimEnv.bank_segments() classified no boundary segment as a map "
                              f"cut-off (polygon bounds {tuple(round(b, 2) for b in env.polygon_shape.bounds)})")
    geom = MultiLineString(lines)
    return geom, len(lines), float(geom.length), None


def _cutoff_geometry(env: "PySimEnv") -> Optional[MultiLineString]:
    """Map cut-off edges as one geometry, or None if there are none (see `_describe_cutoffs`)."""

    return _describe_cutoffs(env)[0]


_WORKER_CUTOFF = []  # per process: [geometry] once computed (see _worker_cutoff_geometry)


def _worker_cutoff_geometry(env: "PySimEnv") -> Optional[MultiLineString]:
    """_cutoff_geometry, computed once per worker process instead of once per episode."""

    if CUTOFF_CLEARANCE <= 0:
        return None
    if not _WORKER_CUTOFF:
        _WORKER_CUTOFF.append(_cutoff_geometry(env))
    return _WORKER_CUTOFF[0]


def _too_close_to_cutoff(cutoff: Optional[MultiLineString], point, clearance: Optional[float] = None) -> bool:
    """True if *point* lies within *clearance* (default CUTOFF_CLEARANCE) of a map cut-off edge."""

    if clearance is None:
        clearance = CUTOFF_CLEARANCE
    if cutoff is None or clearance <= 0:
        return False
    return cutoff.distance(Point(float(point[0]), float(point[1]))) < clearance


def _sample_points(sampler: Callable[[int], np.ndarray], n: int,
                   cutoff: Optional[MultiLineString], clearance: Optional[float] = None,
                   max_rounds: int = 20) -> np.ndarray:
    """
    Up to *n* points from *sampler(k)*, resampling the ones within *clearance*
    (default CUTOFF_CLEARANCE) of a cut-off edge, for at most *max_rounds*
    rounds. Returns an (m, 2) float32 array, m <= n.
    """

    points = np.zeros((0, 2), dtype=np.float32)
    for _ in range(max_rounds):
        need = n - len(points)
        if need <= 0:
            break
        candidates = np.asarray(sampler(need), dtype=np.float32).reshape(-1, 2)
        if len(candidates) == 0:
            break
        keep = np.array([not _too_close_to_cutoff(cutoff, p, clearance) for p in candidates], dtype=bool)
        points = np.vstack([points, candidates[keep]])
    return points[:n]


@dataclass
class SimConfig:
    """Simulator paths and timeout."""

    lib_path: str = ""
    config_path: str = ""
    timeout_seconds: float = TIMEOUT_SECONDS


@dataclass
class SimOutput:
    """Result of one simulation."""

    status: str = ""
    time_seconds: float = 0.0
    x_input: Optional[np.ndarray] = None
    y_output: Optional[np.ndarray] = None
    trajectory: Optional[np.ndarray] = None
    info: Optional[dict] = field(default_factory=dict)


@dataclass
class SimSetup:
    """Parameters of one simulation (episode)."""

    exercise_path: str = ""
    output_path: str = ""
    x_pos: float = 0.0
    y_pos: float = 0.0
    dt: float = 0.1
    horizon_seconds: float = 3.0
    seed: Optional[int] = None  # per-episode RNG seed; None = non-reproducible
    episode_kind: str = "random"  # "random" (bank/uniform starts), "straight" (see STRAIGHT_*;
                                  # start types "straight" and "straight_nocur") or
                                  # "bank_straight" (see BANK_STRAIGHT_*)
    ship_model: str = DEFAULT_SHIP_MODEL.name     # key of SHIP_PROFILES, by name
    rudder_max_deg: float = 0.0                   # rudder scale of the run (resolve_rudder_max_deg)
    thrust_range: Tuple[float, float] = THRUST_ACTION_RANGE   # thrust_action range of random episodes
    configuration: str = DEFAULT_CONFIGURATION   # key of CONFIGURATIONS
    output: SimOutput = field(default_factory=SimOutput)


def plot_start_points(env: "PySimEnv", xy_uniform: np.ndarray, xy_bank: np.ndarray, out_png: Path,
                      xy_bank_straight: Optional[np.ndarray] = None) -> None:
    """
    Save a plot of the start points, with real banks and map cut-off edges in
    different colors (to check PySimEnv.bank_segments). Never raises.
    """

    try:
        fig, ax = plt.subplots(figsize=(12, 9))
        if hasattr(env, "bank_segments"):
            bank, cutoff = env.bank_segments()
            for segs, color, label in ((bank, "tab:blue", "real bank"),
                                       (cutoff, "tab:red", "map cut-off (not a bank)")):
                for k, seg in enumerate(segs):
                    ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.2 if color == "tab:blue" else 2.5,
                            label=label if k == 0 else None)
        else:
            x_poly, y_poly = env.polygon_shape.exterior.xy
            ax.plot(x_poly, y_poly, "k-", linewidth=1.0, label="river polygon")
        if len(xy_uniform):
            ax.scatter(xy_uniform[:, 0], xy_uniform[:, 1], s=8, c="0.45", label="uniform start")
        if len(xy_bank):
            ax.scatter(xy_bank[:, 0], xy_bank[:, 1], s=8, c="tab:orange", label="near-bank start")
        if xy_bank_straight is not None and len(xy_bank_straight):
            ax.scatter(xy_bank_straight[:, 0], xy_bank_straight[:, 1], s=8, c="tab:green",
                       label="bank-straight start")
        ax.set_aspect("equal", "box")
        ax.set_title("Start points (check that red edges are map cut-offs, not real banks)")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        fig.savefig(out_png, bbox_inches="tight", dpi=200)
        print(f"Start-point plot saved as '{out_png}'.")
    except Exception as exc:  # plotting must never break data generation
        print(f"Warning: could not create start-point plot: {exc}")
    finally:
        plt.close("all")


def plot_xy_points_and_trajectories(env: PySimEnv, xy: np.ndarray, trajectories: list,
                                    out_png: Union[str, Path] = "generated_points.png",
                                    show: bool = True) -> None:
    """
    Plot the river polygon with start points (N, 2) and trajectories; saves
    *out_png* and, if *show*, opens a window (blocks until it is closed).
    """

    try:
        if show:
            plt.switch_backend("TkAgg")  # main thread, no worker threads alive here
        fig, ax = plt.subplots(figsize=(10, 10))

        river_poly = env.polygon_shape
        x_poly, y_poly = river_poly.exterior.xy
        ax.plot(x_poly, y_poly, "k-", linewidth=1.0)
        ax.fill(x_poly, y_poly, color="lightblue", alpha=0.30)

        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=20,
            c="darkred",
            alpha=0.9,
            edgecolors="white",
            linewidth=0.4,
            label="Start positions"
        )

        for traj in trajectories:
            ax.plot(
                traj[:, 0], traj[:, 1],
                linestyle="-",
                linewidth=1.0,
                alpha=0.35,
                color="green",
            )

        if hasattr(env, "checkpoints"):
            for cp in env.checkpoints:
                if "pos" in cp:
                    ax.plot(cp["pos"][0], cp["pos"][1], "go", markersize=4)

        ax.set_aspect("equal", "box")
        ax.set_title("Sampled Points + Trajectories")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend()

        plt.savefig(out_png, bbox_inches='tight', dpi=300)
        print(f"\nPath visualization saved as '{out_png}'.")

        if show:
            plt.show()
    except Exception as exc:  # plotting must never break data generation
        print(f"Warning: could not create trajectory plot: {exc}")
    finally:
        plt.close('all')
        if show:
            # Destroy the Tk objects here, in the main thread, then go back to Agg.
            gc.collect()
            plt.switch_backend("Agg")


def load_supervised_dataset(file_name: str, plot: bool = True,
                            configuration: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load a dataset CSV written by collect_supervised_dataset.

    CSV column layout of the full configuration (41 columns). The confined
    configuration has no cols 7-10 (wind/current): 37 columns, X = 33, with
    rudder_action at col 7, the rays at cols 9-32, Y at cols 33-35 and
    episode_id at col 36.
        X – inputs (37 columns)
          col  0: x                (m, world-frame position)
          col  1: y                (m, world-frame position)
          col  2: heading          (rad, [-pi, pi), PySimEnv convention: 0 = East, CCW)
          col  3: surge            (m/s, body frame, over ground, forward positive)
          col  4: sway             (m/s, body frame, over ground, PORT positive -- FH's own
                                    convention and the framework's, see fw.ships.vessel_state)
          col  5: yaw_rate         (rad/s)
          col  6: UKC              (m)
          col  7: wind speed       (m/s)
          col  8: wind direction   (rad, FH WindState.Direction converted with np.radians)
          col  9: current speed    (m/s)
          col 10: current direction(rad, 0 = East, CCW, direction the current flows toward)
          col 11: rudder_action    (normalized [-1, 1])
          col 12: thrust_action    (normalized [-1, 1])
          col 13: ray_0            (proximity straight ahead, 0=far, 1=close)
          col 14: ray_15           (proximity 15° clockwise from the bow)
               …                   (ray_90 = starboard, ray_270 = port)
          col 36: ray_345          (proximity 15° to port of the bow)
          See fw.environments.ray_features for the exact definition.

        Y – one-step delta outputs (3 columns)
          col 37: delta_surge      (m/s)
          col 38: delta_sway       (m/s)
          col 39: delta_yaw_rate   (rad/s)

        episode_id – grouping key (1 column)
          col 40: episode_id       (int, stored as float; identifies which
                                     simulation rollout a row came from.
                                     Use this for episode-level, not
                                     row-level, train/val/test splitting.
                                     Ids of failed episodes are missing.)

    Args:
        file_name: CSV path.
        plot: Also show a plot of the positions.
        configuration: "full" or "confined"; None = whichever matches the
            column count.

    Returns:
        (x_inputs, y_outputs, episode_ids, number_of_rows)
    """

    data = np.loadtxt(file_name, delimiter=",", ndmin=2)

    if configuration is not None:
        configuration_profile(configuration)   # ValueError for an unknown name
    widths = {name: nbr_x_features(name) + NBR_Y_FEATURES + 1 for name in CONFIGURATIONS}
    if configuration is None:
        matching = [name for name, w in widths.items() if data.ndim == 2 and data.shape[1] == w]
        configuration = matching[0] if matching else DEFAULT_CONFIGURATION
    expected_cols = widths[configuration]
    if data.ndim != 2 or data.shape[1] != expected_cols:
        raise ValueError(
            f"Expected CSV with {expected_cols} columns ({configuration!r} configuration; "
            f"known widths {widths}), got shape {data.shape}. "
            "Re-generate the dataset or check the file path."
        )

    x_end = nbr_x_features(configuration)
    y_end = x_end + NBR_Y_FEATURES
    x_values = data[:, 0:x_end].astype(np.float32)
    y_values = data[:, x_end:y_end].astype(np.float32)
    episode_ids = np.rint(data[:, y_end]).astype(np.int64)
    counter = x_values.shape[0]

    if plot:
        start_xy = x_values[:, 0:2].astype(np.float64)
        env = PySimEnv(render_mode=None, time_step=0.1, wind=False, current=False)
        plot_xy_points_and_trajectories(env, start_xy, [])

    return x_values, y_values, episode_ids, counter


def fh_heading_to_env_rad(fh_heading_deg: float) -> float:
    """FH heading (degrees, FH_HEADING_CONVENTION) -> radians, 0 = East, counter-clockwise, [-pi, pi)."""

    if FH_HEADING_CONVENTION == "math":
        heading_deg = fh_heading_deg
    elif FH_HEADING_CONVENTION == "compass":
        heading_deg = 90.0 - fh_heading_deg
    else:
        raise ValueError(f"Unknown FH_HEADING_CONVENTION {FH_HEADING_CONVENTION!r}")
    return float(PySimEnv.normalize_angle(np.radians(heading_deg)))


def env_rad_to_fh_heading_deg(heading_rad: float) -> float:
    """Inverse of fh_heading_to_env_rad: radians (0 = East, CCW) -> FH degrees in [0, 360)."""

    deg = float(np.degrees(heading_rad))
    if FH_HEADING_CONVENTION == "math":
        fh = deg
    elif FH_HEADING_CONVENTION == "compass":
        fh = 90.0 - deg
    else:
        raise ValueError(f"Unknown FH_HEADING_CONVENTION {FH_HEADING_CONVENTION!r}")
    return fh % 360.0


_WORKER_STRAIGHT_ZONE = []  # per process: [stop zone] once computed (see _straight_stop_zone)


def _straight_stop_zone(env: "PySimEnv", cutoff: Optional[MultiLineString]):
    """
    Area where stop_reason would end an episode for geometric reasons: within
    MIN_BANK_DISTANCE of the river boundary (banks, islands; the "bank" stop
    fires when ANY ray sees a bank that close) or within CUTOFF_CLEARANCE of a
    map cut-off edge ("near_cutoff"). Computed once per worker process.
    """

    if not _WORKER_STRAIGHT_ZONE:
        zone = env.polygon_shape.boundary.buffer(MIN_BANK_DISTANCE)
        if cutoff is not None and CUTOFF_CLEARANCE > 0:
            zone = zone.union(cutoff.buffer(CUTOFF_CLEARANCE))
        _WORKER_STRAIGHT_ZONE.append(zone)
    return _WORKER_STRAIGHT_ZONE[0]


def _free_distance(zone, start: np.ndarray, heading_rad: float, length: float) -> float:
    """
    Distance from `start` along `heading_rad` (0 = East, CCW) to the first
    point inside the stop zone (see _straight_stop_zone), capped at `length`;
    0 if `start` itself is inside it.
    """

    end = start + length * np.array([np.cos(heading_rad), np.sin(heading_rad)])
    line = LineString([tuple(start), tuple(end)])
    hit = line.intersection(zone)
    if hit.is_empty:
        return float(length)
    return float(min(length, Point(tuple(start)).distance(hit)))


def _straight_heading(env: "PySimEnv", cutoff: Optional[MultiLineString], start: np.ndarray,
                      lookahead: float, needed: float, rng: np.random.Generator) -> Tuple[float, float]:
    """
    Heading (radians, 0 = East, CCW) with enough free water from `start`. The
    candidate headings (every STRAIGHT_HEADING_STEP_DEG, shifted by one random
    offset within the step) are evaluated as they will be sailed, up to
    `lookahead` m; a uniformly random one among those with at least
    min(needed, lookahead) free distance is returned (or, if none has that,
    one with at least 0.9 * best), with its free distance.
    """

    offset = rng.uniform(0.0, STRAIGHT_HEADING_STEP_DEG)
    cands = np.radians(np.arange(0.0, 360.0, STRAIGHT_HEADING_STEP_DEG) + offset)
    # Clip the (large) zone to this start's reach once; same results, ~50x faster.
    zone = _straight_stop_zone(env, cutoff).intersection(Point(tuple(start)).buffer(lookahead + 1.0))
    free = np.array([_free_distance(zone, start, h, lookahead) for h in cands])
    # Enough free water if any candidate has it; otherwise the near-best ones.
    target = min(needed, lookahead)
    best = float(free.max())
    ok = np.flatnonzero(free >= (target if best >= target else 0.9 * best))
    k = int(rng.choice(ok))
    return float(PySimEnv.normalize_angle(cands[k])), float(free[k])


_WORKER_REAL_BANK = []  # per process: [real-bank geometry or None] (see _worker_real_bank)


def _worker_real_bank(env: "PySimEnv") -> Optional[MultiLineString]:
    """Real banks (PySimEnv.bank_segments, without map cut-offs) as one geometry, once per process."""

    if not _WORKER_REAL_BANK:
        geom = None
        if hasattr(env, "bank_segments"):
            bank, _ = env.bank_segments()
            lines = [LineString(np.asarray(seg, dtype=np.float64)) for seg in bank if len(seg) >= 2]
            geom = MultiLineString(lines) if lines else None
        _WORKER_REAL_BANK.append(geom)
    return _WORKER_REAL_BANK[0]


def _bank_straight_heading(env: "PySimEnv", cutoff: Optional[MultiLineString], start: np.ndarray,
                           lookahead: float, needed: float,
                           rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Heading (radians, 0 = East, CCW) parallel to the nearest real bank, plus
    a jitter of U(-BANK_STRAIGHT_HEADING_JITTER_DEG, +...). Of the two
    directions along the bank, a random one among those with at least
    min(needed, lookahead) free water (see _straight_heading), else the one
    with the most.

    Returns:
        (heading_rad, free_m, bank_distance_m)

    Raises:
        RuntimeError: If PySimEnv provides no real-bank geometry.
    """

    bank = _worker_real_bank(env)
    if bank is None:
        raise RuntimeError("bank_straight episode: PySimEnv.bank_segments() gives no real-bank segments.")
    p = Point(float(start[0]), float(start[1]))
    nearest = bank.interpolate(bank.project(p))
    normal = np.array([p.x - nearest.x, p.y - nearest.y], dtype=np.float64)
    bank_distance = float(np.hypot(*normal))
    if bank_distance < 1e-6:
        raise RuntimeError(f"bank_straight episode: start {tuple(start)} lies on a bank.")
    tangent_rad = float(np.arctan2(normal[0], -normal[1]))   # normal rotated by +90 deg
    jitter = np.radians(rng.uniform(-BANK_STRAIGHT_HEADING_JITTER_DEG, BANK_STRAIGHT_HEADING_JITTER_DEG))
    cands = np.array([tangent_rad, tangent_rad + np.pi]) + jitter
    zone = _straight_stop_zone(env, cutoff).intersection(p.buffer(lookahead + 1.0))
    free = np.array([_free_distance(zone, start, h, lookahead) for h in cands])
    target = min(needed, lookahead)
    ok = np.flatnonzero(free >= target)
    k = int(rng.choice(ok)) if len(ok) else int(np.argmax(free))
    return float(PySimEnv.normalize_angle(cands[k])), float(free[k]), bank_distance


def straight_lookahead(horizon_seconds: float, current_margin: float = STRAIGHT_CURRENT_MARGIN) -> float:
    """
    Free water wanted ahead of a straight episode: STRAIGHT_LOOKAHEAD, or more
    for long horizons (`current_margin`: see straight_current_margin).
    """

    return max(STRAIGHT_LOOKAHEAD,
               1.1 * (INIT_SURGE_RANGE[1] + current_margin) * float(horizon_seconds))


def stop_reason(env: "PySimEnv", ship_pos: np.ndarray, ukc: float, ray_features: np.ndarray,
                cutoff: Optional[MultiLineString] = None) -> Optional[str]:
    """
    Check a state against the stop limits. Returns None, or:
      "left_open_water" - ship center outside PySimEnv's open water;
      "near_cutoff"     - ship center within CUTOFF_CLEARANCE of a map cut-off edge;
      "bank"            - nearest ray hit < MIN_BANK_DISTANCE (a safety buffer
                          measured from the ship center, not hull contact);
      "ukc"             - keel clearance < MIN_KEEL_CLEARANCE.
    """

    if not env.ship_in_open_water(ship_pos):
        return "left_open_water"
    if _too_close_to_cutoff(cutoff, ship_pos):
        return "near_cutoff"
    nearest_bank = RAY_MAX_DISTANCE * (1.0 - float(np.max(ray_features)))
    if nearest_bank < MIN_BANK_DISTANCE:
        return "bank"
    if ukc < MIN_KEEL_CLEARANCE:
        return "ukc"
    return None


def get_current_velocity(_ship_interface):
    """
    Current speed (m/s) and direction (deg, 0 = East, counter-clockwise, flowing
    toward), from FH's body-frame velocities over ground and through water.

    FH's body frame is x forward, y PORT positive (consistent with its
    counter-clockwise heading), so the raw components are rotated directly.
    The dataset stores this sway unchanged (port positive, like the framework).
    """

    psi = fh_heading_to_env_rad(_ship_interface.getShipHeading())

    r = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi),  np.cos(psi)]
    ])

    vog = _ship_interface.getShipVelocityOverGround()
    vtw = _ship_interface.getShipVelocityThroughWater()

    vog_earth = r @ np.array([vog.x, vog.y])
    vtw_earth = r @ np.array([vtw.x, vtw.y])

    current = vog_earth - vtw_earth

    current_speed = np.linalg.norm(current)
    current_direction_deg = np.degrees(np.arctan2(current[1], current[0]))
    if current_direction_deg < 0:
        current_direction_deg += 360

    return current_speed, current_direction_deg


def _probe_rudder_max_worker(sim_config: SimConfig, exercise_path: str, output_path: str,
                             x: float, y: float) -> float:
    """
    In a worker process: initialize FH on `exercise_path` at (x, y) and
    return the rudder controls' getMaximumAngle() [deg].

    Raises:
        RuntimeError: Initialization failed, the ship has no rudder controls,
            or its rudders report different maxima.
    """

    t0 = time.monotonic()

    def step(text: str) -> None:
        """Progress line of the probe (flushed: the worker's output is the only trace)."""
        print(f"[rudder probe {time.monotonic() - t0:6.1f} s] {text}", flush=True)

    step("worker started, loading the simulator library")
    if not SIM_AVAILABLE and sim_config.lib_path not in sys.path:
        sys.path.append(sim_config.lib_path)
    from sim import Config, Exercise, MathModel

    Path(output_path).mkdir(parents=True, exist_ok=True)
    config = Config(sim_config.config_path)
    config.SetOutputDir(str(output_path))
    config.SetMathModelFrequency(DEFAULT_MATH_MODEL_FREQUENCY)
    config.SetRewindEnabled(False)
    config.ClearConnections()
    exercise = Exercise(str(exercise_path), config)
    ship_config = exercise.getShipConfig()
    ship_config.setInitialPosition(float(x), float(y))
    ship_config.setInitialPropellerRps(0.0)
    ship_config.setInitialRudderAngle(0.0)
    ship_config.setInitialHeading(0.0)
    ship_config.setInitialSurgeVelocity(0.0)
    ship_config.setInitialSwayVelocity(0.0)
    ship_config.setInitialYawVelocity(0.0)

    math_model = MathModel()
    try:
        step(f"MathModel.Initialize() on {Path(exercise_path).name} at ({x:.1f}, {y:.1f})")
        if not math_model.Initialize(config, exercise):
            raise RuntimeError(f"MathModel initialization failed for {exercise_path} at ({x:.1f}, {y:.1f}).")
        step("initialized; enableBridge() + simulateSeconds(INITIALIZATION_TIME)")
        math_model.enableBridge()
        math_model.simulateSeconds(INITIALIZATION_TIME)   # as run_simulation, before reading controls
        step("reading getRudderControls() / getMaximumAngle()")
        rudders = list(math_model.getShipInterface(0).getRudderControls())
        if not rudders:
            raise RuntimeError(f"{Path(exercise_path).name}: the ship has no rudder controls.")
        maxima = sorted({float(r.getMaximumAngle()) for r in rudders})
        if len(maxima) != 1:
            raise RuntimeError(f"{Path(exercise_path).name}: rudders report different maxima {maxima} deg.")
        step(f"{len(rudders)} rudder(s), maximum {maxima[0]:g} deg")
        return maxima[0]
    finally:
        for method in ("Terminate", "Dispose"):
            step(f"math model {method}()")
            try:
                getattr(math_model, method)()
            except Exception as exc:
                print(f"Warning: Error during math model {method}() (rudder probe): {exc}", flush=True)
        step("done")


PROBE_ATTEMPTS = 3   # start points the rudder probe tries before giving up


def default_sim_config() -> SimConfig:
    """SimConfig for SIMEXE_REPO_FOLDER (library, config file, TIMEOUT_SECONDS)."""

    return SimConfig(lib_path=str(Path(simexe_path) / 'exe' / 'installed'),
                     config_path=str(Path(simexe_path) / 'config' / 'simXdrive.config.xml'),
                     timeout_seconds=TIMEOUT_SECONDS)


def probe_fh_rudder_max_deg(ship_model: Union[ShipModel, str], configuration: str,
                            sim_config: Optional[SimConfig] = None,
                            start_xy: Optional[Tuple[float, float]] = None) -> float:
    """
    FH's own rudder maximum (getMaximumAngle, deg) for the ship's exercise in
    `configuration`, from one short FH session in a separate process (as the
    episodes run).

    Args:
        ship_model: Ship with classic rudder controls (ShipModel or name).
        configuration: "full" / "confined" (selects the exercise).
        sim_config: Default: default_sim_config().
        start_xy: Initial position; default: a random point in the river.

    Raises:
        FileNotFoundError: SIMEXE_REPO_FOLDER or a simexe input is missing
            (checked before FH is started).
        RuntimeError: The probe timed out, or failed at PROBE_ATTEMPTS start
            points (the last error is chained).
    """

    ship_model = _as_ship_model(ship_model)
    sim_config = sim_config or default_sim_config()
    exercise_path, _ = exercise_paths(ship_model, configuration)
    _check_simexe_setup(sim_config, exercise_path, time_start_xmls=False)
    if start_xy is not None:
        candidates = [tuple(float(v) for v in start_xy)]
    else:
        # Start points filtered like the episodes' (away from map cut-off edges).
        env = PySimEnv(render_mode=None, time_step=0.1, wind=False, current=False)
        cutoff = _cutoff_geometry(env) if CUTOFF_CLEARANCE > 0 else None
        candidates = [tuple(float(v) for v in p)
                      for p in _sample_points(env.sample_point_in_river, PROBE_ATTEMPTS, cutoff)]
        if not candidates:
            raise RuntimeError("Rudder probe: no start point could be sampled in the river.")
    print(f"Probing FH's rudder maximum for {ship_model.name} ({exercise_path.name}) in a separate FH "
          f"process; this takes as long as starting one episode (timeout {sim_config.timeout_seconds:g} s) ...",
          flush=True)
    last_error = None
    for x, y in candidates:
        scratch = Path(tempfile.mkdtemp(prefix="rudder_probe_"))
        pool = multiprocessing.Pool(1, initializer=_ignore_sigint)
        try:
            res = pool.apply_async(_probe_rudder_max_worker,
                                   args=[sim_config, str(exercise_path), str(scratch), x, y])
            try:
                value = float(res.get(sim_config.timeout_seconds))
            except multiprocessing.TimeoutError:
                raise RuntimeError(f"Rudder probe timed out after {sim_config.timeout_seconds:g} s "
                                   f"({exercise_path.name}).") from None
            except Exception as exc:   # e.g. Initialize() failed at this point: try the next one
                last_error = exc
                print(f"Rudder probe at ({x:.1f}, {y:.1f}) failed: {type(exc).__name__}: {exc}", flush=True)
                continue
        finally:
            pool.terminate()
            pool.join()
            shutil.rmtree(scratch, ignore_errors=True)
        print(f"FH rudder maximum (getMaximumAngle) of {ship_model.name} in {exercise_path.name}: {value:g} deg")
        return value
    raise RuntimeError(f"Rudder probe failed at {len(candidates)} start point(s) ({exercise_path.name}); "
                       f"last error: {type(last_error).__name__}: {last_error}") from last_error


def resolve_rudder_max_deg(ship_model: Union[ShipModel, str], configuration: str, **probe_kwargs) -> float:
    """
    Rudder scale of a run: the profile's fixed value, or FH's own maximum
    (probe_fh_rudder_max_deg, `probe_kwargs` passed on) when it is None.
    """

    value = ship_rudder_max_deg(ship_model)
    return value if value is not None else probe_fh_rudder_max_deg(ship_model, configuration, **probe_kwargs)


def _fh_state(ship_interface, env: "PySimEnv", wind_current: bool) -> np.ndarray:
    """
    FH's current state as a row's state part, in the framework's conventions:
    [x, y, heading, surge, sway, yaw_rate, UKC] and, with `wind_current`,
    [wind speed, wind dir, current speed, current dir] after it.
    """

    pos = ship_interface.getShipPosition()
    vog = ship_interface.getShipVelocityOverGround()
    state = [
        pos.x,
        pos.y,
        fh_heading_to_env_rad(ship_interface.getShipHeading()),
        vog.x,                                    # surge
        vog.y,                                    # sway, port positive (FH = framework)
        np.radians(ship_interface.getShipYawRate()),
        ship_interface.getKeelClearance(),
    ]
    if wind_current:
        wind_state = ship_interface.getWindState()
        current_speed, current_direction = get_current_velocity(ship_interface)
        state += [
            wind_state.Speed,
            env.normalize_angle(np.radians(wind_state.Direction)),
            current_speed,
            env.normalize_angle(np.radians(current_direction)),
        ]
    return np.array(state, dtype=np.float64)


def _wrap180(angle_deg: float) -> float:
    """Angle difference [deg] wrapped to [-180, 180)."""

    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _actuator_angles(steering: str, controls, azipod_command_deg: float) -> list:
    """
    (actual, target) angle [deg] of every steering control.

    Azipods: target = the commanded direction itself (NOT FH's
    getDesiredDirection, which might only follow the command after the next
    simulation step -- the check would then pass at once and settle nothing).
    Rudders: target = FH's getDesiredAngle (a rudder starts at its commanded
    angle via setInitialRudderAngle, so this only guards against a lag).
    """

    if steering == "azipod":
        return [(float(c.getActualDirection()), float(azipod_command_deg)) for c in controls]
    return [(float(c.getActualAngle()), float(c.getDesiredAngle())) for c in controls]


def _actuators_settled(steering: str, controls, azipod_command_deg: float) -> bool:
    """True if every steering control is within ACTUATOR_SETTLE_TOL_DEG of its target (angles wrapped)."""

    return all(abs(_wrap180(actual - target)) <= ACTUATOR_SETTLE_TOL_DEG
               for actual, target in _actuator_angles(steering, controls, azipod_command_deg))


def run_simulation(sim_config: SimConfig, sim_setup: SimSetup) -> SimSetup:
    """Run one episode in FH (inside a pool worker process) and return its rows."""

    # Always import here (a conditional import would make the names local and unbound).
    if not SIM_AVAILABLE and sim_config.lib_path not in sys.path:
        sys.path.append(sim_config.lib_path)
    from sim import Config, Exercise, MathModel, SimException

    output_path = Path(sim_setup.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logfile_path = output_path / "math_model.log"

    # Root logger on purpose: this worker process only runs simulations, and
    # the FH bridge's own log messages must end up in this file too.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(str(logfile_path), mode="w")
    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    env = PySimEnv(
        render_mode=None,
        time_step=sim_setup.dt,
        wind=False,
        current=False,
    )
    cutoff = _worker_cutoff_geometry(env)

    sim_setup.output.status = "PASSED"
    sim_setup.output.time_seconds = -1.0

    math_model = None
    time_started = time.monotonic()

    try:
        config = Config(sim_config.config_path)
        config.SetOutputDir(str(sim_setup.output_path))
        config.SetMathModelFrequency(DEFAULT_MATH_MODEL_FREQUENCY)
        config.SetRewindEnabled(False)
        config.ClearConnections()

        exercise = Exercise(sim_setup.exercise_path, config)
        ship_config = exercise.getShipConfig()
        rng = np.random.default_rng(sim_setup.seed)

        ship = ship_profile(sim_setup.ship_model)
        conf = configuration_profile(sim_setup.configuration)
        # Framework rudder sign (positive action = ship turns to port) -> FH's
        # command sign: +1 azipod (Tripoli), -1 classic rudder (Myzako).
        rudder_sign = fh_rudder_sign(ShipModel[sim_setup.ship_model])
        current_margin = straight_current_margin(sim_setup.configuration)
        rudder_max_deg = float(sim_setup.rudder_max_deg)   # rudder_action = +-1 <-> +-this [deg]
        if not rudder_max_deg > 0:
            raise ValueError(f"SimSetup.rudder_max_deg must be positive, got {sim_setup.rudder_max_deg}")

        # One constant (rudder, thrust) action per episode, by design: it is
        # applied at every step (and used for the initial rudder angle and rps).
        bank_straight = sim_setup.episode_kind == "bank_straight"
        straight = sim_setup.episode_kind == "straight" or bank_straight
        if straight:
            # Steady straight sailing (see STRAIGHT_*).
            if rng.random() < STRAIGHT_RUDDER_ZERO_SHARE:
                rudder_action = 0.0
            else:
                rudder_action = rng.uniform(-1.0, 1.0) * STRAIGHT_RUDDER_MAX_DEG / rudder_max_deg
            thrust_action = rng.uniform(*STRAIGHT_THRUST_RANGE)
            action = np.array([rudder_action, thrust_action], dtype=np.float32)
            init_surge = float(np.clip(ship.straight_surge_per_thrust * float(action[1])
                                       * rng.uniform(*STRAIGHT_SURGE_SPREAD), *INIT_SURGE_RANGE))
            init_sway = float(rng.uniform(*STRAIGHT_INIT_SWAY_RANGE))
            init_yaw_rate = float(rng.uniform(*STRAIGHT_INIT_YAW_RATE_RANGE))
            # Free water this episode needs: its own speed plus a following current.
            needed_m = 1.1 * (init_surge + current_margin) * float(sim_setup.horizon_seconds)
            start_xy = np.array([sim_setup.x_pos, sim_setup.y_pos], dtype=np.float64)
            lookahead_m = straight_lookahead(sim_setup.horizon_seconds, current_margin)
            if bank_straight:
                heading_rad, free_m, bank_m = _bank_straight_heading(env, cutoff, start_xy, lookahead_m,
                                                                     needed_m, rng)
                sim_setup.output.info["bank_distance_m"] = bank_m
            else:
                heading_rad, free_m = _straight_heading(env, cutoff, start_xy, lookahead_m, needed_m, rng)
            init_heading_fh = env_rad_to_fh_heading_deg(heading_rad)
            sim_setup.output.info["straight_free_m"] = free_m
        else:
            # Same draws, in the same order, as before the straight episodes existed.
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            t_lo, t_hi = (float(v) for v in sim_setup.thrust_range)
            if (t_lo, t_hi) != (-1.0, 1.0):   # same draw, rescaled: the default stays bit-identical
                action[1] = np.float32(t_lo + 0.5 * (float(action[1]) + 1.0) * (t_hi - t_lo))
            init_heading_fh = float(rng.uniform(0.0, 360.0))
            init_surge = float(rng.uniform(*INIT_SURGE_RANGE))
            init_sway = float(rng.uniform(*INIT_SWAY_RANGE))
            init_yaw_rate = float(rng.uniform(*INIT_YAW_RATE_RANGE))

        ship_config.setInitialPosition(sim_setup.x_pos, sim_setup.y_pos)
        # Initial rps as FH will run the propeller for this lever: a ship without
        # reverse thrust runs a negative lever down to 0 rps (see ShipProfile).
        initial_rps = _PROPELLER_RPS_SCALE * float(action[1])
        if initial_rps < 0 and not ship.reverse_thrust:
            initial_rps = 0.0
        ship_config.setInitialPropellerRps(float(initial_rps))
        # FH's sign, as the per-step command below (on the Myzako it used to be the
        # opposite, so every episode started with the rudder on the wrong side).
        ship_config.setInitialRudderAngle(float(rudder_sign * rudder_max_deg * action[0]))
        ship_config.setInitialHeading(float(init_heading_fh))
        ship_config.setInitialSurgeVelocity(init_surge)
        ship_config.setInitialSwayVelocity(init_sway)
        ship_config.setInitialYawVelocity(init_yaw_rate)

        math_model = MathModel()
        if not math_model.Initialize(config, exercise):
            error_msg = "MathModel initialization failed!"
            print(error_msg)
            sim_setup.output.status = "FAILED"
            sim_setup.output.info["error"] = error_msg
            return sim_setup

        print(f"Math model initialized! Starting simulation at "
              f"[{sim_setup.x_pos:10.3f} {sim_setup.y_pos:10.3f}]")

        math_model.enableBridge()

        if conf.wind_current:
            environment = math_model.getEnvironmentInterface()
            wind_field_control = environment.getWindFieldControl()
            wind_field_transform = wind_field_control.getFieldTransform()
            wind_field_transform.Rotation = float(rng.uniform(0.0, 360.0))
            wind_field_transform.Offset = 0.0
            # Half-normal scale factor of the wind field, clipped (straight
            # episodes: STRAIGHT_CALM_SHARE of them without wind).
            wind_scale = float(np.clip(np.abs(rng.normal(loc=0.0, scale=WIND_SCALE_SIGMA)),
                                       0.0, WIND_SCALE_MAX))
            if straight and rng.random() < STRAIGHT_CALM_SHARE:
                wind_scale = 0.0
            wind_field_transform.Scale = wind_scale
            wind_field_control.setFieldTransform(wind_field_transform)

        math_model.simulateSeconds(INITIALIZATION_TIME)  # initialize velocities

        ship_interface = math_model.getShipInterface(0)

        if conf.wind_current:
            # Wind warm-up (see WIND_WARMUP_TIME): the first recorded state must
            # carry this episode's wind, not the exercise default. A lagging
            # WindState is frozen (bit-identical), so exact equality detects it.
            settled_wind = (ship_interface.getWindState().Speed, ship_interface.getWindState().Direction)
            math_model.simulateSeconds(WIND_WARMUP_TIME)
            warmed_wind = (ship_interface.getWindState().Speed, ship_interface.getWindState().Direction)
            if warmed_wind == settled_wind:
                error_msg = (f"WindState unchanged after WIND_WARMUP_TIME={WIND_WARMUP_TIME} s "
                             f"({warmed_wind}): still the exercise default -- increase WIND_WARMUP_TIME.")
                print(error_msg)
                sim_setup.output.status = "FAILED"
                sim_setup.output.info["error"] = error_msg
                return sim_setup

        # Controls the action is applied to (see ShipProfile.steering). Missing
        # controls fail the episode: the recorded action would have no effect.
        if ship.steering == "azipod":
            steering_controls = list(ship_interface.getAzipodControls())
            thrust_controls = steering_controls
            needed = {"azipod": steering_controls}
        else:
            steering_controls = list(ship_interface.getRudderControls())
            thrust_controls = list(ship_interface.getPropellerControls())
            needed = {"rudder": steering_controls, "propeller": thrust_controls}
        missing = [name for name, controls in needed.items() if not controls]
        if missing:
            error_msg = (f"Ship has no {' / '.join(missing)} controls ({sim_setup.ship_model}, "
                         f"steering '{ship.steering}'): the action cannot be applied. Does the "
                         f"exercise match the ship?")
            print(error_msg)
            sim_setup.output.status = "FAILED"
            sim_setup.output.info["error"] = error_msg
            return sim_setup
        if ship.steering == "rudder":
            # setControlValue(+-1) moves the rudder to +-FH's OWN maximum: that must
            # be the run's scale, or the stored rudder_action would mean other angles.
            maxima = sorted({float(rudder.getMaximumAngle()) for rudder in steering_controls})
            if len(maxima) != 1 or not np.isclose(maxima[0], rudder_max_deg):
                error_msg = (f"FH's rudder maximum {maxima} deg differs from the run's rudder scale "
                             f"{rudder_max_deg:g} deg (probed before the run): this episode's "
                             f"rudder_action would mean other angles. Does {Path(sim_setup.exercise_path).name} "
                             f"use another ship configuration?")
                print(error_msg)
                sim_setup.output.status = "FAILED"
                sim_setup.output.info["error"] = error_msg
                return sim_setup

        if ship.steering == "azipod":
            rudder_command = float(rudder_sign * action[0] * rudder_max_deg)   # direction [deg]
        else:
            rudder_command = float(rudder_sign * action[0])   # normalized to FH's own maximum
        thrust_command = float(action[1])

        def apply_command() -> None:
            """Set this episode's (constant) steering and thrust command on all controls."""
            if ship.steering == "azipod":
                for azipod in steering_controls:
                    azipod.setDesiredDirection(rudder_command)
                    azipod.setEngineLeverValue(thrust_command)
            else:
                for rudder in steering_controls:
                    rudder.setControlValue(rudder_command)
                for propeller in thrust_controls:
                    propeller.engage()
                    propeller.setEngineLeverValue(thrust_command)

        # Settle the actuators (see ACTUATOR_SETTLE_MAX_SECONDS) before the first row.
        settle_steps = 0
        max_settle_steps = int(round(ACTUATOR_SETTLE_MAX_SECONDS / sim_setup.dt))
        apply_command()
        while not _actuators_settled(ship.steering, steering_controls, rudder_command):
            if settle_steps >= max_settle_steps:
                error_msg = (f"Steering actuators not at their commanded angle after "
                             f"{ACTUATOR_SETTLE_MAX_SECONDS:g} s: (actual, target) {_actuator_angles(ship.steering, steering_controls, rudder_command)}")
                print(error_msg)
                sim_setup.output.status = "FAILED"
                sim_setup.output.info["error"] = error_msg
                return sim_setup
            math_model.simulateSeconds(sim_setup.dt)
            settle_steps += 1
            apply_command()
        sim_setup.output.info["settle_seconds"] = settle_steps * sim_setup.dt

        # [x, y, heading, surge, sway, yaw_rate, UKC] (+ [wind speed, wind dir,
        # current speed, current dir] in the full configuration)
        input_state = _fh_state(ship_interface, env, conf.wind_current)
        if conf.wind_current:
            initial_wind_state = ship_interface.getWindState()
            print(f"Wind direction: {initial_wind_state.Direction} - wind speed: {initial_wind_state.Speed}")

        initial_ray_features = compute_ray_features(
            env,
            np.array([input_state[0], input_state[1]]),
            float(input_state[2]),
        )

        steps_per_episode = max(1, int(round(sim_setup.horizon_seconds / sim_setup.dt)))

        curr_ship_pos = np.array([input_state[0], input_state[1]], dtype=np.float32)

        input_states = []
        output_states = []
        traj = [curr_ship_pos.copy()]
        ray_features = initial_ray_features

        # A row is recorded only if its start and end state are within the stop
        # limits; the episode stops at the first violating step, earlier rows are kept.
        termination = stop_reason(env, curr_ship_pos, float(input_state[6]), ray_features, cutoff)
        if termination is not None:
            termination = "start_" + termination
        else:
            termination = "horizon"
            for step in range(steps_per_episode):
                apply_command()
                math_model.simulateSeconds(sim_setup.dt)

                next_input_state = _fh_state(ship_interface, env, conf.wind_current)
                next_ship_pos = np.array([next_input_state[0], next_input_state[1]], dtype=np.float32)
                traj.append(next_ship_pos.copy())

                new_surge = next_input_state[3]
                new_sway = next_input_state[4]     # port positive (FH = framework)
                new_yaw_rate = next_input_state[5]

                next_ray_features = compute_ray_features(
                    env,
                    np.array([next_input_state[0], next_input_state[1]]),
                    float(next_input_state[2]),
                )

                reason = stop_reason(env, next_ship_pos, float(next_input_state[6]), next_ray_features, cutoff)
                if reason is not None:
                    termination = reason
                    break

                input_states.append(np.concatenate([input_state, action, ray_features], axis=0))
                output_states.append(
                    np.array(
                        [
                            new_surge - input_state[3],
                            new_sway - input_state[4],
                            new_yaw_rate - input_state[5],
                        ],
                        dtype=np.float64,
                    )
                )

                input_state = next_input_state
                ray_features = next_ray_features

        sim_setup.output.trajectory = np.array(traj, dtype=np.float32)
        sim_setup.output.info["termination"] = termination
        sim_setup.output.info["steps"] = len(input_states)

        if not input_states:
            sim_setup.output.status = "EMPTY"
            sim_setup.output.info["error"] = f"no valid transitions (termination: {termination})"
        else:
            sim_setup.output.x_input = np.array(input_states, dtype=np.float64)
            sim_setup.output.y_output = np.array(output_states, dtype=np.float64)

    except SimException as e:
        print(f"Simulator exception: {e}")
        _handle_exception(sim_setup, e, "SimException")

    except Exception as e:
        print(f"Unexpected error: {e}")
        _handle_exception(sim_setup, e, type(e).__name__)

    finally:
        if math_model is not None:
            # Separate try blocks: Dispose() must run even if Terminate() fails.
            try:
                math_model.Terminate()
            except Exception as e:
                print(f"Warning: Error during math model Terminate(): {e}")
            try:
                math_model.Dispose()
            except Exception as e:
                print(f"Warning: Error during math model Dispose(): {e}")

            # Clean the scratch directory, keep math_model.log.
            try:
                for item in output_path.iterdir():
                    if item.name == "math_model.log":
                        continue
                    try:
                        if item.is_file():
                            item.unlink()
                    except (PermissionError, OSError) as e:
                        print(f"Warning: Could not delete {item}: {e}")
            except (PermissionError, OSError) as e:
                print(f"Warning: Could not list directory {output_path}: {e}")
        sim_setup.output.time_seconds = time.monotonic() - time_started  # also on early returns

    return sim_setup


def _handle_exception(sim_setup: SimSetup, exception: Exception, exception_type: str) -> None:
    """Mark the episode FAILED and store the exception details."""

    sim_setup.output.status = "FAILED"
    sim_setup.output.info["message"] = str(exception)
    sim_setup.output.info["exception_type"] = exception_type

    tb = traceback.extract_tb(exception.__traceback__)
    if tb:
        filename, line_number, function_name, _ = tb[-1]
        sim_setup.output.info["traceback"] = (
            f"Exception occurred at {filename}, "
            f"line {line_number} in {function_name}"
        )
    else:
        sim_setup.output.info["traceback"] = "No traceback available"


def run_in_process(sim_setup: SimSetup, sim_config: SimConfig, pool: Pool) -> SimSetup:
    """Run one simulation in a pool worker; returns status TIMEOUT after sim_config.timeout_seconds."""

    res = pool.apply_async(run_simulation, args=[sim_config, sim_setup])

    try:
        return res.get(sim_config.timeout_seconds)
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        sim_setup.output.status = "TIMEOUT"
        sim_setup.output.time_seconds = sim_config.timeout_seconds
        return sim_setup


def _ignore_sigint():
    """Pool worker initializer: only the main process reacts to Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _check_simexe_setup(sim_config: SimConfig, exercise_path: Path,
                        extra_exercises: Iterable[Path] = (), time_start_xmls: bool = True) -> None:
    """
    Fail fast if SIMEXE_REPO_FOLDER or a required simexe input is missing
    (the current/tide XMLs only with `time_start_xmls`).
    """

    if not simexe_path:
        raise FileNotFoundError("Environment variable SIMEXE_REPO_FOLDER is not set; it must "
                                "point to the simexe repository.")
    required = {
        "simexe library folder": Path(sim_config.lib_path),
        "simexe config": Path(sim_config.config_path),
        "exercise": Path(exercise_path),
        **{f"exercise ({Path(p).name})": Path(p) for p in extra_exercises},
        **({"current XML": CURRENT_XML_PATH, "tide XML": TIDE_XML_PATH} if time_start_xmls else {}),
    }
    missing = [f"  {name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("simexe setup incomplete (SIMEXE_REPO_FOLDER="
                                f"{simexe_path}). Missing:\n" + "\n".join(missing))


# ----------------------------------------------------------------------
# Checkpoints: "<output stem>_checkpoints/" holds meta.json (run signature)
# and shard_<first>_<last>.npz files with every attempted episode of that
# range (failed ones without rows). No pickle needed to load them.
# ----------------------------------------------------------------------

def _atomic_savez(path: Path, **arrays) -> None:
    """np.savez via a temp file + os.replace, so a crash never leaves a truncated shard."""

    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp_path, "wb") as fh:  # file handle, so savez doesn't append ".npz"
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)  # never leave a stray temp file
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a text file via a temp file + os.replace, so a crash never leaves it half-written."""

    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)  # never leave a stray temp file
        raise


def _atomic_savetxt(path: Union[str, Path], data: np.ndarray, **kwargs) -> None:
    """np.savetxt via a temp file + os.replace, so a crash never leaves a truncated CSV."""

    path = Path(path)
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp_path, "wb") as fh:
            np.savetxt(fh, data, **kwargs)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)  # never leave a stray temp file
        raise


def _write_checkpoint_shard(ckpt_dir: Path, indices: list, episode_meta: dict,
                            episode_results: dict, n_chunks: int, n_x_features: int) -> Path:
    """
    Write episodes *indices* (contiguous) to one shard.

    episode_meta: index -> (reason, start_x, start_y, start_type), all episodes.
    episode_results: index -> (x, y, trajectory), episodes with rows only.
    n_chunks: chunks these episodes took (to replay the tide RNG on resume).
    n_x_features: row width (see nbr_x_features), for an empty x array.
    """

    idx = np.asarray(indices, dtype=np.int64)
    passed = [i for i in indices if i in episode_results]
    x_parts = [episode_results[i][0] for i in passed]
    y_parts = [episode_results[i][1] for i in passed]
    t_parts = [episode_results[i][2] for i in passed]

    path = ckpt_dir / f"shard_{indices[0]:07d}_{indices[-1]:07d}.npz"
    _atomic_savez(
        path,
        ep_index=idx,
        ep_reason=np.array([episode_meta[i][0] for i in indices], dtype=str),
        ep_start_xy=np.array([[episode_meta[i][1], episode_meta[i][2]] for i in indices], dtype=np.float32),
        ep_start_type=np.array([episode_meta[i][3] for i in indices], dtype=str),
        passed_index=np.asarray(passed, dtype=np.int64),
        rows_per_passed=np.array([len(x) for x in x_parts], dtype=np.int64),
        # float32: same precision as the final CSV.
        x=(np.vstack(x_parts) if x_parts else np.zeros((0, n_x_features))).astype(np.float32),
        y=(np.vstack(y_parts) if y_parts else np.zeros((0, NBR_Y_FEATURES))).astype(np.float32),
        traj_len=np.array([len(t) for t in t_parts], dtype=np.int64),
        traj=(np.vstack(t_parts) if t_parts else np.zeros((0, 2))).astype(np.float32),
        n_chunks=np.int64(n_chunks),
    )
    return path


def _load_checkpoint_shards(ckpt_dir: Path):
    """
    Load all shards in episode order. Returns (episodes, n_done, n_chunks) with
    episodes = [(index, reason, start_xy, start_type, x, y, traj)] (x/y/traj None
    without rows); the episodes must cover 0..n_done-1 without gaps.
    """

    shard_paths = sorted(ckpt_dir.glob("shard_*.npz"))
    episodes, n_done, n_chunks = [], 0, 0
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as d:
            ep_index = d["ep_index"]
            if len(ep_index) == 0 or ep_index[0] != n_done or np.any(np.diff(ep_index) != 1):
                raise ValueError(
                    f"Checkpoint shard {path.name} does not continue at episode {n_done}; "
                    f"the checkpoint directory {ckpt_dir} is inconsistent. Remove it (or pass "
                    "resume=False) to start over.")
            row_ends = np.cumsum(d["rows_per_passed"])
            traj_ends = np.cumsum(d["traj_len"])
            # Read each array ONCE: NpzFile.__getitem__ decompresses/reads the whole
            # array on every access, and a slice is only a view that keeps that whole
            # array alive. Indexing d["x"] inside the per-episode loop therefore held
            # one full copy of the shard per episode (MemoryError, and quadratic time).
            x_all, y_all, traj_all = d["x"], d["y"], d["traj"]
            rows = {}
            for k, i in enumerate(d["passed_index"]):
                r0 = row_ends[k - 1] if k else 0
                t0 = traj_ends[k - 1] if k else 0
                rows[int(i)] = (x_all[r0:row_ends[k]], y_all[r0:row_ends[k]],
                                traj_all[t0:traj_ends[k]])
            for j, i in enumerate(ep_index):
                x, y, traj = rows.get(int(i), (None, None, None))
                episodes.append((int(i), str(d["ep_reason"][j]), d["ep_start_xy"][j],
                                 str(d["ep_start_type"][j]), x, y, traj))
            n_done = int(ep_index[-1]) + 1
            n_chunks += int(d["n_chunks"])
    return episodes, n_done, n_chunks


def validate_start_settings(bank_fraction: float, bank_band: Tuple[float, float],
                            straight_fraction: float, bank_straight_fraction: float = 0.0,
                            bank_straight_band: Tuple[float, float] = BANK_STRAIGHT_BAND,
                            thrust_range: Tuple[float, float] = THRUST_ACTION_RANGE) -> Tuple[float, float]:
    """
    Check the start-point settings of collect_supervised_dataset (cheap: call
    it before anything slow, such as the rudder probe).

    Returns:
        (band_min, band_max) as floats.

    Raises:
        ValueError: For an invalid fraction or bank band.
    """

    if not 0.0 <= bank_fraction <= 1.0:
        raise ValueError(f"bank_fraction must be in [0, 1], got {bank_fraction}")
    # Sum, not 1 - bank_fraction: 1.0 - 0.8 = 0.19999999999999996 would reject 0.8 + 0.2.
    if not 0.0 <= bank_straight_fraction <= 1.0:   # also rejects NaN
        raise ValueError(f"bank_straight_fraction must be in [0, 1], got {bank_straight_fraction}")
    if bank_fraction + bank_straight_fraction > 1.0 + 1e-9:
        raise ValueError(f"bank_fraction + bank_straight_fraction must be <= 1, got "
                         f"{bank_fraction} + {bank_straight_fraction}")
    rest = bank_fraction + bank_straight_fraction
    if not (straight_fraction >= 0.0 and rest + straight_fraction <= 1.0 + 1e-9):  # also rejects NaN
        raise ValueError(f"straight_fraction must be in [0, 1 - bank_fraction - bank_straight_fraction] = "
                         f"[0, {1.0 - rest:g}] (straight episodes use uniform starts), got {straight_fraction}")
    band_min, band_max = (float(v) for v in bank_band)
    if bank_fraction > 0 and not 0.0 <= band_min < band_max:
        raise ValueError(f"bank_band must be (min, max) with 0 <= min < max, got {bank_band}")
    t_lo, t_hi = (float(v) for v in thrust_range)
    if not -1.0 <= t_lo < t_hi <= 1.0:
        raise ValueError(f"thrust_range must be (low, high) with -1 <= low < high <= 1, got {thrust_range}")
    bs_min, bs_max = (float(v) for v in bank_straight_band)
    if bank_straight_fraction > 0 and not MIN_BANK_DISTANCE < bs_min < bs_max:
        raise ValueError(f"bank_straight_band must be (min, max) with MIN_BANK_DISTANCE "
                         f"({MIN_BANK_DISTANCE:g} m) < min < max, got {bank_straight_band}")
    return band_min, band_max


def _empty_result(n_x_features: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Empty result with the normal column shapes (`n_x_features`: see nbr_x_features)."""

    return (np.zeros((0, n_x_features), dtype=np.float32),
            np.zeros((0, NBR_Y_FEATURES), dtype=np.float32),
            np.zeros((0,), dtype=np.int64), 0)


def _warn_if_stale_csv(output_file: str) -> None:
    """Warn that an existing output_file is from an earlier run (nothing was written)."""

    if Path(output_file).exists():
        print(f"WARNING: {output_file} was NOT overwritten and still contains the data of "
              "an EARLIER run.")


def collect_supervised_dataset(
        nbr_samples: int = 5000,
        horizon_seconds: float = 3.0,
        dt: float = 0.1,
        workers: int = 4,
        output_file: str = "ship_dynamics_dataset.csv",
        seed: Optional[int] = None,
        bank_fraction: float = BANK_START_FRACTION,
        bank_band: Tuple[float, float] = BANK_START_BAND,
        straight_fraction: float = STRAIGHT_FRACTION,
        bank_straight_fraction: float = BANK_STRAIGHT_FRACTION,
        bank_straight_band: Tuple[float, float] = BANK_STRAIGHT_BAND,
        thrust_range: Tuple[float, float] = THRUST_ACTION_RANGE,
        stop_file: Optional[str] = None,
        checkpoint_every: Optional[int] = CHECKPOINT_EVERY,
        resume: bool = True,
        show_plot: bool = False,
        ship_model: Union[ShipModel, str] = DEFAULT_SHIP_MODEL,
        configuration: str = DEFAULT_CONFIGURATION,
        rudder_max_deg: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generate the supervised dataset by simulating episodes in FH.

    Writes a CSV with 41 columns (full) or 37 columns (confined) (layout: see
    load_supervised_dataset). Rows of one episode are correlated: split
    train/val/test by episode_id.

    Episodes run in chunks of `workers`; in the full configuration each chunk
    gets one random tidal offset. An episode stops at the first state outside
    the stop limits (stop_reason); its earlier rows are kept.

    Early stop (checked between chunks): create the stop file, or press
    Ctrl+C once; collected data is then saved as usual. A second Ctrl+C aborts
    without writing the CSV (checkpoints are kept).

    Checkpoints: every `checkpoint_every` episodes a shard is written to
    "<output stem>_checkpoints/". Calling again with the same parameters
    resumes after the last saved episode. With the same `seed` and `workers`,
    the resumed run uses the same start points, episode seeds, tidal offsets
    and row shuffle as an uninterrupted run; the data is then identical only
    if the simulator itself is deterministic and its inputs are unchanged
    (input files are fingerprinted in the run signature, the simulator
    binaries are not). The directory is removed after a complete run and
    kept after an early stop.

    Args:
        nbr_samples: Number of episodes (start points).
        horizon_seconds: Maximum simulated time per episode.
        dt: Time step (multiple of the math-model step).
        workers: Parallel workers = chunk size; 0/None = cpu_count()-1.
        output_file: CSV path.
        seed: Base seed for start points, episodes, tidal offsets and shuffle.
        bank_fraction: Target fraction of near-bank starts (missing ones become uniform).
        bank_band: (min, max) distance in m from the nearest real bank.
        straight_fraction: Fraction of ALL episodes run as "straight" episodes
            (see STRAIGHT_*), taken from the uniform starts (at most all of them).
        bank_straight_fraction: Fraction of ALL episodes run as "bank_straight"
            episodes (see BANK_STRAIGHT_*), sampled bank_straight_band m from
            a real bank; taken from the uniform starts. 0 = none.
        bank_straight_band: (min, max) distance in m from the nearest real bank.
        thrust_range: (low, high) thrust_action range of the random episodes
            (see THRUST_ACTION_RANGE); straight episodes use STRAIGHT_THRUST_RANGE.
        stop_file: Stop-file path; default "STOP_COLLECTION" next to output_file.
        checkpoint_every: Episodes per checkpoint; 0/None disables checkpoints.
        resume: False deletes an existing checkpoint and starts over.
        show_plot: Open the trajectory plot window at the end (blocks until
            closed); the PNG is always saved.
        ship_model: Ship to simulate (ShipModel or name): a key of SHIP_PROFILES.
        configuration: "full" (ship-bank + UKC + wind + current) or "confined"
            (ship-bank + UKC); see CONFIGURATIONS.
        rudder_max_deg: Rudder scale of the run (rudder_action = +-1 <-> +-this
            many degrees). None = resolve_rudder_max_deg (for the Myzako: probe
            FH's getMaximumAngle). A ship with a fixed scale only accepts that
            value. Checked against ship_factory before the run.

    Returns:
        (x_inputs, y_outputs, episode_ids, number of episodes with rows)
    """

    if nbr_samples <= 0:
        raise ValueError("nbr_samples must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if checkpoint_every is not None and checkpoint_every < 0:
        raise ValueError("checkpoint_every must be positive, 0 or None")
    if workers is not None and workers < 0:
        raise ValueError("workers must be >= 0 or None (0/None = cpu_count()-1)")
    ship_model = _as_ship_model(ship_model)
    ship = ship_profile(ship_model)
    conf = configuration_profile(configuration)
    fh_rudder_sign(ship_model)           # raises for a ship without a measured FH rudder sign
    fixed_rudder_max = ship_rudder_max_deg(ship_model)
    if rudder_max_deg is not None and fixed_rudder_max is not None \
            and not np.isclose(float(rudder_max_deg), fixed_rudder_max):
        raise ValueError(f"{ship_model.name} has a fixed rudder scale of {fixed_rudder_max:g} deg, "
                         f"got rudder_max_deg={rudder_max_deg}.")
    if rudder_max_deg is not None and not float(rudder_max_deg) > 0:
        raise ValueError(f"rudder_max_deg must be positive, got {rudder_max_deg}")
    validate_start_settings(bank_fraction, bank_band, straight_fraction,   # before the (slow) probe
                            bank_straight_fraction, bank_straight_band, thrust_range)
    n_x = nbr_x_features(configuration)
    current_margin = straight_current_margin(configuration)
    # simulateSeconds only advances whole math-model steps.
    timed = [("dt", dt), ("INITIALIZATION_TIME", INITIALIZATION_TIME)]
    if conf.wind_current:
        timed.append(("WIND_WARMUP_TIME", WIND_WARMUP_TIME))
    for name, seconds in timed:
        steps = seconds * DEFAULT_MATH_MODEL_FREQUENCY
        if abs(steps - round(steps)) > 1e-9 or round(steps) < 1:
            raise ValueError(f"{name}={seconds} s must be a multiple of the math-model step "
                             f"1/{DEFAULT_MATH_MODEL_FREQUENCY:g} s")
    requested_samples = nbr_samples  # nbr_samples is reassigned after sampling

    out_dir = Path(output_file).resolve().parent
    out_stem = Path(output_file).stem

    sim_config = default_sim_config()

    exercise_path, exercise_path_no_current = exercise_paths(ship_model, configuration)
    uses_no_current = (exercise_path_no_current is not None and straight_fraction > 0
                       and STRAIGHT_NO_CURRENT_SHARE > 0)

    _check_simexe_setup(sim_config, exercise_path,
                        extra_exercises=(exercise_path_no_current,) if uses_no_current else (),
                        time_start_xmls=conf.randomize_time_start)

    # Rudder scale of this run (probes FH for a ship whose scale is FH's own
    # maximum), refused before the long run if it differs from ship_factory.
    if rudder_max_deg is None:
        rudder_max_deg = resolve_rudder_max_deg(ship_model, configuration, sim_config=sim_config)
    rudder_max_deg = float(rudder_max_deg)
    check_ship_against_factory(ship_model, rudder_max_deg)

    print(f"Ship {ship_model.name} ({ship.steering}, rudder +-{rudder_max_deg:g} deg), "
          f"configuration '{configuration}': exercise {exercise_path.name}"
          + (f", no-current exercise {exercise_path_no_current.name}" if uses_no_current else ""))
    uses_static_tide = (not conf.randomize_time_start) or uses_no_current

    output_path = out_dir / "_sim_scratch"

    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    env = PySimEnv(render_mode=None, time_step=dt, wind=False, current=False)

    # Cut-off handling is either explicitly disabled (CUTOFF_CLEARANCE = 0) or
    # must have a geometry: never silently run without it.
    cutoff_geom, cutoff_n, cutoff_length = None, 0, 0.0
    if CUTOFF_CLEARANCE > 0:
        cutoff_geom, cutoff_n, cutoff_length, cutoff_problem = _describe_cutoffs(env)
        if cutoff_problem is not None:
            boundary_png = out_dir / "start_points.png"
            plot_start_points(env, np.zeros((0, 2)), np.zeros((0, 2)), boundary_png)
            see_plot = (f" Check '{boundary_png}' (real bank blue, cut-off red)."
                        if Path(boundary_png).exists() else "")
            raise RuntimeError(
                f"Cut-off handling is enabled (CUTOFF_CLEARANCE={CUTOFF_CLEARANCE:g} m), but "
                f"{cutoff_problem}.{see_plot} bank_segments() only detects straight map edges "
                f"that lie on the polygon's bounding box (within its edge_tol); if this map "
                f"really has no cut-off edges, set CUTOFF_CLEARANCE = 0 to disable cut-off "
                f"handling explicitly."
            )
        print(f"Map cut-off edges: {cutoff_n} segment(s), {cutoff_length:.0f} m in total "
              f"(start points and episodes kept >= {CUTOFF_CLEARANCE:g} m away).")

    setups = []
    x_inputs = []
    y_outputs = []
    episode_ids = []

    band_min, band_max = validate_start_settings(bank_fraction, bank_band, straight_fraction,
                                                 bank_straight_fraction, bank_straight_band, thrust_range)
    bs_min, bs_max = (float(v) for v in bank_straight_band)
    if bank_straight_fraction > 0 and _worker_real_bank(env) is None:
        raise RuntimeError("bank_straight_fraction > 0, but PySimEnv.bank_segments() gives no real-bank "
                           "segments to sail along.")
    if bank_fraction > 0 and band_min <= MIN_BANK_DISTANCE:
        print(f"Warning: bank_band starts at {band_min} m, at or below MIN_BANK_DISTANCE "
              f"({MIN_BANK_DISTANCE} m) -- such starts end immediately as start_bank.")
    if bank_fraction > 0 and band_max > RAY_MAX_DISTANCE:
        print(f"Warning: bank_band ends at {band_max} m, beyond the ray range "
              f"({RAY_MAX_DISTANCE} m) -- the model cannot see a bank that far away.")

    # Independent seeds for episodes, tidal offsets, shuffle and start points.
    # spawn(5)[:4] equals the earlier spawn(4), so these four streams are unchanged.
    if seed is not None:
        episode_ss, tide_ss, shuffle_ss, start_ss, kind_ss = np.random.SeedSequence(seed).spawn(5)
        env._rng = np.random.default_rng(start_ss)  # used by PySimEnv's samplers
        tide_rng = np.random.default_rng(tide_ss)
        shuffle_rng = np.random.default_rng(shuffle_ss)
        kind_rng = np.random.default_rng(kind_ss)
    else:
        episode_ss = None
        tide_rng = None
        shuffle_rng = np.random.default_rng()
        kind_rng = np.random.default_rng()

    # Start points, away from the map cut-off edges.
    def uniform_sampler(k):
        """`k` start points sampled uniformly inside the river polygon."""
        return env.sample_point_in_river(k)

    def bank_sampler(k):
        """`k` start points within [band_min, band_max] m of a real bank."""
        return env.sample_point_near_bank(k, band_min, band_max)

    def bank_straight_sampler(k):
        """`k` start points within [bs_min, bs_max] m of a real bank."""
        return env.sample_point_near_bank(k, bs_min, bs_max)

    n_bank = int(round(bank_fraction * nbr_samples))
    # min(): the two roundings must not add up to more than nbr_samples.
    n_bank_straight = min(int(round(bank_straight_fraction * nbr_samples)), nbr_samples - n_bank)
    xy_uniform = _sample_points(uniform_sampler, nbr_samples - n_bank - n_bank_straight, cutoff_geom)
    xy_bank = (_sample_points(bank_sampler, n_bank, cutoff_geom) if n_bank > 0
               else np.zeros((0, 2), dtype=np.float32))
    # Sampled after the bank starts: with bank_straight_fraction = 0 nothing changes.
    xy_bank_straight = (_sample_points(bank_straight_sampler, n_bank_straight, cutoff_geom)
                        if n_bank_straight > 0 else np.zeros((0, 2), dtype=np.float32))
    if len(xy_bank_straight) < n_bank_straight:
        missing = n_bank_straight - len(xy_bank_straight)
        print(f"Warning: only {len(xy_bank_straight)} of {n_bank_straight} bank-straight start points could "
              f"be sampled (band {bs_min:g}-{bs_max:g} m). Using {missing} extra uniform start points instead.")
        xy_uniform = np.vstack([xy_uniform, _sample_points(uniform_sampler, missing, cutoff_geom)])
    if len(xy_bank) < n_bank:
        missing = n_bank - len(xy_bank)
        print(f"Warning: only {len(xy_bank)} of {n_bank} near-bank start points could be sampled "
              f"(band {band_min:g}-{band_max:g} m; see the PySimEnv warning and start_points.png). "
              f"Using {missing} extra uniform start points instead.")
        xy_uniform = np.vstack([xy_uniform, _sample_points(uniform_sampler, missing, cutoff_geom)])
    n_sampled = len(xy_uniform) + len(xy_bank) + len(xy_bank_straight)
    if n_sampled < nbr_samples:
        print(f"Warning: only {n_sampled} of {nbr_samples} start points could "
              "be sampled" + (f" at least {CUTOFF_CLEARANCE:g} m from the map cut-off edges."
                              if cutoff_geom is not None else "."))

    # Shuffle, so near-bank starts are spread over all chunks.
    xy = np.vstack([xy_uniform, xy_bank, xy_bank_straight])
    # Wide dtype: resumed episodes write their start type back into this array.
    start_types = np.array(["uniform"] * len(xy_uniform) + ["bank"] * len(xy_bank)
                           + ["bank_straight"] * len(xy_bank_straight), dtype="<U16")
    order = env._rng.permutation(len(xy))
    xy, start_types = xy[order], start_types[order]
    # "straight" episodes (see STRAIGHT_*): a random subset of the uniform starts.
    uniform_idx = np.flatnonzero(start_types == "uniform")
    n_straight = min(len(uniform_idx), int(round(straight_fraction * len(xy))))
    n_no_current = 0
    if n_straight > 0:
        straight_idx = kind_rng.choice(uniform_idx, n_straight, replace=False)
        start_types[straight_idx] = "straight"
        # ... of which STRAIGHT_NO_CURRENT_SHARE without current (the ship's no-current
        # exercise; only in a configuration that has one).
        n_no_current = int(round(STRAIGHT_NO_CURRENT_SHARE * n_straight)) if uses_no_current else 0
        if n_no_current > 0:
            start_types[kind_rng.choice(straight_idx, n_no_current, replace=False)] = "straight_nocur"
    print(f"Start points: {int((start_types == 'uniform').sum())} uniform, {len(xy_bank)} near-bank "
          f"({band_min:g}-{band_max:g} m from a real bank), {n_straight} straight ({n_no_current} without current) "
          f"(uniform start, rudder ~0, thrust {STRAIGHT_THRUST_RANGE[0]:g}-{STRAIGHT_THRUST_RANGE[1]:g})"
          + (f", {len(xy_bank_straight)} bank-straight ({bs_min:g}-{bs_max:g} m from a real bank, "
             f"parallel to it, rudder ~0)" if n_bank_straight > 0 else ""))
    plot_start_points(env, xy_uniform, xy_bank, out_dir / "start_points.png", xy_bank_straight)

    x0 = xy[:, 0]
    y0 = xy[:, 1]

    nbr_samples = len(xy)
    if nbr_samples == 0:
        print("Warning: no start points could be sampled!")
        _warn_if_stale_csv(output_file)
        return _empty_result(n_x)
    trajectories = []
    counter = 0
    spawn_counter = 0  # finished simulations, for the cache cleanup
    cache_clears_done = 0
    episode_results = {}     # index -> (x, y, trajectory)
    termination_counts = {}
    rows_per_termination = {}

    if episode_ss is not None:
        episode_seeds = [int(s) for s in episode_ss.generate_state(nbr_samples)]
    else:
        episode_seeds = [None] * nbr_samples
    all_start_types = ("uniform", "bank", "straight", "straight_nocur", "bank_straight")
    rows_per_start_type = {st: 0 for st in all_start_types}
    kept_per_start_type = {st: 0 for st in all_start_types}

    for i in range(nbr_samples):
        setup = SimSetup(
            exercise_path=str(exercise_path_no_current if start_types[i] == "straight_nocur" else exercise_path),
            output_path=str(output_path / str(i)),
            x_pos=float(x0[i]),
            y_pos=float(y0[i]),
            dt=dt,
            horizon_seconds=horizon_seconds,
            seed=episode_seeds[i],
            episode_kind=("straight" if start_types[i] in ("straight", "straight_nocur")
                          else "bank_straight" if start_types[i] == "bank_straight" else "random"),
            ship_model=ship_model.name,
            rudder_max_deg=rudder_max_deg,
            thrust_range=(float(thrust_range[0]), float(thrust_range[1])),
            configuration=configuration,
            output=SimOutput(status="", time_seconds=0.0)
        )
        setups.append(setup)

    num_workers = workers if workers else max(1, multiprocessing.cpu_count() - 1)

    # ---- Checkpoints and resume ----
    episode_meta = {}  # index -> (reason, start_x, start_y, start_type)

    def _record_episode(_index, _reason, _start_xy, _start_type, _x_ep=None, _y_ep=None, _traj_ep=None):
        """Book-keeping for live and resumed episodes."""
        nonlocal counter
        termination_counts[_reason] = termination_counts.get(_reason, 0) + 1
        episode_meta[_index] = (_reason, float(_start_xy[0]), float(_start_xy[1]), _start_type)
        if _x_ep is not None:
            n_rows = len(_x_ep)
            rows_per_termination[_reason] = rows_per_termination.get(_reason, 0) + n_rows
            rows_per_start_type[_start_type] += n_rows
            kept_per_start_type[_start_type] += 1
            episode_results[_index] = (_x_ep, _y_ep, _traj_ep)
            counter += 1

    checkpointing = bool(checkpoint_every)
    ckpt_dir = out_dir / f"{out_stem}_checkpoints"
    # Everything that changes the data; a resume with a different signature is refused.
    run_signature = {
        "version": _CHECKPOINT_VERSION,
        "nbr_samples": int(requested_samples),
        "horizon_seconds": float(horizon_seconds),
        "dt": float(dt),
        "seed": None if seed is None else int(seed),
        "bank_fraction": float(bank_fraction),
        "bank_band": [band_min, band_max],
        "ship_model": ship_model.name,
        "configuration": configuration,
        "steering": ship.steering,
        "straight_fraction": float(straight_fraction),
        "straight": {
            "rudder_max_deg": float(STRAIGHT_RUDDER_MAX_DEG),
            "rudder_zero_share": float(STRAIGHT_RUDDER_ZERO_SHARE),
            "thrust_range": [float(v) for v in STRAIGHT_THRUST_RANGE],
            "surge_per_thrust": float(ship.straight_surge_per_thrust),
            "surge_spread": [float(v) for v in STRAIGHT_SURGE_SPREAD],
            "init_sway_range": [float(v) for v in STRAIGHT_INIT_SWAY_RANGE],
            "init_yaw_rate_range": [float(v) for v in STRAIGHT_INIT_YAW_RATE_RANGE],
            "calm_share": float(STRAIGHT_CALM_SHARE),
            "no_current_share": float(STRAIGHT_NO_CURRENT_SHARE),
            "heading_step_deg": float(STRAIGHT_HEADING_STEP_DEG),
            "lookahead_m": float(straight_lookahead(horizon_seconds, current_margin)),
            "current_margin": float(current_margin),
            "heading_choice": "uniform_among_sufficient_v2",
        },
        "exercise": exercise_path.name,
        "exercise_no_current": exercise_path_no_current.name if uses_no_current else None,
        "math_model_frequency": float(DEFAULT_MATH_MODEL_FREQUENCY),
        "initialization_time": float(INITIALIZATION_TIME),
        "wind_warmup_time": float(WIND_WARMUP_TIME) if conf.wind_current else None,
        "min_keel_clearance": float(MIN_KEEL_CLEARANCE),
        "min_bank_distance": float(MIN_BANK_DISTANCE),
        "cutoff_clearance": float(CUTOFF_CLEARANCE),
        "cutoff_segments": int(cutoff_n),
        "cutoff_length_m": round(float(cutoff_length), 1),
        "time_start_range": ([int(v) for v in CURRENT_TIME_START_RANGE]
                             if conf.randomize_time_start else None),
        "rudder_max_deg": float(rudder_max_deg),
        "rudder_scale_source": "profile" if fixed_rudder_max is not None else "fh_getMaximumAngle",
        "propeller_rps_scale": float(_PROPELLER_RPS_SCALE),
        "reverse_thrust": bool(ship.reverse_thrust),
        "thrust_range": [float(v) for v in thrust_range],
        "actuator_settle": {"max_seconds": float(ACTUATOR_SETTLE_MAX_SECONDS),
                            "tol_deg": float(ACTUATOR_SETTLE_TOL_DEG)},
        "init_surge_range": [float(v) for v in INIT_SURGE_RANGE],
        "init_sway_range": [float(v) for v in INIT_SWAY_RANGE],
        "init_yaw_rate_range": [float(v) for v in INIT_YAW_RATE_RANGE],
        "wind_scale_sigma": float(WIND_SCALE_SIGMA) if conf.wind_current else None,
        "wind_scale_max": float(WIND_SCALE_MAX) if conf.wind_current else None,
        "ray_max_distance": float(RAY_MAX_DISTANCE),
        "ray_angles_deg": [int(v) for v in RAY_ANGLES_DEG],
        "ray_layout": RAY_LAYOUT_VERSION,
        "fh_heading_convention": FH_HEADING_CONVENTION,
        "n_x_features": int(n_x),
        "n_y_features": int(NBR_Y_FEATURES),
        # Simulator input files (timeStart masked); the simexe binaries are not covered.
        "input_files": {
            "exercise": _file_fingerprint(exercise_path),
            "exercise_no_current": _file_fingerprint(exercise_path_no_current) if uses_no_current else None,
            "config": _file_fingerprint(sim_config.config_path),
            "current_xml": _file_fingerprint(CURRENT_XML_PATH) if conf.randomize_time_start else None,
            "tide_xml": _file_fingerprint(TIDE_XML_PATH) if conf.randomize_time_start else None,
            # Read by the no-current exercises (see STATIC_TIDE_XML_PATH); None if not used or not present.
            "static_tide_xml": (_file_fingerprint(STATIC_TIDE_XML_PATH)
                                if uses_static_tide and STATIC_TIDE_XML_PATH.exists() else None),
        },
    }
    # Only when used, so the signature (and resuming) of runs without them is unchanged.
    if bank_straight_fraction > 0:
        run_signature["bank_straight_fraction"] = float(bank_straight_fraction)
        run_signature["bank_straight"] = {
            "band": [bs_min, bs_max],
            "heading_jitter_deg": float(BANK_STRAIGHT_HEADING_JITTER_DEG),
            "heading_choice": "bank_tangent_v1",
        }
    n_resumed = 0
    chunks_done = 0    # including earlier sessions
    if checkpointing:
        meta_path = ckpt_dir / "meta.json"
        if ckpt_dir.exists() and not resume:
            print(f"[Checkpoint] resume=False: removing existing checkpoint directory {ckpt_dir}")
            shutil.rmtree(ckpt_dir)
        if ckpt_dir.exists():
            if not meta_path.exists():
                raise ValueError(f"Checkpoint directory {ckpt_dir} has no meta.json. Remove it "
                                 "(or pass resume=False) to start over.")
            saved_meta = json.loads(meta_path.read_text())
            saved_signature = saved_meta.get("signature", saved_meta)
            if saved_signature != run_signature:
                raise ValueError(
                    f"Checkpoint in {ckpt_dir} was made with different run parameters:\n"
                    f"  checkpoint: {saved_signature}\n  this run:   {run_signature}\n"
                    "Use the same parameters to resume, or pass resume=False (deletes the "
                    "checkpoint) / choose another output_file to start a new run.")
            resumed, n_resumed, chunks_done = _load_checkpoint_shards(ckpt_dir)
            if n_resumed > len(setups):
                raise ValueError(f"Checkpoint holds {n_resumed} episodes but this run has only "
                                 f"{len(setups)} start points.")
            for index, reason, start_xy, start_type, x_ep, y_ep, traj_ep in resumed:
                xy[index] = start_xy
                start_types[index] = start_type
                _record_episode(index, reason, start_xy, start_type, x_ep, y_ep, traj_ep)
            if n_resumed:
                print(f"[Checkpoint] Resuming: {n_resumed} of {len(setups)} episodes restored from "
                      f"{ckpt_dir} ({counter} with rows); continuing at episode {n_resumed}.")
                if seed is None:
                    print("[Checkpoint] Note: no seed given, so the remaining episodes use new "
                          "random start points, actions and tidal offsets.")
                saved_workers = saved_meta.get("num_workers")
                if seed is not None and saved_workers is not None and saved_workers != num_workers:
                    print(f"[Checkpoint] Warning: the checkpoint was made with {saved_workers} "
                          f"workers, this session uses {num_workers}. The data stays valid, but "
                          "the remaining episodes get different tidal offsets than in an "
                          "uninterrupted run, so the dataset is not identical to one.")
            # Replay the tide draws of the chunks already done.
            if tide_rng is not None and conf.randomize_time_start:
                low, high = CURRENT_TIME_START_RANGE
                for _ in range(chunks_done):
                    tide_rng.integers(low, high, endpoint=True)
        else:
            ckpt_dir.mkdir(parents=True)
            # num_workers is not part of the signature (another count only warns).
            _atomic_write_text(meta_path, json.dumps({"signature": run_signature,
                                             "num_workers": int(num_workers)}, indent=2))
        print(f"[Checkpoint] Saving progress every {checkpoint_every} episodes to {ckpt_dir}")

    last_saved = n_resumed   # episodes 0..last_saved-1 are saved
    chunks_since_save = 0

    def _save_checkpoint(upto: int) -> None:
        """Write episodes last_saved..upto-1 to a new shard."""
        nonlocal last_saved, chunks_since_save
        if not checkpointing or upto <= last_saved:
            return
        path = _write_checkpoint_shard(ckpt_dir, list(range(last_saved, upto)), episode_meta,
                                       episode_results, chunks_since_save, n_x)
        print(f"[Checkpoint] Episodes {last_saved}-{upto - 1} saved to {path.name}")
        last_saved = upto
        chunks_since_save = 0

    # ---- Early stop: stop file or Ctrl+C ----
    stop_path = Path(stop_file) if stop_file else out_dir / "STOP_COLLECTION"
    if stop_path.exists():
        print(f"Removing stale stop file {stop_path}")
        stop_path.unlink()

    stop_requested = [False]
    previous_sigint = None

    def _on_sigint(signum, frame):
        """
        First Ctrl+C: request a graceful stop after the current chunk and restore the previous
        SIGINT handler, so a second Ctrl+C aborts without saving.
        """

        stop_requested[0] = True
        signal.signal(signal.SIGINT, previous_sigint)
        print("\n[Stop] Ctrl+C received: finishing the current chunk, then saving what "
              "was collected. Press Ctrl+C again to abort WITHOUT saving.")

    try:
        previous_sigint = signal.signal(signal.SIGINT, _on_sigint)
    except ValueError:  # not in the main thread: only the stop file works
        previous_sigint = None

    t_start = time.monotonic()

    def _check_stop() -> Optional[str]:
        """Reason to stop early ("Ctrl+C" or the stop file), or None to continue."""

        if stop_requested[0]:
            return "Ctrl+C"
        if stop_path.exists():
            return f"stop file {stop_path}"
        return None

    print(f"To stop gracefully after the current chunk: create '{stop_path}'"
          + (" or press Ctrl+C once." if previous_sigint is not None else "."))

    early_stop_reason = None
    n_attempted = n_resumed  # episodes of completed chunks

    # Replaceable pool: a timed-out simulation keeps its worker busy, so the
    # pool is restarted after a chunk with a TIMEOUT.
    pool_holder = [multiprocessing.Pool(num_workers, initializer=_ignore_sigint)]

    def execute_with_param(_setup):
        """Run one simulation setup in the current worker pool (pool_holder[0], which may be replaced)."""
        return run_in_process(_setup, sim_config=sim_config, pool=pool_holder[0])

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:

            # Chunks of num_workers episodes. Each chunk must finish completely
            # before the shared tide XMLs are rewritten for the next one (simexe
            # reads them itself). Checkpoints and early stop happen at this
            # barrier, when nothing is running.
            for chunk_start in range(n_resumed, len(setups), num_workers):
                chunk = setups[chunk_start:chunk_start + num_workers]
                chunk_end = chunk_start + len(chunk) - 1

                if conf.randomize_time_start:
                    new_time_start = randomize_current_time_start(rng=tide_rng)
                    print(f"timeStart randomized to {new_time_start}s for simulations "
                          f"{chunk_start}-{chunk_end} in:")
                    for _xml_path in TIME_START_XML_PATHS:
                        print(f"  - {_xml_path}")
                else:
                    print(f"Executing simulations {chunk_start}-{chunk_end}:")

                future_to_test = {
                    executor.submit(execute_with_param, setup): (chunk_start + i, setup)
                    for i, setup in enumerate(chunk)
                }

                chunk_timed_out = False
                for future in as_completed(future_to_test):
                    spawn_counter += 1
                    index, _ = future_to_test[future]

                    try:
                        sim_setup = future.result()

                        if sim_setup.output.status == "TIMEOUT":
                            chunk_timed_out = True
                        reason = sim_setup.output.info.get("termination") or sim_setup.output.status or "FAILED"

                        if sim_setup.output.status == "PASSED":
                            # Assembled in episode order after the loop (reproducible shuffle).
                            _record_episode(index, reason, xy[index], str(start_types[index]),
                                            sim_setup.output.x_input,
                                            sim_setup.output.y_output,
                                            sim_setup.output.trajectory)
                            print(f"Simulation {index} succeeded: {len(sim_setup.output.x_input)} rows "
                                  f"(ended: {sim_setup.output.info.get('termination')}, actuators settled in "
                                  f"{sim_setup.output.info.get('settle_seconds', 0.0):.1f} s)")
                        else:
                            _record_episode(index, reason, xy[index], str(start_types[index]))
                            # "error": explicit checks; "message": exceptions.
                            status = sim_setup.output.status or "FAILED"
                            detail = (
                                sim_setup.output.info.get("error")
                                or sim_setup.output.info.get("message")
                            )
                            if detail:
                                print(f"Simulation {index} failed [{status}]: {detail}")
                            else:
                                print(f"Simulation {index} failed [{status}]")
                    except Exception as exc:
                        _record_episode(index, "EXCEPTION", xy[index], str(start_types[index]))
                        print(f"Simulation {index} generated an exception: {exc}")

                # Barrier: all simulations of this chunk are done.
                n_attempted = chunk_end + 1
                chunks_done += 1
                chunks_since_save += 1

                if spawn_counter // CACHE_CLEAR_INTERVAL > cache_clears_done:
                    cache_clears_done = spawn_counter // CACHE_CLEAR_INTERVAL
                    print(f"[Cache cleanup] {spawn_counter} simulations finished - clearing {CACHE_FOLDER} …")
                    clear_cache_folder(CACHE_FOLDER)

                if checkpointing and n_attempted - last_saved >= checkpoint_every:
                    _save_checkpoint(n_attempted)

                if n_attempted < len(setups):
                    early_stop_reason = _check_stop()
                    if early_stop_reason:
                        elapsed = time.monotonic() - t_start
                        print(f"[Stop] {early_stop_reason}: stopping after {n_attempted} of "
                              f"{len(setups)} episodes ({elapsed:.0f} s). Saving collected data.")
                        break

                if chunk_timed_out and n_attempted < len(setups):
                    print(f"[Pool restart] a simulation in chunk {chunk_start}-{chunk_end} timed out; "
                          f"terminating its worker processes and starting a fresh pool.")
                    pool_holder[0].terminate()
                    pool_holder[0].join()
                    pool_holder[0] = multiprocessing.Pool(num_workers, initializer=_ignore_sigint)
    finally:
        pool_holder[0].terminate()
        pool_holder[0].join()
        # Save the completed chunks not saved yet (also after an abort or error).
        try:
            _save_checkpoint(n_attempted)
        except Exception as exc:  # never mask the original exception
            print(f"Warning: could not write the final checkpoint: {exc}")
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        stop_path.unlink(missing_ok=True)

    attempted_types = start_types[:n_attempted]
    if early_stop_reason:
        print(f"Run stopped early ({early_stop_reason}).")
    print(f"Episodes with usable rows: {counter} of {n_attempted} run "
          f"({nbr_samples} planned)")
    print("Episode endings (horizon = ran full length; bank / ukc / left_open_water / near_cutoff ="
          " stopped at a limit with rows kept; start_* / EMPTY = no rows):")
    for reason in sorted(termination_counts):
        rows = rows_per_termination.get(reason, 0)
        print(f"  {reason:<22} {termination_counts[reason]:>6} episodes  {rows:>9} rows")
    print("By start type (episodes with rows / started, rows):")
    for st in all_start_types:
        print(f"  {st:<22} {kept_per_start_type[st]:>6} / {int((attempted_types == st).sum())} episodes  "
              f"{rows_per_start_type[st]:>9} rows")

    for index in sorted(episode_results):
        x_ep, y_ep, traj_ep = episode_results[index]
        x_inputs.append(x_ep)
        y_outputs.append(y_ep)
        episode_ids.append(np.full(len(x_ep), index, dtype=np.int64))
        trajectories.append(traj_ep)

    if counter == 0 or not x_inputs:
        print("Warning: No successful simulations!")
        _warn_if_stale_csv(output_file)
        return _empty_result(n_x)

    x_values = np.vstack(x_inputs).astype(np.float32)
    y_values = np.vstack(y_outputs).astype(np.float32)
    episode_ids = np.concatenate(episode_ids)

    # Shuffle rows (still split by episode_id, not by row).
    perm = shuffle_rng.permutation(len(x_values))
    x_values = x_values[perm]
    y_values = y_values[perm]
    episode_ids = episode_ids[perm]

    store_data = np.hstack([x_values, y_values, episode_ids.reshape(-1, 1).astype(np.float32)])
    # %.9g round-trips float32 exactly (also tiny deltas).
    _atomic_savetxt(output_file, store_data, delimiter=",", fmt="%.9g")
    print(f"Dataset written to {output_file} ({len(store_data)} rows).")
    # Remove the checkpoint after a complete run.
    if checkpointing and n_attempted >= len(setups):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"[Checkpoint] Run complete: removed {ckpt_dir}")
    elif checkpointing:
        print(f"[Checkpoint] Run incomplete ({n_attempted} of {len(setups)} episodes): checkpoint "
              f"kept in {ckpt_dir}. Run again with the same parameters to continue.")
    plot_xy_points_and_trajectories(env, xy[:n_attempted], trajectories,
                                    out_dir / "generated_points.png", show=show_plot)

    return x_values, y_values, episode_ids, counter


if __name__ == "__main__":
    x_vals, y_vals, ep_ids, successful_count = collect_supervised_dataset(
        nbr_samples=10000,
        horizon_seconds=5.0,
        dt=0.1,
        workers=max(1, multiprocessing.cpu_count() - 1),
        ship_model=DEFAULT_SHIP_MODEL,
        configuration=DEFAULT_CONFIGURATION,
    )

    print(f"Dataset generation complete. Successful samples: {successful_count}")
    print("Dataset shapes:")
    print(f"X: {x_vals.shape}")
    print(f"Y: {y_vals.shape}")
    print(f"episode_ids: {ep_ids.shape}")

    if successful_count > 0:
        np.save("ship_dynamics_inputs.npy", x_vals)
        np.save("ship_dynamics_outputs.npy", y_vals)
        np.save("ship_dynamics_episode_ids.npy", ep_ids)
        print("Dataset saved to ship_dynamics_inputs.npy, ship_dynamics_outputs.npy "
              "and ship_dynamics_episode_ids.npy")
