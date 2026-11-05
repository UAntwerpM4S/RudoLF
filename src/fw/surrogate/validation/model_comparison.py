"""
Ship-bank test: compares dynamics models against FH (ground truth) for one ship
(SHIP_MODEL: TRIPOLI or MYZAKO_37) sailing parallel to a river bank, over a grid
of (initial surge, distance to the bank).

Models:
    FH          the simexe engine (FhSession, stepped via FhTabularModel.step())
    UGent       UGentSurrogate (Tripoli only)
    Fossen      Fossen
    UA_clean, UA_confined, UA_full
                UAntwerpSurrogate, one preset per slot (UA_PRESETS_BY_SHIP);
                slots without a preset or checkpoint are left out.

Per grid point the ship is placed at the given distance from the port bank,
parallel to it, with rudder 0 and thrust min(1, v/6). After FH settles, the
"before" state and FH's live UKC, wind and current are read; FH is advanced by
SURROGATE_DT and its (du, dv, dr) is finite-differenced. All other models are
evaluated once at the "before" state (_compute_accelerations()).

The UA_* models get FH's live UKC/wind/current and rays cast on the same
PySimEnv polygon (_inject_live_environment), so a bad result points at their
learned dynamics, not at a lookup or geometry mismatch. The rudder scale comes
from each checkpoint; the rudder angle is 0 throughout. States outside the
dataset generator's stop limits are flagged as extrapolation
(_training_domain_issue, diagnostic only).

This module is also the shared base of model_comparison_wind.py and
model_comparison_current.py (FhSession, _inject_live_environment, ...).

FhSession builds FH exactly as fw.main does (env_factory.FhSimEnvironment and
PySimEnv). If MathModel.Initialize() fails with "Cannot open input file:
.../os2_body", look outside FhSession (working directory, environment
variables, imports); see the diagnose_* functions.

Caveats:
    1. FH's (x, y) frame is assumed to equal that of the PySimEnv polygon. On a
       first run, plot the located points against the polygon.
    2. Run a small grid first.

Usage: run_comparison(seed_point) with an (x, y) point near the port bank in
ScheldeSaeftinge_23_002 (see the __main__ block).
"""

import csv
import dataclasses
import logging

import numpy as np

from pathlib import Path
from typing import Optional, Sequence, Tuple

from fw.ships.ship_factory import create_ship, ShipModel
from fw.ships.vessel_state import VesselState
from fw.dynamics.ugent_surrogate import UGentSurrogate
from fw.dynamics.fossen import Fossen
from fw.dynamics.fh_tabular_model import FhTabularModel, fh_rudder_sign
from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate
from fw.surrogate.presets import PRESETS
from fw.dynamics.wcu_lookup import EnvironmentData
from fw.simulators.fh_sim_bridge import FhSimBridge
from fw.surrogate.schema import SURROGATE_DT
from fw.simulators.fh_interface_utils import fh_heading_to_env_rad, get_current_velocity
from fw.surrogate.sim_fh_dataset_generator import (
    _PROPELLER_RPS_SCALE,
    _cutoff_geometry,
    stop_reason,
)
from fw.environments.py_sim_env import PySimEnv
# Shared ray implementation (ray_0 = bow, clockwise), as used by the dataset
# generator and the UA_* surrogates.
from fw.environments.ray_features import (
    RAY_ANGLES_DEG,
    RAY_MAX_DISTANCE,
    compute_ray_features,
    ray_world_angles,
)

# Trajectories of the non-FH models are integrated with the production
# ShipSimulator (no hand-rolled integration), which adds no wind or current.
from fw.simulators.ship_simulator import ShipSimulator
from fw.ships.actuator_command import ActuatorCommand

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except ImportError:
    # Both names defined to satisfy static checkers. Without plt,
    # generate_trajectories writes CSVs only (no PNG).
    matplotlib = None
    plt = None

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Grid and fixed run parameters
# --------------------------------------------------------------------------

V_VALUES = [1, 2, 3, 4, 5, 6]        # initial surge speed [m/s]
DISTANCES = (35.0, 30.0, 25.0, 20.0, 15.0, 11.0)  # perpendicular distance to bank [m]
# DISTANCES = (200.0, 150.0, 100.0, 75.0, 50.0, 35.0, 30.0, 25.0, 20.0, 15.0, 11.0)  # perpendicular distance to bank [m]

RUDDER_ANGLE_RAD = 0.0               # fixed: ship sails parallel to the bank
# SURROGATE_DT (0.1 s) comes from fw.surrogate.schema; the wind/current tests
# import it from here.
INITIALIZATION_TIME = 0.125          # informational: FhSimBridge's hardcoded settle time.
                                      # The dataset generator settles for one 0.1 s step;
                                      # only the starting transient differs.

# Output folder of this script's CSVs (not passed to FhSimBridge, which keeps its
# default output_path). Resolved against the working directory at import time.
OUTPUT_DIR = Path("model_comparison_output").resolve()

# Ship simulated by FH (also used for ship_spec and ray_env). FhSimBridge picks
# the nowindnocurrent exercise. UGent runs for the Tripoli only; UA_* slots for
# every ship with a preset.
SHIP_MODEL = ShipModel.MYZAKO_37

# All UA_* model slots, in report order.
UA_LABELS = ("UA_clean", "UA_confined", "UA_full")

# Per ship: UA_* slot to UAntwerp surrogate preset (fw/surrogate/presets.py).
# Shared with the wind and current tests.
UA_PRESETS_BY_SHIP = {
    ShipModel.TRIPOLI: {
        "UA_clean": "tripoli_clean",
        "UA_confined": "tripoli_confined",
        "UA_full": "tripoli_full",
    },
    ShipModel.MYZAKO_37: {
        "UA_clean": "myzako_clean",
        "UA_confined": "myzako_confined",
        "UA_full": "myzako_full",
    },
}


def ua_presets() -> dict:
    """
    UA_* slot to preset name for SHIP_MODEL (empty if it has none).
    """

    return UA_PRESETS_BY_SHIP.get(SHIP_MODEL, {})


# Surge limits [m/s] of the ShipSimulator for the non-FH models in all three
# tests; None = the ship's own limits. FH is never clipped; steps at the upper
# limit are reported.
SIM_SURGE_LIMITS: Optional[Tuple[float, float]] = (0.0, 8.0)


def sim_surge_limits(ship_spec) -> Tuple[float, float]:
    """
    Surge limits the model simulators use (SIM_SURGE_LIMITS or the ship's own).
    """

    limits = SIM_SURGE_LIMITS if SIM_SURGE_LIMITS is not None else ship_spec.surge_limits
    lo, hi = float(limits[0]), float(limits[1])
    if not lo < hi:
        raise ValueError(f"SIM_SURGE_LIMITS must be (low, high) with low < high, got {limits}.")
    return lo, hi


def make_model_simulator(ship_spec, model) -> ShipSimulator:
    """
    ShipSimulator for a non-FH model, without added wind/current. Only the
    simulator's clipping uses sim_surge_limits; the model keeps ship_spec.
    """

    specs = dataclasses.replace(ship_spec, surge_limits=sim_surge_limits(ship_spec))
    return ShipSimulator(specs=specs, dynamics=model, dt=SURROGATE_DT)


class SurgeClipCounter:
    """
    Counts, per model, the steps whose surge sits at the simulator's upper surge limit.
    """

    def __init__(self, ship_spec, labels):
        """
        Zero counts per label; limits from sim_surge_limits(ship_spec).
        """

        self.limits = sim_surge_limits(ship_spec)
        self.counts = {label: 0 for label in labels}

    def update(self, label: str, u: float) -> None:
        """
        Count a step for label if u is at the upper limit.
        """

        # Upper limit only: a surge clipped at 0 is indistinguishable from a stopped ship.
        if u >= self.limits[1] - 1e-9:
            self.counts[label] += 1

    def report(self, run_id: str) -> None:
        """
        Warn about the models that were clipped in run_id.
        """

        clipped = {label: n for label, n in self.counts.items() if n}
        if clipped:
            logger.warning("%s: surge at the simulator's upper limit (%g m/s) in %s step(s) -- these models "
                           "were clipped, FH is not (set model_comparison.SIM_SURGE_LIMITS to widen).",
                           run_id, self.limits[1], clipped)


# Models that exist for the Tripoli only (UA_* slots depend on UA_PRESETS_BY_SHIP).
TRIPOLI_ONLY_MODELS = ("UGent",)
_skip_warned: set = set()


def tripoli_models_enabled() -> bool:
    """
    True if SHIP_MODEL is the Tripoli (Tripoli-only models apply).
    """

    return SHIP_MODEL == ShipModel.TRIPOLI


def _note_models_skipped(labels) -> None:
    """
    Warn once per model that it is left out for SHIP_MODEL.
    """

    new = [label for label in labels if label not in _skip_warned]
    if new:
        logger.warning("SHIP_MODEL=%s: no %s for this ship -- left out.", SHIP_MODEL.name, ", ".join(new))
        _skip_warned.update(new)


def _note_tripoli_models_skipped() -> None:
    """
    Warn once that the Tripoli-only models are left out.
    """

    _note_models_skipped(TRIPOLI_ONLY_MODELS)


def build_reference_models(ship_spec, ray_env) -> dict:
    """
    Reference models: UGent (Tripoli only; bank distances from ray_env) and Fossen.
    """

    models = {}
    if tripoli_models_enabled():
        ugent = UGentSurrogate(ship_spec, {"ukc_percent": 363})
        ugent.set_bank_distance_provider(ray_env.bank_beam_distances)
        models["UGent"] = ugent
    else:
        _note_tripoli_models_skipped()
    models["Fossen"] = Fossen(ship_spec)
    return models


def build_ua_models(ship_spec, labels: Optional[Sequence[str]] = None) -> dict:
    """
    One UAntwerpSurrogate per label (default: all UA_LABELS), in order.

    Labels without a preset for SHIP_MODEL, with an undefined preset or with a
    missing checkpoint file are left out with a warning.

    Raises:
        ValueError: For an unknown label, or a checkpoint that does not match its
            preset or SHIP_MODEL.
    """

    wanted = UA_LABELS if labels is None else tuple(labels)
    unknown = [label for label in wanted if label not in UA_LABELS]
    if unknown:
        raise ValueError(f"Unknown UA model label(s) {unknown}; expected some of {UA_LABELS}.")
    presets = ua_presets()
    missing = [label for label in wanted if label not in presets]
    if missing:
        _note_models_skipped(missing)
    models = {}
    for label in wanted:
        if label not in presets:
            continue
        if presets[label] not in PRESETS:
            logger.warning("%s left out: preset '%s' is not defined in fw/surrogate/presets.py "
                           "(an older presets.py?). Defined: %s.", label, presets[label], sorted(PRESETS))
            continue
        try:
            models[label] = UAntwerpSurrogate(ship_spec, {"preset": presets[label]}, ship_model=SHIP_MODEL)
        except FileNotFoundError as exc:
            logger.warning("%s left out: %s", label, exc)
    return models

# A bank_point closer than this to a map cut-off edge (see
# PySimEnv.bank_segments) is not on a real bank; _setup_geometry refuses it.
CUTOFF_BANK_TOLERANCE = 1.0          # m

# --------------------------------------------------------------------------
# Trajectory parameters: generate_trajectories fixes the speed and sweeps
# DISTANCES, to show how bank proximity alone changes each model's path.
# --------------------------------------------------------------------------

TRAJECTORY_SPEED = 3                  # initial surge speed [m/s] (wind test: WIND_TRAJECTORY_SPEED)
TRAJECTORY_TARGET_DISTANCE = 300.0    # [m] stop once FH's path length reaches this; every
                                       # model runs the same number of steps as FH
TRAJECTORY_MAX_STEPS = 8000           # safety cap (800 s) if FH never reaches the target
TRAJECTORY_OUTPUT_DIR = OUTPUT_DIR / "trajectories"

# Same palette as plot_comparison.py, so model colors match across plots.
TRAJECTORY_COLORS = {
    "FH": "#2a78d6",
    "UGent": "#eb6834",
    "Fossen": "#1baf7a",
    "UA_clean": "#eda100",
    "UA_confined": "#e87ba4",
    "UA_full": "#008300",
}

# Distinct dash pattern per model, so overlapping trajectories stay visible.
TRAJECTORY_LINESTYLES = {
    "FH": "-",
    "UGent": (0, (1, 1)),          # dotted
    "Fossen": (0, (5, 2)),         # dashed
    "UA_clean": (0, (3, 1, 1, 1)), # dash-dot
    "UA_confined": (0, (5, 1, 1, 1, 1, 1)),  # dash-dot-dot
    "UA_full": (0, (2, 2, 6, 2)),  # long-short dash
}


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _shapely_point(xy):
    """
    shapely Point for an (x, y) pair.
    """

    from shapely.geometry import Point
    return Point(float(xy[0]), float(xy[1]))


def find_bank_reference(
        env: PySimEnv,
        seed_point: Tuple[float, float],
        step: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the bank point nearest seed_point and the unit normal into open water.

    The ship heading is not derived here (callers use ray_env.ship_heading).

    Args:
        env: PySimEnv whose polygon_shape defines the navigable water.
        seed_point: (x, y) near the port bank (needed: the boundary has two
            banks and possibly islands).
        step: Arc-length step [m] for the finite-difference tangent.

    Returns:
        (bank_point, perp_into_water): the (2,) boundary point and the (2,) unit
        normal into the water, verified with polygon_shape.contains().
    """

    boundary = env.polygon_shape.boundary
    seed = _shapely_point(seed_point)

    s = boundary.project(seed)
    length = boundary.length
    s_fwd = min(s + step, length)
    s_bwd = max(s - step, 0.0)

    p0 = boundary.interpolate(s)
    p_fwd = boundary.interpolate(s_fwd)
    p_bwd = boundary.interpolate(s_bwd)

    bank_point = np.array([p0.x, p0.y])
    tangent = np.array([p_fwd.x - p_bwd.x, p_fwd.y - p_bwd.y])
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-9:
        raise RuntimeError(
            f"Degenerate boundary tangent at seed_point={seed_point} "
            f"(projected arc length s={s}); try a different seed_point."
        )
    tangent = tangent / tangent_norm

    # Candidate normal (tangent rotated by -90 deg), flipped if it points out of
    # the water (polygon_shape.contains() is True in open water).
    perp = np.array([tangent[1], -tangent[0]])

    probe = bank_point + perp * 1.0
    if not env.polygon_shape.contains(_shapely_point(probe)):
        perp = -perp

    return bank_point, perp


def locate_point_at_distance(
        env: PySimEnv,
        bank_point: np.ndarray,
        perp_into_water: np.ndarray,
        target_distance: float,
        tol: float = 0.1,
        max_iter: int = 40,
) -> np.ndarray:
    """
    Bisect along perp_into_water for the point whose ray-cast distance back to
    the bank equals target_distance (ray casting, so it matches the ray
    features). Assumes the distance grows monotonically with the offset; a
    spot-check warns otherwise. Targets beyond RAY_MAX_DISTANCE use a longer
    measuring ray (the UA rays then no longer see this bank).

    Args:
        env: PySimEnv (see find_bank_reference).
        bank_point: (2,) point on the bank.
        perp_into_water: (2,) unit normal into the water.
        target_distance: Desired distance to the bank [m].
        tol: Distance tolerance [m].
        max_iter: Maximum bisection iterations.

    Returns:
        (2,) point at about target_distance from the bank.

    Raises:
        RuntimeError: Target not bracketed or no convergence; callers skip that
            distance.
    """

    to_bank = -perp_into_water
    # Measuring range: the sensor range, longer only for targets beyond it.
    reach = RAY_MAX_DISTANCE if target_distance <= RAY_MAX_DISTANCE else 2.0 * float(target_distance)

    def measured_distance(pos: np.ndarray) -> float:
        """
        Ray-cast distance [m] from `pos` toward the bank, capped at `reach`.
        """

        origin = (float(pos[0]), float(pos[1]))
        end = (float(pos[0] + to_bank[0] * reach),
               float(pos[1] + to_bank[1] * reach))
        return float(env.ray_cast_distance(origin, end, max_distance=reach))

    lo, hi = 0.5, reach
    # Stop the search just before the opposite boundary, so the bisection cannot
    # settle on land (the waterway may be narrower than reach).
    start = bank_point + perp_into_water * lo
    far_end = start + perp_into_water * reach
    width = float(env.ray_cast_distance((float(start[0]), float(start[1])),
                                        (float(far_end[0]), float(far_end[1])), max_distance=reach))
    hi = min(hi, lo + 0.999 * width)
    pos_lo = bank_point + perp_into_water * lo
    pos_hi = bank_point + perp_into_water * hi
    d_lo = measured_distance(pos_lo)
    d_hi = measured_distance(pos_hi)

    if not (d_lo <= target_distance <= d_hi):
        raise RuntimeError(
            f"target_distance={target_distance} not bracketed by "
            f"[{d_lo}, {d_hi}] (offsets [{lo}, {hi}] from bank_point); "
            f"cannot locate a point at this distance"
            + (" (is the opposite bank closer than that?)." if target_distance > d_hi else ".")
        )

    d_mid = float("nan")   # stays NaN only if max_iter < 1
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pos_mid = bank_point + perp_into_water * mid
        d_mid = measured_distance(pos_mid)
        if abs(d_mid - target_distance) <= tol:
            # Monotonicity spot-check. Warning only: pos_mid meets the
            # tolerance anyway, and false positives near a boundary are likely.
            probe_step = max(tol, 0.5)
            d_below = measured_distance(bank_point + perp_into_water * max(mid - probe_step, lo))
            d_above = measured_distance(bank_point + perp_into_water * min(mid + probe_step, hi))
            if not (d_below <= d_mid <= d_above):
                print(
                    f"locate_point_at_distance: monotonicity spot-check "
                    f"failed near offset={mid:.3f} (target_distance="
                    f"{target_distance}) -- measured_distance is not "
                    f"increasing with offset here (d_below={d_below:.3f}, "
                    f"d_mid={d_mid:.3f}, d_above={d_above:.3f}). The "
                    f"returned point still satisfies the distance "
                    f"tolerance, but may not be the point closest to the "
                    f"bank at this distance -- treat this grid point's "
                    f"results with extra care.",
                    flush=True,
                )
            return pos_mid
        if d_mid < target_distance:
            lo = mid
        else:
            hi = mid

    raise RuntimeError(
        f"locate_point_at_distance did not converge within {max_iter} "
        f"iterations for target_distance={target_distance} "
        f"(last measured distance was {d_mid})."
    )


def _print_bank_ray_diagnostics(
        ua_models: dict,
        pos: np.ndarray,
        heading_deg: float,
        perp_into_water: np.ndarray,
        ray_env: PySimEnv,
        expected_distance: float,
        log_prefix: str,
) -> None:
    """
    Print the ray features closest to the bank and to open water in metres, for
    comparison with expected_distance (pure geometry check).

    ray_k points k degrees clockwise from the bow; with the ship parallel to the
    bank, ray_90 or ray_270 should face it. The closest ray is printed with its
    angular offset, next to a ray cast along the exact normal. Nothing is
    printed if no model uses rays.
    """

    if not any(hasattr(model, "_env") for model in ua_models.values()):
        return  # no ray-based model

    to_bank = -perp_into_water
    heading_rad = float(np.radians(heading_deg))
    ray_angles = ray_world_angles(heading_rad)
    ray_dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=1)  # (24, 2)

    toward_idx = int(np.argmax(ray_dirs @ to_bank))
    away_idx = int(np.argmax(ray_dirs @ perp_into_water))

    origin = (float(pos[0]), float(pos[1]))

    def _report(label: str, tag: str, feature: float, idx: int, ref_direction: np.ndarray) -> None:
        """
        Print a ray's feature, implied distance, angular offset and ground-truth distance.
        """

        angle_deg = int(RAY_ANGLES_DEG[idx])
        ray_distance = RAY_MAX_DISTANCE * (1.0 - float(feature))
        angular_offset_deg = float(np.degrees(
            np.arccos(np.clip(ray_dirs[idx] @ ref_direction, -1.0, 1.0))
        ))
        end = (
            float(pos[0] + ref_direction[0] * RAY_MAX_DISTANCE),
            float(pos[1] + ref_direction[1] * RAY_MAX_DISTANCE),
        )
        ground_truth = float(ray_env.ray_cast_distance(origin, end, max_distance=RAY_MAX_DISTANCE))
        print(
            f"{log_prefix}: ray diagnostic [{label}] {tag}: ray_{angle_deg} "
            f"(off by {angular_offset_deg:.1f} deg from the exact direction) "
            f"feature={feature:.4f} -> distance={ray_distance:.2f} m "
            f"(exact-direction ground truth={ground_truth:.2f} m, "
            f"expected~={expected_distance:.2f} m)",
            flush=True,
        )

    features = compute_ray_features(ray_env, np.asarray(pos, dtype=float), heading_rad)
    _report("rays", "toward bank", features[toward_idx], toward_idx, to_bank)
    _report("rays", "away from bank", features[away_idx], away_idx, perp_into_water)


def _training_domain_issue(
        ray_env: PySimEnv,
        cutoff_geom,
        x: float,
        y: float,
        heading_rad: float,
        ukc: float,
) -> Optional[str]:
    """
    Check a state against the dataset generator's stop limits (stop_reason).

    The generator records no rows in such states, so UA_* predictions there
    are extrapolation. Diagnostic only.

    Returns:
        None inside the training domain, else the stop reason
        ("left_open_water", "near_cutoff", "bank" or "ukc").
    """

    rays = compute_ray_features(ray_env, (float(x), float(y)), float(heading_rad))
    return stop_reason(ray_env, np.array([x, y], dtype=np.float64), float(ukc), rays, cutoff_geom)


# --------------------------------------------------------------------------
# FH session (thin wrapper around the real simexe engine)
# --------------------------------------------------------------------------

class FhSession:
    """
    One real FH (simexe) simulation for one grid point.

    Bootstrapping (Config/Exercise/MathModel, rewind, enableBridge, settle) is
    done by FhSimBridge, constructed as in PySimEnv: FhTabularModel wraps the
    bridge, initial_vessel_state is set, then reset() is called once. Controls
    and time stepping (advance()) go through FhTabularModel.step().

    Call close() when done, to release the native FH resources.
    """

    def __init__(
            self,
            x: float,
            y: float,
            heading_deg: float,
            v_init: float,
            thrust: float,
            ship_spec,
            exercise_path: Optional[Path] = None,
            wind_field: Optional[Tuple[float, float]] = None,
            rewind_enabled: Optional[bool] = None,
    ):
        """
        Create the FH bridge, set the initial state and controls, and reset it.

        Args:
            x: Start x [m].
            y: Start y [m].
            heading_deg: Start heading [deg], framework convention.
            v_init: Initial surge velocity [m/s].
            thrust: Normalized thrust; sets the initial propeller RPS and is applied on every advance().
            ship_spec: Ship specifications for FhTabularModel.
            exercise_path: Exercise file for FhSimBridge; None = FhSimBridge's default exercise.
            wind_field: Optional (rotation_deg, scale) wind-field transform, set before reset().
            rewind_enabled: Passed to FhSimBridge; None = FhSimBridge's default.

        Raises:
            FileNotFoundError: If `exercise_path` is given but does not exist.
            RuntimeError: If FhSimBridge lacks an initial-condition hook, or the exercise's ship lacks
                the controls FhTabularModel.step() drives. On any failure in reset() the bridge is
                disposed before the exception is re-raised.
        """

        # Construction mirrors env_factory.FhSimEnvironment + PySimEnv. Optional
        # arguments (wind/current tests): rewind_enabled (a session never rewinds, so
        # False is safe), wind_field (must be set before the first simulateSeconds()),
        # exercise_path.
        bridge_kwargs = {}
        if rewind_enabled is not None:
            bridge_kwargs["rewind_enabled"] = bool(rewind_enabled)
        if exercise_path is not None:
            exercise_path = Path(exercise_path)
            if not exercise_path.is_file():
                raise FileNotFoundError(f"FhSession: exercise_path does not exist: {exercise_path}")
            # str(): FhSimBridge passes this straight to the .NET Exercise(String, Config)
            # constructor, which pythonnet cannot call with a pathlib.Path.
            bridge_kwargs["exercise_path"] = str(exercise_path)
        self._bridge = FhSimBridge(
            ship_model=SHIP_MODEL,
            scenario="DDShip_scenario1_nowindnocurrent",
            **bridge_kwargs,
        )
        self._fh_model = FhTabularModel(ship_spec, fh_engine=self._bridge)

        self._bridge.initial_vessel_state = VesselState(
            terminated=False, truncated=False,
            x=float(x), y=float(y), heading=float(np.radians(heading_deg)),
            u=float(v_init), v=0.0, r=0.0,
        )
        # Initial propeller RPS consistent with thrust (as in the dataset generator).
        # Fail fast if FhSimBridge lacks the initial-condition hooks: FH would silently
        # keep the exercise defaults.
        for hook in ("initial_propeller_rps", "initial_rudder_angle", "initial_wind_field_transform"):
            if not hasattr(self._bridge, hook):
                raise RuntimeError(
                    f"FhSession: FhSimBridge has no '{hook}' hook -- the installed "
                    f"fw/simulators/fh_sim_bridge.py is an older version that would ignore it."
                )
        self._bridge.initial_propeller_rps = float(_PROPELLER_RPS_SCALE * thrust)
        # Rudder at RUDDER_ANGLE_RAD already during the settle (advance() only
        # applies it afterwards), in FH's sign convention.
        self._bridge.initial_rudder_angle = float(fh_rudder_sign(SHIP_MODEL) * np.degrees(RUDDER_ANGLE_RAD))
        if wind_field is not None:
            rotation_deg, scale = wind_field
            self._bridge.initial_wind_field_transform = (float(rotation_deg), float(scale), 0.0)
        # The first reset() initializes the math model and settles 0.125 s.
        try:
            self._bridge.reset()
            self._check_propulsion_controls()
        except Exception:
            # No FhSession object will exist, so nobody else can close() it.
            self._bridge.dispose()
            raise

        self._thrust_normalized = float(thrust)

    def _check_propulsion_controls(self) -> None:
        """
        Raise if the exercise's ship lacks the controls FhTabularModel.step()
        drives (azipods for the Tripoli, rudders and propellers otherwise);
        step() would silently apply nothing, e.g. with a mismatched exercise.
        """

        si = self._bridge.ship_interface
        ship_model = self._bridge.ship_model
        if ship_model == ShipModel.TRIPOLI:
            needed = {"azipod": len(si.getAzipodControls())}
        else:
            needed = {"rudder": len(si.getRudderControls()), "propeller": len(si.getPropellerControls())}
        missing = [name for name, n in needed.items() if n == 0]
        if missing:
            raise RuntimeError(
                f"FhSession: ship model {ship_model.name} needs {' and '.join(needed)} controls, but the "
                f"exercise's ship has no {' / '.join(missing)} controls -- rudder and thrust would not "
                f"be applied. Does the exercise ({getattr(self._bridge, 'exercise_path', '?')}) match SHIP_MODEL?"
            )

    @property
    def ship_interface(self):
        """
        FH ship interface of the underlying bridge.
        """

        return self._bridge.ship_interface

    def close(self) -> None:
        """
        Release the native FH resources (idempotent).
        """

        self._bridge.dispose()

    def state(self) -> Tuple[float, float, float, float, float, float]:
        """
        Return (x, y, heading_rad, u, v, r) in the framework convention, built as
        in sim_fh_dataset_generator: heading via fh_heading_to_env_rad, sway
        positive to port (FH's body frame is the framework's).
        """

        pos = self.ship_interface.getShipPosition()
        heading_deg = self.ship_interface.getShipHeading()
        vog = self.ship_interface.getShipVelocityOverGround()
        yaw_rate_deg = self.ship_interface.getShipYawRate()

        heading_rad = fh_heading_to_env_rad(heading_deg)
        return (
            float(pos.x), float(pos.y), float(heading_rad),
            float(vog.x), float(vog.y), float(np.radians(yaw_rate_deg)),
        )

    def advance(self, dt: float) -> None:
        """
        Apply the held controls (rudder RUDDER_ANGLE_RAD, fixed thrust) and advance FH by `dt` seconds.
        """

        # The returned (u, v, r) is discarded; state() is the single place that
        # reads FH's state (before and after, for the finite differences).
        # noinspection PyTypeChecker
        self._fh_model.step(
            state=None,   # step() is typed VesselState; None is passed on purpose (no framework state here)
            rudder_angle=RUDDER_ANGLE_RAD,
            thrust=self._thrust_normalized,
            dt=dt,
        )

    def get_ukc(self) -> float:
        """
        FH's current keel clearance (UKC) [m].
        """

        return float(self.ship_interface.getKeelClearance())

    def get_wind(self) -> Tuple[float, float]:
        """
        FH's WindState at the ship: (speed [m/s], direction [rad], wrapped with PySimEnv.normalize_angle).
        """

        ws = self.ship_interface.getWindState()
        return float(ws.Speed), float(PySimEnv.normalize_angle(np.radians(ws.Direction)))

    def get_current(self) -> Tuple[float, float]:
        """
        FH's current at the ship: (speed [m/s], direction [rad], wrapped with PySimEnv.normalize_angle).
        """

        speed, direction_deg = get_current_velocity(self.ship_interface)
        return float(speed), float(PySimEnv.normalize_angle(np.radians(direction_deg)))

    # ---- wind helpers (model_comparison_wind.py) ----------------------------

    def get_wind_raw(self) -> Tuple[float, float]:
        """
        (speed [m/s], direction [deg]) as FH's WindState reports them, unconverted.
        """

        ws = self.ship_interface.getWindState()
        return float(ws.Speed), float(ws.Direction)

    def get_fh_heading_deg(self) -> float:
        """
        FH's native heading [deg], unconverted (pairs with get_wind_raw()).
        """

        return float(self.ship_interface.getShipHeading())


# --------------------------------------------------------------------------
# Live-environment injection into the UA_* classes
# --------------------------------------------------------------------------

def _inject_live_environment(
        model,
        ukc: float,
        wind_speed: float,
        wind_dir: float,
        current_speed: float,
        current_dir: float,
        ray_env: PySimEnv,
) -> None:
    """
    Make a UAntwerpSurrogate use live FH values and the shared ray_env.

    Patches instance attributes only: get_env returns the given values, the UKC
    lookup is dropped, and _env (ray casting) becomes ray_env. No effect on
    UA_clean.

    Args:
        model: UAntwerpSurrogate instance.
        ukc, wind_speed, wind_dir, current_speed, current_dir: Live FH values.
        ray_env: Shared PySimEnv used for the bank geometry.
    """

    env_data = EnvironmentData(
        ukc=ukc, wind_speed=wind_speed, wind_direction=wind_dir,
        current_speed=current_speed, current_direction=current_dir,
    )
    model.get_env = lambda pos, state=None: env_data
    model.ukc = ukc  # get_ukc() returns self.ukc without a lookup
    if hasattr(model, "wcu_lookup"):
        model.wcu_lookup = None
    if hasattr(model, "_env"):
        model._env = ray_env


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _setup_geometry(seed_point: Tuple[float, float], log_prefix: str = "run_comparison"):
    """
    Bank and heading setup shared by run_comparison and generate_trajectories.

    Also returns the map cut-off geometry (None if unavailable) for the
    training-domain checks.

    Returns:
        (ray_env, bank_point, perp_into_water, heading_deg, ship_spec, cutoff_geom)

    Raises:
        RuntimeError: If seed_point projects onto a map cut-off edge instead of
            a real bank.
    """

    ship = create_ship(SHIP_MODEL)
    ship_spec = ship.specifications

    # One PySimEnv for all bank geometry and all ray casting (also injected into
    # the UA_* models), so a single polygon is used. start_pos=seed_point is
    # needed for ship_heading below.
    ray_env = PySimEnv(
        render_mode=None, time_step=SURROGATE_DT, wind=False, current=False,
        start_pos=np.array(seed_point, dtype=np.float32), ship_model=SHIP_MODEL,
    )

    bank_point, perp_into_water = find_bank_reference(ray_env, seed_point)

    cutoff_geom = _cutoff_geometry(ray_env)
    if cutoff_geom is None:
        print(
            f"{log_prefix}: no map cut-off segments available "
            f"(PySimEnv.bank_segments) -- the bank_point cut-off check and the "
            f"near_cutoff training-domain check are disabled.",
            flush=True,
        )
    else:
        cutoff_dist = float(cutoff_geom.distance(_shapely_point(bank_point)))
        if cutoff_dist < CUTOFF_BANK_TOLERANCE:
            raise RuntimeError(
                f"seed_point={seed_point} projects onto a map cut-off edge "
                f"(bank_point=({bank_point[0]:.3f}, {bank_point[1]:.3f}) is "
                f"{cutoff_dist:.3f} m from it), not onto a real bank -- choose "
                f"a seed_point closer to a real bank (see the start-point plot "
                f"of the dataset generator: real banks blue, cut-offs red)."
            )

    # Heading as in PySimEnv._initialize_simulation(): toward the second
    # checkpoint, i.e. along the channel, not the bank tangent.
    heading_deg = float(np.degrees(ray_env.ship_heading))
    if heading_deg < 0:
        heading_deg += 360.0

    # One line of geometry context per run.
    print(
        f"{log_prefix}: seed_point={seed_point} -> "
        f"bank_point=({bank_point[0]:.3f}, {bank_point[1]:.3f}) "
        f"({np.hypot(bank_point[0] - seed_point[0], bank_point[1] - seed_point[1]):.3f} "
        f"m from seed_point), heading_deg={heading_deg:.3f}, "
        f"perp_into_water=({perp_into_water[0]:.4f}, {perp_into_water[1]:.4f})",
        flush=True,
    )

    return ray_env, bank_point, perp_into_water, heading_deg, ship_spec, cutoff_geom


def run_comparison(seed_point: Tuple[float, float]) -> None:
    """
    Run the single-step (v, distance) grid for SHIP_MODEL.

    Writes one CSV per (distance, v) and a combined all_results.csv to
    OUTPUT_DIR, each model row with its (du, dv, dr) error vs. FH (0 for FH).
    Points where FhSession cannot be created are skipped. Trajectories are
    produced separately by generate_trajectories.

    Args:
        seed_point: (x, y) near the port bank in ScheldeSaeftinge_23_002.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ray_env, bank_point, perp_into_water, heading_deg, ship_spec, cutoff_geom = _setup_geometry(seed_point)

    ref_models = build_reference_models(ship_spec, ray_env)

    # UA_full is not part of the grid. Live FH environment is injected per point.
    ua_models = build_ua_models(ship_spec, ("UA_clean", "UA_confined"))

    summary_path = OUTPUT_DIR / "all_results.csv"
    summary_file = open(summary_path, "w", newline="")
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow([
        "model", "distance", "v", "du", "dv", "dr",
        "du_error", "dv_error", "dr_error",
        "fh_ukc", "fh_wind_speed", "fh_wind_dir",
        "fh_current_speed", "fh_current_dir",
    ])

    try:
        _run_grid(ray_env, bank_point, perp_into_water, heading_deg, ship_spec, ref_models, ua_models,
                  summary_writer, cutoff_geom)
    finally:
        summary_file.close()


def _run_grid(ray_env, bank_point, perp_into_water, heading_deg, ship_spec, ref_models, ua_models,
              summary_writer, cutoff_geom=None) -> None:
    """
    (distance, v) grid loop of run_comparison (split out so the summary file is closed in a finally).
    """

    for distance in DISTANCES:
        try:
            pos = locate_point_at_distance(ray_env, bank_point, perp_into_water, float(distance))
        except RuntimeError as exc:
            print(f"SKIPPED distance={distance}: {exc}", flush=True)
            continue
        print(
            f"run_comparison: distance={distance} -> pos=({pos[0]:.3f}, {pos[1]:.3f})",
            flush=True,
        )
        # Ray geometry check, once per distance (independent of speed).
        _print_bank_ray_diagnostics(
            ua_models, pos, heading_deg, perp_into_water, ray_env,
            expected_distance=float(distance), log_prefix="run_comparison",
        )

        for v in V_VALUES:
            thrust = float(min(1.0, v / 6.0))

            run_id = f"dist{distance}_v{v}"
            # FhSession can fail for points FH cannot reach (most likely too
            # close to the bank); skip only this point.
            try:
                fh = FhSession(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    heading_deg=heading_deg,
                    v_init=float(v),
                    thrust=thrust,
                    ship_spec=ship_spec,
                )
            except RuntimeError as exc:
                print(
                    f"SKIPPED {run_id}: FhSession construction failed at "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}), heading_deg="
                    f"{heading_deg:.3f} -- most likely too close to the bank "
                    f"or otherwise outside the exercise's navigable area at "
                    f"this speed/thrust. No CSV written for this point. "
                    f"Original error: {exc}",
                    flush=True,
                )
                continue

            try:
                x0, y0, heading0, u0, v0_, r0 = fh.state()
                # FH's live environment at the "before" state, fed to the UA_* models.
                fh_ukc = fh.get_ukc()
                fh_wind_speed, fh_wind_dir = fh.get_wind()
                fh_current_speed, fh_current_dir = fh.get_current()

                # Flag (not skip) points outside the training domain.
                issue = _training_domain_issue(ray_env, cutoff_geom, x0, y0, heading0, fh_ukc)
                if issue is not None:
                    print(
                        f"run_comparison: {run_id} is OUTSIDE the dataset's "
                        f"training domain (generator stop reason '{issue}') -- "
                        f"UA_* results at this point are extrapolation.",
                        flush=True,
                    )

                fh.advance(SURROGATE_DT)
                x1, y1, heading1, u1, v1_, r1 = fh.state()

                fh_du = (u1 - u0) / SURROGATE_DT
                fh_dv = (v1_ - v0_) / SURROGATE_DT
                fh_dr = (r1 - r0) / SURROGATE_DT

                state = VesselState(
                    terminated=False, truncated=False,
                    x=x0, y=y0, heading=heading0, u=u0, v=v0_, r=r0,
                )

                results = [("FH", fh_du, fh_dv, fh_dr)]

                for label, model in ref_models.items():     # UGent (Tripoli only), Fossen
                    du, dv, dr = model._compute_accelerations(state, RUDDER_ANGLE_RAD, thrust)
                    results.append((label, du, dv, dr))

                # UA_* models with FH's live environment; rays cast at `state`.
                for label, model in ua_models.items():
                    _inject_live_environment(
                        model, fh_ukc, fh_wind_speed, fh_wind_dir,
                        fh_current_speed, fh_current_dir, ray_env,
                    )
                    du, dv, dr = model._compute_accelerations(state, RUDDER_ANGLE_RAD, thrust)
                    results.append((label, du, dv, dr))

                # Error vs. FH (0.0 for the FH row).
                results_with_error = [
                    (label, du, dv, dr, du - fh_du, dv - fh_dv, dr - fh_dr)
                    for label, du, dv, dr in results
                ]

                csv_path = OUTPUT_DIR / f"{run_id}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "model", "distance", "v", "du", "dv", "dr",
                        "du_error", "dv_error", "dr_error",
                        "fh_ukc", "fh_wind_speed", "fh_wind_dir",
                        "fh_current_speed", "fh_current_dir",
                    ])
                    for label, du, dv, dr, du_err, dv_err, dr_err in results_with_error:
                        writer.writerow([
                            label, distance, v, du, dv, dr,
                            du_err, dv_err, dr_err,
                            fh_ukc, fh_wind_speed, fh_wind_dir,
                            fh_current_speed, fh_current_dir,
                        ])

                for label, du, dv, dr, du_err, dv_err, dr_err in results_with_error:
                    summary_writer.writerow([
                        label, distance, v, du, dv, dr,
                        du_err, dv_err, dr_err,
                        fh_ukc, fh_wind_speed, fh_wind_dir,
                        fh_current_speed, fh_current_dir,
                    ])

                print(f"Wrote {csv_path}", flush=True)
            finally:
                # Release this point's native FH resources.
                fh.close()


# --------------------------------------------------------------------------
# Trajectory visualization: the integrated counterpart of run_comparison
# --------------------------------------------------------------------------

def generate_trajectories(
        seed_point: Tuple[float, float],
        speed: float = TRAJECTORY_SPEED,
        distances: Sequence[float] = DISTANCES,
        target_distance: float = TRAJECTORY_TARGET_DISTANCE,
        max_steps: int = TRAJECTORY_MAX_STEPS,
) -> None:
    """
    Roll out trajectories of all models at a fixed speed for each distance to
    the bank, from the same start as run_comparison, and plot them.

    FH is stepped with FhSession.advance(SURROGATE_DT); every other model has its
    own ShipSimulator stepped with integrate() (fixed rudder and thrust). The
    UA_* models get FH's live environment at FH's position each step, but cast
    rays from their own position. All models run as many steps as FH needs for
    target_distance (capped at max_steps).

    Writes one CSV (model, step, x, y) per distance to TRAJECTORY_OUTPUT_DIR and,
    if matplotlib is available, OUTPUT_DIR/trajectories_overview.png.

    Args:
        seed_point: As in run_comparison.
        speed: Initial surge speed [m/s].
        distances: Distances to the bank [m], one subplot each.
        target_distance: FH path length [m] at which to stop.
        max_steps: Safety cap on steps per distance.
    """

    TRAJECTORY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ray_env, bank_point, perp_into_water, heading_deg, ship_spec, cutoff_geom = _setup_geometry(
        seed_point, log_prefix="generate_trajectories",
    )

    thrust = float(min(1.0, speed / 6.0))
    actuator = ActuatorCommand(rudder_angle=RUDDER_ANGLE_RAD, thrust=thrust)

    # Built once: loading the checkpoints and lookup CSVs per distance is slow,
    # and the models keep no state between distances (their environment is
    # injected every step).
    ua_models = build_ua_models(ship_spec)

    trajectories_by_distance = {}

    for distance in distances:
        try:
            pos = locate_point_at_distance(ray_env, bank_point, perp_into_water, float(distance))
        except RuntimeError as exc:
            print(f"SKIPPED trajectory at distance={distance}: {exc}", flush=True)
            continue
        print(
            f"generate_trajectories: distance={distance} -> "
            f"pos=({pos[0]:.3f}, {pos[1]:.3f})",
            flush=True,
        )

        # Skip distances where FH cannot be initialized (as in run_comparison).
        try:
            fh = FhSession(
                x=float(pos[0]), y=float(pos[1]), heading_deg=heading_deg,
                v_init=float(speed), thrust=thrust, ship_spec=ship_spec,
            )
        except RuntimeError as exc:
            print(
                f"SKIPPED trajectory at distance={distance}: FhSession "
                f"construction failed at pos=({pos[0]:.3f}, {pos[1]:.3f}), "
                f"heading_deg={heading_deg:.3f} -- most likely too close to "
                f"the bank or otherwise outside the exercise's navigable "
                f"area at this speed/thrust. No CSV written for this "
                f"distance. Original error: {exc}",
                flush=True,
            )
            continue

        x0, y0, heading0, u0, v0_, r0 = fh.state()
        init_state = VesselState(
            terminated=False, truncated=False,
            x=x0, y=y0, heading=heading0, u=u0, v=v0_, r=r0,
        )

        # UGent's bank distances are cast from its own position, like the UA_* rays.
        non_fh_models = {
            **build_reference_models(ship_spec, ray_env),
            **ua_models,
        }

        # Ray geometry check at FH's settled start pose, once per distance.
        _print_bank_ray_diagnostics(
            ua_models, np.array([x0, y0], dtype=float), float(np.degrees(heading0)), perp_into_water,
            ray_env, expected_distance=float(distance),
            log_prefix="generate_trajectories",
        )

        # One ShipSimulator per non-FH model, all starting from FH's initial state.
        sims = {}
        for label, model in non_fh_models.items():
            sim = make_model_simulator(ship_spec, model)
            # A separate VesselState per simulator (reset() stores it as is).
            sim.reset(VesselState(
                terminated=init_state.terminated, truncated=init_state.truncated,
                x=init_state.x, y=init_state.y, heading=init_state.heading,
                u=init_state.u, v=init_state.v, r=init_state.r,
            ))
            sims[label] = sim

        paths = {label: [(x0, y0)] for label in sims}
        paths["FH"] = [(x0, y0)]
        surge_clip = SurgeClipCounter(ship_spec, sims)

        fh_path_length = 0.0
        prev_fh_xy = (x0, y0)
        step_i = 0
        # First step where FH leaves the training domain (reported, not cut short).
        domain_exit = None
        fh_heading = heading0

        try:
            while fh_path_length < target_distance and step_i < max_steps:
                # FH's live environment at its current position, fed to the UA_* models.
                fh_ukc = fh.get_ukc()
                if domain_exit is None:
                    issue = _training_domain_issue(
                        ray_env, cutoff_geom, prev_fh_xy[0], prev_fh_xy[1], fh_heading, fh_ukc,
                    )
                    if issue is not None:
                        domain_exit = (step_i, issue)
                fh_wind_speed, fh_wind_dir = fh.get_wind()
                fh_current_speed, fh_current_dir = fh.get_current()
                for label, model in ua_models.items():
                    _inject_live_environment(
                        model, fh_ukc, fh_wind_speed, fh_wind_dir,
                        fh_current_speed, fh_current_dir, ray_env,
                    )

                for label, sim in sims.items():
                    sim.integrate(actuator, SURROGATE_DT)
                    paths[label].append((float(sim.state.x), float(sim.state.y)))
                    surge_clip.update(label, float(sim.state.u))

                fh.advance(SURROGATE_DT)
                xf, yf, fh_heading, _uf, _vf, _rf = fh.state()
                fh_path_length += float(np.hypot(xf - prev_fh_xy[0], yf - prev_fh_xy[1]))
                prev_fh_xy = (xf, yf)
                paths["FH"].append((xf, yf))

                step_i += 1

            if step_i >= max_steps:
                print(
                    f"generate_trajectories: distance={distance} hit "
                    f"max_steps={max_steps} before FH's own path reached "
                    f"target_distance={target_distance} (covered "
                    f"{fh_path_length:.1f} m instead) -- FH may have "
                    f"stalled (e.g. thrust too low to make headway). "
                    f"Trajectory written as-is, just shorter than requested.",
                    flush=True,
                )
        finally:
            fh.close()

        surge_clip.report(f"generate_trajectories: distance={distance}")
        if domain_exit is not None:
            exit_step, exit_reason = domain_exit
            print(
                f"generate_trajectories: distance={distance} -- FH left the "
                f"dataset's training domain at step {exit_step} "
                f"(t={exit_step * SURROGATE_DT:.1f} s, generator stop reason "
                f"'{exit_reason}'); UA_* behaviour after that point is "
                f"extrapolation.",
                flush=True,
            )

        trajectories_by_distance[distance] = paths

        # Lateral deviation of each final point from the initial heading line; a
        # model without bank effects should report about 0 m at rudder 0.
        for label, points in paths.items():
            xf, yf = points[-1]
            dx, dy = xf - x0, yf - y0
            lateral_deviation = abs(dx * np.sin(heading0) - dy * np.cos(heading0))
            print(
                f"generate_trajectories: distance={distance}, model={label} "
                f"-> {len(points) - 1} steps, final=({xf:.2f}, {yf:.2f}), "
                f"lateral_deviation_from_initial_heading={lateral_deviation:.3f} m",
                flush=True,
            )

        csv_path = TRAJECTORY_OUTPUT_DIR / f"trajectory_dist{distance}_v{speed}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "step", "x", "y"])
            for label, points in paths.items():
                for step_idx, (x, y) in enumerate(points):
                    writer.writerow([label, step_idx, x, y])
        print(f"Wrote {csv_path}", flush=True)

    _plot_trajectories(trajectories_by_distance, ray_env, seed_point, speed)


def _plot_trajectories(trajectories_by_distance, ray_env: PySimEnv, seed_point: Tuple[float, float], speed: float) -> None:
    """
    Save one PNG with a subplot per distance: all model paths over the river
    boundary. Skipped with a note if matplotlib is unavailable.
    """

    if plt is None:
        print(
            "_plot_trajectories: matplotlib is not importable in this "
            "environment -- skipping the PNG. The per-distance trajectory "
            "CSVs in TRAJECTORY_OUTPUT_DIR were still written and can be "
            "plotted separately.",
            flush=True,
        )
        return

    if not trajectories_by_distance:
        print(
            "_plot_trajectories: no distances produced a trajectory "
            "(every FhSession construction failed) -- nothing to plot.",
            flush=True,
        )
        return

    distances_plotted = sorted(trajectories_by_distance.keys(), reverse=True)
    n = len(distances_plotted)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil
    # Minimum width so the 6-entry legend fits with few subplots.
    fig_width = max(6.0 * ncols, 11.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, 6.0 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    boundary = ray_env.polygon_shape.boundary
    boundary_xy = np.array(boundary.coords)

    for ax, distance in zip(axes_flat, distances_plotted):
        ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="#9a9a9a", linewidth=1.0, zorder=1, label="_nolegend_")

        paths = trajectories_by_distance[distance]
        for label, points in paths.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            color = TRAJECTORY_COLORS.get(label, "#000000")
            linestyle = TRAJECTORY_LINESTYLES.get(label, "-")
            ax.plot(xs, ys, color=color, linestyle=linestyle,
                     linewidth=2.0 if label == "FH" else 1.5,
                     label=label, zorder=3 if label == "FH" else 2)
            ax.scatter([xs[0]], [ys[0]], color=color, s=25, zorder=4, marker="o")

        all_x = [p[0] for pts in paths.values() for p in pts]
        all_y = [p[1] for pts in paths.values() for p in pts]
        pad = 20.0
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"distance={distance} m, v={speed} m/s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, linewidth=0.3, alpha=0.5)

    for ax in axes_flat[n:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=len(labels), frameon=False,
    )
    fig.suptitle(f"Model trajectories vs. FH -- seed_point={seed_point}, v={speed} m/s")
    # Bottom margin for x labels plus the figure legend; wider hspace so row
    # titles do not collide with the x labels of the row above.
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.subplots_adjust(hspace=0.35)

    png_path = OUTPUT_DIR / "trajectories_overview.png"
    try:
        # str(): on Windows, some Pillow versions raise OSError [Errno 22] for a
        # WindowsPath here. A save failure must not lose the written CSVs.
        fig.savefig(str(png_path), dpi=150)
        print(f"Wrote {png_path}", flush=True)
    except OSError as exc:
        print(
            f"_plot_trajectories: failed to save {png_path} ({exc}) -- "
            f"the trajectory CSVs above were already written successfully "
            f"and are unaffected; only this PNG is missing. If this "
            f"recurs, try a different OUTPUT_DIR (e.g. one that isn't "
            f"inside a OneDrive-synced or antivirus-scanned folder) or "
            f"plot the CSVs separately.",
            flush=True,
        )
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# FH initialization diagnostics (not part of the comparison; run manually)
# --------------------------------------------------------------------------

def diagnose_fh_initialization() -> None:
    """
    Run the known-working FH path (env_factory + PySimEnv, FH_TABULAR without
    wind and current) in this process. If it also fails with an "os2_body" error,
    the cause lies outside this script (process environment); if it succeeds,
    FhSession's setup still differs.
    """

    from fw.dynamics.dynamics_model import Model
    from fw.configuration.config import get_config_parameters
    from fw.configuration import env_factory

    framework_config = dict(get_config_parameters())
    # Explicit overrides, independent of config.py's current defaults.
    framework_config["ship_model"] = SHIP_MODEL
    framework_config["dynamics_model"] = Model.FH_TABULAR
    framework_config["wind"] = False
    framework_config["current"] = False

    env_factory.initialize_all_environments()
    env_wrapper = env_factory.build_default_environment(
        env_factory.fh_sim_env_name, framework_config
    )

    print(
        "diagnose_fh_initialization: calling PySimEnv.reset() "
        "(FH_TABULAR / Tripoli / no wind or current) via the exact construction "
        "fw.main.py uses, in THIS process...",
        flush=True,
    )
    env_wrapper.gym_env.reset()
    print(
        "diagnose_fh_initialization: PySimEnv.reset() SUCCEEDED in this "
        "process. If run_comparison()'s FhSession still fails with the "
        "os2_body error after this succeeds, something in FhSession's "
        "own construction is the difference after all, despite round 3's "
        "argument-for-argument trace -- worth a closer side-by-side look. "
        "If THIS call itself failed instead, the cause is outside this "
        "script's code entirely.",
        flush=True,
    )


def diagnose_fh_session_at_known_good_position(
        heading_deg: float = 0.0, v_init: float = 1.0, thrust: float = 0.5,
) -> None:
    """
    Create an FhSession at DEFAULT_FRAMEWORK_SHIP_POS, a known-good position.

    Success means the bank-derived position or heading is the problem; failure
    points at FhSession or its heading/speed arguments.
    """

    from fw.common.default_positions import DEFAULT_FRAMEWORK_SHIP_POS

    ship = create_ship(SHIP_MODEL)
    ship_spec = ship.specifications
    x, y = DEFAULT_FRAMEWORK_SHIP_POS

    print(
        f"diagnose_fh_session_at_known_good_position: constructing "
        f"FhSession at DEFAULT_FRAMEWORK_SHIP_POS=({x:.1f}, {y:.1f}), "
        f"heading_deg={heading_deg:.1f}, v_init={v_init:.1f} "
        f"(NOT a bank-derived point)...",
        flush=True,
    )
    fh = FhSession(
        x=float(x), y=float(y), heading_deg=heading_deg,
        v_init=v_init, thrust=thrust, ship_spec=ship_spec,
    )
    try:
        print(
            f"diagnose_fh_session_at_known_good_position: FhSession "
            f"SUCCEEDED at the known-good position. state()={fh.state()} "
            f"-- this points at find_bank_reference/locate_point_at_distance's "
            f"derived (x, y)/heading as the actual cause, not FhSession's "
            f"construction.",
            flush=True,
        )
    finally:
        fh.close()


def diagnose_fh_position_sweep(
        good_pos: Optional[Tuple[float, float]] = None,
        bad_pos: Tuple[float, float] = (2128.498, 26.037),
        bad_heading_deg: float = 32.084,
        n_steps: int = 6,
) -> None:
    """
    Separate position from heading as the cause of an FH initialization failure
    by sweeping FhSession (failures are reported, not raised):

      1. Position: from good_pos to bad_pos at heading 0.
      2. Heading: at bad_pos, from 0 to bad_heading_deg.
    """

    from fw.common.default_positions import DEFAULT_FRAMEWORK_SHIP_POS

    if good_pos is None:
        good_pos = DEFAULT_FRAMEWORK_SHIP_POS

    ship = create_ship(SHIP_MODEL)
    ship_spec = ship.specifications

    def try_point(x: float, y: float, heading_deg: float, label: str) -> None:
        """
        Try an FhSession at one pose and print the outcome.
        """

        try:
            fh = FhSession(
                x=float(x), y=float(y), heading_deg=float(heading_deg),
                v_init=1.0, thrust=0.5, ship_spec=ship_spec,
            )
        except RuntimeError as exc:
            print(
                f"{label}: x={x:.3f}, y={y:.3f}, heading_deg={heading_deg:.3f} "
                f"-> FAILED ({exc})",
                flush=True,
            )
            return
        try:
            print(
                f"{label}: x={x:.3f}, y={y:.3f}, heading_deg={heading_deg:.3f} "
                f"-> OK, state()={fh.state()}",
                flush=True,
            )
        finally:
            fh.close()

    print(
        "=== diagnose_fh_position_sweep: part 1, position sweep "
        "(heading=0.0) ===",
        flush=True,
    )
    for i in range(n_steps + 1):
        t = i / n_steps
        sweep_x = good_pos[0] + t * (bad_pos[0] - good_pos[0])
        sweep_y = good_pos[1] + t * (bad_pos[1] - good_pos[1])
        try_point(sweep_x, sweep_y, 0.0, f"position_sweep[{i}/{n_steps}]")

    print(
        "=== diagnose_fh_position_sweep: part 2, heading sweep "
        "(pos=bad_pos) ===",
        flush=True,
    )
    for i in range(n_steps + 1):
        t = i / n_steps
        sweep_heading_deg = t * bad_heading_deg
        try_point(bad_pos[0], bad_pos[1], sweep_heading_deg, f"heading_sweep[{i}/{n_steps}]")


if __name__ == "__main__":
    generate_trajectories((2245.0, -47.0))
    # run_comparison((2245.0, -47.0))
