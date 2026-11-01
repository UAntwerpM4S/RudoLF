"""
Compares 5 dynamics models against FH (ground truth) over a grid of
(initial surge speed, perpendicular distance-to-bank) points, for one ship
model at a time (SHIP_MODEL: TRIPOLI or MYZAKO_37) sailing parallel to a
river bank.

Models compared:
    FH          -- the real simexe engine, advanced via
                   fw.dynamics.fh_tabular_model.FhTabularModel.step() (see
                   FhSession below) for control-application and time-
                   advancement, but NOT for (du, dv, dr): FhTabularModel has
                   no `_compute_accelerations()` (it reads velocities
                   directly from the FH engine after stepping, specifically
                   to avoid finite-differencing noise -- see its module
                   docstring). This script needs (du, dv, dr) for every
                   model on a common footing, so for FH specifically it is
                   still derived as (u_after - u_before) / dt,
                   (v_after - v_before) / dt, (r_after - r_before) / dt
                   across one SURROGATE_DT step, by reading FhSession.state()
                   before and after calling FhSession.advance() (which
                   itself calls FhTabularModel.step() internally) --
                   finite-differenced by explicit request, since every other
                   model here predicts instantaneously, not integrated.
    UGent       -- fw.dynamics.ugent_surrogate.UGentSurrogate
    Fossen      -- fw.dynamics.fossen.Fossen
    UA_clean    -- fw.dynamics.uantwerp_surrogate.UAntwerpSurrogate, preset "tripoli_clean"
    UA_confined -- fw.dynamics.uantwerp_surrogate.UAntwerpSurrogate, preset "tripoli_confined"
                   (MYZAKO_37: preset "myzako_confined"; UA_clean: "myzako_clean")
    UA_full     -- fw.dynamics.uantwerp_surrogate.UAntwerpSurrogate, preset "tripoli_full"
                   (see UA_PRESETS_BY_SHIP / build_ua_models; a slot without
                   a preset for SHIP_MODEL is left out)

This module is also the shared base of the wind test (model_comparison_wind.py
imports FhSession, _inject_live_environment, ... from here); the FhSession
arguments it uses (exercise_path, wind_field, rewind_enabled) are optional and
default to the exact ship-bank behaviour.

For each (v, distance) grid point, the ship is placed at that perpendicular
distance from the bank (port side), heading parallel to the bank, with
rudder_angle=0 and thrust=min(1.0, v/6.0). FH is initialized, allowed to
settle for INITIALIZATION_TIME (FhSimBridge's own 0.125 s -- the dataset
generator now uses one 0.1 s step, see NOTE), then the "before" state and FH's own live
UKC/wind/current are captured. FH is advanced by SURROGATE_DT and the
"after" state is captured to finite-difference (du, dv, dr). The other 5
models are each evaluated ONCE at the "before" state via their own
`_compute_accelerations()` -- i.e. instantaneous, not integrated over
SURROGATE_DT the way FH's own physics engine was.

NOTE (design history, most recent first -- read before interpreting
UA_clean/confined/full's numbers):

  - Consolidation: the three UAntwerpSurrogateTripoli{Clean,Confined,Full}
    classes are replaced by the single fw.dynamics.uantwerp_surrogate.
    UAntwerpSurrogate, configured per slot by a preset (UA_PRESETS_BY_SHIP). Same
    checkpoints, same raw input rows, same rudder scale (read from the
    checkpoint: 35 deg Tripoli, 75 deg Myzako), same get_env/_env hooks, so the results are
    unchanged. UA_clean (no UKC/rays) still has no `_env` and never calls
    get_env. The ship-bank test runs trajectories at TRAJECTORY_SPEED = 3;
    the wind test has its own WIND_TRAJECTORY_SPEED.

  - Round 17 -- aligned with the updated sim_fh_dataset_generator.py and
    the shared fw/environments/ray_features.py:
      * Ray diagnostics (`_print_bank_ray_diagnostics`) use the shared
        `compute_ray_features` / `ray_world_angles` (ray_0 = bow, clockwise,
        PySimEnv heading convention). For a ship parallel to the bank the
        "toward bank" ray is ray_90 or ray_270.
      * RAY_MAX_DISTANCE is imported from ray_features.py, not redefined.
      * FhSession.state() converts FH's heading with the generator's own
        `fh_heading_to_env_rad`.
      * Map cut-off edges: `_setup_geometry` refuses a seed_point that
        projects onto a cut-off edge instead of a real bank.
      * Training-domain check (`_training_domain_issue`): grid points, and
        the first trajectory step, where FH's state violates one of the
        generator's stop limits (near a cut-off, bank closer than
        MIN_BANK_DISTANCE, UKC too small, outside open water) are flagged
        as extrapolation for the UA_* surrogates. Diagnostic only: no
        output CSV columns changed, nothing is skipped.
      * Settle time: the generator now settles for one math-model step
        (0.1 s), FhSimBridge still hardcodes 0.125 s. Only the starting
        transient differs; the "before" state is read from FH either way.
      * Only surrogates trained on datasets from the current generator
        (current ray layout) are supported.

  - FH init crash, three rounds -- STILL NOT CONFIRMED FIXED as of this
    writing (2026-09-30). All three real runs hit the IDENTICAL native
    error (`Error: Cannot open input file: .../cache/.../os2_body`, then
    `MathModel.Initialize()` returning False), which is what makes this
    worth tracking carefully: each round removed one more difference from
    the one confirmed-working construction without yet finding the actual
    cause.
      Round 1: FhSession's math-model bootstrapping was originally
    hand-rolled (its own Config/Exercise/MathModel calls) and called
    `config.SetRewindEnabled(False)` with no `config.SetRewindConfig(...)`
    at all -- replaced with `fw.simulators.fh_sim_bridge.FhSimBridge`
    (Walter's own bridge, which always pairs `SetRewindEnabled(True)` with
    `SetRewindConfig(...)`). Same crash.
      Round 2: still passed explicit `library_path`/`config_path`/
    `exercise_path` to `FhSimBridge` (re-derived via
    `exercise_files.resolve_exercise_file` + a hand-built `SimConfig`),
    and an explicit per-grid-point `output_path`. Walter confirmed Tripoli
    + wind/current DOES work through `fw.main.py` ->
    `fw.configuration.env_factory.FhSimEnvironment`, whose `FhSimBridge`
    call passes ONLY `ship_model=...` and `scenario=...` -- so
    `library_path`/`config_path`/`exercise_path` were dropped, matching
    FhSimBridge's own defaults. Same crash.
      Round 3 (current state): traced the confirmed-working construction
    all the way through -- `env_factory.py` -> `PySimEnv.__init__` ->
    `PySimEnv.create_simulator()` / `PySimEnv._initialize_simulation()` --
    line by line (see FhSession's own docstring for the full trace) and
    matched it exactly: `output_path` (the one remaining explicit
    argument) is now dropped too, `FhTabularModel` is constructed BEFORE
    `bridge.reset()` (matching `create_simulator()`'s order, not after, as
    an earlier round had it), and `scenario="DDShip_scenario1_nowindnocurrent"`
    (config.py's own value) is now passed for exact parity, even though it
    has no effect for ship_model=TRIPOLI. This is now believed to be a
    faithful 1:1 replication of the confirmed-working call sequence, but
    -- as with the previous two rounds -- THIS HAS NOT YET BEEN RUN by
    Walter to confirm it actually resolves the crash, so it should not be
    read as fixed until he reports back. If this round ALSO reproduces the
    same crash, the cause is not in any FhSimBridge/FhTabularModel
    constructor argument or call order at all, and the next place to look
    is something outside this class entirely -- e.g. what else
    `fw.main.py`'s process does before constructing its environment
    (working directory, environment variables, prior imports) that this
    standalone script does not.
      Separately (not part of chasing this crash, but done at the same
    time and worth keeping regardless of how the crash is eventually
    resolved): each grid point's FhSimBridge is explicitly disposed via
    `FhSession.close()` after that point's CSV is written (wrapped in
    `try/finally` in `run_comparison`) -- the original hand-rolled version
    never released its MathModel instances at all.

  - FhSession delegates control-application and time-advancement (in
    `advance()`) to Walter's own `fw.dynamics.fh_tabular_model.FhTabularModel
    .step()`, instead of a hand-rolled `_apply_controls()` that only knew
    the azipod control path. FhSession exposes `ship_model`/`ship_interface`
    /`math_model` attributes so it can be passed as `FhTabularModel`'s own
    `fh_engine` argument (`FhTabularModel(ship_spec, fh_engine=self)`).
    `FhTabularModel.step()` bundles "apply controls, then
    math_model.simulateSeconds(dt)" in one call, so it can only replace
    FhSession's PER-STEP control application (inside `advance()`), not the
    one-time control-surface priming that used to happen once in `__init__`
    right after the initial settle (that priming had no effect on the
    "before" state() read anyway, since no time is advanced between setting
    it and reading state() -- see FhSession's docstring). `du/dv/dr` for FH
    are unaffected: still finite-differenced from `state()` calls before
    and after `advance()`, not from `step()`'s own returned (u, v, r).

  - The 3 UA checkpoints are evaluated
through Walter's own UAntwerpSurrogate class directly (one instance per
preset, each loading its preset's checkpoint) -- not through a hand-built
helper that constructs the raw input vector itself. However,
each class's OWN environmental/ray sourcing (UKC/wind/current via its own
WcuLookup CSV nearest-neighbor lookup; rays via its own separately-
constructed PySimEnv) is overridden, per request, to use live values from
this same FH session instead -- see `_inject_live_environment`, called
once per grid point for every UA_* model just before
`_compute_accelerations`. Concretely:
  - UKC/wind/current fed to UA_confined/UA_full are FH's own live values at
    the "before" state (fh_ukc/fh_wind_speed/fh_wind_dir/
    fh_current_speed/fh_current_dir, also written to each output CSV row),
    not a CSV lookup.
  - Rays fed to UA_confined/UA_full are cast from `state`'s own (x, y,
    heading) -- i.e. the same post-settle position/heading used for every
    other input and for FH's own ground-truth delta -- via `ray_env`, the
    SAME PySimEnv instance this script already uses for its own
    bank-finding geometry, rather than each class's own separate PySimEnv
    instance. This also means the classes' ray-casting and this script's
    bank/distance-placement geometry can never disagree about which
    polygon or coordinate frame is in play (see caveat 1 below).
  - This makes the comparison "does the surrogate predict correctly given
    identical, ground-truth inputs" (as an earlier, hand-built version of
    this script also did) rather than "does the surrogate predict
    correctly as actually deployed" (as an intermediate version of this
    script, which called the 3 classes unmodified, did) -- a bad
    confined/full result now points at the surrogate's learned dynamics,
    not at a stale WcuLookup CSV or a second, independent polygon.
  - UA_clean (open_water: no UKC, no rays) has no `_env` attribute and
    never calls `get_env` (its input row only uses heading/surge/sway/
    yaw_rate/rudder/thrust); `_inject_live_environment` is still called on
    it for uniformity, but is a harmless no-op there.
  - UAntwerpSurrogate converts the rudder angle with the rudder scale stored
    in the checkpoint (the data's rudder_max_deg: 35 deg Tripoli, 75 deg
    Myzako, equal to the ship's max rudder angle in ship_factory); RUDDER_ANGLE_RAD is
    0.0 throughout this script anyway.

Two notes worth reading before trusting this script's output:

  1. FH's own (x, y) coordinate frame and `ray_env`'s polygon (loaded from
     config.py's 'overall_contour'/'depth_contour', the same PySimEnv
     instance now shared by this script's own bank-finding geometry AND
     every UA_* class's ray-casting -- see NOTE above) are expected to be
     the same coordinate system by design (per Walter), so this is not
     flagged as an open risk. It's still the one assumption this script's
     entire geometry (bank-finding, distance placement, ray inputs) rests
     on, so on the first real run it's worth a quick visual sanity check:
     plot the located (x, y) points against the polygon and confirm they
     land in the river channel, at roughly the expected distance from a
     visible bank.

  2. UNTESTED -- every real API call below (Config/Exercise/MathModel,
     ship_config.*, ship_interface.*, PySimEnv.polygon_shape/ray_cast_*)
     was cross-referenced against sim_fh_dataset_generator.py's actual,
     working use of the same APIs, and this file was syntax-checked, but
     it has never been executed end-to-end: this sandbox has no `sim`
     package (SIMEXE_REPO_FOLDER dependency), no tkinter/gymnasium/shapely,
     no real exercise (.tab) file, and no real contour CSVs. Run it for
     real, on a small grid first, before trusting the full output.

Usage: `run_comparison(seed_point)` needs a real (x, y) point on the port
bank in area ScheldeSaeftinge_23_002 to anchor the bank-following geometry
(see find_bank_reference below) -- there is no default; supply one at
call time (see `if __name__ == "__main__"` at the bottom).
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
from fw.surrogate.sim_fh_dataset_generator import (
    _PROPELLER_RPS_SCALE,
    _cutoff_geometry,
    fh_heading_to_env_rad,
    get_current_velocity,
    stop_reason,
)
from fw.environments.py_sim_env import PySimEnv
# Round 17 -- the single shared ray implementation (bow = ray_0, clockwise,
# PySimEnv heading convention), the same one the dataset generator and the
# UA_* surrogates use. See fw/environments/ray_features.py.
from fw.environments.ray_features import (
    RAY_ANGLES_DEG,
    RAY_MAX_DISTANCE,
    compute_ray_features,
    ray_world_angles,
)

# Round 11 -- trajectory visualization. ShipSimulator/ActuatorCommand are
# Walter's own production integrator (fw/simulators/ship_simulator.py), used
# here exactly as `generate_trajectories` was asked to: instead of this
# script hand-rolling its own position-integration formula for UGent/
# Fossen/UA_* (risking yet another subtly-wrong heading convention, as
# happened in round 8), each model's multi-step trajectory is advanced via
# the SAME ShipSimulator.integrate() the rest of the framework trains and
# runs with. ShipSimulator.integrate() adds EnvironmentModel wind/current
# accelerations only for models with models_own_wind_current = False
# (Fossen, Nomoto, UGent, UA_clean, UA_confined); FH and UA_full model their
# own. This script runs with wind=False/current=False throughout, so
# EnvironmentModel contributes nothing here either way.
from fw.simulators.ship_simulator import ShipSimulator
from fw.ships.actuator_command import ActuatorCommand

try:
    import matplotlib
    matplotlib.use("Agg")  # headless -- this script has no display, and
                            # shouldn't depend on one being available
    import matplotlib.pyplot as plt
except ImportError:
    # Both names set here, even though only `plt` is ever read again below
    # (`matplotlib` itself isn't referenced outside this block) -- a static
    # checker (e.g. PyCharm's "should also be defined in the except block"
    # inspection) otherwise flags `matplotlib` as possibly-undefined on any
    # path where the try block raised partway through, since Python itself
    # doesn't guarantee the name wasn't bound before the failing line.
    matplotlib = None
    plt = None  # generate_trajectories' plotting step checks this and
                # degrades to "CSVs only, no PNG" with a clear message,
                # rather than failing the whole script over a missing
                # plotting dependency when only the CSVs were needed.

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Grid and fixed run parameters
# --------------------------------------------------------------------------

V_VALUES = [1, 2, 3, 4, 5, 6]        # initial surge speed [m/s]
DISTANCES = [35.0, 30.0, 25.0, 20.0, 15.0, 11.0]  # perpendicular distance to bank [m]
# DISTANCES = [200.0, 150.0, 100.0, 75.0, 50.0, 35.0, 30.0, 25.0, 20.0, 15.0, 11.0]  # perpendicular distance to bank [m]

RUDDER_ANGLE_RAD = 0.0               # fixed: ship sails parallel to the bank
# SURROGATE_DT (0.1 s) is imported from fw.surrogate.schema, the single source
# of the surrogate time step; the wind/current tests import it from here.
INITIALIZATION_TIME = 0.125          # documentation-only: FhSimBridge hardcodes this
                                      # 0.125 s settle internally (not passed in) --
                                      # kept here so the value is visible without having
                                      # to open fh_sim_bridge.py. Round 17: the dataset
                                      # generator now settles for its INITIALIZATION_TIME
                                      # (one math-model step, 0.1 s) instead, so the two
                                      # differ; see the module docstring NOTE.

# This script's own output (the CSVs this script writes, one per grid
# point) only -- NOT passed to FhSimBridge, which is now left to use its
# own default output_path for exact parity with the confirmed-working
# construction (see FhSession's docstring). Relative to the current working
# directory at IMPORT time (resolved to an absolute path then, so a later
# chdir does not move it).
OUTPUT_DIR = Path("model_comparison_output").resolve()

# Ship simulated by FH (and used for ship_spec / ray_env). FhSimBridge picks
# the exercise from it: TRIPOLI -> DDShip_Tripoli_scenario_nowindnocurrent.tab,
# MYZAKO_37 -> DDShip_scenario1_nowindnocurrent_myzako_37.tab.
# UGent is a Tripoli model (TRIPOLI_ONLY_MODELS); the UA_* slots run for every
# ship that has a preset for them (UA_PRESETS_BY_SHIP). A slot without one is
# left out automatically, so a Myzako run compares FH, Fossen (Myzako
# coefficients), UA_clean (preset "myzako_clean") and UA_confined
# (preset "myzako_confined").
SHIP_MODEL = ShipModel.MYZAKO_37

# All UA_* model slots, in report order.
UA_LABELS = ("UA_clean", "UA_confined", "UA_full")

# Per ship: UA_* model slot -> UAntwerp surrogate preset (fw/surrogate/presets.py:
# ship, scenario, checkpoint, UKC lookup CSV). Shared with the wind and
# current tests. Add a slot here once its checkpoint is trained.
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
    """UA_* slot -> preset name for the current SHIP_MODEL (empty if it has none)."""

    return UA_PRESETS_BY_SHIP.get(SHIP_MODEL, {})


# Surge limits [m/s] of the ShipSimulator that integrates every NON-FH model
# in all three tests (ship-bank, wind, current). None = the ship's own
# ShipSpecifications.surge_limits (Tripoli: 0-5 m/s). FH itself is never
# clipped and the training data reaches ~6 m/s, so a start speed near the
# upper limit -- or a following current, since the models start from FH's
# speed over ground -- clips the models but not FH. Example: (0.0, 8.0).
# Every run reports the steps in which a model sat at the upper limit.
SIM_SURGE_LIMITS: Optional[Tuple[float, float]] = (0.0, 8.0)


def sim_surge_limits(ship_spec) -> Tuple[float, float]:
    """Surge limits the model simulators use (SIM_SURGE_LIMITS or the ship's own)."""

    limits = SIM_SURGE_LIMITS if SIM_SURGE_LIMITS is not None else ship_spec.surge_limits
    lo, hi = float(limits[0]), float(limits[1])
    if not lo < hi:
        raise ValueError(f"SIM_SURGE_LIMITS must be (low, high) with low < high, got {limits}.")
    return lo, hi


def make_model_simulator(ship_spec, model) -> ShipSimulator:
    """
    ShipSimulator for one non-FH model (no wind/current added on top), with
    the surge limits of `sim_surge_limits`. Only the simulator's clipping
    uses the changed limits; the model keeps `ship_spec` unchanged.
    """

    specs = dataclasses.replace(ship_spec, surge_limits=sim_surge_limits(ship_spec))
    return ShipSimulator(specs=specs, dynamics=model, dt=SURROGATE_DT, wind=False, current=False)


class SurgeClipCounter:
    """Counts, per model, the steps whose surge sits at the simulator's upper surge limit."""

    def __init__(self, ship_spec, labels):
        """Start a zero count for each model label, with the simulator's surge limits for `ship_spec`."""
        self.limits = sim_surge_limits(ship_spec)
        self.counts = {label: 0 for label in labels}

    def update(self, label: str, u: float) -> None:
        """Count one step for `label` if its surge `u` is at the simulator's upper surge limit."""
        # Upper limit only: at a lower limit of 0, a clipped surge cannot be
        # told apart from a ship that simply stopped.
        if u >= self.limits[1] - 1e-9:
            self.counts[label] += 1

    def report(self, run_id: str) -> None:
        """Log a warning listing the models whose surge was clipped at the upper limit in run `run_id`."""

        clipped = {label: n for label, n in self.counts.items() if n}
        if clipped:
            logger.warning("%s: surge at the simulator's upper limit (%g m/s) in %s step(s) -- these models "
                           "were clipped, FH is not (set model_comparison.SIM_SURGE_LIMITS to widen).",
                           run_id, self.limits[1], clipped)


# Models that exist for the Tripoli only (UGent's identified coefficients).
# The UA_* slots depend on UA_PRESETS_BY_SHIP instead.
TRIPOLI_ONLY_MODELS = ("UGent",)
_skip_warned: set = set()


def tripoli_models_enabled() -> bool:
    """True if SHIP_MODEL is the Tripoli, i.e. the Tripoli-only models (UGent) apply."""

    return SHIP_MODEL == ShipModel.TRIPOLI


def _note_models_skipped(labels) -> None:
    """Log once per model that it is left out because it does not exist for SHIP_MODEL."""

    new = [label for label in labels if label not in _skip_warned]
    if new:
        logger.warning("SHIP_MODEL=%s: no %s for this ship -- left out.", SHIP_MODEL.name, ", ".join(new))
        _skip_warned.update(new)


def _note_tripoli_models_skipped() -> None:
    """Log once that the Tripoli-only models are left out for a non-Tripoli SHIP_MODEL."""

    _note_models_skipped(TRIPOLI_ONLY_MODELS)


def build_reference_models(ship_spec, ray_env) -> dict:
    """
    The non-surrogate reference models: UGent (Tripoli only; bank distances
    from `ray_env`, cast from the state passed to each call) and Fossen.
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
    One UAntwerpSurrogate per label (default: all UA_LABELS), in `labels`
    order, for the labels that have a preset for SHIP_MODEL (see
    UA_PRESETS_BY_SHIP); the others are left out with a warning.

    Each checkpoint is checked against its preset (scenario and ship) and
    the preset's ship against SHIP_MODEL.

    A slot whose checkpoint file does not exist (yet), or whose preset is
    not defined in fw/surrogate/presets.py, is left out with a warning, so
    the other models still run. A checkpoint that exists but does not match
    its preset still raises.

    Raises:
        ValueError: If a label is not one of UA_LABELS, or a checkpoint does
            not match its preset / SHIP_MODEL.
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

# RAY_MAX_DISTANCE is imported from fw.environments.ray_features (round 17)
# instead of being redefined here, so this script's bank-finding geometry
# (locate_point_at_distance) and ray diagnostics always use the same sensor
# range as the dataset generator and the UA_* surrogates.

# A bank_point closer than this to a map cut-off edge (see
# PySimEnv.bank_segments) is not on a real bank; _setup_geometry refuses it.
CUTOFF_BANK_TOLERANCE = 1.0          # m

# --------------------------------------------------------------------------
# Trajectory-visualization parameters (round 11) -- separate from the
# (v, distance) grid above: generate_trajectories fixes v at TRAJECTORY_SPEED
# and sweeps DISTANCES, rather than sweeping both, so the resulting plot
# shows how bank proximity alone changes each model's path at one speed
# (chosen by Walter -- the alternative, sweeping speed at a fixed distance,
# would show the opposite comparison).
# --------------------------------------------------------------------------

TRAJECTORY_SPEED = 3                  # fixed initial surge speed [m/s] for
                                       # every ship-bank trajectory -- not a
                                       # technical constraint. The wind test
                                       # uses its own WIND_TRAJECTORY_SPEED.
TRAJECTORY_TARGET_DISTANCE = 300.0    # [m] stop a trajectory once FH's OWN
                                       # cumulative path length reaches this
                                       # -- a tracked distance long enough to show
                                       # genuine divergence between models
                                       # without running past where FH itself
                                       # might reach open water/map edge.
                                       # Every model is stepped for the SAME
                                       # number of steps as FH took to reach
                                       # this (not its own path length), so
                                       # all 6 trajectories span identical
                                       # elapsed time at a given grid point.
TRAJECTORY_MAX_STEPS = 8000           # Safety cap (8000 * SURROGATE_DT =
                                       # 800 s) in case FH's own speed decays
                                       # toward zero (e.g. thrust too low to
                                       # make headway against current) and
                                       # target_distance is never reached --
                                       # without this, that would hang.
TRAJECTORY_OUTPUT_DIR = OUTPUT_DIR / "trajectories"

# Same categorical palette used in this script's own companion analysis
# plots (plot_comparison.py), kept identical here so a reader comparing a
# trajectory plot against an error-metric plot sees the same model -> color
# mapping in both, rather than two independently-chosen palettes.
TRAJECTORY_COLORS = {
    "FH": "#2a78d6",
    "UGent": "#eb6834",
    "Fossen": "#1baf7a",
    "UA_clean": "#eda100",
    "UA_confined": "#e87ba4",
    "UA_full": "#008300",
}

# Round 12 -- color alone isn't enough to tell two trajectories apart when
# they're genuinely close or overlapping. A distinct dash pattern per
# model means an exact overlap is still visible as a different pattern
# rather than vanishing under whichever line is drawn on top.
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
    """shapely Point for an (x, y) pair."""
    from shapely.geometry import Point
    return Point(float(xy[0]), float(xy[1]))


def find_bank_reference(
        env: PySimEnv,
        seed_point: Tuple[float, float],
        step: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Locate the nearest point on the river-polygon boundary to `seed_point`,
    and derive the unit vector pointing from the bank INTO open water at
    that point.

    Round 10: this used to also derive and return a "bank-following
    heading" (the boundary tangent direction, via atan2) for the caller to
    place the ship's heading with. That heading is NOT what run_comparison
    actually uses any more (see its own round-8 note) -- it uses
    ray_env.ship_heading (direction toward the next trajectory checkpoint)
    instead, since that's what matches Walter's own confirmed-correct
    heading value. Keeping the unused tangent heading around invited
    exactly the confusion an external review (ChatGPT) flagged: "is this
    calculated-but-unused value a bug?" It never fed anything -- only the
    tangent VECTOR (not its angle) was ever needed here, to derive
    `perp_into_water` below -- so the angle itself is no longer computed at
    all, and this function's return signature has dropped it.

    Args:
        env: PySimEnv instance whose `polygon_shape` (shapely Polygon,
            loaded from config.py's overall_contour/depth_contour) defines
            the navigable water area.
        seed_point: An (x, y) point near the port bank, supplied by the
            caller -- there is no way to derive this from nothing, since
            the polygon boundary has two banks (and possibly islands).
        step: Arc-length step (m) used for a finite-difference tangent
            estimate at the projected point.

    Returns:
        (bank_point, perp_into_water):
            bank_point: (2,) array, the boundary point nearest `seed_point`.
            perp_into_water: (2,) unit vector, perpendicular to the bank,
                pointing from `bank_point` into open water (verified via
                `polygon_shape.contains()`, NOT assumed from tangent
                rotation direction -- see module docstring caveat 1).
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

    # Candidate perpendicular (rotate tangent by -90 deg); direction
    # verified below via polygon_shape.contains(), not assumed.
    perp = np.array([tangent[1], -tangent[0]])

    probe = bank_point + perp * 1.0
    # env.polygon_shape.contains(...) == True means "in open water" (see
    # ship_in_open_water's docstring in py_sim_env.py) -- NOT "in an
    # obstacle". An earlier version of this function had this inverted.
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
    Find the point along `perp_into_water` from `bank_point` whose
    ray-cast distance back to the bank equals `target_distance`, via
    bisection.

    Uses `env.ray_cast_distance` (the same mechanism
    the shared ray features in fw.environments.ray_features use) rather than a straight-line
    Euclidean offset, so `target_distance` matches what the ray-cast
    proximity features (and, indirectly, the confined/full UA checkpoints)
    would actually measure at the returned point -- consistent with the
    real geometry rather than an assumed channel shape.

    Assumption (round 10, in response to an external review flagging this):
    bisection only finds a correct answer if measured_distance(pos)
    increases monotonically as `offset` grows from `bank_point` along
    `perp_into_water` -- true for a straight or gently-curved channel, but
    not guaranteed in general (e.g. a sharply re-entrant bank could make it
    non-monotonic). This isn't fixed here (a channel this irregular would
    need a different search strategy entirely, not a bigger hammer on
    bisection), but is now checked for -- see the two-point spot-check
    below -- since silently converging to the wrong point on a
    non-monotonic function is worse than a clear warning.

    Args:
        env: PySimEnv instance (see find_bank_reference).
        bank_point: (2,) point on the bank boundary.
        perp_into_water: (2,) unit vector from `bank_point` into open
            water (see find_bank_reference -- this also fixes the ray
            direction used to measure distance back to the bank: it is
            `-perp_into_water`, not an independently assumed rotation of
            the bank heading).
        target_distance: Desired perpendicular distance to the bank [m].
        tol: Acceptable error in the measured distance [m].
        max_iter: Maximum bisection iterations.

    Returns:
        (2,) point at approximately `target_distance` from the bank.

    Distances beyond RAY_MAX_DISTANCE are located too (with a longer
    measuring ray, out to 2 * target_distance, but never past the opposite
    boundary): the confined/full models' rays then no longer see THIS bank.
    The opposite bank may still be within ray range in a narrow stretch.

    Raises:
        RuntimeError: If the target is not bracketed within
            [0.5, search range] (e.g. the opposite bank is closer than
            `target_distance`), or bisection does not converge within
            `max_iter` iterations. The callers skip that distance.
    """

    # Ray direction back toward the bank -- derived from the empirically
    # verified perp_into_water, not an independently hardcoded rotation of
    # the bank heading (an earlier version of this function used
    # heading_rad - pi/2 here, which could disagree with perp_into_water).
    to_bank = -perp_into_water
    # Search/measuring range: the sensor range as before; longer only for
    # targets beyond it (the ray-cast would otherwise cap at RAY_MAX_DISTANCE).
    reach = RAY_MAX_DISTANCE if target_distance <= RAY_MAX_DISTANCE else 2.0 * float(target_distance)

    def measured_distance(pos: np.ndarray) -> float:
        """Ray-cast distance [m] from `pos` toward the bank, capped at `reach`."""

        origin = (float(pos[0]), float(pos[1]))
        end = (float(pos[0] + to_bank[0] * reach),
               float(pos[1] + to_bank[1] * reach))
        return float(env.ray_cast_distance(origin, end, max_distance=reach))

    lo, hi = 0.5, reach
    # Stay in this stretch of water: stop the search just before the opposite
    # boundary along perp_into_water (beyond it, the measured distance would
    # come from outside the river and the bisection could settle on land).
    # Needed for every target, also within RAY_MAX_DISTANCE: a waterway can
    # be narrower than the search range. In water wider than `reach` the ray
    # finds no boundary (width = reach) and `hi` stays `reach`.
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

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pos_mid = bank_point + perp_into_water * mid
        d_mid = measured_distance(pos_mid)
        if abs(d_mid - target_distance) <= tol:
            # Round 10: cheap two-point spot-check of the monotonicity
            # assumption this bisection relies on (see docstring above) --
            # a converged result can still be the wrong point if the
            # measured distance isn't actually increasing with offset here.
            # Warning only, not raised: pos_mid still satisfies
            # abs(d_mid - target_distance) <= tol either way, and a false
            # positive from the probe step being too small/large near a
            # boundary is more likely than genuine non-monotonicity.
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
    Print the ray-cast proximity features that point closest to the bank
    and closest to open water, converted back to metres, so they can be
    sanity-checked against `expected_distance` (the perpendicular
    distance-to-bank `locate_point_at_distance` placed this point at).

    Round 17: the features come from the single shared
    `fw.environments.ray_features.compute_ray_features` (the same function
    the dataset generator and the UA_* surrogates use), called with
    `ray_env` -- the same shared PySimEnv instance `locate_point_at_distance`
    itself used -- so this is purely a geometry check, independent of
    UKC/wind/current. The features are identical for every UA_* model, so
    they are computed and printed once; nothing is printed if no model in
    `ua_models` uses rays (UA_clean has none, see its own docstring).

    Ray layout (see ray_features.py): ray_k points k degrees CLOCKWISE from
    the bow, so ray_0 = ahead, ray_90 = starboard, ray_270 = port. For a
    ship sailing parallel to the bank, the "toward bank" ray should
    therefore be ray_90 (bank on starboard) or ray_270 (bank on port), with
    an angular offset close to 0 -- anything else means the heading and the
    bank direction are not parallel at this point.

    Each of the 24 rays is fired at a FIXED 15-degree step relative to
    heading, so there generally isn't a ray pointed EXACTLY at the bank or
    exactly into open water -- this prints whichever ray is CLOSEST in
    direction to each of those two references, labelled with its actual
    angular offset, next to a ground-truth straight-line ray-cast
    (`ray_env.ray_cast_distance`) along the EXACT bank-normal direction. A
    gap between the two is attributable to that angular offset, not
    necessarily a bug.
    """

    if not any(hasattr(model, "_env") for model in ua_models.values()):
        return  # only UA_clean (no rays) in play

    to_bank = -perp_into_water
    heading_rad = float(np.radians(heading_deg))
    # World-frame unit direction of every ray, from the shared module.
    ray_angles = ray_world_angles(heading_rad)
    ray_dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=1)  # (24, 2)

    toward_idx = int(np.argmax(ray_dirs @ to_bank))
    away_idx = int(np.argmax(ray_dirs @ perp_into_water))

    origin = (float(pos[0]), float(pos[1]))

    def _report(label: str, tag: str, feature: float, idx: int, ref_direction: np.ndarray) -> None:
        """
        Print one ray's feature, the distance it implies, its angular offset from `ref_direction` and
        the ground-truth ray-cast distance along `ref_direction`.
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
    Round 17: check a state against the dataset generator's own stop limits
    (`sim_fh_dataset_generator.stop_reason`: outside open water, within
    CUTOFF_CLEARANCE of a map cut-off edge, nearest ray hit closer than
    MIN_BANK_DISTANCE, UKC below MIN_KEEL_CLEARANCE). The generator never
    records a row in such a state, so the UA_* surrogates never saw one
    during training: their predictions there are extrapolation, and a large
    error vs. FH there says little about the surrogate itself.

    Diagnostic only -- nothing in this script is skipped or changed because
    of it. Uses the same shared ray features (and the same `ray_env`
    polygon) as the generator.

    Returns:
        None if the state is inside the training domain, otherwise the
        generator's stop reason ("left_open_water", "near_cutoff", "bank"
        or "ukc").
    """

    rays = compute_ray_features(ray_env, (float(x), float(y)), float(heading_rad))
    return stop_reason(ray_env, np.array([x, y], dtype=np.float64), float(ukc), rays, cutoff_geom)


# --------------------------------------------------------------------------
# FH session (thin wrapper around the real simexe engine)
# --------------------------------------------------------------------------

class FhSession:
    """
    Drives a single, real FH (simexe) simulation for one grid point.

    Math-model bootstrapping (Config/Exercise/MathModel construction,
    rewind configuration, enableBridge(), the initial settle) is delegated
    entirely to `fw.simulators.fh_sim_bridge.FhSimBridge` -- Walter's own,
    production FH engine bridge -- rather than reimplemented here.

    An earlier version of this class hand-rolled that bootstrapping
    directly (its own Config/Exercise/MathModel calls) and got the rewind
    configuration wrong: it called `config.SetRewindEnabled(False)` and
    never called `config.SetRewindConfig(...)` at all -- `FhSimBridge`
    always pairs `SetRewindEnabled(True)` with `SetRewindConfig(...)`, so
    this was still worth fixing on its own merits, even though it turned
    out NOT to be the cause of the native `MathModel.Initialize()` crash
    this class has been chasing across several rounds since -- see the
    NOTE near the top of `__init__` for the full, still-unresolved history
    and the module docstring's own NOTE for the current overall status.

    This also matches the framework's own real usage pattern for FH, as
    seen in `fw.environments.py_sim_env.PySimEnv`: construct an
    `FhSimBridge` externally, set `.initial_vessel_state`, call
    `.reset()` once, then wrap it with `FhTabularModel(ship_spec, bridge)`
    -- `FhSimBridge` already exposes the `.ship_model` / `.ship_interface`
    / `.math_model` shape `FhTabularModel` expects, so it is passed
    directly as `FhTabularModel`'s `fh_engine` argument; FhSession no
    longer needs to expose that shape itself.

    Per-step control application and time-advancement (`advance()`) are
    delegated to `FhTabularModel.step()`, as before.

    Call `close()` after finishing with a session (see `run_comparison`),
    to `dispose()` the underlying FhSimBridge and release its native FH
    resources -- the earlier hand-rolled version never disposed of its
    MathModel instances, which leaks native resources across a
    multi-hundred-point grid.
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
        Create the FH bridge, set the initial state and controls, and reset it (see the class docstring).

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

        # rewind_enabled (added for model_comparison_wind.py): None keeps
        # FhSimBridge's default (rewind on). FhSession never rewinds (one
        # session per grid point), so False is safe; it configures FH exactly
        # like sim_fh_dataset_generator (SetRewindEnabled(False)).
        # wind_field (added for model_comparison_wind.py): optional
        # (rotation_deg, scale) for the exercise's wind-field transform. It is
        # handed to FhSimBridge.initial_wind_field_transform BEFORE reset(),
        # because FH ignores a transform set after its first simulateSeconds()
        # (confirmed by the first wind run: every probe gave the same wind).
        # None leaves the exercise's own wind untouched.
        # exercise_path (added for model_comparison_wind.py): None keeps the
        # exact confirmed-working construction below (FhSimBridge's own
        # default exercise). When given, it is forwarded to FhSimBridge's own
        # `exercise_path` argument -- used by the wind comparison to load a
        # wind-only (no current) Tripoli exercise.
        # Traced 1:1 against fw.configuration.env_factory.FhSimEnvironment +
        # fw.environments.py_sim_env.PySimEnv's actual FH_TABULAR construction
        # (the one confirmed, by Walter, to run Tripoli + wind/current
        # successfully end-to-end via fw.main.py, 2026-09-30). Two earlier
        # attempts at this class each still hit the same native
        # "Cannot open input file: .../os2_body" MathModel.Initialize()
        # failure, so this round matches every step of that real sequence
        # exactly, in the same order, rather than approximating it:
        #
        #   1. env_factory.py: `FhSimBridge(ship_model=..., scenario=...)` --
        #      ONLY these two args. No library_path/config_path/exercise_path
        #      (all left to FhSimBridge's own defaults -- an earlier version
        #      of this class re-derived and passed those explicitly), and no
        #      output_path either (an earlier version passed a custom,
        #      per-grid-point one -- dropped here too, now that it's the last
        #      remaining unmatched argument; every FhSimBridge instance below
        #      shares FhSimBridge's own default output_path as a result,
        #      which is a one-at-a-time, non-concurrent reuse since each
        #      session is closed before the next grid point's session is
        #      constructed).
        #   2. PySimEnv.create_simulator(): `FhTabularModel(ship_spec,
        #      fh_bridge)` is constructed BEFORE the bridge is ever reset (it
        #      just stores the reference -- see FhTabularModel.__init__).
        #   3. PySimEnv._initialize_simulation(): builds the initial
        #      VesselState (x, y, heading, u, v=0, r=0) exactly as below.
        #   4. PySimEnv.__init__ (immediately after _initialize_simulation()):
        #      `self._fh_sim.initial_vessel_state = <that VesselState>`.
        #   5. Only later, on the first gym-level `env.reset()`, is
        #      `self._fh_sim.reset()` actually called (via
        #      PySimEnv.env_specific_reset()) -- i.e. `initial_vessel_state`
        #      is always set well before `reset()` is called, never after or
        #      interleaved with anything else, which this class also does.
        #
        # scenario is passed as config.py's own 'scenario' value
        # ("DDShip_scenario1_nowindnocurrent") purely for exact parity with
        # env_factory.py's `scenario=self.configuration["scenario"]"` --
        # FhSimBridge ignores it for ship_model=TRIPOLI regardless (its
        # scenario_suffix is hardcoded per Walter's own fh_sim_bridge.py).
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
        # Round 7: without this, the ship's initial surge velocity (u=v_init
        # above) and its initial propeller RPS were inconsistent -- u set to
        # a real cruising speed while the propeller itself started at
        # ship_config's own default (effectively stationary). Mirrors
        # sim_fh_dataset_generator.py's own `ship_config.setInitialPropellerRps
        # (_PROPELLER_RPS_SCALE * action[1])` usage, and Walter's own
        # instruction to set it "around" the setInitialPosition/setInitial*
        # velocity calls in FhSimBridge._initialize_math_model -- see that
        # method (and FhSimBridge.initial_propeller_rps) for where this is
        # actually applied. `thrust` here is the same normalized throttle
        # value passed to FhTabularModel.step() every advance() call below,
        # so the initial condition and the ongoing per-step control agree.
        # Fail fast if FhSimBridge does not support the initial-condition hooks
        # set below: assigning them to an older bridge just creates unused
        # attributes, and FH silently keeps the exercise defaults.
        for hook in ("initial_propeller_rps", "initial_rudder_angle", "initial_wind_field_transform"):
            if not hasattr(self._bridge, hook):
                raise RuntimeError(
                    f"FhSession: FhSimBridge has no '{hook}' hook -- the installed "
                    f"fw/simulators/fh_sim_bridge.py is an older version that would ignore it."
                )
        self._bridge.initial_propeller_rps = float(_PROPELLER_RPS_SCALE * thrust)
        # Round 16 -- without this, the ship's initial rudder angle during
        # FhSimBridge's own internal 0.125 s settle (see reset() below) was
        # left at ship_config's own unspecified default, not this script's
        # own RUDDER_ANGLE_RAD=0.0 -- FhSimBridge never called
        # setInitialRudderAngle at all before this was added (see
        # FhSimBridge.initial_rudder_angle). RUDDER_ANGLE_RAD is only
        # actually applied starting from the first explicit advance() call
        # below (via FhTabularModel.step()), AFTER the "before" state this
        # script reads is already captured -- setting it here too makes the
        # ship sail straight (rudder=0) from the very start of the settle,
        # not just from the first advance() onward, matching this script's
        # own fixed-rudder-angle design throughout.
        # FH's own sign (fh_rudder_sign), like every later command via FhTabularModel.
        self._bridge.initial_rudder_angle = float(fh_rudder_sign(SHIP_MODEL) * np.degrees(RUDDER_ANGLE_RAD))
        if wind_field is not None:
            rotation_deg, scale = wind_field
            self._bridge.initial_wind_field_transform = (float(rotation_deg), float(scale), 0.0)
        # First reset() on a fresh bridge runs the full
        # Config/Exercise/MathModel.Initialize()/enableBridge() sequence
        # and settles for 0.125 s internally (matching INITIALIZATION_TIME
        # above) -- see FhSimBridge._initialize_math_model. Raises
        # RuntimeError itself if Initialize() fails or exercise_path
        # doesn't exist. Any failure disposes the bridge: no FhSession object
        # will exist, so nobody else could close() it.
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
        Fail-fast guard: the exercise's ship must have the controls
        FhTabularModel.step() will drive for this bridge's ship model --
        azipods for the TRIPOLI, rudders AND propellers for any other ship
        (as the Myzako dataset generators). Otherwise step() would silently
        apply nothing (e.g. SHIP_MODEL = MYZAKO_37 with a Tripoli exercise).
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
        """FH ship interface of the underlying bridge."""
        return self._bridge.ship_interface

    def close(self) -> None:
        """Releases this session's native FH resources. Safe to call
        multiple times (FhSimBridge.dispose() is idempotent)."""
        self._bridge.dispose()

    def state(self) -> Tuple[float, float, float, float, float, float]:
        """
        Returns (x, y, heading_rad, u, v, r) in this codebase's convention
        -- exactly as sim_fh_dataset_generator.run_simulation builds its
        state: heading via fh_heading_to_env_rad (round 17: the generator's
        own conversion, so a change of FH_HEADING_CONVENTION there applies
        here too), yaw_rate via radians(...), and sway positive to port
        (vog.x is surge, vog.y is sway; FH's body frame is the framework's).
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
        """Apply the held controls (rudder RUDDER_ANGLE_RAD, fixed thrust) and advance FH by `dt` seconds."""

        # Applies the held control (rudder=0, RUDDER_ANGLE_RAD converted to
        # degrees; fixed thrust) and advances the FH engine by dt seconds,
        # via FhTabularModel.step() -- see class docstring. The returned
        # (u, v, r) is discarded here: state() below re-reads it directly
        # from the engine afterwards, keeping state() the single place that
        # reads position and velocity together (used for both the "before"
        # and "after" reads that FH's du/dv/dr are finite-differenced from).
        self._fh_model.step(
            state=None,
            rudder_angle=RUDDER_ANGLE_RAD,
            thrust=self._thrust_normalized,
            dt=dt,
        )

    def get_ukc(self) -> float:
        """FH's current keel clearance (UKC) [m]."""
        return float(self.ship_interface.getKeelClearance())

    def get_wind(self) -> Tuple[float, float]:
        """FH's WindState at the ship: (speed [m/s], direction [rad], wrapped with PySimEnv.normalize_angle)."""
        ws = self.ship_interface.getWindState()
        return float(ws.Speed), float(PySimEnv.normalize_angle(np.radians(ws.Direction)))

    def get_current(self) -> Tuple[float, float]:
        """FH's current at the ship: (speed [m/s], direction [rad], wrapped with PySimEnv.normalize_angle)."""
        speed, direction_deg = get_current_velocity(self.ship_interface)
        return float(speed), float(PySimEnv.normalize_angle(np.radians(direction_deg)))

    # ---- wind helpers (added for model_comparison_wind.py) -------------------

    def get_wind_raw(self) -> Tuple[float, float]:
        """(speed [m/s], direction [deg]) exactly as FH's WindState reports
        them -- FH's own native direction convention, no conversion."""
        ws = self.ship_interface.getWindState()
        return float(ws.Speed), float(ws.Direction)

    def get_fh_heading_deg(self) -> float:
        """FH's own native heading [deg] (getShipHeading), no conversion --
        used together with get_wind_raw() to form a relative wind angle in
        one and the same (FH) frame."""
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
    Override a UAntwerpSurrogate instance's
    environmental/ray sourcing to use live FH values (and a shared PySimEnv
    for ray-casting) instead of its own WcuLookup CSV lookup and its own
    separately-constructed PySimEnv -- per request, so UKC/wind/current/
    rays are the actual live values FH is simulating at this grid point,
    not a nearest-neighbor CSV lookup or a second, independent polygon
    instance.

    This works by replacing two INSTANCE attributes (not touching the
    class itself), which is enough because every one of these classes'
    `_build_input_vector` reads environmental data exclusively through
    `self.get_env(pos)` and rays exclusively through the shared
    `fw.environments.ray_features.compute_ray_features(self._env, pos, heading)`:

      - `model.get_env` is replaced with a plain function that always
        returns the given (ukc, wind_speed, wind_dir, current_speed,
        current_dir) as an EnvironmentData, regardless of `pos`. An
        instance attribute shadows the class method and is called
        unbound (no implicit `self`), so a function of `(pos, state=None)`
        -- the class method's signature without `self` -- is correct here.
        The values are used as given (no conversion for the model's state).
      - `model._env` (the class's own internal PySimEnv, used only for
        ray-casting) is replaced with the shared `ray_env` -- the SAME
        PySimEnv instance this script already uses for its own
        bank-finding geometry, so the classes' ray-casting and this
        script's bank/distance geometry can never disagree about which
        polygon or coordinate frame is in play.

    UA_clean (an open_water UAntwerpSurrogate) has no `_env` attribute and never
    calls `get_env` (its input row only uses
    heading/surge/sway/yaw_rate/rudder/thrust -- UKC and rays are left as
    zero placeholders that select_and_encode never reads for that
    checkpoint's trained feature set). Patching it here is a harmless
    no-op for that class, kept for uniformity rather than special-cased.

    Args:
        model: A UAntwerpSurrogate instance.
        ukc, wind_speed, wind_dir, current_speed, current_dir: Live values
            read from the same FH session used for ground truth at this
            grid point (see run_comparison).
        ray_env: The shared PySimEnv instance used for this script's own
            bank-finding geometry (ray_env in run_comparison).
    """

    env_data = EnvironmentData(
        ukc=ukc, wind_speed=wind_speed, wind_direction=wind_dir,
        current_speed=current_speed, current_direction=current_dir,
    )
    model.get_env = lambda pos, state=None: env_data
    model.ukc = ukc  # get_ukc() returns self.ukc when there is no lookup
    # Drop the lookup so get_ukc(pos) cannot bypass the injected environment.
    if hasattr(model, "wcu_lookup"):
        model.wcu_lookup = None
    if hasattr(model, "_env"):
        model._env = ray_env


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _setup_geometry(seed_point: Tuple[float, float], log_prefix: str = "run_comparison"):
    """
    Shared bank-finding/heading setup used by both `run_comparison` and
    `generate_trajectories` (round 11) -- pulled out of `run_comparison`
    verbatim (no behavior change) so the trajectory-visualization feature
    doesn't duplicate this geometry logic, which would risk the two
    functions silently disagreeing about bank_point/heading_deg for the
    same seed_point.

    Round 17: also returns the map cut-off geometry (`cutoff_geom`, from
    PySimEnv.bank_segments via the dataset generator's own helper, or None
    if unavailable) for the training-domain checks, and refuses a
    `seed_point` whose nearest boundary point lies on a map cut-off edge
    rather than on a real bank -- all the distance-to-bank geometry would
    then be measured to an artificial map edge.

    Returns:
        (ray_env, bank_point, perp_into_water, heading_deg, ship_spec, cutoff_geom)
    """

    ship = create_ship(SHIP_MODEL)
    ship_spec = ship.specifications

    # One PySimEnv instance reused for this script's own bank-finding
    # geometry across the whole grid -- construction is not cheap and the
    # polygon doesn't change between grid points. Also injected into every
    # UA_confined/UA_full model below (see _inject_live_environment) in
    # place of each class's own separately-constructed PySimEnv, so there
    # is only ever one polygon/coordinate frame in play (see module
    # docstring caveat 1).
    #
    # start_pos=seed_point is required here (round 8) for the heading fix
    # below -- without it, PySimEnv defaults initial_ship_pos to a
    # meaningless (5.0, 5.0) placeholder, which would make ray_env.ship_heading
    # (computed toward checkpoints[1] from initial_ship_pos) equally
    # meaningless. ship_model=SHIP_MODEL is passed for consistency with the
    # ship actually being simulated below, though checkpoint/heading
    # construction itself does not depend on it (verified: it only depends
    # on the trajectory CSV, CHECKPOINTS_DISTANCE, and initial_ship_pos).
    ray_env = PySimEnv(
        render_mode=None, time_step=SURROGATE_DT, wind=False, current=False,
        start_pos=np.array(seed_point, dtype=np.float32), ship_model=SHIP_MODEL,
    )

    # bank_point/perp_into_water (the perpendicular distance-to-bank axis)
    # are the only geometry needed from here -- see find_bank_reference's
    # own docstring (round 10) for why it no longer also returns a bank-
    # tangent heading.
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

    # Round 8 -- ROOT CAUSE of both the wrong heading AND (most likely) the
    # os2_body/Initialize() crash: find_bank_reference's heading was the
    # river-polygon BOUNDARY TANGENT (32.084 deg for this seed_point) --
    # visibly wrong per Walter's own known-good heading for this seed_point,
    # 55.28704136874908 deg. That number is reproduced here to 6 significant
    # figures (55.28704478 deg) by constructing a PySimEnv with
    # start_pos=seed_point and reading its own `ship_heading` -- i.e. the
    # SAME calculation real production code uses in
    # PySimEnv._initialize_simulation(): the direction from seed_point
    # toward the second evenly-spaced TRAJECTORY checkpoint (parallel to the
    # shipping channel/route), not parallel to the river bank boundary.
    # Those two directions are only roughly similar (a bank runs roughly
    # alongside its channel, not exactly), which is why 32.084 and 55.287
    # looked superficially plausible as the same kind of number without
    # actually being close enough to be right.
    heading_deg = float(np.degrees(ray_env.ship_heading))
    if heading_deg < 0:
        heading_deg += 360.0

    # Diagnostic logging (round 5): diagnose_fh_session_at_known_good_position()
    # proved FhSession itself is fine at seed_point == DEFAULT_FRAMEWORK_SHIP_POS
    # == (2245.0, -47.0), so the remaining suspect is THIS derived geometry --
    # how far find_bank_reference's projected bank_point sits from seed_point,
    # and what heading_deg it derives, before any grid-point offsetting even
    # happens. Left in deliberately (not removed once this is diagnosed) since
    # it is one line per run and cheap context for any future seed_point.
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
    Run the full (v, distance) grid comparison for SHIP_MODEL and write:
      - one CSV per (distance, v) combination to OUTPUT_DIR (as before), and
      - one combined `all_results.csv` across the whole grid (round 10),
        with each model row already carrying its (du_error, dv_error,
        dr_error) vs. FH at that same point.
    A point skipped because FhSession construction failed (see round 9,
    inside `_run_grid`) contributes no rows to either output.

    This does NOT generate trajectory plots -- see `generate_trajectories`
    (round 11) for that, a separate entry point since it runs a different
    kind of simulation (multi-step rollouts vs. this function's single-step
    evaluation at every grid point -- see module docstring for why).

    Args:
        seed_point: An (x, y) point near the port bank (see
            find_bank_reference) -- required, no default. Must be supplied
            by the caller for the area the exercise file (resolved below)
            actually simulates (ScheldeSaeftinge_23_002).
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ray_env, bank_point, perp_into_water, heading_deg, ship_spec, cutoff_geom = _setup_geometry(seed_point)

    # UGent (Tripoli only) and Fossen; UGent's bank distances come from the
    # shared ray_env, at the state passed to each _compute_accelerations call.
    ref_models = build_reference_models(ship_spec, ray_env)

    # One UAntwerpSurrogate per preset (UA_PRESETS_BY_SHIP), each loading its
    # preset's checkpoint. Their own environmental/ray sourcing (WcuLookup
    # CSV lookup, own PySimEnv ray-casting) is overridden per grid point, in
    # the loop below, via _inject_live_environment -- see module docstring
    # NOTE. UA_full is left out of the grid, as before.
    ua_models = build_ua_models(ship_spec, ("UA_clean", "UA_confined"))

    # Round 10 (in response to an external review flagging that this script
    # writes 36 separate per-point CSVs with no built-in error metrics,
    # leaving every comparison to an ad-hoc external script): also write
    # ONE combined CSV across the whole grid, with each model row's error
    # vs. FH's own (du, dv, dr) at that same point already computed --
    # error columns are 0.0 for the FH row itself (FH is its own
    # reference), not blank/NaN, so the column stays numeric throughout.
    # This is in ADDITION to the per-point CSVs below, which are unchanged
    # in name/location and also gain the same error columns for
    # consistency -- nothing existing is removed.
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
    """Body of the (distance, v) grid loop, split out of run_comparison so
    `summary_file` above always gets closed via try/finally regardless of
    how the loop ends (round 10, added alongside the combined all_results.csv
    output)."""

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
        # Sanity-check the ray-cast geometry at this grid point once (it's
        # the same `pos` for every v below -- ray features don't depend on
        # speed) -- see _print_bank_ray_diagnostics' own docstring.
        _print_bank_ray_diagnostics(
            ua_models, pos, heading_deg, perp_into_water, ray_env,
            expected_distance=float(distance), log_prefix="run_comparison",
        )

        for v in V_VALUES:
            thrust = float(min(1.0, v / 6.0))

            run_id = f"dist{distance}_v{v}"
            # Round 9: constructing FhSession itself can fail (RuntimeError,
            # MathModel.Initialize() returning False) for grid points that
            # are simply not achievable for this ship/scenario -- most
            # plausibly a genuine "too close to the bank/too shallow" limit
            # at small distance values, not a bug in this script (Walter's
            # own read on the first occurrence of this: the ship likely
            # sailed into/too close to the bank). Previously this crashed
            # the whole grid, losing every remaining (distance, v)
            # combination after the first failure. Now: skip only this one
            # point, log why, and keep going -- so e.g. a hard failure at
            # distance=10 doesn't cost the distance=5 results too, and
            # doesn't lose a full re-run of everything that already
            # succeeded.
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
                # FH's own live values at the "before" state. These ARE fed
                # to every UA_* model below (via _inject_live_environment),
                # not just logged for reference -- see module docstring
                # NOTE.
                fh_ukc = fh.get_ukc()
                fh_wind_speed, fh_wind_dir = fh.get_wind()
                fh_current_speed, fh_current_dir = fh.get_current()

                # Round 17: flag (not skip) grid points the dataset
                # generator would never have recorded -- see
                # _training_domain_issue.
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

                # Each UA_* class still builds its own input vector
                # internally, but its environmental/ray sourcing is
                # redirected to these live FH values first (see module
                # docstring NOTE and _inject_live_environment) --
                # ray-casting also happens at `state`'s own (x, y,
                # heading), i.e. the same post-settle position/heading
                # used for every other input, via the shared ray_env
                # instance.
                for label, model in ua_models.items():
                    _inject_live_environment(
                        model, fh_ukc, fh_wind_speed, fh_wind_dir,
                        fh_current_speed, fh_current_dir, ray_env,
                    )
                    du, dv, dr = model._compute_accelerations(state, RUDDER_ANGLE_RAD, thrust)
                    results.append((label, du, dv, dr))

                # Error vs. FH's own (du, dv, dr) at this same point -- 0.0
                # for the FH row itself. Computed once here and written to
                # both the per-point CSV and the combined all_results.csv
                # below, so nothing needs recomputing downstream just to
                # compare a model against FH.
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
                # Each grid point owns its own FhSimBridge (fresh
                # MathModel), which holds native FH resources -- release
                # them before the next grid point instead of accumulating
                # ~36 undisposed instances (DISTANCES x V_VALUES) over the
                # full grid. Wrapped in try/finally so a failure mid-point
                # still disposes of what was already constructed.
                fh.close()


# --------------------------------------------------------------------------
# Trajectory visualization (round 11) -- separate from run_comparison's
# single-step (v, distance) grid above. See module docstring's top section
# for why FH is finite-differenced over one SURROGATE_DT step there rather
# than integrated: this function is the integrated counterpart, built for
# visualizing path shape rather than measuring instantaneous (du, dv, dr)
# agreement.
# --------------------------------------------------------------------------

def generate_trajectories(
        seed_point: Tuple[float, float],
        speed: float = TRAJECTORY_SPEED,
        distances=DISTANCES,
        target_distance: float = TRAJECTORY_TARGET_DISTANCE,
        max_steps: int = TRAJECTORY_MAX_STEPS,
) -> None:
    """
    For a fixed initial speed and each perpendicular distance-to-bank in
    `distances`, roll out a genuine multi-step trajectory for all 6 models
    (FH + the 5 models run_comparison also evaluates) from the same
    starting point/heading run_comparison would use, and plot them overlaid
    on the river-bank geometry.

    Unlike run_comparison (one instantaneous (du, dv, dr) evaluation per
    grid point), this integrates every model forward in time:

      - FH: the real engine, via repeated FhSession.advance(SURROGATE_DT) +
        fh.state() reads -- genuinely simulated, not approximated.
      - UGent, Fossen, UA_clean, UA_confined, UA_full: each wrapped in its
        OWN `fw.simulators.ship_simulator.ShipSimulator` instance (Walter's
        own production integrator, per his own request -- see the
        ShipSimulator import comment above for why this was used instead
        of a hand-rolled position-integration formula), stepped with
        `ShipSimulator.integrate()` directly (bypassing `.step()`'s
        actuator-dynamics/action-mapping layer, since this script -- like
        run_comparison -- always wants a FIXED physical rudder_angle=
        RUDDER_ANGLE_RAD and thrust, not a normalized RL action run through
        rate-limiting/smoothing). Each model keeps its own ShipSimulator
        (and therefore its own running VesselState), so the 5 trajectories
        are free to diverge from each other and from FH over time -- this
        is the point of the plot.

    UA_clean/UA_confined/UA_full's environmental/ray sourcing is
    overridden every step via `_inject_live_environment`, exactly as
    run_comparison does per grid point -- but here it is called EVERY step,
    using FH's live (ukc, wind, current) at FH's OWN current position at
    that step, while rays are cast from each UA_* model's OWN current
    (already-diverged) position/heading (via its ShipSimulator's running
    state, fed to `_compute_accelerations`). This is a deliberate
    extension of run_comparison's existing single-step design (same
    FH-sourced environment, model's-own-state ray casting) to multiple
    steps, not a new assumption -- but it does mean the UA_* models are
    fed UKC/wind/current as experienced by FH's path, not their own, which
    could matter for a long rollout at a confined/full checkpoint if the
    two paths diverge enough to be in meaningfully different water. Worth
    keeping in mind when interpreting UA_confined/UA_full specifically.

    All 6 models are run for the SAME number of steps per grid point --
    however many FH itself took to cover `target_distance` (capped at
    `max_steps`) -- not however far each model's own path happens to be,
    so all 6 trajectories at a given grid point span identical elapsed
    time, making them directly comparable in the plot.

    Writes one CSV per (distance, speed) to TRAJECTORY_OUTPUT_DIR (columns:
    model, step, x, y) and one combined overview PNG
    (OUTPUT_DIR/trajectories_overview.png, one subplot per distance) if
    matplotlib is importable; otherwise only the CSVs are written, with a
    printed note explaining why no PNG was produced.

    Args:
        seed_point: Same meaning as in run_comparison.
        speed: Fixed initial surge speed [m/s] for every trajectory.
        distances: Perpendicular distances-to-bank to generate a
            trajectory for, one subplot each.
        target_distance: Stop a trajectory once FH's own cumulative path
            length reaches this [m] (see TRAJECTORY_TARGET_DISTANCE).
        max_steps: Safety cap on steps per grid point (see
            TRAJECTORY_MAX_STEPS).
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

        # Same per-point resilience as run_comparison (round 9): a point
        # that's too close to the bank for FH to even initialize should
        # not abort every other distance's trajectory.
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

        # UGent (Tripoli only): bank distances from the shared ray_env, cast
        # from UGent's OWN (diverging) position/heading at every integrate()
        # call -- same model's-own-state convention as the UA_* ray casting.
        non_fh_models = {
            **build_reference_models(ship_spec, ray_env),
            **ua_models,
        }

        # Sanity-check the ray-cast geometry at FH's own actual settled
        # starting pose (x0, y0, heading0) for this distance -- see
        # _print_bank_ray_diagnostics' own docstring. Printed once per
        # distance, at the start of the trajectory, not every rollout step.
        _print_bank_ray_diagnostics(
            ua_models, (x0, y0), float(np.degrees(heading0)), perp_into_water,
            ray_env, expected_distance=float(distance),
            log_prefix="generate_trajectories",
        )

        # One ShipSimulator per non-FH model, each seeded at FH's own
        # initial state (so every trajectory starts from the identical
        # point/heading/velocity FH itself started this grid point from),
        # wind=False/current=False to match this script's own FH scenario
        # (see the ShipSimulator import comment for the one caveat this
        # setting happens to paper over for UA_clean/confined/full).
        sims = {}
        for label, model in non_fh_models.items():
            sim = make_model_simulator(ship_spec, model)
            # Each sim gets its own VesselState instance (not a shared
            # reference to init_state) -- ShipSimulator.reset() stores it
            # directly as self._state and integrate() replaces it with a
            # freshly-constructed VesselState every step anyway, so sharing
            # a single instance here wouldn't itself be unsafe, but a
            # dedicated dataclass instance per sim is clearer than relying
            # on that.
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
        # Round 17: first step at which FH's own state leaves the dataset's
        # training domain (see _training_domain_issue) -- reported once per
        # distance, the trajectory itself is NOT cut short.
        domain_exit = None
        fh_heading = heading0

        try:
            while fh_path_length < target_distance and step_i < max_steps:
                # FH's own live environment at ITS current position/time --
                # fed to every UA_* model this step, same as run_comparison
                # does per grid point (see module docstring NOTE and
                # _inject_live_environment's own docstring).
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

        # Round 12 -- quick numeric check (not just the plot) that every
        # model actually moved laterally: perpendicular distance of each
        # model's FINAL point from the straight line through the start
        # point at the initial heading. A model without bank terms (or
        # without bank information) under this script's fixed
        # rudder_angle=0.0 and v0=r0=0 reports ~0.0 m here.
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
    One PNG, one subplot per distance in `trajectories_by_distance`, each
    showing every model's path overlaid on the river-bank boundary for
    visual context. Skipped (with a printed note, not an exception) if
    matplotlib isn't importable -- the CSVs generate_trajectories already
    wrote are still usable without it.
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
    # Round 12 fix: with few distances (e.g. ncols=1 after most grid points
    # were skipped), 6.0*ncols made the figure too narrow for the 6-entry
    # legend below, clipping it off the right edge -- caught the same way
    # as the round-11 layout bugs, by actually rendering a 1-distance case
    # before delivery, not by inspection. A floor on the total width fixes
    # it without changing the (already-correct) multi-distance layout.
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
    # Bottom margin sized for both the per-subplot x-axis labels AND the
    # figure-level legend below them (round 11 fix: an earlier version
    # reserved only 4% here, which overlapped the legend with the x-axis
    # labels of the bottom row of subplots -- see the smoke-tested PNG this
    # was caught in before delivery). hspace is widened for the same
    # reason: with >3 distances (2 rows), the default spacing let row 2's
    # titles collide with row 1's x-axis labels -- also only caught by
    # actually rendering a 4-distance test case, not a 2-distance one.
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.subplots_adjust(hspace=0.35)

    png_path = OUTPUT_DIR / "trajectories_overview.png"
    try:
        # Round 13: on Windows, passing a pathlib.Path straight to
        # fig.savefig() can raise "OSError: [Errno 22] Invalid argument"
        # deep inside Pillow's PNG writer (Figure.savefig ->
        # FigureCanvasAgg.print_png -> PIL.Image.save -> open(filename,
        # "w+b")) even though the exact same directory just accepted the
        # six trajectory CSVs above via plain `open()` -- a known
        # Pillow/Windows path-handling quirk with certain Pillow versions'
        # handling of WindowsPath objects specifically in this call chain,
        # not a sign the directory or permissions are wrong. Passing a
        # plain str sidesteps it. Confirmed as the fix on Walter's own
        # Windows/PyCharm setup, where this crashed `generate_trajectories`
        # after all 6 distances' CSVs had already been written successfully
        # -- also now caught so a plotting-only failure can't discard that
        # otherwise-complete run.
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
# Diagnostic (NOT part of the grid comparison) -- see module docstring NOTE,
# round 4.
# --------------------------------------------------------------------------

def diagnose_fh_initialization() -> None:
    """
    Isolates whether the recurring "os2_body" / MathModel.Initialize()
    failure is caused by something in THIS script's code, or by something
    about the process/environment this script runs in that `fw.main.py`'s
    process does not share (working directory, environment variables,
    antivirus/cache-locking timing, import order, etc.) -- see the module
    docstring's NOTE for the full history. FhSession's own construction
    has already been traced to match `fw.main.py`'s FH_TABULAR path
    argument-for-argument (round 3) and still failed, so this stops trying
    to out-guess that construction and instead runs the ACTUAL known-
    working code path -- `fw.configuration.env_factory` +
    `fw.environments.py_sim_env.PySimEnv` -- directly, inside this same
    script's process.

    Two possible outcomes:
      - This ALSO fails the same way: proves the cause is not in any of
        this script's code (FhSession/FhSimBridge construction has already
        been matched 1:1 -- see round 3 above), but in something about
        this process/environment that `fw.main.py`'s process does not
        share. The fix at that point belongs outside this script (working
        directory, environment variables, antivirus exclusions for the
        simexe cache folder, etc.), not in more Python changes here.
      - This SUCCEEDS: would mean something in this script's own imports
        or setup still differs in a way not yet identified, worth
        investigating further from there.

    Call this manually, before run_comparison(), to test it in the same
    run where FhSession has been failing:

        from fw.surrogate.validation.model_comparison import diagnose_fh_initialization
        diagnose_fh_initialization()
    """

    from fw.dynamics.dynamics_model import Model
    from fw.configuration.config import get_config_parameters
    from fw.configuration import env_factory

    framework_config = dict(get_config_parameters())
    # Overridden explicitly rather than relying on whatever config.py's
    # CURRENT defaults happen to be (e.g. env_type/wind/current/ship_model
    # may have been reverted since the confirmed-working run), so this
    # diagnostic always exercises FH_TABULAR + Tripoli + wind/current
    # regardless of config.py's present state.
    framework_config["ship_model"] = SHIP_MODEL
    framework_config["dynamics_model"] = Model.FH_TABULAR
    framework_config["wind"] = True
    framework_config["current"] = True

    env_factory.initialize_all_environments()
    env_wrapper = env_factory.build_default_environment(
        env_factory.fh_sim_env_name, framework_config
    )

    print(
        "diagnose_fh_initialization: calling PySimEnv.reset() "
        "(FH_TABULAR / Tripoli / wind+current) via the exact construction "
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
    Round 4 result: diagnose_fh_initialization() SUCCEEDED using
    `fw.common.default_positions.DEFAULT_FRAMEWORK_SHIP_POS` ==
    `(2245.0, -47.0)` -- the exact value Walter has been passing as
    `seed_point` to `run_comparison()` -- as the ship's ACTUAL starting
    position (config.py's own default `start_pos`, left untouched by that
    diagnostic).

    But `run_comparison()` never places the ship AT `seed_point` -- it only
    uses it to SEED `find_bank_reference()`, which projects it onto the
    nearest river-boundary point, then `locate_point_at_distance()` offsets
    that by 5-30 m into open water (see those two functions above). The
    (x, y) actually handed to `FhSession` for each grid point is therefore
    a DERIVED point, not `(2245.0, -47.0)` itself -- in Walter's own
    earlier traceback it was x=2128.4979139552474, y=26.037217255416262,
    heading_deg=32.08432624764385, quite different from the seed.

    This function isolates whether that derivation -- not FhSession's
    construction (already proven fine by diagnose_fh_initialization()) --
    is the actual cause: it calls FhSession directly, the SAME class
    run_comparison() uses, but AT the confirmed-good `(2245.0, -47.0)`
    position instead of a bank-derived one.

      - If this SUCCEEDS: confirms the bank-derived (x, y)/heading
        combination is what MathModel.Initialize() is choking on, not
        FhSession/FhSimBridge construction at all -- next place to look is
        `find_bank_reference`/`locate_point_at_distance` (is the derived
        point actually inside the exercise's simulated/navigable area? is
        the heading convention actually matching what FH expects there?),
        not any more changes to FhSession.
      - If this ALSO fails: FhSession's construction is implicated after
        all despite round 3's trace, and/or heading_deg/v_init/thrust
        themselves matter in a way position alone does not -- try
        heading_deg=0.0 (this function's default) first in that case, to
        rule heading out on its own.

    Call this manually:

        from fw.surrogate.validation.model_comparison import diagnose_fh_session_at_known_good_position
        diagnose_fh_session_at_known_good_position()
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
    Round 6: run_comparison((2245.0, -47.0)) failed at its very FIRST grid
    point (distance=30, v=1): pos=(2128.498, 26.037), heading_deg=32.084 --
    ~167 m from seed_point/DEFAULT_FRAMEWORK_SHIP_POS. But
    diagnose_fh_session_at_known_good_position() already proved FhSession
    initializes fine at that seed_point with heading=0.0. Note that
    PySimEnv.polygon_shape (used by find_bank_reference/
    locate_point_at_distance for both the bank projection and the ray-cast
    distance check) is built from `depth_contour`
    ("env_Sche_250cm_no_scale.csv"), the SAME file config.py's own default
    uses for real runs -- not a mismatched/looser `overall_contour` -- so a
    contour-file mismatch has already been ruled out as the cause.

    That leaves two live hypotheses, not yet distinguished:
      (a) FH's exercise only supports ship positions within some bounded
          region around its own baked-in route/area, tighter than the
          depth-contour polygon used for this script's own geometry
          (i.e. bad_pos being ~167 m from the confirmed-good seed_point is
          itself the problem, independent of heading), or
      (b) the heading is the problem (32 deg vs the confirmed-good 0 deg),
          independent of position.
    Both could also be true together. This sweeps each axis independently,
    with try/except around FhSession's constructor (which raises
    RuntimeError on MathModel.Initialize() failure) so a failure at one
    point doesn't stop the rest of the sweep from running and reporting:

      1. Position sweep: n_steps points linearly interpolated from
         good_pos to bad_pos, heading held at 0.0 throughout -- isolates
         whether failure is purely distance/position-based, and roughly
         where along that line it starts failing.
      2. Heading sweep: bad_pos held fixed, heading swept from 0.0 to
         bad_heading_deg across n_steps -- isolates whether heading alone
         is enough to fail Initialize(), independent of position.

    Call this manually:

        from fw.surrogate.validation.model_comparison import diagnose_fh_position_sweep
        diagnose_fh_position_sweep()
    """

    from fw.common.default_positions import DEFAULT_FRAMEWORK_SHIP_POS

    if good_pos is None:
        good_pos = DEFAULT_FRAMEWORK_SHIP_POS

    ship = create_ship(SHIP_MODEL)
    ship_spec = ship.specifications

    def try_point(x: float, y: float, heading_deg: float, label: str) -> None:
        """Try to start an FhSession at one pose and print whether it succeeded (with its first state)."""

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
        x = good_pos[0] + t * (bad_pos[0] - good_pos[0])
        y = good_pos[1] + t * (bad_pos[1] - good_pos[1])
        try_point(x, y, 0.0, f"position_sweep[{i}/{n_steps}]")

    print(
        "=== diagnose_fh_position_sweep: part 2, heading sweep "
        "(pos=bad_pos) ===",
        flush=True,
    )
    for i in range(n_steps + 1):
        t = i / n_steps
        heading_deg = t * bad_heading_deg
        try_point(bad_pos[0], bad_pos[1], heading_deg, f"heading_sweep[{i}/{n_steps}]")


if __name__ == "__main__":
    generate_trajectories((2245.0, -47.0))
    # run_comparison((2245.0, -47.0))
