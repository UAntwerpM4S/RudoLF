"""
UAntwerp surrogate dynamics: one class for every surrogate scenario.

The checkpoint's scenario (fw.surrogate.schema) determines the inputs:

    scenario       UKC (lookup)   rays   wind/current
    open_water     -              -      -
    confined_ukc   yes            yes    -
    full_env       yes            yes    yes

Configuration (ua_surrogate_config, see config.py)::

    {"preset": "tripoli_confined",          # required, see fw.surrogate.presets
     "checkpoint": "x.pth",                 # optional override of the preset's
     "lookup_csv": "x.csv",                 # optional override (None = no lookup)
     "env_smoothing_tau": 10.0,             # optional, seconds (None/absent = off)
     "wind_current": "random"}              # optional, "lookup" (default) or "random"

Relative checkpoint paths resolve against fw/checkpoints, relative lookup CSVs
against fw/dynamics, independent of the working directory. The checkpoint must
match the preset (scenario and ship). The rudder angle is converted to
rudder_action with the rudder scale stored in the checkpoint.

Hooks used by the model-comparison tests:
  * get_env(pos, state=None): the only source of UKC/wind/current for the
    input row; may be replaced by an instance attribute with that signature.
  * _env: the ray caster (scenarios with rays); may be any PySimEnv-like object.
  * ukc / get_ukc(pos): last UKC used.
  * _compute_accelerations(state, rudder_angle, thrust).

Environment from the lookup CSV (get_env, nearest dataset row):
  * UKC for scenarios with a UKC input.
  * Current (full_env): the row's earth-fixed current, as stored.
  * Wind (full_env): the CSV stores the apparent wind of the recording ship
    (FH WindState: relative to its bow, positive from starboard). It is
    converted to true wind with that ship's heading, surge and sway, then back
    to apparent wind for this ship (same conversion as model_comparison_wind).
  Without a lookup or a row: default UKC, zero wind and current. Rows come
  from episodes with random tides and wind fields, so values can jump when the
  nearest row changes. The nearest row is always used; distances beyond
  LOOKUP_WARN_DISTANCE_M are warned about once and then counted
  (lookup_far_count, lookup_max_distance) to expose coverage gaps.

Random wind and current (wind_current = "random", full_env, meant for RL):
reset_episode(rng) draws per episode a true wind (direction uniform, speed
|N(0, RANDOM_WIND_SPEED_SIGMA)| clipped to RANDOM_WIND_SPEED_MAX), constant
earth-fixed and converted to this ship's apparent wind every step, and a
current speed (uniform in [0, RANDOM_CURRENT_SPEED_MAX]) with a flood/ebb sign
(50 %). The current flows along the river line (CurrentLineFollower): a
moving average (time constant env_smoothing_tau, else CURRENT_LINE_TAU) of
the line of the lookup's current at the ship's position over reliable rows
(at least CURRENT_AXIS_MIN_SPEED), with flood/ebb rows treated alike. The
drawn sign orients it at the first reliable row; then the direction moves
gradually towards the line, the slower the weaker the line, so it neither
jumps nor flips. Tuned on the real lookups with
fw.surrogate.check_random_current_lookup. Before the first reliable
row, and without a lookup, the current is zero. UKC still comes from the
lookup. episode_conditions() reports the draw. For presets without
wind/current the setting has no effect.

Smoothing (env_smoothing_tau, meant for RL): the model input then gets an
exponential moving average (time constant tau) of the UKC and, with "lookup",
the earth-fixed true wind and the earth-fixed current ("random": the UKC),
restarted by reset_episode(); the wind is converted back to this ship's
apparent wind. get_env and get_ukc(pos) stay raw; get_ukc(None) returns the UKC of the last
model input. observed_wind_current() reports the wind and current of the
last model input.
"""

import logging
import numpy as np
import pandas as pd

from typing import Tuple, Optional, Dict
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import DynamicsBase
from fw.dynamics.wcu_lookup import WcuLookup, EnvironmentData
# Shared ray implementation, also used by the dataset generator.
from fw.environments.ray_features import RAY_ANGLES_DEG, compute_ray_features, ray_world_angles
from fw.surrogate.presets import CHECKPOINT_DIR, get_preset, resolve_checkpoint, resolve_lookup_csv
from fw.surrogate.schema import (
    SURROGATE_DT,
    TARGET_NAMES,
    apparent_to_true_wind,
    env_body_components,
    scenario_for_raw_width,
)

logger = logging.getLogger(__name__)


def _finite_position(pos) -> bool:
    """
    True if `pos` is an (x, y) pair of finite numbers (None counts as not finite).
    """

    try:
        return bool(np.all(np.isfinite(np.asarray(pos[:2], dtype=float))))
    except (TypeError, ValueError):
        return False

# Keys accepted in ua_surrogate_config.
CONFIG_KEYS = ("preset", "checkpoint", "lookup_csv", "env_smoothing_tau", "wind_current")
WIND_CURRENT_SOURCES = ("lookup", "random")

# wind_current = "random": per-episode distributions (see module docstring).
RANDOM_WIND_SPEED_SIGMA = 8.6     # m/s, true wind
RANDOM_WIND_SPEED_MAX = 25.0      # m/s
RANDOM_CURRENT_SPEED_MAX = 2.1    # m/s, p99.9 of the full_env datasets
CURRENT_AXIS_MIN_SPEED = 0.05     # m/s; slower lookup currents give no reliable direction
CURRENT_LINE_TAU = 10.0           # s; river-line average without env_smoothing_tau
CURRENT_LINE_STRENGTH_EXPONENT = 2.0  # direction follows the line at a rate ~ strength**this

# Lookup rows farther than this [m] are reported. Lookups are 5 m grids of the
# dataset positions, so inside covered water the nearest row is a few metres away.
LOOKUP_WARN_DISTANCE_M = 50.0

_RAY_IDX_BOW = int(np.flatnonzero(np.isclose(np.asarray(RAY_ANGLES_DEG, dtype=float), 0.0))[0])
_RAY_IDX_STBD = int(np.flatnonzero(np.isclose(np.asarray(RAY_ANGLES_DEG, dtype=float), 90.0))[0])


def _bow_and_starboard(heading_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    World-frame unit vectors of bow and starboard (heading convention from the ray module).
    """

    angles = ray_world_angles(float(heading_rad))
    bow, stbd = angles[_RAY_IDX_BOW], angles[_RAY_IDX_STBD]
    return np.array([np.cos(bow), np.sin(bow)]), np.array([np.cos(stbd), np.sin(stbd)])


def apparent_wind_for_state(app_speed: float, app_dir: float,
                            rec_heading: float, rec_u: float, rec_v: float,
                            heading: float, u: float, v: float) -> Tuple[float, float]:
    """
    Re-express an apparent wind measured on one ship for another ship.

    Wind convention (FH WindState, as in the dataset): direction the wind comes
    from, relative to the bow, positive from starboard. In (ahead, starboard)
    components apparent = true + (u, -v), since sway v is positive to port.

    Args:
        app_speed, app_dir: Apparent wind [m/s, rad] on the recording ship.
        rec_heading, rec_u, rec_v: Heading [rad, PySimEnv convention], surge
            and sway [m/s, port +] of the recording ship.
        heading, u, v: The same for the ship the wind is wanted for.

    Returns:
        (speed [m/s], direction [rad] in [-pi, pi)) of the apparent wind on
        the second ship.
    """

    return apparent_wind_from_true_world(true_wind_world(app_speed, app_dir, rec_heading, rec_u, rec_v),
                                         heading, u, v)


def true_wind_world(app_speed: float, app_dir: float, heading: float, u: float, v: float) -> np.ndarray:
    """
    Earth-fixed true wind (world-frame vector pointing where the wind comes
    from) from the apparent wind on a ship (convention: apparent_wind_for_state).
    """

    ahead, stbd_comp = apparent_to_true_wind(app_speed, app_dir, u, v)
    bow, stbd = _bow_and_starboard(heading)
    return ahead * bow + stbd_comp * stbd


def apparent_wind_from_true_world(true_world: np.ndarray, heading: float, u: float,
                                  v: float) -> Tuple[float, float]:
    """
    Apparent wind (speed [m/s], direction [rad] in [-pi, pi)) on a ship from
    the earth-fixed true wind of true_wind_world.
    """

    bow, stbd = _bow_and_starboard(heading)
    own = np.array([true_world @ bow + u, true_world @ stbd - v])
    direction = float(np.arctan2(own[1], own[0]))
    return float(np.hypot(own[0], own[1])), float((direction + np.pi) % (2.0 * np.pi) - np.pi)


class CurrentLineFollower:
    """
    Direction of one episode's "random" current along the river line.

    The line is a moving average (time constant `tau`) of the unit vectors
    of twice the lookup current's direction over reliable rows (speed at
    least CURRENT_AXIS_MIN_SPEED, finite), so flood and ebb rows reinforce
    each other and cross-flow rows only weaken it. The first reliable row,
    oriented by `sign`, gives the first direction. Afterwards the direction
    moves towards the line's orientation closest to it, as a moving average
    with time constant `tau` scaled by the line strength to the power
    CURRENT_LINE_STRENGTH_EXPONENT: a weak line (cross flow, sparse rows)
    barely moves it, so it neither jumps nor flips. None (zero current)
    until the first reliable row.
    """

    def __init__(self, sign: float, tau: Optional[float] = None):
        """
        Args:
            sign: +1: flow as the first reliable row's current, -1: opposite.
            tau: Time constant [s] of the line and direction averages; None for CURRENT_LINE_TAU.
        """

        self.sign = float(sign)
        self.tau = float(tau) if tau is not None else CURRENT_LINE_TAU
        self._line: Optional[np.ndarray] = None
        self._vec: Optional[np.ndarray] = None

    @property
    def line_strength(self) -> float:
        """
        Length of the averaged doubled-angle vector (1 = all rows on one line, 0 = no line).
        """

        return 0.0 if self._line is None else float(np.hypot(self._line[0], self._line[1]))

    def step(self, axis: float, axis_speed: float, dt: float) -> Optional[float]:
        """
        Direction [rad, flowing toward, in [-pi, pi)] after a step of `dt` [s]
        at a lookup row with current direction `axis` and speed `axis_speed`;
        None before the first reliable row.
        """

        alpha = 1.0 - np.exp(-dt / self.tau)
        if np.isfinite(axis) and np.isfinite(axis_speed) and axis_speed >= CURRENT_AXIS_MIN_SPEED:
            doubled = np.array([np.cos(2.0 * axis), np.sin(2.0 * axis)])
            if self._line is None:
                self._line = doubled
                first = axis if self.sign > 0 else axis + np.pi
                self._vec = np.array([np.cos(first), np.sin(first)])
                return float((first + np.pi) % (2.0 * np.pi) - np.pi)
            self._line = self._line + alpha * (doubled - self._line)
        if self._line is None:
            return None
        current = np.arctan2(self._vec[1], self._vec[0])
        line = 0.5 * np.arctan2(self._line[1], self._line[0])
        if np.cos(line - current) < 0.0:
            line += np.pi
        weight = alpha * self.line_strength ** CURRENT_LINE_STRENGTH_EXPONENT
        self._vec = self._vec + weight * (np.array([np.cos(line), np.sin(line)]) - self._vec)
        return float(np.arctan2(self._vec[1], self._vec[0]))


def make_scenario_lookup(csv, scenario, preset_name: str = "?") -> WcuLookup:
    """
    WcuLookup on `csv`, reading only the columns `scenario` needs.

    Raises:
        ValueError: If the CSV width does not match the scenario's dataset layout.
    """

    # Lookup CSVs carry no identity, so the width is the only layout check
    # (otherwise e.g. a confined_ukc CSV's rudder/ray columns would be read
    # as wind and current by a full_env preset).
    sc = scenario
    expected = sc.raw_input_dim + len(TARGET_NAMES) + 1     # raw row + targets + episode_id
    n_cols = len(pd.read_csv(csv, header=None, nrows=1).columns)
    if n_cols != expected:
        try:
            found = scenario_for_raw_width(n_cols - len(TARGET_NAMES) - 1).name
        except ValueError:
            found = "no known scenario"
        raise ValueError(
            f"Lookup CSV {csv} has {n_cols} columns ({found}); a '{sc.name}' dataset/lookup CSV "
            f"has {expected} (raw row {sc.raw_input_dim} + {len(TARGET_NAMES)} targets + episode_id). "
            f"Wrong lookup_csv for preset '{preset_name}'?")
    cols = dict(x_col=sc.col("x"), y_col=sc.col("y"), ukc_col=sc.col("ukc"),
                wind_speed_col=None, wind_dir_col=None, current_speed_col=None, current_dir_col=None)
    if sc.includes_wind_current:
        cols.update(wind_speed_col=sc.col("wind_speed"), wind_dir_col=sc.col("wind_direction"),
                    current_speed_col=sc.col("current_speed"), current_dir_col=sc.col("current_direction"),
                    heading_col=sc.col("heading"), surge_col=sc.col("surge"), sway_col=sc.col("sway"))
    return WcuLookup(str(csv), **cols)


def parse_env_smoothing_tau(value) -> Optional[float]:
    """
    Validate ua_surrogate_config["env_smoothing_tau"]: None (off) or a positive
    finite number of seconds.

    Raises:
        ValueError: For anything else (including bools).
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"ua_surrogate_config['env_smoothing_tau'] must be None or a positive number of "
                         f"seconds, got {value!r}.")
    return float(value)


class UAntwerpSurrogate(DynamicsBase):
    """
    3-DOF dynamics from a trained neural-network surrogate, predicting body-frame
    accelerations from state, actuator commands and, depending on the scenario,
    UKC, ray features, wind and current. The surrogate predicts velocity changes
    over SURROGATE_DT, the only time step step() accepts.

    Attributes:
        preset: fw.surrogate.presets.SurrogatePreset in use.
        scenario: schema.ScenarioSchema of the checkpoint.
        surrogate: fw.surrogate.inference_surrogate.SurrogatePredictor.
        rudder_max_deg: Rudder angle [deg] that rudder_action = 1 stands for.
        wcu_lookup: WcuLookup for the UKC, or None.
        default_ukc: UKC [m] used without (a hit in) the lookup.
        ukc: Last UKC used [m].
        lookup_far_count: Lookups whose nearest row was farther than
            LOOKUP_WARN_DISTANCE_M (the row is still used).
        lookup_max_distance: Largest lookup distance seen so far [m].
        _env: Ray caster (scenarios with rays only); created on first use if
            none was given.
    """

    def __init__(self,
                 ship_spec: ShipSpecifications,
                 ua_surrogate_config: Optional[Dict] = None,
                 ship_model: Optional[ShipModel] = None,
                 ray_caster=None):
        """
        Args:
            ship_spec:           Vessel specifications.
            ua_surrogate_config: {"preset": ..., optional "checkpoint", "lookup_csv",
                                 "env_smoothing_tau", "wind_current"}.
            ship_model:          Ship the caller simulates; if given, it must be
                                 the preset's ship.
            ray_caster:          Object with ray_cast_distances_batch / ray_cast_distance
                                 (normally a PySimEnv). None: a PySimEnv is created
                                 on first use (scenarios with rays only).

        Raises:
            ValueError: On a missing/unknown preset, unknown config keys, a
                checkpoint/ship that does not match the preset, a lookup CSV
                with the wrong column count, or an invalid env_smoothing_tau or wind_current.
            FileNotFoundError: If the checkpoint does not exist.
        """

        config = dict(ua_surrogate_config or {})
        unknown = sorted(set(config) - set(CONFIG_KEYS))
        if unknown:
            raise ValueError(f"ua_surrogate_config: unknown key(s) {unknown}; accepted: {list(CONFIG_KEYS)} "
                             f"(the former 'wcu_enabled'/'file_name' are replaced by 'preset'/'checkpoint').")
        if "preset" not in config:
            raise ValueError("ua_surrogate_config needs a 'preset', e.g. {'preset': 'tripoli_confined'} "
                             "(see fw.surrogate.presets.PRESETS).")

        self.preset = get_preset(config["preset"])
        self.env_smoothing_tau = parse_env_smoothing_tau(config.get("env_smoothing_tau"))
        self.wind_current_source = config.get("wind_current", "lookup")
        if self.wind_current_source not in WIND_CURRENT_SOURCES:
            raise ValueError(f"ua_surrogate_config['wind_current'] must be one of {WIND_CURRENT_SOURCES}, "
                             f"got {self.wind_current_source!r}.")
        # Per episode (reset_episode): moving average and last model input,
        # both as [ukc, true wind x, y, current x, y] (earth-fixed), and the
        # "random" draw: true wind world vector, current speed and sign, and
        # the current's direction follower (CurrentLineFollower).
        self._env_ema: Optional[np.ndarray] = None
        self._env_world: Optional[np.ndarray] = None
        self._random_wind_world: Optional[np.ndarray] = None
        self._random_current_speed = 0.0
        self._random_current_sign = 1.0
        self._random_current: Optional[CurrentLineFollower] = None
        if ship_model is not None and ship_model != self.preset.ship_model:
            raise ValueError(f"UAntwerp surrogate preset '{self.preset.name}' is for "
                             f"{self.preset.ship_model.name}, but ship_model={ship_model.name}.")

        # Lazy import: torch is only needed once a surrogate is used.
        from fw.surrogate.inference_surrogate import SurrogatePredictor

        checkpoint = resolve_checkpoint(config.get("checkpoint", self.preset.checkpoint))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint of UAntwerp surrogate preset '{self.preset.name}' not found: "
                                    f"{checkpoint} (relative checkpoint paths are resolved against "
                                    f"{CHECKPOINT_DIR}).")
        self.surrogate = SurrogatePredictor(str(checkpoint))
        if self.surrogate.scenario != self.preset.scenario:
            raise ValueError(f"Checkpoint {checkpoint} is a '{self.surrogate.scenario.name}' model, but preset "
                             f"'{self.preset.name}' expects '{self.preset.scenario.name}'.")
        if self.surrogate.ship_model != self.preset.ship_model.name:
            raise ValueError(f"Checkpoint {checkpoint} was trained on {self.surrogate.ship_model} data, but "
                             f"preset '{self.preset.name}' is for {self.preset.ship_model.name}.")

        self.ship_spec: ShipSpecifications = ship_spec
        self.scenario = self.surrogate.scenario
        self.rudder_max_deg = self.surrogate.rudder_max_deg
        if float(ship_spec.max_rudder_angle) > np.radians(self.rudder_max_deg) + 1e-9:
            logger.warning(
                "Ship max rudder angle %.1f deg exceeds the %.1f deg the surrogate (%s) was trained "
                "with: larger rudder angles are extrapolation.",
                np.degrees(ship_spec.max_rudder_angle), self.rudder_max_deg, checkpoint,
            )

        # UKC (and for full_env wind/current): nearest-neighbor lookup in a CSV
        # with the scenario's raw column layout.
        self.default_ukc = float(self.preset.default_ukc)
        self.ukc = self.default_ukc
        self.lookup_far_count = 0
        self.lookup_max_distance = 0.0
        self.wcu_lookup = None
        if self.scenario.includes_ukc:
            csv = resolve_lookup_csv(config.get("lookup_csv", self.preset.lookup_csv))
            if csv is not None and csv.exists():
                self.wcu_lookup = self._make_lookup(csv)
            else:
                # Deliberately a warning (lookup CSVs are large and often left out).
                if not self.scenario.includes_wind_current:
                    consequence = ""
                elif self._uses_random_wind_current:
                    consequence = " and ZERO current (the random wind still applies)"
                else:
                    consequence = " and ZERO wind/current (the run has no wind or current at all)"
                logger.warning(
                    "UAntwerp surrogate '%s': %s; falling back to default UKC value of %s%s",
                    self.preset.name,
                    "no lookup CSV configured" if csv is None else f"ship data file {csv} not found",
                    self.default_ukc, consequence)

        if self.scenario.includes_rays:
            self._env = ray_caster

    @property
    def models_own_wind_current(self) -> bool:
        """
        True only for full_env (wind/current from the lookup or, with
        wind_current = "random", drawn per episode).
        """

        return bool(self.scenario.includes_wind_current)

    def _make_lookup(self, csv) -> WcuLookup:
        """
        WcuLookup on `csv` for this preset's scenario (make_scenario_lookup).
        """

        return make_scenario_lookup(csv, self.scenario, self.preset.name)

    def _ray_caster(self):
        """
        The ray caster, creating the default PySimEnv on first use.
        """

        if self._env is None:
            # Imported locally to avoid a circular import (py_sim_env imports this module).
            from fw.environments.py_sim_env import PySimEnv
            self._env = PySimEnv(render_mode=None, time_step=SURROGATE_DT, wind=False, current=False)
        return self._env

    def get_ukc(self, pos: Optional[np.ndarray]):
        """
        Under-keel clearance at `pos`: the nearest lookup row (no side effects),
        or the UKC of the last model input without a position or a lookup
        (default UKC without a lookup; also for a non-finite position).

        Args:
            pos: World-frame position [x, y] in metres, or None.

        Returns:
            Under-keel clearance in metres.
        """

        if pos is None or self.wcu_lookup is None:
            return self.ukc
        if not _finite_position(pos):
            # Do not raise here (the KD-tree refuses NaN); _compute_accelerations
            # raises a clear FloatingPointError instead.
            return self.default_ukc
        ukc = self.wcu_lookup.get_ukc(float(pos[0]), float(pos[1]))
        return self.default_ukc if ukc is None else float(ukc)

    def get_env(self, pos: Optional[np.ndarray], state: Optional[VesselState] = None) -> EnvironmentData:
        """
        Environmental conditions at `pos` from the nearest lookup row (see module
        docstring); default UKC and zero wind/current without a lookup or row.

        Args:
            pos: World-frame position [x, y] in metres.
            state: This ship's state, needed to express the full_env wind as its
                apparent wind. None: wind fields are None (only UKC/current
                wanted, e.g. model_comparison_wind.UkcSource).
        """

        calm = 0.0 if state is not None or not self.scenario.includes_wind_current else None
        # Non-finite position: no-row defaults instead of a cryptic KD-tree error;
        # _compute_accelerations then raises a clear FloatingPointError.
        sample = (None if self.wcu_lookup is None or not _finite_position(pos)
                  else self.wcu_lookup.get_sample(pos[0], pos[1]))
        if sample is None or sample.env.ukc is None:
            self.ukc = self.default_ukc
            return EnvironmentData(ukc=self.ukc, wind_speed=calm, wind_direction=calm,
                                   current_speed=0.0, current_direction=0.0)

        self._note_lookup_distance(pos, getattr(sample, "distance", None))
        self.ukc = sample.env.ukc
        if not self.scenario.includes_wind_current:
            return EnvironmentData(ukc=self.ukc, wind_speed=0.0, wind_direction=0.0,
                                   current_speed=0.0, current_direction=0.0)

        wind_speed = wind_dir = None
        if state is not None:
            wind_speed, wind_dir = apparent_wind_for_state(
                sample.env.wind_speed, sample.env.wind_direction,
                sample.heading, sample.surge, sample.sway,
                float(state.heading), float(state.u), float(state.v),
            )
        return EnvironmentData(ukc=self.ukc, wind_speed=wind_speed, wind_direction=wind_dir,
                               current_speed=sample.env.current_speed,
                               current_direction=sample.env.current_direction)

    def _model_env(self, pos: np.ndarray, state: VesselState) -> EnvironmentData:
        """
        Environment for the model input at `pos`: get_env, with this episode's
        wind and current instead with "random", smoothed with env_smoothing_tau
        if set (one update per call, i.e. per SURROGATE_DT step; with "random"
        only the UKC). Remembers it for observed_wind_current; self.ukc becomes
        the UKC used.
        """

        env = self.get_env(pos, state)
        use_random = self._uses_random_wind_current
        if use_random:
            env = self._random_wind_current(env, state)
        world = self._env_to_world(env, state)
        if self.env_smoothing_tau is not None:
            if self._env_ema is None:
                self._env_ema = world
            else:
                alpha = 1.0 - np.exp(-SURROGATE_DT / self.env_smoothing_tau)
                self._env_ema = self._env_ema + alpha * (world - self._env_ema)
            # "random": wind/current come from the episode draw (current line
            # averaged by CurrentLineFollower), so only the UKC is averaged here.
            world = np.concatenate([self._env_ema[:1], world[1:]]) if use_random else self._env_ema
            env = self._world_to_env(world, state)
            self.ukc = env.ukc
        self._env_world = world
        return env

    @property
    def _uses_random_wind_current(self) -> bool:
        """
        True for "random" wind/current on a scenario with wind and current.
        """

        return self.wind_current_source == "random" and self.scenario.includes_wind_current

    def _random_wind_current(self, env: EnvironmentData, state: VesselState) -> EnvironmentData:
        """
        `env` with this episode's random wind (apparent for `state`) and current
        (drawn speed and sign along the lookup's current line, see module docstring).

        Raises:
            RuntimeError: If no wind/current was drawn (reset_episode without rng).
        """

        if self._random_wind_world is None:
            raise RuntimeError("UAntwerp surrogate with wind_current='random': no wind/current drawn for this "
                               "episode; call reset_episode(rng) (ShipSimulator.reset(state, rng)).")
        ws, wd = apparent_wind_from_true_world(self._random_wind_world, float(state.heading), float(state.u),
                                               float(state.v))
        if self.wcu_lookup is None:
            cs, cd = 0.0, 0.0
        else:
            cd = self._random_current.step(float(env.current_direction), float(env.current_speed), SURROGATE_DT)
            # Zero current until the first reliable row (slack water at the start).
            cs, cd = (0.0, 0.0) if cd is None else (self._random_current_speed, cd)
        return EnvironmentData(ukc=env.ukc, wind_speed=ws, wind_direction=wd, current_speed=cs,
                               current_direction=cd)

    def episode_conditions(self) -> Dict[str, object]:
        """
        This episode's drawn wind and current with "random" (wind as true wind,
        direction it comes from, 0 = East, CCW); {"wind_current": "lookup"} for
        full_env with the lookup; empty without wind/current.
        """

        if not self.scenario.includes_wind_current:
            return {}
        if not self._uses_random_wind_current:
            return {"wind_current": "lookup"}
        if self._random_wind_world is None:
            return {"wind_current": "random (not drawn)"}
        wx, wy = self._random_wind_world
        return {"wind_speed": round(float(np.hypot(wx, wy)), 2),
                "wind_from_deg": round(float(np.degrees(np.arctan2(wy, wx)) % 360.0), 1),
                "current_speed": round(self._random_current_speed, 3),
                "current_sign": int(self._random_current_sign)}

    def _env_to_world(self, env: EnvironmentData, state: VesselState) -> np.ndarray:
        """
        [ukc, true wind x, y, current x, y]: `env` with this ship's apparent wind
        made earth-fixed (true_wind_world) and the current as a world vector.
        """

        out = np.zeros(5, dtype=np.float64)
        out[0] = float(env.ukc)
        if self.scenario.includes_wind_current:
            out[1:3] = true_wind_world(float(env.wind_speed), float(env.wind_direction),
                                       float(state.heading), float(state.u), float(state.v))
            cs, cd = float(env.current_speed), float(env.current_direction)
            out[3:5] = (cs * np.cos(cd), cs * np.sin(cd))
        return out

    def _world_to_env(self, world: np.ndarray, state: VesselState) -> EnvironmentData:
        """
        Inverse of _env_to_world for this ship's `state`.
        """

        if not self.scenario.includes_wind_current:
            return EnvironmentData(ukc=float(world[0]), wind_speed=0.0, wind_direction=0.0,
                                   current_speed=0.0, current_direction=0.0)
        ws, wd = apparent_wind_from_true_world(world[1:3], float(state.heading), float(state.u), float(state.v))
        return EnvironmentData(ukc=float(world[0]), wind_speed=ws, wind_direction=wd,
                               current_speed=float(np.hypot(world[3], world[4])),
                               current_direction=float(np.arctan2(world[4], world[3])))

    def observed_wind_current(self, state: VesselState) -> np.ndarray:
        """
        Wind and current of the last model input (smoothed if configured), the
        wind as apparent wind for `state` (see DynamicsBase). Before the first
        step of an episode, the environment at `state` starts the episode's
        moving average. Zeros for scenarios without wind and current.
        """

        if not self.scenario.includes_wind_current:
            return np.zeros(4, dtype=np.float64)
        if self._env_world is None:
            self._model_env(np.array([state.x, state.y]), state)
        env = self._world_to_env(self._env_world, state)
        return np.array(env_body_components(env.wind_speed, env.wind_direction, env.current_speed,
                                            env.current_direction, float(state.heading)), dtype=np.float64)

    def reset_episode(self, rng: Optional[np.random.Generator] = None) -> None:
        """
        Restart the environment smoothing and forget the last model input
        (get_ukc(None) returns the default UKC until the next one). With
        "random" wind/current, draw this episode's wind and current from `rng`
        (draw order: wind direction, wind speed, current speed, current sign);
        without `rng` none is drawn and the next model step raises.
        """

        self._env_ema = None
        self._env_world = None
        self.ukc = self.default_ukc
        self._random_wind_world = None
        self._random_current = None
        if self._uses_random_wind_current and rng is not None:
            wind_dir = float(rng.uniform(0.0, 2.0 * np.pi))
            wind_speed = float(np.clip(np.abs(rng.normal(0.0, RANDOM_WIND_SPEED_SIGMA)), 0.0, RANDOM_WIND_SPEED_MAX))
            self._random_wind_world = wind_speed * np.array([np.cos(wind_dir), np.sin(wind_dir)])
            self._random_current_speed = float(rng.uniform(0.0, RANDOM_CURRENT_SPEED_MAX))
            self._random_current_sign = 1.0 if rng.random() < 0.5 else -1.0
            self._random_current = CurrentLineFollower(self._random_current_sign, self.env_smoothing_tau)

    def _note_lookup_distance(self, pos, distance: float) -> None:
        """
        Track the lookup distance; warn the first time it exceeds LOOKUP_WARN_DISTANCE_M.
        """

        if distance is None:          # lookup without distance information
            return
        distance = float(distance)
        if distance > self.lookup_max_distance:
            self.lookup_max_distance = distance
        if distance > LOOKUP_WARN_DISTANCE_M:
            self.lookup_far_count += 1
            if self.lookup_far_count == 1:
                logger.warning(
                    "UAntwerp surrogate '%s': the nearest lookup row to (%.1f, %.1f) is %.1f m away "
                    "(> %.0f m); its UKC%s %s used anyway. The lookup does not cover this area well. "
                    "Further cases are only counted (lookup_far_count).",
                    self.preset.name, float(pos[0]), float(pos[1]), distance, LOOKUP_WARN_DISTANCE_M,
                    ", wind and current" if self.scenario.includes_wind_current else "",
                    "are" if self.scenario.includes_wind_current else "is")

    def step(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
        dt: float,
        enable_rudder_angle_scaling: Optional[bool] = False,
    ) -> Tuple[float, float, float]:
        """
        Advance the dynamics by one surrogate time step.

        Args:
            state:                       Current vessel state.
            rudder_angle:                Rudder command [rad]; positive = the ship turns to PORT
                                         (framework convention, see DynamicsBase).
            thrust:                      Normalized thrust command in [-1, 1].
            dt:                          Integration time step [s]; must equal SURROGATE_DT.
            enable_rudder_angle_scaling: Interface compatibility only; ignored.

        Returns:
            Updated body-frame velocities (u, v, r).

        Raises:
            ValueError: If dt differs from SURROGATE_DT.
        """

        if not np.isclose(dt, SURROGATE_DT):
            raise ValueError(
                f"UAntwerpSurrogate requires dt == {SURROGATE_DT} s (surrogate training step), got {dt}. "
                f"Ensure ShipSimulator.dt is an exact integer multiple of {SURROGATE_DT} s "
                f"so no fractional remainder sub-step is produced."
            )
        return super().step(state, rudder_angle, thrust, dt, enable_rudder_angle_scaling)

    def _build_input_vector(self, state: VesselState, rudder_angle: float, thrust: float) -> np.ndarray:
        """
        Assemble the raw input row in the scenario's column order, exactly as
        the dataset generator stores it (selection, encoding and normalization
        happen in SurrogatePredictor). UKC/wind/current come from _model_env, rays
        from compute_ray_features, each only if the scenario uses them.

        Returns:
            float32 array of shape (scenario.raw_input_dim,).
        """

        values = {
            # x, y are never model inputs; 0.0 placeholder if unknown (only
            # allowed for scenarios that do not need the position).
            "x": state.x if state.x is not None else 0.0,
            "y": state.y if state.y is not None else 0.0,
            "heading": state.heading,
            "surge": state.u,
            "sway": state.v,
            "yaw_rate": state.r,
            "rudder_action": np.degrees(rudder_angle) / self.rudder_max_deg,
            "thrust_action": thrust,
        }

        if self.scenario.includes_ukc or self.scenario.includes_wind_current:
            env = self._model_env(np.array([state.x, state.y]), state)
            values.update(ukc=env.ukc,
                          wind_speed=env.wind_speed, wind_direction=env.wind_direction,
                          current_speed=env.current_speed, current_direction=env.current_direction)

        if self.scenario.includes_rays:
            if _finite_position((state.x, state.y)) and np.isfinite(float(state.heading)):
                values["rays"] = compute_ray_features(
                    self._ray_caster(),
                    np.array([state.x, state.y]),
                    float(state.heading),
                )
            else:
                # NaN pose: skip ray casting; _compute_accelerations raises.
                values["rays"] = np.full(len(RAY_ANGLES_DEG), np.nan, dtype=np.float32)

        return self.scenario.build_raw_row(values)

    def _compute_accelerations(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
    ) -> Tuple[float, float, float]:
        """
        Body-frame accelerations: the surrogate's velocity changes over
        SURROGATE_DT divided by SURROGATE_DT.

        Args:
            state:        Current vessel state.
            rudder_angle: Rudder command [rad]; positive turns the ship to port.
            thrust:       Normalized thrust command in [-1, 1].

        Returns:
            Predicted accelerations (du, dv, dr).

        Raises:
            ValueError: If heading is None, or x/y is None for a scenario that
                needs the position.
            FloatingPointError: If any input is non-finite.
        """

        required = [("heading", state.heading)]
        if self.scenario.includes_ukc or self.scenario.includes_wind_current or self.scenario.includes_rays:
            required += [("x", state.x), ("y", state.y)]
        missing = [name for name, val in required if val is None]
        if missing:
            raise ValueError(
                f"UAntwerpSurrogate ({self.scenario.name}) requires state fields {missing} to evaluate the surrogate."
            )

        x_raw = self._build_input_vector(state, rudder_angle, thrust)
        if not np.all(np.isfinite(x_raw)):
            # Fail fast: a NaN input would otherwise surface later as a NaN observation.
            bad = [name for name, val in zip(self.scenario.raw_columns, x_raw) if not np.isfinite(val)]
            raise FloatingPointError(
                f"UAntwerpSurrogate ({self.scenario.name}): non-finite input(s) {bad} "
                f"at x={state.x}, y={state.y}."
            )
        delta_u, delta_v, delta_r = self.surrogate.predict_single(x_raw)
        return float(delta_u / SURROGATE_DT), float(delta_v / SURROGATE_DT), float(delta_r / SURROGATE_DT)
