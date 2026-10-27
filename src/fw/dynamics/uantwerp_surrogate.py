"""
UAntwerp surrogate dynamics -- ONE class for every surrogate scenario.

Replaces the previous per-scenario copies (UAntwerpSurrogateTripoliClean /
Confined / Full and the old generic UAntwerpSurrogate). What a given instance
does is fully determined by its checkpoint's scenario (fw.surrogate.schema):

    scenario       UKC (lookup)   rays   wind/current   (previous class)
    open_water     -              -      -              ...TripoliClean
    confined_ukc   yes            yes    -              ...TripoliConfined
    full_env       yes            yes    yes            ...TripoliFull

Configuration (`ua_surrogate_config`, see config.py)::

    {"preset": "tripoli_confined",          # required, see fw.surrogate.presets
     "checkpoint": "x.pth",                 # optional override of the preset's
     "lookup_csv": "x.csv"}                 # optional override (None = no lookup)

Relative checkpoint paths are resolved against fw/checkpoints
(presets.CHECKPOINT_DIR), relative lookup CSVs against fw/dynamics
(presets.LOOKUP_CSV_DIR), independent of the working directory.

The checkpoint must match the preset (scenario and ship); the rudder angle is
converted to the model's rudder_action with the rudder scale stored in the
checkpoint (the generator's RUDDER_MAX_DEG for that data).

Hooks used by the model-comparison tests (kept on purpose):
  * `get_env(pos, state=None)`: the ONLY source of UKC/wind/current for the
    input row; may be replaced by an instance attribute (a function of
    `(pos, state=None)`; its values are used as given).
  * `_env`: the ray caster (only present for scenarios with rays); may be
    replaced by another PySimEnv-like object.
  * `ukc` / `get_ukc(pos)`: last UKC used.
  * `_compute_accelerations(state, rudder_angle, thrust)`.

Environment from the lookup CSV (`get_env`, nearest dataset row):
  * UKC: for scenarios with a UKC input.
  * Current (full_env): the row's earth-fixed current, as stored.
  * Wind (full_env): the CSV stores the APPARENT wind of the ship that
    recorded the row (FH WindState: relative to its bow, positive = from
    starboard). It is turned into the true wind with that ship's heading,
    surge and sway, and back into the apparent wind for THIS ship's heading,
    surge and sway -- the same conversion as the wind test
    (model_comparison_wind._true_wind_world / _apparent_wind_for).
  Without a lookup (or no row found): default UKC, zero wind and current.
  The rows come from episodes with random tide times and wind fields, so the
  looked-up wind/current can jump when the nearest row changes.
  The nearest row is used however far away it is (unchanged behaviour), but
  a lookup farther than LOOKUP_WARN_DISTANCE_M is reported: a warning the
  first time, and counted in `lookup_far_count` (largest distance seen in
  `lookup_max_distance`), so gaps in the lookup's coverage become visible.
"""

import logging
import numpy as np

from typing import Tuple, Optional, Dict
from fw.ships.vessel_state import VesselState
from fw.ships.ship_specs import ShipSpecifications
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import DynamicsBase
from fw.dynamics.wcu_lookup import WcuLookup, EnvironmentData
# The single shared ray implementation, also used by the dataset generator.
from fw.environments.ray_features import RAY_ANGLES_DEG, compute_ray_features, ray_world_angles
from fw.surrogate.presets import CHECKPOINT_DIR, get_preset, resolve_checkpoint, resolve_lookup_csv
from fw.surrogate.schema import SURROGATE_DT

logger = logging.getLogger(__name__)

# Keys accepted in ua_surrogate_config.
CONFIG_KEYS = ("preset", "checkpoint", "lookup_csv")

# A lookup row farther away than this [m] is reported (see the module
# docstring). The lookups are 5 m grids of the positions the dataset visited,
# so inside the covered water the nearest row is a few metres away.
LOOKUP_WARN_DISTANCE_M = 50.0

_RAY_IDX_BOW = int(np.flatnonzero(np.isclose(np.asarray(RAY_ANGLES_DEG, dtype=float), 0.0))[0])
_RAY_IDX_STBD = int(np.flatnonzero(np.isclose(np.asarray(RAY_ANGLES_DEG, dtype=float), 90.0))[0])


def _bow_and_starboard(heading_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """World-frame unit vectors of the bow and starboard directions, from the
    shared ray module (so no heading convention is assumed here)."""

    angles = ray_world_angles(float(heading_rad))
    bow, stbd = angles[_RAY_IDX_BOW], angles[_RAY_IDX_STBD]
    return np.array([np.cos(bow), np.sin(bow)]), np.array([np.cos(stbd), np.sin(stbd)])


def apparent_wind_for_state(app_speed: float, app_dir: float,
                            rec_heading: float, rec_u: float, rec_v: float,
                            heading: float, u: float, v: float) -> Tuple[float, float]:
    """
    Re-express an apparent wind measured on one ship for another ship.

    Wind convention (FH WindState, as stored in the dataset): direction the
    wind comes FROM, relative to the bow, positive = from starboard; as
    (ahead, starboard) components apparent = true + (u, -v), because sway v
    is positive to PORT (fw.ships.vessel_state, the dataset convention).

    Args:
        app_speed, app_dir: Apparent wind [m/s, rad] on the recording ship.
        rec_heading, rec_u, rec_v: Heading [rad, PySimEnv convention], surge
            and sway [m/s, port +] of the recording ship.
        heading, u, v: The same for the ship the wind is wanted for.

    Returns:
        (speed [m/s], direction [rad] in [-pi, pi)) of the apparent wind on
        the second ship.
    """

    rel = app_speed * np.array([np.cos(app_dir), np.sin(app_dir)]) - np.array([rec_u, -rec_v])
    bow, stbd = _bow_and_starboard(rec_heading)
    true_world = rel[0] * bow + rel[1] * stbd
    bow, stbd = _bow_and_starboard(heading)
    own = np.array([true_world @ bow + u, true_world @ stbd - v])
    direction = float(np.arctan2(own[1], own[0]))
    return float(np.hypot(own[0], own[1])), float((direction + np.pi) % (2.0 * np.pi) - np.pi)


class UAntwerpSurrogate(DynamicsBase):
    """
    Surrogate-model-based 3-DOF vessel dynamics.

    Uses a trained neural-network surrogate (fw.surrogate.inference_surrogate)
    to predict body-frame surge, sway and yaw accelerations from the vessel
    state, actuator commands and -- depending on the checkpoint's scenario --
    under-keel clearance, ray-cast proximity features, wind and current.

    The surrogate predicts velocity changes over `SURROGATE_DT`, which is
    therefore the only time step `step()` accepts.

    Attributes:
        preset: fw.surrogate.presets.SurrogatePreset in use.
        scenario: schema.ScenarioSchema of the checkpoint.
        surrogate: fw.surrogate.inference_surrogate.SurrogatePredictor.
        rudder_max_deg: Rudder angle [deg] that rudder_action = 1 stands for.
        wcu_lookup: WcuLookup for the UKC, or None.
        default_ukc: UKC [m] used without (a hit in) the lookup.
        ukc: Last UKC used [m].
        lookup_far_count: Number of lookups whose nearest row was farther
            than LOOKUP_WARN_DISTANCE_M (the row is still used).
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
        Initialize the surrogate dynamics model.

        Args:
            ship_spec:           Vessel specifications.
            ua_surrogate_config: {"preset": ..., optional "checkpoint", "lookup_csv"}.
            ship_model:          Ship the caller simulates; if given, it must be
                                 the preset's ship.
            ray_caster:          Object with `ray_cast_distances_batch` /
                                 `ray_cast_distance` (normally a PySimEnv) to
                                 cast the rays against. None = a PySimEnv is
                                 created on first use (scenarios with rays only).

        Raises:
            ValueError: On a missing/unknown preset, unknown config keys, or a
                checkpoint / ship that does not match the preset.
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
        if ship_model is not None and ship_model != self.preset.ship_model:
            raise ValueError(f"UAntwerp surrogate preset '{self.preset.name}' is for "
                             f"{self.preset.ship_model.name}, but ship_model={ship_model.name}.")

        # Imported here: torch is only needed once a surrogate is actually used.
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

        # UKC (and, for full_env, wind/current): nearest-neighbor lookup in
        # the dataset CSV, only for scenarios with a UKC input. The CSV has
        # the scenario's raw column layout, so the positions come from the schema.
        self.default_ukc = float(self.preset.default_ukc)
        self.ukc = self.default_ukc
        self.lookup_far_count = 0
        self.lookup_max_distance = 0.0
        self.wcu_lookup = None
        if self.scenario.includes_ukc:
            csv = resolve_lookup_csv(config.get("lookup_csv", self.preset.lookup_csv))
            if csv is not None and csv.exists():
                self.wcu_lookup = self._make_lookup(csv)
            elif csv is not None:
                logger.warning("Ship data file %s not found, falling back to default UKC value of %s%s",
                               csv, self.default_ukc,
                               " and zero wind/current" if self.scenario.includes_wind_current else "")

        if self.scenario.includes_rays:
            self._env = ray_caster

    def _make_lookup(self, csv) -> WcuLookup:
        """WcuLookup on `csv`, reading only the columns this scenario needs."""

        sc = self.scenario
        cols = dict(x_col=sc.col("x"), y_col=sc.col("y"), ukc_col=sc.col("ukc"),
                    wind_speed_col=None, wind_dir_col=None, current_speed_col=None, current_dir_col=None)
        if sc.includes_wind_current:
            cols.update(wind_speed_col=sc.col("wind_speed"), wind_dir_col=sc.col("wind_direction"),
                        current_speed_col=sc.col("current_speed"), current_dir_col=sc.col("current_direction"),
                        heading_col=sc.col("heading"), surge_col=sc.col("surge"), sway_col=sc.col("sway"))
        return WcuLookup(str(csv), **cols)

    def _ray_caster(self):
        """The ray caster, creating the default PySimEnv on first use."""

        if self._env is None:
            # Imported locally to avoid a circular import (py_sim_env imports this module).
            from fw.environments.py_sim_env import PySimEnv
            self._env = PySimEnv(render_mode=None, time_step=SURROGATE_DT, wind=False, current=False)
        return self._env

    def get_ukc(self, pos: Optional[np.ndarray]):
        """
        Return the last under-keel clearance used (updated by `get_env`).

        The position argument is accepted for interface compatibility but is
        ignored.

        Returns:
            Under-keel clearance in metres.
        """

        return self.ukc

    def get_env(self, pos: Optional[np.ndarray], state: Optional[VesselState] = None) -> EnvironmentData:
        """
        Environmental conditions at `pos` from the nearest lookup row (see
        the module docstring); default UKC and zero wind/current without a
        lookup or a row.

        Args:
            pos: World-frame position `[x, y]` in metres.
            state: State of THIS ship (heading, surge, sway), needed to give
                the full_env wind as this ship's apparent wind. None: the
                wind fields are None (only UKC/current wanted, e.g. by
                model_comparison_wind.UkcSource).

        Returns:
            Environmental conditions at the requested position.
        """

        calm = 0.0 if state is not None or not self.scenario.includes_wind_current else None
        sample = None if self.wcu_lookup is None else self.wcu_lookup.get_sample(pos[0], pos[1])
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

    def _note_lookup_distance(self, pos, distance: float) -> None:
        """Track the lookup distance; warn the first time it exceeds LOOKUP_WARN_DISTANCE_M."""

        if distance is None:          # a lookup without distance information: nothing to report
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
        Advance the vessel dynamics by one surrogate time step.

        Args:
            state:                       Current vessel state.
            rudder_angle:                Rudder command [rad]; positive = the ship turns to PORT
                                         (framework convention, see DynamicsBase).
            thrust:                      Normalized thrust command in `[-1, 1]`.
            dt:                          Integration time step [s]; must equal SURROGATE_DT.
            enable_rudder_angle_scaling: Passed to the base implementation for interface
                                         compatibility. Ignored by this class.

        Returns:
            Tuple `(u, v, r)` containing the updated body-frame velocities.

        Raises:
            ValueError: If `dt` differs from `SURROGATE_DT` (an explicit check, not
                an assert, so it also holds under `python -O`).
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
        Assemble the RAW input row in the checkpoint scenario's column order
        (schema.<SCENARIO>.raw_columns), exactly as the dataset generator
        stores it. Selection, encoding and normalization happen inside
        SurrogatePredictor; nothing is pre-processed here.

        UKC/wind/current come from `get_env` (only called for scenarios that
        use them), rays from `compute_ray_features` (only for scenarios with
        rays). x, y are part of every raw row but never a model input.

        Returns:
            float32 array of shape `(scenario.raw_input_dim,)`.
        """

        values = {
            # x, y are never model inputs; 0.0 placeholder if unknown (only
            # possible for scenarios that do not need the position, see
            # _compute_accelerations).
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
            env = self.get_env(np.array([state.x, state.y]), state)
            values.update(ukc=env.ukc,
                          wind_speed=env.wind_speed, wind_direction=env.wind_direction,
                          current_speed=env.current_speed, current_direction=env.current_direction)

        if self.scenario.includes_rays:
            values["rays"] = compute_ray_features(
                self._ray_caster(),
                np.array([state.x, state.y]),
                float(state.heading),
            )

        return self.scenario.build_raw_row(values)

    def _compute_accelerations(
        self,
        state: VesselState,
        rudder_angle: float,
        thrust: float,
    ) -> Tuple[float, float, float]:
        """
        Compute body-frame accelerations using the trained surrogate.

        The surrogate predicts body-frame velocity changes over one training
        time step (`SURROGATE_DT`), converted to accelerations by dividing by
        that interval.

        Args:
            state:        Current vessel state.
            rudder_angle: Rudder command [rad]; positive = the ship turns to PORT
                (framework convention, see DynamicsBase).
            thrust:       Normalized thrust command in `[-1, 1]`.

        Returns:
            Tuple `(du, dv, dr)` containing the predicted body-frame accelerations.

        Raises:
            ValueError: If `state.heading` is None, or `state.x`/`state.y` is
                None for a scenario that needs the position.
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
        delta_u, delta_v, delta_r = self.surrogate.predict_single(x_raw)
        return float(delta_u / SURROGATE_DT), float(delta_v / SURROGATE_DT), float(delta_r / SURROGATE_DT)
