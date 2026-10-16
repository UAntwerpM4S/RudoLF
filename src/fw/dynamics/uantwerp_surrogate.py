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
     "checkpoint": "checkpoints/x.pth",     # optional override of the preset's
     "lookup_csv": "x.csv"}                 # optional override (None = no lookup)

The checkpoint must match the preset (scenario and ship); the rudder angle is
converted to the model's rudder_action with the rudder scale stored in the
checkpoint (the generator's RUDDER_MAX_DEG for that data).

Hooks used by the model-comparison tests (kept on purpose):
  * `get_env(pos)`: the ONLY source of UKC/wind/current for the input row;
    may be replaced by an instance attribute (one-argument function).
  * `_env`: the ray caster (only present for scenarios with rays); may be
    replaced by another PySimEnv-like object.
  * `ukc` / `get_ukc(pos)`: last UKC used.
  * `_compute_accelerations(state, rudder_angle, thrust)`.

Wind/current: `get_env` returns zero wind and current (only the UKC comes
from the lookup), as before. For a full_env checkpoint inside PySimEnv the
model therefore always sees calm conditions.
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
from fw.environments.ray_features import compute_ray_features
from fw.surrogate.presets import get_preset, resolve_lookup_csv
from fw.surrogate.schema import SURROGATE_DT

logger = logging.getLogger(__name__)

# Keys accepted in ua_surrogate_config.
CONFIG_KEYS = ("preset", "checkpoint", "lookup_csv")


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

        checkpoint = config.get("checkpoint", self.preset.checkpoint)
        self.surrogate = SurrogatePredictor(checkpoint)
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

        # UKC: nearest-neighbour lookup, only for scenarios with a UKC input.
        self.default_ukc = float(self.preset.default_ukc)
        self.ukc = self.default_ukc
        self.wcu_lookup = None
        if self.scenario.includes_ukc:
            csv = resolve_lookup_csv(config.get("lookup_csv", self.preset.lookup_csv))
            if csv is not None and csv.exists():
                self.wcu_lookup = WcuLookup(str(csv))
            elif csv is not None:
                logger.warning("Ship data file %s not found, falling back to default UKC value of %s",
                               csv, self.default_ukc)

        if self.scenario.includes_rays:
            self._env = ray_caster

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

    def get_env(self, pos: Optional[np.ndarray]) -> EnvironmentData:
        """
        Environmental conditions at `pos`: UKC from the nearest-neighbour
        lookup (or `default_ukc`), zero wind and current.

        Uses `wcu_lookup.get_ukc()` only: the UKC column (6) is the same in
        the confined_ukc and full_env CSV layouts, the wind/current columns
        are not.

        Args:
            pos: World-frame position `[x, y]` in metres.

        Returns:
            Environmental conditions at the requested position.
        """

        if self.wcu_lookup is None:
            ukc = self.default_ukc
        else:
            looked_up_ukc = self.wcu_lookup.get_ukc(pos[0], pos[1])
            ukc = looked_up_ukc if looked_up_ukc is not None else self.default_ukc

        self.ukc = ukc
        return EnvironmentData(ukc=ukc, wind_speed=0.0, wind_direction=0.0,
                               current_speed=0.0, current_direction=0.0)

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
            rudder_angle:                Rudder deflection angle [rad], positive to starboard.
            thrust:                      Normalized thrust command in `[-1, 1]`.
            dt:                          Integration time step [s]; must equal SURROGATE_DT.
            enable_rudder_angle_scaling: Passed to the base implementation for interface
                                         compatibility. Ignored by this class.

        Returns:
            Tuple `(u, v, r)` containing the updated body-frame velocities.

        Raises:
            AssertionError: If `dt` differs from `SURROGATE_DT`.
        """

        assert np.isclose(dt, SURROGATE_DT), (
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
            env = self.get_env(np.array([state.x, state.y]))
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
            rudder_angle: Rudder deflection angle [rad], positive to starboard.
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
