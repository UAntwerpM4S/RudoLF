"""
Scenario schema definitions for the ship-dynamics surrogate pipeline.

This module (numpy only, no torch) is the single source of truth for:

  * which raw columns a training scenario collects, and which of them
    become model inputs (`ScenarioSchema`, `OPEN_WATER`, `CONFINED_UKC`,
    `FULL_ENV`);
  * how a raw row is turned into model features (`select_and_encode`,
    `encoded_feature_names`), including how wind and current are encoded
    (`env_encoding`);
  * which poorer scenario may be derived from a richer one by dropping
    columns (`check_projection`, `project_raw_columns`);
  * the identity every dataset .npz carries (`DatasetIdentity`,
    `npz_metadata`, `read_npz_identity`).

Dataset (dataset.py), predictor (inference_surrogate.py), training
(train_surrogate.py) and the UAntwerp surrogate dynamics model
(fw/dynamics/uantwerp_surrogate.py) all go through these functions, so
they cannot disagree about the feature layout of a scenario.

Three scenarios are defined:

  open_water   - plain dynamics only. No bank/collision rays, no UKC, no
                 wind or current. x, y are still collected in the raw row
                 (for trajectory logging) but are dropped before the model
                 sees them, same as every other scenario.
  confined_ukc - adds bank-collision ray features and under-keel clearance.
                 Still no wind or current.
  full_env     - everything: rays, UKC, wind, and current (37 raw columns,
                 matching sim_fh_dataset_generator.collect_supervised_dataset's
                 column layout exactly).

The raw widths of the three scenarios are all different (8 / 33 / 37), so a
raw width identifies its scenario (`scenario_for_raw_width`).

How wind and current are ENCODED for the model is a separate, model-side
choice (`env_encoding`, see ENV_ENCODING_* below): the raw data layout and
the .npz identity do not depend on it.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Tuple

from fw.environments.ray_features import RAY_ANGLES_DEG, RAY_LAYOUT_VERSION


def _ray_column_names() -> Tuple[str, ...]:
    """
    Ray column names in generation order, from the shared ray definition
    (fw.environments.ray_features.RAY_ANGLES_DEG: 0, 15, ..., 345 deg,
    clockwise from the bow).

    Returns:
        Tuple of column names, e.g. ("ray_0", "ray_15", ..., "ray_345").
    """

    return tuple(f"ray_{int(a)}" for a in RAY_ANGLES_DEG)


# Public: same order as ray_features.RAY_ANGLES_DEG, which the dataset
# generators and the surrogate use to compute the rays.
RAY_COLUMN_NAMES = _ray_column_names()

# Model outputs: one-step body-frame velocity changes over SURROGATE_DT.
TARGET_NAMES = ("delta_surge", "delta_sway", "delta_yaw_rate")

# Sign convention of "sway" (and "delta_sway") in dataset rows and model
# outputs: the framework's (fw.ships.vessel_state) -- positive to PORT, as
# FH's own getShipVelocityOverGround().y and ShipSimulator.integrate. Stored in
# every dataset .npz and checkpoint and checked on load, so data written with
# the earlier starboard-positive sway can never be mixed in.
SWAY_CONVENTION = "port_positive"

# Time step all surrogate datasets are generated with, and the step the
# surrogate predicts deltas over [s].
SURROGATE_DT = 0.1


@dataclass(frozen=True)
class ScenarioSchema:
    """
    Column layout for one training scenario.

    Attributes:
        name: Scenario identifier, e.g. "open_water". Stored in dataset
            files and checkpoints, so data and models always carry their own
            schema rather than relying on the caller to guess it.
        raw_columns: Every column in the raw dataset row, in order. Defines
            `raw_input_dim`.
        input_columns: Ordered subset of `raw_columns` that becomes model
            input (before encoding). x/y are intentionally excluded in every
            scenario -- see module docstring.
        angle_columns: Subset of `input_columns` that are angles in
            [-pi, pi] and must be sin/cos encoded rather than passed raw,
            since a raw radian value has a discontinuity at +-pi that an
            MLP cannot represent.
    """

    name: str
    raw_columns: Tuple[str, ...]
    input_columns: Tuple[str, ...]
    angle_columns: Tuple[str, ...]

    def __post_init__(self):
        """
        Validate internal consistency of the schema at construction time.

        Raises:
            ValueError: If `raw_columns`/`input_columns` contain duplicates,
                or if `input_columns`/`angle_columns` reference a column not
                present in `raw_columns`/`input_columns` respectively.
        """

        if len(set(self.raw_columns)) != len(self.raw_columns):
            raise ValueError(f"Schema '{self.name}': duplicate raw_columns.")
        if len(set(self.input_columns)) != len(self.input_columns):
            raise ValueError(f"Schema '{self.name}': duplicate input_columns.")

        missing_inputs = set(self.input_columns) - set(self.raw_columns)
        if missing_inputs:
            raise ValueError(
                f"Schema '{self.name}': input_columns {sorted(missing_inputs)} "
                f"not present in raw_columns."
            )

        missing_angles = set(self.angle_columns) - set(self.input_columns)
        if missing_angles:
            raise ValueError(
                f"Schema '{self.name}': angle_columns {sorted(missing_angles)} "
                f"not present in input_columns."
            )

    @property
    def raw_input_dim(self) -> int:
        """
        Width of the raw dataset row this schema expects.
        """

        return len(self.raw_columns)

    @property
    def includes_rays(self) -> bool:
        """
        True if the raw row holds the ray proximity features (the surrogate
        then has to ray-cast at inference time).
        """

        return all(c in self.raw_columns for c in RAY_COLUMN_NAMES)

    @property
    def includes_ukc(self) -> bool:
        """
        True if the raw row holds the under-keel clearance.
        """

        return "ukc" in self.raw_columns

    @property
    def includes_wind_current(self) -> bool:
        """
        True if the raw row holds wind and current (speed and direction).
        """

        return any(c in self.raw_columns for c in _ENV_RAW_COLS)

    def col(self, name: str) -> int:
        """
        Position of raw column `name` in `raw_columns`.

        Raises:
            ValueError: If the scenario has no such column.
        """

        try:
            return self.raw_columns.index(name)
        except ValueError:
            raise ValueError(f"Scenario '{self.name}' has no raw column '{name}'.") from None

    def input_col_indices(self) -> Tuple[int, ...]:
        """
        Positions of `input_columns` within `raw_columns`, in input order.
        """

        return tuple(self.raw_columns.index(c) for c in self.input_columns)

    def angle_col_indices(self) -> Tuple[int, ...]:
        """
        Positions of `angle_columns` within `raw_columns`, in input order.
        """

        return tuple(self.raw_columns.index(c) for c in self.input_columns if c in self.angle_columns)

    def angle_mask(self) -> Tuple[bool, ...]:
        """
        Per-`input_columns` boolean mask, True where the column is an angle.
        """

        return tuple(c in self.angle_columns for c in self.input_columns)

    def build_raw_row(self, values: dict) -> np.ndarray:
        """
        Assemble one raw row in this scenario's column order.

        Args:
            values: Mapping column name -> value. Every column of
                `raw_columns` must be present, except the ray columns, which
                may instead be given at once as values["rays"] (a sequence
                of len(RAY_COLUMN_NAMES), in RAY_COLUMN_NAMES order). Keys
                for columns this scenario does not have are ignored.

        Returns:
            float32 array of shape (raw_input_dim,).

        Raises:
            KeyError: If a column of this scenario has no value.
        """

        row = np.empty(self.raw_input_dim, dtype=np.float32)
        rays = values.get("rays")
        for i, name in enumerate(self.raw_columns):
            if name in values:
                row[i] = values[name]
            elif rays is not None and name in RAY_COLUMN_NAMES:
                row[i] = rays[RAY_COLUMN_NAMES.index(name)]
            else:
                raise KeyError(f"build_raw_row: no value for column '{name}' of scenario '{self.name}'.")
        return row


OPEN_WATER = ScenarioSchema(
    name="open_water",
    raw_columns=("x", "y", "heading", "surge", "sway", "yaw_rate", "rudder_action", "thrust_action"),
    input_columns=("heading", "surge", "sway", "yaw_rate", "rudder_action", "thrust_action"),
    angle_columns=("heading",),
)

CONFINED_UKC = ScenarioSchema(
    name="confined_ukc",
    raw_columns=(
        "x", "y", "heading", "surge", "sway", "yaw_rate", "ukc", "rudder_action", "thrust_action",
    ) + RAY_COLUMN_NAMES,
    input_columns=(
        "heading", "surge", "sway", "yaw_rate", "ukc", "rudder_action", "thrust_action",
    ) + RAY_COLUMN_NAMES,
    angle_columns=("heading",),
)

FULL_ENV = ScenarioSchema(
    name="full_env",
    raw_columns=(
        "x", "y", "heading", "surge", "sway", "yaw_rate", "ukc",
        "wind_speed", "wind_direction", "current_speed", "current_direction",
        "rudder_action", "thrust_action",
    ) + RAY_COLUMN_NAMES,
    input_columns=(
        "heading", "surge", "sway", "yaw_rate", "ukc",
        "wind_speed", "wind_direction", "current_speed", "current_direction",
        "rudder_action", "thrust_action",
    ) + RAY_COLUMN_NAMES,
    angle_columns=("heading", "wind_direction", "current_direction"),
)

SCENARIOS = {s.name: s for s in (OPEN_WATER, CONFINED_UKC, FULL_ENV)}

if len({s.raw_input_dim for s in SCENARIOS.values()}) != len(SCENARIOS):
    raise RuntimeError("Scenario raw widths must be unique (scenario_for_raw_width relies on it).")


def get_scenario(name: str) -> ScenarioSchema:
    """
    Look up a scenario schema by name.

    Args:
        name: Scenario identifier -- "open_water", "confined_ukc", or
            "full_env".

    Returns:
        The matching `ScenarioSchema`.

    Raises:
        ValueError: If `name` is not a registered scenario.
    """

    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f"Unknown scenario '{name}'. Available scenarios: {sorted(SCENARIOS)}"
        ) from None


def scenario_for_raw_width(width: int) -> ScenarioSchema:
    """
    The scenario whose raw row is `width` columns wide (widths are unique).

    Raises:
        ValueError: If no scenario has that width.
    """

    for scenario in SCENARIOS.values():
        if scenario.raw_input_dim == width:
            return scenario
    accepted = ", ".join(f"{s.name} ({s.raw_input_dim})" for s in SCENARIOS.values())
    raise ValueError(f"No scenario has a raw width of {width}; known: {accepted}.")


# ----------------------------------------------------------------------
# Environment encoding (model side; the RAW data layout above is unchanged)
# ----------------------------------------------------------------------
# How the wind/current columns of a raw row become model inputs:
#
#   "angles"          - wind_speed and current_speed raw, wind_direction and
#                       current_direction sin/cos encoded like the heading.
#                       Wind is ship-relative (apparent, from FH's WindState),
#                       current earth-fixed, so the model has to learn the
#                       current's rotation by the heading itself (encoded
#                       width 38 for full_env).
#   "body_components" - wind and current as components in the SHIP frame:
#                         wind_from_ahead = ws * cos(wd)
#                         wind_from_stbd  = ws * sin(wd)
#                       (wd: WindState direction relative to the bow, positive
#                       = from starboard; apparent wind, unchanged frame) and
#                         current_to_ahead =  cs * cos(cd - heading)
#                         current_to_stbd  = -cs * sin(cd - heading)
#                       (cd: earth-fixed, 0 = East, counter-clockwise, flowing
#                       toward; heading counter-clockwise, so the relative
#                       angle is counter-clockwise = towards port, hence the
#                       minus sign for starboard). The four columns replace
#                       wind_speed/wind_direction/current_speed/current_direction
#                       in place; only the heading stays sin/cos encoded
#                       (encoded width 36 for full_env). Components go
#                       smoothly to 0 at low speed, where an angle is noise.
#
# For scenarios without wind/current (open_water, confined_ukc) both
# encodings give identical features.
ENV_ENCODING_ANGLES = "angles"
ENV_ENCODING_BODY = "body_components"
ENV_ENCODINGS = (ENV_ENCODING_ANGLES, ENV_ENCODING_BODY)
# Encoding for NEW training runs. Inference always uses the encoding stored
# in the checkpoint.
DEFAULT_ENV_ENCODING = ENV_ENCODING_BODY

_ENV_RAW_COLS = ("wind_speed", "wind_direction", "current_speed", "current_direction")
_ENV_BODY_NAMES = ("wind_from_ahead", "wind_from_stbd", "current_to_ahead", "current_to_stbd")


def check_env_encoding(env_encoding: str) -> str:
    """
    Return `env_encoding` if it is known.

    Raises:
        ValueError: If it is not one of ENV_ENCODINGS.
    """

    if env_encoding not in ENV_ENCODINGS:
        raise ValueError(f"Unknown env_encoding {env_encoding!r}; expected one of {ENV_ENCODINGS}.")
    return env_encoding


def _uses_body_env(scenario: ScenarioSchema, env_encoding: str) -> bool:
    """True if `env_encoding` is body_components AND the scenario has wind/current inputs."""

    check_env_encoding(env_encoding)
    if env_encoding != ENV_ENCODING_BODY:
        return False
    present = [c in scenario.input_columns for c in _ENV_RAW_COLS]
    if not any(present):
        return False          # open_water / confined_ukc: nothing to convert
    if not all(present) or "heading" not in scenario.input_columns:
        raise ValueError(f"Scenario '{scenario.name}': body_components needs all of "
                         f"{_ENV_RAW_COLS} and heading as inputs.")
    return True


def encoded_feature_names(scenario: ScenarioSchema, env_encoding: str) -> Tuple[str, ...]:
    """
    Names of the encoded model-input features, in the exact order produced by
    `select_and_encode(x_raw, scenario, env_encoding)`.
    """

    body = _uses_body_env(scenario, env_encoding)
    non_angle, angle = [], []
    for c in scenario.input_columns:
        if body and c in _ENV_RAW_COLS:
            non_angle.append(_ENV_BODY_NAMES[_ENV_RAW_COLS.index(c)])
        elif c in scenario.angle_columns:
            angle.append(c)
        else:
            non_angle.append(c)
    return tuple(non_angle) + tuple(f"sin_{a}" for a in angle) + tuple(f"cos_{a}" for a in angle)


def encoded_input_dim(scenario: ScenarioSchema, env_encoding: str) -> int:
    """
    Width of the model input for `scenario` with `env_encoding`.
    """

    return len(encoded_feature_names(scenario, env_encoding))


def select_and_encode(x_raw: np.ndarray, scenario: ScenarioSchema, env_encoding: str) -> np.ndarray:
    """
    Select `scenario.input_columns` from a raw X block and sin/cos-encode
    the angular columns; with env_encoding="body_components", wind and
    current become ship-frame components (see ENV_ENCODING_BODY above).

    This is the ONLY place raw-column selection happens. The dataset, the
    normalization stats and `SurrogatePredictor.predict` all call this, so
    they are guaranteed to agree on the feature set.

    Args:
        x_raw: Raw input array, shape `(N, scenario.raw_input_dim)`.
        scenario: The scenario schema to apply.
        env_encoding: "angles" or "body_components".

    Returns:
        float32 feature array, shape `(N, encoded_input_dim(scenario, env_encoding))`,
        in the order of `encoded_feature_names(scenario, env_encoding)`:
        non-angle `input_columns` in order (with body_components: the four
        wind/current columns replaced in place by their components),
        followed by sin(angle) for each angle column (in `input_columns`
        order), followed by cos(angle) for each angle column.

    Raises:
        ValueError: If `x_raw` is not shaped `(N, scenario.raw_input_dim)`.
    """

    if x_raw.ndim != 2 or x_raw.shape[1] != scenario.raw_input_dim:
        raise ValueError(
            f"Expected raw X with shape (N, {scenario.raw_input_dim}) for "
            f"scenario '{scenario.name}', got {x_raw.shape}"
        )

    if not _uses_body_env(scenario, env_encoding):
        indices = list(scenario.input_col_indices())
        selected = x_raw[:, indices]
        angle_mask = np.array(scenario.angle_mask())

        non_angle_vals = selected[:, ~angle_mask]
        angle_vals = selected[:, angle_mask]

        return np.concatenate(
            [non_angle_vals, np.sin(angle_vals), np.cos(angle_vals)],
            axis=1,
        ).astype(np.float32)

    # body_components: compute in float64, then cast once.
    col = {c: x_raw[:, scenario.col(c)].astype(np.float64) for c in _ENV_RAW_COLS + ("heading",)}
    rel_cur = col["current_direction"] - col["heading"]
    body = {
        "wind_speed": col["wind_speed"] * np.cos(col["wind_direction"]),          # wind_from_ahead
        "wind_direction": col["wind_speed"] * np.sin(col["wind_direction"]),      # wind_from_stbd
        "current_speed": col["current_speed"] * np.cos(rel_cur),                  # current_to_ahead
        "current_direction": -col["current_speed"] * np.sin(rel_cur),             # current_to_stbd
    }
    non_angle, angle = [], []
    for c in scenario.input_columns:
        if c in body:
            non_angle.append(body[c])
        elif c in scenario.angle_columns:
            angle.append(x_raw[:, scenario.col(c)].astype(np.float64))
        else:
            non_angle.append(x_raw[:, scenario.col(c)].astype(np.float64))
    angle_arr = np.stack(angle, axis=1) if angle else np.zeros((len(x_raw), 0))
    return np.concatenate(
        [np.stack(non_angle, axis=1), np.sin(angle_arr), np.cos(angle_arr)],
        axis=1,
    ).astype(np.float32)


# ----------------------------------------------------------------------
# Projection: deriving a poorer scenario's data from a richer one
# ----------------------------------------------------------------------

def check_projection(source: ScenarioSchema, target: ScenarioSchema) -> None:
    """
    Raise if data generated for `source` may not be used for a `target` model.

    Allowed: `source == target`, or `target.raw_columns` is an ordered subset
    of `source.raw_columns` AND `source` has no wind/current that `target`
    lacks. So open_water <- confined_ukc is allowed (rays and UKC dropped),
    but nothing <- full_env except full_env itself: wind and current change
    the SIMULATED dynamics, not just which columns get logged, so a full_env
    run's Y targets contain effects that a model without wind/current inputs
    cannot explain.

    Caveat for the allowed case: confined_ukc episodes ran against real river
    geometry, so the Y deltas may contain some bank/shallow-water influence
    that the open_water inputs cannot explain.

    Raises:
        ValueError: If the projection is not allowed.
    """

    if source == target:
        return
    order = [source.raw_columns.index(c) for c in target.raw_columns if c in source.raw_columns]
    if len(order) != len(target.raw_columns) or order != sorted(order):
        raise ValueError(
            f"Cannot derive '{target.name}' data from '{source.name}': '{target.name}' raw columns "
            f"are not an ordered subset of '{source.name}' raw columns."
        )
    if source.includes_wind_current and not target.includes_wind_current:
        raise ValueError(
            f"Cannot derive '{target.name}' data from '{source.name}': the '{source.name}' episodes "
            f"were simulated with wind/current, which a '{target.name}' model has no inputs for."
        )


def project_raw_columns(x_raw: np.ndarray, from_scenario: ScenarioSchema, to_scenario: ScenarioSchema) -> np.ndarray:
    """
    Derive `to_scenario`'s raw X columns from `from_scenario` raw X by
    dropping columns -- no new simulation needed. Only allowed combinations
    (see `check_projection`) are accepted.

    Args:
        x_raw: Raw X array generated for `from_scenario`, shape
            `(N, from_scenario.raw_input_dim)`.
        from_scenario: The scenario `x_raw` was generated for.
        to_scenario: The scenario to project `x_raw` down to.

    Returns:
        Raw X array for `to_scenario`, shape `(N, to_scenario.raw_input_dim)`.

    Raises:
        ValueError: If `x_raw`'s width doesn't match `from_scenario`, or the
            projection is not allowed.
    """

    if x_raw.ndim != 2 or x_raw.shape[1] != from_scenario.raw_input_dim:
        raise ValueError(
            f"Expected raw X with shape (N, {from_scenario.raw_input_dim}) for "
            f"scenario '{from_scenario.name}', got {x_raw.shape}"
        )
    check_projection(from_scenario, to_scenario)
    if from_scenario == to_scenario:
        return x_raw
    indices = [from_scenario.raw_columns.index(c) for c in to_scenario.raw_columns]
    return x_raw[:, indices]


# ----------------------------------------------------------------------
# Dataset identity (.npz)
# ----------------------------------------------------------------------
# Stored in every dataset .npz (generate_ship_data.py), required and checked
# when it is read (dataset.load_raw), and copied into every checkpoint
# (train_surrogate.py).
NPZ_SCENARIO_KEY = "scenario"
NPZ_RAY_LAYOUT_KEY = "ray_layout"
NPZ_SHIP_MODEL_KEY = "ship_model"            # fw.ships.ship_factory.ShipModel name, e.g. "TRIPOLI"
NPZ_RUDDER_MAX_DEG_KEY = "rudder_max_deg"    # rudder_action = +-1 <-> +-this many degrees
NPZ_SWAY_CONVENTION_KEY = "sway_convention"  # SWAY_CONVENTION at generation time


@dataclass(frozen=True)
class DatasetIdentity:
    """
    What a dataset file was generated for.

    Attributes:
        scenario: Scenario name of the raw X layout in the file.
        ray_layout: ray_features.RAY_LAYOUT_VERSION at generation time.
        ship_model: ShipModel name of the simulated ship, e.g. "TRIPOLI".
        rudder_max_deg: Rudder angle [deg] that rudder_action = 1 stands for
            (the generator's RUDDER_MAX_DEG). The surrogate converts a rudder
            angle to its input with this value.
        sway_convention: Sign convention of the sway columns (SWAY_CONVENTION).
    """

    scenario: str
    ray_layout: str
    ship_model: str
    rudder_max_deg: float
    sway_convention: str = SWAY_CONVENTION


def npz_metadata(identity: DatasetIdentity) -> dict:
    """
    Identity entries to store in a dataset .npz (0-d arrays).
    """

    get_scenario(identity.scenario)  # validates the name
    return {NPZ_SCENARIO_KEY: np.array(identity.scenario),
            NPZ_RAY_LAYOUT_KEY: np.array(identity.ray_layout),
            NPZ_SHIP_MODEL_KEY: np.array(identity.ship_model),
            NPZ_RUDDER_MAX_DEG_KEY: np.array(float(identity.rudder_max_deg)),
            NPZ_SWAY_CONVENTION_KEY: np.array(identity.sway_convention)}


def read_npz_identity(data, path) -> DatasetIdentity:
    """
    Read and validate the identity of an opened .npz (`np.load` result).

    Raises:
        ValueError: If a key is missing, the scenario is unknown, or the ray
            layout or sway convention is not the current one.
    """

    keys = (NPZ_SCENARIO_KEY, NPZ_RAY_LAYOUT_KEY, NPZ_SHIP_MODEL_KEY, NPZ_RUDDER_MAX_DEG_KEY,
            NPZ_SWAY_CONVENTION_KEY)
    missing = [k for k in keys if k not in data.files]
    if missing:
        raise ValueError(
            f"{path}: missing identity key(s) {missing}. Regenerate the file with the current "
            f"generate_ship_data.py."
        )
    identity = DatasetIdentity(
        scenario=str(data[NPZ_SCENARIO_KEY]),
        ray_layout=str(data[NPZ_RAY_LAYOUT_KEY]),
        ship_model=str(data[NPZ_SHIP_MODEL_KEY]),
        rudder_max_deg=float(data[NPZ_RUDDER_MAX_DEG_KEY]),
        sway_convention=str(data[NPZ_SWAY_CONVENTION_KEY]),
    )
    get_scenario(identity.scenario)
    if identity.ray_layout != RAY_LAYOUT_VERSION:
        raise ValueError(
            f"{path}: ray_layout={identity.ray_layout!r}, expected {RAY_LAYOUT_VERSION!r}. "
            f"Regenerate this file with the current generator."
        )
    if identity.sway_convention != SWAY_CONVENTION:
        raise ValueError(
            f"{path}: sway_convention={identity.sway_convention!r}, expected {SWAY_CONVENTION!r}. "
            f"Regenerate this file with the current generator."
        )
    return identity
