"""
Scenario schemas for the ship-dynamics surrogate pipeline (numpy only).

Single source of truth for:
  * the raw columns of each scenario and which become model inputs;
  * raw row to model features (select_and_encode, encoded_feature_names),
    including the wind/current encoding (env_encoding);
  * which poorer scenario may be derived from a richer one (check_projection,
    project_raw_columns);
  * the identity every dataset .npz carries (DatasetIdentity).

Scenarios (raw widths 8 / 33 / 37, see scenario_for_raw_width):
  open_water     dynamics only.
  confined_ukc   adds bank rays and under-keel clearance.
  full_env       rays, UKC, wind and current (the layout of
                 sim_fh_dataset_generator.collect_supervised_dataset).
x and y are in every raw row but never a model input. The wind/current
encoding is a model-side choice; raw layout and .npz identity do not depend
on it.
"""

from __future__ import annotations

import json
import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple
from fw.environments.ray_features import RAY_ANGLES_DEG, RAY_LAYOUT_VERSION


def _ray_column_names() -> Tuple[str, ...]:
    """
    Ray column names ("ray_0", ..., "ray_345") in RAY_ANGLES_DEG order (clockwise from the bow).
    """

    return tuple(f"ray_{int(a)}" for a in RAY_ANGLES_DEG)


RAY_COLUMN_NAMES = _ray_column_names()

# Model outputs: one-step body-frame velocity changes over SURROGATE_DT.
TARGET_NAMES = ("delta_surge", "delta_sway", "delta_yaw_rate")

# Sign of "sway" and "delta_sway" in datasets and model outputs: positive to
# port (framework convention, as FH's getShipVelocityOverGround().y). Stored in
# every .npz and checkpoint and checked on load, so starboard-positive data
# cannot be mixed in.
SWAY_CONVENTION = "port_positive"

# Dataset time step and the step the surrogate predicts deltas over [s].
SURROGATE_DT = 0.1


@dataclass(frozen=True)
class ScenarioSchema:
    """
    Column layout for one training scenario.

    Attributes:
        name: Scenario identifier, e.g. "open_water"; stored in datasets and
            checkpoints.
        raw_columns: All columns of a raw dataset row, in order.
        input_columns: Ordered subset of `raw_columns` used as model input
            (before encoding); never x/y.
        angle_columns: Subset of `input_columns` that are angles in [-pi, pi],
            sin/cos encoded because of the discontinuity at +-pi.
    """

    name: str
    raw_columns: Tuple[str, ...]
    input_columns: Tuple[str, ...]
    angle_columns: Tuple[str, ...]

    def __post_init__(self):
        """
        Raises:
            ValueError: On duplicate columns, input_columns not in raw_columns,
                or angle_columns not in input_columns.
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
        Width of the raw dataset row.
        """

        return len(self.raw_columns)

    @property
    def includes_rays(self) -> bool:
        """
        True if the raw row holds the ray features (ray-casting needed at inference).
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
        Per-`input_columns` mask, True where the column is an angle.
        """

        return tuple(c in self.angle_columns for c in self.input_columns)

    def build_raw_row(self, values: dict) -> np.ndarray:
        """
        Assemble one raw row in this scenario's column order.

        Args:
            values: Column name to value for every raw column; the rays may
                instead be given as values["rays"] in RAY_COLUMN_NAMES order.
                Keys of columns this scenario lacks are ignored.

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
    Scenario schema by name ("open_water", "confined_ukc" or "full_env").

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
# Environment encoding (model side only)
# ----------------------------------------------------------------------
# "angles": speeds raw, directions sin/cos encoded; wind apparent (FH
#     WindState), current earth-fixed (encoded width 38 for full_env).
# "body_components": ship-frame components in place of the four columns:
#     wind_from_ahead  =  ws * cos(wd)    (wd: from bow, + from stbd)
#     wind_from_stbd   =  ws * sin(wd)
#     current_to_ahead =  cs * cos(cd - heading)
#     current_to_stbd  = -cs * sin(cd - heading)
#     (cd: earth-fixed, 0 = East, CCW, flowing toward); only the heading stays
#     sin/cos encoded (width 36 for full_env).
# Without wind/current both give identical features.
ENV_ENCODING_ANGLES = "angles"
ENV_ENCODING_BODY = "body_components"
ENV_ENCODINGS = (ENV_ENCODING_ANGLES, ENV_ENCODING_BODY)
# Encoding for new training runs; inference uses the checkpoint's encoding.
DEFAULT_ENV_ENCODING = ENV_ENCODING_BODY

_ENV_RAW_COLS = ("wind_speed", "wind_direction", "current_speed", "current_direction")
_ENV_BODY_NAMES = ("wind_from_ahead", "wind_from_stbd", "current_to_ahead", "current_to_stbd")


def env_body_components(wind_speed, wind_direction, current_speed, current_direction, heading):
    """
    Ship-frame wind and current of the "body_components" encoding, also used
    for the RL observation. Works on floats and numpy arrays.

    Args:
        wind_speed: Apparent wind speed [m/s]
        wind_direction: Apparent wind direction [rad], relative
            to the bow, positive from starboard.
        current_speed: Earth-fixed current speed [m/s]
        current_direction: Earth-fixed current direction [rad],
            0 = East, CCW, flowing toward.
        heading: Ship heading [rad], 0 = East, CCW.

    Returns:
        (wind_from_ahead, wind_from_stbd, current_to_ahead, current_to_stbd).
    """

    rel_cur = current_direction - heading
    return (
        wind_speed * np.cos(wind_direction),
        wind_speed * np.sin(wind_direction),
        current_speed * np.cos(rel_cur),
        -current_speed * np.sin(rel_cur),
    )


def apparent_to_true_wind(app_speed: float, app_dir: float, u: float, v: float) -> Tuple[float, float]:
    """
    True wind in the ship frame from the apparent wind (dataset / FH WindState
    convention): apparent = true + (u, -v) in (from ahead, from starboard)
    components, since sway v is positive to port.

    Returns:
        (true wind from ahead, true wind from starboard) [m/s].
    """

    rel = app_speed * np.array([np.cos(app_dir), np.sin(app_dir)]) - np.array([u, -v])
    return float(rel[0]), float(rel[1])


def check_env_encoding(env_encoding: str) -> str:
    """
    Return `env_encoding` if known.

    Raises:
        ValueError: If it is not one of ENV_ENCODINGS.
    """

    if env_encoding not in ENV_ENCODINGS:
        raise ValueError(f"Unknown env_encoding {env_encoding!r}; expected one of {ENV_ENCODINGS}.")
    return env_encoding


def _uses_body_env(scenario: ScenarioSchema, env_encoding: str) -> bool:
    """
    True if `env_encoding` is body_components AND the scenario has wind/current inputs.
    """

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
    Names of the encoded features, in `select_and_encode` output order.
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
    Select `scenario.input_columns` from raw X and encode them (angles sin/cos;
    with "body_components" wind/current as ship-frame components). The only
    place raw columns are selected.

    Args:
        x_raw: Raw input array, shape (N, scenario.raw_input_dim).
        scenario: The scenario schema to apply.
        env_encoding: "angles" or "body_components".

    Returns:
        float32 array (N, encoded_input_dim), ordered as encoded_feature_names:
        non-angle inputs, then sin of each angle column, then cos.

    Raises:
        ValueError: If x_raw is not shaped (N, scenario.raw_input_dim).
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
    # Each raw column is replaced in place by its body component (_ENV_BODY_NAMES).
    body = dict(zip(_ENV_RAW_COLS, env_body_components(
        col["wind_speed"], col["wind_direction"], col["current_speed"], col["current_direction"],
        col["heading"])))
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

    Allowed: source == target, or target's raw columns are an ordered subset of
    source's and source has no wind/current that target lacks (wind and current
    change the dynamics, not just the logged columns).

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
    Derive `to_scenario` raw X from `from_scenario` raw X by dropping columns
    (only combinations allowed by `check_projection`).

    Args:
        x_raw: Raw X of shape (N, from_scenario.raw_input_dim).
        from_scenario: Scenario `x_raw` was generated for.
        to_scenario: Target scenario.

    Returns:
        Raw X of shape (N, to_scenario.raw_input_dim).

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
# Written by generate_ship_data.py, checked by dataset.load_raw, copied into
# every checkpoint by train_surrogate.py.
NPZ_SCENARIO_KEY = "scenario"
NPZ_RAY_LAYOUT_KEY = "ray_layout"
NPZ_SHIP_MODEL_KEY = "ship_model"            # fw.ships.ship_factory.ShipModel name, e.g. "TRIPOLI"
NPZ_RUDDER_MAX_DEG_KEY = "rudder_max_deg"    # rudder_action = +-1 <-> +-this many degrees
NPZ_SWAY_CONVENTION_KEY = "sway_convention"  # SWAY_CONVENTION at generation time
# Optional provenance (generator settings, seed, exercises, ...) as JSON. Not
# part of the identity: datasets of different provenance may be combined.
NPZ_PROVENANCE_KEY = "generation_json"


@dataclass(frozen=True)
class DatasetIdentity:
    """
    What a dataset file was generated for.

    Attributes:
        scenario: Scenario name of the raw X layout in the file.
        ray_layout: ray_features.RAY_LAYOUT_VERSION at generation time.
        ship_model: ShipModel name of the simulated ship, e.g. "TRIPOLI".
        rudder_max_deg: Rudder angle [deg] for rudder_action = 1; the surrogate
            scales rudder angles to its input with it.
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


def provenance_metadata(provenance: dict) -> dict:
    """
    Provenance entry for a dataset .npz (0-d JSON string array, no pickle needed).
    """

    return {NPZ_PROVENANCE_KEY: np.array(json.dumps(provenance, sort_keys=True))}


def read_npz_provenance(data) -> Optional[dict]:
    """
    Provenance of an opened .npz, or None if it has none.
    """

    if NPZ_PROVENANCE_KEY not in data.files:
        return None
    return json.loads(str(data[NPZ_PROVENANCE_KEY]))


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
