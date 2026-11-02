"""
Dataset files (.npz) and the torch Dataset for surrogate training, for every
scenario (open_water / confined_ukc / full_env, see schema.py).

A dataset file holds raw inputs "X" (in the layout of the scenario it was generated
for), targets "Y" and its identity (schema.DatasetIdentity). A model can be trained
for the file's own scenario or for a poorer one obtained by dropping columns
(schema.check_projection, e.g. open_water from confined_ukc data).

Normalization: X and Y use mean/std from the training split only. Near-constant
features get std = 1.0 (see `safe_std`), so later deviations are not amplified ~1e8.
"""

import logging

import torch
import numpy as np

from pathlib import Path
from typing import Optional, Sequence, Tuple
from torch.utils.data import Dataset

from fw.surrogate.schema import (
    DEFAULT_ENV_ENCODING,
    TARGET_NAMES,
    DatasetIdentity,
    ScenarioSchema,
    check_env_encoding,
    encoded_feature_names,
    get_scenario,
    project_raw_columns,
    read_npz_identity,
    read_npz_provenance,
    select_and_encode,
)

logger = logging.getLogger(__name__)

# Features with std below this are treated as constant (std replaced by 1.0).
STD_FLOOR = 1e-6


def read_identity(path) -> DatasetIdentity:
    """
    Validated identity of a dataset .npz (all keys present, known scenario, current ray layout).
    """

    with np.load(path, allow_pickle=False) as data:
        return read_npz_identity(data, path)


def read_provenance(path) -> Optional[dict]:
    """
    Generator settings of a dataset .npz (see generate_ship_data.generation_provenance),
    or None for files without provenance.
    """

    with np.load(path, allow_pickle=False) as data:
        return read_npz_provenance(data)


def load_raw(path, scenario: Optional[ScenarioSchema] = None) -> Tuple[np.ndarray, np.ndarray, DatasetIdentity]:
    """
    Load the raw X and Y arrays of a dataset .npz (the single entry point for all
    dataset readers).

    Args:
        path: An .npz written by generate_ship_data.py.
        scenario: Scenario to return X for; None means the file's own. Another
            scenario must be derivable from it (schema.check_projection).

    Returns:
        (x_raw, y, identity): float32 X in `scenario`'s raw layout (not encoded or
        normalized), Y as (N, 3), and the file's identity (its `scenario` is the
        source scenario).

    Raises:
        ValueError: Missing/invalid identity, disallowed projection, wrong shapes
            or NaN/inf values.
    """

    with np.load(path, allow_pickle=False) as data:
        identity = read_npz_identity(data, path)
        x_file = data["X"].astype(np.float32)
        y_vals = data["Y"].astype(np.float32)

    source = get_scenario(identity.scenario)
    if x_file.ndim != 2 or x_file.shape[1] != source.raw_input_dim:
        raise ValueError(
            f"{path}: X has shape {x_file.shape}, but the file says scenario '{source.name}' "
            f"(raw width {source.raw_input_dim})."
        )
    if y_vals.ndim == 1:
        y_vals = y_vals.reshape(-1, 1)
    if y_vals.shape != (len(x_file), len(TARGET_NAMES)):
        raise ValueError(f"{path}: Y has shape {y_vals.shape}, expected ({len(x_file)}, {len(TARGET_NAMES)}).")
    target = source if scenario is None else scenario
    x_raw = project_raw_columns(x_file, source, target)
    # Checked after projection: columns the model scenario drops do not matter.
    for name, arr in (("X", x_raw), ("Y", y_vals)):
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            raise ValueError(f"{path}: {name} contains {n_bad} NaN/inf values")
    return x_raw, y_vals, identity


def check_same_identity(paths: Sequence, identities: Sequence[DatasetIdentity]) -> DatasetIdentity:
    """
    Return the common identity; raise unless all are equal (e.g. train/val/test of one run).
    """

    first = identities[0]
    for path, ident in zip(paths[1:], identities[1:]):
        if ident != first:
            raise ValueError(f"{path} has identity {ident}, but {paths[0]} has {first}; "
                             f"all splits must come from the same generation run.")
    return first


def safe_std(values: np.ndarray, names: Optional[Sequence[str]] = None,
             what: str = "feature") -> np.ndarray:
    """
    Per-column float32 std of shape (1, D), with near-constant columns set to 1.0.

    Args:
        values: Array of shape (N, D).
        names: Optional column names for the warning.
        what: Label used in the warning ("input feature" / "target").
    """

    std = values.std(axis=0, keepdims=True).astype(np.float32)
    degenerate = std[0] < STD_FLOOR
    if degenerate.any():
        idx = np.flatnonzero(degenerate)
        use_names = names is not None and len(names) == values.shape[1]
        labels = [names[i] if use_names else str(i) for i in idx]
        logger.warning(
            "%d %s(s) are (near-)constant in the normalization source "
            "(std < %g): %s. Using std=1.0 for them. If these vary in "
            "val/test or at inference time, the model has never seen that "
            "variation -- consider dropping them or regenerating data that "
            "covers it.", len(idx), what, STD_FLOOR, labels,
        )
    std = std + 1e-8
    std[:, degenerate] = 1.0
    return std


def stats_from_arrays(x_vals: np.ndarray, y_vals: np.ndarray, feature_names: Sequence[str]):
    """
    Return (x_mean, x_std, y_mean, y_std) for encoded X and Y, each shaped (1, D).
    """

    if len(x_vals) == 0:
        raise ValueError(
            "Cannot compute normalization stats from an empty dataset (0 rows). "
            "Check that the train split actually received episodes."
        )
    x_mean = x_vals.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = safe_std(x_vals, feature_names, "input feature")
    y_mean = y_vals.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = safe_std(y_vals, TARGET_NAMES, "target")
    return x_mean, x_std, y_mean, y_std


def load_encoded(path, scenario: ScenarioSchema, env_encoding: str) -> Tuple[np.ndarray, np.ndarray, DatasetIdentity]:
    """
    `load_raw`, then select/encode X; returns (x_encoded, y, identity).
    """

    x_raw, y_vals, identity = load_raw(path, scenario)
    return select_and_encode(x_raw, scenario, env_encoding), y_vals, identity


def compute_stats(path, scenario: ScenarioSchema, env_encoding: str):
    """
    Per-feature (x_mean, x_std, y_mean, y_std) of a dataset .npz, each shaped (1, D).
    """

    x_vals, y_vals, _ = load_encoded(path, scenario, env_encoding)
    return stats_from_arrays(x_vals, y_vals, encoded_feature_names(scenario, env_encoding))


class ShipDynamicsDataset(Dataset):
    """
    Torch Dataset over one dataset .npz, encoded for one scenario.

    Attributes:
        scenario: Scenario the model inputs are built for.
        env_encoding: Wind/current encoding used for the inputs.
        identity: Identity of the file (its `scenario` is the source scenario).
        feature_names: Encoded feature names, in input order.
        X, Y: Encoded but unnormalized inputs and targets (numpy).
        x_mean, x_std, y_mean, y_std: Normalization stats (None if not normalized).
    """

    def __init__(
        self,
        path,
        scenario: Optional[ScenarioSchema] = None,
        env_encoding: str = DEFAULT_ENV_ENCODING,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        y_mean: Optional[np.ndarray] = None,
        y_std: Optional[np.ndarray] = None,
        normalize: bool = True,
    ):
        """
        Load a .npz dataset and set up normalization.

        With `normalize` and no stats given, stats are computed from this dataset;
        otherwise the given stats are used. Pass all four stats or none.

        Args:
            path: Path to the `.npz` file.
            scenario: Scenario to encode the inputs for; None = the file's own.
            env_encoding: "angles" or "body_components" (no effect for
                scenarios without wind/current).
            x_mean, x_std, y_mean, y_std: Precomputed normalization stats.
            normalize: If False, `__getitem__` returns unnormalized values.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If only some of the four stats are given, or their
                widths do not match the data.
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        self.env_encoding = check_env_encoding(env_encoding)
        x_raw, self.Y, self.identity = load_raw(path, scenario)
        self.scenario = scenario if scenario is not None else get_scenario(self.identity.scenario)
        self.X = select_and_encode(x_raw, self.scenario, self.env_encoding)
        self.feature_names = list(encoded_feature_names(self.scenario, self.env_encoding))
        self.normalize = normalize

        if normalize:
            given = [s is not None for s in (x_mean, x_std, y_mean, y_std)]
            if any(given) and not all(given):
                raise ValueError("Pass all four of x_mean/x_std/y_mean/y_std, or none.")
            if x_mean is None:
                x_mean, x_std, y_mean, y_std = stats_from_arrays(self.X, self.Y, self.feature_names)

            self.x_mean = np.asarray(x_mean, np.float32).reshape(1, -1)
            self.x_std = np.asarray(x_std, np.float32).reshape(1, -1)
            self.y_mean = np.asarray(y_mean, np.float32).reshape(1, -1)
            self.y_std = np.asarray(y_std, np.float32).reshape(1, -1)

            # Catch stats from another feature layout (would broadcast silently or fail opaquely).
            for name, stat, width in (("x_mean", self.x_mean, self.X.shape[1]),
                                      ("x_std", self.x_std, self.X.shape[1]),
                                      ("y_mean", self.y_mean, self.Y.shape[1]),
                                      ("y_std", self.y_std, self.Y.shape[1])):
                if stat.shape[1] != width:
                    raise ValueError(
                        f"{name} has {stat.shape[1]} columns but the data in {path} has "
                        f"{width}; the stats come from a different feature layout."
                    )

            x_out = (self.X - self.x_mean) / self.x_std
            y_out = (self.Y - self.y_mean) / self.y_std
        else:
            self.x_mean = self.x_std = self.y_mean = self.y_std = None
            x_out, y_out = self.X, self.Y

        # Normalized once here, so __getitem__ is a plain index.
        self._x_t = torch.from_numpy(np.ascontiguousarray(x_out, dtype=np.float32))
        self._y_t = torch.from_numpy(np.ascontiguousarray(y_out, dtype=np.float32))

    def __len__(self):
        """
        Number of samples.
        """

        return len(self._x_t)

    def __getitem__(self, idx):
        """
        `(x, y)` tensors at `idx` (normalized if `self.normalize`).
        """

        return self._x_t[idx], self._y_t[idx]
