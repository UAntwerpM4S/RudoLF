"""
Nearest-neighbor lookup of Under Keel Clearance (UKC) and environmental
conditions (wind speed/direction, current speed/direction) from a dataset CSV
written by the surrogate dataset generator, backed by a scipy cKDTree.

CSV layout: the generator's raw X row (the scenario's raw columns, see
fw.surrogate.schema), then the 3 targets, then episode_id. The column
positions therefore depend on the scenario:

    full_env      x=0, y=1, heading=2, surge=3, sway=4, UKC=6, wind speed=7,
                  wind direction=8, current speed=9, current direction=10
    confined_ukc  x=0, y=1, heading=2, surge=3, sway=4, UKC=6 (no wind/current)

The defaults below are the full_env positions. For a CSV without wind/current
pass None for those columns (fw.dynamics.uantwerp_surrogate derives all
positions from the scenario schema). Only the requested columns are read, so
a large dataset CSV costs a few columns of memory, not the whole file.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Optional, Union
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

Column = Optional[Union[int, str]]


@dataclass(frozen=True, slots=True)
class EnvironmentData:
    """
    Container for environmental conditions returned by a nearest-neighbor lookup.

    All angular quantities are expressed in radians. A field is `None` when no
    dataset point satisfies the requested search constraints, or when the
    lookup was built without that column.

    Attributes:
        ukc: Under-keel clearance in meters.
        wind_speed: Wind speed in meters per second.
        wind_direction: Wind direction in radians.
        current_speed: Current speed in meters per second.
        current_direction: Current direction in radians.
    """

    ukc: float | None               # m
    wind_speed: float | None        # m/s
    wind_direction: float | None    # rad
    current_speed: float | None     # m/s
    current_direction: float | None # rad


@dataclass(frozen=True, slots=True)
class LookupSample:
    """
    The nearest dataset row: its environment plus the state of the ship that
    recorded it (needed to turn its apparent wind into a true wind).

    Attributes:
        env: Environmental conditions of the row.
        heading: Heading of the recording ship [rad], or None if not loaded.
        surge: Surge velocity of the recording ship [m/s], or None.
        sway: Sway velocity of the recording ship [m/s], port positive, or None.
        distance: Distance from the query point to the row [m].
    """

    env: EnvironmentData
    heading: float | None
    surge: float | None
    sway: float | None
    distance: float


_ENV_FIELDS = ("ukc", "wind_speed", "wind_direction", "current_speed", "current_direction")
_STATE_FIELDS = ("heading", "surge", "sway")

# Structured dtype used for batch results
_ENV_DTYPE = np.dtype([(name, np.float32) for name in _ENV_FIELDS])


class WcuLookup:
    """
    Nearest-neighbor lookup for under-keel clearance and environmental data.

    The lookup is backed by a scipy `cKDTree` built from the dataset's
    world-frame `(x, y)` coordinates. Each query returns the values of the
    nearest sampled row. Columns are given by integer position (CSV without
    header) or all by header name (CSV with a header row); None = not loaded.
    """

    def __init__(
        self,
        csv_file: str,
        x_col: Column = 0,
        y_col: Column = 1,
        ukc_col: Column = 6,
        wind_speed_col: Column = 7,
        wind_dir_col: Column = 8,
        current_speed_col: Column = 9,
        current_dir_col: Column = 10,
        heading_col: Column = None,
        surge_col: Column = None,
        sway_col: Column = None,
    ):
        """
        Load the requested columns and build the nearest-neighbor search tree.

        Rows with NaN/inf in the coordinates or in any loaded value are
        discarded before the tree is built.

        Raises:
            ValueError: If x/y/ukc are not given, the columns mix integer
                positions and names, the CSV is too narrow, or no valid rows
                remain.
        """

        if x_col is None or y_col is None or ukc_col is None:
            raise ValueError("WcuLookup needs x_col, y_col and ukc_col.")
        wanted: Dict[str, Column] = {
            "x": x_col, "y": y_col, "ukc": ukc_col,
            "wind_speed": wind_speed_col, "wind_direction": wind_dir_col,
            "current_speed": current_speed_col, "current_direction": current_dir_col,
            "heading": heading_col, "surge": surge_col, "sway": sway_col,
        }
        wanted = {k: c for k, c in wanted.items() if c is not None}
        kinds = {isinstance(c, str) for c in wanted.values()}
        if len(kinds) > 1:
            raise ValueError("WcuLookup: give all columns as integer positions or all as header names.")
        by_name = kinds == {True}

        if by_name:
            df = pd.read_csv(csv_file, usecols=sorted(set(wanted.values())))
        else:
            n_cols = len(pd.read_csv(csv_file, header=None, nrows=1).columns)
            required = max(wanted.values()) + 1
            if n_cols < required:
                raise ValueError(f"CSV '{csv_file}' has {n_cols} columns; need at least {required}.")
            df = pd.read_csv(csv_file, header=None, usecols=sorted(set(wanted.values())),
                             dtype=np.float64)

        data = {k: df[c].to_numpy(dtype=np.float64) for k, c in wanted.items()}
        valid = np.ones(len(df), dtype=bool)
        for arr in data.values():
            valid &= np.isfinite(arr)
        n_dropped = int((~valid).sum())
        if n_dropped:
            logger.warning("WcuLookup: dropped %d row(s) with NaN/inf values in '%s'.", n_dropped, csv_file)
        if not valid.any():
            raise ValueError(f"WcuLookup: no valid rows found in '{csv_file}'.")

        self.points = np.column_stack([data["x"][valid], data["y"][valid]])
        # Loaded fields as float32 arrays; fields that were not requested are None.
        self._values: Dict[str, Optional[np.ndarray]] = {
            name: (data[name][valid].astype(np.float32) if name in data else None)
            for name in _ENV_FIELDS + _STATE_FIELDS
        }
        self.tree = cKDTree(self.points)

    # Read-only access to the loaded columns (None if not loaded).
    @property
    def ukc(self) -> np.ndarray:
        return self._values["ukc"]

    @property
    def wind_speed(self) -> Optional[np.ndarray]:
        return self._values["wind_speed"]

    @property
    def wind_direction(self) -> Optional[np.ndarray]:
        return self._values["wind_direction"]

    @property
    def current_speed(self) -> Optional[np.ndarray]:
        return self._values["current_speed"]

    @property
    def current_direction(self) -> Optional[np.ndarray]:
        return self._values["current_direction"]

    def has(self, field: str) -> bool:
        """True if column `field` (e.g. "wind_speed", "heading") was loaded."""

        return self._values.get(field) is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _query(self, x: float, y: float, max_distance: float | None) -> tuple[int, float]:
        """
        Nearest row for one query point: `(index, distance)`. The index equals
        `len(self)` when no row lies within `max_distance`.
        """

        kwargs = {}
        if max_distance is not None:
            kwargs["distance_upper_bound"] = max_distance
        distance, idx = self.tree.query([x, y], **kwargs)
        return int(idx), float(distance)

    def _query_batch(self, xy: np.ndarray, max_distance: float | None, workers: int):
        """Nearest rows for `(N, 2)` query points: `(indices, distances)`."""

        kwargs: dict = {"workers": workers}
        if max_distance is not None:
            kwargs["distance_upper_bound"] = max_distance
        distance, idx = self.tree.query(xy, **kwargs)
        return idx, distance

    def _value(self, name: str, idx: int) -> Optional[float]:
        arr = self._values[name]
        return None if arr is None else float(arr[idx])

    def _env_at(self, idx: int) -> EnvironmentData:
        if idx >= len(self):
            return EnvironmentData(None, None, None, None, None)
        return EnvironmentData(*(self._value(name, idx) for name in _ENV_FIELDS))

    # ------------------------------------------------------------------
    # Public API - single point
    # ------------------------------------------------------------------

    def get_env(self, x: float, y: float, max_distance: float | None = None,
                return_distance: bool = False) -> EnvironmentData | tuple[EnvironmentData, float]:
        """
        Environmental conditions of the dataset row nearest to `(x, y)`.

        Args:
            x, y: World-frame position [m].
            max_distance: Optional search radius [m]; no row within it -> all
                fields None (distance inf).
            return_distance: Also return the distance to the row.
        """

        idx, distance = self._query(x, y, max_distance)
        env = self._env_at(idx)
        return (env, distance) if return_distance else env

    def get_sample(self, x: float, y: float, max_distance: float | None = None) -> Optional[LookupSample]:
        """
        Nearest dataset row with its environment and the recording ship's
        heading/surge/sway (None for columns that were not loaded).

        Returns:
            The row, or None if no row lies within `max_distance`.
        """

        idx, distance = self._query(x, y, max_distance)
        if idx >= len(self):
            return None
        return LookupSample(env=self._env_at(idx), heading=self._value("heading", idx),
                            surge=self._value("surge", idx), sway=self._value("sway", idx),
                            distance=distance)

    def get_ukc(self, x: float, y: float, max_distance: float | None = None,
                return_distance: bool = False) -> float | None | tuple[float | None, float]:
        """
        Under-keel clearance [m] of the nearest row, or None if no row lies
        within `max_distance`.
        """

        idx, distance = self._query(x, y, max_distance)
        result = None if idx >= len(self) else float(self.ukc[idx])
        return (result, distance) if return_distance else result

    # ------------------------------------------------------------------
    # Public API - batch
    # ------------------------------------------------------------------

    def get_env_batch(self, xy: np.ndarray, max_distance: float | None = None, workers: int = -1) -> np.ndarray:
        """
        Environmental conditions for `(N, 2)` query points, as a structured
        float32 array with the EnvironmentData fields. Missing rows and
        columns that were not loaded are NaN.
        """

        xy = np.asarray(xy, dtype=np.float64)
        idx, _ = self._query_batch(xy, max_distance, workers)
        out = np.full(len(xy), np.nan, dtype=_ENV_DTYPE)
        valid = idx < len(self)
        vi = idx[valid]
        for name in _ENV_FIELDS:
            arr = self._values[name]
            if arr is not None:
                out[name][valid] = arr[vi]
        return out

    def get_ukc_batch(self, xy: np.ndarray, max_distance: float | None = None, workers: int = -1) -> np.ndarray:
        """Under-keel clearance for `(N, 2)` query points; NaN where no row is found."""

        xy = np.asarray(xy, dtype=np.float64)
        idx, _ = self._query_batch(xy, max_distance, workers)
        out = np.full(len(xy), np.nan, dtype=np.float32)
        valid = idx < len(self)
        out[valid] = self.ukc[idx[valid]]
        return out

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of valid dataset points in the lookup tree."""

        return len(self.points)

    def __repr__(self) -> str:
        loaded = [name for name, arr in self._values.items() if arr is not None]
        return f"WcuLookup(n_points={len(self.points)}, columns={loaded})"
