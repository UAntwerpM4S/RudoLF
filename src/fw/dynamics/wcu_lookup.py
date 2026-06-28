"""Nearest-neighbor lookup of Under Keel Clearance (UKC) and environmental
conditions (wind speed/direction, current speed/direction) from a CSV of
supervised-dataset points, backed by a scipy cKDTree.

CSV column layout (16 columns, matching collect_supervised_dataset output):
  X – inputs (13 columns)
    col  0: x                (m, world-frame position)
    col  1: y                (m, world-frame position)
    col  2: heading          (rad, normalized to [-pi, pi])
    col  3: surge            (m/s, body-frame longitudinal velocity)
    col  4: sway             (m/s, body-frame lateral velocity)
    col  5: yaw_rate         (rad/s)
    col  6: UKC              (m)
    col  7: wind speed       (m/s)
    col  8: wind direction   (rad)
    col  9: current speed    (m/s)
    col 10: current direction(rad)
    col 11: rudder_action    (normalized [-1, 1])
    col 12: thrust_action    (normalized [-1, 1])
  Y – one-step delta outputs (3 columns)
    col 13: delta_surge      (m/s)
    col 14: delta_sway       (m/s)
    col 15: delta_yaw_rate   (rad/s)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True, slots=True)
class EnvironmentData:
    """Environmental conditions returned from a nearest-neighbor lookup.

    All angular quantities are in radians. None values indicate that no
    point was found within the requested max_distance.
    """

    ukc: float | None               # m
    wind_speed: float | None        # m/s
    wind_direction: float | None    # rad
    current_speed: float | None     # m/s
    current_direction: float | None # rad


# Structured dtype used for batch results
_ENV_DTYPE = np.dtype([
    ("ukc",               np.float32),
    ("wind_speed",        np.float32),
    ("wind_direction",    np.float32),
    ("current_speed",     np.float32),
    ("current_direction", np.float32),
])


class WcuLookup:
    """Nearest-neighbor lookup for wind, current and UKC.

    The tree is built on (x, y) world-frame coordinates (cols 0 and 1).
    Querying returns the five environmental scalars stored at the nearest
    sampled point.

    Column indices default to the layout produced by
    `collect_supervised_dataset`, but each can be overridden by passing
    either an integer (positional) or a string (header name).

    Backward-compatible API
    -----------------------
    `get_ukc` / `get_ukc_batch` still exist and behave exactly as before.
    The new `get_env` / `get_env_batch` methods return all five fields as
    an :class:`EnvironmentData` instance or a structured NumPy array.
    """

    def __init__(
        self,
        csv_file: str,
        x_col: int | str = 0,
        y_col: int | str = 1,
        ukc_col: int | str = 6,
        wind_speed_col: int | str = 7,
        wind_dir_col: int | str = 8,
        current_speed_col: int | str = 9,
        current_dir_col: int | str = 10,
    ):
        df = pd.read_csv(csv_file, header=None)

        # Validate that we have enough columns for every positional index used.
        int_cols = [
            c for c in (
                x_col, y_col, ukc_col,
                wind_speed_col, wind_dir_col,
                current_speed_col, current_dir_col,
            )
            if isinstance(c, int)
        ]
        if int_cols:
            required = max(int_cols) + 1
            if len(df.columns) < required:
                raise ValueError(
                    f"CSV '{csv_file}' has {len(df.columns)} columns; "
                    f"need at least {required}."
                )

        def _select(col: int | str) -> pd.Series:
            return df.iloc[:, col] if isinstance(col, int) else df[col]

        x   = _select(x_col).to_numpy(dtype=np.float64)
        y   = _select(y_col).to_numpy(dtype=np.float64)
        ukc = _select(ukc_col).to_numpy(dtype=np.float32)
        ws  = _select(wind_speed_col).to_numpy(dtype=np.float32)
        wd  = _select(wind_dir_col).to_numpy(dtype=np.float32)
        cs  = _select(current_speed_col).to_numpy(dtype=np.float32)
        cd  = _select(current_dir_col).to_numpy(dtype=np.float32)

        points = np.column_stack([x, y])

        # Drop rows with NaN/inf in coordinates or any of the five env fields.
        coord_valid = np.isfinite(points).all(axis=1)
        env_valid   = (
            np.isfinite(ukc) & np.isfinite(ws) & np.isfinite(wd)
            & np.isfinite(cs) & np.isfinite(cd)
        )
        valid   = coord_valid & env_valid
        n_dropped = int((~valid).sum())
        if n_dropped:
            import warnings
            warnings.warn(
                f"UKCLookup: dropped {n_dropped} row(s) with NaN/inf "
                f"coordinates or environmental values in '{csv_file}'."
            )

        points = points[valid]
        ukc = ukc[valid]
        ws  = ws[valid]
        wd  = wd[valid]
        cs  = cs[valid]
        cd  = cd[valid]

        if len(points) == 0:
            raise ValueError(f"UKCLookup: no valid rows found in '{csv_file}'.")

        self.points = points
        self.ukc               = ukc
        self.wind_speed        = ws
        self.wind_direction    = wd
        self.current_speed     = cs
        self.current_direction = cd
        self.tree = cKDTree(self.points)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _query(
        self,
        x: float,
        y: float,
        max_distance: float | None,
    ) -> tuple[int, float]:
        """Return (idx, distance) for the nearest point to (x, y).

        idx is `len(self.ukc)` (i.e., out-of-range) when no point lies
        within *max_distance*.
        """
        kwargs = {}
        if max_distance is not None:
            kwargs["distance_upper_bound"] = max_distance
        distance, idx = self.tree.query([x, y], **kwargs)
        return int(idx), float(distance)

    def _query_batch(
        self,
        xy: np.ndarray,
        max_distance: float | None,
        workers: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (idx_array, distance_array) for an (N, 2) query array."""
        kwargs: dict = {"workers": workers}
        if max_distance is not None:
            kwargs["distance_upper_bound"] = max_distance
        distance, idx = self.tree.query(xy, **kwargs)
        return idx, distance

    # ------------------------------------------------------------------
    # Public API – single point
    # ------------------------------------------------------------------

    def get_env(
        self,
        x: float,
        y: float,
        max_distance: float | None = None,
        return_distance: bool = False,
    ) -> EnvironmentData | tuple[EnvironmentData, float]:
        """Look up all environmental conditions nearest to (x, y).

        Parameters
        ----------
        x, y:
            World-frame coordinates (m).
        max_distance:
            If given, only points within this radius are considered.
            All fields in the returned :class:`EnvironmentData` are `None`
            when no point qualifies.
        return_distance:
            When `True`, returns `(EnvironmentData, distance)` where
            *distance* is `inf` if no point was within *max_distance*.

        Returns
        -------
        :class:`EnvironmentData` (or a `(EnvironmentData, float)` tuple).
        """
        idx, distance = self._query(x, y, max_distance)
        if idx >= len(self.ukc):
            env = EnvironmentData(
                ukc=None,
                wind_speed=None,
                wind_direction=None,
                current_speed=None,
                current_direction=None,
            )
        else:
            env = EnvironmentData(
                ukc=float(self.ukc[idx]),
                wind_speed=float(self.wind_speed[idx]),
                wind_direction=float(self.wind_direction[idx]),
                current_speed=float(self.current_speed[idx]),
                current_direction=float(self.current_direction[idx]),
            )
        return (env, distance) if return_distance else env

    def get_ukc(
        self,
        x: float,
        y: float,
        max_distance: float | None = None,
        return_distance: bool = False,
    ) -> float | None | tuple[float | None, float]:
        """Look up UKC only (backward-compatible).

        See :meth:`get_env` for full environmental data.
        """
        idx, distance = self._query(x, y, max_distance)
        result = None if idx >= len(self.ukc) else float(self.ukc[idx])
        return (result, distance) if return_distance else result

    # ------------------------------------------------------------------
    # Public API – batch
    # ------------------------------------------------------------------

    def get_env_batch(
        self,
        xy: np.ndarray,
        max_distance: float | None = None,
        workers: int = -1,
    ) -> np.ndarray:
        """Vectorized lookup for many points at once.

        Parameters
        ----------
        xy:
            (N, 2) array-like of (x, y) pairs.
        max_distance:
            Points with no neighbor within this radius get NaN in all fields.
        workers:
            Passed to `cKDTree.query`; `-1` uses all CPU cores.

        Returns
        -------
        Structured NumPy array of shape (N,) with dtype fields
        `ukc`, `wind_speed`, `wind_direction`, `current_speed`,
        `current_direction` (all `float32`).  Missing values are NaN.
        """
        xy  = np.asarray(xy, dtype=np.float64)
        idx, _ = self._query_batch(xy, max_distance, workers)

        out = np.full(len(xy), np.nan, dtype=_ENV_DTYPE)
        valid = idx < len(self.ukc)
        vi   = idx[valid]
        out["ukc"][valid]               = self.ukc[vi]
        out["wind_speed"][valid]        = self.wind_speed[vi]
        out["wind_direction"][valid]    = self.wind_direction[vi]
        out["current_speed"][valid]     = self.current_speed[vi]
        out["current_direction"][valid] = self.current_direction[vi]
        return out

    def get_ukc_batch(
        self,
        xy: np.ndarray,
        max_distance: float | None = None,
        workers: int = -1,
    ) -> np.ndarray:
        """Vectorized UKC-only lookup (backward-compatible).

        Returns an (N,) float32 array with NaN where no point was found.
        """
        xy  = np.asarray(xy, dtype=np.float64)
        idx, _ = self._query_batch(xy, max_distance, workers)
        out = np.full(len(xy), np.nan, dtype=np.float32)
        valid = idx < len(self.ukc)
        out[valid] = self.ukc[idx[valid]]
        return out

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"UKCLookup(n_points={len(self.points)})"
