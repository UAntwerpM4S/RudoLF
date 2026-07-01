"""
Nearest-neighbor lookup of Under Keel Clearance (UKC) and environmental
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

import numpy as np
import pandas as pd

from dataclasses import dataclass
from scipy.spatial import cKDTree


@dataclass(frozen=True, slots=True)
class EnvironmentData:
    """
    Container for environmental conditions returned by a nearest-neighbor lookup.

    All angular quantities are expressed in radians. A field is `None` when no
    dataset point satisfies the requested search constraints.

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


# Structured dtype used for batch results
_ENV_DTYPE = np.dtype([
    ("ukc",               np.float32),
    ("wind_speed",        np.float32),
    ("wind_direction",    np.float32),
    ("current_speed",     np.float32),
    ("current_direction", np.float32),
])


class WcuLookup:
    """
    Nearest-neighbor lookup for under-keel clearance and environmental data.

    The lookup is backed by a scipy `cKDTree` constructed from the dataset's
    world-frame `(x, y)` coordinates. Each query returns the environmental
    values associated with the nearest sampled point.

    By default, the column indices match the CSV layout produced by
    `collect_supervised_dataset`. Individual columns may be specified either by
    their integer position or by their header name.

    The legacy `get_ukc()` and `get_ukc_batch()` methods remain available for
    backward compatibility. The newer `get_env()` and `get_env_batch()`
    methods return all available environmental variables.
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
        """
        Load a dataset and build the nearest-neighbor search tree.

        Rows containing invalid coordinates or environmental values (NaN or infinity)
        are discarded before constructing the lookup structure.

        Args:
            csv_file: Path to the CSV dataset.
            x_col: Column containing the x-coordinate.
            y_col: Column containing the y-coordinate.
            ukc_col: Column containing under-keel clearance values.
            wind_speed_col: Column containing wind speed values.
            wind_dir_col: Column containing wind direction values.
            current_speed_col: Column containing current speed values.
            current_dir_col: Column containing current direction values.

        Raises:
            ValueError: If the CSV does not contain the required columns or no valid
                rows remain after filtering invalid data.
        """

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
            """
            Return a DataFrame column by index or column name.

            Args:
                col: Column identifier, specified either as a zero-based integer
                    index or as a column name.

            Returns:
                The selected pandas Series.
            """

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
        """
        Find the nearest dataset point for a single query location.

        Args:
            x: World-frame x-coordinate in meters.
            y: World-frame y-coordinate in meters.
            max_distance: Optional maximum search radius in meters.

        Returns:
            A tuple `(index, distance)` where `index` is the nearest dataset
            index and `distance` is the Euclidean distance in meters. If no point
            lies within `max_distance`, the returned index equals
            `len(self.ukc)`.
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
        """
        Perform nearest-neighbor lookup for multiple query points.

        Args:
            xy: Array of shape `(N, 2)` containing world-frame `(x, y)`
                coordinates.
            max_distance: Optional maximum search radius in meters.
            workers: Number of worker threads passed to `cKDTree.query()`.
                A value of `-1` uses all available CPU cores.

        Returns:
            A tuple `(indices, distances)` containing the nearest dataset index
            and Euclidean distance for each query point.
        """

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
        """
        Retrieve environmental conditions nearest to a world-frame position.

        Args:
            x: World-frame x-coordinate in meters.
            y: World-frame y-coordinate in meters.
            max_distance: Optional maximum search radius in meters. If no dataset
                point falls within this distance, all returned fields are `None`.
            return_distance: If `True`, also returns the distance to the nearest
                dataset point.

        Returns:
            An `EnvironmentData` instance. If `return_distance` is `True`,
            returns a tuple `(environment, distance)` where `distance` is
            `float('inf')` when no matching point is found.
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
        """
        Retrieve only the under-keel clearance for a query location.

        This method is retained for backward compatibility. Use `get_env()` to
        retrieve the complete set of environmental conditions.

        Args:
            x: World-frame x-coordinate in meters.
            y: World-frame y-coordinate in meters.
            max_distance: Optional maximum search radius in meters.
            return_distance: If `True`, also returns the lookup distance.

        Returns:
            The under-keel clearance in meters, or `None` if no dataset point lies
            within `max_distance`. If `return_distance` is `True`, returns a
            tuple `(ukc, distance)`.
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
        """
        Retrieve environmental conditions for multiple query locations.

        Args:
            xy: Array-like object of shape `(N, 2)` containing world-frame
                `(x, y)` coordinates.
            max_distance: Optional maximum search radius in meters. Query points
                without a nearby dataset point receive `NaN` values.
            workers: Number of worker threads passed to `cKDTree.query()`.
                A value of `-1` uses all available CPU cores.

        Returns:
            A structured NumPy array of shape `(N,)` containing the fields
            `ukc`, `wind_speed`, `wind_direction`, `current_speed`, and
            `current_direction`. All fields have dtype `float32`. Missing values
            are represented by `NaN`.
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
        """
        Retrieve under-keel clearance for multiple query locations.

        This method is retained for backward compatibility.

        Args:
            xy: Array-like object of shape `(N, 2)` containing world-frame
                `(x, y)` coordinates.
            max_distance: Optional maximum search radius in meters.
            workers: Number of worker threads passed to `cKDTree.query()`.
                A value of `-1` uses all available CPU cores.

        Returns:
            A one-dimensional `float32` NumPy array containing the under-keel
            clearance for each query point. Entries corresponding to unmatched
            locations are `NaN`.
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
        """
        Return the number of valid dataset points available for lookup.

        Returns:
            The number of points stored in the lookup tree.
        """

        return len(self.points)

    def __repr__(self) -> str:
        """
        Return a developer-friendly string representation of the lookup object.

        Returns:
            A string containing the class name and the number of indexed dataset
            points.
        """

        return f"UKCLookup(n_points={len(self.points)})"
