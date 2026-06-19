"""Nearest-neighbor lookup of Under Keel Clearance (UKC) values from a CSV
of surveyed/sampled points, backed by a scipy cKDTree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree


class UKCLookup:
    """Nearest-neighbor UKC lookup.

    Expects a CSV with at least 7 columns where:
      - column 0: x coordinate
      - column 1: y coordinate
      - column 6: UKC value

    Coordinates and UKC are looked up positionally by default (matching the
    original implementation), but can be selected by name via
    `x_col` / `y_col` / `ukc_col` if the CSV has a header you trust more
    than column order.
    """

    def __init__(
        self,
        csv_file: str,
        x_col: int | str = 0,
        y_col: int | str = 1,
        ukc_col: int | str = 6,
    ):
        df = pd.read_csv(csv_file)

        min_cols = max(
            (c for c in (x_col, y_col, ukc_col) if isinstance(c, int)),
            default=-1,
        )
        if len(df.columns) <= min_cols:
            raise ValueError(
                f"CSV '{csv_file}' has {len(df.columns)} columns; "
                f"need at least {min_cols + 1}."
            )

        def _select(col: int | str) -> pd.Series:
            return df.iloc[:, col] if isinstance(col, int) else df[col]

        x = _select(x_col)
        y = _select(y_col)
        ukc = _select(ukc_col)

        points = np.column_stack(
            [x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64)]
        )
        ukc = ukc.to_numpy(dtype=np.float32)

        # Drop rows with NaN coordinates or NaN UKC rather than silently
        # returning NaN later. Warn so a bad input file doesn't fail silently.
        valid = np.isfinite(points).all(axis=1) & np.isfinite(ukc)
        n_dropped = int((~valid).sum())
        if n_dropped:
            import warnings

            warnings.warn(
                f"UKCLookup: dropped {n_dropped} row(s) with NaN/inf "
                f"coordinates or UKC value in '{csv_file}'."
            )
        points = points[valid]
        ukc = ukc[valid]

        if len(points) == 0:
            raise ValueError(f"UKCLookup: no valid rows found in '{csv_file}'.")

        self.points = points
        self.ukc = ukc
        self.tree = cKDTree(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def get_ukc(
        self,
        x: float,
        y: float,
        max_distance: float | None = None,
        return_distance: bool = False,
    ) -> float | None | tuple[float | None, float]:
        """Look up the UKC nearest to (x, y).

        Returns None if no point exists within `max_distance` (when given).
        If `return_distance` is True, returns (ukc_or_None, distance) instead,
        where distance is the distance to the nearest point found (inf if
        none was within `max_distance`).
        """
        kwargs = {}
        if max_distance is not None:
            kwargs["distance_upper_bound"] = max_distance

        distance, idx = self.tree.query([x, y], **kwargs)

        # scipy returns idx == n (out of range) and distance == inf when no
        # point is found within distance_upper_bound.
        if idx >= len(self.ukc):
            result = None
        else:
            result = float(self.ukc[idx])

        return (result, float(distance)) if return_distance else result

    def get_ukc_batch(
        self,
        xy: np.ndarray,
        max_distance: float | None = None,
        workers: int = -1,
    ) -> np.ndarray:
        """Vectorized lookup for many points at once.

        `xy` should be an (N, 2) array-like of (x, y) pairs.
        Returns an (N,) float array of UKC values, with NaN where no point
        was found within `max_distance`.
        """
        xy = np.asarray(xy, dtype=np.float64)
        kwargs = {"workers": workers}
        if max_distance is not None:
            kwargs["distance_upper_bound"] = max_distance

        distance, idx = self.tree.query(xy, **kwargs)

        out = np.full(len(xy), np.nan, dtype=np.float32)
        valid = idx < len(self.ukc)
        out[valid] = self.ukc[idx[valid]]
        return out

    def __repr__(self) -> str:
        return f"UKCLookup(n_points={len(self.points)})"
