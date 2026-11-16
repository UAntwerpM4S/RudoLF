"""
Suggest WIND_OBS_SCALE and CURRENT_OBS_SCALE (fw.environments.wind_current_obs)
from full_env dataset or lookup CSVs.

The CSVs have the generator's raw full_env row (no header). For every row the
four observation features are computed with env_body_components, exactly as
the RL observation does; per quantity the given percentile of the larger
absolute component (|ahead|, |stbd|) is printed, plus the share of rows that
the suggested scale would clip. The same percentile of |yaw_rate| is printed
as well (a candidate scale for the observed yaw rate).

Usage (from src):

    python -m fw.surrogate.compute_wind_current_obs_scales \
        --csv data/tripoli_full/ship_dynamics_dataset.csv data/myzako_full/ship_dynamics_dataset.csv
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Sequence
from fw.surrogate.schema import FULL_ENV, TARGET_NAMES, env_body_components

_COLUMNS = ("heading", "yaw_rate", "wind_speed", "wind_direction", "current_speed", "current_direction")


def load_env_columns(csv_file: Path, chunk_rows: int = 1_000_000) -> Dict[str, np.ndarray]:
    """
    Read only the heading, yaw-rate and wind/current columns of a full_env CSV.

    Raises:
        ValueError: If the CSV does not have the full_env dataset layout
            (raw row + targets + episode_id).
    """

    expected = FULL_ENV.raw_input_dim + len(TARGET_NAMES) + 1
    n_cols = len(pd.read_csv(csv_file, header=None, nrows=1).columns)
    if n_cols != expected:
        raise ValueError(f"{csv_file}: {n_cols} columns, but a full_env dataset/lookup CSV has {expected} "
                         f"(raw row {FULL_ENV.raw_input_dim} + {len(TARGET_NAMES)} targets + episode_id).")
    usecols = [FULL_ENV.col(c) for c in _COLUMNS]
    parts = {c: [] for c in _COLUMNS}
    for chunk in pd.read_csv(csv_file, header=None, usecols=usecols, dtype=np.float64, chunksize=chunk_rows):
        for c in _COLUMNS:
            parts[c].append(chunk[FULL_ENV.col(c)].to_numpy())
    return {c: np.concatenate(v) for c, v in parts.items()}


def suggest_scales(columns: Dict[str, np.ndarray], percentile: float) -> Dict[str, Dict[str, float]]:
    """
    Per quantity ("wind", "current"): the percentile of max(|ahead|, |stbd|),
    the maximum, and the share of finite rows above that percentile.

    Raises:
        ValueError: If `percentile` is outside [0, 100] or a quantity has no finite rows.
    """

    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}.")

    wa, ws, ca, cs = env_body_components(columns["wind_speed"], columns["wind_direction"],
                                         columns["current_speed"], columns["current_direction"],
                                         columns["heading"])
    out = {}
    for name, a, b in (("wind", wa, ws), ("current", ca, cs)):
        m = np.maximum(np.abs(a), np.abs(b))
        m = m[np.isfinite(m)]
        if m.size == 0:
            raise ValueError(f"No finite {name} rows in the given CSV(s).")
        p = float(np.percentile(m, percentile))
        out[name] = {"percentile": p, "max": float(m.max()), "clipped_share": float(np.mean(m > p)),
                     "rows": int(m.size)}
    return out


def main(argv: Sequence[str] = None) -> int:
    """
    Command-line entry point; prints the suggestions.
    """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, nargs="+", required=True, help="full_env dataset or lookup CSVs.")
    parser.add_argument("--percentile", type=float, default=99.9, help="Percentile used as scale (default 99.9).")
    args = parser.parse_args(argv)

    merged: Dict[str, list] = {c: [] for c in _COLUMNS}
    for csv_file in args.csv:
        cols = load_env_columns(csv_file)
        for c in _COLUMNS:
            merged[c].append(cols[c])
        print(f"{csv_file}: {len(cols['heading'])} rows")
    columns = {c: np.concatenate(v) for c, v in merged.items()}

    result = suggest_scales(columns, args.percentile)
    for name, r in result.items():
        print(f"{name}: p{args.percentile:g} of max(|ahead|, |stbd|) = {r['percentile']:.3f} m/s, "
              f"max = {r['max']:.3f} m/s, clipped at that scale: {100 * r['clipped_share']:.3f} % "
              f"of {r['rows']} rows")
    yaw = np.abs(columns["yaw_rate"])
    yaw = yaw[np.isfinite(yaw)]
    if yaw.size:
        print(f"yaw rate: p{args.percentile:g} of |yaw_rate| = {np.percentile(yaw, args.percentile):.4f} rad/s, "
              f"max = {yaw.max():.4f} rad/s, of {yaw.size} rows")
    print(f"Suggested: WIND_OBS_SCALE = {result['wind']['percentile']:.1f}, "
          f"CURRENT_OBS_SCALE = {result['current']['percentile']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
