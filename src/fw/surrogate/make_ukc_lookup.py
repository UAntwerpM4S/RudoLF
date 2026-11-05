"""
Build a small UKC lookup CSV from a (large) confined_ukc dataset CSV.

The UAntwerp surrogate (fw.dynamics.uantwerp_surrogate) looks up the UKC at a
position with a nearest-neighbor search (fw.dynamics.wcu_lookup.WcuLookup).
The dataset CSV has one row per simulation step (millions of rows), but
WcuLookup only needs a few columns. This tool bins positions on a square grid
and keeps one row per occupied cell, in the same column layout as the dataset
CSV (raw row, 3 targets, episode_id; other columns 0), so it is a drop-in
replacement for the preset's lookup_csv.

- confined_ukc: UKC comes from the static tide and depends on position, so a
  cell stores its mean x, y (stays in the water) and median UKC.
- full_env: tide, wind and current vary in time, so a median would mix moments
  (e.g. flood and ebb cancelling out). Each cell keeps one real row (closest to
  the cell's mean position), so UKC, wind, current and the recording ship's
  heading/surge/sway (needed to convert the recorded apparent wind) belong to
  the same moment. Only ENV_LOOKUP_COLUMNS are kept.

An --out name ending in .gz gives a gzip file that WcuLookup reads as is (a 5 m
grid of the river is tens of MB as plain CSV, a few MB compressed; GitHub
refuses files over 100 MB).

Before writing, whole episodes (default 10 %) are held out and the UKC at their
positions is compared with the full nearest-neighbor lookup and each grid size.
The within-cell spread shows how much UKC is not explained by position.

Usage (from src):

    python -m fw.surrogate.make_ukc_lookup --csv <dataset.csv> --cell 5 \
        --compare 2 5 10 20 --out fw/dynamics/ukc_lookup_myzako_confined.csv.gz

    python -m fw.surrogate.make_ukc_lookup --csv data/myzako_full/ship_dynamics_dataset.csv \
        --cell 5 --out fw/dynamics/env_lookup_myzako_full.csv.gz
"""

from __future__ import annotations

import sys
import time
import gzip
import logging
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.spatial import cKDTree
from typing import Dict, Optional, Sequence, Tuple
from fw.surrogate.schema import TARGET_NAMES, ScenarioSchema, scenario_for_raw_width

logger = logging.getLogger(__name__)

# Columns a full_env lookup keeps (what fw.dynamics.uantwerp_surrogate's WcuLookup reads);
# all other columns of the written rows are set to 0.
ENV_LOOKUP_COLUMNS = ("x", "y", "heading", "surge", "sway", "ukc",
                      "wind_speed", "wind_direction", "current_speed", "current_direction")

# Columns after the raw row in a dataset CSV: the targets, then episode_id.
_TRAILING_COLS = len(TARGET_NAMES) + 1


def detect_scenario(csv_file: Path) -> Tuple[ScenarioSchema, int]:
    """
    Scenario of a dataset CSV (from its width) and its number of columns.

    Raises:
        ValueError: If the width matches no scenario.
    """

    n_cols = len(pd.read_csv(csv_file, header=None, nrows=1).columns)
    return scenario_for_raw_width(n_cols - _TRAILING_COLS), n_cols


def load_columns(csv_file: Path, scenario: ScenarioSchema, n_cols: int,
                 required: Sequence[str] = ()) -> Dict[str, np.ndarray]:
    """
    x, y, UKC and episode_id (plus the raw columns in `required`) of every
    row in which all of them are finite, and each such row's 0-based row
    index in the CSV (`"row"`).

    Uses the pyarrow CSV engine when installed (much faster), else pandas' C engine.
    """

    cols = {"x": scenario.col("x"), "y": scenario.col("y"), "ukc": scenario.col("ukc"),
            "episode_id": n_cols - 1}
    for name in required:
        cols.setdefault(name, scenario.col(name))
    usecols = sorted(set(cols.values()))
    kwargs = dict(header=None, usecols=usecols)
    try:
        df = pd.read_csv(csv_file, engine="pyarrow", **kwargs)
    except (ImportError, ValueError):
        df = pd.read_csv(csv_file, dtype=np.float64, **kwargs)
    # pyarrow may renumber the selected columns 0..n-1: relabel by file position.
    df.columns = usecols
    data = {k: df[c].to_numpy(dtype=np.float64) for k, c in cols.items()}
    data["row"] = np.arange(len(df), dtype=np.int64)     # 0-based row index in the CSV
    valid = np.ones(len(df), dtype=bool)
    for arr in data.values():
        valid &= np.isfinite(arr)
    if (~valid).any():
        logger.warning("Dropped %d row(s) with NaN/inf values.", int((~valid).sum()))
    if not valid.any():
        raise ValueError(f"No valid rows in '{csv_file}'.")
    return {k: v[valid] for k, v in data.items()}


def grid_reduce(x: np.ndarray, y: np.ndarray, ukc: np.ndarray, cell: float):
    """
    One point per occupied `cell` x `cell` [m] square.

    Returns:
        (cx, cy, cukc, counts, spread): mean x/y and median UKC per cell, the
        number of rows per cell, and |UKC - cell median| per input row.
    """

    if not cell > 0:
        raise ValueError(f"Cell size must be > 0, got {cell}.")
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    _, inv = np.unique(np.stack([ix, iy], axis=1), axis=0, return_inverse=True)
    inv = inv.ravel()
    counts = np.bincount(inv)
    cx = np.bincount(inv, weights=x) / counts
    cy = np.bincount(inv, weights=y) / counts
    order = np.lexsort((ukc, inv))                  # by cell, then by UKC
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    sorted_ukc = ukc[order]
    cukc = 0.5 * (sorted_ukc[starts + (counts - 1) // 2] + sorted_ukc[starts + counts // 2])
    spread = np.abs(ukc - cukc[inv])
    return cx, cy, cukc, counts, spread


def grid_representative(x: np.ndarray, y: np.ndarray, cell: float) -> np.ndarray:
    """
    Index of one representative row per occupied `cell` x `cell` [m] square:
    the row closest to the mean position of its cell (sorted by row index).
    """

    if not cell > 0:
        raise ValueError(f"Cell size must be > 0, got {cell}.")
    ix = np.floor(x / cell).astype(np.int64)
    iy = np.floor(y / cell).astype(np.int64)
    _, inv = np.unique(np.stack([ix, iy], axis=1), axis=0, return_inverse=True)
    inv = inv.ravel()
    counts = np.bincount(inv)
    cx = np.bincount(inv, weights=x) / counts
    cy = np.bincount(inv, weights=y) / counts
    d2 = (x - cx[inv]) ** 2 + (y - cy[inv]) ** 2
    order = np.lexsort((d2, inv))                   # by cell, then by distance to the cell mean
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return np.sort(order[starts])


def read_rows(csv_file: Path, rows: np.ndarray, chunk_rows: int = 1_000_000) -> np.ndarray:
    """
    The CSV rows with the given (sorted, 0-based) indices, all columns, read in chunks.
    """

    parts, start = [], 0
    for chunk in pd.read_csv(csv_file, header=None, chunksize=chunk_rows, dtype=np.float64):
        stop = start + len(chunk)
        lo, hi = np.searchsorted(rows, [start, stop])
        if hi > lo:
            parts.append(chunk.to_numpy()[rows[lo:hi] - start])
        start = stop
    out = np.concatenate(parts) if parts else np.zeros((0, 0))
    if len(out) != len(rows):
        raise ValueError(f"'{csv_file}': read {len(out)} of {len(rows)} requested rows (file changed?).")
    return out


def error_stats(err: np.ndarray) -> Dict[str, float]:
    """
    MAE, RMSE, 95th/99th percentile and maximum of |err| [m].
    """

    a = np.abs(err)
    return {"mae": float(a.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def _fmt(stats: Dict[str, float]) -> str:
    """
    One table row of error_stats [m].
    """

    return "  ".join(f"{stats[k]:8.3f}" for k in ("mae", "rmse", "p95", "p99", "max"))


def evaluate(data: Dict[str, np.ndarray], cells: Sequence[float], holdout_frac: float, seed: int,
             representative: bool = False) -> None:
    """
    Log the UKC error at held-out episodes' positions for the full
    nearest-neighbor lookup and the grid lookup per cell size.
    """

    episodes = np.unique(data["episode_id"])
    n_hold = int(round(holdout_frac * len(episodes)))
    if n_hold < 1 or n_hold >= len(episodes):
        logger.warning("Too few episodes (%d) for a %.0f %% hold-out: accuracy report skipped.",
                       len(episodes), 100 * holdout_frac)
        return
    rng = np.random.default_rng(seed)
    held = np.isin(data["episode_id"], rng.choice(episodes, size=n_hold, replace=False))
    tr = ~held
    xy_tr = np.column_stack([data["x"][tr], data["y"][tr]])
    xy_ho = np.column_stack([data["x"][held], data["y"][held]])
    truth = data["ukc"][held]
    ukc_tr = data["ukc"][tr]

    logger.info("Accuracy on %d held-out episodes (%d rows); lookups built from the other %d episodes "
                "(%d rows). |UKC error| vs the UKC FH recorded there [m]:",
                n_hold, int(held.sum()), len(episodes) - n_hold, int(tr.sum()))
    logger.info("  %-26s %8s  %8s  %8s  %8s  %8s  %10s", "lookup", "MAE", "RMSE", "p95", "p99", "max", "points")

    t0 = time.perf_counter()
    dist, idx = cKDTree(xy_tr).query(xy_ho, workers=-1)
    full_pred = ukc_tr[idx]
    logger.info("  %-26s %s  %10d   (%.1f s)", "full nearest-neighbour", _fmt(error_stats(full_pred - truth)),
                len(xy_tr), time.perf_counter() - t0)

    for cell in cells:
        cx, cy, cukc, _, spread = grid_reduce(xy_tr[:, 0], xy_tr[:, 1], ukc_tr, cell)
        if representative:      # full_env: one real row per cell instead of mean position / median UKC
            rep_idx = grid_representative(xy_tr[:, 0], xy_tr[:, 1], cell)
            cx, cy, cukc = xy_tr[rep_idx, 0], xy_tr[rep_idx, 1], ukc_tr[rep_idx]
        _, gidx = cKDTree(np.column_stack([cx, cy])).query(xy_ho, workers=-1)
        grid_pred = cukc[gidx]
        logger.info("  %-26s %s  %10d   (vs full: MAE %.3f, max %.3f; within-cell spread MAE %.3f%s)",
                    f"grid {cell:g} m", _fmt(error_stats(grid_pred - truth)), len(cx),
                    float(np.abs(grid_pred - full_pred).mean()), float(np.abs(grid_pred - full_pred).max()),
                    float(spread.mean()), " around the cell median" if representative else "")
    logger.info("  Held-out positions: median distance to the nearest training row %.2f m, p99 %.2f m, max %.1f m.",
                float(np.median(dist)), float(np.percentile(dist, 99)), float(dist.max()))


def _savetxt(out: Path, arr: np.ndarray, fmt) -> None:
    """
    np.savetxt to `out`, gzip-compressed if the name ends in .gz. The gzip
    header is fixed (mtime 0, no name), so equal content gives equal bytes.
    """

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".gz":
        with open(out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            np.savetxt(gz, arr, fmt=fmt, delimiter=",")
    else:
        np.savetxt(out, arr, fmt=fmt, delimiter=",")


def _stem(csv_file: Path) -> str:
    """
    File name without .csv / .csv.gz (for the default output name).
    """

    name = csv_file.name
    for ext in (".csv.gz", ".gz", ".csv"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return csv_file.stem


def write_lookup(out: Path, scenario: ScenarioSchema, n_cols: int, cx, cy, cukc) -> None:
    """
    Write the grid points in the dataset CSV layout (other columns 0, episode_id -1).
    """

    arr = np.zeros((len(cx), n_cols), dtype=np.float64)
    arr[:, scenario.col("x")] = cx
    arr[:, scenario.col("y")] = cy
    arr[:, scenario.col("ukc")] = cukc
    arr[:, -1] = -1
    fmt = ["%d"] * n_cols
    fmt[scenario.col("x")] = fmt[scenario.col("y")] = "%.3f"
    fmt[scenario.col("ukc")] = "%.4f"
    _savetxt(out, arr, fmt)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Command-line entry point; returns the exit code.
    """

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="confined_ukc or full_env dataset CSV (no header).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV; a name ending in .gz is written gzip-compressed (pandas and WcuLookup "
                        "read it directly). Default: <csv stem>_ukc_grid<cell>m.csv.gz / _env_grid<cell>m.csv.gz "
                        "next to --csv.")
    p.add_argument("--cell", type=float, default=5.0, help="Grid cell size of the written lookup [m] (default 5).")
    p.add_argument("--compare", type=float, nargs="*", default=None,
                   help="Cell sizes [m] to compare in the accuracy report (default: just --cell).")
    p.add_argument("--holdout-frac", type=float, default=0.1,
                   help="Fraction of episodes held out for the accuracy report (default 0.1; 0 = no report).")
    p.add_argument("--seed", type=int, default=0, help="Seed of the hold-out selection (default 0).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scenario, n_cols = detect_scenario(args.csv)
    if not scenario.includes_ukc:
        logger.error("'%s' is a %s CSV: it has no UKC column.", args.csv, scenario.name)
        return 2
    representative = scenario.includes_wind_current

    t0 = time.perf_counter()
    # full_env: representative rows need finite wind/current/heading/speeds, or
    # WcuLookup drops them and their cell has no lookup row.
    data = load_columns(args.csv, scenario, n_cols, ENV_LOOKUP_COLUMNS if representative else ())
    logger.info("Read %d rows (%d episodes) of %s from '%s' in %.1f s.", len(data["x"]),
                len(np.unique(data["episode_id"])), scenario.name, args.csv, time.perf_counter() - t0)
    u = data["ukc"]
    logger.info("UKC: min %.2f, median %.2f, max %.2f m.", float(u.min()), float(np.median(u)), float(u.max()))

    if args.holdout_frac > 0:
        evaluate(data, args.compare if args.compare else [args.cell], args.holdout_frac, args.seed,
                 representative)
        if representative:
            logger.info("  (full_env: the UKC of every lookup varies with the tide of the recorded episode; "
                        "wind and current are not evaluated -- the grid keeps real rows, like the full CSV.)")

    if representative:
        t0 = time.perf_counter()
        rep_idx = grid_representative(data["x"], data["y"], args.cell)
        rows = read_rows(args.csv, data["row"][rep_idx])
        out = args.out or args.csv.with_name(f"{_stem(args.csv)}_env_grid{args.cell:g}m.csv.gz")
        # Keep only what the lookup reads (same layout, other columns 0: much smaller file).
        keep = [scenario.col(c) for c in ENV_LOOKUP_COLUMNS]
        unused = np.ones(rows.shape[1], dtype=bool)
        unused[keep] = False
        unused[-1] = False                               # episode_id
        rows[:, unused] = 0.0
        fmt = ["%d"] * rows.shape[1]
        for c in ENV_LOOKUP_COLUMNS:
            fmt[scenario.col(c)] = "%.3f" if c in ("x", "y") else "%.6g"
        _savetxt(out, rows, fmt)                          # gzip-compressed if `out` ends in .gz
        logger.info("Grid %g m (all rows): %d real rows kept, one per occupied cell (closest to the cell's "
                    "mean position); columns %s kept, the others set to 0 (%.1f s).", args.cell, len(rows),
                    ", ".join(ENV_LOOKUP_COLUMNS), time.perf_counter() - t0)
        logger.info("Wrote '%s' (%.1f MB; input %.1f MB).", out, out.stat().st_size / 1e6,
                    args.csv.stat().st_size / 1e6)
        return 0

    cx, cy, cukc, counts, spread = grid_reduce(data["x"], data["y"], data["ukc"], args.cell)
    out = args.out or args.csv.with_name(f"{_stem(args.csv)}_ukc_grid{args.cell:g}m.csv.gz")
    write_lookup(out, scenario, n_cols, cx, cy, cukc)
    logger.info("Grid %g m (all rows): %d points (median %d rows/cell); within-cell |UKC - median|: "
                "mean %.3f, p95 %.3f, max %.3f m.", args.cell, len(cx), int(np.median(counts)),
                float(spread.mean()), float(np.percentile(spread, 95)), float(spread.max()))
    logger.info("Wrote '%s' (%.1f MB; input %.1f MB).", out, out.stat().st_size / 1e6,
                args.csv.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
