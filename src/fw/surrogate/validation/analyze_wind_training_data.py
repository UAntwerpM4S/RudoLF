"""
Wind coverage of a wind/current training CSV (41 columns, sim_fh_dataset_generator
layout), over ALL rows.

Background: FH's WindState updates at 1 Hz, so the stored wind (col 7/8) is held
constant for ~10 consecutive 0.1 s rows. Bit-identical wind within an episode is
therefore normal and NOT a sign of stale data -- the old "stale rows" section was
removed for that reason. Episode-start staleness (the first second after the
settle still showing the exercise default) cannot be detected reliably from this
CSV: the generator shuffles rows, and the default's apparent value depends on each
episode's start heading and speed. The generator fix (WIND_WARMUP_TIME) prevents it
in regenerated datasets.

Reports:
  1. 1 Hz hold check: rows per (episode, wind value) group.
  2. Port vs starboard balance, apparent (as stored, col 7/8, rel. bow, + = from
     STARBOARD, i.e. clockwise -- FH's WindState convention, determined from
     turning episodes) and true (reconstructed as apparent - (u, -v), cols 3/4,
     in (ahead, starboard) components; sway v is PORT positive, as the
     generator stores it since 2026-10-19 -- older CSVs have starboard-positive
     sway and need the opposite sign).
  3. True-wind speed percentiles and the share of rows / episodes at low wind --
     relevant for how well "no wind" is covered.
  4. True-wind direction histogram.
  5. Current speed coverage (share of rows / episodes at low or zero current)
     and current direction relative to the heading.

Writes a PNG next to the CSV. Needs numpy, pandas, matplotlib only.

Usage:
    python analyze_wind_training_data.py <dataset.csv>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COL_HEADING, COL_SURGE, COL_SWAY, COL_WIND_SPEED, COL_WIND_DIR = 2, 3, 4, 7, 8
COL_CUR_SPEED, COL_CUR_DIR, COL_EPISODE = 9, 10, 40
DIR_BINS = np.arange(-180.0, 180.0 + 15.0, 15.0)     # deg, positive = from starboard
SPEED_BINS = np.arange(0.0, 41.0, 1.0)               # m/s
LOW_WIND_LIMITS = (0.5, 1.0, 2.0, 3.0, 5.0)          # m/s, true wind
LOW_CURRENT_LIMITS = (0.01, 0.05, 0.1, 0.2, 0.5)     # m/s


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()
    path = Path(args.csv)

    cols = [COL_HEADING, COL_SURGE, COL_SWAY, COL_WIND_SPEED, COL_WIND_DIR, COL_CUR_SPEED, COL_CUR_DIR, COL_EPISODE]
    print(f"Reading {path} ...", flush=True)
    df = pd.read_csv(path, header=None, usecols=cols, dtype=np.float64)
    df.columns = ["hdg", "u", "v", "ws", "wd", "cs", "cd", "ep"]
    df["ep"] = np.rint(df["ep"].to_numpy()).astype(np.int64)
    n_rows = len(df)
    n_eps = int(df["ep"].nunique())
    rows_per_ep = df.groupby("ep").size()

    print(f"\nFile: {path}")
    print(f"Rows: {n_rows:,}   episodes: {n_eps:,}   rows/episode: {n_rows / max(n_eps, 1):.1f} "
          f"(min {rows_per_ep.min()}, max {rows_per_ep.max()})")

    # 1. 1 Hz hold
    hold = df.groupby(["ep", "ws", "wd"], sort=False).size()
    print("\n== 1. 1 Hz WindState hold (rows per identical wind value within an episode) ==")
    print(f"Groups: {len(hold):,}   rows/group: mean {hold.mean():.2f}, median {hold.median():.0f}, "
          f"p95 {hold.quantile(0.95):.0f}, max {hold.max()}   (expected ~10 at 10 Hz rows)")

    u, v, ws, wd = (df[c].to_numpy() for c in ("u", "v", "ws", "wd"))
    app_deg = _wrap_deg(np.degrees(wd))
    tx = ws * np.cos(wd) - u                  # true = apparent - (u, -v), (ahead, starboard) components
    ty = ws * np.sin(wd) + v                  # sway v port positive
    true_speed = np.hypot(tx, ty)
    true_deg = _wrap_deg(np.degrees(np.arctan2(ty, tx)))

    # 2. balance
    print("\n== 2. Starboard (>0 deg) vs port (<0 deg), all rows ==")
    for label, d, s in (("apparent", app_deg, ws), ("true (reconstructed)", true_deg, true_speed)):
        port = d < 0.0
        p, q = int(port.sum()), int((~port).sum())
        print(f"{label:22s}: port {p:>12,} rows (mean {s[port].mean() if p else float('nan'):5.2f} m/s) | "
              f"starboard {q:>12,} rows (mean {s[~port].mean() if q else float('nan'):5.2f} m/s) | "
              f"port/stbd ratio {p / max(q, 1):.2f}")

    # 3. speed / low wind
    print("\n== 3. True-wind speed, all rows ==")
    print("Percentiles: " + ", ".join(f"p{q}={np.percentile(true_speed, q):.2f}"
                                      for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)) + " m/s")
    ep_median = pd.Series(true_speed).groupby(df["ep"].to_numpy()).median()
    print("Share below limit (episodes counted by their median true wind):")
    for lim in LOW_WIND_LIMITS:
        print(f"  < {lim:3g} m/s: {100.0 * np.mean(true_speed < lim):6.2f}% of rows, "
              f"{100.0 * np.mean(ep_median < lim):6.2f}% of episodes")
    print("Episode-median true wind percentiles: " + ", ".join(
        f"p{q}={np.percentile(ep_median, q):.2f}" for q in (1, 5, 10, 50, 90, 99)) + " m/s")

    # 4. direction histogram
    print("\n== 4. True-wind direction histogram, all rows, 15-deg bins, positive = from starboard ==")
    h = np.histogram(true_deg, bins=DIR_BINS)[0]
    tot = max(h.sum(), 1)
    for lo, hi, c in zip(DIR_BINS[:-1], DIR_BINS[1:], h):
        print(f"  [{lo:+6.0f},{hi:+6.0f}) {c:>12,}  {100.0 * c / tot:5.2f}%  {'#' * int(round(200.0 * c / tot))}")

    # 5. current
    cs = df["cs"].to_numpy()
    ep_cur = pd.Series(cs).groupby(df["ep"].to_numpy()).median()
    print("\n== 5. Current (col 9 speed, col 10 world direction), all rows ==")
    print("Speed percentiles: " + ", ".join(f"p{q}={np.percentile(cs, q):.3f}"
                                            for q in (0, 1, 5, 10, 25, 50, 75, 90, 99, 100)) + " m/s")
    print("Share below limit (episodes counted by their median current):")
    for lim in LOW_CURRENT_LIMITS:
        print(f"  < {lim:4g} m/s: {100.0 * np.mean(cs < lim):6.2f}% of rows, "
              f"{100.0 * np.mean(ep_cur < lim):6.2f}% of episodes")
    rel = _wrap_deg(np.degrees(df["cd"].to_numpy() - df["hdg"].to_numpy()))
    print("Current direction relative to heading (deg, 0 = flowing ahead, + = towards port):")
    h = np.histogram(rel, bins=np.arange(-180.0, 181.0, 45.0))[0]
    print("  " + "  ".join(f"[{lo:+4.0f},{lo + 45:+4.0f}): {100.0 * c / max(h.sum(), 1):5.1f}%"
                           for lo, c in zip(np.arange(-180.0, 180.0, 45.0), h)))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available -- no PNG.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    centers = 0.5 * (DIR_BINS[:-1] + DIR_BINS[1:])
    scenters = 0.5 * (SPEED_BINS[:-1] + SPEED_BINS[1:])
    for ax, (name, d) in zip(axes[0], (("Apparent", app_deg), ("True (reconstructed)", true_deg))):
        ax.bar(centers, np.histogram(d, bins=DIR_BINS)[0], width=13, color="#2a78d6")
        ax.set_title(f"{name} wind direction rel. bow (+ = from starboard)")
        ax.set_xlabel("deg")
        ax.set_ylabel("rows")
        ax.axvline(0, color="#666", lw=0.8)
    for ax, (name, s) in zip(axes[1], (("Apparent", ws), ("True (reconstructed)", true_speed))):
        ax.bar(scenters, np.histogram(s, bins=SPEED_BINS)[0], width=0.85, color="#eb6834")
        ax.set_title(f"{name} wind speed")
        ax.set_xlabel("m/s")
        ax.set_ylabel("rows")
    fig.suptitle(f"{path.name}: {n_rows:,} rows, {n_eps:,} episodes")
    fig.tight_layout()
    out = path.with_name(path.stem + "_wind_coverage.png")
    fig.savefig(str(out), dpi=130)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
