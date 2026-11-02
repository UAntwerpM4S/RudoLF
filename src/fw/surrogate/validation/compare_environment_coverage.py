#!/usr/bin/env python3
"""
One slide comparing the environment coverage of several datasets (e.g. Tripoli full vs Myzako full).

Reads only the small tables written by plot_environment_coverage.py (summary.csv,
thresholds.csv, directions.csv). Per dataset:
    top row     UKC, true wind speed, current speed: median, 25-75 % and 5-95 % ranges
    bottom row  share of time below thresholds, and per 30-deg sector of the true wind
                direction and of the current relative to the ship

Usage (from src):
    python -m fw.surrogate.validation.compare_environment_coverage \\
        data/tripoli_full/ship_dynamics_dataset_coverage data/myzako_full/ship_dynamics_dataset_coverage \\
        --labels Tripoli Myzako --out coverage_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")        # categorical slots 1-3, fixed order

RANGES = (("UKC", "Under-keel clearance", "m"),
          ("true wind speed", "True wind speed", "m/s"),
          ("current speed", "Current speed", "m/s"))
THRESHOLDS = (("UKC", 2.0, "UKC < 2 m"), ("UKC", 3.0, "UKC < 3 m"),
              ("true wind speed", 1.0, "wind < 1 m/s"), ("true wind speed", 10.0, "wind < 10 m/s"),
              ("current speed", 0.1, "current < 0.1 m/s"), ("current speed", 1.0, "current < 1 m/s"))
ROSES = (("true wind direction", "True wind, comes from", "N,NE,E,SE,S,SW,W,NW"),
         ("current relative to the ship", "Current relative to the ship", "following,,to stbd,,head,,to port,"))


def _read(path: Path) -> List[Dict[str, str]]:
    """
    Read a CSV file as a list of dicts ([] if the file does not exist).
    """

    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load(folder: Path) -> Dict[str, object]:
    """
    Load summary, thresholds and directions of one coverage folder, keyed for lookup.
    """

    summary = {r["quantity"]: r for r in _read(folder / "summary.csv")}
    if not summary:
        raise ValueError(f"{folder}: no summary.csv (run plot_environment_coverage first).")
    thresholds = {(r["quantity"], float(r["below"])): float(r["share_of_time_pct"])
                  for r in _read(folder / "thresholds.csv")}
    directions = {r["quantity"]: r for r in _read(folder / "directions.csv")}
    return {"summary": summary, "thresholds": thresholds, "directions": directions}


def _style(ax, polar: bool = False) -> None:
    """
    Recessive axes and grid.
    """

    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if polar:
        ax.spines["polar"].set_color(GRID)
    else:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)


def plot(datasets: Sequence[Dict[str, object]], labels: Sequence[str], out: Path,
         title: Optional[str], dpi: int) -> None:
    """
    Draw and save the comparison slide.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(datasets)
    colors = SERIES[:n]
    fig = plt.figure(figsize=(16.0, 9.0), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 6, hspace=0.6, wspace=0.9, left=0.10, right=0.97, top=0.82, bottom=0.08)

    # Top row: percentile ranges.
    for k, (quantity, name, unit) in enumerate(RANGES):
        ax = fig.add_subplot(grid[0, 2 * k:2 * k + 2])
        _style(ax)
        ax.grid(axis="y", visible=False)
        for i, (data, color) in enumerate(zip(datasets, colors)):
            row = data["summary"].get(quantity)
            if row is None:
                continue
            y = n - 1 - i
            p = {q: float(row[f"p{q}"]) for q in (5, 25, 50, 75, 95)}
            ax.plot([p[5], p[95]], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot([p[25], p[75]], [y, y], color=color, linewidth=9, solid_capstyle="round")
            ax.plot(p[50], y, "o", markersize=11, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(p[95], y + 0.28, f"median {p[50]:.2f}, 5-95 %: {p[5]:.2f}-{p[95]:.2f}",
                    ha="right", va="bottom", fontsize=10, color=INK_2)
        ax.set_yticks(range(n))
        ax.set_yticklabels(list(labels)[::-1], fontsize=12, color=INK)
        ax.set_ylim(-0.6, n - 0.2)
        ax.set_xlabel(f"{name} [{unit}]", fontsize=12, color=INK)
        ax.set_title(name, fontsize=14, color=INK, loc="left", pad=22)
        ax.text(0.0, 1.02, "dot = median, thick = 25-75 %, thin = 5-95 % of the time",
                transform=ax.transAxes, fontsize=10, color=INK_2, va="bottom")

    # Bottom left: share of time below thresholds (grouped horizontal bars).
    ax = fig.add_subplot(grid[1, 0:2])
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(THRESHOLDS))[::-1].astype(float)
    height = 0.8 / n
    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        values = [data["thresholds"].get((q, t), np.nan) for q, t, _ in THRESHOLDS]
        bars = ax.barh(y + ((n - 1) / 2 - i) * height, values, height=height * 0.9, color=color, label=label)
        for bar, v in zip(bars, values):
            if np.isfinite(v):
                ax.text(v + 1.0, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", ha="left",
                        va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([t[2] for t in THRESHOLDS], fontsize=11, color=INK)
    ax.set_xlabel("share of time [%]", fontsize=12, color=INK)
    ax.set_xlim(0, 100)
    ax.set_title("Share of time below thresholds", fontsize=14, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=11, labelcolor=INK, loc="upper right")

    # Bottom right: 30-deg sector shares as closed lines.
    for k, (quantity, name, ticks) in enumerate(ROSES):
        ax = fig.add_subplot(grid[1, 2 + 2 * k:4 + 2 * k], projection="polar")
        _style(ax, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for data, color, label in zip(datasets, colors, labels):
            row = data["directions"].get(quantity)
            if row is None:
                continue
            sectors = sorted(int(c) for c in row if c.isdigit())
            values = [float(row[f"{c:03d}"]) for c in sectors]
            theta = np.radians(sectors + [sectors[0]])
            ax.plot(theta, values + [values[0]], color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, label=label)
        ax.set_ylim(0, None)
        ax.set_xticks(np.radians(np.arange(0, 360, 45)))
        ax.set_xticklabels(ticks.split(","), fontsize=10, color=INK_2)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
        ax.tick_params(axis="y", labelsize=9, labelcolor=INK_2)
        ax.set_rlabel_position(202.5)
        ax.set_title(f"{name}\n(share of time per 30° sector)", fontsize=12, color=INK, pad=16)

    fig.suptitle(title or "Environment coverage of the surrogate datasets: " + " vs ".join(labels),
                 fontsize=16, color=INK, x=0.02, ha="left")
    rows = "   |   ".join(f"{label}: {int(d['summary']['UKC']['rows']):,} rows "
                          f"({int(d['summary']['UKC']['rows']) * 0.1 / 3600:,.0f} h simulated)"
                          for label, d in zip(labels, datasets))
    fig.text(0.02, 0.905, rows, fontsize=11, color=INK_2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Parse the arguments and write the comparison slide.
    """

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path,
                        help="coverage folders written by plot_environment_coverage (2 or 3)")
    parser.add_argument("--labels", nargs="+", default=None, help="one label per folder")
    parser.add_argument("--out", type=Path, default=Path("coverage_comparison.png"))
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if not 1 <= len(args.folders) <= len(SERIES):
        parser.error(f"give 1 to {len(SERIES)} folders")
    labels = args.labels or [f.name.replace("ship_dynamics_dataset_coverage", "").strip("_") or f.parent.name
                             for f in args.folders]
    if len(labels) != len(args.folders):
        parser.error("--labels needs one label per folder")
    plot([load(f) for f in args.folders], labels, args.out, args.title, args.dpi)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
