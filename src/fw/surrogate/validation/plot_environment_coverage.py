#!/usr/bin/env python3
"""
Summarize which environmental conditions a surrogate dataset covers.

Reads a dataset CSV (generator layout: raw row, 3 targets, episode_id; .csv or .csv.gz)
in chunks and summarizes the distribution of:

    UKC                          (confined and full datasets)
    true wind speed / direction  (full; the weather the ship sailed in)
    apparent wind speed / angle  (full; what the surrogate is given)
    current speed / direction    (full; earth-fixed)
    current relative to the ship (full; following / head / beam)

Every row is one 0.1 s step, so all shares are shares of simulated time.

Figure conventions (dataset conventions are in fw/CONVENTIONS.md):
    true wind     compass direction the wind comes FROM (N = 0 deg, clockwise)
    apparent wind angle off the bow it comes FROM (0 = ahead, + = starboard)
    current       compass direction it flows TOWARD (N = 0 deg, clockwise)
    current rel.  angle off the bow it flows TOWARD (0 = following current,
                  180 = head current, + = toward starboard)

True wind is reconstructed as apparent = true + (u, -v) in (ahead, starboard) components
(sway port positive). FH updates wind at 1 Hz, so apparent wind may lag u, v by up to
~1 s; the resulting error is small.

Percentiles come from fine histograms (0.01 m, 0.01 m/s, 0.001 m/s for current, 1 deg);
min, max, mean and std are exact.

Outputs in <out> (default <csv folder>/<csv stem>_coverage/):
    overview.png                16:9 slide with every distribution
    ukc.png, wind_speed.png, wind_rose_true.png, wind_angle_apparent.png,
    current_speed.png, current_rose.png, current_relative.png   (one panel each)
    summary.csv                 min, percentiles, max, mean, std per quantity
    thresholds.csv              share of time below thresholds (e.g. UKC < 2 m)
    directions.csv              share of time per 30-deg sector (full datasets)
The summary is also printed. Needs numpy, pandas and matplotlib only.

Usage (from src):
    python -m fw.surrogate.validation.plot_environment_coverage data/tripoli_full/ship_dynamics_dataset.csv
    python -m fw.surrogate.validation.plot_environment_coverage a.csv b.csv --dpi 200
    python -m fw.surrogate.validation.plot_environment_coverage big.csv --max-rows 2000000   # quick look
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fw.surrogate.make_ukc_lookup import detect_scenario
from fw.surrogate.schema import SURROGATE_DT, ScenarioSchema

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ settings
CHUNK_ROWS = 1_000_000

# Fine histograms for percentiles; out-of-range values still count in min/max/mean
# and in the under-/overflow bins.
UKC_EDGES = np.arange(0.0, 100.0 + 0.01, 0.01)            # m
WIND_EDGES = np.arange(0.0, 60.0 + 0.01, 0.01)            # m/s
CURRENT_EDGES = np.arange(0.0, 5.0 + 0.001, 0.001)        # m/s
DIR_EDGES = np.arange(0.0, 360.0 + 1.0, 1.0)              # deg, [0, 360)

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
UKC_BELOW = (1.0, 2.0, 3.0, 5.0)                          # m
WIND_BELOW = (1.0, 2.0, 5.0, 10.0)                        # m/s
CURRENT_BELOW = (0.1, 0.25, 0.5, 1.0)                     # m/s

# Speed classes of the roses. Wind: Beaufort 0-1, 2, 3, 4, 5, 6+.
WIND_CLASSES = (0.0, 1.6, 3.4, 5.5, 8.0, 10.8, np.inf)
WIND_CLASS_LABELS = ("< 1.6 (Bft 0-1)", "1.6-3.4 (Bft 2)", "3.4-5.5 (Bft 3)",
                     "5.5-8.0 (Bft 4)", "8.0-10.8 (Bft 5)", "> 10.8 (Bft 6+)")
CURRENT_CLASSES = (0.0, 0.25, 0.5, 1.0, 1.5, np.inf)
CURRENT_CLASS_LABELS = ("< 0.25", "0.25-0.5", "0.5-1.0", "1.0-1.5", "> 1.5")
ROSE_SECTOR_DEG = 10.0
TABLE_SECTOR_DEG = 30.0

# Colors: light surface, recessive ink and grid, sequential blue ramp (light = low speed).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAR = "#2a78d6"
# Ordinal steps 250..700 (lightest still has 2:1 contrast on the surface).
SEQ_BLUE = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")


# ------------------------------------------------------------------ accumulators
@dataclass
class Scalar:
    """
    Streaming statistics of one scalar quantity: exact moments, histogram percentiles.
    """

    name: str
    unit: str
    edges: np.ndarray
    below: Tuple[float, ...] = ()
    counts: np.ndarray = field(init=False)
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    lo: float = np.inf
    hi: float = -np.inf
    n_below: Dict[float, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Allocate the histogram (one extra bin each side for under- / overflow).
        """

        self.counts = np.zeros(len(self.edges) + 1, dtype=np.int64)
        self.n_below = {t: 0 for t in self.below}

    def add(self, values: np.ndarray) -> None:
        """
        Add finite values to the statistics.
        """

        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if not len(v):
            return
        self.n += len(v)
        self.total += float(v.sum())
        self.total_sq += float(np.square(v).sum())
        self.lo = min(self.lo, float(v.min()))
        self.hi = max(self.hi, float(v.max()))
        # Bin i (1..len(edges)-1) holds [edges[i-1], edges[i]); 0 = underflow, last = overflow.
        self.counts += np.bincount(np.searchsorted(self.edges, v, side="right"),
                                   minlength=len(self.counts))[:len(self.counts)]
        for t in self.below:
            self.n_below[t] += int((v < t).sum())

    @property
    def mean(self) -> float:
        """
        Exact mean (NaN without data).
        """

        return self.total / self.n if self.n else float("nan")

    @property
    def std(self) -> float:
        """
        Exact population standard deviation (NaN without data).
        """

        if not self.n:
            return float("nan")
        return float(np.sqrt(max(self.total_sq / self.n - self.mean ** 2, 0.0)))

    def percentile(self, q: float) -> float:
        """
        q-th percentile from the histogram (bin upper edge; clamped to min / max).
        """

        if not self.n:
            return float("nan")
        rank = q / 100.0 * self.n
        idx = int(np.searchsorted(np.cumsum(self.counts), rank, side="left"))
        if idx <= 0:
            return self.lo
        if idx >= len(self.edges):
            return self.hi
        return float(min(max(self.edges[idx], self.lo), self.hi))

    def share_below(self, threshold: float) -> float:
        """
        Share [%] of the values below `threshold`.
        """

        return 100.0 * self.n_below[threshold] / self.n if self.n else float("nan")

    def bin_shares(self, step: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        (left edges, share [%]) over the in-range part, regrouped into `step`-wide bins.
        """

        fine = self.counts[1:len(self.edges)]                 # in-range bins
        width = float(self.edges[1] - self.edges[0])
        k = max(1, int(round(step / width)))
        n_bins = int(np.ceil(len(fine) / k))
        padded = np.zeros(n_bins * k, dtype=np.int64)
        padded[:len(fine)] = fine
        grouped = padded.reshape(n_bins, k).sum(axis=1)
        lefts = self.edges[0] + np.arange(n_bins) * k * width
        return lefts, 100.0 * grouped / max(self.n, 1)


@dataclass
class Direction:
    """
    Streaming histogram of a direction [deg, 0..360) in 1-deg bins, optionally split by speed class.
    """

    name: str
    convention: str
    classes: Tuple[float, ...] = (0.0, np.inf)
    counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """
        Allocate the (direction bin, speed class) histogram.
        """

        self.counts = np.zeros((len(DIR_EDGES) - 1, len(self.classes) - 1), dtype=np.int64)

    def add(self, direction_deg: np.ndarray, speed: Optional[np.ndarray] = None) -> None:
        """
        Add directions (any range; wrapped to [0, 360)) with their speeds.
        """

        d = np.asarray(direction_deg, dtype=np.float64)
        s = np.zeros_like(d) if speed is None else np.asarray(speed, dtype=np.float64)
        ok = np.isfinite(d) & np.isfinite(s)
        d, s = np.mod(d[ok], 360.0), s[ok]
        d_bin = np.minimum(d.astype(np.int64), len(DIR_EDGES) - 2)
        c_bin = np.clip(np.searchsorted(self.classes, s, side="right") - 1, 0, len(self.classes) - 2)
        np.add.at(self.counts, (d_bin, c_bin), 1)

    @property
    def n(self) -> int:
        """
        Number of directions added.
        """

        return int(self.counts.sum())

    def sector_shares(self, sector_deg: float, centred: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        (sector centres [deg], share [%] per sector and speed class).

        With `centred`, sector 0 is centred on 0 deg (e.g. N for 30 deg:
        345..15 deg), as on a compass rose.
        """

        k = int(round(sector_deg))
        shift = k // 2 if centred else 0
        rolled = np.roll(self.counts, shift, axis=0)          # bin 345..359 moves to the front
        grouped = rolled.reshape(-1, k, rolled.shape[1]).sum(axis=1)
        centres = np.arange(grouped.shape[0]) * sector_deg + (0.0 if centred else sector_deg / 2)
        return centres, 100.0 * grouped / max(self.n, 1)


# ------------------------------------------------------------------ conversions
def compass_from_math(angle_rad: np.ndarray) -> np.ndarray:
    """
    Convert a math angle (rad, CCW from East) to a compass bearing (deg, clockwise from North, [0, 360)).
    """

    return np.mod(90.0 - np.degrees(angle_rad), 360.0)


def true_wind(app_speed: np.ndarray, app_dir: np.ndarray, heading: np.ndarray,
              u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    True wind (speed [m/s], compass direction it comes from [deg]) from the
    stored apparent wind and the ship's motion.

    Args:
        app_speed, app_dir: Apparent wind [m/s, rad off the bow, + = from starboard].
        heading: Ship heading [rad, CCW from East] (framework convention).
        u, v: Surge and sway [m/s, sway port positive].
    """

    # (ahead, starboard) components of the "comes from" vector: apparent = true + (u, -v).
    ahead = app_speed * np.cos(app_dir) - u
    stbd = app_speed * np.sin(app_dir) + v
    # World unit vectors: bow at `heading`, starboard 90 deg clockwise of it.
    wx = ahead * np.cos(heading) + stbd * np.cos(heading - np.pi / 2)
    wy = ahead * np.sin(heading) + stbd * np.sin(heading - np.pi / 2)
    return np.hypot(wx, wy), compass_from_math(np.arctan2(wy, wx))


def current_relative_deg(current_dir: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """
    Direction the current flows TOWARD, clockwise off the bow [deg, 0..360): 0 = following, 180 = head.
    """
    return np.mod(-np.degrees(current_dir - heading), 360.0)


# ------------------------------------------------------------------ reading
@dataclass
class Coverage:
    """
    All accumulated statistics of one dataset.
    """

    scenario: ScenarioSchema
    rows: int = 0
    episodes: set = field(default_factory=set)
    ukc: Scalar = field(default_factory=lambda: Scalar("UKC", "m", UKC_EDGES, UKC_BELOW))
    wind_true: Scalar = field(default_factory=lambda: Scalar("true wind speed", "m/s", WIND_EDGES, WIND_BELOW))
    wind_app: Scalar = field(default_factory=lambda: Scalar("apparent wind speed", "m/s", WIND_EDGES, WIND_BELOW))
    current: Scalar = field(default_factory=lambda: Scalar("current speed", "m/s", CURRENT_EDGES, CURRENT_BELOW))
    dir_wind_true: Direction = field(default_factory=lambda: Direction(
        "true wind direction", "compass, comes FROM", WIND_CLASSES))
    dir_wind_app: Direction = field(default_factory=lambda: Direction(
        "apparent wind angle", "off the bow, comes FROM, + = starboard", WIND_CLASSES))
    dir_current: Direction = field(default_factory=lambda: Direction(
        "current direction", "compass, flows TOWARD", CURRENT_CLASSES))
    dir_current_rel: Direction = field(default_factory=lambda: Direction(
        "current relative to the ship", "off the bow, flows TOWARD, 0 = following", CURRENT_CLASSES))

    @property
    def has_env(self) -> bool:
        """
        True for datasets with wind and current (full_env).
        """

        return self.scenario.includes_wind_current


def read_coverage(csv_file: Path, chunk_rows: int = CHUNK_ROWS, max_rows: Optional[int] = None) -> Coverage:
    """
    Stream `csv_file` and accumulate the coverage statistics.

    Raises:
        ValueError: For a dataset without UKC (open water), or a width that
            matches no scenario.
    """

    scenario, n_cols = detect_scenario(csv_file)
    if not scenario.includes_ukc:
        raise ValueError(f"{csv_file}: '{scenario.name}' data has no UKC, wind or current to summarize.")
    names = ["ukc"]
    if scenario.includes_wind_current:
        names += ["heading", "surge", "sway", "wind_speed", "wind_direction", "current_speed", "current_direction"]
    cols = {name: scenario.col(name) for name in names}
    cols["episode_id"] = n_cols - 1
    usecols = sorted(set(cols.values()))

    cov = Coverage(scenario)
    reader = pd.read_csv(csv_file, header=None, usecols=usecols, dtype=np.float64,
                         chunksize=chunk_rows, nrows=max_rows)
    for chunk in reader:
        chunk.columns = usecols
        d = {k: chunk[c].to_numpy() for k, c in cols.items()}
        cov.rows += len(chunk)
        cov.episodes.update(np.unique(d["episode_id"][np.isfinite(d["episode_id"])]).tolist())
        cov.ukc.add(d["ukc"])
        if scenario.includes_wind_current:
            speed, from_deg = true_wind(d["wind_speed"], d["wind_direction"], d["heading"], d["surge"], d["sway"])
            cov.wind_true.add(speed)
            cov.dir_wind_true.add(from_deg, speed)
            cov.wind_app.add(d["wind_speed"])
            cov.dir_wind_app.add(np.degrees(d["wind_direction"]), d["wind_speed"])
            cov.current.add(d["current_speed"])
            cov.dir_current.add(compass_from_math(d["current_direction"]), d["current_speed"])
            cov.dir_current_rel.add(current_relative_deg(d["current_direction"], d["heading"]),
                                    d["current_speed"])
        logger.info("%s: %d rows read", csv_file.name, cov.rows)
    if cov.rows == 0:
        raise ValueError(f"{csv_file}: no rows.")
    return cov


def dataset_title(csv_file: Path, cov: Coverage, title: Optional[str] = None) -> str:
    """
    Figure title: ship and configuration from the generator's sidecar JSON when present.
    """

    if title:
        return title
    stem = csv_file.stem.replace(".csv", "")
    # Without a sidecar, the folder name (e.g. "tripoli_full") beats the default file name.
    label = csv_file.parent.name if stem == "ship_dynamics_dataset" and csv_file.parent.name else stem
    sidecar = csv_file.with_name(stem + "_generation.json")
    try:
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
        label = f"{prov.get('ship_model', '?')} - {prov.get('configuration', cov.scenario.name)}"
    except (OSError, ValueError, AttributeError):
        pass
    hours = cov.rows * SURROGATE_DT / 3600.0
    return f"{label}  |  {len(cov.episodes):,} episodes, {cov.rows:,} rows ({hours:,.1f} h simulated)"


# ------------------------------------------------------------------ tables
def summary_rows(cov: Coverage) -> List[Dict[str, object]]:
    """
    One summary dict per scalar quantity (percentiles, moments, shares below thresholds).
    """

    quantities = [cov.ukc] + ([cov.wind_true, cov.wind_app, cov.current] if cov.has_env else [])
    rows = []
    for s in quantities:
        row: Dict[str, object] = {"quantity": s.name, "unit": s.unit, "rows": s.n,
                                  "min": s.lo, "mean": s.mean, "std": s.std}
        row.update({f"p{q}": s.percentile(q) for q in PERCENTILES})
        row["max"] = s.hi
        rows.append(row)
    return rows


def threshold_rows(cov: Coverage) -> List[Dict[str, object]]:
    """
    Share of time [%] below each threshold, one row per (quantity, threshold).
    """

    quantities = [cov.ukc] + ([cov.wind_true, cov.wind_app, cov.current] if cov.has_env else [])
    return [{"quantity": s.name, "unit": s.unit, "below": t, "share_of_time_pct": s.share_below(t)}
            for s in quantities for t in s.below]


def direction_rows(cov: Coverage) -> List[Dict[str, object]]:
    """
    Share of time per TABLE_SECTOR_DEG sector for every direction quantity.
    """

    rows = []
    if not cov.has_env:
        return rows
    for dirq in (cov.dir_wind_true, cov.dir_wind_app, cov.dir_current, cov.dir_current_rel):
        centres, shares = dirq.sector_shares(TABLE_SECTOR_DEG)
        total = shares.sum(axis=1)
        row: Dict[str, object] = {"quantity": dirq.name, "convention": dirq.convention}
        row.update({f"{c:03.0f}": float(t) for c, t in zip(centres, total)})
        row["min_sector_share"] = float(total.min())
        row["max_sector_share"] = float(total.max())
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """
    Write dict rows (same keys per quantity group; union of keys as header).
    """

    if not rows:
        return
    header: List[str] = []
    for r in rows:
        header += [k for k in r if k not in header]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


def print_summary(title: str, cov: Coverage) -> None:
    """
    Print the summary tables.
    """

    print(f"\n== {title}")
    shares = threshold_rows(cov)
    for r in summary_rows(cov):
        pct = "  ".join(f"p{q} {r[f'p{q}']:.2f}" for q in (5, 50, 95))
        below = "  ".join(f"<{t['below']:g}: {t['share_of_time_pct']:.1f}%"
                          for t in shares if t["quantity"] == r["quantity"])
        print(f"  {r['quantity']:<20} [{r['unit']}]  min {r['min']:.2f}  {pct}  max {r['max']:.2f}"
              f"  mean {r['mean']:.2f}  |  {below}")
    for r in direction_rows(cov):
        print(f"  {r['quantity']:<29} 30-deg sectors: min {r['min_sector_share']:.1f}%  "
              f"max {r['max_sector_share']:.1f}%  ({r['convention']})")


# ------------------------------------------------------------------ figures
def _style(ax, polar: bool = False) -> None:
    """
    Recessive axes and grid, readable at slide size.
    """

    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if not polar:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)


def plot_scalar(ax, s: Scalar, step: float, title: str) -> None:
    """
    Histogram (share of time per bin) with the 5th, 50th and 95th percentiles marked.
    """

    _style(ax)
    lefts, shares = s.bin_shares(step)
    hi_idx = int(np.searchsorted(lefts, s.percentile(99.9) if s.n else 0.0)) + 2
    lo_idx = max(int(np.searchsorted(lefts, s.lo if s.n else 0.0, side="right")) - 2, 0)
    lefts, shares = lefts[lo_idx:hi_idx], shares[lo_idx:hi_idx]
    ax.bar(lefts, shares, width=step * 0.92, align="edge", color=BAR, linewidth=0)
    top = shares.max() if len(shares) else 1.0
    for q, ls in ((5, ":"), (50, "-"), (95, ":")):
        ax.axvline(s.percentile(q), color=INK_2, linewidth=1.2, linestyle=ls)
    ax.set_ylim(0, top * 1.08)
    ax.set_xlabel(f"{s.name} [{s.unit}]", fontsize=12, color=INK)
    ax.set_ylabel("share of time [%]", fontsize=12, color=INK)
    ax.set_title(title, fontsize=14, color=INK, loc="left", pad=34)
    ax.text(0.0, 1.02, f"median {s.percentile(50):.2f} {s.unit}  (solid line)\n"
            f"5-95%: {s.percentile(5):.2f} - {s.percentile(95):.2f} {s.unit}  (dotted)",
            transform=ax.transAxes, fontsize=10, color=INK_2, va="bottom", linespacing=1.3)


def plot_rose(ax, dirq: Direction, labels: Sequence[str], title: str, tick_labels: str) -> None:
    """
    Stacked polar histogram: share of time per direction sector, split by speed class.
    """

    _style(ax, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)                                   # clockwise, as a compass / off the bow
    centres, shares = dirq.sector_shares(ROSE_SECTOR_DEG)
    theta = np.radians(centres)
    width = np.radians(ROSE_SECTOR_DEG) * 0.9
    bottom = np.zeros(len(centres))
    colors = SEQ_BLUE[:len(labels)]                              # light = low speed
    for k, label in enumerate(labels):
        ax.bar(theta, shares[:, k], width=width, bottom=bottom, color=colors[k],
               edgecolor=SURFACE, linewidth=0.6, label=label)
        bottom += shares[:, k]
    ax.set_title(title, fontsize=13, color=INK, pad=18)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(tick_labels.split(","), fontsize=11, color=INK_2)
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%" if y >= 1 else f"{y:.1f}%")
    ax.set_rlabel_position(112.5)                                # between the E/SE labels
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK_2)
    for label in ax.get_yticklabels():                           # stay readable over the bars
        label.set_bbox(dict(facecolor=SURFACE, edgecolor="none", alpha=0.8, pad=1))
        label.set_zorder(5)
    ax.spines["polar"].set_color(GRID)


COMPASS_LABELS = "N,NE,E,SE,S,SW,W,NW"
BOW_LABELS = "ahead,,starboard,,astern,,port,"
REL_CURRENT_LABELS = "following,,to stbd,,head,,to port,"


def _panels(cov: Coverage) -> List[Tuple[str, str, callable]]:
    """
    (file stem, kind, draw function) of every panel for this dataset.
    """

    panels = [("ukc", "xy", lambda ax: plot_scalar(ax, cov.ukc, 0.25, "Under-keel clearance"))]
    if cov.has_env:
        panels += [
            ("wind_speed", "xy", lambda ax: plot_scalar(ax, cov.wind_true, 0.5, "True wind speed")),
            ("current_speed", "xy", lambda ax: plot_scalar(ax, cov.current, 0.05, "Current speed")),
            ("wind_rose_true", "polar", lambda ax: plot_rose(
                ax, cov.dir_wind_true, WIND_CLASS_LABELS, "True wind, comes from\n(compass)",
                COMPASS_LABELS)),
            ("wind_angle_apparent", "polar", lambda ax: plot_rose(
                ax, cov.dir_wind_app, WIND_CLASS_LABELS, "Apparent wind, comes from\n(off the bow: what the model sees)",
                BOW_LABELS)),
            ("current_rose", "polar", lambda ax: plot_rose(
                ax, cov.dir_current, CURRENT_CLASS_LABELS, "Current, flows toward\n(compass)",
                COMPASS_LABELS)),
            ("current_relative", "polar", lambda ax: plot_rose(
                ax, cov.dir_current_rel, CURRENT_CLASS_LABELS, "Current relative to the ship\n(flows toward, off the bow)",
                REL_CURRENT_LABELS)),
        ]
    return panels


def _legend(fig, ax, anchor, title: str) -> None:
    """
    Speed-class legend of a rose, at `anchor` (figure coordinates, upper left).
    """

    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=anchor, frameon=False,
                     fontsize=10, labelcolor=INK_2, title=title, title_fontsize=10, alignment="left",
                     labelspacing=0.3, handlelength=1.6)
    leg.get_title().set_color(INK)


def save_figures(cov: Coverage, title: str, out_dir: Path, dpi: int) -> List[Path]:
    """
    Write overview.png and one PNG per panel; returns the written paths.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    panels = _panels(cov)

    # One figure per panel.
    for stem, kind, draw in panels:
        fig = plt.figure(figsize=(9.0, 6.0) if kind == "xy" else (8.5, 7.0), facecolor=SURFACE)
        ax = fig.add_subplot(111, projection="polar" if kind == "polar" else None)
        if kind == "polar":
            fig.subplots_adjust(left=0.06, right=0.68, top=0.80, bottom=0.08)
        else:
            fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.11)
        draw(ax)
        if kind == "polar":
            _legend(fig, ax, (0.72, 0.72), "wind speed [m/s]" if "wind" in stem else "current speed [m/s]")
        fig.suptitle(title, fontsize=11, color=INK_2, x=0.01, ha="left")
        path = out_dir / f"{stem}.png"
        fig.savefig(path, dpi=dpi, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)

    # Overview slide (16:9).
    fig = plt.figure(figsize=(16.0, 9.0), facecolor=SURFACE)
    if cov.has_env:
        grid = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.42, left=0.05, right=0.97, top=0.86, bottom=0.05)
        slots = {"ukc": grid[0, 0], "wind_speed": grid[0, 1], "current_speed": grid[0, 2],
                 "wind_rose_true": grid[1, 0], "wind_angle_apparent": grid[1, 1],
                 "current_rose": grid[1, 2], "current_relative": grid[1, 3]}
    else:
        grid = fig.add_gridspec(1, 1, left=0.08, right=0.95, top=0.85, bottom=0.12)
        slots = {"ukc": grid[0, 0]}
    for stem, kind, draw in panels:
        ax = fig.add_subplot(slots[stem], projection="polar" if kind == "polar" else None)
        draw(ax)
        if stem == "wind_rose_true":
            wind_ax = ax
        elif stem == "current_rose":
            current_ax = ax
    if cov.has_env:
        # Top-right cell: both rose legends, stacked (side by side does not always fit).
        pos = fig.add_subplot(grid[0, 3]).get_position()
        fig.axes[-1].remove()
        _legend(fig, wind_ax, (pos.x0, pos.y1 + 0.04), "wind speed in the roses [m/s]")
        _legend(fig, current_ax, (pos.x0, pos.y1 + 0.04 - 0.215), "current speed in the roses [m/s]")
    fig.suptitle(title, fontsize=15, color=INK, x=0.02, ha="left")
    path = out_dir / "overview.png"
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    written.insert(0, path)
    return written


# ------------------------------------------------------------------ main
def default_out_dir(csv_file: Path) -> Path:
    """
    <csv folder>/<csv stem>_coverage (".csv.gz" counts as one suffix).
    """

    stem = csv_file.name[:-len(".csv.gz")] if csv_file.name.endswith(".csv.gz") else csv_file.stem
    return csv_file.with_name(stem + "_coverage")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Parse the arguments, summarize every CSV, write tables and figures.
    """

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", type=Path, help="dataset CSV(s) (.csv or .csv.gz)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (one CSV only; default <csv folder>/<stem>_coverage)")
    parser.add_argument("--title", default=None, help="figure title (one CSV only)")
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS, help="rows read per chunk")
    parser.add_argument("--max-rows", type=int, default=None, help="read only the first N rows (quick look)")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
    parser.add_argument("--no-figures", action="store_true", help="tables only")
    args = parser.parse_args(argv)
    if len(args.csv) > 1 and (args.out or args.title):
        parser.error("--out and --title need a single CSV")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    for csv_file in args.csv:
        cov = read_coverage(csv_file, args.chunk_rows, args.max_rows)
        title = dataset_title(csv_file, cov, args.title)
        out_dir = args.out or default_out_dir(csv_file)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / "summary.csv", summary_rows(cov))
        write_csv(out_dir / "thresholds.csv", threshold_rows(cov))
        write_csv(out_dir / "directions.csv", direction_rows(cov))
        print_summary(title, cov)
        if args.max_rows is not None:
            print(f"  (first {args.max_rows:,} rows only: the rows are shuffled, so this is a random sample)")
        if not args.no_figures:
            for p in save_figures(cov, title, out_dir, args.dpi):
                print(f"  wrote {p}")
        tables = ["summary.csv", "thresholds.csv"] + (["directions.csv"] if cov.has_env else [])
        print("  wrote " + ", ".join(str(out_dir / t) for t in tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
