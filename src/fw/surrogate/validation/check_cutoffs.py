"""
check_cutoffs.py -- check map cut-off detection (PySimEnv.bank_segments) for
the river polygon the dataset generator uses.

If the depth contour has an entry in py_sim_env.MAP_CUTOFF_LINES, the
explicit cut-off lines are used: this lists each detected cut-off run and
saves cutoff_check.png with the configured lines (black dashed) and the
segments classified as cut-off (red), so the line coordinates can be
verified. Otherwise, it compares the bounding-box heuristic over a range of
edge tolerances, one panel per tolerance.

Usage (from the folder you run the generator from, e.g. src):
    python -m fw.surrogate.check_cutoffs
    python fw/surrogate/check_cutoffs.py
"""

import os
import sys
import traceback
from pathlib import Path

# Allow running by file path: put the folder that contains the `fw` package
# (src) on the import path. Not needed with `python -m`.
_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import MultiLineString
from shapely.ops import linemerge

from fw.environments import py_sim_env
from fw.environments.py_sim_env import PySimEnv

TOLERANCES = (0.01, 1.0, 10.0, 25.0, 50.0, 100.0)  # m; heuristic only; 0.01 is the default


def _print_runs(cutoff: np.ndarray) -> None:
    """Print each connected cut-off run with its end points and length."""

    length = float(np.linalg.norm(cutoff[:, 1] - cutoff[:, 0], axis=1).sum()) if len(cutoff) else 0.0
    print(f"  {len(cutoff)} cut-off segment(s), {length:.1f} m in total")
    if len(cutoff):
        merged = linemerge(MultiLineString([seg for seg in cutoff]))
        for k, run in enumerate(getattr(merged, "geoms", [merged])):
            (x0, y0), (x1, y1) = run.coords[0], run.coords[-1]
            print(f"   run {k}: ({x0:.1f}, {y0:.1f}) -> ({x1:.1f}, {y1:.1f}), {run.length:.1f} m")


def _draw(ax, poly, cutoff, title, lines=()) -> None:
    for ring in [poly.exterior, *poly.interiors]:
        xy = np.asarray(ring.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", linewidth=0.8)
    for seg in cutoff:
        ax.plot(seg[:, 0], seg[:, 1], color="tab:red", linewidth=3)
    for line in lines:
        xy = np.asarray(line, dtype=np.float64)
        ax.plot(xy[:, 0], xy[:, 1], "k--", linewidth=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", "box")


def main() -> None:
    env = PySimEnv(render_mode=None, time_step=0.1, wind=False, current=False)
    poly = env.polygon_shape
    minx, miny, maxx, maxy = poly.bounds
    contour = os.path.basename(str(env.depth_contour))
    print(f"Depth contour: {contour}")
    print(f"Polygon: {len(poly.exterior.coords)} exterior vertices, {len(poly.interiors)} interior ring(s)")
    print(f"Bounds: x [{minx:.2f}, {maxx:.2f}]  y [{miny:.2f}, {maxy:.2f}]")

    lines = py_sim_env.MAP_CUTOFF_LINES.get(contour)
    if lines:
        print(f"\nExplicit cut-off lines (MAP_CUTOFF_LINES, tolerance "
              f"{py_sim_env.CUTOFF_LINE_TOLERANCE:g} m): {len(lines)}")
        _, cutoff = env.bank_segments()
        cutoff = np.asarray(cutoff, dtype=np.float64).reshape(-1, 2, 2)
        _print_runs(cutoff)
        fig, ax = plt.subplots(figsize=(12, 10))
        _draw(ax, poly, cutoff, "Configured cut-off lines (black dashed), classified cut-offs (red)", lines)
    else:
        print("\nNo MAP_CUTOFF_LINES entry for this contour: bounding-box heuristic")
        found = {}
        for tol in TOLERANCES:
            print(f"\nedge_tol={tol:g} m:")
            try:
                _, cutoff = env.bank_segments(edge_tol=tol)
            except Exception:
                print("  bank_segments() RAISED:")
                traceback.print_exc()
                continue
            found[tol] = np.asarray(cutoff, dtype=np.float64).reshape(-1, 2, 2)
            _print_runs(found[tol])
        ncols = 3
        nrows = -(-len(TOLERANCES) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
        for ax, tol in zip(axes.flat, TOLERANCES):
            status = f"{len(found[tol])} cut-off segment(s)" if tol in found else "bank_segments() raised"
            _draw(ax, poly, found.get(tol, []), f"edge_tol = {tol:g} m: {status}")
        for ax in list(axes.flat)[len(TOLERANCES):]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig("cutoff_check.png", dpi=150)
    print("\nSaved cutoff_check.png (red = classified as map cut-off)")


if __name__ == "__main__":
    main()
