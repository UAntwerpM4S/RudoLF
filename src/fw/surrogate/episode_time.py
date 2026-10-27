"""
Time since the start of the episode, per row of a dataset split.

The generator shuffles the rows of every split and stores no step index, so
the order within an episode is reconstructed from the data itself (see
`step_index`). This module

  * provides `step_index` (also used by validation/time_error_diagnostic and
    validation/bank_error_diagnostic), and
  * as a script, adds that time as an extra array `t_episode` [s] to existing
    dataset .npz files, so that train_surrogate --min-episode-time can leave
    out the start of every episode (where FH's hidden states -- roll -- are
    still settling from the artificial initial state).

Rows of episodes whose order cannot be reconstructed get t_episode = -1
(train_surrogate --min-episode-time drops them).

Usage (from src; rewrites the files in place, all other arrays unchanged):

    python -m fw.surrogate.episode_time ^
        data\\tripoli_confined\\ship_dynamics_dataset_train.npz ^
        data\\tripoli_confined\\ship_dynamics_dataset_val.npz ^
        data\\tripoli_confined\\ship_dynamics_dataset_test.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from pathlib import Path
from typing import Optional, Sequence, Tuple
from scipy.spatial import cKDTree

from fw.surrogate.dataset import load_raw
from fw.surrogate.schema import SURROGATE_DT, get_scenario

logger = logging.getLogger(__name__)

T_EPISODE_KEY = "t_episode"
# Matching space: position [m] plus velocities scaled up, so that a successor
# is found both when the ship moves (position separates the rows, ~0.1-0.6 m
# per step) and when it is (nearly) stopped (only the velocities differ).
VEL_SCALE = 100.0          # surge/sway [m/s] -> x100
YAW_SCALE = 1.0e4          # yaw rate [rad/s] -> x1e4
MATCH_TOL = 0.1            # max distance in that space between predicted and found successor
STATE_COLUMNS = ("x", "y", "heading", "surge", "sway", "yaw_rate")


def _match_space(x, y, u, v, r) -> np.ndarray:
    """Points in the successor-matching space (see VEL_SCALE / YAW_SCALE)."""

    return np.column_stack([x, y, VEL_SCALE * u, VEL_SCALE * v, YAW_SCALE * r])


def step_index(state: np.ndarray, delta: np.ndarray, episode_id: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Step index of every row within its episode (-1 where the chain could not
    be reconstructed).

    A row's successor is the row of the same episode closest to this row's
    predicted next state: velocities = state + targets (exact up to float32
    rounding), position = one kinematic step of SURROGATE_DT. The row nobody
    points to is the first one.

    Args:
        state: (N, 6) x, y, heading [rad, 0 = East, CCW], surge, sway (port
            positive), yaw_rate [rad/s].
        delta: (N, 3) delta_surge, delta_sway, delta_yaw_rate.
        episode_id: (N,) episode of every row.

    Returns:
        (steps, n_ok, n_bad): per-row step index, and the numbers of episodes
        reconstructed / skipped.
    """

    steps = np.full(len(state), -1, dtype=np.int64)
    n_ok = n_bad = 0
    if len(state) == 0:
        return steps, n_ok, n_bad
    order = np.argsort(episode_id, kind="stable")
    bounds = np.flatnonzero(np.diff(episode_id[order])) + 1
    for rows in np.split(order, bounds):
        if len(rows) == 1:
            steps[rows] = 0
            n_ok += 1
            continue
        x, yy, h, u, v, r = state[rows].astype(np.float64).T
        du, dv, dr = delta[rows].astype(np.float64).T
        hm = h + 0.5 * SURROGATE_DT * r                   # mid-step heading
        x_next = x + SURROGATE_DT * (u * np.cos(hm) - v * np.sin(hm))
        y_next = yy + SURROGATE_DT * (u * np.sin(hm) + v * np.cos(hm))
        here = _match_space(x, yy, u, v, r)
        nxt = _match_space(x_next, y_next, u + du, v + dv, r + dr)
        dist, succ = cKDTree(here).query(nxt)
        has_succ = dist <= MATCH_TOL
        pointed = np.zeros(len(rows), dtype=bool)
        pointed[succ[has_succ]] = True
        starts = np.flatnonzero(~pointed)
        ends = np.flatnonzero(~has_succ)
        if len(starts) != 1 or len(ends) != 1:
            n_bad += 1
            continue
        chain = [int(starts[0])]
        seen = {chain[0]}
        while has_succ[chain[-1]]:
            j = int(succ[chain[-1]])
            if j in seen:
                break
            chain.append(j)
            seen.add(j)
        if len(chain) != len(rows):
            n_bad += 1
            continue
        steps[rows[chain]] = np.arange(len(chain))
        n_ok += 1
    return steps, n_ok, n_bad


def episode_time(path) -> Tuple[np.ndarray, int, int]:
    """
    Time since episode start [s] per row of a dataset .npz (-1 = not
    reconstructable), plus the numbers of episodes reconstructed / skipped.
    """

    x_raw, y, identity = load_raw(path)
    source = get_scenario(identity.scenario)
    with np.load(path, allow_pickle=False) as z:
        episode_id = z["episode_id"].astype(np.int64)
    state = x_raw[:, [source.col(c) for c in STATE_COLUMNS]]
    steps, n_ok, n_bad = step_index(state, y, episode_id)
    t = np.where(steps >= 0, steps * SURROGATE_DT, -1.0).astype(np.float32)
    return t, n_ok, n_bad


def add_episode_time(path: Path) -> None:
    """Write `t_episode` into the .npz at `path` (atomically; other arrays unchanged)."""

    t, n_ok, n_bad = episode_time(path)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != T_EPISODE_KEY}
    arrays[T_EPISODE_KEY] = t
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    ok = t >= 0
    logger.info("%s: %d rows, %d episodes ordered, %d skipped (%d rows get t_episode = -1); "
                "t_episode up to %.1f s.", path, len(t), n_ok, n_bad, int((~ok).sum()),
                float(t[ok].max()) if ok.any() else float("nan"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; returns the exit code."""

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", type=Path, nargs="+", help="Dataset .npz files to add t_episode to (in place).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for f in args.files:
        add_episode_time(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
