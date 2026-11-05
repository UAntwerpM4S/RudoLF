"""
Time since the start of the episode, per row of a dataset split.

Rows are shuffled and carry no step index, so the order within an episode is
reconstructed from the data (step_index, also used by the time/bank error
diagnostics). As a script, it adds the array t_episode [s] to dataset .npz files,
so train_surrogate --min-episode-time can drop episode starts, where FH's hidden
states (roll) still settle from the artificial initial state. Rows of episodes
that cannot be ordered get t_episode = -1 (also dropped).

t_episode counts from the first recorded row, i.e. after FH start-up (0.1 s),
wind warm-up (full configuration, 1 s) and actuator settling (0-3 s).

Compare time windows in whole steps (steps_from_times / step_threshold), not on
the float32 times: float32(k * 0.1) is not exactly k / 10.

Usage (from src; rewrites the files in place, other arrays unchanged):

    python -m fw.surrogate.episode_time ^
        data\\tripoli_confined\\ship_dynamics_dataset_train.npz ^
        data\\tripoli_confined\\ship_dynamics_dataset_val.npz ^
        data\\tripoli_confined\\ship_dynamics_dataset_test.npz
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import numpy as np

from pathlib import Path
from scipy.spatial import cKDTree
from fw.surrogate.dataset import load_raw
from typing import Optional, Sequence, Tuple
from fw.surrogate.schema import SURROGATE_DT, get_scenario

logger = logging.getLogger(__name__)

T_EPISODE_KEY = "t_episode"
# Matching space: position [m] plus scaled-up velocities, so successors are found
# both when moving (position separates rows, ~0.1-0.6 m per step) and when
# (nearly) stopped (only velocities differ).
VEL_SCALE = 100.0          # surge/sway [m/s] -> x100
YAW_SCALE = 1.0e4          # yaw rate [rad/s] -> x1e4
MATCH_TOL = 0.1            # max distance in that space between predicted and found successor
STATE_COLUMNS = ("x", "y", "heading", "surge", "sway", "yaw_rate")


def _match_space(x, y, u, v, r) -> np.ndarray:
    """
    Points in the successor-matching space (see VEL_SCALE / YAW_SCALE).
    """

    return np.column_stack([x, y, VEL_SCALE * u, VEL_SCALE * v, YAW_SCALE * r])


def step_index(state: np.ndarray, delta: np.ndarray, episode_id: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Step index of every row within its episode (-1 where the chain could not be
    reconstructed).

    A row's successor is the other row of the same episode closest to its
    predicted next state (velocities = state + targets, position = one kinematic
    step of SURROGATE_DT). The worst-matching row is the last one (n rows have
    n - 1 links); if even it matches within MATCH_TOL (closed track, e.g. a full
    turning circle), it must be > 3x worse than every other link, else the episode
    is skipped. The row nobody points to is the first. Ambiguous episodes (e.g. a
    stopped ship with identical rows) are skipped, never mis-ordered.

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
        # k=2: a row is never its own successor (slow ships predict near themselves).
        dist2, succ2 = cKDTree(here).query(nxt, k=2)
        self_hit = succ2[:, 0] == np.arange(len(rows))
        dist = np.where(self_hit, dist2[:, 1], dist2[:, 0])
        succ = np.where(self_hit, succ2[:, 1], succ2[:, 0])
        has_succ = dist <= MATCH_TOL
        # n rows have n - 1 links: the worst-matching row is the last one.
        i_end = int(np.argmax(dist))
        if dist[i_end] <= MATCH_TOL:
            # Closed track: cut only where the worst link is clearly worse, else skip.
            second = float(np.partition(dist, -2)[-2])
            if not dist[i_end] > 3.0 * second:
                n_bad += 1
                continue
        has_succ[i_end] = False
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


def steps_from_times(t: np.ndarray) -> np.ndarray:
    """
    Step indices of t_episode values [s] (-1 stays -1, i.e. not reconstructed).
    """

    t = np.asarray(t, dtype=np.float64)
    return np.where(t >= 0, np.rint(t / SURROGATE_DT), -1).astype(np.int64)


def step_threshold(seconds: float) -> int:
    """
    First step index at or after `seconds` [s] (e.g. 0.7 s -> step 7, 0.75 s ->
    step 8), robust against float rounding of `seconds`.
    """

    return int(np.ceil(float(seconds) / SURROGATE_DT - 1e-6))


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
    """
    Write `t_episode` into the .npz at `path` (atomically; other arrays unchanged).
    """

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
    """
    Command-line entry point; returns the exit code.
    """

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
