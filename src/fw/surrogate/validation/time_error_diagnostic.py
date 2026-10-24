"""
Surrogate error vs. time since the start of the episode, on a dataset split.

Question it answers: is the error concentrated in the first seconds of the
episodes? Every episode starts from a randomly set state with the commanded
rudder/azipod angle and thrust; if the simulator's actual actuator state
(e.g. an azipod still slewing to its commanded direction, a propeller still
spinning up) lags behind the command, the surrogate -- which only sees the
command -- cannot explain those first steps, and its error is large there
and small later on.

The rows of a split are shuffled and carry no step index. The time since
episode start is taken from the file's t_episode array when present (see
fw.surrogate.episode_time), otherwise reconstructed the same way on the fly
(episode_time.step_index). Episodes whose order cannot be reconstructed are
skipped (and counted).

Per time bin and target it reports the MAE in physical units, the MAE
relative to the target's spread (MAE / std of the target over all rows), the
gain sum(pred*true)/sum(true^2) and the number of rows.

Usage (from src):

    python -m fw.surrogate.validation.time_error_diagnostic ^
        --checkpoint tripoli_confined_A80.pth ^
        --data data\\tripoli_confined\\ship_dynamics_dataset_test.npz

Relative --checkpoint paths are resolved like the presets do (fw/checkpoints).
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from pathlib import Path
from typing import Optional, Sequence

from fw.surrogate.dataset import load_raw
from fw.surrogate.episode_time import STATE_COLUMNS, T_EPISODE_KEY, step_index  # noqa: F401 (re-export)
from fw.surrogate.presets import resolve_checkpoint
from fw.surrogate.schema import SURROGATE_DT, TARGET_NAMES, check_projection, get_scenario, project_raw_columns

logger = logging.getLogger(__name__)

DEFAULT_TIME_BINS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 1e9)
BATCH = 65536


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; returns the exit code."""

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Checkpoint (relative = in fw/checkpoints).")
    p.add_argument("--data", type=Path, required=True, help="Split to evaluate (e.g. the _test.npz).")
    p.add_argument("--bins", type=float, nargs="+", default=None,
                   help=f"Time bin edges [s] (default {list(DEFAULT_TIME_BINS[:-1])} + rest).")
    p.add_argument("--split-thrust", action="store_true",
                   help="Also show the rows with thrust_action >= 0 and < 0 separately.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from fw.surrogate.inference_surrogate import SurrogatePredictor  # torch only needed here

    ckpt = resolve_checkpoint(args.checkpoint)
    predictor = SurrogatePredictor(str(ckpt))
    x_raw, y, identity = load_raw(args.data)
    source = get_scenario(identity.scenario)
    if identity.ship_model != predictor.ship_model:
        logger.error("%s is %s data, but the checkpoint is a %s model.", args.data, identity.ship_model,
                     predictor.ship_model)
        return 2
    with np.load(args.data) as z:
        episode_id = z["episode_id"].astype(np.int64)
        stored = z[T_EPISODE_KEY] if T_EPISODE_KEY in z.files else None

    if stored is not None:      # written by fw.surrogate.episode_time
        steps = np.where(stored >= 0, np.rint(stored / SURROGATE_DT), -1).astype(np.int64)
        logger.info("%s: %d rows; episode time from its %s array (%d rows without).",
                    args.data, len(y), T_EPISODE_KEY, int((steps < 0).sum()))
    else:
        state = x_raw[:, [source.col(c) for c in STATE_COLUMNS]]
        steps, n_ok, n_bad = step_index(state, y, episode_id)
        logger.info("%s: %d rows, %d episodes ordered, %d skipped (chain not reconstructable).",
                    args.data, len(y), n_ok, n_bad)
    ok = steps >= 0
    if not ok.any():
        logger.error("No episode could be ordered.")
        return 2

    check_projection(source, predictor.scenario)
    x_model = project_raw_columns(x_raw[ok], source, predictor.scenario)
    pred = np.concatenate([predictor.predict(x_model[i:i + BATCH]) for i in range(0, len(x_model), BATCH)])
    true, t = y[ok], steps[ok] * SURROGATE_DT
    spread = y.std(axis=0)

    if args.bins is None:
        bins = list(DEFAULT_TIME_BINS)
    else:   # from 0, strictly increasing, plus an open last bin
        bins = [0.0] + sorted({float(v) for v in args.bins if float(v) > 0.0}) + [1e9]
    which = np.digitize(t, bins) - 1
    logger.info("Checkpoint %s (%s, %s). MAE in physical units; rel = MAE / std of the target.",
                ckpt.name, predictor.ship_model, predictor.scenario.name)
    thrust = x_raw[ok][:, source.col("thrust_action")]
    groups = [("all rows", np.ones(len(t), dtype=bool))]
    if args.split_thrust:
        groups += [("thrust_action >= 0", thrust >= 0), ("thrust_action < 0", thrust < 0)]
    for title, sel in groups:
        logger.info("")
        logger.info("=== %s (%d rows)", title, int(sel.sum()))
        header = "   ".join(f"{name:>30}" for name in TARGET_NAMES)
        logger.info("  %-12s %9s  %s", "time [s]", "n", header)
        logger.info("  %-12s %9s  %s", "", "", "   ".join(f"{'MAE':>10} {'rel':>8} {'gain':>10}" for _ in TARGET_NAMES))
        for b in range(len(bins) - 1):
            m = (which == b) & sel
            label = f"{bins[b]:g}-{bins[b + 1]:g}" if bins[b + 1] < 1e8 else f">= {bins[b]:g}"
            if m.sum() < 20:
                if m.any():
                    logger.info("  %-12s %9d   (too few rows)", label, int(m.sum()))
                continue
            cells = []
            for k in range(len(TARGET_NAMES)):
                tr, pr = true[m, k], pred[m, k]
                mae = float(np.abs(pr - tr).mean())
                tt = float(np.sum(tr ** 2))
                gain = float(np.sum(pr * tr) / tt) if tt > 0 else float("nan")
                cells.append(f"{mae:10.3e} {mae / spread[k]:8.3f} {gain:10.3f}")
            logger.info("  %-12s %9d  %s", label, int(m.sum()), "   ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
