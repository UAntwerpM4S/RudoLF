"""
Surrogate error binned by any raw input column of a dataset split.

Question it answers: does the error (or the under-prediction) depend on one
condition -- wind speed, current speed, surge, UKC, rudder, ... ? E.g. if the
yaw-rate gain drops with wind speed, the wind's effect on the ship is not
fully explained by the model's inputs.

Per bin and per target it reports the MAE in physical units, the MAE
relative to the target's spread (MAE / std of the target over all rows), the
gain sum(pred*true)/sum(true^2) (1 = right size, < 1 = predictions too small)
and the number of rows.

Usage (from src):

    python -m fw.surrogate.validation.column_error_diagnostic ^
        --checkpoint myzako_full_A80.pth ^
        --data data\\myzako_full\\ship_dynamics_dataset_test.npz --by wind_speed

    --by     raw column name of the data's scenario (e.g. wind_speed,
             current_speed, surge, ukc, rudder_action, thrust_action)
    --bins   bin edges; default: the deciles of that column
    --abs    bin on the absolute value (e.g. --by rudder_action --abs)
    --min-time  only rows at least this long after the episode start [s]
             (needs the file's t_episode array, see fw.surrogate.episode_time)
    --where COL LO HI   only rows with LO <= COL <= HI (repeatable), e.g.
             --where rudder_action -0.03 0.03  (straight rudder only)
             --where current_speed 0 0.001     (no current)

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
from fw.surrogate.episode_time import T_EPISODE_KEY
from fw.surrogate.presets import resolve_checkpoint
from fw.surrogate.schema import TARGET_NAMES, check_projection, get_scenario, project_raw_columns

logger = logging.getLogger(__name__)

BATCH = 65536


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; returns the exit code."""

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Checkpoint (relative = in fw/checkpoints).")
    p.add_argument("--data", type=Path, required=True, help="Split to evaluate (e.g. the _test.npz).")
    p.add_argument("--by", required=True, help="Raw column to bin on (e.g. wind_speed).")
    p.add_argument("--bins", type=float, nargs="+", default=None, help="Bin edges (default: deciles).")
    p.add_argument("--abs", action="store_true", help="Bin on the absolute value of the column.")
    p.add_argument("--min-time", type=float, default=None,
                   help="Only rows at least this many seconds after their episode start (needs t_episode).")
    p.add_argument("--where", nargs=3, action="append", default=[], metavar=("COL", "LO", "HI"),
                   help="Only rows with LO <= COL <= HI (raw column; repeatable).")
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
    if args.by not in source.raw_columns:
        logger.error("Column '%s' is not in the %s data. Columns: %s", args.by, source.name,
                     ", ".join(source.raw_columns))
        return 2
    spread = y.std(axis=0)          # over ALL rows of the split, before any filter
    for col, lo, hi in args.where:
        if col not in source.raw_columns:
            logger.error("--where column '%s' is not in the %s data.", col, source.name)
            return 2
    keep = np.ones(len(y), dtype=bool)
    for col, lo, hi in args.where:
        v = x_raw[:, source.col(col)]
        keep &= (v >= float(lo)) & (v <= float(hi))

    if args.min_time is not None:
        with np.load(args.data, allow_pickle=False) as z:
            if T_EPISODE_KEY not in z.files:
                logger.error("%s has no '%s' array: run 'python -m fw.surrogate.episode_time %s' first.",
                             args.data, T_EPISODE_KEY, args.data)
                return 2
            t = z[T_EPISODE_KEY]
        keep &= (t >= 0) & (t >= args.min_time - 1e-6)
    if args.where or args.min_time is not None:
        x_raw, y = x_raw[keep], y[keep]
        logger.info("Rows after the filters (%s): %d.",
                    ", ".join([f"{lo} <= {c} <= {hi}" for c, lo, hi in args.where]
                              + ([f"t_episode >= {args.min_time:g} s"] if args.min_time is not None else [])),
                    len(y))
        if len(y) == 0:
            logger.error("No rows left after the filters.")
            return 2

    check_projection(source, predictor.scenario)
    x_model = project_raw_columns(x_raw, source, predictor.scenario)
    pred = np.concatenate([predictor.predict(x_model[i:i + BATCH]) for i in range(0, len(x_model), BATCH)])

    value = x_raw[:, source.col(args.by)].astype(np.float64)
    if args.abs:
        value = np.abs(value)
    if args.bins is None:
        edges = np.unique(np.quantile(value, np.linspace(0.0, 1.0, 11)))
    else:
        edges = np.array(sorted(set(args.bins)), dtype=np.float64)
    if len(edges) < 2:
        logger.error("Need at least two distinct bin edges (column '%s' is constant?).", args.by)
        return 2
    edges[-1] = np.nextafter(edges[-1], np.inf)        # include the maximum in the last bin
    which = np.digitize(value, edges) - 1

    name = f"|{args.by}|" if args.abs else args.by
    logger.info("Checkpoint %s (%s, %s) on %s: %d rows, binned by %s. MAE in physical units; "
                "rel = MAE / std of the target.", ckpt.name, predictor.ship_model, predictor.scenario.name,
                args.data, len(y), name)
    outside = int(((which < 0) | (which >= len(edges) - 1)).sum())
    if outside:
        logger.info("%d rows fall outside the bins and are not shown.", outside)
    logger.info("")
    logger.info("  %-21s %9s  %s", name, "n", "   ".join(f"{t:>30}" for t in TARGET_NAMES))
    logger.info("  %-21s %9s  %s", "", "", "   ".join(f"{'MAE':>10} {'rel':>8} {'gain':>10}" for _ in TARGET_NAMES))
    for b in range(len(edges) - 1):
        m = which == b
        label = f"{edges[b]:.4g} - {edges[b + 1]:.4g}"
        if m.sum() < 20:
            if m.any():
                logger.info("  %-21s %9d   (too few rows)", label, int(m.sum()))
            continue
        cells = []
        for k in range(len(TARGET_NAMES)):
            tr, pr = y[m, k], pred[m, k]
            mae = float(np.abs(pr - tr).mean())
            tt = float(np.sum(tr ** 2))
            gain = float(np.sum(pr * tr) / tt) if tt > 0 else float("nan")
            cells.append(f"{mae:10.3e} {mae / spread[k]:8.3f} {gain:10.3f}")
        logger.info("  %-21s %9d  %s", label, int(m.sum()), "   ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
