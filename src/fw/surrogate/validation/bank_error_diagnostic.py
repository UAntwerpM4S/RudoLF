"""
Surrogate error vs. distance to the bank, on a dataset split.

Shows whether a confined/full surrogate under-predicts the bank effect close
to the bank, and whether that is a data problem (few samples there) or a
model/loss problem (enough samples, but shrunk predictions).

The bank distance per row comes from its ray features,
d = RAY_MAX_DISTANCE * (1 - feature) (RAY_MAX_DISTANCE if no bank in range).
By default only beam rays count (60-120 and 240-300 deg from the bow, a bank
alongside as in the model_comparison ship-bank test); --rays all uses all 24.
Per distance bin and target it reports:

    n        rows in the bin (and in the training split, with --train)
    MAE      mean |pred - true|
    bias     mean (pred - true)
    gain     sum(pred * true) / sum(true^2): least-squares scale of the
             prediction (< 1 = predictions shrunk towards zero)
    R2       1 - SSE / SST within the bin
    |true| p99, |pred| p99   size of the tails

Bank suction acts in a direction that depends on the bank side, so bias
mostly cancels; gain and p99 show under-prediction regardless of side.
A second table covers rows with a near-centred rudder
(|rudder_action| <= --straight-rudder), where the bank dominates sway and yaw.

Usage (from src):

    python -m fw.surrogate.validation.bank_error_diagnostic ^
        --checkpoint best_surrogate_myzako_confined.pth ^
        --data data\\myzako_confined\\ship_dynamics_dataset_test.npz ^
        --train data\\myzako_confined\\ship_dynamics_dataset_train.npz

--min-time / --max-time keep only rows in that time window since episode start
[s] (from t_episode or reconstructed as in fw.surrogate.episode_time; episodes
that cannot be ordered are dropped). This separates the bank effect from the
start transient, since bank-straight episodes start next to the bank.

A relative --checkpoint is resolved in fw/checkpoints. Writes
<checkpoint stem>_bank_error.png next to --data unless --plot is given.
"""

from __future__ import annotations

import sys
import logging
import argparse
import numpy as np

from pathlib import Path
from fw.surrogate.dataset import load_raw
from typing import Dict, Optional, Sequence
from fw.surrogate.presets import resolve_checkpoint
from fw.environments.ray_features import RAY_ANGLES_DEG, RAY_MAX_DISTANCE
from fw.surrogate.schema import RAY_COLUMN_NAMES, TARGET_NAMES, ScenarioSchema

logger = logging.getLogger(__name__)

DEFAULT_BINS = (0.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 199.999, RAY_MAX_DISTANCE + 1e-6)
BATCH = 65536


BEAM_RAY_SECTORS_DEG = ((60, 120), (240, 300))


def bank_distance(x_raw: np.ndarray, scenario: ScenarioSchema, rays: str = "beam") -> np.ndarray:
    """
    Distance [m] to the nearest bank per row, over the beam rays ("beam") or all rays ("all").
    """

    names = RAY_COLUMN_NAMES
    if rays == "beam":
        names = [n for n, a in zip(RAY_COLUMN_NAMES, RAY_ANGLES_DEG)
                 if any(lo <= a <= hi for lo, hi in BEAM_RAY_SECTORS_DEG)]
    elif rays != "all":
        raise ValueError(f"rays must be 'beam' or 'all', got {rays!r}.")
    idx = [scenario.col(name) for name in names]
    features = np.clip(x_raw[:, idx], 0.0, 1.0)
    return RAY_MAX_DISTANCE * (1.0 - features.max(axis=1))


def predict(predictor, x_raw: np.ndarray) -> np.ndarray:
    """
    Predictions for all rows, in batches.
    """

    return np.concatenate([predictor.predict(x_raw[i:i + BATCH]) for i in range(0, len(x_raw), BATCH)])


def bin_stats(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    MAE, bias, gain, R2 and the p99 of |true| and |pred| of one bin and target.
    """

    err = pred - true
    sst = float(np.sum((true - true.mean()) ** 2))
    tt = float(np.sum(true ** 2))
    return {
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "gain": float(np.sum(pred * true) / tt) if tt > 0 else float("nan"),
        "r2": 1.0 - float(np.sum(err ** 2)) / sst if sst > 0 else float("nan"),
        "true_p99": float(np.percentile(np.abs(true), 99)),
        "pred_p99": float(np.percentile(np.abs(pred), 99)),
    }


def _label(lo: float, hi: float) -> str:
    """
    Bin label; the last bin (no bank within ray range) is shown as 'no bank'.
    """

    if lo >= RAY_MAX_DISTANCE - 1e-3:
        return "no bank"
    return f"{lo:g}-{min(hi, RAY_MAX_DISTANCE):.0f} m"


def report(title: str, d: np.ndarray, true: np.ndarray, pred: np.ndarray, bins: Sequence[float],
           d_train: Optional[np.ndarray]) -> Dict[str, list]:
    """
    Log one table (all targets, per distance bin); return the gains per target for the plot.
    """

    which = np.digitize(d, bins) - 1
    counts_train = None if d_train is None else np.bincount(np.digitize(d_train, bins) - 1, minlength=len(bins))
    logger.info("")
    logger.info("=== %s (%d rows)", title, len(d))
    gains: Dict[str, list] = {name: [] for name in TARGET_NAMES}
    for t, name in enumerate(TARGET_NAMES):
        logger.info("  %s", name)
        logger.info("    %-12s %9s %10s %10s %11s %7s %7s %11s %11s",
                    "bank dist", "n", "n train", "MAE", "bias", "gain", "R2", "|true| p99", "|pred| p99")
        for b in range(len(bins) - 1):
            m = which == b
            n_tr = "-" if counts_train is None else f"{int(counts_train[b]):d}"
            if m.sum() < 20:
                logger.info("    %-12s %9d %10s   (too few rows)", _label(bins[b], bins[b + 1]), int(m.sum()), n_tr)
                gains[name].append(float("nan"))
                continue
            s = bin_stats(true[m, t], pred[m, t])
            gains[name].append(s["gain"])
            logger.info("    %-12s %9d %10s %10.3e %11.3e %7.3f %7.3f %11.3e %11.3e",
                        _label(bins[b], bins[b + 1]), int(m.sum()), n_tr, s["mae"], s["bias"], s["gain"], s["r2"],
                        s["true_p99"], s["pred_p99"])
    return gains


def plot(path: Path, bins: Sequence[float], gains_all: Dict[str, list], gains_straight: Dict[str, list],
         counts: np.ndarray, title: str) -> None:
    """
    Gain per distance bin (all rows and straight rudder) and the row counts, as a PNG.
    """

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed: no plot written.")
        return
    labels = [_label(bins[b], bins[b + 1]) for b in range(len(bins) - 1)]
    pos = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, ls in zip(TARGET_NAMES, ("-", "--", "-.")):
        axes[0].plot(pos, gains_all[name], ls, marker="o", label=f"{name} (all)")
        axes[0].plot(pos, gains_straight[name], ls, marker="x", alpha=0.6, label=f"{name} (straight rudder)")
    axes[0].axhline(1.0, color="grey", lw=0.8)
    axes[0].set_xticks(pos, labels, rotation=30)
    axes[0].set_ylabel("gain = sum(pred*true) / sum(true^2)")
    axes[0].set_title("Prediction scale vs. distance to bank (1 = right size)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].bar(pos, counts[:len(labels)])
    axes[1].set_xticks(pos, labels, rotation=30)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("rows in the evaluated split")
    axes[1].set_title("Rows per bin")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("")
    logger.info("Wrote %s", path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Command-line entry point; returns the exit code.
    """

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Checkpoint (relative = in fw/checkpoints).")
    p.add_argument("--data", type=Path, required=True, help="Split to evaluate (e.g. the _test.npz).")
    p.add_argument("--train", type=Path, default=None,
                   help="Optional training split: also show its rows per bin (loads the whole file).")
    p.add_argument("--bins", type=float, nargs="+", default=None,
                   help=f"Distance bin edges [m] (default {list(DEFAULT_BINS[:-2])} + 'no bank').")
    p.add_argument("--straight-rudder", type=float, default=0.05,
                   help="|rudder_action| limit of the straight-rudder table (default 0.05).")
    p.add_argument("--rays", choices=("beam", "all"), default="beam",
                   help="Rays that define the bank distance: beam (60-120/240-300 deg, default) or all.")
    p.add_argument("--min-time", type=float, default=None,
                   help="Keep only rows at least this long [s] after the start of their episode.")
    p.add_argument("--max-time", type=float, default=None,
                   help="Keep only rows less than this long [s] after the start of their episode.")
    p.add_argument("--plot", type=Path, default=None, help="PNG path (default: next to --data).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from fw.surrogate.inference_surrogate import SurrogatePredictor  # torch only needed here

    ckpt = resolve_checkpoint(args.checkpoint)
    predictor = SurrogatePredictor(str(ckpt))
    scenario = predictor.scenario
    if not scenario.includes_rays:
        logger.error("Checkpoint %s is a %s model: no ray features, no bank distance.", ckpt, scenario.name)
        return 2

    x_raw, y, identity = load_raw(args.data, scenario)
    mismatch = predictor.data_mismatch(identity)
    if mismatch:
        logger.error("%s: %s.", args.data, mismatch)
        return 2
    if args.bins is None:
        bins = list(DEFAULT_BINS)
    else:
        # User edges below the "no bank" bin, starting at 0, strictly increasing.
        user = sorted({float(b) for b in args.bins if 0.0 < float(b) < 199.999})
        bins = [0.0] + user + [199.999, RAY_MAX_DISTANCE + 1e-6]

    title_extra = ""
    if args.min_time is not None or args.max_time is not None:
        from fw.surrogate.episode_time import (STATE_COLUMNS, T_EPISODE_KEY, step_index, step_threshold,
                                               steps_from_times)
        with np.load(args.data) as z:
            episode_id = z["episode_id"].astype(np.int64)
            stored = z[T_EPISODE_KEY] if T_EPISODE_KEY in z.files else None
        # Compare in whole steps (stored float32 or reconstructed times alike):
        # [min_time, max_time) maps to [step_threshold(min), step_threshold(max)).
        if stored is not None:      # written by fw.surrogate.episode_time
            steps = steps_from_times(stored)
            n_ok, n_bad = len(np.unique(episode_id[steps >= 0])), len(np.unique(episode_id[steps < 0]))
        else:
            state = x_raw[:, [scenario.col(c) for c in STATE_COLUMNS]]
            steps, n_ok, n_bad = step_index(state, y, episode_id)
        keep = steps >= 0
        if args.min_time is not None:
            keep &= steps >= step_threshold(args.min_time)
        if args.max_time is not None:
            keep &= steps < step_threshold(args.max_time)
        logger.info("Time filter [%s, %s) s: %d episodes ordered, %d skipped; keeping %d of %d rows.",
                    args.min_time, args.max_time, n_ok, n_bad, int(keep.sum()), len(y))
        x_raw, y = x_raw[keep], y[keep]
        if len(y) == 0:
            logger.error("No rows left after the time filter.")
            return 2
        title_extra = f", t in [{args.min_time if args.min_time is not None else 0:g}, " \
                      f"{args.max_time if args.max_time is not None else 'end'}) s"

    d = bank_distance(x_raw, scenario, args.rays)
    d_train = None
    if args.train is not None:
        x_tr, _, _ = load_raw(args.train, scenario)
        d_train = bank_distance(x_tr, scenario, args.rays)
        del x_tr
    pred = predict(predictor, x_raw)
    logger.info("Checkpoint %s (%s, %s) on %s: %d rows; bank distance from the %s rays.", ckpt.name,
                predictor.ship_model, scenario.name, args.data, len(y), args.rays)

    gains_all = report("All rows", d, y, pred, bins, d_train)
    straight = np.abs(x_raw[:, scenario.col("rudder_action")]) <= args.straight_rudder
    gains_straight = report(f"Straight rudder (|rudder_action| <= {args.straight_rudder:g})",
                            d[straight], y[straight], pred[straight], bins, None)

    counts = np.bincount(np.digitize(d, bins) - 1, minlength=len(bins))
    out = args.plot or args.data.with_name(f"{ckpt.stem}_bank_error.png")
    plot(out, bins, gains_all, gains_straight, counts, f"{ckpt.name} on {args.data.name}{title_extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
