"""
Surrogate error vs. under-keel clearance (UKC), on bank-free rows of a split.

Question it answers: how much accuracy does a model WITHOUT a UKC input
(open_water, "clean") lose against one WITH it (confined_ukc), and at which
water depths? On rows without any bank in ray range the confined model's
rays are all zero, so the only information it has that the clean model lacks
is the UKC: the difference between the two is the effect of water depth.

Rows of the split are filtered on the nearest bank (from their own ray
features, all rays, like train_surrogate --min-bank-distance) and binned by
their UKC. Per bin and per target, for every checkpoint:

    MAE     mean |pred - true|
    gain    sum(pred * true) / sum(true^2) (1 = right size, < 1 = too small)
    R2      1 - SSE / SST within the bin

Usage (from src):

    python -m fw.surrogate.validation.ukc_error_diagnostic ^
        --checkpoint best_surrogate_myzako_clean.pth best_surrogate_myzako_confined.pth ^
        --data data\\myzako_confined\\ship_dynamics_dataset_test.npz

Relative --checkpoint paths are resolved like the presets do (fw/checkpoints).
Writes ukc_error.png next to --data unless --plot is given.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from pathlib import Path
from typing import Optional, Sequence

from fw.environments.ray_features import RAY_MAX_DISTANCE
from fw.surrogate.dataset import load_raw
from fw.surrogate.presets import resolve_checkpoint
from fw.surrogate.schema import RAY_COLUMN_NAMES, TARGET_NAMES, check_projection, get_scenario, project_raw_columns

logger = logging.getLogger(__name__)

DEFAULT_UKC_BINS = (0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
BATCH = 65536


def predict(predictor, x_raw: np.ndarray) -> np.ndarray:
    """Predictions for all rows, in batches."""

    return np.concatenate([predictor.predict(x_raw[i:i + BATCH]) for i in range(0, len(x_raw), BATCH)])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; returns the exit code."""

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", required=True,
                   help="One or more checkpoints (relative = in fw/checkpoints), e.g. clean and confined.")
    p.add_argument("--data", type=Path, required=True,
                   help="Split to evaluate; must have rays and UKC (confined_ukc or full_env data).")
    p.add_argument("--min-bank-distance", type=float, default=RAY_MAX_DISTANCE,
                   help=f"Keep rows whose nearest bank is at least this far [m] (default {RAY_MAX_DISTANCE:g} = "
                        "no bank in ray range); 0 = all rows.")
    p.add_argument("--bins", type=float, nargs="+", default=list(DEFAULT_UKC_BINS),
                   help=f"UKC bin edges [m] (default {list(DEFAULT_UKC_BINS)}).")
    p.add_argument("--plot", type=Path, default=None, help="PNG path (default: ukc_error.png next to --data).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from fw.surrogate.inference_surrogate import SurrogatePredictor  # torch only needed here

    x_raw, y, identity = load_raw(args.data)
    source = get_scenario(identity.scenario)
    if not (source.includes_rays and source.includes_ukc):
        logger.error("%s is %s data: it needs ray and UKC columns.", args.data, source.name)
        return 2
    rays = np.clip(x_raw[:, [source.col(c) for c in RAY_COLUMN_NAMES]], 0.0, 1.0)
    keep = RAY_MAX_DISTANCE * (1.0 - rays.max(axis=1)) >= args.min_bank_distance
    x_raw, y = x_raw[keep], y[keep]
    if len(y) == 0:
        logger.error("%s: no rows with no bank within %g m (lower --min-bank-distance).",
                     args.data, args.min_bank_distance)
        return 2
    ukc = x_raw[:, source.col("ukc")]
    logger.info("%s: %d rows with no bank within %g m (%.1f %%); UKC %.2f-%.2f m, median %.2f m.",
                args.data, len(y), args.min_bank_distance, 100.0 * keep.mean(), float(ukc.min()),
                float(ukc.max()), float(np.median(ukc)))

    bins = sorted(args.bins)
    which = np.digitize(ukc, bins) - 1
    labels = [f"{bins[b]:g}-{bins[b + 1]:g} m" for b in range(len(bins) - 1)]
    counts = [int((which == b).sum()) for b in range(len(labels))]
    outside = len(y) - sum(counts)
    if outside:
        logger.warning("%d rows have a UKC outside the bins %s and are not shown.", outside, bins)

    results = {}   # name -> (len(labels), 3) MAE array
    for ckpt_arg in args.checkpoint:
        ckpt = resolve_checkpoint(ckpt_arg)
        predictor = SurrogatePredictor(str(ckpt))
        if predictor.ship_model != identity.ship_model:
            logger.error("%s is a %s model, the data is %s.", ckpt.name, predictor.ship_model, identity.ship_model)
            return 2
        check_projection(source, predictor.scenario)
        pred = predict(predictor, project_raw_columns(x_raw, source, predictor.scenario))
        mae = np.full((len(labels), len(TARGET_NAMES)), np.nan)
        logger.info("")
        logger.info("=== %s (%s)", ckpt.name, predictor.scenario.name)
        logger.info("  %-10s %9s  %s", "UKC", "n", "   ".join(f"{t:>33}" for t in TARGET_NAMES))
        logger.info("  %-10s %9s  %s", "", "", "   ".join(f"{'MAE':>11} {'gain':>9} {'R2':>11}" for _ in TARGET_NAMES))
        for b, label in enumerate(labels):
            m = which == b
            if m.sum() < 20:
                logger.info("  %-10s %9d   (too few rows)", label, int(m.sum()))
                continue
            cells = []
            for t in range(len(TARGET_NAMES)):
                tr, pr = y[m, t], pred[m, t]
                err = pr - tr
                tt, sst = float(np.sum(tr ** 2)), float(np.sum((tr - tr.mean()) ** 2))
                mae[b, t] = float(np.abs(err).mean())
                gain = float(np.sum(pr * tr) / tt) if tt > 0 else float("nan")
                r2 = 1.0 - float(np.sum(err ** 2)) / sst if sst > 0 else float("nan")
                cells.append(f"{mae[b, t]:11.3e} {gain:9.3f} {r2:11.3f}")
            logger.info("  %-10s %9d  %s", label, int(m.sum()), "   ".join(cells))
        results[ckpt.name] = mae

    out = args.plot or args.data.with_name("ukc_error.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed: no plot written.")
        return 0
    pos = np.arange(len(labels))
    fig, axes = plt.subplots(1, len(TARGET_NAMES) + 1, figsize=(5 * (len(TARGET_NAMES) + 1), 4.5))
    for t, name in enumerate(TARGET_NAMES):
        for ckpt_name, mae in results.items():
            axes[t].plot(pos, mae[:, t], marker="o", label=ckpt_name)
        axes[t].set_xticks(pos, labels, rotation=35)
        axes[t].set_title(f"MAE {name}")
        axes[t].set_xlabel("UKC")
        axes[t].grid(alpha=0.3)
        axes[t].legend(fontsize=7)
    axes[-1].bar(pos, counts)
    axes[-1].set_xticks(pos, labels, rotation=35)
    axes[-1].set_yscale("log")
    axes[-1].set_title("Rows per UKC bin")
    axes[-1].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Error vs. UKC on {args.data.name} (no bank within {args.min_bank_distance:g} m)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
