#!/usr/bin/env python3
"""
generate_ship_data.py

Example CLI script for generating supervised-learning datasets
using the standalone dataset generator + any ship simulator
(Python class implementing reset(), step(), and .state).
"""

import os
import argparse
import numpy as np

from fw.surrogate.sim_fh_dataset_generator import collect_supervised_dataset


# ================================================================
# TRAIN/VAL/TEST SPLITTING
# ================================================================
def split_and_save(x_vals, y_vals, save_prefix, train=0.8, val=0.1):
    """
    Split dataset into train/val/test and save as .npz.
    """

    num = x_vals.shape[0]
    idx = np.random.permutation(num)

    n_train = int(train * num)
    n_val = int(val * num)

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    np.savez(save_prefix + "_train.npz", X=x_vals[idx_train], Y=y_vals[idx_train])
    np.savez(save_prefix + "_val.npz", X=x_vals[idx_val], Y=y_vals[idx_val])
    np.savez(save_prefix + "_test.npz", X=x_vals[idx_test], Y=y_vals[idx_test])

    print(f"Saved: {save_prefix}_train.npz  ({len(idx_train)} samples)")
    print(f"Saved: {save_prefix}_val.npz    ({len(idx_val)} samples)")
    print(f"Saved: {save_prefix}_test.npz   ({len(idx_test)} samples)")


# ================================================================
# MAIN CLI ENTRYPOINT
# ================================================================
def main():
    """
    Main.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        type=str,
                        default="one_step",
                        choices=["vectorized", "one_step", "parallel"],
                        help="Dataset generation mode")

    parser.add_argument("--samples",
                        type=int,
                        default=20000,
                        help="Number of samples")

    parser.add_argument("--horizon",
                        type=float,
                        default=80.0,
                        help="Rollout duration (seconds)")

    parser.add_argument("--save",
                        type=str,
                        # default="dataset.npz",
                        default="data/ship_dynamics_dataset.csv",
                        help="Output filename")

    parser.add_argument("--gpu",
                        action="store_true",
                        help="Enable GPU (if available)")

    parser.add_argument("--workers",
                        type=int,
                        default=None,
                        help="Number of parallel workers (parallel mode)")

    parser.add_argument("--split",
                        action="store_true",
                        default=True,
                        help="Split dataset into train/val/test sets")

    args = parser.parse_args()

    # ============================================================
    # Run selected mode
    # ============================================================

    if args.mode == "one_step":
        print(f"[one_step] Generating {args.samples} samples (max. horizon={args.horizon}s)...")

        x_values, y_values, nbr_samples = collect_supervised_dataset(
            nbr_samples=args.samples,
            horizon_seconds=args.horizon,
            output_file=args.save,
        )

    elif args.mode in ("vectorized", "parallel"):
        raise NotImplementedError(
            f"Mode '{args.mode}' is not yet implemented. "
            "Use --mode one_step (the only currently supported mode)."
        )

    else:
        # argparse choices= already guards this, but be explicit
        raise ValueError(f"Unknown mode: {args.mode}")

    if nbr_samples == 0:
        print("Warning: no samples were collected - dataset files were not written.")
        return

    print(f"Saved dataset to {args.save} ({nbr_samples} samples)")

    # ============================================================
    # Optional splitting
    # ============================================================
    if args.split:
        prefix = os.path.splitext(args.save)[0]
        split_and_save(x_values, y_values, prefix)


if __name__ == "__main__":
    main()
