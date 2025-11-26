#!/usr/bin/env python3
"""
generate_ship_data.py

Example CLI script for generating supervised-learning datasets
using the standalone dataset generator + any ship simulator
(Python class implementing reset(), step(), and .state).
"""

import argparse
import numpy as np
import os

from sim_dataset_generator import (
    collect_supervised_dataset,
)


# ================================================================
# ENV FACTORY
# ================================================================
def make_env():
    """
    Your environment factory.
    Replace PySimEnv with another sim implementation if desired.
    """
    from py_sim_env import PySimEnv
    return PySimEnv(render_mode=None)


# ================================================================
# TRAIN/VAL/TEST SPLITTING
# ================================================================
def split_and_save(X, Y, save_prefix, train=0.8, val=0.1):
    """
    Split dataset into train/val/test and save as .npz.
    """
    N = X.shape[0]
    idx = np.random.permutation(N)

    N_train = int(train * N)
    N_val = int(val * N)
    N_test = N - N_train - N_val

    idx_train = idx[:N_train]
    idx_val = idx[N_train:N_train + N_val]
    idx_test = idx[N_train + N_val:]

    np.savez(save_prefix + "_train.npz", X=X[idx_train], Y=Y[idx_train])
    np.savez(save_prefix + "_val.npz", X=X[idx_val], Y=Y[idx_val])
    np.savez(save_prefix + "_test.npz", X=X[idx_test], Y=Y[idx_test])

    print(f"Saved: {save_prefix}_train.npz  ({len(idx_train)} samples)")
    print(f"Saved: {save_prefix}_val.npz    ({len(idx_val)} samples)")
    print(f"Saved: {save_prefix}_test.npz   ({len(idx_test)} samples)")


# ================================================================
# MAIN CLI ENTRYPOINT
# ================================================================
def main():
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
                        default=3.0,
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
    nbr_samples = 0
    X = None
    Y = None
    if args.mode == "one_step":
        print(f"[One_step] Generating {args.samples} samples (max. horizon={args.horizon}s)...")
        X, Y, nbr_samples = collect_supervised_dataset(
            N_samples=args.samples,
            horizon_seconds=args.horizon,
            output_file=args.save,
        )

    print(f"Saved dataset to {args.save} ({nbr_samples} samples)")

    # ============================================================
    # Optional splitting
    # ============================================================
    if args.split:
        prefix = os.path.splitext(args.save)[0]
        split_and_save(X, Y, prefix)


if __name__ == "__main__":
    main()
