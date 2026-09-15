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
def split_and_save(x_vals, y_vals, episode_ids, save_prefix, train=0.8, val=0.1):
    """
    Split dataset into train/val/test BY EPISODE and save as .npz.

    Splitting is done on unique `episode_ids`, not on individual rows: all
    rows belonging to one simulation rollout are assigned to the same
    split. Splitting by row instead (the previous behavior) leaks
    temporally-correlated, near-duplicate rows from the same episode
    across the train/val/test boundary, which makes validation/test
    metrics optimistically biased and doesn't test generalization to
    genuinely unseen trajectories.

    Because episodes have variable length (a rollout can terminate early,
    e.g. leaving open water or insufficient keel clearance), the
    resulting ROW counts per split will only approximately match
    `train`/`val`/`1-train-val` — the fractions are exact in terms of
    episode count, not row count.

    Args:
        x_vals: Input array, shape (N, 37).
        y_vals: Output array, shape (N, 3).
        episode_ids: Per-row episode index, shape (N,), aligned with
            x_vals/y_vals.
        save_prefix: Path prefix for the three output .npz files.
        train: Fraction of EPISODES assigned to the train split.
        val: Fraction of EPISODES assigned to the val split (remainder
            goes to test).
    """

    unique_episodes = np.unique(episode_ids)
    num_episodes = len(unique_episodes)

    perm = np.random.permutation(num_episodes)
    shuffled_episodes = unique_episodes[perm]

    n_train = int(train * num_episodes)
    n_val = int(val * num_episodes)

    train_episodes = shuffled_episodes[:n_train]
    val_episodes = shuffled_episodes[n_train:n_train + n_val]
    test_episodes = shuffled_episodes[n_train + n_val:]

    train_mask = np.isin(episode_ids, train_episodes)
    val_mask = np.isin(episode_ids, val_episodes)
    test_mask = np.isin(episode_ids, test_episodes)

    np.savez(save_prefix + "_train.npz", X=x_vals[train_mask], Y=y_vals[train_mask])
    np.savez(save_prefix + "_val.npz", X=x_vals[val_mask], Y=y_vals[val_mask])
    np.savez(save_prefix + "_test.npz", X=x_vals[test_mask], Y=y_vals[test_mask])

    print(f"Saved: {save_prefix}_train.npz  ({train_mask.sum()} rows from {len(train_episodes)} episodes)")
    print(f"Saved: {save_prefix}_val.npz    ({val_mask.sum()} rows from {len(val_episodes)} episodes)")
    print(f"Saved: {save_prefix}_test.npz   ({test_mask.sum()} rows from {len(test_episodes)} episodes)")


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
                        default=4,
                        help="Number of parallel simulation worker processes. "
                             "Kept at collect_supervised_dataset's own conservative "
                             "default (4) rather than defaulting to all CPU cores: "
                             "each simulation is a heavy per-process physics/math-model "
                             "run, and oversubscribing cores tends to cause simulations "
                             "to exceed sim_config.timeout_seconds and fail with TIMEOUT "
                             "under load. Raise this deliberately once you've confirmed "
                             "your machine can sustain more concurrent simulations "
                             "within the timeout.")

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

        x_values, y_values, episode_ids, nbr_samples = collect_supervised_dataset(
            nbr_samples=args.samples,
            horizon_seconds=args.horizon,
            output_file=args.save,
            workers=args.workers,
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
        split_and_save(x_values, y_values, episode_ids, prefix)


if __name__ == "__main__":
    main()
