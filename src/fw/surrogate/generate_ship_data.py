#!/usr/bin/env python3
"""
generate_ship_data.py

CLI for generating the surrogate training data with the FH dataset
generator (sim_fh_dataset_generator.collect_supervised_dataset): writes the
full CSV and, unless --no-split, train/val/test .npz files split by episode.

Each .npz carries its identity (schema.DatasetIdentity):
  * scenario       - from the generator's row width (NBR_X_FEATURES): 37 =
                     full_env, 33 = confined_ukc, 8 = open_water. The
                     generator is whatever is installed as
                     sim_fh_dataset_generator.py; --straight-fraction only
                     applies to the wind/current generators;
  * ray_layout     - ray_features.RAY_LAYOUT_VERSION;
  * ship_model     - --ship-model (default TRIPOLI);
  * sway_convention - schema.SWAY_CONVENTION (sway positive to port);
  * rudder_max_deg - the generator's RUDDER_MAX_DEG (rudder_action = +-1 <->
                     +-RUDDER_MAX_DEG). Must equal the ship's maximum rudder
                     angle in fw.ships.ship_factory, otherwise generation is
                     refused (a mismatch means the wrong generator or the
                     wrong --ship-model).
"""

import os
import inspect
import argparse
import numpy as np

# schema (numpy only), not dataset.py: data generation must not need torch.
from fw.environments.ray_features import RAY_LAYOUT_VERSION
from fw.ships.ship_factory import ShipModel, create_ship
from fw.surrogate.schema import SWAY_CONVENTION, DatasetIdentity, npz_metadata, scenario_for_raw_width
from fw.surrogate import sim_fh_dataset_generator as _generator
from fw.surrogate.sim_fh_dataset_generator import (
    BANK_START_BAND,
    BANK_START_FRACTION,
    NBR_X_FEATURES,
    RUDDER_MAX_DEG,
    collect_supervised_dataset,
)

# Only the wind/current generators have "straight" episodes; the others
# (Tripoli confined, Myzako) neither define STRAIGHT_FRACTION nor accept
# straight_fraction=. None = not supported by the installed generator.
STRAIGHT_FRACTION = getattr(_generator, "STRAIGHT_FRACTION", None)
_SUPPORTS_STRAIGHT = "straight_fraction" in inspect.signature(collect_supervised_dataset).parameters

# The generator's raw row width identifies the scenario it generates.
GENERATOR_SCENARIO = scenario_for_raw_width(NBR_X_FEATURES)


def dataset_identity(ship_model: ShipModel) -> DatasetIdentity:
    """
    Identity of the data the current generator produces for `ship_model`.

    Raises:
        ValueError: If the generator's RUDDER_MAX_DEG differs from the ship's
            maximum rudder angle in ship_factory.
    """

    ship_max_deg = float(np.degrees(create_ship(ship_model).specifications.max_rudder_angle))
    if not np.isclose(ship_max_deg, float(RUDDER_MAX_DEG)):
        raise ValueError(
            f"sim_fh_dataset_generator.RUDDER_MAX_DEG = {RUDDER_MAX_DEG} deg, but "
            f"{ship_model.name}'s max rudder angle in ship_factory is {ship_max_deg:g} deg. "
            f"Wrong generator for this ship, or wrong --ship-model?"
        )
    return DatasetIdentity(scenario=GENERATOR_SCENARIO.name, ray_layout=RAY_LAYOUT_VERSION,
                           ship_model=ship_model.name, rudder_max_deg=float(RUDDER_MAX_DEG),
                           sway_convention=SWAY_CONVENTION)


# ================================================================
# TRAIN/VAL/TEST SPLITTING
# ================================================================
def split_and_save(x_vals, y_vals, episode_ids, save_prefix, identity: DatasetIdentity,
                   train=0.8, val=0.1, seed=None):
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
    `train`/`val`/`1-train-val`, the fractions are exact in terms of
    episode count, not row count.

    Args:
        x_vals: Input array, shape (N, raw width of identity.scenario).
        y_vals: Output array, shape (N, 3).
        episode_ids: Per-row episode index, shape (N,), aligned with
            x_vals/y_vals.
        save_prefix: Path prefix for the three output .npz files.
        identity: Dataset identity stored in every .npz (see dataset_identity).
        train: Fraction of EPISODES assigned to the train split.
        val: Fraction of EPISODES assigned to the val split (remainder
            goes to test).
        seed: Optional seed for the episode shuffle, so the same data gives
            the same split. Each .npz also stores the per-row `episode_id`,
            so the split can be inspected or reproduced afterward, plus the
            identity (`schema.npz_metadata`), which dataset.load_raw checks.
    """

    unique_episodes = np.unique(episode_ids)
    num_episodes = len(unique_episodes)

    perm = np.random.default_rng(seed).permutation(num_episodes)
    shuffled_episodes = unique_episodes[perm]

    n_train = int(train * num_episodes)
    n_val = int(val * num_episodes)

    train_episodes = shuffled_episodes[:n_train]
    val_episodes = shuffled_episodes[n_train:n_train + n_val]
    test_episodes = shuffled_episodes[n_train + n_val:]

    train_mask = np.isin(episode_ids, train_episodes)
    val_mask = np.isin(episode_ids, val_episodes)
    test_mask = np.isin(episode_ids, test_episodes)

    for name, episodes in (("train", train_episodes), ("val", val_episodes), ("test", test_episodes)):
        if len(episodes) == 0:
            print(f"WARNING: the {name} split has 0 episodes ({num_episodes} episodes in total). "
                  f"Training will fail on it -- generate more episodes or change the fractions.")

    # All three splits are first written to temporary files and only then
    # renamed, so a crash while writing never leaves a truncated split or a
    # mix of old and new splits behind. (The three renames themselves are not
    # one atomic step, but they take microseconds and cannot run out of disk.)
    pending = []
    try:
        for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            path = save_prefix + f"_{name}.npz"
            tmp = f"{path}.tmp{os.getpid()}"
            pending.append((tmp, path))
            with open(tmp, "wb") as f:
                np.savez(f, X=x_vals[mask], Y=y_vals[mask], episode_id=episode_ids[mask],
                         **npz_metadata(identity))
        for tmp, path in pending:
            os.replace(tmp, path)
    finally:
        for tmp, _ in pending:
            if os.path.exists(tmp):
                os.remove(tmp)

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

    parser.add_argument("--samples",
                        type=int,
                        default=10000,
                        help="Number of EPISODES (start positions) to simulate; each "
                             "yields up to horizon/dt rows")

    parser.add_argument("--horizon",
                        type=float,
                        default=80.0,
                        help="Rollout duration (seconds)")

    parser.add_argument("--save",
                        type=str,
                        default="data/ship_dynamics_dataset.csv",
                        help="Output filename")

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
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Split dataset into train/val/test sets by episode "
                             "(--no-split to skip)")

    parser.add_argument("--bank-fraction",
                        type=float,
                        default=BANK_START_FRACTION,
                        help="Fraction of episodes starting near a real bank (0 = all "
                             "starts uniform in the river)")

    parser.add_argument("--bank-band",
                        type=float,
                        nargs=2,
                        metavar=("MIN_M", "MAX_M"),
                        default=list(BANK_START_BAND),
                        help="Distance band (m, from ship centre to nearest real bank) "
                             "for near-bank starts")

    parser.add_argument("--straight-fraction",
                        type=float,
                        default=STRAIGHT_FRACTION,
                        help="Fraction of ALL episodes run as 'straight' episodes (open water, "
                             "rudder ~0, constant forward thrust; taken from the uniform starts, "
                             "so at most 1 - bank-fraction). 0 = none. Only for generators "
                             "that support it (the wind/current ones)")

    parser.add_argument("--ship-model",
                        choices=[m.name for m in ShipModel],
                        default=ShipModel.TRIPOLI.name,
                        help="Ship simulated by the generator; stored in the .npz identity and "
                             "checked against the generator's RUDDER_MAX_DEG")

    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Base seed for episode actions/initial conditions/wind, "
                             "tidal offsets and the train/val/test split")

    args = parser.parse_args()

    # nargs=2 guarantees exactly two values; unpacking makes that visible to the
    # type checker (tuple(list) is inferred as a variable-length tuple).
    bank_min, bank_max = args.bank_band
    bank_band = (float(bank_min), float(bank_max))

    # Before the (long) generation run: refuse an inconsistent ship/generator.
    identity = dataset_identity(ShipModel[args.ship_model])
    print(f"Dataset identity: {identity}")

    # ============================================================
    # Generate
    # ============================================================

    extra = {}
    if _SUPPORTS_STRAIGHT:
        extra["straight_fraction"] = args.straight_fraction
    elif args.straight_fraction:
        parser.error("--straight-fraction: the installed sim_fh_dataset_generator has no straight episodes.")

    print(f"Generating {args.samples} episodes (max. horizon={args.horizon}s)...")

    x_values, y_values, episode_ids, nbr_samples = collect_supervised_dataset(
        nbr_samples=args.samples,
        horizon_seconds=args.horizon,
        output_file=args.save,
        workers=args.workers,
        seed=args.seed,
        bank_fraction=args.bank_fraction,
        bank_band=bank_band,
        **extra,
    )

    if nbr_samples == 0:
        print("Warning: no samples were collected - dataset files were not written.")
        return

    print(f"Saved dataset to {args.save} ({len(x_values)} rows from {nbr_samples} episodes)")

    # ============================================================
    # Optional splitting
    # ============================================================
    if args.split:
        prefix = os.path.splitext(args.save)[0]
        split_and_save(x_values, y_values, episode_ids, prefix, identity, seed=args.seed)


if __name__ == "__main__":
    main()
