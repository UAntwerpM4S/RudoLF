#!/usr/bin/env python3
"""
CLI for generating surrogate training data with the FH dataset generator
(sim_fh_dataset_generator.collect_supervised_dataset): writes the full CSV and,
unless --no-split, train/val/test .npz files split by episode.

--ship-model (TRIPOLI / MYZAKO_37) and --configuration (full = ship-bank + UKC +
wind + current, confined = ship-bank + UKC) select the data.

Each .npz carries its schema.DatasetIdentity:
  * scenario: from the row width (37 = full_env, 33 = confined_ukc)
  * ray_layout: ray_features.RAY_LAYOUT_VERSION
  * ship_model: --ship-model
  * sway_convention: schema.SWAY_CONVENTION (sway positive to port)
  * rudder_max_deg: angle of rudder_action = +-1; 35 deg for TRIPOLI, FH's
    getMaximumAngle() for MYZAKO_37 (probed before the run). Must equal the
    ship's maximum rudder angle in ship_factory, else generation is refused.
"""

import os
import json
import argparse
import datetime
import numpy as np

from typing import Optional

# schema (numpy only), not dataset.py: data generation must not need torch.
from fw.environments.ray_features import RAY_LAYOUT_VERSION
from fw.ships.ship_factory import ShipModel
from fw.surrogate.schema import (SWAY_CONVENTION, DatasetIdentity, npz_metadata, provenance_metadata,
                                 scenario_for_raw_width)
from fw.surrogate.sim_fh_dataset_generator import (
    BANK_START_BAND,
    BANK_START_FRACTION,
    BANK_STRAIGHT_BAND,
    BANK_STRAIGHT_FRACTION,
    CONFIGURATIONS,
    DEFAULT_CONFIGURATION,
    DEFAULT_SHIP_MODEL,
    SHIP_PROFILES,
    STRAIGHT_FRACTION,
    THRUST_ACTION_RANGE,
    check_ship_against_factory,
    collect_supervised_dataset,
    nbr_x_features,
    resolve_rudder_max_deg,
    validate_start_settings,
)


def dataset_identity(ship_model: ShipModel, configuration: str = DEFAULT_CONFIGURATION,
                     rudder_max_deg: Optional[float] = None) -> DatasetIdentity:
    """
    Identity of the data the generator produces for ship_model in configuration.

    Args:
        ship_model: Ship to simulate.
        configuration: "full" / "confined".
        rudder_max_deg: The run's rudder scale; None = resolve_rudder_max_deg
            (probes FH for a ship whose scale is FH's own maximum).

    Raises:
        ValueError: If the ship has no generator profile, the configuration is
            unknown, or the rudder scale differs from the ship's maximum rudder
            angle in ship_factory.
    """

    if rudder_max_deg is None:
        rudder_max_deg = resolve_rudder_max_deg(ship_model, configuration)
    check_ship_against_factory(ship_model, rudder_max_deg)
    scenario = scenario_for_raw_width(nbr_x_features(configuration))
    return DatasetIdentity(scenario=scenario.name, ray_layout=RAY_LAYOUT_VERSION,
                           ship_model=ship_model.name, rudder_max_deg=float(rudder_max_deg),
                           sway_convention=SWAY_CONVENTION)


def generation_provenance(args, ship_model: ShipModel, rudder_max_deg: float, n_episodes: int,
                          n_rows: int, started_utc: Optional[str] = None) -> dict:
    """
    How a dataset was generated (CLI settings, rudder scale, FH exercises,
    generator data version, row/episode counts). Stored in every .npz
    (schema.NPZ_PROVENANCE_KEY) and as <csv stem>_generation.json.
    Informational only; not part of the dataset identity.
    """

    from fw.surrogate.sim_fh_dataset_generator import _CHECKPOINT_VERSION, exercise_paths

    exercise, no_current = exercise_paths(ship_model, args.configuration)
    settings = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in sorted(vars(args).items())}
    return {
        "started_utc": started_utc,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "generator_data_version": int(_CHECKPOINT_VERSION),
        "cli_settings": settings,
        "ship_model": ship_model.name,
        "configuration": args.configuration,
        "rudder_max_deg": float(rudder_max_deg),
        "exercise": os.path.basename(str(exercise)),
        "exercise_no_current": os.path.basename(str(no_current)) if no_current else None,
        "episodes_with_rows": int(n_episodes),
        "rows": int(n_rows),
    }


def split_and_save(x_vals, y_vals, episode_ids, save_prefix, identity: DatasetIdentity,
                   train=0.8, val=0.1, seed=None, provenance: Optional[dict] = None):
    """
    Split the dataset into train/val/test by episode and save as .npz.

    All rows of one rollout go to the same split; a row-level split would leak
    near-duplicate, correlated rows across splits and bias val/test metrics.
    The fractions are exact in episodes, approximate in rows (episodes vary in
    length).

    Args:
        x_vals: Inputs, shape (N, raw width of identity.scenario).
        y_vals: Targets, shape (N, 3).
        episode_ids: Per-row episode index, shape (N,).
        save_prefix: Path prefix of the three .npz files.
        identity: DatasetIdentity stored in every .npz (checked by dataset.load_raw).
        train: Fraction of episodes for train.
        val: Fraction of episodes for val (the rest is test).
        seed: Optional seed of the episode shuffle (reproducible split). Each
            .npz also stores the per-row episode_id.
        provenance: Optional generation_provenance() dict, stored in every .npz.
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

    # Write all splits to temporary files first, then rename, so a crash never
    # leaves a truncated split or a mix of old and new splits.
    pending = []
    try:
        for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            path = save_prefix + f"_{name}.npz"
            tmp = f"{path}.tmp{os.getpid()}"
            pending.append((tmp, path))
            with open(tmp, "wb") as f:
                np.savez(f, X=x_vals[mask], Y=y_vals[mask], episode_id=episode_ids[mask],
                         **npz_metadata(identity),
                         **(provenance_metadata(provenance) if provenance is not None else {}))
        for tmp, path in pending:
            os.replace(tmp, path)
    finally:
        for tmp, _ in pending:
            if os.path.exists(tmp):
                os.remove(tmp)

    print(f"Saved: {save_prefix}_train.npz  ({train_mask.sum()} rows from {len(train_episodes)} episodes)")
    print(f"Saved: {save_prefix}_val.npz    ({val_mask.sum()} rows from {len(val_episodes)} episodes)")
    print(f"Saved: {save_prefix}_test.npz   ({test_mask.sum()} rows from {len(test_episodes)} episodes)")


def main():
    """
    Command-line entry point.
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
                             "so at most 1 - bank-fraction). 0 = none")

    parser.add_argument("--bank-straight-fraction",
                        type=float,
                        default=BANK_STRAIGHT_FRACTION,
                        help="Fraction of ALL episodes run as 'bank-straight' episodes (start "
                             "--bank-straight-band m from a real bank, heading parallel to it, "
                             "rudder ~0, constant forward thrust; taken from the uniform starts). "
                             "0 = none (default)")

    parser.add_argument("--bank-straight-band",
                        type=float,
                        nargs=2,
                        metavar=("MIN_M", "MAX_M"),
                        default=list(BANK_STRAIGHT_BAND),
                        help="Distance band (m, from ship centre to nearest real bank) for "
                             "bank-straight starts (min must exceed the 10 m bank stop limit)")

    parser.add_argument("--thrust-range",
                        type=float,
                        nargs=2,
                        metavar=("LOW", "HIGH"),
                        default=list(THRUST_ACTION_RANGE),
                        help="thrust_action range of the random episodes (default -1 1). The RL "
                             "framework maps its thrust to [min_thrust, max_thrust] = [0.1, 1], so "
                             "'0 1' spends no data on negative thrust")

    parser.add_argument("--ship-model",
                        choices=[m.name for m in SHIP_PROFILES],
                        default=DEFAULT_SHIP_MODEL.name,
                        help="Ship to simulate (TRIPOLI: azipods, +-35 deg; MYZAKO_37: rudder + "
                             "propeller, +-75 deg); stored in the .npz identity")

    parser.add_argument("--configuration",
                        choices=sorted(CONFIGURATIONS),
                        default=DEFAULT_CONFIGURATION,
                        help="full: ship-bank + UKC + wind + current (scenario full_env); "
                             "confined: ship-bank + UKC (scenario confined_ukc)")

    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Base seed for episode actions/initial conditions/wind, "
                             "tidal offsets and the train/val/test split")

    args = parser.parse_args()

    # Unpack the nargs=2 pairs so the type checker sees fixed-length tuples.
    bank_min, bank_max = args.bank_band
    bank_band = (float(bank_min), float(bank_max))
    bs_min, bs_max = args.bank_straight_band
    bank_straight_band = (float(bs_min), float(bs_max))
    t_lo, t_hi = args.thrust_range
    thrust_range = (float(t_lo), float(t_hi))

    # Validate before the long run: cheap checks first, then the rudder scale
    # (the Myzako probe starts FH and can take up to TIMEOUT_SECONDS).
    ship_model = ShipModel[args.ship_model]
    try:
        validate_start_settings(args.bank_fraction, bank_band, args.straight_fraction,
                                args.bank_straight_fraction, bank_straight_band, thrust_range)
    except ValueError as exc:
        parser.error(str(exc))
    rudder_max_deg = resolve_rudder_max_deg(ship_model, args.configuration)
    identity = dataset_identity(ship_model, args.configuration, rudder_max_deg)
    print(f"Dataset identity: {identity}")

    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    print(f"Generating {args.samples} episodes (max. horizon={args.horizon}s, {ship_model.name}, "
          f"configuration '{args.configuration}')...")

    x_values, y_values, episode_ids, nbr_samples = collect_supervised_dataset(
        nbr_samples=args.samples,
        horizon_seconds=args.horizon,
        output_file=args.save,
        workers=args.workers,
        seed=args.seed,
        bank_fraction=args.bank_fraction,
        bank_band=bank_band,
        straight_fraction=args.straight_fraction,
        bank_straight_fraction=args.bank_straight_fraction,
        bank_straight_band=bank_straight_band,
        thrust_range=thrust_range,
        ship_model=ship_model,
        configuration=args.configuration,
        rudder_max_deg=rudder_max_deg,
    )

    if nbr_samples == 0:
        print("Warning: no samples were collected - dataset files were not written.")
        return

    print(f"Saved dataset to {args.save} ({len(x_values)} rows from {nbr_samples} episodes)")

    # Provenance is informational: a failure here is reported, never fatal.
    provenance = None

    try:
        provenance = generation_provenance(args, ship_model, rudder_max_deg, nbr_samples, len(x_values),
                                           started_utc=started_utc)
        provenance_metadata(provenance)          # check serializability now, not while writing splits
    except Exception as exc:  # noqa: BLE001 - never lose the dataset over the provenance
        provenance = None
        print(f"WARNING: could not record the generation settings ({type(exc).__name__}: {exc}); "
              f"the dataset is written without them.")
    if provenance is not None:
        provenance_path = os.path.splitext(args.save)[0] + "_generation.json"
        tmp = f"{provenance_path}.tmp{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                json.dump(provenance, f, indent=2, sort_keys=True)
            os.replace(tmp, provenance_path)
            print(f"Saved generation settings to {provenance_path}")
        except Exception as exc:  # noqa: BLE001 - never lose the dataset over the provenance
            print(f"WARNING: could not write {provenance_path} ({type(exc).__name__}: {exc}); the .npz "
                  f"files still carry the generation settings.")
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    if args.split:
        prefix = os.path.splitext(args.save)[0]
        split_and_save(x_values, y_values, episode_ids, prefix, identity, seed=args.seed, provenance=provenance)


if __name__ == "__main__":
    main()
