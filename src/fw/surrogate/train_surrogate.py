# train_surrogate.py
"""
Training script for the ship dynamics surrogate MLP -- every scenario.

The model scenario (--scenario) defaults to the scenario of the training
files; a poorer scenario can be trained from richer data when allowed
(schema.check_projection), e.g.:

    # UA_full (full_env data, full_env model)
    python -m fw.surrogate.train_surrogate --train-file .._train.npz --val-file .._val.npz \
        --test-file .._test.npz --checkpoint checkpoints/best_surrogate_tripoli_full.pth
    # UA_clean (open_water model from confined_ukc data)
    python -m fw.surrogate.train_surrogate --scenario open_water ... \
        --checkpoint checkpoints/best_surrogate_tripoli_clean.pth

Inputs (see schema.encoded_feature_names): the scenario's input columns
(x, y always dropped), angles sin/cos encoded; for full_env, wind/current per
--env-encoding (body_components: 36 inputs, angles: 38).
Output (3-D): [delta_surge, delta_sway, delta_yaw_rate]

The checkpoint stores the scenario, env encoding, the data's identity (ship
model, rudder scale, ray layout) and the feature layout; SurrogatePredictor
checks all of them on load.

Several datasets of the same identity (same scenario, ship, rudder scale, ray
layout, sway convention), e.g. a main run plus a supplementary run with
bank-straight episodes, are trained on together by giving several files per
split (in the same order for the three splits):

    python -m fw.surrogate.train_surrogate \
        --train-file main_train.npz extra_train.npz \
        --val-file main_val.npz extra_val.npz \
        --test-file main_test.npz extra_test.npz ...

The rows are concatenated per split; normalization stats come from all
training files together. The test results are reported per test file too.

--min-bank-distance D keeps only rows whose nearest bank (from the row's own
ray features, all rays) is at least D m away, in all three splits. Meant for
an open_water ("clean") model trained from confined_ukc data: it cannot see
the bank, so rows with bank interaction would only add unexplainable noise.
D = 200 (the ray range) keeps only rows without any bank in ray range:

    python -m fw.surrogate.train_surrogate --scenario open_water \
        --min-bank-distance 200 ... --checkpoint checkpoints/best_surrogate_myzako_clean.pth

--min-ukc U keeps only rows with an under-keel clearance of at least U m (from
the data's UKC column), in all three splits: e.g. a clean model for deep
water only.
"""

import random
import torch
import argparse
import logging
import numpy as np
import torch.nn.functional as F

from collections.abc import Sized
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset
from fw.environments.ray_features import RAY_MAX_DISTANCE
from fw.surrogate.dataset import check_same_identity, load_raw, read_identity, read_provenance, stats_from_arrays
from fw.surrogate.episode_time import T_EPISODE_KEY
from fw.surrogate.models.mlp_surrogate import ACTIVATIONS, SurrogateMLP, activation_from_name
from fw.surrogate.schema import (
    DEFAULT_ENV_ENCODING,
    ENV_ENCODINGS,
    SCENARIOS,
    RAY_COLUMN_NAMES,
    TARGET_NAMES,
    check_projection,
    encoded_feature_names,
    get_scenario,
    project_raw_columns,
    select_and_encode,
)
from fw.surrogate.utils import load_checkpoint, data_path, get_device, save_checkpoint

logger = logging.getLogger(__name__)

LOSS_LABELS = {"mse": "MSE", "l1": "MAE", "huber": "Huber"}


def _num_samples(loader: DataLoader) -> int:
    """
    Number of samples in a map-style loader's dataset.

    torch types `DataLoader.dataset` as `Dataset`, which deliberately does not
    define `__len__` (iterable datasets have no length). The TensorDatasets
    built here do; the isinstance check narrows the type for the checker and fails
    loudly if a dataset without a length is ever passed in.
    """

    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError(f"{type(dataset).__name__} has no __len__; a map-style dataset is required.")
    return len(dataset)


def make_loss_fn(name: str = "mse", huber_delta: float = 1.0):
    """
    Return the training/validation criterion, computed on NORMALIZED targets.

    Args:
        name: "mse" (mean squared error), "l1" (mean absolute error) or
            "huber" (quadratic for |error| < huber_delta, linear beyond).
        huber_delta: Transition point for "huber", in normalized units
            (i.e. target standard deviations).

    Returns:
        Callable `loss_fn(pred, target, reduction="mean")`, same signature
        as `F.mse_loss`.
    """

    if name == "mse":
        return F.mse_loss
    if name == "l1":
        return F.l1_loss
    if name == "huber":
        return lambda pred, target, reduction="mean": F.huber_loss(
            pred, target, reduction=reduction, delta=huber_delta)
    raise ValueError(f"Unknown loss '{name}'. Choose from {sorted(LOSS_LABELS)}.")


def evaluate(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        loss_fn=None,
):
    """
    Compute the normalized loss (MSE unless `loss_fn` is given) and
    denormalized MAE.

    Args:
        model: Trained (or in-training) SurrogateMLP.
        loader: DataLoader yielding normalized (X, Y) batches.
        device: Torch device.
        y_mean: Output normalization mean.
        y_std: Output normalization std.
        loss_fn: Criterion from `make_loss_fn`; defaults to MSE so existing
            callers keep their behavior. Pass the TRAINING criterion so that
            early stopping and the LR scheduler judge what is optimized.

    Returns:
        Tuple:
            mean_loss: Mean normalized loss (per-sample weighted).
            mae: Mean absolute error in original physical units
                 for each output dimension.
    """

    model.eval()

    if _num_samples(loader) == 0:
        raise ValueError(
            "evaluate() was given an empty loader (0 samples). This usually "
            "means your episode-level train/val/test split produced a split "
            "with too few episodes for its fraction to round up to >=1 "
            "(int(fraction * num_episodes) == 0) -- common when smoke-testing "
            "with a small --samples count and a low simulation success rate. "
            "Generate more episodes, or adjust the split fractions, rather "
            "than silently dividing by zero below."
        )

    loss_fn = loss_fn or F.mse_loss
    total_loss = 0.0
    total_samples = 0
    total_abs_error = None

    # MAE in physical units only needs the scale: (pred - y) * y_std. y_mean is
    # kept in the signature for callers.
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    with torch.no_grad():
        for x_vals, y_vals in loader:
            x_vals = x_vals.to(device, non_blocking=True)
            y_vals = y_vals.to(device, non_blocking=True)

            pred = model(x_vals)

            loss = loss_fn(pred, y_vals, reduction="mean")
            total_loss += loss.item() * x_vals.size(0)

            # pred_phys - y_phys == (pred - y) * y_std
            abs_error = torch.abs((pred - y_vals) * y_std_t)

            if total_abs_error is None:
                total_abs_error = abs_error.sum(dim=0)
            else:
                total_abs_error += abs_error.sum(dim=0)

            total_samples += x_vals.size(0)

    mean_loss = total_loss / total_samples
    mae = total_abs_error / total_samples

    return mean_loss, mae.cpu().numpy()


PathOrPaths = Union[str, Path, Sequence[Union[str, Path]]]


def _as_paths(files: PathOrPaths) -> List[str]:
    """One path or a sequence of paths -> list of path strings."""

    if isinstance(files, (str, Path)):
        return [str(files)]
    return [str(f) for f in files]


def nearest_bank_distance(x_raw: np.ndarray, source) -> np.ndarray:
    """
    Distance [m] to the nearest bank per raw row of scenario `source`, from
    its ray features (minimum over all rays; RAY_MAX_DISTANCE = none in range).

    Raises:
        ValueError: If `source` has no ray columns.
    """

    if not source.includes_rays:
        raise ValueError(f"Scenario '{source.name}' has no ray columns: cannot filter on bank distance.")
    rays = np.clip(x_raw[:, [source.col(c) for c in RAY_COLUMN_NAMES]], 0.0, 1.0)
    return RAY_MAX_DISTANCE * (1.0 - rays.max(axis=1))


def load_split(files: PathOrPaths, scenario, env_encoding,
               min_bank_distance: Optional[float] = None,
               min_ukc: Optional[float] = None,
               min_episode_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encoded (unnormalized) X and Y of one split, concatenated over `files`.

    Args:
        files: One or more dataset .npz files.
        scenario: Model scenario (the files' scenario, or one derived from it).
        env_encoding: Wind/current encoding.
        min_bank_distance: If given, keep only rows whose nearest bank (from
            the file's own ray features, before projection) is at least this
            many metres away.
        min_ukc: If given, keep only rows whose UKC (the file's own UKC
            column, before projection) is at least this many metres.
        min_episode_time: If given, keep only rows at least this many seconds
            after the start of their episode (the file's t_episode array,
            written by fw.surrogate.episode_time); rows of episodes whose
            order could not be reconstructed (t_episode < 0) are left out too.

    Returns:
        `(x, y)` float32 arrays.
    """

    xs, ys = [], []
    for f in _as_paths(files):
        if min_bank_distance is None and min_ukc is None and min_episode_time is None:
            x_raw, y_vals, _ = load_raw(f, scenario)    # projected to the model scenario
        else:
            x_raw, y_vals, identity = load_raw(f)       # the file's own scenario (with rays/UKC)
            source = get_scenario(identity.scenario)
            keep = np.ones(len(x_raw), dtype=bool)
            what = []
            if min_bank_distance is not None:
                keep &= nearest_bank_distance(x_raw, source) >= min_bank_distance
                what.append(f"no bank within {min_bank_distance:g} m")
            if min_ukc is not None:
                if not source.includes_ukc:
                    raise ValueError(f"{f}: scenario '{source.name}' has no UKC column: cannot filter on UKC.")
                keep &= x_raw[:, source.col("ukc")] >= min_ukc
                what.append(f"UKC >= {min_ukc:g} m")
            if min_episode_time is not None:
                with np.load(f, allow_pickle=False) as z:
                    if T_EPISODE_KEY not in z.files:
                        raise ValueError(f"{f} has no '{T_EPISODE_KEY}' array: run "
                                         f"'python -m fw.surrogate.episode_time {f}' first.")
                    t_episode = z[T_EPISODE_KEY]
                if t_episode.shape != (len(x_raw),):
                    raise ValueError(f"{f}: '{T_EPISODE_KEY}' has shape {t_episode.shape}, expected ({len(x_raw)},).")
                keep &= (t_episode >= 0) & (t_episode >= min_episode_time - 1e-6)
                what.append(f"t_episode >= {min_episode_time:g} s")
            logger.info("%s: %d of %d rows (%.1f %%) have %s; the rest is left out.",
                        f, int(keep.sum()), len(keep), 100.0 * keep.mean() if len(keep) else 0.0,
                        " and ".join(what))
            x_raw = project_raw_columns(x_raw[keep], source, scenario)
            y_vals = y_vals[keep]
        x_vals = select_and_encode(x_raw, scenario, env_encoding)
        xs.append(x_vals)
        ys.append(y_vals)
    if len(xs) == 1:
        return xs[0], ys[0]
    return np.concatenate(xs), np.concatenate(ys)


def normalized_dataset(x_vals: np.ndarray, y_vals: np.ndarray, x_mean, x_std, y_mean, y_std) -> TensorDataset:
    """TensorDataset of the normalized arrays (same arithmetic as ShipDynamicsDataset)."""

    x_out = (x_vals - np.asarray(x_mean, np.float32).reshape(1, -1)) / np.asarray(x_std, np.float32).reshape(1, -1)
    y_out = (y_vals - np.asarray(y_mean, np.float32).reshape(1, -1)) / np.asarray(y_std, np.float32).reshape(1, -1)
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(x_out, dtype=np.float32)),
                         torch.from_numpy(np.ascontiguousarray(y_out, dtype=np.float32)))


def log_provenance(train_files: Sequence[str], all_files: Sequence[str]) -> dict:
    """
    Log how each dataset file was generated (its provenance, see
    generate_ship_data.generation_provenance) and return {file: provenance or
    None} for the checkpoint. When several training datasets are combined,
    the generator settings in which they differ are listed: combining them is
    allowed (e.g. a main dataset plus a bank-straight one), this only makes
    the difference visible. Informational only: nothing is checked or changed.
    """

    provenance = {}
    unreadable = set()
    for f in all_files:
        try:
            p = read_provenance(f)
            if p is not None and not isinstance(p, dict):
                raise ValueError(f"expected a JSON object, got {type(p).__name__}")
            provenance[f] = p
        except Exception as exc:  # noqa: BLE001 - informational only, never stop training over it
            logger.warning("%s: could not read its generation provenance (%s).", f, exc)
            provenance[f] = None
            unreadable.add(f)
    missing = [f for f in all_files if provenance[f] is None and f not in unreadable]
    if missing:
        logger.info("No generation provenance recorded in %d file(s) (written before it was recorded): %s",
                    len(missing), ", ".join(missing))
    try:
        _log_provenance_summary(train_files, provenance)
    except Exception as exc:  # noqa: BLE001 - informational only, never stop training over it
        logger.warning("Could not summarise the generation provenance (%s).", exc)
    return provenance


def _cli_settings(p: dict) -> dict:
    """The provenance's generator CLI settings ({} if absent or malformed)."""

    s = p.get("cli_settings")
    return s if isinstance(s, dict) else {}


def _log_provenance_summary(train_files: Sequence[str], provenance: dict) -> None:
    """Log the provenance of every training file and the settings in which they differ (see log_provenance)."""

    for f in train_files:
        p = provenance[f]
        if p is not None:
            s = _cli_settings(p)
            logger.info("%s: generated %s, %s %s, %s episodes, seed %s, thrust range %s, bank %s / straight %s / "
                        "bank-straight %s", f, p.get("created_utc"), p.get("ship_model"), p.get("configuration"),
                        p.get("episodes_with_rows"), s.get("seed"), s.get("thrust_range"), s.get("bank_fraction"),
                        s.get("straight_fraction"), s.get("bank_straight_fraction"))
    known = [(f, provenance[f]) for f in train_files if provenance[f] is not None]
    if len(known) > 1:
        ignore = {"save", "workers", "split"}
        settings = [{k: v for k, v in _cli_settings(p).items() if k not in ignore} for _, p in known]
        keys = sorted(set().union(*settings))
        differing = [k for k in keys if len({repr(st.get(k)) for st in settings}) > 1]
        for k in ("configuration", "exercise", "generator_data_version"):
            if k not in differing and len({repr(p.get(k)) for _, p in known}) > 1:
                differing.append(k)
        if differing:
            logger.info("The training datasets were generated with different settings: %s. Combining them "
                        "is allowed; check that this is intended.", ", ".join(differing))


def make_dataloaders(train_file: PathOrPaths, val_file: PathOrPaths, test_file: PathOrPaths, scenario,
                     env_encoding, batch_size=256, num_workers=0, seed=None, drop_last=False, pin_memory=False,
                     min_bank_distance: Optional[float] = None, min_ukc: Optional[float] = None,
                     min_episode_time: Optional[float] = None):
    """
    Build train / val / test DataLoaders with shared normalization stats.

    Normalization statistics are computed from the training split only and
    then re-applied identically to the validation and test splits. Each split
    may consist of several .npz files (rows concatenated).

    Args:
        train_file:  Path(s) to training .npz file(s).
        val_file:    Path(s) to validation .npz file(s).
        test_file:   Path(s) to test .npz file(s).
        scenario:    schema.ScenarioSchema of the model.
        env_encoding: Wind/current encoding ("angles" / "body_components");
                     the same one is used for all three splits.
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.
        seed:        If given, seeds the train loader's shuffle order so
                     runs are reproducible independently of other RNG use.
        drop_last:   Drop the last incomplete train batch. Recommended with
                     BatchNorm: a tiny last batch gives noisy batch stats,
                     and a batch of 1 makes BatchNorm1d raise.
        pin_memory:  Pin host memory (useful when training on CUDA).
        min_bank_distance, min_ukc, min_episode_time: Row filters applied to
                     all splits (see load_split).

    Returns:
        Tuple of (train_loader, val_loader, test_loader,
                  x_mean, x_std, y_mean, y_std).
    """

    # Stats from the (concatenated) training split only; val/test reuse them.
    x_train, y_train = load_split(train_file, scenario, env_encoding, min_bank_distance, min_ukc,
                                  min_episode_time)
    stats = stats_from_arrays(x_train, y_train, encoded_feature_names(scenario, env_encoding))
    x_mean, x_std, y_mean, y_std = (np.asarray(s, np.float32).reshape(1, -1) for s in stats)

    train_ds = normalized_dataset(x_train, y_train, *stats)
    del x_train, y_train
    val_ds = normalized_dataset(*load_split(val_file, scenario, env_encoding, min_bank_distance, min_ukc,
                                                min_episode_time), *stats)
    test_ds = normalized_dataset(*load_split(test_file, scenario, env_encoding, min_bank_distance, min_ukc,
                                                 min_episode_time), *stats)
    # Fail before training, not after it: an empty val split would only fail
    # after the first epoch, an empty test split only after the whole run.
    empty = [name for name, ds in (("validation", val_ds), ("test", test_ds)) if len(ds) == 0]
    if empty:
        filters = ", ".join(f"{k}={v}" for k, v in (("min_bank_distance", min_bank_distance),
                                                    ("min_ukc", min_ukc),
                                                    ("min_episode_time", min_episode_time)) if v is not None)
        raise ValueError(
            f"The {' and '.join(empty)} split{'s have' if len(empty) > 1 else ' has'} 0 samples"
            + (f" after the row filters ({filters}) -- or the file itself has no rows. Relax the "
               f"filters, or check the files."
               if filters else
               ". The episode-level split probably gave it no episodes (too few episodes for its "
               "fraction); generate more episodes or check the files."))

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    # Only drop the last batch when that still leaves at least one batch.
    drop_last = drop_last and len(train_ds) > batch_size

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=drop_last,
                              generator=generator, **common)
    val_loader   = DataLoader(val_ds,  shuffle=False, **common)
    test_loader  = DataLoader(test_ds, shuffle=False, **common)

    logger.info("Samples: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))

    return train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    """
    Entry point for surrogate training.

    Args:
        argv: Optional list of CLI arguments (useful for testing).
    """

    parser = argparse.ArgumentParser(
        description="Train an MLP surrogate (any scenario) -> 3 outputs (delta surge/sway/yaw rate)."
    )
    parser.add_argument("--train-file", nargs="+", default=[data_path("ship_dynamics_dataset_train.npz")],
                        help="Training .npz file(s); several = datasets of one identity trained together")
    parser.add_argument("--val-file", nargs="+", default=[data_path("ship_dynamics_dataset_val.npz")],
                        help="Validation .npz file(s), same number and order as --train-file")
    parser.add_argument("--test-file", nargs="+", default=[data_path("ship_dynamics_dataset_test.npz")],
                        help="Test .npz file(s), same number and order as --train-file")
    parser.add_argument("--batch-size", type=int, default=128)  # 256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0)  # 1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)  # early stopping patience
    parser.add_argument("--loss", choices=sorted(LOSS_LABELS), default="l1",
                        help="Training/validation criterion on normalized targets: mse, "
                             "l1 (mean absolute error) or huber")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber transition point, in target standard deviations")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Relative val-loss improvement needed to count as 'better'")
    parser.add_argument("--dropout", type=float, default=0.0)  # 0.05)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[128, 128])  # [128, 128])
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), default="ReLU",
                        help="Hidden-layer activation; stored in the checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_surrogate.pth")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=None,
                        help="Scenario of the MODEL. Default: the scenario of the training files. "
                             "A poorer scenario is derived from richer data when allowed "
                             "(e.g. open_water from confined_ukc). Stored in the checkpoint.")
    parser.add_argument("--env-encoding", choices=list(ENV_ENCODINGS),
                        default=DEFAULT_ENV_ENCODING,
                        help="Wind/current input encoding: body_components (ship-frame components, "
                             "default) or angles (speed + sin/cos direction). No effect for "
                             "scenarios without wind/current. Stored in the checkpoint.")
    parser.add_argument("--min-bank-distance", type=float, default=None,
                        help="Keep only rows whose nearest bank (from the data's ray features, all rays) "
                             f"is at least this many metres away, in all splits; {RAY_MAX_DISTANCE:g} = no "
                             "bank in ray range. For an open_water model from confined/full data. "
                             "Default: no filter")
    parser.add_argument("--min-ukc", type=float, default=None,
                        help="Keep only rows with an under-keel clearance of at least this many metres "
                             "(the data's UKC column), in all splits. Default: no filter")
    parser.add_argument("--min-episode-time", type=float, default=None,
                        help="Keep only rows at least this many seconds after the start of their episode, "
                             "in all splits (leaves out the start transient). Needs the files' t_episode "
                             "array: run 'python -m fw.surrogate.episode_time <files>' first. "
                             "Default: no filter")
    args = parser.parse_args(argv)

    # Identity of the data: all splits of all datasets must have the same identity.
    train_files, val_files, test_files = (_as_paths(f) for f in (args.train_file, args.val_file, args.test_file))
    if not len(train_files) == len(val_files) == len(test_files):
        parser.error(f"--train-file, --val-file and --test-file need the same number of files "
                     f"(one per dataset), got {len(train_files)}, {len(val_files)}, {len(test_files)}.")
    all_files = train_files + val_files + test_files
    resolved = [str(Path(f).resolve()) for f in all_files]
    if len(set(resolved)) != len(resolved):
        parser.error("A file is given more than once (its rows would be counted twice).")
    data_identity = check_same_identity(all_files, [read_identity(f) for f in all_files])
    if len(train_files) > 1:
        logger.info("Training on %d datasets: %s", len(train_files), ", ".join(train_files))
    data_provenance = log_provenance(train_files, all_files)
    source_scenario = get_scenario(data_identity.scenario)
    scenario = get_scenario(args.scenario) if args.scenario else source_scenario
    check_projection(source_scenario, scenario)
    if args.min_bank_distance is not None:
        if not source_scenario.includes_rays:
            parser.error(f"--min-bank-distance needs data with ray features; the files are "
                         f"'{source_scenario.name}'.")
        if not 0.0 < args.min_bank_distance <= RAY_MAX_DISTANCE:
            parser.error(f"--min-bank-distance must be in (0, {RAY_MAX_DISTANCE:g}] m, got {args.min_bank_distance}.")
    if args.min_ukc is not None and not source_scenario.includes_ukc:
        parser.error(f"--min-ukc needs data with a UKC column; the files are '{source_scenario.name}'.")
    if args.min_episode_time is not None and not args.min_episode_time >= 0.0:   # also rejects NaN
        parser.error(f"--min-episode-time must be >= 0 s, got {args.min_episode_time}.")
    logger.info("Data: scenario=%s, ship_model=%s, rudder_max_deg=%g -> model scenario=%s",
                source_scenario.name, data_identity.ship_model, data_identity.rudder_max_deg, scenario.name)

    if args.patience >= args.epochs:
        logger.warning(
            "patience (%d) >= epochs (%d): early stopping can never trigger; "
            "training always runs all epochs.", args.patience, args.epochs
        )

    loss_fn = make_loss_fn(args.loss, args.huber_delta)
    loss_label = LOSS_LABELS[args.loss]

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()

    train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std = make_dataloaders(
        train_files, val_files, test_files, scenario, args.env_encoding,
        batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, drop_last=args.batchnorm, pin_memory=device.type == "cuda",
        min_bank_distance=args.min_bank_distance, min_ukc=args.min_ukc,
        min_episode_time=args.min_episode_time,
    )

    # Checked before the peek below: on an empty split, next(iter(...)) would
    # raise a bare StopIteration instead of a readable error.
    n_train = _num_samples(train_loader)
    if n_train == 0:
        raise ValueError(f"Training split {args.train_file} contains 0 samples.")
    if args.batchnorm and n_train < 2:
        raise ValueError(
            f"--batchnorm needs at least 2 training samples, got {n_train}."
        )

    # Derive dims from data (guards against accidental mismatch with whatever
    # dataset schema is currently in train_file/val_file/test_file).
    # Peek once: each iter() on a DataLoader starts a fresh set of workers.
    x_peek, y_peek = next(iter(train_loader))
    input_dim, output_dim = x_peek.shape[1], y_peek.shape[1]
    del x_peek, y_peek

    # Separate, non-dropping, unshuffled loader over the TRAIN set for the
    # eval-mode train loss: the training loader may use drop_last (BatchNorm),
    # which would silently leave the last partial batch out of that number.
    train_eval_loader = DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False,
        num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory,
    )

    target_names = list(TARGET_NAMES)
    if output_dim != len(target_names):
        # zip() further down would otherwise silently drop or mislabel targets
        raise ValueError(
            f"Dataset has output_dim={output_dim}, but {len(target_names)} target "
            f"names are configured: {target_names}"
        )

    feature_names = list(encoded_feature_names(scenario, args.env_encoding))
    if input_dim != len(feature_names):
        raise ValueError(f"Data gives input_dim={input_dim}, but scenario '{scenario.name}' with "
                         f"env_encoding={args.env_encoding!r} defines {len(feature_names)} features.")

    logger.info("Model: scenario=%s, input_dim=%d (env_encoding=%s), output_dim=%d, hidden=%s, activation=%s, "
                "dropout=%s, batchnorm=%s, loss=%s",
                scenario.name, input_dim, args.env_encoding, output_dim, args.hidden_sizes, args.activation,
                args.dropout, args.batchnorm,
                loss_label if args.loss != "huber" else f"Huber(delta={args.huber_delta})")

    model = SurrogateMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=activation_from_name(args.activation),
        use_batch_norm=args.batchnorm,
        dropout=args.dropout,
    ).to(device)

    # AdamW: decoupled weight decay. With plain Adam, weight_decay is folded into
    # the gradient and rescaled by the adaptive denominator, which makes it a
    # weaker and less predictable regulariser. Identical to Adam when wd = 0.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # LR-scheduler patience well below early-stopping patience, otherwise
    # training can stop before the LR has ever been reduced.
    # (Default patience 25 -> 8, same as the previous hardcoded value.)
    sched_patience = max(1, args.patience // 3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=sched_patience
    )

    best_val   = float("inf")
    best_epoch = -1
    no_improve = 0
    ckpt_path  = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss  = 0.0
        n_samples     = 0
        grad_norm_max = 0.0
        counter = 0

        for x_vals, y_vals in train_loader:
            x_vals = x_vals.to(device, non_blocking=True)
            y_vals = y_vals.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_vals)

            loss = loss_fn(pred, y_vals, reduction="mean")
            if counter == 0 and logger.isEnabledFor(logging.DEBUG):
                # per-target loss of the first batch (same criterion as training)
                logger.debug("Epoch %03d first-batch loss per target: %s", epoch,
                             loss_fn(pred, y_vals, reduction="none").mean(dim=0).tolist())
            counter += 1
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            grad_norm_max = max(grad_norm_max, float(grad_norm))
            optimizer.step()

            running_loss += loss.item() * x_vals.size(0)
            n_samples    += x_vals.size(0)

        # Loss as seen during training: dropout ON, BatchNorm on batch statistics,
        # averaged over changing weights. NOT directly comparable to val loss.
        train_running_loss = running_loss / n_samples
        if not np.isfinite(train_running_loss):
            raise FloatingPointError(
                f"Training loss became {train_running_loss} in epoch {epoch}. Check the data "
                f"for extreme values (python -m fw.surrogate.validation.diagnose_splits) or lower --lr."
            )

        # Loss on the training set under the SAME conditions as validation
        # (eval mode, end-of-epoch weights). Compare THIS with val_loss.
        train_eval_loss, _ = evaluate(model, train_eval_loader, device, y_mean, y_std, loss_fn)
        val_loss, val_mae  = evaluate(model, val_loader,   device, y_mean, y_std, loss_fn)

        gap = val_loss / max(train_eval_loss, 1e-12)
        mae_str = ", ".join(f"{n}={v:.6f}" for n, v in zip(target_names, val_mae))
        logger.info(
            "Epoch %03d - train(run): %.6f  train(eval): %.6f  val: %.6f  "
            "val/train: %.2f  lr: %.2e  max|g|: %.2f  MAE[%s]",
            epoch, train_running_loss, train_eval_loss, val_loss,
            gap, optimizer.param_groups[0]["lr"], grad_norm_max, mae_str
        )

        scheduler.step(val_loss)

        if val_loss < best_val * (1.0 - args.min_delta):
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            save_checkpoint({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss":             val_loss,
                "loss":                 args.loss,
                "train_eval_loss":      train_eval_loss,
                "input_dim":            input_dim,
                "output_dim":           output_dim,
                "target_names":         target_names,
                # Feature-set fingerprint; SurrogatePredictor refuses to run
                # if the schema's selection for this scenario has changed since.
                "input_cols":           list(scenario.input_col_indices()),
                "angle_cols":           list(scenario.angle_col_indices()),
                "feature_names":        feature_names,
                # Wind/current encoding; SurrogatePredictor encodes with it.
                "env_encoding":         args.env_encoding,
                # Model scenario and data identity (checked by SurrogatePredictor
                # and by the UAntwerp surrogate against its preset / ship).
                "scenario":             scenario.name,
                "source_scenario":      source_scenario.name,
                "ship_model":           data_identity.ship_model,
                "rudder_max_deg":       data_identity.rudder_max_deg,
                "sway_convention":      data_identity.sway_convention,
                "ray_layout":           data_identity.ray_layout,
                "raw_input_dim":        scenario.raw_input_dim,
                # Row filter of the training data (None = all rows).
                "min_bank_distance":    args.min_bank_distance,
                "min_ukc":              args.min_ukc,
                "min_episode_time":     args.min_episode_time,
                # How the data was generated (informational; None for older files).
                "data_provenance":      data_provenance,
                "hidden_sizes":         args.hidden_sizes,
                "activation":           args.activation,
                "dropout":              args.dropout,
                "use_batchnorm":        args.batchnorm,
                "seed":                 args.seed,
                "args":                 {k: str(v) for k, v in vars(args).items()},
                # Normalization stats - required by SurrogatePredictor at inference
                "x_mean": x_mean,
                "x_std":  x_std,
                "y_mean": y_mean,
                "y_std":  y_std,
            }, ckpt_path)
            logger.info("  ✔ new best (val_loss=%.6f) saved to %s", val_loss, ckpt_path)
        else:
            no_improve += 1

        if no_improve >= args.patience:
            logger.info(
                "Early stopping: no improvement in %d epochs "
                "(best epoch %d, best val_loss %.6f)", args.patience, best_epoch, best_val
            )
            break

    # Final evaluation on held-out test set, using the best checkpoint
    if best_epoch < 0:
        # Without this, a checkpoint left at the same path by an EARLIER run
        # would be loaded and its results reported as this run's.
        raise RuntimeError(
            f"No checkpoint was saved in this run (epochs={args.epochs}, val loss never "
            f"finite/improving), so there is nothing to evaluate. {ckpt_path} was not "
            f"written by this run."
        )
    checkpoint = load_checkpoint(ckpt_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_mae = evaluate(model, test_loader, device, y_mean, y_std, loss_fn)

    print("\nFinal Test Results")
    print("------------------")
    print(f"Best epoch                : {checkpoint.get('epoch', -1)}")
    print(f"Test Loss (normalized {loss_label}): {test_loss:.6f}")
    print(f"Best Validation Loss      : {checkpoint.get('val_loss', float('nan')):.6f}")
    print(f"Train Loss (eval mode)    : {checkpoint.get('train_eval_loss', float('nan')):.6f}")

    for name, mae in zip(target_names, test_mae):
        print(f"MAE {name:<15}: {mae:.6f}")

    if len(test_files) > 1:
        # Per dataset: a combined number can hide a poor fit on the smaller dataset.
        for test_file in test_files:
            loader = DataLoader(normalized_dataset(*load_split(test_file, scenario, args.env_encoding,
                                                               args.min_bank_distance, args.min_ukc,
                                                               args.min_episode_time),
                                                   x_mean, x_std, y_mean, y_std),
                                batch_size=args.batch_size, shuffle=False)
            if _num_samples(loader) == 0:
                print(f"\nTest file {test_file}: 0 rows (after the row filters) -- not evaluated.")
                continue
            file_loss, file_mae = evaluate(model, loader, device, y_mean, y_std, loss_fn)
            print(f"\nTest file {test_file} ({_num_samples(loader)} rows)")
            print(f"  Test Loss (normalized {loss_label}): {file_loss:.6f}")
            for name, mae in zip(target_names, file_mae):
                print(f"  MAE {name:<15}: {mae:.6f}")


if __name__ == "__main__":
    from fw.common.tools.logging_config import setup_logging

    setup_logging(run_type="train_surrogate", env_name="surrogate", to_file=False)
    main()
