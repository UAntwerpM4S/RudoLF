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
"""

import random
import torch
import argparse
import logging
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader
from fw.surrogate.dataset import ShipDynamicsDataset, check_same_identity, compute_stats, read_identity
from fw.surrogate.models.mlp_surrogate import SurrogateMLP
from fw.surrogate.schema import (
    DEFAULT_ENV_ENCODING,
    ENV_ENCODINGS,
    SCENARIOS,
    TARGET_NAMES,
    check_projection,
    encoded_feature_names,
    get_scenario,
)
from fw.surrogate.utils import load_checkpoint, data_path, get_device, save_checkpoint

logger = logging.getLogger(__name__)

LOSS_LABELS = {"mse": "MSE", "l1": "MAE", "huber": "Huber"}


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
            callers keep their behaviour. Pass the TRAINING criterion so that
            early stopping and the LR scheduler judge what is optimized.

    Returns:
        Tuple:
            mean_loss: Mean normalized loss (per-sample weighted).
            mae: Mean absolute error in original physical units
                 for each output dimension.
    """

    model.eval()

    if len(loader.dataset) == 0:
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


def make_dataloaders(train_file, val_file, test_file, scenario, env_encoding, batch_size=256, num_workers=0,
                     seed=None, drop_last=False, pin_memory=False):
    """
    Build train / val / test DataLoaders with shared normalization stats.

    Normalization statistics are computed from the training split only and
    then re-applied identically to the validation and test splits.

    Args:
        train_file:  Path to training .npz file.
        val_file:    Path to validation .npz file.
        test_file:   Path to test .npz file.
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

    Returns:
        Tuple of (train_loader, val_loader, test_loader,
                  x_mean, x_std, y_mean, y_std).
    """

    x_mean, x_std, y_mean, y_std = compute_stats(train_file, scenario, env_encoding)

    stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, normalize=True)
    train_ds = ShipDynamicsDataset(train_file, scenario, env_encoding, **stats)
    val_ds   = ShipDynamicsDataset(val_file,   scenario, env_encoding, **stats)
    test_ds  = ShipDynamicsDataset(test_file,  scenario, env_encoding, **stats)

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
    parser.add_argument("--train-file", default=data_path("ship_dynamics_dataset_train.npz"))
    parser.add_argument("--val-file", default=data_path("ship_dynamics_dataset_val.npz"))
    parser.add_argument("--test-file", default=data_path("ship_dynamics_dataset_test.npz"))
    parser.add_argument("--batch-size", type=int, default=128)  # 256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0)  # 1e-4)
    parser.add_argument("--epochs", type=int, default=21)  # 300)
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
    args = parser.parse_args(argv)

    # Identity of the data: all three splits from one generation run.
    split_files = [str(args.train_file), str(args.val_file), str(args.test_file)]
    data_identity = check_same_identity(split_files, [read_identity(f) for f in split_files])
    source_scenario = get_scenario(data_identity.scenario)
    scenario = get_scenario(args.scenario) if args.scenario else source_scenario
    check_projection(source_scenario, scenario)
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
        *split_files, scenario, args.env_encoding,
        batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, drop_last=args.batchnorm, pin_memory=device.type == "cuda",
    )

    # Derive dims from data (guards against accidental mismatch with whatever
    # dataset schema is currently in train_file/val_file/test_file).
    # Peek once: each iter() on a DataLoader starts a fresh set of workers.
    x_peek, y_peek = next(iter(train_loader))
    input_dim, output_dim = x_peek.shape[1], y_peek.shape[1]
    del x_peek, y_peek

    if args.batchnorm and len(train_loader.dataset) < 2:
        raise ValueError(
            f"--batchnorm needs at least 2 training samples, got {len(train_loader.dataset)}."
        )

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

    logger.info("Model: scenario=%s, input_dim=%d (env_encoding=%s), output_dim=%d, hidden=%s, dropout=%s, batchnorm=%s, loss=%s",
                scenario.name, input_dim, args.env_encoding, output_dim, args.hidden_sizes, args.dropout, args.batchnorm,
                loss_label if args.loss != "huber" else f"Huber(delta={args.huber_delta})")

    model = SurrogateMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=torch.nn.ReLU,
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
            if counter == 0:
                # per-target loss of the first batch (same criterion as training)
                print(loss_fn(pred, y_vals, reduction="none").mean(dim=0))
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
                f"for extreme values (python -m fw.surrogate.diagnose_splits) or lower --lr."
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
                "ray_layout":           data_identity.ray_layout,
                "raw_input_dim":        scenario.raw_input_dim,
                "hidden_sizes":         args.hidden_sizes,
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


if __name__ == "__main__":
    from fw.common.tools.logging_config import setup_logging

    setup_logging(run_type="train_surrogate", env_name="surrogate", to_file=False)
    main()
