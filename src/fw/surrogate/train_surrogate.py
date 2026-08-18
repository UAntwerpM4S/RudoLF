# train_surrogate.py
"""
Training script for the ship dynamics surrogate MLP.

Input  (8-D): [x, y, heading, surge, sway, yaw_rate, rudder_action, thrust_action]
Output (3-D): [delta_surge, delta_sway, delta_yaw_rate]
"""

import torch
import argparse
import logging
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader
from fw.surrogate.dataset import ShipDynamicsDataset
from fw.surrogate.models.mlp_surrogate import SurrogateMLP
from fw.surrogate.utils import load_checkpoint, data_path, get_device, save_checkpoint

logger = logging.getLogger(__name__)


def evaluate(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        y_mean: np.ndarray,
        y_std: np.ndarray,
):
    """
    Compute normalized MSE loss and denormalized MAE.

    Args:
        model: Trained (or in-training) SurrogateMLP.
        loader: DataLoader yielding normalized (X, Y) batches.
        device: Torch device.
        y_mean: Output normalization mean.
        y_std: Output normalization std.

    Returns:
        Tuple:
            mean_mse_loss: Mean normalized MSE loss.
            mae: Mean absolute error in original physical units
                 for each output dimension.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_abs_error = None

    y_mean_t = torch.as_tensor(
        y_mean,
        dtype=torch.float32,
        device=device,
    )

    y_std_t = torch.as_tensor(
        y_std,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        for x_vals, y_vals in loader:
            x_vals = x_vals.to(device)
            y_vals = y_vals.to(device)

            pred = model(x_vals)

            loss = F.mse_loss(pred, y_vals, reduction="mean")
            total_loss += loss.item() * x_vals.size(0)

            pred_phys = pred * y_std_t + y_mean_t
            y_phys = y_vals * y_std_t + y_mean_t

            abs_error = torch.abs(pred_phys - y_phys)

            if total_abs_error is None:
                total_abs_error = abs_error.sum(dim=0)
            else:
                total_abs_error += abs_error.sum(dim=0)

            total_samples += x_vals.size(0)

    mean_mse_loss = total_loss / total_samples
    mae = total_abs_error / total_samples

    return mean_mse_loss, mae.cpu().numpy()


def make_dataloaders(train_file, val_file, test_file, batch_size=256, num_workers=0):
    """
    Build train / val / test DataLoaders with shared normalization stats.

    Normalization statistics are computed from the training split only and
    then re-applied identically to the validation and test splits.

    Args:
        train_file:  Path to training .npz file.
        val_file:    Path to validation .npz file.
        test_file:   Path to test .npz file.
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader, test_loader,
                  x_mean, x_std, y_mean, y_std).
    """

    x_mean, x_std, y_mean, y_std = ShipDynamicsDataset.compute_stats(train_file)

    train_ds = ShipDynamicsDataset(train_file, x_mean, x_std, y_mean, y_std, normalize=True)
    val_ds   = ShipDynamicsDataset(val_file,   x_mean, x_std, y_mean, y_std, normalize=True)
    test_ds  = ShipDynamicsDataset(test_file,  x_mean, x_std, y_mean, y_std, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    """
    Entry point for surrogate training.

    Args:
        argv: Optional list of CLI arguments (useful for testing).
    """

    parser = argparse.ArgumentParser(
        description="Train MLP surrogate: 8 inputs -> 3 outputs"
    )
    parser.add_argument("--train-file", default=data_path("ship_dynamics_dataset_train.npz"))
    parser.add_argument("--val-file", default=data_path("ship_dynamics_dataset_val.npz"))
    parser.add_argument("--test-file", default=data_path("ship_dynamics_dataset_test.npz"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)  # early stopping patience
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_surrogate.pth")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()

    train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std = make_dataloaders(
        str(args.train_file), str(args.val_file), str(args.test_file),
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Derive dims from data (guards against accidental mismatch with whatever
    # dataset schema is currently in train_file/val_file/test_file)
    input_dim  = next(iter(train_loader))[0].shape[1]
    output_dim = next(iter(train_loader))[1].shape[1]

    logger.info("Model: input_dim=%d, output_dim=%d, hidden=%s, dropout=%s, batchnorm=%s",
                input_dim, output_dim, args.hidden_sizes, args.dropout, args.batchnorm)

    model = SurrogateMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=torch.nn.ReLU,
        use_batch_norm=args.batchnorm,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    best_val  = float("inf")
    best_epoch = -1
    no_improve = 0
    ckpt_path  = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples  = 0

        for x_vals, y_vals in train_loader:
            x_vals = x_vals.to(device)
            y_vals = y_vals.to(device)
            optimizer.zero_grad()
            pred = model(x_vals)
            loss = F.mse_loss(pred, y_vals, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item() * x_vals.size(0)
            n_samples  += x_vals.size(0)

        train_epoch_loss = epoch_loss / n_samples

        val_loss, val_mae = evaluate(
            model,
            val_loader,
            device,
            y_mean,
            y_std,
        )

        mae_str = ", ".join(f"{v:.6f}" for v in val_mae)

        logger.info(
            "Epoch %03d - train: %.6f  val: %.6f  MAE=[%s]",
            epoch, train_epoch_loss, val_loss, mae_str
        )

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-9:
            best_val   = val_loss
            best_epoch = epoch
            no_improve = 0
            save_checkpoint({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":          val_loss,
                "input_dim":         input_dim,
                "output_dim":        output_dim,
                "hidden_sizes":      args.hidden_sizes,
                "dropout":           args.dropout,
                "use_batchnorm":     args.batchnorm,
                # Normalization stats – required by SurrogatePredictor at inference
                "x_mean": x_mean,
                "x_std":  x_std,
                "y_mean": y_mean,
                "y_std":  y_std,
            }, ckpt_path)
            logger.info("  \u2714 new best (val_loss=%.6f) saved to %s", val_loss, ckpt_path)
        else:
            no_improve += 1

        if no_improve >= args.patience:
            logger.info(
                "Early stopping: no improvement in %d epochs "
                "(best epoch %d, best val_loss %.6f)", args.patience, best_epoch, best_val
            )
            break

    # Final evaluation on held-out test set
    checkpoint = load_checkpoint(ckpt_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_mae = evaluate(
        model,
        test_loader,
        device,
        y_mean,
        y_std,
    )

    target_names = [
        "delta_surge",
        "delta_sway",
        "delta_yaw_rate",
    ]

    print("\nFinal Test Results")
    print("------------------")
    print(f"Test Loss (normalized MSE): {test_loss:.6f}")
    print(f"Best Validation Loss      : {checkpoint.get('val_loss', float('nan')):.6f}")

    for name, mae in zip(target_names, test_mae):
        print(f"MAE {name:<15}: {mae:.6f}")


if __name__ == "__main__":
    from fw.common.tools.logging_config import setup_logging

    setup_logging(run_type="train_surrogate", env_name="surrogate", to_file=False)
    main()
