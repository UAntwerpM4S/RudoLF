# train_surrogate.py
"""
Training script for the ship dynamics surrogate MLP.

Input  (8-D): [x, y, heading, surge, sway, yaw_rate, rudder_action, thrust_action]
Output (3-D): [delta_surge, delta_sway, delta_yaw_rate]
"""
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader
from fw.surrogate.dataset import ShipDynamicsDataset
from fw.surrogate.models.mlp_surrogate import SurrogateMLP
from fw.surrogate.utils import load_checkpoint, data_path, get_device, save_checkpoint


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute mean MSE loss over a DataLoader.

    Args:
        model:  Trained (or in-training) SurrogateMLP.
        loader: DataLoader yielding normalized (X, Y) batches.
        device: Torch device.

    Returns:
        Mean MSE loss (scalar).
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            loss = F.mse_loss(pred, Y, reduction="mean")
            total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def make_dataloaders(train_file, val_file, test_file, batch_size=256, num_workers=0):
    """Build train / val / test DataLoaders with shared normalization stats.

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
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    """Entry point for surrogate training.

    Args:
        argv: Optional list of CLI arguments (useful for testing).
    """
    parser = argparse.ArgumentParser(
        description="Train MLP surrogate: 8 inputs -> 3 outputs"
    )
    parser.add_argument("--train-file",    default=data_path("ship_dynamics_dataset_train.npz"))
    parser.add_argument("--val-file",      default=data_path("ship_dynamics_dataset_val.npz"))
    parser.add_argument("--test-file",     default=data_path("ship_dynamics_dataset_test.npz"))
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--epochs",        type=int,   default=300)
    parser.add_argument("--patience",      type=int,   default=25,
                        help="Early-stopping patience (epochs without improvement)")
    parser.add_argument("--dropout",       type=float, default=0.05)
    parser.add_argument("--hidden-sizes",  nargs="+",  type=int, default=[128, 128])
    parser.add_argument("--batchnorm",     action="store_true")
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--num-workers",   type=int,   default=0)
    parser.add_argument("--checkpoint",    type=str,   default="checkpoints/best_surrogate.pth")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()

    train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std = make_dataloaders(
        str(args.train_file), str(args.val_file), str(args.test_file),
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Derive dims from data (guards against accidental mismatch)
    input_dim  = next(iter(train_loader))[0].shape[1]   # should be 8
    output_dim = next(iter(train_loader))[1].shape[1]   # should be 3

    print(f"Model: input_dim={input_dim}, output_dim={output_dim}, "
          f"hidden={args.hidden_sizes}, dropout={args.dropout}, "
          f"batchnorm={args.batchnorm}")

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

        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, Y, reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)
            n_samples  += X.size(0)

        train_epoch_loss = epoch_loss / n_samples
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} – train: {train_epoch_loss:.6f}  val: {val_loss:.6f}")

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
            print(f"  ✔ new best (val_loss={val_loss:.6f}) saved to {ckpt_path}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(
                f"Early stopping: no improvement in {args.patience} epochs "
                f"(best epoch {best_epoch}, best val_loss {best_val:.6f})"
            )
            break

    # Final evaluation on held-out test set
    checkpoint = load_checkpoint(ckpt_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, device)
    print(
        f"\nFinal Test Loss: {test_loss:.6f} "
        f"(best val_loss: {checkpoint.get('val_loss', float('nan')):.6f})"
    )


if __name__ == "__main__":
    main()
