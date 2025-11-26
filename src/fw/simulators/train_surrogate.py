# train_surrogate.py
import argparse
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ShipDynamicsDataset
from models.mlp_surrogate import SurrogateMLP


def data_path(*relative):
    """Return an absolute path inside the local data/ folder (project-local)."""
    base = Path(__file__).resolve().parent
    return base / "data" / Path(*relative)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
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
    # compute input stats from train file and ensure same normalization applied to val/test
    # in make_dataloaders():
    x_mean, x_std, y_mean, y_std = ShipDynamicsDataset.compute_stats(train_file)

    train_ds = ShipDynamicsDataset(train_file, x_mean, x_std, y_mean, y_std, normalize=True)
    val_ds = ShipDynamicsDataset(val_file, x_mean, x_std, y_mean, y_std, normalize=True)
    test_ds = ShipDynamicsDataset(test_file, x_mean, x_std, y_mean, y_std, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    parser = argparse.ArgumentParser()
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

    input_dim = next(iter(train_loader))[0].shape[1]
    output_dim = next(iter(train_loader))[1].shape[1]

    model = SurrogateMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=torch.nn.ReLU,
        dropout=args.dropout,
        use_batchnorm=args.batchnorm,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, verbose=True)

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, Y, reduction="mean")
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)
            n_samples += X.size(0)

        train_epoch_loss = epoch_loss / n_samples
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} - train_loss: {train_epoch_loss:.6f}, val_loss: {val_loss:.6f}")

        # scheduler step based on validation loss
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            # save model state + training metadata
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_sizes": args.hidden_sizes,
                "dropout": args.dropout,
                "use_batchnorm": args.batchnorm,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }, ckpt_path)
            print(f"  ✔ new best (val_loss={val_loss:.6f}) saved to {ckpt_path}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping: no improvement in {args.patience} epochs (best epoch {best_epoch}, best val_loss {best_val:.6f})")
            break

    # Load best and evaluate on test set
    checkpoint = load_checkpoint(ckpt_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.6f} (best val_loss: {checkpoint.get('val_loss', float('nan')):.6f})")


if __name__ == "__main__":
    main()
