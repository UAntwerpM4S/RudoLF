import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import ShipDynamicsDataset
from models.mlp_surrogate import SurrogateMLP
from utils import data_path
from pathlib import Path

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = F.mse_loss(pred, Y)
            total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths resolved relative to THIS project folder
    train_file = data_path("ship_dynamics_dataset_train.npz")
    val_file   = data_path("ship_dynamics_dataset_val.npz")
    test_file  = data_path("ship_dynamics_dataset_test.npz")

    # Load datasets
    train_ds = ShipDynamicsDataset(train_file)
    val_ds   = ShipDynamicsDataset(val_file)
    test_ds  = ShipDynamicsDataset(test_file)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=256)
    test_loader  = DataLoader(test_ds, batch_size=256)

    model = SurrogateMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    ckpt_path = Path("checkpoints/best_surrogate.pth")
    ckpt_path.parent.mkdir(exist_ok=True)

    for epoch in range(1, 101):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, Y)
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d}: val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path, _use_new_zipfile_serialization=True)
            print(f"  ✔ new best model saved to {ckpt_path}")

    # ---- Test evaluation ----
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    test_loss = evaluate(model, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.6f}")

if __name__ == "__main__":
    main()
