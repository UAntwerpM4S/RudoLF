import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class ShipDynamicsDataset(Dataset):
    def __init__(self, path, normalize=True):
        # Ensure path is a Path object → solves string vs Path issues
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        data = np.load(path)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)

        self.normalize = normalize

        if normalize:
            self.x_mean = self.X.mean(axis=0, keepdims=True)
            self.x_std = self.X.std(axis=0, keepdims=True) + 1e-8

            self.y_mean = self.Y.mean(axis=0, keepdims=True)
            self.y_std = self.Y.std(axis=0, keepdims=True) + 1e-8
        else:
            self.x_mean = None
            self.x_std = None
            self.y_mean = None
            self.y_std = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.normalize:
            x = (x - self.x_mean) / self.x_std
            y = (y - self.y_mean) / self.y_std

        return torch.from_numpy(x), torch.from_numpy(y)
