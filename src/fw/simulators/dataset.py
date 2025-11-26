# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class ShipDynamicsDataset(Dataset):
    """
    Normalization strategy (Option A):
      - Inputs X normalized using x_mean/x_std
      - Targets Y normalized using y_mean/y_std
    """

    def __init__(
        self,
        path: str,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        y_mean: Optional[np.ndarray] = None,
        y_std: Optional[np.ndarray] = None,
        normalize: bool = True,
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        data = np.load(path)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)

        self.normalize = normalize

        if normalize:
            # Compute stats if not provided
            if x_mean is None:
                self.x_mean = self.X.mean(axis=0, keepdims=True)
                self.x_std = self.X.std(axis=0, keepdims=True) + 1e-8
                self.y_mean = self.Y.mean(axis=0, keepdims=True)
                self.y_std = self.Y.std(axis=0, keepdims=True) + 1e-8
            else:
                self.x_mean = x_mean
                self.x_std = x_std
                self.y_mean = y_mean
                self.y_std = y_std
        else:
            self.x_mean = None
            self.x_std = None
            self.y_mean = None
            self.y_std = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].reshape(-1)
        y = self.Y[idx].reshape(-1)

        if self.normalize:
            x = (x - self.x_mean.reshape(-1)) / self.x_std.reshape(-1)
            y = (y - self.y_mean.reshape(-1)) / self.y_std.reshape(-1)

        return torch.from_numpy(x), torch.from_numpy(y)

    @staticmethod
    def compute_stats(path: str):
        path = Path(path)
        data = np.load(path)
        X = data["X"].astype(np.float32)
        Y = data["Y"].astype(np.float32)

        x_mean = X.mean(axis=0, keepdims=True)
        x_std = X.std(axis=0, keepdims=True) + 1e-8
        y_mean = Y.mean(axis=0, keepdims=True)
        y_std = Y.std(axis=0, keepdims=True) + 1e-8

        return x_mean, x_std, y_mean, y_std
