import torch
import numpy as np

from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset


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
        """
        Load a `.npz` dataset (keys "X", "Y") and set up normalization.

        If `normalize` is True and `x_mean` is None, per-feature mean/std
        are computed from this dataset's own `X`/`Y` (std has `1e-8`
        added to avoid division by zero); otherwise the given
        `x_mean`/`x_std`/`y_mean`/`y_std` are used as-is. Note only
        `x_mean` is checked to decide which path to take - callers should
        pass all four stats together or none of them.

        Args:
            path: Path to the `.npz` file containing `X` and `Y` arrays.
            x_mean: Precomputed per-feature input mean; if None (and
                `normalize` is True), computed from this dataset's `X`.
            x_std: Precomputed per-feature input std (paired with
                `x_mean`).
            y_mean: Precomputed per-feature target mean (paired with
                `x_mean`).
            y_std: Precomputed per-feature target std (paired with
                `x_mean`).
            normalize: If False, `__getitem__` returns raw `X`/`Y` values
                and no stats are computed or stored.

        Raises:
            FileNotFoundError: If `path` does not exist.
        """

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
        """
        Return the number of samples in the dataset.
        """

        return len(self.X)

    def __getitem__(self, idx):
        """
        Return the (optionally normalized) sample at `idx`.

        Args:
            idx: Sample index into `X`/`Y`.

        Returns:
            `(x, y)` tensors, normalized using the stored stats if
            `self.normalize` is True.
        """

        x = self.X[idx].reshape(-1)
        y = self.Y[idx].reshape(-1)

        if self.normalize:
            x = (x - self.x_mean.reshape(-1)) / self.x_std.reshape(-1)
            y = (y - self.y_mean.reshape(-1)) / self.y_std.reshape(-1)

        return torch.from_numpy(x), torch.from_numpy(y)

    @staticmethod
    def compute_stats(path: str):
        """
        Compute per-feature mean/std for X and Y from a `.npz` dataset.

        Args:
            path: Path to the `.npz` file containing `X` and `Y` arrays.

        Returns:
            `(x_mean, x_std, y_mean, y_std)`, each shaped
            `(1, num_features)`; std has `1e-8` added to avoid division
            by zero.
        """

        path = Path(path)
        data = np.load(path)
        x_vals = data["X"].astype(np.float32)
        y_vals = data["Y"].astype(np.float32)

        x_mean = x_vals.mean(axis=0, keepdims=True)
        x_std = x_vals.std(axis=0, keepdims=True) + 1e-8
        y_mean = y_vals.mean(axis=0, keepdims=True)
        y_std = y_vals.std(axis=0, keepdims=True) + 1e-8

        return x_mean, x_std, y_mean, y_std
