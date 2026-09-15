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

    Column selection strategy:
      Raw "X" arrays are 37-wide, matching
      sim_fh_dataset_generator.collect_supervised_dataset's column layout
      (x, y, heading, surge, sway, yaw_rate, UKC, wind_speed, wind_dir,
      current_speed, current_dir, rudder_action, thrust_action, ray_0..ray_345).
      `_select_and_encode` is the single source of truth for which raw
      columns become model inputs, and is used by both `__init__` and
      `compute_stats` so the two can never see a different feature set.

      - x, y (world position) are dropped: they have ~0 correlation with
        the one-step dynamics deltas and would just add noise/overfitting
        risk for a body-frame dynamics model.
      - heading, wind_dir, current_dir are angles in [-pi, pi] and are
        sin/cos encoded rather than passed raw, since a raw radian value
        has a discontinuity at +-pi that an MLP cannot represent (e.g.
        headings of 3.13 and -3.13 rad are ~0.01 rad apart physically but
        ~6.28 apart numerically).
      - All 24 ray proximity features are kept (collision-influenced
        dynamics are in scope for this model).
    """

    COL_X = 0
    COL_Y = 1
    COL_HEADING = 2
    COL_SURGE = 3
    COL_SWAY = 4
    COL_YAW_RATE = 5
    COL_UKC = 6
    COL_WIND_SPEED = 7
    COL_WIND_DIR = 8
    COL_CURRENT_SPEED = 9
    COL_CURRENT_DIR = 10
    COL_RUDDER = 11
    COL_THRUST = 12
    RAY_COLS = list(range(13, 37))  # 24 ray proximity features

    RAW_INPUT_DIM = 37  # width of the raw "X" arrays this class consumes

    # Raw columns selected as model inputs, in order. x/y intentionally
    # excluded (see class docstring).
    INPUT_COLS = (
        [COL_HEADING, COL_SURGE, COL_SWAY, COL_YAW_RATE, COL_UKC,
         COL_WIND_SPEED, COL_WIND_DIR, COL_CURRENT_SPEED, COL_CURRENT_DIR,
         COL_RUDDER, COL_THRUST]
        + RAY_COLS
    )

    # Subset of INPUT_COLS that are angles in [-pi, pi] and must be
    # sin/cos encoded instead of passed through raw.
    ANGLE_COLS = [COL_HEADING, COL_WIND_DIR, COL_CURRENT_DIR]

    @classmethod
    def select_and_encode(cls, x_raw: np.ndarray) -> np.ndarray:
        """
        Select `INPUT_COLS` from a raw 37-wide X block and sin/cos-encode
        the angular columns.

        This is the ONLY place raw-column selection happens; both
        `__init__` and `compute_stats` call this so they are guaranteed to
        agree on the feature set (fixes the previous bug where the two
        used separately hardcoded, inconsistent slices).

        Args:
            x_raw: Raw input array, shape `(N, 37)`.

        Returns:
            Feature array, shape `(N, len(INPUT_COLS) + len(ANGLE_COLS))`:
            non-angle INPUT_COLS in order, followed by sin(angle) for each
            ANGLE_COLS entry, followed by cos(angle) for each ANGLE_COLS
            entry.

        Raises:
            ValueError: If `x_raw` is not shaped `(N, RAW_INPUT_DIM)`.
        """

        if x_raw.ndim != 2 or x_raw.shape[1] != cls.RAW_INPUT_DIM:
            raise ValueError(
                f"Expected raw X with shape (N, {cls.RAW_INPUT_DIM}), "
                f"got {x_raw.shape}"
            )

        selected = x_raw[:, cls.INPUT_COLS]
        angle_mask = np.isin(cls.INPUT_COLS, cls.ANGLE_COLS)

        non_angle_vals = selected[:, ~angle_mask]
        angle_vals = selected[:, angle_mask]

        return np.concatenate(
            [non_angle_vals, np.sin(angle_vals), np.cos(angle_vals)],
            axis=1,
        ).astype(np.float32)

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
        self.X = self.select_and_encode(data["X"].astype(np.float32))
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
        x_vals = ShipDynamicsDataset.select_and_encode(data["X"].astype(np.float32))
        y_vals = data["Y"].astype(np.float32)

        x_mean = x_vals.mean(axis=0, keepdims=True)
        x_std = x_vals.std(axis=0, keepdims=True) + 1e-8
        y_mean = y_vals.mean(axis=0, keepdims=True)
        y_std = y_vals.std(axis=0, keepdims=True) + 1e-8

        return x_mean, x_std, y_mean, y_std
