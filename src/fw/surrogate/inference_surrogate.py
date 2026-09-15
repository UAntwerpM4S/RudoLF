import torch
import numpy as np

from fw.surrogate.dataset import ShipDynamicsDataset
from fw.surrogate.models.mlp_surrogate import SurrogateMLP


class SurrogatePredictor:
    """
    Loads trained surrogate model and returns UNNORMALIZED predictions.

    `predict` / `predict_single` take RAW rows straight from the dataset's
    "X" arrays (shape `(N, ShipDynamicsDataset.RAW_INPUT_DIM)`, i.e. 37
    columns: x, y, heading, surge, ..., ray_345). Column selection and
    angle sin/cos encoding are applied internally via
    `ShipDynamicsDataset._select_and_encode`, the same function used at
    training time -- callers should never pre-slice or pre-encode the
    input themselves, or they risk silently drifting out of sync with
    whatever `ShipDynamicsDataset` currently does (this previously
    happened: this class used to normalize whatever it was given with no
    selection/encoding at all, and a separate caller had its own stale
    hardcoded slice -- see plot_val_vs_simulator.py history).
    """

    def __init__(self, checkpoint, device=None):
        """
        Load a trained `SurrogateMLP` and its normalization stats.

        Args:
            checkpoint: Path to a checkpoint saved by `train_surrogate.py`
                (via `utils.save_checkpoint`), containing the model
                state dict, architecture args, and `x_mean`/`x_std`/
                `y_mean`/`y_std` normalization stats.
            device: Device to load the model onto; defaults to CUDA if
                available, else CPU.
        """

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)

        # normalization stats
        self.x_mean = ckpt["x_mean"].astype(np.float32)
        self.x_std  = ckpt["x_std"].astype(np.float32)
        self.y_mean = ckpt["y_mean"].astype(np.float32)
        self.y_std  = ckpt["y_std"].astype(np.float32)

        self.input_dim = ckpt["input_dim"]
        self.output_dim = ckpt["output_dim"]

        self.model = SurrogateMLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_sizes=tuple(ckpt["hidden_sizes"]),
            use_batch_norm=ckpt["use_batchnorm"],
            dropout=ckpt["dropout"],
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _normalize_x(self, x_vals):
        """
        Normalize raw input features using the loaded x_mean/x_std.

        Args:
            x_vals: Raw input array, shape `(N, input_dim)`.

        Returns:
            Normalized array, same shape as `x_vals`.
        """

        return (x_vals - self.x_mean) / self.x_std

    def _unnormalize_y(self, y_norm):
        """
        Convert normalized model outputs back to raw units.

        Args:
            y_norm: Normalized model output, shape `(N, output_dim)`.

        Returns:
            Un-normalized predictions, same shape as `y_norm`.
        """

        return y_norm * self.y_std + self.y_mean

    def predict(self, x_vals):
        """
        Predict outputs for one or more RAW input rows.

        Args:
            x_vals: Raw, unnormalized, un-selected input straight from a
                dataset "X" array -- `np.ndarray` of shape
                `(N, ShipDynamicsDataset.RAW_INPUT_DIM)` (37 columns), or
                `(RAW_INPUT_DIM,)` for a single sample. Do NOT pre-slice
                or pre-encode; that happens inside this method.

        Returns:
            `np.ndarray` of shape `(N, output_dim)` with un-normalized
            predictions.
        """

        if x_vals.ndim == 1:
            x_vals = x_vals.reshape(1, -1)

        x_selected = ShipDynamicsDataset._select_and_encode(x_vals.astype(np.float32))
        x_n = self._normalize_x(x_selected)

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_n).to(self.device)
            pred_norm = self.model(x_tensor).cpu().numpy()

        # UNNORMALIZE OUTPUTS
        pred_raw = self._unnormalize_y(pred_norm)
        return pred_raw

    def predict_single(self, x_vals):
        """
        Predict outputs for a single RAW input row.

        Args:
            x_vals: Raw unnormalized, un-selected input for one sample
                (a list or 1-D array of length
                `ShipDynamicsDataset.RAW_INPUT_DIM`, i.e. 37).

        Returns:
            `np.ndarray` of shape `(output_dim,)` with un-normalized
            predictions.
        """

        return self.predict(np.array(x_vals))[0]
