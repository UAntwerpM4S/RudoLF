import torch
import numpy as np

from models.mlp_surrogate import SurrogateMLP


class SurrogatePredictor:
    """
    Loads trained surrogate model and returns UNNORMALIZED predictions.
    """

    def __init__(self, checkpoint, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=self.device)

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

    def _normalize_x(self, X):
        return (X - self.x_mean) / self.x_std

    def _unnormalize_y(self, Ynorm):
        return Ynorm * self.y_std + self.y_mean

    def predict(self, X):
        """
        X: np.ndarray (N, input_dim) raw unnormalized input data
        Returns: np.ndarray (N, output_dim) UNnormalized predictions
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Xn = self._normalize_x(X.astype(np.float32))

        with torch.no_grad():
            X_tensor = torch.from_numpy(Xn).to(self.device)
            pred_norm = self.model(X_tensor).cpu().numpy()

        # UNNORMALIZE OUTPUTS
        pred_raw = self._unnormalize_y(pred_norm)
        return pred_raw

    def predict_single(self, X):
        return self.predict(np.array(X))[0]
