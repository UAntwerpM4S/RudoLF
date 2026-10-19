"""
SurrogatePredictor: loads a checkpoint written by train_surrogate.py and
returns UNNORMALIZED one-step predictions -- one implementation for every
scenario. The scenario, wind/current encoding, ship model and rudder scale
are read from the checkpoint, never assumed by the caller.
"""

import logging

import torch
import numpy as np

from fw.environments.ray_features import RAY_LAYOUT_VERSION
from fw.surrogate.schema import (
    SWAY_CONVENTION,
    TARGET_NAMES,
    check_env_encoding,
    encoded_feature_names,
    get_scenario,
    select_and_encode,
)
from fw.surrogate.models.mlp_surrogate import SurrogateMLP, activation_from_name
from fw.surrogate.utils import get_device

logger = logging.getLogger(__name__)

# Keys every checkpoint must have (written by train_surrogate.py).
REQUIRED_CHECKPOINT_KEYS = (
    "model_state_dict", "input_dim", "output_dim", "hidden_sizes", "use_batchnorm", "dropout",
    "x_mean", "x_std", "y_mean", "y_std", "target_names",
    "scenario", "ray_layout", "raw_input_dim", "input_cols", "angle_cols", "feature_names",
    "env_encoding", "ship_model", "rudder_max_deg", "sway_convention",
)


class SurrogatePredictor:
    """
    Trained surrogate MLP plus its normalization stats and identity.

    `predict` / `predict_single` take RAW rows in the layout of the
    checkpoint's scenario (`self.scenario.raw_columns`, width
    `self.raw_input_dim`), exactly as stored in a dataset "X" array. Column
    selection, encoding and normalization happen inside, with the same
    `schema.select_and_encode` used at training time; callers must never
    pre-slice or pre-encode the input themselves.

    Attributes:
        scenario: schema.ScenarioSchema of the checkpoint.
        env_encoding: Wind/current encoding the model was trained with.
        ship_model: ShipModel name the training data was simulated for.
        rudder_max_deg: Rudder angle [deg] that rudder_action = 1 stands for.
        raw_input_dim: Width of a raw input row.
        input_dim / output_dim: Model widths.
    """

    def __init__(self, checkpoint, device=None):
        """
        Load a trained `SurrogateMLP` and its normalization stats.

        Args:
            checkpoint: Path to a checkpoint saved by `train_surrogate.py`.
            device: Device to load the model onto; defaults to
                utils.get_device() (CUDA if available, else CPU).

        Raises:
            ValueError: If a required key is missing, or the checkpoint's
                feature layout does not match the current schema.
        """

        self.checkpoint = str(checkpoint)
        self.device = device or get_device()
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)

        missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in ckpt]
        if missing:
            raise ValueError(
                f"Checkpoint {checkpoint} has no {missing}: it was not written by the current "
                f"train_surrogate.py. Retrain it on a dataset from the current generator."
            )

        self.scenario = get_scenario(ckpt["scenario"])
        self.env_encoding = check_env_encoding(ckpt["env_encoding"])
        self.ship_model = str(ckpt["ship_model"])
        self.rudder_max_deg = float(ckpt["rudder_max_deg"])
        self.raw_input_dim = self.scenario.raw_input_dim
        self.input_dim = int(ckpt["input_dim"])
        self.output_dim = int(ckpt["output_dim"])
        self._check_feature_set(ckpt)

        self.x_mean = np.asarray(ckpt["x_mean"], np.float32).reshape(1, -1)
        self.x_std = np.asarray(ckpt["x_std"], np.float32).reshape(1, -1)
        self.y_mean = np.asarray(ckpt["y_mean"], np.float32).reshape(1, -1)
        self.y_std = np.asarray(ckpt["y_std"], np.float32).reshape(1, -1)
        for name, stat, width in (("x_mean", self.x_mean, self.input_dim), ("x_std", self.x_std, self.input_dim),
                                  ("y_mean", self.y_mean, self.output_dim), ("y_std", self.y_std, self.output_dim)):
            if stat.shape[1] != width:
                raise ValueError(f"Checkpoint {checkpoint}: {name} has {stat.shape[1]} columns, expected {width}.")

        # Checkpoints without "activation" were all trained with ReLU (it was
        # hardcoded in train_surrogate.py before it was stored).
        self.activation = str(ckpt.get("activation", "ReLU"))
        self.model = SurrogateMLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_sizes=tuple(ckpt["hidden_sizes"]),
            activation=activation_from_name(self.activation),
            # Older BatchNorm checkpoints still have hidden Linear biases.
            hidden_bias="net.0.bias" in ckpt["model_state_dict"],
            use_batch_norm=ckpt["use_batchnorm"],
            dropout=ckpt["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _check_feature_set(self, ckpt) -> None:
        """Fail loudly if the checkpoint does not match the current schema."""

        names = list(encoded_feature_names(self.scenario, self.env_encoding))
        expected = {
            "ray_layout": RAY_LAYOUT_VERSION,
            "sway_convention": SWAY_CONVENTION,
            "raw_input_dim": self.scenario.raw_input_dim,
            "input_cols": list(self.scenario.input_col_indices()),
            "angle_cols": list(self.scenario.angle_col_indices()),
            "feature_names": names,
            "input_dim": len(names),
            "output_dim": len(TARGET_NAMES),
            "target_names": list(TARGET_NAMES),
        }
        for key, value in expected.items():
            found = [x.item() if hasattr(x, "item") else x for x in ckpt[key]] if isinstance(value, list) else ckpt[key]
            if found != value:
                raise ValueError(
                    f"Checkpoint {self.checkpoint} (scenario '{self.scenario.name}', env_encoding "
                    f"{self.env_encoding!r}) has {key}={found!r}, but the current schema gives "
                    f"{value!r}. Retrain on a dataset from the current generator."
                )

    def predict(self, x_vals) -> np.ndarray:
        """
        Predict outputs for one or more RAW input rows.

        Args:
            x_vals: Raw, unnormalized input in the checkpoint scenario's raw
                layout: shape `(N, raw_input_dim)`, or `(raw_input_dim,)` for
                a single sample.

        Returns:
            `np.ndarray` of shape `(N, output_dim)` with un-normalized
            predictions (deltas over schema.SURROGATE_DT).
        """

        x_vals = np.asarray(x_vals, dtype=np.float32)
        if x_vals.ndim == 1:
            x_vals = x_vals.reshape(1, -1)

        x_n = ((select_and_encode(x_vals, self.scenario, self.env_encoding) - self.x_mean) / self.x_std)
        with torch.no_grad():
            pred_norm = self.model(torch.from_numpy(x_n.astype(np.float32)).to(self.device)).cpu().numpy()
        return pred_norm * self.y_std + self.y_mean

    def predict_single(self, x_vals) -> np.ndarray:
        """
        Predict outputs for a single RAW input row (length `raw_input_dim`).

        Returns:
            `np.ndarray` of shape `(output_dim,)`.
        """

        return self.predict(np.asarray(x_vals))[0]
