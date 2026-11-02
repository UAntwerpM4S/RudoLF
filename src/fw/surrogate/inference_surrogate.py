"""
SurrogatePredictor: loads a train_surrogate.py checkpoint and returns
unnormalized one-step predictions for any scenario. Scenario, wind/current
encoding, ship model and rudder scale come from the checkpoint.
"""

import logging

from typing import Optional

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

    predict / predict_single take raw rows in the scenario's layout (width
    raw_input_dim), as in a dataset "X" array. Selection, encoding and
    normalization happen inside (schema.select_and_encode, as in training);
    callers must not pre-slice or pre-encode.

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
        Args:
            checkpoint: Path to a train_surrogate.py checkpoint.
            device: Torch device; default utils.get_device().

        Raises:
            ValueError: If a required key is missing or the feature layout does
                not match the current schema.
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

        # Checkpoints without "activation" were trained with ReLU.
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

    def data_mismatch(self, identity) -> Optional[str]:
        """
        Reason why data with this schema.DatasetIdentity cannot be evaluated with
        the model (other ship or rudder scale), or None. The scenario is checked
        separately (schema.check_projection).
        """

        if str(identity.ship_model) != self.ship_model:
            return f"the data is {identity.ship_model} data, but the checkpoint is a {self.ship_model} model"
        if abs(float(identity.rudder_max_deg) - self.rudder_max_deg) > 1e-6:
            return (f"the data's rudder scale is {float(identity.rudder_max_deg):g} deg, the checkpoint's "
                    f"{self.rudder_max_deg:g} deg (rudder_action = 1 would mean a different angle)")
        return None

    def _check_feature_set(self, ckpt) -> None:
        """
        Fail loudly if the checkpoint does not match the current schema.
        """

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
        Predict for raw input rows, shape (N, raw_input_dim) or (raw_input_dim,).

        Returns:
            (N, output_dim) unnormalized deltas over schema.SURROGATE_DT.
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
        Predict for one raw row; returns shape (output_dim,).
        """

        return self.predict(np.asarray(x_vals))[0]
