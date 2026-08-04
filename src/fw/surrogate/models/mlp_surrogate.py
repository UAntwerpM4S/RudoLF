import torch
import torch.nn as nn

from typing import Sequence


class SurrogateMLP(nn.Module):
    """
    Configurable MLP for regression.

    Default architecture (tunable):
      input_dim -> [Linear -> Activation -> (BatchNorm) -> Dropout] x N -> Linear(output_dim)

    Final layer has no activation (regression).
    """

    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 3,
        hidden_sizes: Sequence[int] = (128, 128, 128),
        activation: nn.Module = nn.ReLU,
        use_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initialize the instance.

        Args:
            input_dim: Description.
            output_dim: Description.
            hidden_sizes: Description.
            activation: Description.
            use_batch_norm: Description.
            dropout: Description.
        """

        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward.

        Args:
            x: Description.
        """

        return self.net(x)

    def _initialize_weights(self):
        """
        Initialize weights.
        """

        # Xavier initialization for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
