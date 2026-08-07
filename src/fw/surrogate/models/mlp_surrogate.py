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
        Build the MLP layer stack.

        Args:
            input_dim: Number of input features.
            output_dim: Number of regression output targets.
            hidden_sizes: Width of each hidden layer, in order.
            activation: Activation module class applied after each hidden
                `Linear` layer (and after `BatchNorm1d`, when enabled).
            use_batch_norm: Whether to insert `BatchNorm1d` after each
                hidden `Linear` layer, before the activation.
            dropout: Dropout probability applied after each hidden layer's
                activation. `0.0` disables dropout.
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
        Run the MLP forward pass.

        Args:
            x: Input tensor of shape `(N, input_dim)`.

        Returns:
            torch.Tensor: Output tensor of shape `(N, output_dim)`.
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
