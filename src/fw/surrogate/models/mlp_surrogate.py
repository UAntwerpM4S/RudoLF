import torch
import torch.nn as nn

from typing import Sequence


class SurrogateMLP(nn.Module):
    """
    Configurable MLP for ship dynamics regression.

    Input  (8-D): [x, y, heading, surge, sway, yaw_rate, rudder_action, thrust_action]
    Output (3-D): [delta_surge, delta_sway, delta_yaw_rate]

    Architecture:
      input_dim -> [Linear -> Activation -> (BatchNorm) -> Dropout] x N -> Linear(output_dim)

    Final layer has no activation (regression).
    """

    def __init__(
        self,
        input_dim: int = 8,    # x, y, heading, surge, sway, yaw_rate, rudder, thrust
        output_dim: int = 3,   # delta_surge, delta_sway, delta_yaw_rate
        hidden_sizes: Sequence[int] = (128, 128, 128),
        activation: nn.Module = nn.ReLU,
        use_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        """Initialize the instance.

        Args:
            input_dim:       Number of input features (default 8).
            output_dim:      Number of output targets (default 3).
            hidden_sizes:    Sequence of hidden layer widths.
            activation:      Activation class (instantiated per layer).
            use_batch_norm:  Whether to insert BatchNorm1d after each linear layer.
            dropout:         Dropout probability (0.0 disables dropout).
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
        """Forward.

        Args:
            x: Input tensor of shape (batch, input_dim).
        """
        return self.net(x)

    def _initialize_weights(self):
        """Xavier initialization for linear layers, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
