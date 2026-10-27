import torch
import torch.nn as nn

from typing import Dict, Literal, Optional, Sequence, Type

# Activations a checkpoint may name (train_surrogate.py stores the name,
# inference_surrogate.py resolves it here).
ACTIVATIONS: Dict[str, Type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
}

# Kaiming nonlinearity name for ReLU-family activations; anything else gets
# Xavier init. Keys typed as the activation classes passed around (so lookups
# with a Type[nn.Module] type-check); values as a Literal, because
# nn.init.kaiming_uniform_ annotates `nonlinearity` with a Literal of names.
_KaimingNonlinearity = Literal["relu", "leaky_relu"]
_KAIMING_NONLINEARITY: Dict[Type[nn.Module], _KaimingNonlinearity] = {
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
}


def activation_from_name(name: str) -> Type[nn.Module]:
    """Activation class for a name stored in a checkpoint (see ACTIVATIONS)."""

    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(f"Unknown activation {name!r}; known: {sorted(ACTIVATIONS)}.") from None


class SurrogateMLP(nn.Module):
    """
    Configurable MLP for regression.

    Default architecture (tunable):
      input_dim -> [Linear -> (BatchNorm) -> Activation -> (Dropout)] x N -> Linear(output_dim)

    Final layer has no activation (regression). With BatchNorm, the hidden
    Linear layers have no bias (BatchNorm's own shift replaces it).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        hidden_sizes: Sequence[int] = (256, 256, 128),
        activation: Type[nn.Module] = nn.ReLU,
        use_batch_norm: bool = False,
        dropout: float = 0.0,
        hidden_bias: Optional[bool] = None,
    ):
        """
        Build the MLP layer stack.

        Args:
            input_dim: Number of input features (required: it depends on the
                scenario and env encoding, see schema.encoded_input_dim).
            output_dim: Number of regression output targets.
            hidden_sizes: Width of each hidden layer, in order.
            activation: Activation module class applied after each hidden
                `Linear` layer (and after `BatchNorm1d`, when enabled).
            use_batch_norm: Whether to insert `BatchNorm1d` after each
                hidden `Linear` layer, before the activation.
            dropout: Dropout probability applied after each hidden layer's
                activation. `0.0` disables dropout.
            hidden_bias: Bias on the hidden Linear layers. None (default) =
                `not use_batch_norm`. Inference passes what the checkpoint's
                weights contain, so BatchNorm models trained when the bias
                was still there keep loading.
        """

        super().__init__()

        self.activation_name = activation.__name__
        if hidden_bias is None:
            hidden_bias = not use_batch_norm
        layers = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h, bias=hidden_bias))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

        self._initialize_weights(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the MLP forward pass.

        Args:
            x: Input tensor of shape `(N, input_dim)`.

        Returns:
            torch.Tensor: Output tensor of shape `(N, output_dim)`.
        """

        return self.net(x)

    def _initialize_weights(self, activation: Type[nn.Module]):
        """
        Hidden Linear layers: Kaiming (He) uniform for ReLU-family
        activations, Xavier uniform otherwise. Output layer: Xavier uniform.
        Biases zero. BatchNorm keeps PyTorch's default (weight 1, bias 0).
        """

        nonlinearity = _KAIMING_NONLINEARITY.get(activation)
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for i, m in enumerate(linears):
            if i < len(linears) - 1 and nonlinearity is not None:
                nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
