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

# Kaiming nonlinearity name for ReLU-family activations (others get Xavier init).
# Values typed as a Literal to match nn.init.kaiming_uniform_'s annotation.
_KaimingNonlinearity = Literal["relu", "leaky_relu"]
_KAIMING_NONLINEARITY: Dict[Type[nn.Module], _KaimingNonlinearity] = {
    nn.ReLU: "relu",
    nn.LeakyReLU: "leaky_relu",
}


def activation_from_name(name: str) -> Type[nn.Module]:
    """
    Activation class for a name stored in a checkpoint (see ACTIVATIONS).
    """

    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(f"Unknown activation {name!r}; known: {sorted(ACTIVATIONS)}.") from None


class SurrogateMLP(nn.Module):
    """
    Configurable regression MLP:
      input_dim -> [Linear -> (BatchNorm) -> Activation -> (Dropout)] x N -> Linear(output_dim)

    No output activation. With BatchNorm, hidden Linear layers default to no bias.
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
        Args:
            input_dim: Number of input features (depends on scenario and env
                encoding, see schema.encoded_input_dim).
            output_dim: Number of regression targets.
            hidden_sizes: Width of each hidden layer.
            activation: Activation class applied after each hidden layer.
            use_batch_norm: Insert BatchNorm1d after each hidden Linear, before
                the activation.
            dropout: Dropout probability after each hidden activation (0 disables).
            hidden_bias: Bias on hidden Linear layers; None means
                `not use_batch_norm`. Inference passes what the checkpoint contains,
                so older BatchNorm checkpoints with biases still load.
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
        Map (N, input_dim) inputs to (N, output_dim) outputs.
        """

        return self.net(x)

    def _initialize_weights(self, activation: Type[nn.Module]):
        """
        Kaiming uniform for hidden layers with ReLU-family activations, Xavier
        uniform otherwise and for the output layer; zero biases.
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
