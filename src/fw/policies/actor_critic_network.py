import torch
import torch.nn as nn
import numpy as np

from typing import Tuple


class ActorCriticNetwork(nn.Module):
    """Actor-Critic neural network with shared feature extractor.

    This network has a shared trunk for feature extraction, and two heads:
    one for predicting the action distribution (actor) and one for estimating
    the state value (critic).

    Attributes:
        shared_network (nn.Sequential): Common layers for processing the input state.
        actor_mean (nn.Linear): Linear layer producing the mean of the action distribution.
        actor_log_std (nn.Parameter): Learnable parameter for the log standard deviation of actions.
        critic (nn.Linear): Linear layer producing the value estimate of the input state.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        """Initializes the ActorCriticNetwork.

        Args:
            input_dim (int): Dimension of the input state space.
            output_dim (int): Dimension of the output action space.
            log_std_min (float): Lower clamp on the learnable log-std before
                exponentiation. The default of ``-20`` matches the historical
                "effectively unbounded" behavior. Raise it (e.g. ``-1.0``) to
                floor the policy's exploration noise and prevent the Gaussian
                from collapsing toward a deterministic policy on tasks where
                rare-event coverage matters.
            log_std_max (float): Upper clamp on the learnable log-std before
                exponentiation. Default ``2.0`` (std ≈ 7.4); rarely needs
                changing.
        """
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Helper to initialize layers orthogonally
        def init_layer(layer, gain=np.sqrt(2)):
            """Init layer.

            Args:
                layer: Description.
                gain: Description.
            """
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0)
            return layer

        # Shared feature extraction layers
        self.shared_network = nn.Sequential(
            init_layer(nn.Linear(input_dim, 256)),
            nn.ReLU(),
            init_layer(nn.Linear(256, 128)),
            nn.ReLU(),
        )

        # Actor head: gain of 0.01 makes initial actions close to mean (near 0)
        self.actor_mean = init_layer(nn.Linear(128, output_dim), gain=0.01)

        # Actor head: start with a slightly smaller standard deviation for stability
        self.actor_log_std = nn.Parameter(torch.ones(output_dim) * -0.5)

        # Critic head: gain of 1.0 is standard for value estimation
        self.critic = init_layer(nn.Linear(128, 1), gain=1.0)


    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the forward pass.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - action_mean (torch.Tensor): Mean of the action distribution (after tanh activation).
                - action_std (torch.Tensor): Standard deviation of the action distribution.
                - state_value (torch.Tensor): Estimated value of the input state.
        """
        # Pass the state through the shared feature extractor
        shared_features = self.shared_network(state)

        # Compute the mean of the action distribution
        action_mean = torch.tanh(self.actor_mean(shared_features))

        log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
        action_std = torch.exp(log_std)

        # Compute the value estimate for the input state
        state_value = self.critic(shared_features)

        return action_mean, action_std, state_value

