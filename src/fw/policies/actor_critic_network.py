import torch
import torch.nn as nn
import numpy as np

from typing import Tuple


class ActorCriticNetwork(nn.Module):
    """
    Actor-critic network: a shared trunk with an actor head (Gaussian action
    distribution) and a critic head (state value).

    Attributes:
        shared_network (nn.Sequential): Common layers for processing the input state.
        actor_mean (nn.Linear): Linear layer producing the mean of the action distribution.
        actor_log_std (nn.Parameter): Learnable parameter for the log standard deviation of actions.
        critic (nn.Linear): Linear layer producing the value estimate of the input state.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim (int): Dimension of the input state space.
            output_dim (int): Dimension of the output action space.
        """

        super().__init__()

        def init_layer(layer, gain=np.sqrt(2)):
            """
            Orthogonal weight init (with `gain`) and zero bias, in place; returns the layer.
            """

            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0)
            return layer

        self.shared_network = nn.Sequential(
            init_layer(nn.Linear(input_dim, 256)),
            nn.ReLU(),
            init_layer(nn.Linear(256, 128)),
            nn.ReLU(),
        )

        # Small gain: initial action means close to 0.
        self.actor_mean = init_layer(nn.Linear(128, output_dim), gain=0.01)

        # Initial std exp(-0.5) for stability.
        self.actor_log_std = nn.Parameter(torch.ones(output_dim) * -0.5)

        # Gain 1.0 is standard for the value head.
        self.critic = init_layer(nn.Linear(128, 1), gain=1.0)


    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - action_mean: mean of the Normal distribution (raw, not tanh-squashed;
                  PPOPolicy.predict()/calc_log_probs() squash the sampled action).
                - action_std: standard deviation of the action distribution.
                - state_value: estimated value of the input state.
        """

        shared_features = self.shared_network(state)

        action_mean = self.actor_mean(shared_features)

        # Clamp log_std before exponentiation for numerical stability.
        log_std = torch.clamp(self.actor_log_std, -20, 2)
        action_std = torch.exp(log_std)

        state_value = self.critic(shared_features)

        return action_mean, action_std, state_value

