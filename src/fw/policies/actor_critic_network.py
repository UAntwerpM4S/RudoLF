import torch
import torch.nn as nn

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

    def __init__(self, input_dim: int, output_dim: int):
        """Initializes the ActorCriticNetwork.

        Args:
            input_dim (int): Dimension of the input state space.
            output_dim (int): Dimension of the output action space.
        """
        super().__init__()

        # Shared feature extraction layers
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Actor head: outputs mean of the action distribution
        self.actor_mean = nn.Linear(128, output_dim)

        # Actor head: learnable log standard deviation for actions
        self.actor_log_std = nn.Parameter(torch.zeros(output_dim))

        # Critic head: outputs the predicted value of the state
        self.critic = nn.Linear(128, 1)


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

        # Compute the standard deviation of the action distribution
        # Clamp log_std to reasonable range before exponentiation to avoid instability
        action_std = torch.exp(self.actor_log_std.clamp(-20, 2))

        # Compute the value estimate for the input state
        state_value = self.critic(shared_features)

        return action_mean, action_std, state_value
