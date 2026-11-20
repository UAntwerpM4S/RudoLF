import torch
import torch.nn as nn
import numpy as np

from typing import Tuple


class _StraightThroughClamp(torch.autograd.Function):
    """
    Clamp whose backward passes the gradient through unchanged (torch.clamp
    passes none at or beyond a bound). The forward value equals torch.clamp
    for every input, including +-inf.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """
        Clamp x into [low, high].
        """

        return torch.clamp(x, low, high)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Identity gradient for x; none for the bounds.
        """

        return grad_outputs[0], None, None


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

    # Bounds of actor_log_std, enforced on the value in action_std() and on the
    # parameter itself by clamp_log_std_().
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

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

        action_std = self.action_std()

        state_value = self.critic(shared_features)

        return action_mean, action_std, state_value


    def action_std(self) -> torch.Tensor:
        """
        Standard deviation of the (pre-tanh) action distribution, shape (output_dim,).

        log_std is clamped before exponentiation for numerical stability, with
        the value of torch.clamp but a straight-through gradient: torch.clamp
        passes none at or beyond a bound, which would leave the std stuck there.
        Inside the bounds this equals a plain clamp.
        """

        log_std = _StraightThroughClamp.apply(self.actor_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return torch.exp(log_std)


    def clamp_log_std_(self) -> None:
        """
        Clamp the actor_log_std parameter itself into [LOG_STD_MIN, LOG_STD_MAX].

        Call after every optimizer step, so the parameter cannot drift beyond a
        bound (where the std no longer changes) and would first have to travel
        back before the std responds again.
        """

        with torch.no_grad():
            self.actor_log_std.clamp_(self.LOG_STD_MIN, self.LOG_STD_MAX)

