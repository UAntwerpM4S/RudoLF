import torch
import torch.nn as nn

from typing import Tuple, Optional


class LstmNetwork(nn.Module):
    """
    LSTM-based actor-critic network that outputs values for all timesteps.
    """

    def __init__(self, input_dim: int, output_dim: int, lstm_hidden_size: int = 128):
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True)

        # Post-LSTM layers
        self.post_lstm = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
        )

        # Actor and critic heads
        self.actor_mean = nn.Linear(64, output_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(output_dim))
        self.critic = nn.Linear(64, 1)

    # In your lstm_network.py - update the forward method:

    def forward(self, state: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # Handle single step vs sequence
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (batch, 1, features)

        batch_size = state.size(0)

        # 🛠️ FIX: Ensure hidden state matches current batch size
        if hidden_state is None:
            hidden_state = self.get_initial_hidden_state(batch_size)
        else:
            # If hidden state exists but batch size doesn't match, recreate it
            hidden_batch_size = hidden_state[0].size(1)
            if hidden_batch_size != batch_size:
                hidden_state = self.get_initial_hidden_state(batch_size)

        # LSTM forward
        lstm_out, hidden_out = self.lstm(state, hidden_state)

        # Use last output only (for compatibility with your current training)
        if lstm_out.shape[1] > 1:
            features = lstm_out[:, -1, :]  # Last timestep
        else:
            features = lstm_out.squeeze(1)

        # Post-processing
        shared_features = self.post_lstm(features)

        # Outputs
        action_mean = torch.tanh(self.actor_mean(shared_features))
        action_std = torch.exp(self.actor_log_std.clamp(-20, 2))
        state_value = self.critic(shared_features)

        return action_mean, action_std, state_value, hidden_out

    def get_initial_hidden_state(self, batch_size: int = 1):
        device = next(self.parameters()).device
        return (torch.zeros(1, batch_size, self.lstm_hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm_hidden_size, device=device))
