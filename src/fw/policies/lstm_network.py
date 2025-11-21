import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional


class LstmNetwork(nn.Module):
    """
    LSTM-based actor-critic network that outputs values for all timesteps.
    """

    def __init__(
            self,
            obs_dim: int = 367,
            action_dim: int = 2,
            lstm_hidden_size: int = 256,
            lstm_layers: int = 2,
            fc_hidden_size: int = 256,
            dropout: float = 0.1
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # Input preprocessing layers
        self.input_bn = nn.BatchNorm1d(obs_dim)
        self.input_fc = nn.Linear(obs_dim, lstm_hidden_size)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Shared feature extraction after LSTM - now processes all timesteps
        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Actor head (policy network) - for all timesteps
        self.actor_mean = nn.Sequential(
            nn.Linear(fc_hidden_size, fc_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden_size // 2, action_dim),
            nn.Tanh()  # Actions are in [-1, 1]
        )

        # Log standard deviation for action distribution (learned parameter)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function) - for all timesteps
        self.critic = nn.Sequential(
            nn.Linear(fc_hidden_size, fc_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden_size // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with appropriate scaling."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weight initialization
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Initialize action log std to reasonable values
        nn.init.constant_(self.actor_log_std, -0.5)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden states."""
        h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return h0, c0

    def forward(
            self,
            obs: torch.Tensor,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network that outputs values for all timesteps.

        Args:
            obs: Observation tensor of shape (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden: LSTM hidden state tuple (h, c) or None

        Returns:
            action_mean: Action means for all timesteps (batch_size, seq_len, action_dim)
            action_std: Action stds for all timesteps (batch_size, seq_len, action_dim)
            values: Value estimates for all timesteps (batch_size, seq_len, 1)
            new_hidden: Updated LSTM hidden states
        """
        # Handle input dimensions
        if obs.dim() == 2:
            # Single timestep: (batch_size, obs_dim) -> (batch_size, 1, obs_dim)
            obs = obs.unsqueeze(1)
            single_timestep = True
        else:
            # Multiple timesteps: (batch_size, seq_len, obs_dim)
            single_timestep = False

        batch_size, seq_len, _ = obs.shape
        device = obs.device

        # Input preprocessing
        original_shape = obs.shape
        obs_flat = obs.reshape(-1, self.obs_dim)

        # Batch normalization (only if batch size > 1)
        if obs_flat.shape[0] > 1:
            obs_normalized = self.input_bn(obs_flat)
        else:
            obs_normalized = obs_flat

        obs_preprocessed = F.relu(self.input_fc(obs_normalized))
        obs_preprocessed = obs_preprocessed.view(original_shape[0], original_shape[1], -1)

        # LSTM forward pass
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        lstm_out, new_hidden = self.lstm(obs_preprocessed, hidden)

        # Process all timesteps through shared layers
        lstm_out_flat = lstm_out.reshape(-1, self.lstm_hidden_size)
        features = self.shared_fc(lstm_out_flat)
        features = features.reshape(batch_size, seq_len, -1)

        # Actor (policy) - for all timesteps
        action_mean_flat = self.actor_mean(features.reshape(-1, features.shape[-1]))
        action_mean = action_mean_flat.reshape(batch_size, seq_len, self.action_dim)

        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Critic (value function) - for all timesteps
        values_flat = self.critic(features.reshape(-1, features.shape[-1]))
        values = values_flat.reshape(batch_size, seq_len, 1)

        # If input was single timestep, squeeze back to match input format
        if single_timestep:
            action_mean = action_mean.squeeze(1)
            action_std = action_std.squeeze(1)
            values = values.squeeze(1)

        return action_mean, action_std, values, new_hidden
