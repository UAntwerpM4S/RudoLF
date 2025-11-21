# lstm_policy.py
import os
import zipfile
import tempfile
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

from fw.policies.lstm_network import LstmNetwork
from fw.policies.ppo_policy import PPOPolicy

POLICY_FILE_NAME = "policy.pth"


def get_device(device_str: str = "auto") -> torch.device:
    """Return torch.device based on string. 'auto' prefers cuda if available."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class LstmPolicy(PPOPolicy):
    """
    Policy wrapper around LstmNetwork with convenient save/load and predict utilities.
    """

    def __init__(self, input_dim: int, output_dim: int, device: str = "cpu",
                 lstm_hidden_size: int = 128):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device(device)
        self.lstm_hidden_size = lstm_hidden_size

        # Use LSTM network
        self.network = LstmNetwork(input_dim, output_dim, lstm_hidden_size).to(self.device)
        self.optimizer = None  # Will be set by the model

    def _get_constructor_parameters(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": self.device.type,
            "lstm_hidden_size": self.lstm_hidden_size
        }

    def predict_single_episode(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict for a complete episode (single environment)
        Returns: actions, log_probs, values for the entire episode
        """
        states_tensor = torch.FloatTensor(states).to(self.device)

        # Run through LSTM sequentially to maintain proper hidden states
        actions, log_probs, values = [], [], []
        hidden = None

        for t in range(len(states)):
            # Single step prediction
            state_step = states_tensor[t:t + 1].unsqueeze(1)  # (1, 1, features)

            with torch.no_grad():
                action_mean, action_std, value, hidden = self.network(state_step, hidden)

                # Sample action
                dist = torch.distributions.Normal(action_mean, action_std)
                raw_action = dist.rsample()
                action = torch.tanh(raw_action)
                log_prob = self.calc_log_probs(dist, raw_action, action)

            actions.append(action.cpu().numpy()[0])
            log_probs.append(log_prob.cpu().numpy()[0])
            values.append(value.cpu().numpy()[0, 0])

        return np.array(actions), np.array(log_probs), np.array(values)

    # Keep the original predict method for compatibility with existing code
    def predict(self, state: np.ndarray):
        """Single step prediction for compatibility"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            # Use None hidden state for single step (starts fresh each time)
            action_mean, action_std, value, _ = self.network(state.unsqueeze(1), None)

            dist = torch.distributions.Normal(action_mean, action_std)
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            log_prob = self.calc_log_probs(dist, raw_action, action)

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.squeeze(-1).cpu().numpy()
