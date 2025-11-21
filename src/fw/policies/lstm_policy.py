# lstm_policy.py
import os
import zipfile
import tempfile
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

from fw.policies.lstm_network import LstmNetwork

POLICY_FILE_NAME = "policy.pth"


def get_device(device_str: str = "auto") -> torch.device:
    """Return torch.device based on string. 'auto' prefers cuda if available."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class LstmPolicy:
    """
    Policy wrapper around LstmNetwork with convenient save/load and predict utilities.
    """

    def __init__(self, input_dim: int, output_dim: int, device: str = "auto"):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.device = get_device(device)
        # keep network defaults for other hyperparams (could be extended)
        self.network = LstmNetwork(obs_dim=self.input_dim, action_dim=self.output_dim).to(self.device)
        self._hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return {
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "device": self.device.type,
        }

    def reset_hidden(self, batch_size: int = 1) -> None:
        self._hidden_state = self.network.init_hidden(batch_size, self.device)

    def save(self, filename: str) -> None:
        """Save model into a single zip file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            params_path = os.path.join(tmp_dir, POLICY_FILE_NAME)
            torch.save(
                {
                    "model_state_dict": self.network.state_dict(),
                    "meta": self._get_constructor_parameters(),
                },
                params_path,
            )
            # ensure .zip extension for convenience
            zip_path = f"{filename}.zip" if not filename.endswith(".zip") else filename
            with zipfile.ZipFile(zip_path, "w") as z:
                z.write(params_path, arcname=POLICY_FILE_NAME)

    @classmethod
    def load(cls, filename: str, device_str: str = "auto") -> "LstmPolicy":
        device = get_device(device_str)
        zip_path = f"{filename}.zip" if not filename.endswith(".zip") else filename
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                with z.open(POLICY_FILE_NAME) as f:
                    saved = torch.load(f, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {zip_path}: {e}")

        meta = saved.get("meta", {})
        # fallback: try to infer sizes from state_dict
        if not meta:
            sd = saved["model_state_dict"]
            input_dim = sd["input_fc.weight"].shape[1]
            output_dim = sd["actor_out.weight"].shape[0]
            meta = {"input_dim": int(input_dim), "output_dim": int(output_dim), "device": device.type}

        model = cls(int(meta["input_dim"]), int(meta["output_dim"]), device.type)
        model.network.load_state_dict(saved["model_state_dict"])
        model.network.to(device)
        return model

    @staticmethod
    def calc_log_probs(
        dist: torch.distributions.Normal,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the log probabilities of actions with an adjustment for tanh squashing.

        Args:
            dist (torch.distributions.Normal): The normal distribution of actions.
            actions (torch.Tensor): The unbounded sampled actions from the distribution.

        Returns:
            torch.Tensor: The log probabilities of the actions.
        """
        return dist.log_prob(actions).sum(dim=-1)


    def _predict(self,
                 state: torch.Tensor,
                 hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 deterministic: bool = False) -> Tuple:
        """Predict the action and log probability for a given state.

        Args:
            state (torch.Tensor): The input state tensor.
            hidden: LSTM hidden state tuple (h, c) or None
            deterministic: If True, return mean action without sampling

        Returns:
            Tuple: A tuple containing:
                - action (torch.Tensor): The predicted action.
                - log_prob (torch.Tensor): The log probability of the action.
                - value (torch.Tensor): The predicted value for the state.
        """
        action_mean, action_std, value, new_hidden = self.network(state, hidden)  # Get network outputs (mean, std, value)

        # Create action distribution
        action_dist = torch.distributions.Normal(action_mean, action_std)

        # Sample action
        if deterministic:
            raw_action = action_mean
        else:
            raw_action = action_dist.sample()

        # Clip actions to valid range [-1, 1]
        action = torch.clamp(raw_action, -1.0, 1.0)

        # Compute log probability of the squashed action
        log_prob = self.calc_log_probs(action_dist, action)

        return action, log_prob, value.squeeze(-1), new_hidden  # Return action, log_prob, and value (squeezed)


    from typing import Union
    def predict(self,
                state: Union[np.ndarray, torch.Tensor],
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                deterministic: bool = False) -> Tuple:
        """Predict an action for a given state.

        Args:
            state (np.ndarray): The input state as a NumPy array or a tensor.
            hidden: LSTM hidden state tuple (h, c) or None
            deterministic: If True, return mean action without sampling

        Returns:
            Tuple: A tuple containing:
                - action (np.ndarray): The predicted action.
                - log_prob (np.ndarray): The log probability of the action.
                - value (np.ndarray): The predicted value for the state.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)  # Convert to tensor if necessary

        # Add batch dimension if input is from a single environment
        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():  # Disable gradient computation for inference
            action, log_prob, value, new_hidden = self._predict(state, hidden, deterministic)

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy(), new_hidden  # Move to CPU and return as NumPy arrays
