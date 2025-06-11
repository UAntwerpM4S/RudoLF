import os
import torch
import zipfile
import tempfile
import numpy as np

from typing import Tuple
from fw.policies.actor_critic_network import ActorCriticNetwork

POLICY_FILE_NAME = 'policy.pth'


def get_device(device: str = "auto") -> torch.device:
    """Retrieve the PyTorch device to be used for computations.

    Args:
        device (str, optional): The device to use. Options are 'auto', 'cuda', or 'cpu'.
            If 'auto', defaults to using GPU ('cuda') if available, otherwise falls back to CPU.

    Returns:
        torch.device: The appropriate PyTorch device.

    """
    # Default to CUDA if device is 'auto'
    if device == "auto":
        device = "cuda"

    # Convert the device string to a torch.device object
    device = torch.device(device)

    # If CUDA is requested but not available, fallback to CPU
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


class PPOPolicy:
    """Proximal Policy Optimization (PPO) Policy class for Actor-Critic reinforcement learning.

    This class encapsulates the network architecture, model saving/loading functionality,
    and prediction logic for PPO. It uses an Actor-Critic network for making predictions and
    updating the model.

    Attributes:
        input_dim (int): The dimensionality of the input state space.
        output_dim (int): The dimensionality of the output action space.
        device (torch.device): The device on which the model should run (CPU or CUDA).
        network (ActorCriticNetwork): The neural network model used for the Actor-Critic method.
    """

    def __init__(self, input_dim: int, output_dim: int, device: str = "cpu"):
        """Initialize the PPOPolicy with the given dimensions and device.

        Args:
            input_dim (int): The dimensionality of the state space.
            output_dim (int): The dimensionality of the action space.
            device (str, optional): The device to run computations on. Default is 'cpu'.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = get_device(device)  # Set the device
        self.network = ActorCriticNetwork(input_dim, output_dim).to(self.device)  # Initialize the network


    def _get_constructor_parameters(self) -> dict:
        """Retrieve the constructor parameters of the policy.

        Returns:
            dict: A dictionary containing the parameters used to construct the PPOPolicy.
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": self.device.type,
        }


    def save(self, filename: str) -> None:
        """Save the model state and constructor parameters to a file.

        Args:
            filename (str): The path to the file where the model should be saved.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Paths for temp files
            params_path = os.path.join(tmp_dir, POLICY_FILE_NAME)

            torch.save(
                {
                    "model_state_dict": self.network.state_dict(),
                    "data": self._get_constructor_parameters(),
                },
                params_path,
			)

            with zipfile.ZipFile(f"{filename}.zip", 'w') as zip_file:
                zip_file.write(params_path, arcname=POLICY_FILE_NAME)


    @classmethod
    def load(cls, filename: str, device: str = "cpu") -> "PPOPolicy":
        """Load a PPOPolicy model from a file.

        Args:
            filename (str): The path to the file from which the model should be loaded.
            device (str, optional): The device to load the model onto ('cpu', 'cuda', or 'auto'). Default is 'cpu'.

        Returns:
            PPOPolicy: The loaded PPOPolicy instance.

        Raises:
            RuntimeError: If the model file is corrupted or the device doesn't match.
        """
        device = get_device(device)

        try:
            with zipfile.ZipFile(f"{filename}.zip", 'r') as zip_file:
                with zip_file.open(POLICY_FILE_NAME) as policy_file:
                    saved_variables = torch.load(policy_file, map_location=device, weights_only=False)
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"Failed to load model: {e}")

        data = saved_variables.get("data", {})
        if not data:
            # Infer missing data from model state_dict
            model_state_dict = saved_variables["model_state_dict"]
            data = {
                "input_dim": model_state_dict["shared_network.0.weight"].shape[1],
                "output_dim": model_state_dict["actor_mean.weight"].shape[0],
                "device": device.type,
            }

        if device.type != torch.device(data["device"]).type:
            raise RuntimeError(f'Expected a "{device}" device, but got a "{data["device"]}" device instead!')

        model = cls(**data)  # Initialize model with constructor data
        model.network.load_state_dict(saved_variables["model_state_dict"])  # Load the model weights
        model.network.to(device)  # Move model to the specified device
        return model


    @staticmethod
    def calc_log_probs(
        dist: torch.distributions.Normal,
        raw_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the log probabilities of actions with an adjustment for tanh squashing.

        Args:
            dist (torch.distributions.Normal): The normal distribution of actions.
            raw_actions (torch.Tensor): The unbounded sampled actions from the distribution.
            actions (torch.Tensor): The squashed actions (after applying tanh).

        Returns:
            torch.Tensor: The log probabilities of the actions.
        """
        log_probs = dist.log_prob(raw_actions)  # Calculate log probability of raw actions
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6)  # Adjust for tanh squashing
        return log_probs.sum(dim=-1)  # Sum across the action dimensions


    def _predict(self, state: torch.Tensor) -> Tuple:
        """Predict the action and log probability for a given state.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            Tuple: A tuple containing:
                - action (torch.Tensor): The predicted action.
                - log_prob (torch.Tensor): The log probability of the action.
                - value (torch.Tensor): The predicted value for the state.
        """
        action_mean, action_std, value = self.network(state)  # Get network outputs (mean, std, value)

        # Create a normal distribution with the mean and std
        dist = torch.distributions.Normal(action_mean, action_std)

        # Sample raw action and squash it using tanh
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        # Compute log probability of the squashed action
        log_prob = self.calc_log_probs(dist, raw_action, action)

        return action, log_prob, value.squeeze(-1)  # Return action, log_prob, and value (squeezed)


    def predict(self, state: np.ndarray) -> Tuple:
        """Predict an action for a given state.

        Args:
            state (np.ndarray): The input state as a NumPy array or a tensor.

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
            action, log_prob, value = self._predict(state)

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()  # Move to CPU and return as NumPy arrays
