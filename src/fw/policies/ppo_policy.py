import os
import logging
import torch
import zipfile
import tempfile
import numpy as np

from typing import Tuple, Dict, Any, Union
from fw.policies.actor_critic_network import ActorCriticNetwork
from fw.common.tools.ppo_utils import normalize_policy_zip_path

logger = logging.getLogger(__name__)

POLICY_FILE_NAME = 'policy.pth'

def get_device(device: str = "auto") -> torch.device:
    """
    Retrieve the PyTorch device to be used for computations.

    Args:
        device (str, optional): The device to use. Options are 'auto', 'cuda', or 'cpu'.
            If 'auto', defaults to using GPU ('cuda') if available, otherwise falls back to CPU.

    Returns:
        torch.device: The appropriate PyTorch device.

    """


    # Default to CUDA if device is 'auto'
    if device == "auto":
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    # If CUDA is requested but not available, fallback to CPU
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


class PPOPolicy:
    """
    Proximal Policy Optimization (PPO) Policy class for Actor-Critic reinforcement learning.

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
        """
        Initialize the PPOPolicy with the given dimensions and device.

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
        """
        Retrieve the constructor parameters of the policy.

        Returns:
            dict: A dictionary containing the parameters used to construct the PPOPolicy.
        """

        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": self.device.type,
        }


    def serialize_policy(self) -> Dict[str, Any]:
        """
        Serialize the policy into a dictionary format.

        Captures both the model parameters and the minimal constructor
        arguments required to reconstruct the policy.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "model_state_dict": The state dictionary of the policy network.
                - "data": Constructor parameters needed to rebuild the policy.
        """

        return {
            "model_state_dict": self.network.state_dict(),
            "data": self._get_constructor_parameters(),
        }


    @staticmethod
    def infer_constructor_data(saved_variables: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Infer constructor parameters for policy reconstruction.

        Attempts to retrieve constructor data from the saved variables. If
        unavailable, infers required parameters (e.g., input/output dimensions)
        directly from the model state dictionary.

        Args:
            saved_variables (Dict[str, Any]): Serialized policy data containing
                model weights and optional constructor metadata.
            device (torch.device): Target device for the reconstructed model.

        Returns:
            Dict[str, Any]: Dictionary of constructor parameters including:
                - "input_dim": Input feature dimension.
                - "output_dim": Output action dimension.
                - "device": Device type as a string.

        Raises:
            KeyError: If required keys are missing in `saved_variables` when
                inference is attempted.
        """

        data = saved_variables.get("data", {})

        if not data:
            # Infer missing data from model state_dict
            model_state_dict = saved_variables["model_state_dict"]
            data = {
                "input_dim": model_state_dict["shared_network.0.weight"].shape[1],
                "output_dim": model_state_dict["actor_mean.weight"].shape[0],
                "device": device.type,
            }

        # Filter to only the keys accepted by the current constructor, so
        # checkpoints saved with extra metadata fields don't break loading.
        allowed_keys = {"input_dim", "output_dim", "device"}
        data = {k: v for k, v in data.items() if k in allowed_keys}

        return data


    @classmethod
    def build_policy_from_state(cls, saved_variables: Dict[str, Any], device: torch.device):
        """
        Reconstruct a policy instance from serialized state data.

        Initializes a new policy using inferred or stored constructor parameters,
        loads the model weights, and moves the network to the specified device.

        Args:
            saved_variables (Dict[str, Any]): Serialized policy containing
                model weights and constructor metadata.
            device (torch.device): Target device for the reconstructed model.

        Returns:
            PPOPolicy: Reconstructed policy instance with loaded weights.

        """

        data = dict(cls.infer_constructor_data(saved_variables, device))

        # The device stored in the checkpoint is where it was TRAINED. The
        # tensors are already remapped by torch.load(map_location=device), so
        # the policy is simply rebuilt on the requested device (a GPU-trained
        # checkpoint must load for CPU inference / transfer learning).
        trained_on = data.get("device")
        if trained_on is not None and torch.device(trained_on).type != device.type:
            logger.info("Loading a policy trained on '%s' onto '%s'.", trained_on, device)
        data["device"] = str(device)

        model = cls(**data)  # Initialize model with constructor data
        model.network.load_state_dict(saved_variables["model_state_dict"])  # Load the model weights
        model.network.to(device)  # Move model to the specified device
        return model


    def save(self, filename: str) -> None:
        """
        Save the model state and constructor parameters to a file.

        Args:
            filename (str): The path to the file where the model should be saved.
        """

        zip_path = normalize_policy_zip_path(filename)
        parent_dir = os.path.dirname(zip_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Paths for temp files
            params_path = os.path.join(tmp_dir, POLICY_FILE_NAME)

            torch.save(
                self.serialize_policy(),
                params_path,
			)

            with zipfile.ZipFile(zip_path, 'w') as zip_file:
                zip_file.write(params_path, arcname=POLICY_FILE_NAME)


    @classmethod
    def load(cls, filename: str, device: str = "cpu") -> "PPOPolicy":
        """
        Load a PPOPolicy model from a file.

        Args:
            filename (str): The path to the file from which the model should be loaded.
            device (str, optional): The device to load the model onto ('cpu', 'cuda', or 'auto'). Default is 'cpu'.

        Returns:
            PPOPolicy: The loaded PPOPolicy instance.

        Raises:
            RuntimeError: If the model file is corrupted or the device doesn't match.
        """

        device = get_device(device)
        zip_path = normalize_policy_zip_path(filename)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                with zip_file.open(POLICY_FILE_NAME) as policy_file:
                    saved_variables = torch.load(policy_file, map_location=device, weights_only=False)
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"Failed to load model: {e}")

        return cls.build_policy_from_state(saved_variables, device)


    # Squashed actions are clamped to +-SQUASH_CLAMP before the log-density is
    # evaluated. The SAME clamp is applied when the log-prob is computed during
    # the rollout and when it is recomputed from the stored (squashed) action in
    # the PPO update, so both evaluate exactly the same point and the importance
    # ratio is exactly 1 before the first optimizer step. Without that, samples
    # with |raw| > atanh(0.999) ~= 3.8 were evaluated at different points in the
    # rollout (exact raw sample) and the update (atanh of the clamped action),
    # which produced ratios of 1e4..1e18 for saturated actions.
    SQUASH_CLAMP = 0.999

    @classmethod
    def squashed_log_prob(
        cls,
        dist: torch.distributions.Normal,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log-probability of tanh-squashed actions, computed from the squashed
        action only (no dependency on the raw Gaussian sample).

        Args:
            dist (torch.distributions.Normal): Pre-squash Gaussian.
            actions (torch.Tensor): Squashed actions in [-1, 1].

        Returns:
            torch.Tensor: Log-probabilities summed over the action dimensions.
        """

        a = actions.clamp(-cls.SQUASH_CLAMP, cls.SQUASH_CLAMP)
        raw = torch.atanh(a)
        log_probs = dist.log_prob(raw) - torch.log(1 - a.pow(2) + 1e-6)
        return log_probs.sum(dim=-1)

    @staticmethod
    def calc_log_probs(
        dist: torch.distributions.Normal,
        raw_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the log probabilities of actions with an adjustment for tanh squashing.

        Kept for backward compatibility. The rollout and the PPO update use
        `squashed_log_prob`, which depends on the squashed action only, so the
        two are always evaluated at the same point. `raw_actions` is ignored.

        Args:
            dist (torch.distributions.Normal): The normal distribution of actions.
            raw_actions (torch.Tensor): Ignored (see above).
            actions (torch.Tensor): The squashed actions (after applying tanh).

        Returns:
            torch.Tensor: The log probabilities of the actions.
        """

        return PPOPolicy.squashed_log_prob(dist, actions)


    def _predict(self, state: torch.Tensor, deterministic: bool = False) -> Tuple:
        """
        Predict the action and log probability for a given state.

        Args:
            state (torch.Tensor): The input state tensor.
            deterministic (boolean): Whether the action should be deterministic or not.

        Returns:
            Tuple: A tuple containing:
                - action (torch.Tensor): The predicted action.
                - log_prob (torch.Tensor): The log probability of the action.
                - value (torch.Tensor): The predicted value for the state.
        """

        action_mean, action_std, value = self.network(state)  # Get network outputs (mean, std, value)

        # Create a normal distribution with the mean and std
        dist = torch.distributions.Normal(action_mean, action_std)

        if deterministic:
            # Use the mean as the raw (pre-squash) action
            raw_action = action_mean
        else:
            # Sample in R
            raw_action = dist.rsample()

        # Sample raw action and squash it using tanh
        action = torch.tanh(raw_action)

        # Compute log probability of the squashed action (same evaluation
        # point as in the PPO update, see `squashed_log_prob`).
        log_prob = self.squashed_log_prob(dist, action)

        return action, log_prob, value.squeeze(-1)  # Return action, log_prob, and value (squeezed)


    def predict(self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict an action for a given state.

        Args:
            state (np.ndarray): The input state as a NumPy array or a tensor.
            deterministic (boolean): Whether the action should be deterministic or not.

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
            action, log_prob, value = self._predict(state, deterministic)

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()  # Move to CPU and return as NumPy arrays
