import os
import torch
import logging
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
    Return the torch device; 'auto' and 'cuda' fall back to CPU without CUDA.

    Args:
        device (str, optional): 'auto', 'cuda', 'cpu' or another torch device string.

    Returns:
        torch.device: The device to use.
    """


    if device == "auto":
        device = torch.device("cuda")
    else:
        device = torch.device(device)

    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


class PPOPolicy:
    """
    PPO actor-critic policy: network, save/load and (tanh-squashed) prediction.

    Attributes:
        input_dim (int): Observation dimension.
        output_dim (int): Action dimension.
        device (torch.device): Device the network runs on.
        network (ActorCriticNetwork): The actor-critic network.
    """

    def __init__(self, input_dim: int, output_dim: int, device: str = "cpu"):
        """
        Args:
            input_dim (int): Observation dimension.
            output_dim (int): Action dimension.
            device (str, optional): Device to run on (see get_device).
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = get_device(device)
        self.network = ActorCriticNetwork(input_dim, output_dim).to(self.device)


    def _get_constructor_parameters(self) -> dict:
        """
        Return the constructor arguments (input_dim, output_dim, device).
        """

        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": self.device.type,
        }


    def serialize_policy(self) -> Dict[str, Any]:
        """
        Serialize the policy.

        Returns:
            Dict[str, Any]: "model_state_dict" (network weights) and "data"
                (constructor arguments).
        """

        return {
            "model_state_dict": self.network.state_dict(),
            "data": self._get_constructor_parameters(),
        }


    @staticmethod
    def infer_constructor_data(saved_variables: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Return the constructor arguments stored in a checkpoint.

        Without stored "data", input/output dimensions are inferred from the
        weights. Unknown keys are dropped so extra metadata does not break loading.

        Args:
            saved_variables (Dict[str, Any]): Serialized policy (see serialize_policy).
            device (torch.device): Target device, used when inferring.

        Returns:
            Dict[str, Any]: "input_dim", "output_dim" and "device".

        Raises:
            KeyError: If the weights needed for inference are missing.
        """

        data = saved_variables.get("data", {})

        if not data:
            model_state_dict = saved_variables["model_state_dict"]
            data = {
                "input_dim": model_state_dict["shared_network.0.weight"].shape[1],
                "output_dim": model_state_dict["actor_mean.weight"].shape[0],
                "device": device.type,
            }

        allowed_keys = {"input_dim", "output_dim", "device"}
        data = {k: v for k, v in data.items() if k in allowed_keys}

        return data


    @classmethod
    def build_policy_from_state(cls, saved_variables: Dict[str, Any], device: torch.device):
        """
        Rebuild a policy on `device` from serialized state.

        Args:
            saved_variables (Dict[str, Any]): Serialized policy (see serialize_policy).
            device (torch.device): Target device.

        Returns:
            PPOPolicy: Policy with the loaded weights.
        """

        data = dict(cls.infer_constructor_data(saved_variables, device))

        # The stored device is where the policy was trained; tensors are already
        # remapped by torch.load, so rebuild on the requested device.
        trained_on = data.get("device")
        if trained_on is not None and torch.device(trained_on).type != device.type:
            logger.info("Loading a policy trained on '%s' onto '%s'.", trained_on, device)
        data["device"] = str(device)

        model = cls(**data)
        model.network.load_state_dict(saved_variables["model_state_dict"])
        model.network.to(device)
        return model


    def save(self, filename: str) -> None:
        """
        Save weights and constructor arguments as a .zip containing policy.pth.

        Args:
            filename (str): Target path (".zip" is added if missing).
        """

        zip_path = normalize_policy_zip_path(filename)
        parent_dir = os.path.dirname(zip_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
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
        Load a policy saved with `save`.

        Args:
            filename (str): Policy path (".zip" is added if missing).
            device (str, optional): Device to load onto ('cpu', 'cuda' or 'auto').

        Returns:
            PPOPolicy: The loaded policy.

        Raises:
            RuntimeError: If the file or its policy.pth entry is missing.
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


    # Upper bound on log(new_prob / old_prob) in the PPO update. Without it,
    # exp() overflows to inf above ~88 (float32) and min(surr1, surr2) can
    # backpropagate 0 * inf = NaN into every weight; well below 88 the gradient
    # norm can already overflow, which skips the whole minibatch. Samples
    # above the bound get zero policy gradient (their loss term stays finite
    # but large); samples below it are unaffected.
    MAX_LOG_RATIO = 20.0

    @staticmethod
    def calc_log_probs(
        dist: torch.distributions.Normal,
        raw_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log-probability of tanh-squashed actions with the tanh correction.

        Args:
            dist (torch.distributions.Normal): Pre-squash Gaussian.
            raw_actions (torch.Tensor): Pre-squash (raw) actions.
            actions (torch.Tensor): Squashed actions, tanh(raw_actions).

        Returns:
            torch.Tensor: Log-probabilities summed over the action dimensions.
        """

        log_probs = dist.log_prob(raw_actions)
        log_probs = log_probs - torch.log(1 - actions.pow(2) + 1e-6)
        return log_probs.sum(dim=-1)

    @classmethod
    def raw_log_prob(
        cls,
        dist: torch.distributions.Normal,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log-probability of the tanh-squashed action, evaluated at the raw action.

        Used both in the rollout and in the PPO update, so both evaluate the
        exact same point.

        Args:
            dist (torch.distributions.Normal): Pre-squash Gaussian.
            raw_actions (torch.Tensor): Pre-squash (raw) actions.

        Returns:
            torch.Tensor: Log-probabilities summed over the action dimensions.
        """

        return cls.calc_log_probs(dist, raw_actions, torch.tanh(raw_actions))


    def _predict_with_raw(self, state: torch.Tensor, deterministic: bool = False) -> Tuple:
        """
        Predict squashed action, raw action, log-probability and value (tensors).

        Args:
            state (torch.Tensor): Batched observations.
            deterministic (bool): Use the mean instead of sampling.

        Returns:
            Tuple: (action in [-1, 1], raw pre-tanh action, log_prob, value).
        """

        action_mean, action_std, value = self.network(state)

        dist = torch.distributions.Normal(action_mean, action_std)

        if deterministic:
            raw_action = action_mean
        else:
            raw_action = dist.rsample()

        action = torch.tanh(raw_action)
        log_prob = self.raw_log_prob(dist, raw_action)

        return action, raw_action, log_prob, value.squeeze(-1)


    def _predict(self, state: torch.Tensor, deterministic: bool = False) -> Tuple:
        """
        Predict squashed action, log-probability and value (tensors).

        Args:
            state (torch.Tensor): Batched observations.
            deterministic (bool): Use the mean instead of sampling.

        Returns:
            Tuple: (action in [-1, 1], log_prob, value).
        """

        action, _, log_prob, value = self._predict_with_raw(state, deterministic)
        return action, log_prob, value


    def _as_batched_tensor(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert a state to a float tensor on the policy device; a 1-D state gets
        a batch dimension.
        """

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        return state


    def predict(self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict without gradients; a 1-D state gets a batch dimension.

        Args:
            state (np.ndarray | torch.Tensor): Observation(s).
            deterministic (bool): Use the mean instead of sampling.

        Returns:
            Tuple: (action, log_prob, value) as NumPy arrays.
        """

        state = self._as_batched_tensor(state)

        with torch.no_grad():
            action, log_prob, value = self._predict(state, deterministic)

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()


    def predict_with_raw(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        As `predict`, but also return the raw pre-tanh action, which the PPO
        update needs to evaluate the log-probability at the same point.

        Args:
            state (np.ndarray | torch.Tensor): Observation(s).
            deterministic (bool): Use the mean instead of sampling.

        Returns:
            Tuple: (action, raw_action, log_prob, value) as NumPy arrays.
        """

        state = self._as_batched_tensor(state)

        with torch.no_grad():
            action, raw_action, log_prob, value = self._predict_with_raw(state, deterministic)

        return (
            action.cpu().numpy(),
            raw_action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )
