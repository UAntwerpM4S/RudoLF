import os
import torch
import logging
import zipfile
import tempfile
import numpy as np

from typing import Tuple, Dict, Any, Optional, Union
from fw.policies.actor_critic_network import ActorCriticNetwork
from fw.common.tools.ppo_utils import normalize_policy_zip_path

logger = logging.getLogger(__name__)

POLICY_FILE_NAME = 'policy.pth'

# Per-update PPO diagnostics, averaged over the minibatches of one update.
# entropy is that of the pre-tanh Gaussian (the term the entropy bonus uses);
# approx_kl is the k3 estimator mean((r - 1) - log r), with log r capped at
# PPOPolicy.MAX_LOG_RATIO as in the loss; clip_frac is the fraction of samples
# with |r - 1| > clip_range; action_std is the mean std after the update.
PPO_UPDATE_STAT_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "action_std")


def empty_update_stats(network: Optional[torch.nn.Module] = None) -> Dict[str, float]:
    """
    Update stats for an update that did not happen: all NaN, except action_std,
    which is the network's current mean std if a network is given.
    """

    stats = {key: float("nan") for key in PPO_UPDATE_STAT_KEYS}
    if network is not None:
        stats["action_std"] = mean_action_std(network)
    return stats


def mean_action_std(network: torch.nn.Module) -> float:
    """
    Mean over the action dimensions of the network's current action std.
    """

    with torch.no_grad():
        return float(network.action_std().mean().item())


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


def require_policy_dims(policy: "PPOPolicy", input_dim: int, output_dim: int, source: str) -> None:
    """
    Check that a loaded policy fits the observation and action sizes of its env.

    Args:
        policy: The loaded policy.
        input_dim: Observation size the env provides.
        output_dim: Action size the env expects.
        source: Policy file or slot, named in the error.

    Raises:
        ValueError: If a size differs (e.g. a policy saved before an observation change).
    """

    if policy.input_dim != input_dim or policy.output_dim != output_dim:
        raise ValueError(
            f"Policy '{source}' has observation/action size {policy.input_dim}/{policy.output_dim}, "
            f"but the environment provides {input_dim}/{output_dim}.")


def _yaw_scale_envs(env) -> list:
    """
    The (unwrapped) envs whose observed yaw-rate scale follows a policy: the
    sub-environments of a multi-ship env, else the env itself; envs without
    set_yaw_rate_obs_scale are left out.
    """

    if env is None:
        return []
    subs = getattr(env, "sub_environments", None)
    candidates = list(subs.values() if isinstance(subs, dict) else subs) if subs else [env]
    envs = [getattr(e, "unwrapped", e) for e in candidates]
    return [e for e in envs if hasattr(e, "set_yaw_rate_obs_scale")]


def env_yaw_rate_obs_scale(env) -> Optional[float]:
    """
    The yaw-rate observation scale an env currently uses (the first sub-environment's
    for a multi-ship env), or None if it has none.
    """

    envs = _yaw_scale_envs(env)
    return float(envs[0].yaw_rate_obs_scale) if envs else None


def start_fresh_policy(policy: "PPOPolicy", env) -> None:
    """
    For a newly built (untrained) policy: restore the env's own yaw-rate scale (a
    policy loaded earlier may have set another) and record it in the policy.
    """

    for e in _yaw_scale_envs(env):
        e.set_yaw_rate_obs_scale(None)
    policy.yaw_rate_obs_scale = env_yaw_rate_obs_scale(env)


def policy_yaw_rate_obs_scale(policy: "PPOPolicy", env) -> Optional[float]:
    """
    The yaw-rate observation scale [rad/s] a policy was trained with: its stored
    scale, or for a file without one the env's unstored_policy_yaw_rate_obs_scale
    (multi-ship env: its first sub-environment's). None if env has no yaw-rate scale.
    """

    if policy.yaw_rate_obs_scale is not None:
        return float(policy.yaw_rate_obs_scale)
    envs = _yaw_scale_envs(env)
    return float(envs[0].unstored_policy_yaw_rate_obs_scale) if envs else None


def yaw_rate_obs_scale_changes(policy: "PPOPolicy", env) -> bool:
    """
    Whether loading the policy into env changes the yaw-rate scale of any of its envs.
    """

    scale = policy_yaw_rate_obs_scale(policy, env)
    return any(e.yaw_rate_obs_scale != scale for e in _yaw_scale_envs(env))


def apply_policy_yaw_rate_obs_scale(policy: "PPOPolicy", env, source: str) -> None:
    """
    Make the env observe the yaw rate with the scale the loaded policy was trained
    with, so a policy trained with another dynamics model (e.g. Fossen) sees the
    same observation for the same yaw rate when it runs in this env (e.g. FH).

    A policy file without a stored scale gets the env's
    unstored_policy_yaw_rate_obs_scale (logged), which is then saved with it.
    Envs without a yaw-rate scale are left alone.

    Args:
        policy: The loaded policy.
        env: The env it will run in (multi-ship: all sub-environments).
        source: Policy file or slot, named in the log.
    """

    envs = _yaw_scale_envs(env)
    if not envs:
        return
    if policy.yaw_rate_obs_scale is None:
        policy.yaw_rate_obs_scale = policy_yaw_rate_obs_scale(policy, env)
        logger.info("Policy '%s' has no stored yaw-rate observation scale: using %g, the scale %s "
                    "policies without one were trained with.", source, policy.yaw_rate_obs_scale,
                    type(envs[0]).__name__)
    for e in envs:
        if e.yaw_rate_obs_scale != policy.yaw_rate_obs_scale:
            logger.info("%s observes the yaw rate with scale %g, the scale policy '%s' was trained with "
                        "(own scale %g).", type(e).__name__, policy.yaw_rate_obs_scale, source,
                        e.own_yaw_rate_obs_scale)
            e.set_yaw_rate_obs_scale(policy.yaw_rate_obs_scale)


class PPOPolicy:
    """
    PPO actor-critic policy: network, save/load and (tanh-squashed) prediction.

    Attributes:
        input_dim (int): Observation dimension.
        output_dim (int): Action dimension.
        device (torch.device): Device the network runs on.
        network (ActorCriticNetwork): The actor-critic network.
        yaw_rate_obs_scale (Optional[float]): Yaw-rate observation scale [rad/s]
            the policy is trained with; saved with it and applied to the env it is
            loaded into (apply_policy_yaw_rate_obs_scale). None if not stored.
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
        self.yaw_rate_obs_scale: Optional[float] = None


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
            Dict[str, Any]: "model_state_dict" (network weights), "data"
                (constructor arguments) and, if known, "obs_scales" with the
                yaw-rate observation scale.
        """

        state = {
            "model_state_dict": self.network.state_dict(),
            "data": self._get_constructor_parameters(),
        }
        if self.yaw_rate_obs_scale is not None:
            state["obs_scales"] = {"yaw_rate": float(self.yaw_rate_obs_scale)}
        return state


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

        Raises:
            ValueError: If a stored yaw-rate observation scale is not positive and finite.
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
        yaw_scale = (saved_variables.get("obs_scales") or {}).get("yaw_rate")
        if yaw_scale is not None:
            yaw_scale = float(yaw_scale)
            if not (np.isfinite(yaw_scale) and yaw_scale > 0.0):
                raise ValueError(f"Stored yaw-rate observation scale must be positive and finite, got {yaw_scale}.")
        model.yaw_rate_obs_scale = yaw_scale
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
            ValueError: If its stored yaw-rate observation scale is not positive and finite.
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
