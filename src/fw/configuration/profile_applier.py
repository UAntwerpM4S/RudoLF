"""
Apply a VesselProfile (fw.configuration.config_profile) to existing framework objects:

* CONFIG_PARAMETERS dict: ship_model, dynamics_model
* BaseEnv: smoothing flag, rudder-angle scaling flag, initial speed
* BaseSimulator: actuator-limit overrides
* Raw [rudder, thrust] actions: fixed-thrust substitution only. Rudder-angle
  scaling is a flag on BaseEnv passed down to the dynamics model's step()
  (only FhTabularModel uses it; see that class for the formula).

Usage::

    from fw.configuration.config_profile import ProfileRegistry
    from fw.configuration.profile_applier import apply_profile
    from fw.configuration.config import get_config_parameters

    profile = ProfileRegistry.get("inference_pysim")
    cfg = get_config_parameters()
    apply_profile(profile, config=cfg)

    env = build_default_environment("PySimEnv")
    apply_profile(profile, env=env)

    action = apply_profile(profile, action=agent.predict(obs))
    obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import logging
import warnings
import numpy as np

from typing import Any, Dict, Optional
from fw.configuration.config_profile import ThrustMode, VesselProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_profile(
    profile: VesselProfile,
    *,
    config: Optional[Dict[str, Any]] = None,
    env=None,
    simulator=None,
    action: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Apply *profile* to any combination of targets; pass only what you have.

    config, env and simulator are mutated in place; action is returned as a
    transformed copy.

    Args:
        profile: The VesselProfile to apply.
        config: CONFIG_PARAMETERS dict (from get_config_parameters()).
        env: A BaseEnv; smoothing, rudder-angle scaling and initial speed are
            also set on its sub-environments.
        simulator: A BaseSimulator; actuator limits are patched if overridden.
        action: [rudder, thrust] array in [-1, 1].

    Returns:
        Transformed action array when *action* is not *None*, otherwise *None*.

    Raises:
        ValueError: If *action* does not have at least 2 elements.
    """

    if config is not None:
        ConfigApplier.apply(profile, config)

    if env is not None:
        EnvApplier.apply(profile, env)

    if simulator is not None:
        SimulatorApplier.apply(profile, simulator)

    if action is not None:
        return ActionAdapter.transform(profile, action)

    return None


# ---------------------------------------------------------------------------
# Config applier
# ---------------------------------------------------------------------------

class ConfigApplier:
    """
    Patches a CONFIG_PARAMETERS dict in place.
    """

    @staticmethod
    def apply(profile: VesselProfile, config: Dict[str, Any]) -> None:
        """
        Set ship_model and dynamics_model in *config* when the profile defines them.
        """

        if profile.model.ship_model is not None:
            config["ship_model"] = profile.model.ship_model
            logger.debug("ConfigApplier: set ship_model=%s", config["ship_model"])

        if profile.model.dynamics_model is not None:
            config["dynamics_model"] = profile.model.dynamics_model
            logger.debug("ConfigApplier: set dynamics_model=%s", config["dynamics_model"])


# ---------------------------------------------------------------------------
# Env applier
# ---------------------------------------------------------------------------

class EnvApplier:
    """
    Sets env-level flags on an environment and all its sub-environments.
    """

    @staticmethod
    def apply(profile: VesselProfile, env) -> None:
        """
        Apply the profile's env-level flags to *env* and its sub-environments.
        """

        EnvApplier._set_smoothing(env, profile.actuator.enable_smoothing)
        EnvApplier._set_rudder_angle_scaling(env, profile.transfer.enable_rudder_angle_scaling)
        EnvApplier._set_initial_speed(env, profile.scenario.initial_speed)

        sub_envs = getattr(env, "sub_environments", None)
        if sub_envs is None or not sub_envs:
            return

        iterable = sub_envs.values() if isinstance(sub_envs, dict) else sub_envs
        for sub in iterable:
            EnvApplier._set_smoothing(sub, profile.actuator.enable_smoothing)
            EnvApplier._set_rudder_angle_scaling(sub, profile.transfer.enable_rudder_angle_scaling)
            EnvApplier._set_initial_speed(sub, profile.scenario.initial_speed)

    @staticmethod
    def _set_smoothing(env, value: bool) -> None:
        """
        Set enable_smoothing on *env* if the attribute exists.
        """

        if hasattr(env, "enable_smoothing"):
            env.enable_smoothing = value
            logger.debug(
                "EnvApplier: set enable_smoothing=%s on %s",
                value, type(env).__name__,
            )

    @staticmethod
    def _set_rudder_angle_scaling(env, value: bool) -> None:
        """
        Set enable_rudder_angle_scaling on *env* if the attribute exists.
        """

        if hasattr(env, "enable_rudder_angle_scaling"):
            env.enable_rudder_angle_scaling = value
            logger.debug(
                "EnvApplier: set enable_rudder_angle_scaling=%s on %s",
                value, type(env).__name__,
            )

    @staticmethod
    def _set_initial_speed(env, value: Optional[float]) -> None:
        """
        Set the initial surge speed (m/s), applied on the next reset(). None keeps
        the env default; envs without an initial_speed attribute are skipped.
        """

        if value is None:
            return
        if hasattr(env, "initial_speed"):
            env.initial_speed = value
            logger.debug(
                "EnvApplier: set initial_speed=%.3f m/s on %s",
                value, type(env).__name__,
            )
        else:
            logger.debug(
                "EnvApplier: %s has no 'initial_speed' attribute - skipped.",
                type(env).__name__,
            )


# ---------------------------------------------------------------------------
# Simulator applier
# ---------------------------------------------------------------------------

class SimulatorApplier:
    """
    Patches the simulator's actuator limits from the profile.
    """

    @staticmethod
    def apply(profile: VesselProfile, simulator) -> None:
        """
        Patch actuator limits on *simulator* if the profile overrides them.
        """

        if profile.actuator.angle_rad is not None or profile.actuator.rate_rad is not None:
            SimulatorApplier._patch_specs(profile, simulator)

    @staticmethod
    def _patch_specs(profile: VesselProfile, simulator) -> None:
        """
        Override rudder limits on the simulator's (mutable) ShipSpecifications
        and pass the specs on to the actuator model to keep both in sync.
        """

        specs = (
            getattr(simulator, "specs", None)
            or getattr(simulator, "_specs", None)
        )
        if specs is None:
            warnings.warn(
                f"SimulatorApplier: could not locate 'specs' or '_specs' on "
                f"{type(simulator).__name__}. Actuator limit overrides not applied.",
                UserWarning,
                stacklevel=3,
            )
            return

        if profile.actuator.angle_rad is not None:
            specs.max_rudder_angle = profile.actuator.angle_rad
            logger.debug(
                "SimulatorApplier: set max_rudder_angle=%.4f rad", profile.actuator.angle_rad
            )

        if profile.actuator.rate_rad is not None:
            specs.max_rudder_rate = profile.actuator.rate_rad
            logger.debug(
                "SimulatorApplier: set max_rudder_rate=%.4f rad/s", profile.actuator.rate_rad
            )

        actuators = getattr(simulator, "_actuators", None)
        if actuators is not None and hasattr(actuators, "specifications"):
            actuators.specifications = specs


# ---------------------------------------------------------------------------
# Action adapter
# ---------------------------------------------------------------------------

class ActionAdapter:
    """
    Transforms a raw [rudder, thrust] action using profile settings.
    """

    @staticmethod
    def transform(
        profile: VesselProfile,
        action: np.ndarray,
    ) -> np.ndarray:
        """
        Return a copy of *action*; with ThrustMode.FIXED the thrust (index 1)
        is replaced by profile.thrust.fixed_value.

        Args:
            profile: Profile whose thrust settings to apply.
            action: Array of shape (N,), N >= 2: [rudder, thrust, ...] in [-1, 1].

        Returns:
            New float32 array with the same shape as *action*.

        Raises:
            ValueError: If *action* has fewer than 2 elements.
        """

        action = np.array(action, dtype=np.float32, copy=True)

        if action.shape[0] < 2:
            raise ValueError(
                f"ActionAdapter.transform expects an action with at least 2 "
                f"elements [rudder, thrust], got shape {action.shape}."
            )

        if profile.thrust.mode == ThrustMode.FIXED:
            action[1] = profile.thrust.fixed_value

        return action
