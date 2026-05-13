"""
profile_applier.py – Apply a :class:`~fw.profile.VesselProfile` to the framework.

Translates a :class:`VesselProfile` into side effects on existing framework
objects so that no call-sites outside this module need to be rewritten:

* ``CONFIG_PARAMETERS`` dict  – ship_model, dynamics_model
* :class:`~fw.environments.base_env.BaseEnv`  – smoothing flag (+ sub-envs)
* :class:`~fw.simulators.base_simulator.BaseSimulator`  – initial vessel
  state, actuator-limit overrides
* Raw ``[rudder, thrust]`` action arrays  – rudder scaling, fixed-thrust

Typical usage
-------------
::

    from fw.profile import ProfileRegistry
    from fw.profile_applier import apply_profile

    profile = ProfileRegistry.get("inference_pysim")

    # 1. Patch CONFIG_PARAMETERS at startup
    from fw.config import get_config_parameters
    cfg = get_config_parameters()
    apply_profile(profile, config=cfg)

    # 2. Configure smoothing after the env is built
    env = build_default_environment("PySimEnv")
    apply_profile(profile, env=env)

    # 3. Rescale / fix thrust at every step
    raw_action = agent.predict(obs)
    action = apply_profile(profile, action=raw_action)
    obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import math
import logging
import warnings
import numpy as np

from typing import Any, Dict, Optional
from fw.ships.vessel_state import VesselState
from fw.config_profile import ThrustMode, VesselProfile

EPS = 1e-6
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
    Apply *profile* settings to any combination of framework targets.

    Each keyword argument is independent; pass only what you have at hand.
    ``config``, ``env``, and ``simulator`` are mutated in-place.  When
    ``action`` is supplied the function returns a transformed copy — the
    original array is never modified.

    Args:
        profile: The :class:`~fw.profile.VesselProfile` to apply.
        config: Mutable ``CONFIG_PARAMETERS`` dict from
            :func:`~fw.config.get_config_parameters`. Updated in-place.
        env: A :class:`~fw.environments.base_env.BaseEnv` instance.
            ``enable_smoothing`` is propagated to sub-environments too.
        simulator: A :class:`~fw.simulators.base_simulator.BaseSimulator`
            instance.  Reset to ``profile.scenario.initial_state`` when set,
            and actuator limits are patched when overrides are present.
        action: NumPy array of shape ``(2,)`` with ``[rudder, thrust]`` in
            ``[-1, 1]``.  Returns the transformed array.

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
        return ActionAdapter.transform(profile, action, simulator=simulator)

    return None


# ---------------------------------------------------------------------------
# Config applier
# ---------------------------------------------------------------------------

class ConfigApplier:
    """
    Patches a ``CONFIG_PARAMETERS`` dict in-place.
    """

    @staticmethod
    def apply(profile: VesselProfile, config: Dict[str, Any]) -> None:
        """
        Patch ship_model and dynamics_model into *config* from *profile*.
        """

        config["ship_model"] = profile.model.ship_model
        config["dynamics_model"] = profile.model.dynamics_model
        logger.debug(
            "ConfigApplier: set ship_model=%s, dynamics_model=%s",
            config["ship_model"], config["dynamics_model"],
        )


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
        Apply env-level flags from *profile* to *env* and all its sub-environments.
        """

        EnvApplier._set_smoothing(env, profile.actuator.enable_smoothing)
        EnvApplier._set_rudder_angle_scaling(env, profile.transfer.enable_rudder_angle_scaling)

        sub_envs = getattr(env, "sub_environments", None)
        if sub_envs is None or not sub_envs:
            return

        iterable = sub_envs.values() if isinstance(sub_envs, dict) else sub_envs
        for sub in iterable:
            EnvApplier._set_smoothing(sub, profile.actuator.enable_smoothing)
            EnvApplier._set_rudder_angle_scaling(sub, profile.transfer.enable_rudder_angle_scaling)

    @staticmethod
    def _set_smoothing(env, value: bool) -> None:
        """
        Set `enable_smoothing` on *env* if the attribute exists.
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
        Set `enable_rudder_angle_scaling` on *env* if the attribute exists.
        """

        if hasattr(env, "enable_rudder_angle_scaling"):
            env.enable_rudder_angle_scaling = value
            logger.debug(
                "EnvApplier: set enable_rudder_angle_scaling=%s on %s",
                value, type(env).__name__,
            )


# ---------------------------------------------------------------------------
# Simulator applier
# ---------------------------------------------------------------------------

class SimulatorApplier:
    """
    Resets the simulator to the profile's initial state and patches actuator limits.
    """

    @staticmethod
    def apply(profile: VesselProfile, simulator) -> None:
        """
        Reset *simulator* state and patch actuator limits from *profile*.
        """

        if profile.scenario.initial_state is not None:
            SimulatorApplier._reset_state(profile, simulator)

        if profile.actuator.angle_rad is not None or profile.actuator.rate_rad is not None:
            SimulatorApplier._patch_specs(profile, simulator)

    @staticmethod
    def _reset_state(profile: VesselProfile, simulator) -> None:
        """
        Call `simulator.reset` with the initial state defined in *profile*.
        """

        def nz(value, fallback):
            """Return *fallback* when *value* is numerically zero, otherwise *value*."""
            return fallback if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=EPS) else value

        orig = simulator.state
        s = profile.scenario.initial_state
        simulator.reset(VesselState(
            terminated=False,
            truncated=False,
            x=nz(s.x, orig.x),
            y=nz(s.y, orig.y),
            heading=nz(s.heading, orig.heading),
            u=nz(s.u, orig.u),
            v=nz(s.v, orig.v),
            r=nz(s.r, orig.r),
        ))
        logger.debug(
            "SimulatorApplier: reset to x=%.1f y=%.1f heading=%.3f rad",
            s.x, s.y, s.heading,
        )

    @staticmethod
    def _patch_specs(profile: VesselProfile, simulator) -> None:
        """
        Override actuator limits on the simulator's :class:`ShipSpecifications`.

        :class:`~fw.ships.ship_specs.ShipSpecifications` is a frozen dataclass;
        `object.__setattr__` is used to write the new values in-place.  The
        updated specs object is then propagated to the actuator model so both
        stay in sync.
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
            object.__setattr__(specs, "max_rudder_angle", profile.actuator.angle_rad)
            logger.debug(
                "SimulatorApplier: set max_rudder_angle=%.4f rad", profile.actuator.angle_rad
            )

        if profile.actuator.rate_rad is not None:
            object.__setattr__(specs, "max_rudder_rate", profile.actuator.rate_rad)
            logger.debug(
                "SimulatorApplier: set max_rudder_rate=%.4f rad/s", profile.actuator.rate_rad
            )

        # Keep the actuator model in sync with the updated specs
        actuators = getattr(simulator, "_actuators", None)
        if actuators is not None and hasattr(actuators, "specifications"):
            actuators.specifications = specs


# ---------------------------------------------------------------------------
# Action adapter
# ---------------------------------------------------------------------------

class ActionAdapter:
    """
    Transforms a raw ``[rudder, thrust]`` action using profile settings.
    """

    @staticmethod
    def transform(
        profile: VesselProfile,
        action: np.ndarray,
        *,
        simulator=None,  # noqa: ARG004  kept for API compatibility
    ) -> np.ndarray:
        """
        Return a transformed copy of *action*.

        When ``thrust_mode == FIXED`` the thrust component (index 1) is
        replaced with ``profile.thrust.fixed_value`` directly.  The original
        array is never modified.

        Args:
            profile: Profile whose thrust settings to apply.
            action: Array of shape ``(N,)`` with ``N >= 2``.  Index 0 is
                rudder, index 1 is thrust, both nominally in ``[-1, 1]``.
            simulator: Accepted for API compatibility; currently unused.

        Returns:
            New ``float32`` array with the same shape as *action*.

        Raises:
            ValueError: If *action* has fewer than 2 elements.
        """

        action = np.array(action, dtype=np.float32, copy=True)

        if action.shape[0] < 2:
            raise ValueError(
                f"ActionAdapter.transform expects an action with at least 2 "
                f"elements [rudder, thrust], got shape {action.shape}."
            )

        # --- Fixed thrust -----------------------------------------------------
        if profile.thrust.mode == ThrustMode.FIXED:
            action[1] = profile.thrust.fixed_value

        return action
