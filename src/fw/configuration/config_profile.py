"""
Vessel and run-time profiles for RL training and inference.

A VesselProfile bundles the settings of one run: initial speed, actuator
limits and smoothing, thrust mode (free or fixed), rudder-angle scaling for
sim-to-sim transfer, and ship/dynamics model selection.

Profiles and their sub-configs are frozen dataclasses; derive variants with
`VesselProfile.with_overrides`. Profiles serialize to/from JSON and can be
stored as named presets in ProfileRegistry.

Typical usage
-------------
>>> from fw.configuration.config_profile import (
...     VesselProfile, ActuatorConfig, ThrustConfig, ThrustMode,
...     ScenarioConfig, ModelConfig, RunMode, ProfileRegistry,
... )

# Build a custom profile
>>> p = VesselProfile(
...     name="my_run",
... )
>>> p.validate()

# Derive an inference variant without modifying the original
>>> p_infer = p.with_overrides(
...     name="my_run_inference",
...     actuator=ActuatorConfig(enable_smoothing=False),
...     thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=0.8),
... )

# Load a built-in preset
>>> p = ProfileRegistry.get("default_profile")
"""

from __future__ import annotations

import copy
import json
import math
import hashlib
import warnings
import dataclasses

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from fw.dynamics.dynamics_model import Model
from fw.ships.ship_factory import ShipModel


class ThrustMode(Enum):
    """
    Thrust control strategy during a run.

    Attributes:
        FREE: The agent controls thrust freely within [min_thrust, max_thrust].
        FIXED: Thrust is clamped to :attr:`ThrustConfig.fixed_value` every step.
    """

    FREE = auto()
    FIXED = auto()


class RunMode(Enum):
    """
    Intended use-case for a profile.

    Attributes:
        TRAIN: Standard reinforcement-learning training.
        INFERENCE: Running a trained policy (no weight updates).
        TRANSFER: Inference in a different simulator than used for training;
            typically with `enable_rudder_angle_scaling=True`.
    """

    TRAIN = auto()
    INFERENCE = auto()
    TRANSFER = auto()


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioConfig:
    """
    Scenario-level settings.

    Attributes:
        initial_speed: Initial surge velocity override [m/s], applied via the
            env's `initial_speed` setter after each reset. None keeps the
            environment default.
        initial_thrust: Thrust the actuators start an episode with, in thrust
            units ([min_thrust, max_thrust] of the ship, checked at reset). None:
            the fixed thrust in ThrustMode.FIXED, else the thrust that holds
            initial_speed (Fossen, Nomoto, UGent and B01 surrogates; calm water),
            else min_thrust.
    """

    initial_speed: Optional[float] = None
    initial_thrust: Optional[float] = None


@dataclass(frozen=True)
class ActuatorConfig:
    """
    Actuator constraint overrides and smoothing toggle.

    Attributes:
        max_rudder_angle_deg: Max rudder angle override [deg]; None keeps the
            ship-specification value.
        max_rudder_rate_deg_s: Max rudder rate override [deg/s]; None keeps the
            ship-specification value.
        enable_smoothing: Whether actuator rate limiting/smoothing is active.
    """

    max_rudder_angle_deg: Optional[float] = None
    max_rudder_rate_deg_s: Optional[float] = None
    enable_smoothing: bool = True

    @property
    def angle_rad(self) -> Optional[float]:
        """
        `max_rudder_angle_deg` converted to radians, or *None*.
        """

        return (
            None if self.max_rudder_angle_deg is None
            else math.radians(self.max_rudder_angle_deg)
        )

    @property
    def rate_rad(self) -> Optional[float]:
        """
        `max_rudder_rate_deg_s` converted to radians per second, or *None*.
        """

        return (
            None if self.max_rudder_rate_deg_s is None
            else math.radians(self.max_rudder_rate_deg_s)
        )


@dataclass(frozen=True)
class ThrustConfig:
    """
    Thrust control settings.

    Attributes:
        mode: ThrustMode.FREE or ThrustMode.FIXED.
        fixed_value: Thrust action that replaces the policy's in FIXED mode, on the
            action scale (-1 = min_thrust, 1 = max_thrust). Must be in [-1, 1].
    """

    mode: ThrustMode = ThrustMode.FREE
    fixed_value: float = 1.0


@dataclass(frozen=True)
class TransferConfig:
    """
    Sim-to-sim transfer settings.

    Attributes:
        enable_rudder_angle_scaling: Rescale the rudder action every step to
            compensate for different max rudder angles between the training and
            inference simulators. The scale is computed by the simulator at
            runtime. False forwards the raw policy output.
    """

    enable_rudder_angle_scaling: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """
    Ship and dynamics model selection.

    Attributes:
        ship_model: ShipModel preset; None keeps the framework default.
        dynamics_model: Dynamics Model used by the simulator; None keeps the
            framework default.
    """

    ship_model: Optional[ShipModel] = None
    dynamics_model: Optional[Model] = None


# ---------------------------------------------------------------------------
# Main profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VesselProfile:
    """
    Complete configuration profile for one RL training or inference run (immutable).

    Attributes:
        name: Human-readable identifier.
        description: Free-text description of the profile's purpose.
        version: Schema version string.
        run_mode: Intended use case (RunMode).
        scenario: Episode start-state settings.
        actuator: Actuator constraint overrides and smoothing toggle.
        thrust: Thrust control strategy.
        transfer: Sim-to-sim rudder rescaling.
        model: Ship and dynamics model selection.
    """

    name: str = "default"
    description: str = ""
    version: str = "1.0"

    run_mode: RunMode = RunMode.TRAIN

    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    thrust: ThrustConfig = field(default_factory=ThrustConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # ------------------------------------------------------------------
    # Derived convenience
    # ------------------------------------------------------------------

    def with_overrides(self, **kwargs) -> "VesselProfile":
        """
        Return a copy with selected top-level fields replaced (dataclasses.replace).

        Sub-configs must be passed as typed instances, not dicts.

        Raises:
            TypeError: On an unknown field name or a plain dict value.

        Example::

            inference = training_profile.with_overrides(
                name="my_inference",
                actuator=ActuatorConfig(enable_smoothing=False),
                thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=0.8),
            )
        """

        field_map = {f.name: f for f in dataclasses.fields(self)}

        unknown = set(kwargs) - field_map.keys()
        if unknown:
            raise TypeError(
                f"with_overrides() got unexpected field name(s): "
                f"{', '.join(sorted(unknown))}"
            )

        # dataclasses.replace would silently accept a dict, failing much later.
        for name, value in kwargs.items():
            expected = field_map[name].type
            if isinstance(value, dict):
                raise TypeError(
                    f"with_overrides(): field '{name}' received a plain dict. "
                    f"Pass a typed instance (e.g. {expected}(...)) instead."
                )

        return dataclasses.replace(self, **kwargs)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Raise ValueError listing all invalid fields.

        Suspicious but legal combinations (TRANSFER without rudder scaling)
        only emit a UserWarning.
        """

        errors: List[str] = []

        if self.thrust.mode == ThrustMode.FIXED:
            # "not (a <= x <= b)" also rejects NaN
            if not -1.0 <= self.thrust.fixed_value <= 1.0:
                errors.append(
                    f"thrust.fixed_value must be in [-1, 1] (action scale), "
                    f"got {self.thrust.fixed_value}."
                )

        # "not (a <= x < b)" also rejects NaN
        if (self.scenario.initial_speed is not None
                and not (0.0 <= self.scenario.initial_speed < float("inf"))):
            errors.append(
                f"scenario.initial_speed must be a finite, non-negative speed, "
                f"got {self.scenario.initial_speed}."
            )

        # "not (x >= 0)" also rejects NaN
        if (self.scenario.initial_thrust is not None
                and not (0.0 <= self.scenario.initial_thrust < float("inf"))):
            errors.append(
                f"scenario.initial_thrust must be a finite, non-negative thrust, "
                f"got {self.scenario.initial_thrust}."
            )

        if (self.actuator.max_rudder_angle_deg is not None
                and self.actuator.max_rudder_angle_deg <= 0):
            errors.append(
                f"actuator.max_rudder_angle_deg must be positive, "
                f"got {self.actuator.max_rudder_angle_deg}."
            )

        if (self.actuator.max_rudder_rate_deg_s is not None
                and self.actuator.max_rudder_rate_deg_s <= 0):
            errors.append(
                f"actuator.max_rudder_rate_deg_s must be positive, "
                f"got {self.actuator.max_rudder_rate_deg_s}."
            )

        if errors:
            raise ValueError(
                f"VesselProfile '{self.name}' validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

        if self.run_mode == RunMode.TRANSFER and not self.transfer.enable_rudder_angle_scaling:
            warnings.warn(
                f"Profile '{self.name}': run_mode is TRANSFER but "
                "transfer.enable_rudder_angle_scaling is False. "
                "Did you forget to enable rudder angle scaling?",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable dict (enums stored by name).
        """

        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "run_mode": self.run_mode.name,
            "scenario": {
                "initial_speed": self.scenario.initial_speed,
                "initial_thrust": self.scenario.initial_thrust,
            },
            "actuator": {
                "max_rudder_angle_deg": self.actuator.max_rudder_angle_deg,
                "max_rudder_rate_deg_s": self.actuator.max_rudder_rate_deg_s,
                "enable_smoothing": self.actuator.enable_smoothing,
            },
            "thrust": {
                "mode": self.thrust.mode.name,
                "fixed_value": self.thrust.fixed_value,
            },
            "transfer": {
                "enable_rudder_angle_scaling": self.transfer.enable_rudder_angle_scaling,
            },
            "model": {
                "ship_model": self.model.ship_model.name if self.model.ship_model is not None else None,
                "dynamics_model": self.model.dynamics_model.name if self.model.dynamics_model is not None else None,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VesselProfile":
        """
        Reconstruct a VesselProfile from a dict.

        Unknown sub-config keys are ignored and missing blocks use their
        defaults, so other schema versions stay loadable. 'name' is required.
        """

        scenario_d = d.get("scenario", {})
        initial_speed = scenario_d.get("initial_speed") if scenario_d else None
        initial_thrust = scenario_d.get("initial_thrust") if scenario_d else None

        thrust_d = d.get("thrust", {})
        model_d = d.get("model", {})

        return cls(
            name=d["name"],
            description=d.get("description", ""),
            version=d.get("version", "1.0"),
            run_mode=RunMode[d.get("run_mode", RunMode.TRAIN.name)],
            scenario=ScenarioConfig(
                initial_speed=(
                    float(initial_speed) if initial_speed is not None else None
                ),
                initial_thrust=(
                    float(initial_thrust) if initial_thrust is not None else None
                ),
            ),
            actuator=ActuatorConfig(
                **_filter_fields(ActuatorConfig, d.get("actuator", {}))
            ),
            thrust=ThrustConfig(
                mode=ThrustMode[thrust_d.get("mode", ThrustMode.FREE.name)],
                fixed_value=thrust_d.get("fixed_value", ThrustConfig().fixed_value),
            ),
            transfer=TransferConfig(
                **_filter_fields(TransferConfig, d.get("transfer", {}))
            ),
            model=ModelConfig(
                ship_model=(
                    ShipModel[model_d["ship_model"]]
                    if model_d.get("ship_model") is not None
                    else None
                ),
                dynamics_model=(
                    Model[model_d["dynamics_model"]]
                    if model_d.get("dynamics_model") is not None
                    else None
                ),
            ),
        )

    def to_json(self, *, indent: int = 2) -> str:
        """
        Serialize to a JSON string.
        """

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "VesselProfile":
        """
        Deserialize from a JSON string.
        """

        return cls.from_dict(json.loads(s))

    def fingerprint(self) -> str:
        """
        Return a stable MD5 hex digest (non-cryptographic) of the sorted-key JSON.

        Useful for tagging checkpoints or experiment logs.
        """

        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(canonical.encode(), usedforsecurity=False).hexdigest()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Return a human-readable summary of the profile's key settings.
        """

        lines = [f"VesselProfile '{self.name}' [{self.run_mode.name}]"]
        if self.description:
            lines.append(f"  description  : {self.description}")
        lines.append(f"  ship_model   : {self.model.ship_model.name if self.model.ship_model is not None else '(config default)'}")
        lines.append(f"  dynamics     : {self.model.dynamics_model.name if self.model.dynamics_model is not None else '(config default)'}")

        if self.scenario.initial_speed is not None:
            lines.append(f"  initial_speed: {self.scenario.initial_speed} m/s")
        else:
            lines.append("  initial_speed: (environment default)")
        if self.scenario.initial_thrust is not None:
            lines.append(f"  initial_thrust: {self.scenario.initial_thrust} (thrust)")
        elif self.thrust.mode == ThrustMode.FIXED:
            lines.append("  initial_thrust: (fixed thrust)")
        else:
            lines.append("  initial_thrust: (holds initial_speed, else min_thrust)")

        lines.append(f"  smoothing    : {self.actuator.enable_smoothing}")
        if self.actuator.max_rudder_angle_deg is not None:
            lines.append(f"  max rudder   : {self.actuator.max_rudder_angle_deg}°")
        if self.actuator.max_rudder_rate_deg_s is not None:
            lines.append(f"  max rud.rate : {self.actuator.max_rudder_rate_deg_s}°/s")

        lines.append(f"  thrust_mode  : {self.thrust.mode.name}")
        if self.thrust.mode == ThrustMode.FIXED:
            lines.append(f"  fixed_thrust : {self.thrust.fixed_value}")
        lines.append(f"  rudder_angle_scaling: {self.transfer.enable_rudder_angle_scaling}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_fields(dc_type: type, d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return only the keys of `d` that are fields of `dc_type`.
    """

    known = {f.name for f in dataclasses.fields(dc_type)}
    return {k: v for k, v in d.items() if k in known}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ProfileRegistry:
    """
    In-memory registry of named VesselProfile presets.

    Profiles are frozen, so references are stored directly and `get` returns
    a shallow copy.
    """

    _registry: Dict[str, VesselProfile] = {}

    @classmethod
    def register(cls, profile: VesselProfile, *, overwrite: bool = False) -> None:
        """
        Validate and add a profile to the registry.

        Args:
            profile: The VesselProfile to register.
            overwrite: Replace an existing entry with the same name.

        Raises:
            KeyError: On a name collision when `overwrite` is False.
            ValueError: If the profile fails validation.
        """

        if profile.name in cls._registry and not overwrite:
            raise KeyError(
                f"Profile '{profile.name}' already exists. "
                "Pass overwrite=True to replace it."
            )
        # Validate now so a bad preset fails at import time.
        profile.validate()
        cls._registry[profile.name] = profile

    @classmethod
    def get(cls, name: str) -> VesselProfile:
        """
        Return a (shallow) copy of a registered profile.

        Raises:
            KeyError: If no profile with that name is registered.
        """

        if name not in cls._registry:
            available = ", ".join(cls._registry) or "(none)"
            raise KeyError(f"Unknown profile '{name}'. Available: {available}")
        return copy.copy(cls._registry[name])

    @classmethod
    def list_names(cls) -> List[str]:
        """
        Return all registered profile names in insertion order.
        """

        return list(cls._registry.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a profile from the registry.

        Raises:
            KeyError: If no profile with that name is registered.
        """

        if name not in cls._registry:
            raise KeyError(f"No profile named '{name}'.")
        del cls._registry[name]


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

ProfileRegistry.register(VesselProfile(
    name="default_profile",
    description="Default configuration profile.",
    run_mode=RunMode.TRAIN,
    actuator=ActuatorConfig(enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FREE),
))

ProfileRegistry.register(VesselProfile(
    name="fh_simulator_profile",
    description="Configuration profile for the FH simulator.",
    run_mode=RunMode.TRAIN,
    model=ModelConfig(
        ship_model=ShipModel.MYZAKO_37,
        dynamics_model=Model.FH_TABULAR,
    ),
    actuator=ActuatorConfig(max_rudder_angle_deg=75.0, enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FREE),
))

ProfileRegistry.register(VesselProfile(
    name="multi_policy_colav_profile",
    description="Configuration profile for multi-policy COLAV.",
    run_mode=RunMode.TRAIN,
    model=ModelConfig(
        ship_model=ShipModel.MYZAKO_37,
        dynamics_model=Model.FOSSEN_SWAY_FLIP,
    ),
    actuator=ActuatorConfig(max_rudder_angle_deg=46.5, enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FREE),
))

ProfileRegistry.register(VesselProfile(
    name="ugent_surrogate_profile",
    description="Configuration profile for UGent's surrogate model.",
    run_mode=RunMode.TRAIN,
    model=ModelConfig(
        ship_model=ShipModel.TRIPOLI,
        dynamics_model=Model.UGENT_SURROGATE,
    ),
    scenario=ScenarioConfig(initial_speed=0.0),
    actuator=ActuatorConfig(max_rudder_angle_deg=35.0, enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FREE),
))

ProfileRegistry.register(VesselProfile(
    name="ugent_b01_surrogate_profile",
    description="Configuration profile for UGent's B01 model-scale (towing tank) surrogate model.",
    run_mode=RunMode.TRAIN,
    model=ModelConfig(
        ship_model=ShipModel.B01,
        dynamics_model=Model.UGENT_B01_SURROGATE,
    ),
    # Model-scale start speed [m/s], below the B01 surge limit of 1.0 m/s (steady
    # speed at full thrust drops from ~0.93 m/s at 379 % UKC to ~0.41 at 35 %).
    # Refused by the full-scale PySim environments (SUPPORTS_MODEL_SCALE_SHIPS).
    scenario=ScenarioConfig(initial_speed=0.8),
    actuator=ActuatorConfig(max_rudder_angle_deg=35.0, enable_smoothing=True),
    thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=1.0),
))

ProfileRegistry.register(VesselProfile(
    name="uantwerp_surrogate_profile",
    description="Configuration profile for UAntwerp's surrogate model.",
    run_mode=RunMode.TRAIN,
    model=ModelConfig(
        ship_model=ShipModel.MYZAKO_37,
        dynamics_model=Model.UANTWERP,
    ),
    actuator=ActuatorConfig(max_rudder_angle_deg=75.0, enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FREE),
))

ProfileRegistry.register(VesselProfile(
    name="myzako_pysim_to_fhsim_transfer_profile",
    description=(
        "Run a PySim-trained policy inside FhSim with rudder angle scaling."
    ),
    run_mode=RunMode.TRANSFER,
    model=ModelConfig(
        ship_model=ShipModel.MYZAKO_37,
        dynamics_model=Model.FH_TABULAR,
    ),
    actuator=ActuatorConfig(enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FREE),
    transfer=TransferConfig(enable_rudder_angle_scaling=True),
))

ProfileRegistry.register(VesselProfile(
    name="tripoli_pysim_to_fhsim_transfer_profile",
    description=(
        "Run a PySim-trained policy inside FhSim with rudder angle scaling."
    ),
    run_mode=RunMode.TRANSFER,
    model=ModelConfig(
        ship_model=ShipModel.TRIPOLI,
        dynamics_model=Model.FH_TABULAR,
    ),
    actuator=ActuatorConfig(max_rudder_angle_deg=35.0, enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=1.0),
    transfer=TransferConfig(enable_rudder_angle_scaling=True),
))
