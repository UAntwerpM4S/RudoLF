"""
config_profile.py – Vessel & run-time profiles for RL training and inference.

A :class:`VesselProfile` bundles every parameter that characterizes a single
training or inference run:

* initial vessel state (position, heading, body-fixed velocities)
* actuator constraints (max rudder angle / rate overrides)
* actuator smoothing on/off
* thrust mode (free-range or fixed value)
* rudder-angle conversion scale  (sim-to-sim transfer)
* ship model and dynamics model selection

All profile objects and their sub-configs are **frozen dataclasses**, meaning
they are immutable after construction.  Use :meth:`VesselProfile.with_overrides`
to derive modified copies without touching the original.

Profiles can be serialized to/from JSON and stored as named presets in the
:class:`ProfileRegistry`.

Typical usage
-------------
>>> from fw.config_profile import (
...     VesselProfile, ActuatorConfig, ThrustConfig, ThrustMode,
...     ModelConfig, RunMode, ProfileRegistry,
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
import dataclasses
import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
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
        TRANSFER: Inference in a *different* simulator than the one used for
            training; typically combined with ``enable_rudder_angle_scaling=True``.
    """

    TRAIN = auto()
    INFERENCE = auto()
    TRANSFER = auto()


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InitialVesselState:
    """
    Vessel state snapshot used at the start of every episode.

    Units and coordinate conventions match
    :class:`~fw.ships.vessel_state.VesselState`.

    Attributes:
        x: Earth-fixed x position [m].
        y: Earth-fixed y position [m].
        heading: Yaw angle in the earth-fixed frame [rad].
        u: Surge (forward) velocity [m/s].
        v: Sway (starboard) velocity [m/s].
        r: Yaw rate [rad/s].
    """

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    u: float = 0.0
    v: float = 0.0
    r: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return the state fields as a plain ``float``-valued dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InitialVesselState":
        """Construct an :class:`InitialVesselState` from a raw dictionary, ignoring unknown keys."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in known})


@dataclass(frozen=True)
class ScenarioConfig:
    """
    Scenario-level settings.

    Attributes:
        initial_state: Override the environment's built-in start state.
            *None* means the environment's default is used.
    """

    initial_state: Optional[InitialVesselState] = None


@dataclass(frozen=True)
class ActuatorConfig:
    """
    Actuator constraint overrides and smoothing toggle.

    Attributes:
        max_rudder_angle_deg: Override for max rudder deflection [deg].
            *None* keeps the ship-specification default.
        max_rudder_rate_deg_s: Override for max rudder rate [deg/s].
            *None* keeps the ship-specification default.
        enable_smoothing: Whether actuator rate-limiting / smoothing is
            active during simulation steps.
    """

    max_rudder_angle_deg: Optional[float] = None
    max_rudder_rate_deg_s: Optional[float] = None
    enable_smoothing: bool = True

    @property
    def angle_rad(self) -> Optional[float]:
        """``max_rudder_angle_deg`` converted to radians, or *None*."""
        return (
            None if self.max_rudder_angle_deg is None
            else math.radians(self.max_rudder_angle_deg)
        )

    @property
    def rate_rad(self) -> Optional[float]:
        """``max_rudder_rate_deg_s`` converted to radians per second, or *None*."""
        return (
            None if self.max_rudder_rate_deg_s is None
            else math.radians(self.max_rudder_rate_deg_s)
        )


@dataclass(frozen=True)
class ThrustConfig:
    """
    Thrust control settings.

    Attributes:
        mode: :attr:`ThrustMode.FREE` or :attr:`ThrustMode.FIXED`.
        fixed_value: Normalised thrust in ``[0.1, 1.0]`` used when
            ``mode == ThrustMode.FIXED``.  Ignored otherwise.
    """

    mode: ThrustMode = ThrustMode.FREE
    fixed_value: float = 1.0


@dataclass(frozen=True)
class TransferConfig:
    """
    Sim-to-sim transfer settings.

    Attributes:
        enable_rudder_angle_scaling: Whether to rescale the rudder action at
            every step to compensate for differing maximum rudder angles between
            the original (training) and target (inference) simulators.  The
            scale is computed internally by the simulator at runtime — no manual
            angle configuration is required.  *False* means the raw policy
            output is forwarded without rescaling.
    """

    enable_rudder_angle_scaling: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """
    Ship and dynamics model selection.

    Attributes:
        ship_model: Which :class:`~fw.ships.ship_factory.ShipModel` preset
            to load.
        dynamics_model: Which :class:`~fw.dynamics.dynamics_model.Model`
            family to use inside the simulator.
    """

    ship_model: ShipModel = ShipModel.MYZAKO_DEFAULT
    dynamics_model: Model = Model.FOSSEN


# ---------------------------------------------------------------------------
# Main profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VesselProfile:
    """
    Complete configuration profile for one RL training or inference run.

    All fields are immutable after construction.  Use
    :meth:`with_overrides` to create modified copies.

    Attributes:
        name: Human-readable identifier.
        description: Free-text description of the profile's purpose.
        version: Schema version string for forwards-compatibility checks.
        run_mode: Intended use-case (:class:`RunMode`).
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
        Return a new :class:`VesselProfile` with selected top-level fields
        replaced.

        Uses :func:`dataclasses.replace` so type safety is preserved:
        callers must pass typed sub-config instances
        (e.g. ``ActuatorConfig(...)``) rather than plain dicts.
        Unchanged fields are carried over from *self* — no JSON round-trip,
        no risk of silently dropping nested fields.

        Raises:
            TypeError: If any key in *kwargs* is not a valid field name, or
                if a sub-config field receives a plain :class:`dict` instead
                of the expected dataclass type.

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

        # Guard against passing a plain dict for a dataclass-typed field.
        # dataclasses.replace would accept it silently, leading to hard-to-
        # debug AttributeErrors later when the profile is used.
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
        Raise :exc:`ValueError` for any obviously invalid field combination.

        All violations are collected and reported together rather than
        failing on the first one.  Suspicious-but-legal combinations emit
        a :mod:`warnings` warning instead of raising.
        """
        errors: List[str] = []

        if self.thrust.mode == ThrustMode.FIXED:
            if not (0.1 <= self.thrust.fixed_value <= 1.0):
                errors.append(
                    f"thrust.fixed_value must be in [0.1, 1.0], "
                    f"got {self.thrust.fixed_value}."
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
        Return a JSON-serializable dictionary representation.

        Enum fields are stored by name/value rather than as raw Enum objects
        so the output is safe for :func:`json.dumps` without a custom encoder.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "run_mode": self.run_mode.name,
            "scenario": {
                "initial_state": (
                    self.scenario.initial_state.to_dict()
                    if self.scenario.initial_state else None
                )
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
                "ship_model": self.model.ship_model.name,
                "dynamics_model": self.model.dynamics_model.name,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VesselProfile":
        """
        Reconstruct a :class:`VesselProfile` from a dictionary.

        Unknown keys inside each sub-config are silently ignored so that
        profiles serialized by a newer schema version remain loadable by
        older code.
        """
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            version=d.get("version", "1.0"),
            run_mode=RunMode[d["run_mode"]],
            scenario=ScenarioConfig(
                initial_state=(
                    InitialVesselState.from_dict(d["scenario"]["initial_state"])
                    if d["scenario"]["initial_state"] else None
                )
            ),
            actuator=ActuatorConfig(
                **_filter_fields(ActuatorConfig, d["actuator"])
            ),
            thrust=ThrustConfig(
                mode=ThrustMode[d["thrust"]["mode"]],
                fixed_value=d["thrust"]["fixed_value"],
            ),
            transfer=TransferConfig(
                **_filter_fields(TransferConfig, d["transfer"])
            ),
            model=ModelConfig(
                ship_model=ShipModel[d["model"]["ship_model"]],
                dynamics_model=Model[d["model"]["dynamics_model"]],
            ),
        )

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "VesselProfile":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    def fingerprint(self) -> str:
        """
        Return a stable hex digest of the profile's serialized content.

        Useful for tagging model checkpoints or experiment logs.  The digest
        is computed over the canonical JSON representation (keys sorted) so
        it is stable across Python versions and dict insertion orders.

        Note: MD5 is used purely as a fast, non-cryptographic content hash.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(canonical.encode(), usedforsecurity=False).hexdigest()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a human-readable summary of the profile's key settings."""
        lines = [f"VesselProfile '{self.name}' [{self.run_mode.name}]"]
        if self.description:
            lines.append(f"  description  : {self.description}")
        lines.append(f"  ship_model   : {self.model.ship_model.name}")
        lines.append(f"  dynamics     : {self.model.dynamics_model.name}")

        s = self.scenario.initial_state
        if s is not None:
            lines.append(
                f"  initial_state: x={s.x}, y={s.y}, "
                f"heading={math.degrees(s.heading):.1f}°, "
                f"u={s.u}, v={s.v}, r={s.r}"
            )
        else:
            lines.append("  initial_state: (environment default)")

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
    Return only the keys from *d* that are declared fields on *dc_type*.

    This makes :meth:`VesselProfile.from_dict` tolerant of extra keys written
    by a newer schema version without silently dropping required ones.
    """
    known = {f.name for f in dataclasses.fields(dc_type)}
    return {k: v for k, v in d.items() if k in known}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ProfileRegistry:
    """
    In-memory registry of named :class:`VesselProfile` presets.

    Because :class:`VesselProfile` is a frozen dataclass, the registry stores
    direct references (no deep-copy on write).  :meth:`get` returns a shallow
    copy — sufficient for immutable objects — so callers can hold and pass the
    profile freely without any risk of mutating the registry entry.
    """

    _registry: Dict[str, VesselProfile] = {}

    @classmethod
    def register(cls, profile: VesselProfile, *, overwrite: bool = False) -> None:
        """
        Add a profile to the registry.

        Args:
            profile: The :class:`VesselProfile` to register.
            overwrite: Replace an existing entry with the same name when
                *True*.  Raises :exc:`KeyError` on collision when *False*.
        """
        if profile.name in cls._registry and not overwrite:
            raise KeyError(
                f"Profile '{profile.name}' already exists. "
                "Pass overwrite=True to replace it."
            )
        # Validate before committing so a mis-configured preset is caught at
        # import time rather than silently sitting in the registry until a
        # caller happens to call validate() themselves.
        profile.validate()
        cls._registry[profile.name] = profile

    @classmethod
    def get(cls, name: str) -> VesselProfile:
        """
        Retrieve a registered profile by name.

        Returns a shallow copy — safe because the profile and all its
        sub-configs are frozen — so the caller can hold the object freely.

        Raises:
            KeyError: If no profile with that name is registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry) or "(none)"
            raise KeyError(f"Unknown profile '{name}'. Available: {available}")
        return copy.copy(cls._registry[name])

    @classmethod
    def list_names(cls) -> List[str]:
        """Return all registered profile names in insertion order."""
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
    description="Standard profile with Fossen dynamics.",
    run_mode=RunMode.TRAIN,
    model=ModelConfig(
        ship_model=ShipModel.MYZAKO_DEFAULT,
        dynamics_model=Model.FOSSEN,
    ),
    actuator=ActuatorConfig(enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FREE),
))

ProfileRegistry.register(VesselProfile(
    name="pysim_to_fhsim_transfer_profile",
    description=(
        "Run a PySim-trained policy inside FhSim with rudder angle scaling."
    ),
    run_mode=RunMode.TRANSFER,
    model=ModelConfig(
        ship_model=ShipModel.MYZAKO_DEFAULT,
        dynamics_model=Model.FOSSEN,
    ),
    actuator=ActuatorConfig(enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=0.8),
    transfer=TransferConfig(enable_rudder_angle_scaling=True),
))

ProfileRegistry.register(VesselProfile(
    name="ugent_surrogate_training_profile",
    description="Training profile for UGent's surrogate model.",
    run_mode=RunMode.TRAIN,
    model=ModelConfig(
        ship_model=ShipModel.TRIPOLI,
        dynamics_model=Model.UGENT_SURROGATE,
    ),
    scenario=ScenarioConfig(initial_state=InitialVesselState(
        x= 0.0,
        y= 0.0,
        heading= 0.0,
        u= 4.8,
        v= 0.0,
        r= 0.0)),
    actuator=ActuatorConfig(max_rudder_angle_deg=35.0, enable_smoothing=True),
    thrust=ThrustConfig(mode=ThrustMode.FIXED, fixed_value=1.0),
))

ProfileRegistry.register(VesselProfile(
    name="train_lunar_default",
    description=(
        "LunarLander training profile. Ship-model and dynamics fields are "
        "unused by the Lunar environment but kept for schema consistency."
    ),
    run_mode=RunMode.TRAIN,
    model=ModelConfig(
        ship_model=ShipModel.MYZAKO_DEFAULT,
        dynamics_model=Model.FOSSEN,
    ),
    actuator=ActuatorConfig(enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FREE),
))

ProfileRegistry.register(VesselProfile(
    name="train_pysim_polar_default",
    description="PySimPolarEnv training profile with Fossen dynamics.",
    run_mode=RunMode.TRAIN,
    model=ModelConfig(
        ship_model=ShipModel.MYZAKO_DEFAULT,
        dynamics_model=Model.FOSSEN,
    ),
    actuator=ActuatorConfig(enable_smoothing=False),
    thrust=ThrustConfig(mode=ThrustMode.FREE),
))
