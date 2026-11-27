"""
Build pipeline environments without the factory cache and resolve block profiles.
"""

from __future__ import annotations

from typing import Any, Dict
from fw.surrogate.presets import get_preset
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.configuration.config import load_config_parameters
from fw.configuration.profile_applier import apply_profile
from fw.scheduling.run_spec import BLOCK_CONFIG_KEYS, PipelineBlock
from fw.configuration.config_profile import ModelConfig, ProfileRegistry, VesselProfile

from fw.configuration.env_factory import (
    resolve_environment_name,
    airsim_env_name,
    COLAV_ENVS,
    SCENARIO_ENVS,
    UA_SURROGATE_ENVS,
    ENVIRONMENT_CLASSES,
    ENVIRONMENT_CONFIG_BUILDERS,
)


# Instantiated directly: the standard env factory caches by canonical name.
# AirSim is not supported by the pipeline (see build_block_environment).
_DIRECT_ENV_BUILDERS = {name: cls for name, cls in ENVIRONMENT_CLASSES.items() if name != airsim_env_name}
_CONFIG_BUILDERS = {name: ENVIRONMENT_CONFIG_BUILDERS[name] for name in _DIRECT_ENV_BUILDERS}

# Dynamics model that reads each surrogate config.
SURROGATE_CONFIG_MODEL = {
    "ua_surrogate_config": Model.UANTWERP,
    "ugent_surrogate_config": Model.UGENT_SURROGATE,
    "ugent_b01_surrogate_config": Model.UGENT_B01_SURROGATE,
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """
    Recursively merge overrides into a copy of base.
    """

    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_block_profile(block: PipelineBlock) -> VesselProfile:
    """
    Resolve the vessel profile for a pipeline block.
    """

    profile = ProfileRegistry.get(block.profile)

    if block.ship_model or block.dynamics_model:
        current_model = profile.model
        profile = profile.with_overrides(model=ModelConfig(
            ship_model=ShipModel[block.ship_model] if block.ship_model else current_model.ship_model,
            dynamics_model=Model[block.dynamics_model] if block.dynamics_model else current_model.dynamics_model,
        ))

    if block.profile_overrides:
        merged = _deep_merge(profile.to_dict(), block.profile_overrides)
        merged["name"] = f"{profile.name}__{block.name}"
        profile = VesselProfile.from_dict(merged)

    profile.validate()
    return profile


# ua_surrogate_config keys that belong to one preset: not inherited from
# config.py when the block selects another preset.
_PRESET_SPECIFIC_UA_KEYS = ("checkpoint", "lookup_csv")


def merge_config_override(key: str, base: Any, override: Any) -> Any:
    """
    A block's config override applied to config.py's value for `key`.

    A dict (surrogate config) is merged over config.py's dict key by key, so
    unset keys such as wind_current keep config.py's value. For
    ua_surrogate_config, config.py's checkpoint and lookup_csv are dropped when
    the block selects another preset. Any other value replaces config.py's.
    """

    if not isinstance(override, dict):
        return override
    merged = dict(base) if isinstance(base, dict) else {}
    if key == "ua_surrogate_config" and override.get("preset", merged.get("preset")) != merged.get("preset"):
        for preset_key in _PRESET_SPECIFIC_UA_KEYS:
            merged.pop(preset_key, None)
    merged.update(override)
    return merged


def describe_block_config(block: PipelineBlock, framework_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    The BLOCK_CONFIG_KEYS entries of a resolved framework config that the
    block's environment and dynamics model read, for logging and the run record.
    """

    canonical_name = resolve_environment_name(block.env_type)
    dynamics_model = framework_config.get("dynamics_model")
    used = {}
    for key in BLOCK_CONFIG_KEYS:
        if key == "scenario":
            if canonical_name in SCENARIO_ENVS:
                used[key] = framework_config.get(key)
        elif dynamics_model == SURROGATE_CONFIG_MODEL[key] and (key != "ua_surrogate_config"
                                                               or canonical_name in UA_SURROGATE_ENVS):
            used[key] = framework_config.get(key)
    return used


def resolved_block_configs(blocks) -> Dict[str, Any]:
    """
    Per block name, describe_block_config of its resolved framework config, or
    {"error": ...} if the block cannot be resolved.
    """

    resolved: Dict[str, Any] = {}
    for block in blocks:
        try:
            framework_config = block_framework_config(block, resolve_block_profile(block))
            resolved[block.name] = describe_block_config(block, framework_config)
        except Exception as exc:
            resolved[block.name] = {"error": str(exc)}
    return resolved


def block_framework_config(block: PipelineBlock, profile: VesselProfile) -> Dict[str, Any]:
    """
    The block's framework configuration: config.py, the block's config
    overrides (merge_config_override), its wind/current flags, then the
    profile.
    """

    framework_config = load_config_parameters()
    for key, value in block.config_overrides.items():
        framework_config[key] = merge_config_override(key, framework_config.get(key), value)
    framework_config["wind"] = block.wind
    framework_config["current"] = block.current
    apply_profile(profile, config=framework_config)
    return framework_config


def check_block_wind_current(block: PipelineBlock, framework_config: Dict[str, Any]) -> None:
    """
    Wind/current combination that PySimEnv would only refuse when the block's
    environment is built: a full_env UAntwerp preset with the flags off (only
    for the PySimEnv-based environments, which use ua_surrogate_config). Any
    other problem (unknown environment or preset) is left to the block.

    Raises:
        ValueError: For that combination.
    """

    try:
        canonical_name = resolve_environment_name(block.env_type)
    except Exception:
        return
    if canonical_name not in UA_SURROGATE_ENVS:
        return
    ua_config = framework_config.get("ua_surrogate_config") or {}
    if framework_config.get("dynamics_model") == Model.UANTWERP and "preset" in ua_config and not block.wind:
        try:
            preset = get_preset(ua_config["preset"])
        except ValueError:
            return
        if preset.scenario.includes_wind_current:
            raise ValueError(f"Block '{block.name}': UAntwerp surrogate preset '{preset.name}' simulates wind and "
                             f"current, so wind and current must be enabled.")


def check_block_dynamics_config(block: PipelineBlock, framework_config: Dict[str, Any], model: ModelConfig) -> None:
    """
    Check the block's environment, dynamics model and surrogate configs before
    the environment is built: Model.UANTWERP on a COLAV environment (not
    supported), a block surrogate config for a dynamics model the block does
    not use, and for Model.UANTWERP the preset in effect (config.py merged with
    the block's override) against the block's ship. An unknown preset is left
    to the block (as in check_block_wind_current); run_loader checks the
    block's own preset name.

    Args:
        block: The block.
        framework_config: Its resolved framework config (block_framework_config).
        model: Its resolved ship and dynamics model (profile.model).

    Raises:
        ValueError: For any of these problems.
    """

    canonical_name = resolve_environment_name(block.env_type)
    if model.dynamics_model == Model.UANTWERP and canonical_name in COLAV_ENVS:
        raise ValueError(f"Block '{block.name}': dynamics model UANTWERP is not supported on {canonical_name}.")

    for key, dynamics_model in SURROGATE_CONFIG_MODEL.items():
        if key in block.config_overrides and model.dynamics_model != dynamics_model:
            raise ValueError(f"Block '{block.name}': '{key}' is only read by dynamics model {dynamics_model.name}, "
                             f"but this block uses {model.dynamics_model.name}.")

    if model.dynamics_model != Model.UANTWERP or canonical_name not in UA_SURROGATE_ENVS:
        return
    preset_name = (framework_config.get("ua_surrogate_config") or {}).get("preset")
    if preset_name is None:
        raise ValueError(f"Block '{block.name}': ua_surrogate_config has no 'preset' (block or config.py).")
    try:
        preset = get_preset(preset_name)
    except ValueError:
        return
    if preset.ship_model != model.ship_model:
        raise ValueError(f"Block '{block.name}': ua_surrogate_config preset '{preset.name}' is for "
                         f"{preset.ship_model.name}, but this block uses {model.ship_model.name}.")


def build_block_environment(block: PipelineBlock, framework_config: Dict[str, Any]):
    """
    Build one block environment and give it the block's unique training name.

    Raises:
        RuntimeError: If `block.env_type` resolves to AirSim (unsupported
            here) or to an environment with no registered direct builder.
    """

    canonical_name = resolve_environment_name(block.env_type)

    if canonical_name == airsim_env_name:
        raise RuntimeError(
            f"Block '{block.name}': AirSimEnv is not supported by the JSON "
            f"pipeline yet - AirSim needs fw.main_with_airsim's dedicated "
            f"UE5 connection/telemetry orchestration, which this pipeline "
            f"does not replicate. Use fw.main for AirSim training, or "
            f"exclude this block."
        )

    builder = _DIRECT_ENV_BUILDERS.get(canonical_name)
    if builder is None:
        raise RuntimeError(
            f"Block '{block.name}': no direct-instantiation builder registered "
            f"for '{canonical_name}' in pipeline_env._DIRECT_ENV_BUILDERS."
        )

    cfg = _CONFIG_BUILDERS[canonical_name](framework_config)

    env = builder(canonical_name, cfg)
    env.gym_env.training_name = block.name
    return env, canonical_name
