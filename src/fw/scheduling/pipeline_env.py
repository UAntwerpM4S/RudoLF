"""
Build pipeline environments without the factory cache and resolve block profiles.
"""

from __future__ import annotations

from typing import Any, Dict
from fw.surrogate.presets import get_preset
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.scheduling.run_spec import PipelineBlock
from fw.configuration.config import load_config_parameters
from fw.configuration.profile_applier import apply_profile
from fw.configuration.config_profile import ModelConfig, ProfileRegistry, VesselProfile

from fw.configuration.env_factory import (
    resolve_environment_name,
    py_sim_env_name,
    multi_sim_env_name,
    fh_sim_env_name,
    sync_sim_env_name,
    airsim_env_name,
    lunar_env_name,
    py_sim_colav_env_name,
    multi_policy_colav_env_name,
    PySimEnvironment,
    MultiPySimEnvironment,
    FhSimEnvironment,
    FhSimSyncEnvironment,
    LunarEnvironment,
    PySimColavEnvironment,
    MultiPolicyColavEnvironment,
    build_pysim_config,
    build_fhsim_config,
    build_fhsimsync_config,
    build_colav_config,
    build_multi_policy_colav_config,
)


# Instantiated directly: the standard env factory caches by canonical name.
_DIRECT_ENV_BUILDERS = {
    py_sim_env_name: PySimEnvironment,
    multi_sim_env_name: MultiPySimEnvironment,
    fh_sim_env_name: FhSimEnvironment,
    sync_sim_env_name: FhSimSyncEnvironment,
    lunar_env_name: LunarEnvironment,
    py_sim_colav_env_name: PySimColavEnvironment,
    multi_policy_colav_env_name: MultiPolicyColavEnvironment,
}

_CONFIG_BUILDERS = {
    py_sim_env_name: build_pysim_config,
    multi_sim_env_name: build_pysim_config,
    fh_sim_env_name: build_fhsim_config,
    sync_sim_env_name: build_fhsimsync_config,
    py_sim_colav_env_name: build_colav_config,
    multi_policy_colav_env_name: build_multi_policy_colav_config,
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


# Environments built from the framework config's ua_surrogate_config (PySimEnv based).
_UA_CONFIG_ENVS = frozenset({py_sim_env_name, multi_sim_env_name, fh_sim_env_name, sync_sim_env_name})


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


def describe_block_config(framework_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    The block-relevant part of a resolved framework config (scenario and the
    surrogate configs), for logging and the run record.
    """

    return {key: framework_config.get(key) for key in
            ("scenario", "ua_surrogate_config", "ugent_surrogate_config", "ugent_b01_surrogate_config")}


def resolved_block_configs(blocks) -> Dict[str, Any]:
    """
    Per block name, describe_block_config of its resolved framework config, or
    {"error": ...} if the block cannot be resolved.
    """

    resolved: Dict[str, Any] = {}
    for block in blocks:
        try:
            resolved[block.name] = describe_block_config(block_framework_config(block, resolve_block_profile(block)))
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
    if canonical_name not in _UA_CONFIG_ENVS:
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

    cfg = {} if canonical_name == lunar_env_name else _CONFIG_BUILDERS[canonical_name](framework_config)

    env = builder(canonical_name, cfg)
    env.gym_env.training_name = block.name
    return env, canonical_name
