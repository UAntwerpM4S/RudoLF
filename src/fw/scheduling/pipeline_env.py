"""
Build pipeline environments without the factory cache and resolve block profiles.
"""

from __future__ import annotations

from typing import Any, Dict
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.scheduling.run_spec import PipelineBlock
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
