"""
pipeline_env.py - Cache-bypassing environment construction and profile
resolution for a JSON-driven pipeline run (see run_spec.py / run_loader.py /
main_pipeline.py).

Why this module exists
-----------------------
`fw.env_factory._Factory` caches at most ONE instance per canonical
environment name (e.g. "PySimEnv") and raises `EnvironmentRegistryError` if
a second request for that same name comes in with a different
configuration (see `_Factory.get`). A pipeline can legitimately want two
blocks with the same `env_type` but a different ship/dynamics model or
scenario, so `build_default_environment` (which goes through that cache)
is not safe to call more than once per env_type across a run.

This module instantiates each block's environment wrapper class directly
instead - the same technique `Trainer._build_eval_environment` already uses
for reward evaluation, for exactly the same reason.

It also overrides the freshly-built environment's `training_name` from the
canonical env name (the factory's default) to the pipeline block's own
unique `name`. Without this, two blocks sharing an `env_type` would still
collide on disk: `SimpleScheduler`/`AgentManager` key saved policy
checkpoints and profile JSON files off `training_name`
(`{training_name}_{agent_type}_policy.zip`), which `env_factory` stamps as
the canonical env name by default - so two "PySimEnv" blocks would silently
overwrite each other's checkpoint. Renaming it to the block's own name
makes every block's checkpoint independently addressable, and makes
`transfer.from_block` / `transfer.external_training_name` resolution simply
equal to the block name in question - no separate bookkeeping dict needed.

NOTE: agent-type selection (see `main_pipeline.py`'s `_select_agent_type`)
must use the *canonical* env name / `include_scenario_one_hot`, not
`training_name`, since `training_name` no longer matches constants like
`multi_sim_env_name` after this override.
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

# Direct-instantiation builders, bypassing the `_Factory` singleton cache.
# AirSim is intentionally excluded here - see main_pipeline.py's explicit
# rejection of AirSim blocks and its docstring for why.
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
    Recursively merge *overrides* onto a copy of *base*.
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
    Resolve the fully-configured `VesselProfile` for one pipeline block.

    Applies, in order: the registered base profile, any `ship_model` /
    `dynamics_model` override, then any free-form `profile_overrides`
    (deep-merged onto the profile's `to_dict()` and round-tripped through
    `VesselProfile.from_dict()`).
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
        # Keep the profile name unique per block so it doesn't silently
        # collide with the registered preset's own name in logs/errors.
        merged["name"] = f"{profile.name}__{block.name}"
        profile = VesselProfile.from_dict(merged)

    profile.validate()
    return profile


def build_block_environment(block: PipelineBlock, framework_config: Dict[str, Any]):
    """
    Instantiate this block's environment, bypassing the factory's cache,
    and stamp its `training_name` to the block's own unique name (see
    module docstring for why).

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

    if canonical_name == lunar_env_name:
        cfg = {}
    else:
        cfg = _CONFIG_BUILDERS[canonical_name](framework_config)

    env = builder(canonical_name, cfg)
    env.gym_env.training_name = block.name
    return env, canonical_name
