"""
Load a JSON pipeline specification and validate it against framework enums and registries.
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any, Dict, Set
from fw.ships.ship_factory import ShipModel
from fw.surrogate.presets import get_preset
from fw.dynamics.dynamics_model import Model
from fw.configuration.config import PolicyStrategy
from fw.configuration.config_profile import ProfileRegistry
from fw.common.tools.config_utils import get_validated_bool
from fw.common.tools.logging_config import resolve_log_level
from fw.configuration.stop_condition import Criterion, StopCondition
from fw.dynamics.uantwerp_surrogate import CONFIG_KEYS as UA_CONFIG_KEYS
from fw.dynamics.ugent_surrogate import CONFIG_KEYS as UGENT_CONFIG_KEYS
from fw.scheduling.run_spec import block_name_error, validate_block_names
from fw.dynamics.ugent_b01_surrogate import CONFIG_KEYS as UGENT_B01_CONFIG_KEYS
from fw.scheduling.run_spec import BLOCK_CONFIG_KEYS, COMMENT_KEY_PREFIX, REWARD_CONFIG_KEYS
from fw.scheduling.run_spec import PipelineBlock, PipelineSpec, StopConditionSpec, TransferSpec
from fw.configuration.env_factory import resolve_environment_name, EnvironmentRegistryError, SCENARIO_ENVS

logger = logging.getLogger(__name__)

_VALID_ACTIONS = {"train", "visualize", "evaluate_reward"}

# Accepted keys of each surrogate config (BLOCK_CONFIG_KEYS).
_SURROGATE_CONFIG_KEYS = {
    "ua_surrogate_config": frozenset(UA_CONFIG_KEYS),
    "ugent_surrogate_config": UGENT_CONFIG_KEYS,
    "ugent_b01_surrogate_config": UGENT_B01_CONFIG_KEYS,
}

_KNOWN_SPEC_KEYS = {"on_block_failure", "log_level", "blocks"}
_KNOWN_BLOCK_KEYS = {
    "name", "env_type", "profile", "ship_model", "dynamics_model", "profile_overrides", "action",
    "reward_configs", "stop_condition", "transfer", "on_failure", "wind", "current", "show_plots",
    "hyperparameter_overrides", *BLOCK_CONFIG_KEYS,
}
_KNOWN_STOP_CONDITION_KEYS = {"criterion", "max_time_steps", "target_reward", "target_loss", "patience"}
_KNOWN_TRANSFER_KEYS = {"policy_strategy", "from_block", "external_training_name", "independent"}
_VALID_FAILURE_MODES = {"continue", "abort"}

# Rejected up front because they would fail later during execution.
_UNIMPLEMENTED_CRITERIA = {"AUTO_CONVERGE"}
_UNSUPPORTED_POLICY_STRATEGIES = {"REUSE_CURRENT_POLICY"}


class PipelineSpecError(ValueError):
    """
    Raised when a pipeline specification fails validation.
    """


def load_pipeline_spec(path: str) -> PipelineSpec:
    """
    Read and validate a pipeline specification file.

    Args:
        path: JSON file shaped like {"on_block_failure": "continue", "blocks": [{...}, ...]}.

    Returns:
        A validated `PipelineSpec`.

    Raises:
        PipelineSpecError: Missing required fields, or a value that is not a known
            enum member or registered name.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _parse_spec(raw, source=str(path))


def _parse_spec(raw: Dict[str, Any], source: str) -> PipelineSpec:
    """
    Parse the top-level specification (`source` is only used in error messages).

    Raises:
        PipelineSpecError: On any invalid field, or a `transfer.from_block` that does
            not name an earlier 'train' block.
    """

    if not isinstance(raw, dict):
        raise PipelineSpecError(f"[{source}] The specification must be a JSON object, got {type(raw).__name__}.")
    _reject_unknown_keys(raw, _KNOWN_SPEC_KEYS, f"[{source}]", "the specification")

    on_block_failure = raw.get("on_block_failure", "continue")
    _validate_choice("on_block_failure", on_block_failure, _VALID_FAILURE_MODES, f"[{source}]")

    log_level = raw.get("log_level", "INFO")
    try:
        resolve_log_level(log_level, where=f"[{source}] 'log_level'")
    except ValueError as exc:
        raise PipelineSpecError(str(exc)) from exc

    raw_blocks = raw.get("blocks")
    if not raw_blocks or not isinstance(raw_blocks, list):
        raise PipelineSpecError(f"[{source}] 'blocks' must be a non-empty list.")

    seen_names: Set[str] = set()
    blocks = []
    for i, raw_block in enumerate(raw_blocks):
        block = _parse_block(raw_block, index=i, seen_names=seen_names, source=source)
        seen_names.add(block.name.casefold())
        blocks.append(block)

    try:
        validate_block_names(blocks)
    except ValueError as exc:
        raise PipelineSpecError(f"[{source}] {exc}") from exc

    for i, block in enumerate(blocks):
        ref = block.transfer.from_block
        if ref is None:
            continue
        earlier_blocks_by_name = {b.name: b for b in blocks[:i]}
        if ref not in earlier_blocks_by_name:
            raise PipelineSpecError(
                f"[{source}] Block '{block.name}' (position {i}) has "
                f"transfer.from_block='{ref}', which must name an earlier "
                f"block in the same file. Earlier blocks: "
                f"{sorted(earlier_blocks_by_name) or '(none)'}."
            )
        ref_action = earlier_blocks_by_name[ref].action
        if ref_action != "train":
            raise PipelineSpecError(
                f"[{source}] Block '{block.name}' (position {i}) has "
                f"transfer.from_block='{ref}', but '{ref}' has "
                f"action='{ref_action}', not 'train' - only 'train' blocks "
                f"save a policy checkpoint, so there is nothing for "
                f"'{block.name}' to reuse from it."
            )

    return PipelineSpec(blocks=blocks, on_block_failure=on_block_failure, log_level=log_level)


def _parse_block(raw_block: Dict[str, Any], index: int, seen_names: Set[str], source: str) -> PipelineBlock:
    """
    Parse one entry of `blocks`.

    Args:
        raw_block: The raw block dict.
        index: Position in `blocks` (for error messages).
        seen_names: Case-folded names of earlier blocks, used to reject duplicates.
        source: Path string for error messages.

    Raises:
        PipelineSpecError: On any missing or invalid field.
    """

    where = f"[{source}] block[{index}]"

    if not isinstance(raw_block, dict):
        raise PipelineSpecError(f"{where}: a block must be a JSON object, got {type(raw_block).__name__}.")
    # A misspelt or unsupported key would otherwise be dropped silently.
    _reject_unknown_keys(raw_block, _KNOWN_BLOCK_KEYS, where, f"block '{raw_block.get('name')}'")

    name = raw_block.get("name")
    if not name or not isinstance(name, str):
        raise PipelineSpecError(f"{where}: 'name' is required and must be a non-empty string.")
    if name.casefold() in seen_names:
        raise PipelineSpecError(f"{where}: duplicate block name '{name}'. Block names must be unique, "
                                f"ignoring case.")
    name_error = block_name_error(name)
    if name_error:
        raise PipelineSpecError(f"{where}: block name {name!r} is used as a file and folder name, so {name_error}.")

    env_type = raw_block.get("env_type")
    if not env_type:
        raise PipelineSpecError(f"{where} ('{name}'): 'env_type' is required.")
    try:
        resolve_environment_name(env_type)
    except EnvironmentRegistryError as exc:
        raise PipelineSpecError(f"{where} ('{name}'): {exc}") from exc

    profile_name = raw_block.get("profile", "default_profile")
    try:
        ProfileRegistry.get(profile_name)
    except KeyError as exc:
        raise PipelineSpecError(f"{where} ('{name}'): {exc}") from exc

    ship_model = raw_block.get("ship_model")
    if ship_model is not None:
        _validate_enum_member("ship_model", ship_model, ShipModel, where, name)

    dynamics_model = raw_block.get("dynamics_model")
    if dynamics_model is not None:
        _validate_enum_member("dynamics_model", dynamics_model, Model, where, name)

    action = raw_block.get("action", "train")
    _validate_choice("action", action, _VALID_ACTIONS, f"{where} ('{name}')")
    reward_configs = raw_block.get("reward_configs")
    if action == "evaluate_reward" and not reward_configs:
        raise PipelineSpecError(
            f"{where} ('{name}'): action='evaluate_reward' requires a "
            f"non-empty 'reward_configs' list."
        )
    if action != "evaluate_reward" and reward_configs:
        raise PipelineSpecError(
            f"{where} ('{name}'): 'reward_configs' is only valid when "
            f"action='evaluate_reward' (got action='{action}'). Likely a "
            f"leftover from changing this block's action."
        )
    if reward_configs:
        _check_reward_configs(reward_configs, env_type, where, name)

    raw_stop_condition = _get_dict(raw_block, "stop_condition", where, name)
    _reject_unknown_keys(raw_stop_condition, _KNOWN_STOP_CONDITION_KEYS, f"{where} ('{name}')", "stop_condition")
    stop_condition = _parse_stop_condition(raw_stop_condition, where, name)
    raw_transfer = _get_dict(raw_block, "transfer", where, name)
    _reject_unknown_keys(raw_transfer, _KNOWN_TRANSFER_KEYS, f"{where} ('{name}')", "transfer")
    transfer = _parse_transfer(raw_transfer, action, where, name)

    on_failure = raw_block.get("on_failure")
    if on_failure is not None:
        _validate_choice("on_failure", on_failure, _VALID_FAILURE_MODES, f"{where} ('{name}')")

    # Checked here, not hours later when PySimEnv is built.
    wind = get_validated_bool(raw_block, "wind", False)
    current = get_validated_bool(raw_block, "current", False)
    if wind != current:
        raise PipelineSpecError(f"{where} ('{name}'): 'wind' ({wind}) and 'current' ({current}) must be "
                                f"enabled or disabled together.")

    config_overrides = _parse_config_overrides(raw_block, where, name)

    block = PipelineBlock(
        name=name,
        env_type=env_type,
        profile=profile_name,
        ship_model=ship_model,
        dynamics_model=dynamics_model,
        profile_overrides=_get_dict(raw_block, "profile_overrides", where, name),
        action=action,
        wind=wind,
        current=current,
        show_plots=get_validated_bool(raw_block, "show_plots", False),
        stop_condition=stop_condition,
        hyperparameter_overrides=_get_dict(raw_block, "hyperparameter_overrides", where, name),
        reward_configs=reward_configs,
        transfer=transfer,
        on_failure=on_failure,
        config_overrides=config_overrides,
    )
    _check_block_config(block, where)
    return block


def _check_reward_configs(reward_configs: Any, env_type: str, where: str, name: str) -> None:
    """
    Check an evaluate_reward block's reward_configs: a list of objects with
    only REWARD_CONFIG_KEYS, each with a "type" resolving to the block's own
    environment (the framework config and profile are the block's).

    Raises:
        PipelineSpecError: For anything else.
    """

    if not isinstance(reward_configs, list):
        raise PipelineSpecError(f"{where} ('{name}'): 'reward_configs' must be a list, "
                                f"got {type(reward_configs).__name__}.")
    block_env = resolve_environment_name(env_type)
    for i, entry in enumerate(reward_configs):
        entry_where = f"{where} ('{name}') reward_configs[{i}]"
        if not isinstance(entry, dict):
            raise PipelineSpecError(f"{entry_where}: must be an object, got {type(entry).__name__}.")
        _reject_unknown_keys(entry, REWARD_CONFIG_KEYS, entry_where, "the reward config")
        if "type" not in entry:
            raise PipelineSpecError(f"{entry_where}: 'type' is required.")
        try:
            entry_env = resolve_environment_name(entry["type"])
        except EnvironmentRegistryError as exc:
            raise PipelineSpecError(f"{entry_where}: {exc}") from exc
        if entry_env != block_env:
            raise PipelineSpecError(f"{entry_where}: 'type' resolves to {entry_env}, but the block's env_type "
                                    f"resolves to {block_env}; they must match.")
        weights = entry.get("reward_weights")
        if weights is not None and not isinstance(weights, dict):
            raise PipelineSpecError(f"{entry_where}: 'reward_weights' must be an object, "
                                    f"got {type(weights).__name__}.")


def _parse_config_overrides(raw_block: Dict[str, Any], where: str, name: str) -> Dict[str, Any]:
    """
    Collect the block's BLOCK_CONFIG_KEYS entries, without comment keys
    (checked against the resolved block by _check_block_config).

    Raises:
        PipelineSpecError: A non-string scenario, a non-dict surrogate config or
            an unknown key in a surrogate config.
    """

    overrides: Dict[str, Any] = {}
    for key in BLOCK_CONFIG_KEYS:
        if key not in raw_block:
            continue
        value = raw_block[key]
        if key == "scenario":
            if not isinstance(value, str) or not value:
                raise PipelineSpecError(f"{where} ('{name}'): 'scenario' must be a non-empty string.")
            overrides[key] = value
            continue
        if not isinstance(value, dict):
            raise PipelineSpecError(f"{where} ('{name}'): '{key}' must be an object, got {type(value).__name__}.")
        _reject_unknown_keys(value, _SURROGATE_CONFIG_KEYS[key], f"{where} ('{name}')", key)
        overrides[key] = {k: v for k, v in value.items() if not str(k).startswith(COMMENT_KEY_PREFIX)}
    return overrides


def _check_block_config(block: PipelineBlock, where: str) -> None:
    """
    Check the block against its resolved configuration
    (pipeline_env.check_block_dynamics_config, for every block): UANTWERP on a
    COLAV env, a surrogate config for another dynamics model, and the UAntwerp
    preset in effect against the block's ship, plus the name of a preset given
    by the block (a config.py typo is left to the block, as in _preflight). A
    scenario on an environment that does not read it is logged as a warning.
    If the block's profile cannot be resolved, the rest is left to the block
    itself (as PipelineScheduler._preflight does).

    Raises:
        PipelineSpecError: For any of these problems.
    """

    # Imported here: pipeline_env imports the environment factories.
    from fw.scheduling import pipeline_env

    if "scenario" in block.config_overrides and resolve_environment_name(block.env_type) not in SCENARIO_ENVS:
        logger.warning("%s ('%s'): 'scenario' has no effect on env_type '%s'; only %s read it.",
                       where, block.name, block.env_type, sorted(SCENARIO_ENVS))

    preset_name = block.config_overrides.get("ua_surrogate_config", {}).get("preset")
    if preset_name is not None:
        try:
            get_preset(preset_name)
        except ValueError as exc:
            raise PipelineSpecError(f"{where} ('{block.name}'): ua_surrogate_config: {exc}") from exc

    try:
        profile = pipeline_env.resolve_block_profile(block)
        framework_config = pipeline_env.block_framework_config(block, profile)
    except Exception:
        return

    try:
        pipeline_env.check_block_dynamics_config(block, framework_config, profile.model)
    except ValueError as exc:
        raise PipelineSpecError(f"{where}: {exc}") from exc


def _get_dict(raw_block: Dict[str, Any], key: str, where: str, name: str) -> Dict[str, Any]:
    """
    The block's `key` entry as a dict ({} when absent or null).

    Raises:
        PipelineSpecError: If the entry is present but not an object.
    """

    value = raw_block.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PipelineSpecError(f"{where} ('{name}'): '{key}' must be an object, got {type(value).__name__}.")
    return value


def _reject_unknown_keys(raw: Dict[str, Any], known: Set[str], where: str, what: str) -> None:
    """
    Raise PipelineSpecError if `raw` has keys outside `known`, ignoring comment
    keys (COMMENT_KEY_PREFIX); `what` names the object in the message.
    """

    unknown = sorted(k for k in set(raw) - set(known) if not str(k).startswith(COMMENT_KEY_PREFIX))
    if unknown:
        raise PipelineSpecError(f"{where}: unknown key(s) {unknown} in {what}; accepted: {sorted(known)} "
                                f"(keys starting with '{COMMENT_KEY_PREFIX}' are comments).")


def _parse_stop_condition(raw: Dict[str, Any], where: str, name: str) -> StopConditionSpec:
    """
    Parse a block's (possibly empty) `stop_condition` dict.

    Raises:
        PipelineSpecError: Unknown or unimplemented criterion, or parameters that
            StopCondition rejects.
    """

    criterion = raw.get("criterion", "NBR_POLICY_UPDATES")
    _validate_enum_member("stop_condition.criterion", criterion, Criterion, where, name)
    if criterion in _UNIMPLEMENTED_CRITERIA:
        raise PipelineSpecError(
            f"{where} ('{name}'): stop_condition.criterion '{criterion}' is "
            f"not implemented by StopCondition yet (see "
            f"fw.configuration.stop_condition.Criterion) and cannot be used here."
        )
    spec = StopConditionSpec(
        criterion=criterion,
        max_time_steps=int(raw.get("max_time_steps", 2000)),
        target_reward=raw.get("target_reward"),
        target_loss=raw.get("target_loss"),
        patience=raw.get("patience"),
    )

    # Build the StopCondition now, as PipelineScheduler will, so a missing
    # target_reward/target_loss fails at load time rather than hours into the run.
    try:
        StopCondition(
            stop_criterion=Criterion[spec.criterion],
            max_time_steps=spec.max_time_steps,
            target_reward=spec.target_reward,
            target_loss=spec.target_loss,
            patience=spec.patience,
        )
    except (ValueError, TypeError, NotImplementedError) as exc:
        raise PipelineSpecError(f"{where} ('{name}'): invalid stop_condition: {exc}") from exc

    return spec


def _parse_transfer(raw: Dict[str, Any], action: str, where: str, name: str) -> TransferSpec:
    """
    Parse a block's (possibly empty) `transfer` dict.

    Args:
        raw: The raw `transfer` dict.
        action: The block's validated action; `independent` is only allowed on "train".
        where: Location string for error messages.
        name: Block name for error messages.

    Raises:
        PipelineSpecError: Unknown/unsupported policy_strategy; both from_block and
            external_training_name set; `independent` combined with another
            transfer field or used on a non-"train" block.
    """

    policy_strategy = raw.get("policy_strategy")
    if policy_strategy is not None:
        _validate_enum_member("transfer.policy_strategy", policy_strategy, PolicyStrategy, where, name)
        if policy_strategy in _UNSUPPORTED_POLICY_STRATEGIES:
            raise PipelineSpecError(
                f"{where} ('{name}'): transfer.policy_strategy "
                f"'{policy_strategy}' is not supported by pipeline blocks - "
                f"AgentManager.set_environment saves-then-reloads a policy "
                f"file for every transfer strategy, so REUSE_CURRENT_POLICY "
                f"would behave identically to REUSE_OTHER_POLICY pointed at "
                f"this block's own env, just under a more confusing name. "
                f"Use 'REUSE_OTHER_POLICY' with an explicit 'from_block' or "
                f"'external_training_name' instead."
            )

    from_block = raw.get("from_block")
    external_training_name = raw.get("external_training_name")
    if from_block and external_training_name:
        raise PipelineSpecError(
            f"{where} ('{name}'): transfer specifies both 'from_block' and "
            f"'external_training_name' - use only one."
        )

    independent = raw.get("independent", False)
    if not isinstance(independent, bool):
        raise PipelineSpecError(
            f"{where} ('{name}'): transfer.independent must be a boolean."
        )
    if independent and (policy_strategy or from_block or external_training_name):
        raise PipelineSpecError(
            f"{where} ('{name}'): transfer.independent cannot be combined "
            f"with 'policy_strategy', 'from_block', or "
            f"'external_training_name' - set only 'independent': true, or "
            f"remove it and use one of the other fields instead."
        )
    if independent and action != "train":
        raise PipelineSpecError(
            f"{where} ('{name}'): transfer.independent is only valid on "
            f"'train' blocks (got action='{action}'). It exists to opt a "
            f"'train' block out of the scheduler's implicit "
            f"previous-train-block chaining default, which never applies "
            f"to '{action}' blocks in the first place - remove "
            f"'independent' here."
        )

    return TransferSpec(
        policy_strategy=policy_strategy,
        from_block=from_block,
        external_training_name=external_training_name,
        independent=independent,
    )


def _validate_choice(field_name: str, value: Any, valid: Set[str], where: str) -> None:
    """
    Raise PipelineSpecError if `value` is not in `valid`.
    """

    if value not in valid:
        raise PipelineSpecError(
            f"{where}: '{field_name}' must be one of {sorted(valid)}, got '{value}'."
        )


def _validate_enum_member(field_name: str, value: str, enum_cls, where: str, block_name: str) -> None:
    """
    Raise PipelineSpecError if `value` is not a member name of `enum_cls`.
    """

    valid_names = {member.name for member in enum_cls}
    if value not in valid_names:
        raise PipelineSpecError(
            f"{where} ('{block_name}'): '{field_name}' = '{value}' is not a "
            f"valid {enum_cls.__name__}. Valid options: {sorted(valid_names)}."
        )
