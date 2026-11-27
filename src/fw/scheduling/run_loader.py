"""
Load a JSON pipeline specification and validate it against framework enums and registries.
"""

from __future__ import annotations

import json

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
from fw.configuration.env_factory import resolve_environment_name, EnvironmentRegistryError
from fw.scheduling.run_spec import PipelineBlock, PipelineSpec, StopConditionSpec, TransferSpec
from fw.scheduling.run_spec import block_name_error, validate_block_names

_VALID_ACTIONS = {"train", "visualize", "evaluate_reward"}

# Block keys that replace the config.py entry of the same name for that block.
# The surrogate configs replace the config.py dict as a whole (no key merge), so
# a block's preset never inherits a checkpoint/lookup_csv meant for another one.
BLOCK_CONFIG_KEYS = ("scenario", "ua_surrogate_config", "ugent_surrogate_config", "ugent_b01_surrogate_config")

_KNOWN_BLOCK_KEYS = {
    "name", "env_type", "profile", "ship_model", "dynamics_model", "profile_overrides", "action",
    "reward_configs", "stop_condition", "transfer", "on_failure", "wind", "current", "show_plots",
    "hyperparameter_overrides", *BLOCK_CONFIG_KEYS,
}
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

    on_block_failure = raw.get("on_block_failure", "continue")
    _validate_choice("on_block_failure", on_block_failure, _VALID_FAILURE_MODES, f"[{source}]")

    log_level = raw.get("log_level", "INFO")
    try:
        resolve_log_level(log_level, where=f"[{source}] 'log_level'")
    except ValueError as exc:
        raise PipelineSpecError(str(exc)) from exc

    raw_blocks = raw.get("blocks")
    if not raw_blocks:
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

    # A misspelt or unsupported key would otherwise be dropped silently.
    unknown_keys = sorted(set(raw_block) - _KNOWN_BLOCK_KEYS)
    if unknown_keys:
        raise PipelineSpecError(f"{where} ('{raw_block.get('name')}'): unknown key(s) {unknown_keys}; "
                                f"accepted: {sorted(_KNOWN_BLOCK_KEYS)}.")

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

    stop_condition = _parse_stop_condition(raw_block.get("stop_condition", {}), where, name)
    transfer = _parse_transfer(raw_block.get("transfer", {}), action, where, name)

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

    return PipelineBlock(
        name=name,
        env_type=env_type,
        profile=profile_name,
        ship_model=ship_model,
        dynamics_model=dynamics_model,
        profile_overrides=raw_block.get("profile_overrides", {}) or {},
        action=action,
        wind=wind,
        current=current,
        show_plots=get_validated_bool(raw_block, "show_plots", False),
        stop_condition=stop_condition,
        hyperparameter_overrides=raw_block.get("hyperparameter_overrides", {}) or {},
        reward_configs=reward_configs,
        transfer=transfer,
        on_failure=on_failure,
        config_overrides=config_overrides,
    )


def _parse_config_overrides(raw_block: Dict[str, Any], where: str, name: str) -> Dict[str, Any]:
    """
    Collect the block's BLOCK_CONFIG_KEYS entries.

    Raises:
        PipelineSpecError: A non-string scenario, a non-dict surrogate config,
            or an unknown UAntwerp preset.
    """

    overrides: Dict[str, Any] = {}
    for key in BLOCK_CONFIG_KEYS:
        if key not in raw_block:
            continue
        value = raw_block[key]
        if key == "scenario":
            if not isinstance(value, str) or not value:
                raise PipelineSpecError(f"{where} ('{name}'): 'scenario' must be a non-empty string.")
        elif not isinstance(value, dict):
            raise PipelineSpecError(f"{where} ('{name}'): '{key}' must be an object, got {type(value).__name__}.")
        overrides[key] = dict(value) if isinstance(value, dict) else value

    ua_config = overrides.get("ua_surrogate_config")
    if ua_config is not None and "preset" in ua_config:
        try:
            get_preset(ua_config["preset"])
        except ValueError as exc:
            raise PipelineSpecError(f"{where} ('{name}'): ua_surrogate_config: {exc}") from exc
    return overrides


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
