"""
Load and validate a JSON pipeline specification against framework enums and registries.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict, Set
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.configuration.config import PolicyStrategy
from fw.configuration.stop_condition import Criterion, StopCondition
from fw.configuration.config_profile import ProfileRegistry
from fw.common.tools.config_utils import get_validated_bool
from fw.common.tools.logging_config import resolve_log_level
from fw.configuration.env_factory import resolve_environment_name, EnvironmentRegistryError
from fw.scheduling.run_spec import PipelineBlock, PipelineSpec, StopConditionSpec, TransferSpec

_VALID_ACTIONS = {"train", "visualize", "evaluate_reward"}
_VALID_FAILURE_MODES = {"continue", "abort"}

# Reject criteria known to fail later during execution.
_UNIMPLEMENTED_CRITERIA = {"AUTO_CONVERGE"}
_UNSUPPORTED_POLICY_STRATEGIES = {"REUSE_CURRENT_POLICY"}


class PipelineSpecError(ValueError):
    """ Raised when a pipeline specification fails validation. """


def load_pipeline_spec(path: str) -> PipelineSpec:
    """
    Read and validate a pipeline specification file.

    Args:
        path: Path to a JSON file shaped like:
            {"on_block_failure": "continue", "blocks": [ {...}, {...} ]}

    Returns:
        A validated `PipelineSpec`.

    Raises:
        PipelineSpecError: If the file is missing required fields, or any
            field does not match a known enum member / registered name.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _parse_spec(raw, source=str(path))


def _parse_spec(raw: Dict[str, Any], source: str) -> PipelineSpec:
    """
    Validate and parse the top-level pipeline specification.

    Args:
        raw: Parsed JSON content of the run file.
        source: Path string used only for error messages.

    Returns:
        A validated `PipelineSpec`.

    Raises:
        PipelineSpecError: On any invalid field or forward/self
            `transfer.from_block` reference.
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
        seen_names.add(block.name)
        blocks.append(block)

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
    Validate and parse one pipeline block.

    Args:
        raw_block: One entry from the run file's `blocks` list.
        index: Position of this block in `blocks`, used in error messages.
        seen_names: Names of blocks already parsed earlier in this file,
            used to reject duplicates.
        source: Path string used only for error messages.

    Returns:
        A validated `PipelineBlock`.

    Raises:
        PipelineSpecError: On any missing or invalid field.
    """

    where = f"[{source}] block[{index}]"

    name = raw_block.get("name")
    if not name or not isinstance(name, str):
        raise PipelineSpecError(f"{where}: 'name' is required and must be a non-empty string.")
    if name in seen_names:
        raise PipelineSpecError(f"{where}: duplicate block name '{name}'. Block names must be unique.")

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

    return PipelineBlock(
        name=name,
        env_type=env_type,
        profile=profile_name,
        ship_model=ship_model,
        dynamics_model=dynamics_model,
        profile_overrides=raw_block.get("profile_overrides", {}) or {},
        action=action,
        wind=get_validated_bool(raw_block, "wind", False),
        current=get_validated_bool(raw_block, "current", False),
        show_plots=get_validated_bool(raw_block, "show_plots", False),
        stop_condition=stop_condition,
        hyperparameter_overrides=raw_block.get("hyperparameter_overrides", {}) or {},
        reward_configs=reward_configs,
        transfer=transfer,
        on_failure=on_failure,
    )


def _parse_stop_condition(raw: Dict[str, Any], where: str, name: str) -> StopConditionSpec:
    """
    Validate and parse a stop-condition specification.

    Args:
        raw: The block's raw `stop_condition` dict (can be empty).
        where: Location string used in error messages.
        name: The owning block's name, used in error messages.

    Returns:
        A validated `StopConditionSpec`.

    Raises:
        PipelineSpecError: If `criterion` is unknown or unimplemented.
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

    # Build the StopCondition once at load time, exactly as PipelineScheduler
    # does when the block starts, so a missing target_reward / target_loss is
    # reported now instead of after hours of earlier blocks.
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
    Validate and parse a transfer specification.

    Args:
        raw: The block's raw `transfer` dict (can be empty).
        action: The owning block's already-validated `action` (one of
            `_VALID_ACTIONS`), used to reject `independent` on any action
            other than "train".
        where: Location string used in error messages.
        name: The owning block's name, used in error messages.

    Returns:
        A validated `TransferSpec`.

    Raises:
        PipelineSpecError: If `policy_strategy` is unknown/unsupported,
            both `from_block` and `external_training_name` are set,
            `independent` is combined with any other transfer field, or
            `independent` is set on a non-"train" block.
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
    Raise if value is not one of the allowed choices.

    Args:
        field_name: Name of the field being validated, used in the
            error message.
        value: The value to check.
        valid: Set of acceptable values.
        where: Location string used in the error message.

    Raises:
        PipelineSpecError: If `value` is not in `valid`.
    """

    if value not in valid:
        raise PipelineSpecError(
            f"{where}: '{field_name}' must be one of {sorted(valid)}, got '{value}'."
        )


def _validate_enum_member(field_name: str, value: str, enum_cls, where: str, block_name: str) -> None:
    """
    Raise if value does not name a member of enum_cls.

    Args:
        field_name: Name of the field being validated, used in the
            error message.
        value: The raw string to check against `enum_cls`'s member names.
        enum_cls: The enum class `value` must name a member of.
        where: Location string used in the error message.
        block_name: The owning block's name, used in the error message.

    Raises:
        PipelineSpecError: If `value` does not match any member name of
            `enum_cls`.
    """

    valid_names = {member.name for member in enum_cls}
    if value not in valid_names:
        raise PipelineSpecError(
            f"{where} ('{block_name}'): '{field_name}' = '{value}' is not a "
            f"valid {enum_cls.__name__}. Valid options: {sorted(valid_names)}."
        )
