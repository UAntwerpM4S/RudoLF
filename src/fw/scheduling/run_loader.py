"""
run_loader.py - Load and validate a JSON pipeline-run file into a PipelineSpec.

Every field is checked against the framework's *actual* enums and registries
(`ShipModel`, `Model`, `ProfileRegistry`, `PolicyStrategy`, `Criterion`,
`env_factory`'s alias table) at load time, so a typo in the run file fails
immediately with a clear message rather than surfacing as a confusing error
deep inside training.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict, Set
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.configuration.config import PolicyStrategy
from fw.configuration.stop_condition import Criterion
from fw.configuration.config_profile import ProfileRegistry
from fw.common.tools.config_utils import get_validated_bool
from fw.configuration.env_factory import resolve_environment_name, EnvironmentRegistryError
from fw.scheduling.run_spec import PipelineBlock, PipelineSpec, StopConditionSpec, TransferSpec
from fw.common.tools.logging_config import resolve_log_level

_VALID_ACTIONS = {"train", "visualize", "evaluate_reward"}
_VALID_FAILURE_MODES = {"continue", "abort"}

# StopCondition.__init__ raises NotImplementedError for this immediately -
# but only when that block's StopCondition is actually built, at
# *execution* time, potentially after earlier blocks already ran for a
# long time. Reject it here instead, at *load* time, for the whole plan.
_UNIMPLEMENTED_CRITERIA = {"AUTO_CONVERGE"}

# PipelineScheduler reuses ONE shared AgentManager across every block in a
# run (see pipeline_scheduler.py's module docstring) - it does NOT build a
# fresh Trainer/AgentManager per block, so there *is* an in-memory "current"
# agent carried across blocks. That's not why REUSE_CURRENT_POLICY is
# rejected here.
#
# It's rejected because AgentManager.set_environment saves-then-reloads a
# policy *file* for BOTH REUSE_CURRENT_POLICY and REUSE_OTHER_POLICY (see
# its source) - it never keeps the in-memory agent object across the
# set_environment() call regardless of strategy. So REUSE_CURRENT_POLICY
# would, in practice, behave exactly like REUSE_OTHER_POLICY pointed at the
# current block's own env, just via a more confusing name. Rejected here
# instead of silently accepted; see main_pipeline.py's docstring for the
# same rationale.
_UNSUPPORTED_POLICY_STRATEGIES = {"REUSE_CURRENT_POLICY"}


class PipelineSpecError(ValueError):
    """Raised when a pipeline run file fails validation."""


def load_pipeline_spec(path: str) -> PipelineSpec:
    """
    Read and validate a pipeline run file.

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
    Validate the top-level run file and parse every block in order.

    Validates `on_block_failure` and that `blocks` is non-empty, parses
    each raw block via `_parse_block` (rejecting duplicate names as it
    goes), then makes a second pass ensuring every `transfer.from_block`
    reference names a strictly earlier block in the same file.

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

    # Second pass: transfer.from_block must point at a *strictly earlier*
    # block, so a run-time lookup by name can never fail on a forward or
    # self reference.
    for i, block in enumerate(blocks):
        ref = block.transfer.from_block
        if ref is None:
            continue
        earlier_names = {b.name for b in blocks[:i]}
        if ref not in earlier_names:
            raise PipelineSpecError(
                f"[{source}] Block '{block.name}' (position {i}) has "
                f"transfer.from_block='{ref}', which must name an earlier "
                f"block in the same file. Earlier blocks: "
                f"{sorted(earlier_names) or '(none)'}."
            )

    return PipelineSpec(blocks=blocks, on_block_failure=on_block_failure, log_level=log_level)


def _parse_block(raw_block: Dict[str, Any], index: int, seen_names: Set[str], source: str) -> PipelineBlock:
    """
    Validate one raw block dict and build its `PipelineBlock`.

    Checks: `name` is a non-empty, unique string; `env_type` resolves via
    `resolve_environment_name`; `profile` is registered; `ship_model` /
    `dynamics_model`, if given, are valid enum members; `action` is one
    of the supported actions; `reward_configs` is present if and only if
    `action` == "evaluate_reward"; `stop_condition` and `transfer` are
    delegated to `_parse_stop_condition`/`_parse_transfer`; `on_failure`,
    if given, is a valid failure mode.

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
    transfer = _parse_transfer(raw_block.get("transfer", {}), where, name)

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
    Validate a raw `stop_condition` dict and build its `StopConditionSpec`.

    `criterion` defaults to "NBR_POLICY_UPDATES" and must name a member
    of `fw.configuration.stop_condition.Criterion`; "AUTO_CONVERGE" is
    rejected here even though it is a valid enum member, since
    `StopCondition.__init__` raises `NotImplementedError` for it (see
    `_UNIMPLEMENTED_CRITERIA`).

    Args:
        raw: The block's raw `stop_condition` dict (may be empty).
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
    return StopConditionSpec(
        criterion=criterion,
        max_time_steps=int(raw.get("max_time_steps", 2000)),
        target_reward=raw.get("target_reward"),
        target_loss=raw.get("target_loss"),
        patience=raw.get("patience"),
    )


def _parse_transfer(raw: Dict[str, Any], where: str, name: str) -> TransferSpec:
    """
    Validate a raw `transfer` dict and build its `TransferSpec`.

    `policy_strategy`, if given, must name a member of
    `fw.configuration.config.PolicyStrategy`; "REUSE_CURRENT_POLICY" is
    rejected even though it is a valid enum member (see
    `_UNSUPPORTED_POLICY_STRATEGIES` for why). `from_block` and
    `external_training_name` are mutually exclusive.

    Args:
        raw: The block's raw `transfer` dict (may be empty).
        where: Location string used in error messages.
        name: The owning block's name, used in error messages.

    Returns:
        A validated `TransferSpec`.

    Raises:
        PipelineSpecError: If `policy_strategy` is unknown/unsupported,
            or both `from_block` and `external_training_name` are set.
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

    return TransferSpec(
        policy_strategy=policy_strategy,
        from_block=from_block,
        external_training_name=external_training_name,
    )


def _validate_choice(field_name: str, value: Any, valid: Set[str], where: str) -> None:
    """
    Raise if `value` is not one of `valid`.

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
    Raise if `value` does not name a member of `enum_cls`.

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
