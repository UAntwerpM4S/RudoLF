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

from fw.configuration.config import PolicyStrategy
from fw.configuration.config_profile import ProfileRegistry
from fw.configuration.stop_condition import Criterion
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.configuration.env_factory import resolve_environment_name, EnvironmentRegistryError

from fw.scheduling.run_spec import PipelineBlock, PipelineSpec, StopConditionSpec, TransferSpec

_VALID_ACTIONS = {"train", "visualize", "evaluate_reward"}
_VALID_FAILURE_MODES = {"continue", "abort"}

# StopCondition.__init__ raises NotImplementedError for this immediately -
# but only when that block's StopCondition is actually built, at
# *execution* time, potentially after earlier blocks already ran for a
# long time. Reject it here instead, at *load* time, for the whole plan.
_UNIMPLEMENTED_CRITERIA = {"AUTO_CONVERGE"}

# Every pipeline block gets its own Trainer/AgentManager (see
# main_pipeline.py), so there is no in-memory "current" agent to carry
# across blocks - any transfer necessarily goes through a saved policy
# file on disk. REUSE_CURRENT_POLICY would, in practice, behave exactly
# like REUSE_OTHER_POLICY pointed at the current block's own env, in a
# confusing way (AgentManager.set_environment saves-then-reloads a file for
# BOTH strategies - see its source). Rejected here instead of silently
# accepted.
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
    _parse_spec
    """

    on_block_failure = raw.get("on_block_failure", "continue")
    _validate_choice("on_block_failure", on_block_failure, _VALID_FAILURE_MODES, f"[{source}]")

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

    return PipelineSpec(blocks=blocks, on_block_failure=on_block_failure)


def _parse_block(raw_block: Dict[str, Any], index: int, seen_names: Set[str], source: str) -> PipelineBlock:
    """
    _parse_block
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
        wind=bool(raw_block.get("wind", False)),
        current=bool(raw_block.get("current", False)),
        show_plots=bool(raw_block.get("show_plots", False)),
        stop_condition=stop_condition,
        hyperparameter_overrides=raw_block.get("hyperparameter_overrides", {}) or {},
        reward_configs=reward_configs,
        transfer=transfer,
        on_failure=on_failure,
    )


def _parse_stop_condition(raw: Dict[str, Any], where: str, name: str) -> StopConditionSpec:
    """
    _parse_stop_condition
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
    _parse_transfer
    """

    policy_strategy = raw.get("policy_strategy")
    if policy_strategy is not None:
        _validate_enum_member("transfer.policy_strategy", policy_strategy, PolicyStrategy, where, name)
        if policy_strategy in _UNSUPPORTED_POLICY_STRATEGIES:
            raise PipelineSpecError(
                f"{where} ('{name}'): transfer.policy_strategy "
                f"'{policy_strategy}' is not supported by pipeline blocks - "
                f"each block gets its own Trainer/agent, so there is no "
                f"in-memory 'current' policy to carry over. Use "
                f"'REUSE_OTHER_POLICY' with an explicit 'from_block' or "
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
    _validate_choice
    """

    if value not in valid:
        raise PipelineSpecError(
            f"{where}: '{field_name}' must be one of {sorted(valid)}, got '{value}'."
        )


def _validate_enum_member(field_name: str, value: str, enum_cls, where: str, block_name: str) -> None:
    """
    _validate_enum_member
    """

    valid_names = {member.name for member in enum_cls}
    if value not in valid_names:
        raise PipelineSpecError(
            f"{where} ('{block_name}'): '{field_name}' = '{value}' is not a "
            f"valid {enum_cls.__name__}. Valid options: {sorted(valid_names)}."
        )
