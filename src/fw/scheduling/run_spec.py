"""
Dataclasses describing JSON-driven pipeline runs (validated by run_loader).
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Block names become file and folder names (model_dir/<block>), so they follow
# the Windows rules: no reserved characters, control characters or device names.
_INVALID_NAME_CHARS = set('/\\:*?"<>|')
_RESERVED_NAME_RE = re.compile(r"(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]|CONIN\$|CONOUT\$)(\..*)?", re.IGNORECASE)


@dataclass
class StopConditionSpec:
    """
    Configuration for constructing a StopCondition.
    """

    criterion: str = "NBR_POLICY_UPDATES"
    max_time_steps: int = 2000
    target_reward: Optional[float] = None
    target_loss: Optional[float] = None
    patience: Optional[int] = None


@dataclass
class TransferSpec:
    """
    Configuration for selecting a block's starting policy.
    """

    policy_strategy: Optional[str] = None
    from_block: Optional[str] = None
    external_training_name: Optional[str] = None
    independent: bool = False


@dataclass
class PipelineBlock:
    """
    One train, visualize or evaluate_reward pipeline step.
    """

    name: str
    env_type: str
    profile: str = "default_profile"
    ship_model: Optional[str] = None
    dynamics_model: Optional[str] = None
    profile_overrides: Dict[str, Any] = field(default_factory=dict)
    action: str = "train"
    wind: bool = False
    current: bool = False
    show_plots: bool = False
    stop_condition: StopConditionSpec = field(default_factory=StopConditionSpec)
    hyperparameter_overrides: Dict[str, Any] = field(default_factory=dict)
    reward_configs: Optional[List[Dict[str, Any]]] = None
    transfer: TransferSpec = field(default_factory=TransferSpec)
    on_failure: Optional[str] = None
    # Block-level overrides of config.py entries (scenario and the surrogate
    # configs, see run_loader.BLOCK_CONFIG_KEYS and
    # pipeline_env.merge_config_override); a key absent here keeps config.py.
    config_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSpec:
    """
    A complete pipeline run and its run-level defaults.
    """

    blocks: List[PipelineBlock] = field(default_factory=list)
    on_block_failure: str = "continue"
    log_level: str = "INFO"


def block_name_error(name: Any) -> Optional[str]:
    """
    Why `name` cannot be used as a block (file and folder) name, or None if it can.
    """

    if not isinstance(name, str) or not name:
        return "it must be a non-empty string"
    if name != name.strip() or name.endswith("."):
        return "it must not have leading/trailing spaces or end with a dot"
    if set(name) & _INVALID_NAME_CHARS or any(ord(c) < 32 for c in name):
        return f"it must not contain control characters or any of {''.join(sorted(_INVALID_NAME_CHARS))}"
    if _RESERVED_NAME_RE.fullmatch(name):
        return "it is a reserved device name on Windows"
    return None


def block_output_subdir(block: PipelineBlock) -> str:
    """
    Folder (under model_dir) for the block's own outputs: the block name, or
    <name>_visualize for a visualize block that loads the policy named after
    itself, so it does not overwrite the outputs of the training it shows.
    """

    transfer = getattr(block, "transfer", None) or TransferSpec()
    reuses_own_name = (
        not transfer.from_block
        and transfer.external_training_name in (None, block.name)
        and transfer.policy_strategy in (None, "REUSE_OTHER_POLICY")
    )
    if block.action == "visualize" and reuses_own_name:
        return f"{block.name}_visualize"
    return block.name


def validate_block_names(blocks: List[PipelineBlock]) -> None:
    """
    Check that every block name is a valid file/folder name and that block
    names and output folders are unique, ignoring case (macOS and Windows
    file systems are case-insensitive).

    Raises:
        ValueError: On the first invalid or clashing name.
    """

    seen_names: Dict[str, str] = {}
    seen_dirs: Dict[str, str] = {}
    for block in blocks:
        error = block_name_error(block.name)
        if error:
            raise ValueError(f"Block name {block.name!r} is used as a file and folder name, so {error}.")
        key = block.name.casefold()
        if key in seen_names:
            raise ValueError(f"Block names '{seen_names[key]}' and '{block.name}' clash: "
                             f"block names must be unique, ignoring case.")
        seen_names[key] = block.name
    for block in blocks:
        subdir = block_output_subdir(block)
        key = subdir.casefold()
        if key in seen_dirs:
            raise ValueError(f"Blocks '{seen_dirs[key]}' and '{block.name}' would both write to the output "
                             f"folder '{subdir}' (ignoring case); rename one of them.")
        seen_dirs[key] = block.name
