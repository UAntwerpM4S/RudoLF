"""
Dataclasses describing validated JSON-driven pipeline runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# These containers are validated by run_loader before execution.
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
    One train, visualize, or evaluate_reward pipeline step.
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


@dataclass
class PipelineSpec:
    """
    A complete pipeline run and its run-level defaults.
    """

    blocks: List[PipelineBlock] = field(default_factory=list)
    on_block_failure: str = "continue"
    log_level: str = "INFO"
