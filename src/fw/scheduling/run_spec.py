"""
run_spec.py - Data containers for a JSON-driven pipeline run.

A pipeline run is an ordered list of `PipelineBlock`s. Each block fully
describes one train / visualize / evaluate_reward step: which environment,
ship model, dynamics model, profile, stop condition, and (optionally) which
earlier block's policy to carry forward as a starting point.

These are plain dataclasses with no validation of their own. Validation
against the framework's actual enums/registries (ShipModel, Model,
ProfileRegistry, PolicyStrategy, Criterion, env_factory's alias table)
happens in `run_loader.py`, so a malformed run file fails loudly - with a
clear message - before any environment is built or any training starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StopConditionSpec:
    """
    Mirrors `fw.stop_condition.StopCondition`'s constructor arguments.

    `criterion` is kept as the raw string name (e.g. "TARGET_REWARD") here;
    `run_loader` validates it against `fw.stop_condition.Criterion` and
    `pipeline_scheduler` converts it to the actual enum member right before
    building a `StopCondition` instance.
    """

    criterion: str = "NBR_POLICY_UPDATES"
    max_time_steps: int = 2000
    target_reward: Optional[float] = None
    target_loss: Optional[float] = None
    patience: Optional[int] = None


@dataclass
class TransferSpec:
    """
    Describes how a block's starting policy should be obtained.

    Exactly one of `from_block` / `external_training_name` should be set
    when a transfer is desired; `policy_strategy`, if given, always wins
    over the inferred choice.

    Chaining default: for a "train" block, if none of `policy_strategy`,
    `from_block`, `external_training_name`, or `independent` is set, the
    scheduler now implicitly reuses the policy of the most recently
    completed "train" block *in this same run* (i.e. behaves as if
    `from_block` pointed at that block). If no earlier "train" block has
    completed yet (e.g. this is the first block), it still starts from a
    fresh (RESET_POLICY) policy - there is nothing to chain from. Set
    `independent=True` to opt a block out of this implicit chaining
    explicitly.

    Attributes:
        policy_strategy: Explicit override - one of "RESET_POLICY",
            "REUSE_CURRENT_POLICY", "REUSE_OTHER_POLICY". When omitted,
            the scheduler infers a strategy per the chaining rules above.
        from_block: Name of an earlier block *in this same run* whose
            trained policy should be loaded as the starting point. Only
            valid for same-process, auto-chained runs.
        external_training_name: The `training_name` portion of a policy
            checkpoint saved by a *previous, separate* invocation of the
            pipeline (i.e. the file is
            `{external_training_name}_{agent_type}_policy.zip` inside the
            configured `model_dir`). Use this to chain across separate
            script runs instead of within one process.
        independent: When `True`, forces this "train" block to start from
            a fresh (RESET_POLICY) policy and explicitly opts out of the
            implicit previous-train-block chaining described above.
            Mutually exclusive with `policy_strategy`, `from_block`, and
            `external_training_name` - set only this field to mark a
            block as independent. Only valid on "train" blocks, since the
            chaining it opts out of never applies to "visualize" or
            "evaluate_reward" blocks in the first place.
    """

    policy_strategy: Optional[str] = None
    from_block: Optional[str] = None
    external_training_name: Optional[str] = None
    independent: bool = False


@dataclass
class PipelineBlock:
    """
    One train / visualize / evaluate_reward step in a pipeline run.

    Attributes:
        name: Unique identifier for this block within the run. Used for
            `transfer.from_block` references, metrics keys, and log output.
        env_type: Environment selector, same values/aliases accepted by
            `fw.env_factory.resolve_environment_name` (e.g. "PySimEnv",
            "FhSimEnv").
        profile: Name of a `VesselProfile` registered in `ProfileRegistry`.
        ship_model: Optional override of the profile's ship model - one of
            `fw.ships.ship_factory.ShipModel`'s member names.
        dynamics_model: Optional override of the profile's dynamics model -
            one of `fw.dynamics.dynamics_model.Model`'s member names.
        profile_overrides: Optional free-form dict, deep-merged onto the
            resolved profile's `to_dict()` representation before being
            reconstructed via `VesselProfile.from_dict()`. Lets a block
            tweak nested settings (e.g. `{"actuator": {"enable_smoothing":
            false}}`) without needing a brand new registered profile.
        action: One of "train", "visualize", "evaluate_reward".
        wind / current: Environment-level toggles, same as `CONFIG_PARAMETERS`.
        show_plots: Whether to run the post-training/visualize evaluation
            at all for this block. Defaults to False - when False, the
            ENTIRE step (episode rollout, metrics computation, figure
            saving, and any interactive window) is skipped, not just the
            interactive window, since running a full evaluation episode
            costs real time and clutters logs for a block nobody's going
            to look at. Set True on a specific block to get its full
            evaluation, including a blocking matplotlib window.
        stop_condition: Stop condition for this block (ignored for
            "visualize" and "evaluate_reward" actions, except max_time_steps
            for evaluate_reward, matching Trainer.evaluate_reward's usage).
        hyperparameter_overrides: Optional dict merged onto
            `get_hyperparameters(agent_type, env_name)`'s defaults.
        reward_configs: Required (non-empty) when action == "evaluate_reward";
            same shape as `CONFIG_PARAMETERS["reward_configs"]`.
        transfer: See `TransferSpec`.
        on_failure: Optional per-block override of the run-level
            `on_block_failure` setting - "continue" or "abort".
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
    A full pipeline run: an ordered list of blocks plus run-level defaults.

    Attributes:
        blocks: Ordered list of `PipelineBlock`s, executed in sequence.
        on_block_failure: Run-level default for what happens when a block
            raises - "continue" (log and move to the next block, matching
            `SimpleScheduler`'s original behavior) or "abort" (stop the
            whole run). Overridable per block via `PipelineBlock.on_failure`.
        log_level: Run-level log level - one of "DEBUG", "INFO", "WARNING",
            "ERROR", "CRITICAL". Applied once, process-wide, via
            `fw.common.tools.logging_config.setup_logging` before any block
            runs. Defaults to "INFO".
    """

    blocks: List[PipelineBlock] = field(default_factory=list)
    on_block_failure: str = "continue"
    log_level: str = "INFO"
