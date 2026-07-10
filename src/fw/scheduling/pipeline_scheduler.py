"""
pipeline_scheduler.py - Drive a sequence of data-driven `PipelineBlock`s
against ONE shared `AgentManager`.

This is a SIBLING to `simple_scheduler.py`, not a replacement for it:
`fw.main`'s existing, unchanged flow still uses `SimpleScheduler` exactly as
before. This scheduler is only ever constructed directly by
`fw.main_pipeline`, for JSON-driven runs.

Why a shared AgentManager instead of one Trainer/AgentManager per block
------------------------------------------------------------------------
Each block may want its own ship/dynamics model, VesselProfile, and
hyperparameters, but should still be able to transfer a policy to/from any
other block regardless of environment type or agent_type context. Rather
than constructing a fresh `Trainer`/`AgentManager` per block (which forces
every block's environment to be known and cached up front, and duplicates
Trainer's own scheduler-loading machinery), this scheduler reuses ONE
`AgentManager` for the whole run and swaps its profile/hyperparameters per
block via the additive `profile=`/`hyperparameters=` parameters added to
`AgentManager.set_environment` (see agent_manager.py's docstring there).

Design notes / things that are NOT obvious from the block spec alone
---------------------------------------------------------------------
* `fw.env_factory._Factory` caches at most ONE instance per canonical
  environment name (e.g. "PySimEnv") and raises `EnvironmentRegistryError`
  if a second, differently-configured request for that same name comes in.
  Since a pipeline can legitimately want two blocks with the same
  `env_type` but different ship/dynamics models, this scheduler bypasses
  that shared cache entirely via `pipeline_env.build_block_environment` -
  the same technique `Trainer._build_eval_environment` already uses for
  reward evaluation, for exactly the same reason.
* `pipeline_env.build_block_environment` also overrides each freshly-built
  environment's `training_name` to the block's own unique `name` (rather
  than the canonical env name env_factory stamps by default). Without
  this, two blocks sharing an `env_type` would silently overwrite each
  other's saved checkpoint on disk, since `AgentManager`/this scheduler key
  saved policies off `training_name`
  (`{training_name}_{agent_type}_policy.zip`). This also means
  `transfer.from_block` / `transfer.external_training_name` resolve
  directly to a `training_name` - no separate checkpoint-name bookkeeping
  dict needed.
* This scheduler talks to `AgentManager.set_environment(other_env_name=...)`
  directly for BOTH "train" and "visualize" actions, passing an explicit,
  already-resolved `other_env_name`. This sidesteps a bug found in an
  earlier iteration of this pipeline (a per-block-Trainer design), where
  `Trainer.visualize()` resolves its policy purely from
  `gym_env.training_name` and has no way to honor a block's own
  `transfer` - here, transfer resolution is identical for every action.
* This scheduler currently assumes ONE `agent_type` for the entire run
  (whatever the `AgentManager` was constructed with). Mixing agent types
  within a single pipeline (e.g. a `MultiSimEnv` block needing MPPO
  alongside a plain PPO block) is not supported - that would need
  `fw.main`'s `_select_agent_type` logic replicated per block, which would
  also mean rebuilding the AgentManager between blocks. Flag this if you
  need it.
"""

from __future__ import annotations

import os
import time
import logging
import traceback

from typing import Any, Dict, Optional, Tuple

from fw.scheduling.base_scheduler import BaseScheduler
from fw.configuration.config import PolicyStrategy, get_hyperparameters, load_config_parameters
from fw.configuration.stop_condition import Criterion, StopCondition
from fw.configuration.profile_applier import apply_profile

from fw.scheduling.run_spec import PipelineBlock, StopConditionSpec, TransferSpec
from fw.scheduling.pipeline_env import resolve_block_profile, build_block_environment

logger = logging.getLogger(__name__)


class PipelineScheduler(BaseScheduler):
    """
    Executes an ordered list of `PipelineBlock`s against a shared `AgentManager`.

    Unlike `SimpleScheduler`, this scheduler does not receive a pre-built
    list of environments up front - each block's environment is built
    lazily, from that block's own configuration, immediately before it
    runs (see module docstring for why).
    """

    def __init__(
        self,
        agent_mgr,
        blocks: Optional[list] = None,
        on_block_failure: str = "continue",
    ):
        """
        Initializes the PipelineScheduler.

        Args:
            agent_mgr: The `AgentManager` instance reused across every
                block (its profile/hyperparameters are swapped per block
                via `set_environment`).
            blocks: Ordered list of `PipelineBlock`s to run. Can also be
                supplied later via `execute(blocks=...)`.
            on_block_failure: Run-level default - "continue" or "abort" -
                for what happens when a block raises. Overridable per block
                via `PipelineBlock.on_failure`.

        Note:
            `BaseScheduler.__init__` requires a non-empty `envs` list up
            front, which doesn't fit here (environments are built lazily,
            per block). We therefore set up the same minimal state
            ourselves instead of calling `super().__init__()`.
        """

        self.agent_mgr = agent_mgr
        self.envs = []
        self._blocks = blocks or []
        self._on_block_failure = on_block_failure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, blocks: Optional[list] = None) -> Dict[str, Any]:
        """
        Run each block in order.

        Args:
            blocks: Optional override of the blocks passed at construction.

        Returns:
            dict: Metrics keyed by block name, e.g.
                {"block_a": {"status": "success", "duration": 12.3}, ...}
        """

        blocks = blocks if blocks is not None else self._blocks
        if not blocks:
            raise RuntimeError("PipelineScheduler.execute() called with no blocks.")

        metrics: Dict[str, Any] = {}
        completed_block_names = set()

        for block in blocks:
            print(f"\n[PipelineScheduler] === Block '{block.name}' ({block.env_type}) ===")
            try:
                result = self._run_block(block, completed_block_names)
                metrics[block.name] = {"status": "success", **result}
                if block.action == "train":
                    completed_block_names.add(block.name)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, mirrors SimpleScheduler
                traceback.print_exc()
                logger.error("[PipelineScheduler] Block '%s' failed: %s", block.name, exc)
                metrics[block.name] = {"status": "failure", "error": str(exc)}

                failure_mode = block.on_failure or self._on_block_failure
                if failure_mode == "abort":
                    print(
                        f"[PipelineScheduler] Aborting pipeline after failure "
                        f"in block '{block.name}'."
                    )
                    break

        print("[PipelineScheduler] Pipeline complete.")
        return metrics

    # ------------------------------------------------------------------
    # Per-block execution
    # ------------------------------------------------------------------

    def _run_block(self, block: PipelineBlock, completed_block_names: set) -> Dict[str, Any]:
        """
        _run_block
        """

        profile = resolve_block_profile(block)

        framework_config = load_config_parameters()
        framework_config["wind"] = block.wind
        framework_config["current"] = block.current
        apply_profile(profile, config=framework_config)

        env, canonical_name = build_block_environment(block, framework_config)
        apply_profile(profile, env=env.gym_env)
        sim = getattr(env.gym_env, "phys_sim", None) or getattr(env.gym_env, "_phys_sim", None)
        if sim is not None:
            apply_profile(profile, simulator=sim)

        print(
            f"[PipelineScheduler] ship={profile.model.ship_model} "
            f"dynamics={profile.model.dynamics_model} action={block.action}"
        )

        if block.action == "evaluate_reward":
            return self._run_evaluate_reward(block, env, profile, framework_config)

        policy_strategy, other_env_name = self._resolve_transfer(block, completed_block_names)
        hyperparameters = self._resolve_hyperparameters(block, canonical_name)

        self.agent_mgr.set_environment(
            env=env.gym_env,
            policy_strategy=policy_strategy,
            other_env_name=other_env_name,
            profile=profile,
            hyperparameters=hyperparameters,
        )

        start = time.time()

        if block.action == "visualize":
            self.agent_mgr.visualize_trained_agent(
                num_episodes=1,
                live_animation=framework_config.get("live_animation", False),
                return_summary=False,
                show_plots=block.show_plots,
            )
        else:  # "train"
            stop_condition = self._build_stop_condition(block.stop_condition)
            stop_condition.reset()
            self.agent_mgr.train(stop_condition=stop_condition)
            self._save_checkpoint(block, env, profile)

            if not block.show_plots:
                # Mirrors Trainer.train()'s show_plots semantics exactly:
                # False means skip the ENTIRE post-training evaluation step
                # (episode rollout, metrics computation, figure saving) -
                # not just the interactive window - since running a full
                # evaluation episode costs real time and clutters logs for
                # an intermediate block nobody's going to look at.
                return {"duration": time.time() - start}

            # The agent already resident in self.agent_mgr is the one that
            # was just trained, so no reload from disk is needed here.
            summary = self.agent_mgr.visualize_trained_agent(
                num_episodes=1,
                live_animation=framework_config.get("live_animation", False),
                return_summary=True,
                show_plots=block.show_plots,
            )
            duration = time.time() - start
            return {"duration": duration, "evaluation_summary": summary}

        return {"duration": time.time() - start}

    def _save_checkpoint(self, block: PipelineBlock, env, profile) -> None:
        """"
        _save_checkpoint
        """

        training_name = env.gym_env.training_name
        save_name = f"{training_name}_{self.agent_mgr.agent_type}_policy"
        self.agent_mgr.agent.save_policy(save_name)

        model_dir = getattr(self.agent_mgr.agent, "model_dir", None) or "models"
        profile_path = os.path.join(model_dir, f"{training_name}_profile.json")
        try:
            os.makedirs(model_dir, exist_ok=True)
            with open(profile_path, "w", encoding="utf-8") as fh:
                fh.write(profile.to_json())
        except OSError as exc:
            logger.warning("[PipelineScheduler] Could not save profile JSON (%s).", exc)

    def _run_evaluate_reward(self, block: PipelineBlock, env, profile, framework_config: dict) -> Dict[str, Any]:
        """
        _run_evaluate_reward
        """

        # Local import: evaluate_reward delegates to the existing, already
        # well-exercised Trainer.evaluate_reward implementation rather than
        # duplicating it, so a fresh, block-scoped Trainer/AgentManager pair
        # is built just for this one block. Note this means evaluate_reward
        # blocks do NOT participate in the transfer chain (their internal
        # agent is separate from self.agent_mgr), and don't receive this
        # block's hyperparameter_overrides (Trainer resolves its own
        # defaults) - a deliberate, minor scope limitation to avoid
        # duplicating Trainer.evaluate_reward's substantial logic here.
        from fw.scheduling.trainer import Trainer

        start = time.time()
        trainer = Trainer(self.agent_mgr.agent_type, [env], profile, "SimpleScheduler", config=framework_config)
        stop_condition = self._build_stop_condition(block.stop_condition)
        results = trainer.evaluate_reward(
            block.reward_configs,
            {str(env): stop_condition},
            verbose=True,
        )
        trainer.export_evaluation_results(results, f"{block.name}_reward_evaluation_results.csv")
        return {"duration": time.time() - start}

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_hyperparameters(self, block: PipelineBlock, canonical_name: str) -> dict:
        """
        _resolve_hyperparameters
        """

        base = get_hyperparameters(self.agent_mgr.agent_type, env_name=canonical_name)
        if block.hyperparameter_overrides:
            base = {**base, **block.hyperparameter_overrides}
        return base

    def _resolve_transfer(
        self, block: PipelineBlock, completed_block_names: set
    ) -> Tuple[PolicyStrategy, Optional[str]]:
        """
        Resolve (policy_strategy, other_env_name) for this block.

        Used uniformly for BOTH "train" and "visualize" actions, since this
        scheduler always passes an explicit `other_env_name` to
        `AgentManager.set_environment` rather than relying on
        `training_name` self-lookup - see module docstring for why this
        matters.

        "No transfer given" means different things depending on the
        action: for "train" it means start from a fresh (RESET_POLICY)
        policy; for "visualize" it means "show me what this block's own
        training produced" - i.e. self-reference by block.name.
        """

        transfer: TransferSpec = block.transfer

        if transfer.policy_strategy:
            strategy = PolicyStrategy[transfer.policy_strategy]
        elif transfer.from_block or transfer.external_training_name:
            strategy = PolicyStrategy.REUSE_OTHER_POLICY
        elif block.action == "visualize":
            strategy = PolicyStrategy.REUSE_OTHER_POLICY
        else:
            strategy = PolicyStrategy.RESET_POLICY

        other_env_name = None
        if strategy == PolicyStrategy.REUSE_OTHER_POLICY:
            if transfer.external_training_name:
                other_env_name = transfer.external_training_name
            elif transfer.from_block:
                if transfer.from_block not in completed_block_names:
                    raise RuntimeError(
                        f"Block '{block.name}' wants to reuse the policy "
                        f"from '{transfer.from_block}', but that block has "
                        f"not completed successfully yet in this run."
                    )
                other_env_name = transfer.from_block
            elif block.action == "visualize":
                # Self-reference default described above.
                other_env_name = block.name
            else:
                raise RuntimeError(
                    f"Block '{block.name}': policy_strategy is "
                    f"REUSE_OTHER_POLICY but neither 'from_block' nor "
                    f"'external_training_name' was given."
                )

        return strategy, other_env_name

    @staticmethod
    def _build_stop_condition(spec: StopConditionSpec) -> StopCondition:
        """
        _build_stop_condition
        """

        return StopCondition(
            stop_criterion=Criterion[spec.criterion],
            max_time_steps=spec.max_time_steps,
            target_reward=spec.target_reward,
            target_loss=spec.target_loss,
            patience=spec.patience,
        )
