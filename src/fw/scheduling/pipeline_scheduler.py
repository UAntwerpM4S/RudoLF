"""
Execute pipeline blocks with one shared AgentManager.
"""

from __future__ import annotations

import os
import time
import logging

from typing import Any, Dict, Optional, Tuple
from fw.scheduling.base_scheduler import BaseScheduler
from fw.configuration.profile_applier import apply_profile
from fw.configuration.stop_condition import Criterion, StopCondition
from fw.configuration.config import PolicyStrategy, get_hyperparameters
from fw.scheduling.run_spec import block_output_subdir, validate_block_names
from fw.scheduling.run_spec import PipelineBlock, StopConditionSpec, TransferSpec
from fw.scheduling.pipeline_env import (
    block_framework_config,
    describe_block_config,
    build_block_environment,
    check_block_wind_current,
    resolve_block_profile,
)

logger = logging.getLogger(__name__)


class PipelineScheduler(BaseScheduler):
    """
    Execute ordered pipeline blocks with one shared AgentManager, so policies
    can transfer between blocks.

    Files shared between blocks (<block>_<agent>_policy.zip, Best_<block>_<agent>_policy.zip,
    <block>_profile.json) are written to model_dir; the block's own outputs
    (checkpoints, training_metrics.csv, evaluation plots) to model_dir/<block>,
    or model_dir/<block>_visualize for a visualize block that loads the policy
    named after itself (block_output_subdir). evaluate_reward blocks keep their
    own output locations.
    """

    def __init__(
        self,
        agent_mgr,
        blocks: Optional[list] = None,
        on_block_failure: str = "continue",
    ):
        """
        Args:
            agent_mgr: AgentManager reused by every block (profile and
                hyperparameters are swapped per block via set_environment).
            blocks: Ordered PipelineBlocks; can also be given to execute().
            on_block_failure: "continue" or "abort" when a block raises;
                PipelineBlock.on_failure overrides it per block.
        """

        super().__init__(agent_mgr)

        self._blocks = blocks or []
        self._on_block_failure = on_block_failure


    def execute(self, blocks: Optional[list] = None) -> Dict[str, Any]:
        """
        Run blocks in order (optionally overriding the constructor's blocks).

        Returns:
            dict: Metrics per block name, e.g. {"block_a": {"status": "success", "duration": 12.3}}.

        Raises:
            RuntimeError: If there are no blocks.
            ValueError: If block names are invalid or clash (ignoring case), or a
                block's wind/current flags do not fit its environment or
                UAntwerp preset (both checked before any block runs).
        """

        blocks = blocks if blocks is not None else self._blocks
        if not blocks:
            raise RuntimeError("PipelineScheduler.execute() called with no blocks.")
        self._preflight(blocks)

        metrics: Dict[str, Any] = {}
        completed_block_names = set()
        last_train_block: Optional[str] = None

        for block in blocks:
            logger.info("=== Block '%s' (%s) ===", block.name, block.env_type)
            try:
                result = self._run_block(block, completed_block_names, last_train_block)
                metrics[block.name] = {"status": "success", **result}
                if block.action == "train":
                    completed_block_names.add(block.name)
                    last_train_block = block.name
            except Exception as exc:
                logger.exception("[PipelineScheduler] Block '%s' failed: %s", block.name, exc)
                metrics[block.name] = {"status": "failure", "error": str(exc)}

                failure_mode = block.on_failure or self._on_block_failure
                if failure_mode == "abort":
                    logger.warning(
                        "Aborting pipeline after failure in block '%s'.", block.name
                    )
                    break

        logger.info("Pipeline complete.")
        return metrics


    @staticmethod
    def _preflight(blocks: list) -> None:
        """
        Check the block names and output folders (validate_block_names, also for
        blocks built in Python) and every block's wind/current flags against
        its resolved configuration (check_block_wind_current) before the first
        block runs. Other resolution errors are left to the block itself.

        Raises:
            ValueError: From validate_block_names or check_block_wind_current.
        """

        validate_block_names(blocks)
        for block in blocks:
            try:
                framework_config = block_framework_config(block, resolve_block_profile(block))
            except Exception:
                continue        # reported by the block when it runs, as before
            check_block_wind_current(block, framework_config)


    def _run_block(
            self, block: PipelineBlock, completed_block_names: set, last_train_block: Optional[str]
    ) -> Dict[str, Any]:
        """
        Build the block's environment and run its action.

        Args:
            block: The block to run.
            completed_block_names: "train" blocks completed so far (validates transfer.from_block).
            last_train_block: Most recent completed "train" block, or None (implicit chaining).

        Returns:
            Dict with "duration", plus "evaluation_summary" if a post-train rollout ran
            ("evaluation_error" instead if it failed).
        """

        profile = resolve_block_profile(block)
        framework_config = block_framework_config(block, profile)

        env, canonical_name = build_block_environment(block, framework_config)
        apply_profile(profile, env=env.gym_env)
        sim = getattr(env.gym_env, "phys_sim", None) or getattr(env.gym_env, "_phys_sim", None)
        if sim is not None:
            apply_profile(profile, simulator=sim)

        logger.info(
            "ship=%s dynamics=%s action=%s",
            profile.model.ship_model, profile.model.dynamics_model, block.action
        )
        logger.info("Block '%s' config: %s", block.name, describe_block_config(framework_config))

        if block.action == "evaluate_reward":
            return self._run_evaluate_reward(block, env, profile, framework_config)

        policy_strategy, other_env_name = self._resolve_transfer(block, completed_block_names, last_train_block)
        hyperparameters = self._resolve_hyperparameters(block, canonical_name)

        self.agent_mgr.set_environment(
            env=env.gym_env,
            policy_strategy=policy_strategy,
            other_env_name=other_env_name,
            profile=profile,
            hyperparameters=hyperparameters,
            require_policy=block.action == "visualize",
            output_subdir=block_output_subdir(block),
        )

        start = time.time()

        if block.action == "visualize":
            self.agent_mgr.visualize_trained_agent(
                num_episodes=1,
                live_animation=framework_config.get("live_animation", False),
                return_summary=False,
                show_plots=block.show_plots,
            )
        else:
            stop_condition = self._build_stop_condition(block.stop_condition)
            stop_condition.reset()
            self.agent_mgr.train(stop_condition=stop_condition)
            self._save_checkpoint(block, env, profile)

            if not block.show_plots:
                return {"duration": time.time() - start}

            try:
                summary = self.agent_mgr.visualize_trained_agent(
                    num_episodes=1,
                    live_animation=framework_config.get("live_animation", False),
                    return_summary=True,
                    show_plots=block.show_plots,
                )
            except Exception as exc:
                # The training is done and its policy saved: the block succeeded.
                logger.exception("Evaluation after training block '%s' failed; the block keeps its "
                                 "trained policy, without an evaluation summary.", block.name)
                return {"duration": time.time() - start, "evaluation_error": str(exc)}
            duration = time.time() - start
            return {"duration": duration, "evaluation_summary": summary}

        return {"duration": time.time() - start}

    def _save_checkpoint(self, block: PipelineBlock, env, profile) -> None:
        """
        Save the trained policy and the profile JSON, named after
        env.gym_env.training_name. `block` is unused.
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
        Run an "evaluate_reward" block (non-empty reward_configs) with a scoped
        Trainer. framework_config is passed so the profile is not applied twice.

        Returns:
            Dict with "duration"; the results are written to CSV.
        """

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


    def _resolve_hyperparameters(self, block: PipelineBlock, canonical_name: str) -> dict:
        """
        Default hyperparameters for the agent and env, with block overrides on top.
        """

        base = get_hyperparameters(self.agent_mgr.agent_type, env_name=canonical_name)
        if block.hyperparameter_overrides:
            base = {**base, **block.hyperparameter_overrides}
        return base

    @staticmethod
    def _resolve_transfer(
            block: PipelineBlock, completed_block_names: set, last_train_block: Optional[str]
    ) -> Tuple[PolicyStrategy, Optional[str]]:
        """
        Resolve (policy strategy, source name) for a block.

        Defaults: independent gives RESET_POLICY; an explicit policy_strategy wins;
        from_block / external_training_name, "visualize" blocks and "train" blocks
        after an earlier completed train block reuse another policy (implicit
        chaining from last_train_block); otherwise RESET_POLICY.

        Args:
            block: The block whose transfer is resolved.
            completed_block_names: "train" blocks completed so far (validates from_block).
            last_train_block: Most recent completed "train" block, or None.

        Raises:
            RuntimeError: If independent is combined with another transfer field or
                used on a non-train block (also checked for blocks built in Python,
                bypassing run_loader), or if a reused policy source is missing.
        """

        transfer: TransferSpec = block.transfer

        if transfer.independent and (
                transfer.policy_strategy or transfer.from_block or transfer.external_training_name
        ):
            raise RuntimeError(
                f"Block '{block.name}': transfer.independent=True cannot be "
                f"combined with 'policy_strategy', 'from_block', or "
                f"'external_training_name'. Set only 'independent': True, "
                f"or remove it and use one of the other fields instead."
            )

        if transfer.independent and block.action != "train":
            raise RuntimeError(
                f"Block '{block.name}': transfer.independent is only valid "
                f"on 'train' blocks (got action='{block.action}')."
            )

        if transfer.independent:
            return PolicyStrategy.RESET_POLICY, None

        if transfer.policy_strategy:
            strategy = PolicyStrategy[transfer.policy_strategy]
        elif transfer.from_block or transfer.external_training_name:
            strategy = PolicyStrategy.REUSE_OTHER_POLICY
        elif block.action == "visualize":
            strategy = PolicyStrategy.REUSE_OTHER_POLICY
        elif block.action == "train" and last_train_block is not None:
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
                other_env_name = block.name
            elif block.action == "train":
                if last_train_block is None:
                    raise RuntimeError(
                        f"Block '{block.name}': policy_strategy is "
                        f"REUSE_OTHER_POLICY but no 'from_block' or "
                        f"'external_training_name' was given, and no "
                        f"earlier 'train' block has completed yet in this "
                        f"run to implicitly chain from."
                    )
                other_env_name = last_train_block
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
        Build a StopCondition from a spec validated by run_loader._parse_stop_condition.
        """

        return StopCondition(
            stop_criterion=Criterion[spec.criterion],
            max_time_steps=spec.max_time_steps,
            target_reward=spec.target_reward,
            target_loss=spec.target_loss,
            patience=spec.patience,
        )
