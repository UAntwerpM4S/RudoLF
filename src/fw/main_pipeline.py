"""
main_pipeline.py - JSON-driven entry point for running a sequence of
train / visualize / evaluate_reward blocks, with optional automatic policy
transfer between them.

This is an ADDITIVE entry point: main.py and config.py are untouched.
`fw.simple_scheduler` is untouched too and continues to be used by
`fw.main` exactly as before - this script uses `fw.pipeline_scheduler`'s
`PipelineScheduler` instead, a sibling scheduler purpose-built for
data-driven, multi-block runs (see its module docstring for the full
rationale).

Use this script when you want a run defined declaratively in a JSON file
instead of hand-editing config.py, e.g. to chain "train in PySimEnv, then
transfer into FhSimEnv" or to sweep several ship/dynamics-model
combinations without touching Python code.

Usage:
    python -m fw.main_pipeline --spec path/to/run.json

See run_spec.py for the schema each block supports, and
example_pipeline.json (next to this file) for a worked example.

Scope, explicitly out for now:
  - AirSim blocks are rejected with a clear message (see pipeline_env.py) -
    fw.main's AirSim path goes through fw.main_with_airsim's dedicated UE5
    connection/telemetry orchestration, which this pipeline does not
    replicate.
  - transfer.policy_strategy "REUSE_CURRENT_POLICY" is rejected at load
    time (see run_loader.py) - AgentManager.set_environment round-trips
    every strategy through a saved policy file regardless, so it never
    behaves differently from REUSE_OTHER_POLICY pointed at the same block.
  - transfer.from_block must reference an earlier "train" block - rejected
    at load time (see run_loader.py) if it names a "visualize" or
    "evaluate_reward" block instead, since only "train" blocks save a
    policy checkpoint.
  - transfer.independent is only valid on "train" blocks - rejected at
    load time (see run_loader.py) on "visualize"/"evaluate_reward" blocks,
    since the implicit previous-train-block chaining it opts out of never
    applies to those actions in the first place. A "train" block with no
    transfer at all implicitly chains from the previous "train" block by
    default (see pipeline_scheduler.py's module docstring); set
    "independent": true to opt a block out of that.
  - stop_condition.criterion "AUTO_CONVERGE" is rejected at load time (see
    run_loader.py) - StopCondition.__init__ raises NotImplementedError for
    it immediately, so this catches it before any block runs rather than
    partway through the pipeline.
  - One agent_type for the whole run (default PPO) - see
    pipeline_scheduler.py's module docstring.
"""

if __name__ == "__main__" or __name__ == "fw.main_pipeline":
    import os

    # Same rationale as main.py: must be set before any torch/numpy import.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import logging
    import argparse
    import dataclasses
    import multiprocessing
    import fw.configuration.config as config_module

    from pathlib import Path
    from datetime import datetime, timezone
    from fw.common.tools.logging_config import setup_logging
    from fw.common.tools.run_bundler import bundle_run_output
    from fw.common.tools.path_utils import resolve_path_relative_to_src
    from fw.configuration.config import PPO_POLICY_NAME, get_hyperparameters

    # Captured before any real work starts, so it can be reused both as
    # setup_logging's timestamp (keeps the log filename and the bundle's
    # cutoff/filename in sync) and as the mtime cutoff for deciding which
    # model_dir/log_dir files belong to this run (see run_bundler.py).
    run_started = datetime.now(timezone.utc)

    # Provisional console-only setup, in case spec loading itself fails
    # (spec.log_level, needed for the real setup, isn't known yet).
    setup_logging(run_type="pipeline", to_file=False)
    logger = logging.getLogger("fw.main_pipeline")

    multiprocessing.set_start_method("spawn", force=True)

    from fw.scheduling.agent_manager import AgentManager
    from fw.configuration.config_profile import ProfileRegistry
    from fw.scheduling.pipeline_scheduler import PipelineScheduler
    from fw.scheduling.run_loader import load_pipeline_spec, PipelineSpecError
    from fw.configuration.env_factory import initialize_all_environments, resolve_environment_name, airsim_env_name

    parser = argparse.ArgumentParser(description="Run a JSON-defined training pipeline.")
    parser.add_argument("--spec", required=True, help="Path to the pipeline JSON file.")
    args = parser.parse_args()

    try:
        spec = load_pipeline_spec(args.spec)
    except PipelineSpecError as exc:
        logger.error("Invalid pipeline spec: %s", exc)
        raise SystemExit(1)

    src_dir = Path(__file__).resolve().parents[1]
    spec_name = Path(args.spec).stem
    config_py_path = Path(config_module.__file__)

    log_path = setup_logging(
        run_type="pipeline",
        env_name=spec_name,
        # main_pipeline.py runs a single agent_type for the whole pipeline
        # (see module docstring "Scope" section: mixing agent types across
        # blocks isn't supported), so it's safe to embed here directly
        # rather than resolve it per-block like fw.main does.
        agent_type=PPO_POLICY_NAME,
        src_dir=src_dir,
        level=spec.log_level,
        timestamp=run_started,
    )
    if log_path is not None:
        logger.info("Logging to file: %s", log_path)

    try:
        # Pre-flight check: reject any AirSim block *before* running anything,
        # not just when that block's turn comes up in the loop below. Without
        # this, an AirSim block sitting at position 3 in the run would only be
        # discovered after blocks 1-2 (potentially long-running) had already
        # executed. `run_loader` only validates env_type against the alias
        # table, not against this pipeline's supported-environment subset, so
        # that check can't happen there - it has to happen here, against every
        # block, before `initialize_all_environments()`.
        for block in spec.blocks:
            if resolve_environment_name(block.env_type) == airsim_env_name:
                logger.error(
                    "Block '%s' uses AirSimEnv, which is not supported by "
                    "main_pipeline.py (see pipeline_env.py's rejection of AirSim). "
                    "Use fw.main for AirSim training, or exclude this block.",
                    block.name,
                )
                raise SystemExit(1)

        initialize_all_environments()

        # NOTE: agent_type is fixed for the whole run (PPO by default). Every
        # block below overrides the AgentManager's *profile* and
        # *hyperparameters* via PipelineScheduler -> set_environment(...), so
        # the profile passed here only matters as a bootstrap value in case the
        # AgentManager needs *something* non-None to construct with. Mixing
        # agent types (e.g. a MultiSimEnv block requiring MPPO) within one
        # pipeline run is not supported yet - see pipeline_scheduler.py's
        # module docstring.
        bootstrap_profile = ProfileRegistry.get(spec.blocks[0].profile)
        agent_mgr = AgentManager(PPO_POLICY_NAME, bootstrap_profile, {})

        scheduler = PipelineScheduler(agent_mgr, spec.blocks, on_block_failure=spec.on_block_failure)
        metrics = scheduler.execute()

        logger.info("Pipeline metrics: %s", metrics)

        if any(m.get("status") == "failure" for m in metrics.values()):
            raise SystemExit(1)
    finally:
        # One bundle for the whole pipeline run (not per block) - see
        # run_bundler.py's module docstring. model_dir can in principle
        # differ per block via hyperparameter_overrides, so every block's
        # resolved model_dir is scanned; bundle_run_output de-duplicates
        # identical paths and namespaces distinct ones separately.
        #
        # All of this prep is its own try/except: bundle_run_output() already
        # can't raise, but building its arguments here (per-block
        # hyperparameter lookups, path resolution, the visualize glob,
        # dataclasses.asdict(spec)) happens in this frame, before that
        # protection applies - a bug here must not crash or mask whatever
        # actually happened in the try block above (including a SystemExit(1)
        # already in flight from a failed block).
        try:
            pipeline_model_dirs = []
            pipeline_policy_files = []
            for block in spec.blocks:
                block_hyperparameters = get_hyperparameters(
                    PPO_POLICY_NAME, env_name=resolve_environment_name(block.env_type)
                )
                block_hyperparameters.update(block.hyperparameter_overrides)
                block_model_dir_value = block_hyperparameters.get("model_dir", "models")
                pipeline_model_dirs.append(block_model_dir_value)

                if block.action == "evaluate_reward":
                    # export_evaluation_results() writes its CSV to
                    # Path("models") relative to the current working
                    # directory rather than through resolve_path_relative_to_src
                    # like everything else (see run_bundler.py's module
                    # docstring) - scan that too.
                    pipeline_model_dirs.append(str(Path.cwd() / "models"))

                if block.action == "visualize":
                    # visualize() only loads an existing policy checkpoint via
                    # PipelineScheduler._resolve_transfer()'s block-chaining
                    # rules - it never (re)writes one, so it wouldn't show up
                    # in the mtime scan on its own. Unlike fw.main (which
                    # reconstructs the exact filename directly, since its rule
                    # is just "visualize_best_trained flag + training_name"),
                    # this deliberately stays glob-only: _resolve_transfer()'s
                    # rules (transfer.external_training_name / transfer.from_block
                    # / block.name / last_train_block) are involved enough that
                    # reimplementing them here - outside the module that owns
                    # them - would be its own source of bugs for a marginal
                    # precision gain. Include every existing policy checkpoint
                    # for this agent_type sitting in the block's model_dir -
                    # broader than strictly necessary, but never misses the one
                    # actually in use.
                    block_model_dir = resolve_path_relative_to_src(block_model_dir_value, src_dir)
                    if block_model_dir.is_dir():
                        pipeline_policy_files.extend(
                            sorted(block_model_dir.glob(f"*_{PPO_POLICY_NAME}_policy.zip"))
                        )

            resolved_config_snapshot = dataclasses.asdict(spec)
        except Exception as exc:
            logger.warning("Could not prepare bundle metadata for this pipeline run: %s", exc)
            pipeline_model_dirs, pipeline_policy_files, resolved_config_snapshot = ["models"], [], None

        bundle_run_output(
            run_started=run_started,
            src_dir=src_dir,
            model_dirs=pipeline_model_dirs or "models",
            log_dirs="logs",
            input_files=[config_py_path, Path(args.spec)],
            resolved_config=resolved_config_snapshot,
            additional_model_files=pipeline_policy_files,
            env_name=spec_name,
            agent_type=PPO_POLICY_NAME,
            run_type="pipeline",
        )
