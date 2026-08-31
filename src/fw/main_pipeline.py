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

    from pathlib import Path
    from datetime import datetime, timezone
    from fw.common.tools.logging_config import setup_logging
    from fw.common.tools.run_bundler import bundle_run_output
    from fw.common.tools.path_utils import resolve_path_relative_to_src
    from fw.configuration.config import PPO_POLICY_NAME, get_hyperparameters

    _SRC_DIR = Path(__file__).resolve().parents[1]
    # Captured before any file (log, model, ...) is written, so mtime-based
    # bundling below reliably picks up everything this run produces.
    _RUN_STARTED = datetime.now(timezone.utc)

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

    log_path = setup_logging(
        run_type="pipeline",
        env_name=Path(args.spec).stem,
        # main_pipeline.py runs a single agent_type for the whole pipeline
        # (see module docstring "Scope" section: mixing agent types across
        # blocks isn't supported), so it's safe to embed here directly
        # rather than resolve it per-block like fw.main does.
        agent_type=PPO_POLICY_NAME,
        src_dir=Path(__file__).resolve().parents[1],
        level=spec.log_level,
    )
    if log_path is not None:
        logger.info("Logging to file: %s", log_path)

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

    try:
        metrics = scheduler.execute()

        logger.info("Pipeline metrics: %s", metrics)

        if any(m.get("status") == "failure" for m in metrics.values()):
            raise SystemExit(1)
    finally:
        # One bundle for the whole pipeline run (not per block), regardless
        # of whether it succeeded - see run_bundler.py's module docstring.
        # model_dir can in principle differ per block via
        # hyperparameter_overrides, so every block's resolved model_dir is
        # scanned; bundle_run_output de-duplicates identical paths.
        _pipeline_model_dirs = []
        for _block in spec.blocks:
            _block_hyperparameters = get_hyperparameters(
                PPO_POLICY_NAME, env_name=resolve_environment_name(_block.env_type)
            )
            _block_hyperparameters.update(_block.hyperparameter_overrides)
            _pipeline_model_dirs.append(_block_hyperparameters.get("model_dir", "models"))

        if any(_block.action == "evaluate_reward" for _block in spec.blocks):
            # Same CWD-relative-path issue as in main.py: any evaluate_reward
            # block's Trainer.export_evaluation_results() call (see
            # pipeline_scheduler.py) writes relative to the current working
            # directory rather than resolving against src_dir. Scanning
            # both covers it regardless of launch directory; de-duplicated
            # when CWD == src_dir.
            _pipeline_model_dirs.append(str(Path.cwd() / "models"))

        # A "visualize" block only *reads* an existing policy checkpoint -
        # it never writes to model_dir - so that checkpoint predates
        # _RUN_STARTED and would be silently skipped by bundle_run_output's
        # mtime-based scan (same rationale as main.py's standalone
        # 'visualize' action). Resolve the expected path per block (using
        # each block's own env_type/hyperparameter_overrides) and pass it
        # through explicitly. For a "visualize" block chained after a
        # "train" block earlier in this same pipeline run, the checkpoint
        # will typically already be caught by the mtime scan above;
        # bundle_run_output dedupes so listing it again here is harmless.
        _pipeline_policy_files = []
        for _block in spec.blocks:
            if _block.action != "visualize":
                continue
            _visualize_training_name = resolve_environment_name(_block.env_type)
            _block_hyperparameters = get_hyperparameters(
                PPO_POLICY_NAME, env_name=_visualize_training_name
            )
            _block_hyperparameters.update(_block.hyperparameter_overrides)
            _block_model_dir = resolve_path_relative_to_src(
                str(_block_hyperparameters.get("model_dir", "models")), _SRC_DIR
            )
            _pipeline_policy_files.append(
                _block_model_dir / f"{_visualize_training_name}_{PPO_POLICY_NAME}_policy.zip"
            )

        bundle_run_output(
            run_started=_RUN_STARTED,
            src_dir=_SRC_DIR,
            model_dirs=_pipeline_model_dirs or "models",
            log_dirs="logs",
            input_files=[Path(args.spec)],
            resolved_config=dataclasses.asdict(spec),
            env_name=Path(args.spec).stem,
            agent_type=PPO_POLICY_NAME,
            run_type="pipeline",
            additional_model_files=_pipeline_policy_files,
        )
