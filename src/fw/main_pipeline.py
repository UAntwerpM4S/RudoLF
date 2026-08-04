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
    import multiprocessing

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    multiprocessing.set_start_method("spawn", force=True)

    from fw.configuration.config import PPO_POLICY_NAME
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
        print(f"[main_pipeline] Invalid pipeline spec: {exc}")
        raise SystemExit(1)

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
            print(
                f"[main_pipeline] Block '{block.name}' uses AirSimEnv, which is not "
                f"supported by main_pipeline.py (see pipeline_env.py's rejection of "
                f"AirSim). Use fw.main for AirSim training, or exclude this block."
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

    print(f"\n[main_pipeline] Pipeline metrics: {metrics}")

    if any(m.get("status") == "failure" for m in metrics.values()):
        raise SystemExit(1)
