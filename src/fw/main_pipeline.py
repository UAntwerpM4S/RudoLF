"""
Run a JSON-defined training, visualization, or reward-evaluation pipeline.
"""

if __name__ == "__main__" or __name__ == "fw.main_pipeline":
    import os

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


    run_started = datetime.now(timezone.utc)
    setup_logging(run_type="pipeline", to_file=False)
    logger = logging.getLogger("fw.main_pipeline")

    multiprocessing.set_start_method("spawn", force=True)

    from fw.scheduling.agent_manager import AgentManager
    from fw.configuration.config_profile import ProfileRegistry
    from fw.scheduling.pipeline_scheduler import PipelineScheduler
    from fw.scheduling.run_loader import load_pipeline_spec, PipelineSpecError
    from fw.configuration.env_factory import multi_sim_env_name, multi_policy_colav_env_name
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
        agent_type=PPO_POLICY_NAME,
        src_dir=src_dir,
        level=spec.log_level,
        timestamp=run_started,
    )
    if log_path is not None:
        logger.info("Logging to file: %s", log_path)

    try:
        for block in spec.blocks:
            if resolve_environment_name(block.env_type) == airsim_env_name:
                logger.error(
                    "Block '%s' uses AirSimEnv, which is not supported by "
                    "main_pipeline.py (see pipeline_env.py's rejection of AirSim). "
                    "Use fw.main for AirSim training, or exclude this block.",
                    block.name,
                )
                raise SystemExit(1)

            # The pipeline only runs single-policy PPO. MultiSimEnv (needs MPPO)
            # would crash and MultiPolicyColavEnv (needs MPSA) would silently
            # train one policy on the one-hot observation, so refuse both.
            if resolve_environment_name(block.env_type) in (multi_sim_env_name, multi_policy_colav_env_name):
                logger.error(
                    "Block '%s' uses %s, which needs a multi-policy agent (MPPO/MPSA); "
                    "main_pipeline.py only runs the single-policy PPO agent. Use fw.main for it.",
                    block.name, resolve_environment_name(block.env_type),
                )
                raise SystemExit(1)

        initialize_all_environments()
        bootstrap_profile = ProfileRegistry.get(spec.blocks[0].profile)
        agent_mgr = AgentManager(PPO_POLICY_NAME, bootstrap_profile, {})

        scheduler = PipelineScheduler(agent_mgr, spec.blocks, on_block_failure=spec.on_block_failure)
        metrics = scheduler.execute()

        logger.info("Pipeline metrics: %s", metrics)

        if any(m.get("status") == "failure" for m in metrics.values()):
            raise SystemExit(1)
    finally:
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
                    pipeline_model_dirs.append(str(Path.cwd() / "models"))

                if block.action == "visualize":
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
