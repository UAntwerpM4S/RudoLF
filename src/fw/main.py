# This file is the top-level runtime entry point for the framework.
#
# The guard intentionally accepts two execution modes:
#   1. `python fw/main.py` or `python -m fw.main`, where __name__ == "__main__".
#   2. `runpy.run_module("fw.main", run_name="fw.main")`, where a caller wants
#      to execute the same startup path from inside another Python process.
#
# All imports that may load NumPy/Torch/BLAS are kept inside the guard so the
# thread-limit environment variables below are applied before those libraries
# initialize their worker pools.
if __name__ == "__main__" or __name__ == "fw.main":
    import os

    # Limit BLAS threads before NumPy/Torch load to avoid subprocess
    # oversubscription.  This matters when the training stack creates
    # subprocesses: if each child process lets BLAS use all CPU cores, total
    # thread count can explode and slow training down instead of speeding it up.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import runpy
    import logging
    import multiprocessing
    import fw.configuration.config as config_module

    from pathlib import Path
    from datetime import datetime, timezone
    from fw.common.tools.run_bundler import bundle_run_output
    from fw.common.tools.path_utils import resolve_path_relative_to_src
    from fw.configuration.config import get_config_parameters, get_hyperparameters
    from fw.common.tools.logging_config import setup_logging, update_log_file_agent_type

    # _SRC_DIR points to the repository's src/ directory.  It is used whenever
    # output bundling or path resolution needs a stable base independent of the
    # current shell working directory.
    _SRC_DIR = Path(__file__).resolve().parents[1]

    # Keep the physical config.py path so run bundles can include the exact
    # configuration file that produced a training/visualization run.
    _CONFIG_INPUT_FILE = Path(config_module.__file__)

    # get_config_parameters() is the single source of truth for runtime
    # settings.  It starts from CONFIG_PARAMETERS, applies the selected profile,
    # and returns the resolved dict.  Downstream code should not apply the same
    # profile to this dict again, otherwise ship_model/dynamics_model style
    # settings can be applied twice.
    framework_config = get_config_parameters()

    # The framework supports four high-level actions:
    #   train           -> train policy/policies and optionally visualize.
    #   visualize       -> load an existing policy and generate trajectory plots.
    #   evaluate_reward -> train/evaluate several reward configurations.
    #   path_following  -> run the classical LOS/PID controller in
    #                      fw.control.path_following_los_pid.
    # Unknown values fall back to train to preserve historical behavior.
    action = str(framework_config.get("action", "train")).strip().lower()
    if action not in {"train", "visualize", "evaluate_reward", "path_following"}:
        action = "train"

    # env_type can be a string ("PySimEnv") or a sequence
    # (["PySimEnv", "FhSimEnv"]).  For logging, collapse the configured set into
    # a readable label before any environment instances exist.
    configured_env_names_for_log = framework_config.get("env_type", "PySimEnv")
    if isinstance(configured_env_names_for_log, list):
        env_name_for_log = "+".join(str(n) for n in configured_env_names_for_log) or "run"
    else:
        env_name_for_log = str(configured_env_names_for_log)

    log_path = setup_logging(
        run_type=action,
        env_name=env_name_for_log,
        agent_type="na",
        src_dir=_SRC_DIR,
        level=framework_config.get("log_level", "INFO"),
    )
    logger = logging.getLogger("fw.main")
    if log_path is not None:
        logger.info("Logging to file: %s", log_path)

    # Force spawn semantics so subprocesses start from a clean interpreter.
    # This is safer around Gym environments, PyTorch, FH Simulator bridges, and
    # GUI/plotting libraries than inheriting state through fork.
    multiprocessing.set_start_method("spawn", force=True)

    from fw.scheduling.trainer import Trainer
    from fw.configuration.profile_applier import apply_profile
    from fw.configuration.config_profile import ProfileRegistry
    from fw.configuration.stop_condition import StopCondition, Criterion
    from fw.configuration.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME, MULTI_POLICY_SINGLE_NAME
    from fw.configuration.env_factory import (
        build_default_environment,
        resolve_environment_name,
        py_sim_env_name,
        fh_sim_env_name,
        sync_sim_env_name,
        airsim_env_name,
        lunar_env_name,
        multi_sim_env_name,
        py_sim_colav_env_name,
        multi_policy_colav_env_name,
        resolve_environment_names,
        initialize_all_environments
    )

    # Retrieve the same profile object used by get_config_parameters().  At this
    # point the config dict has already received profile-level model overrides;
    # this profile object is retained for later side effects on live objects:
    # environment flags, simulator actuator limits, and action transformations.
    profile_name = framework_config.get("profile", "default_profile")
    profile = ProfileRegistry.get(profile_name)

    # reward_configs is only used by action == "evaluate_reward".  The normal
    # train/visualize paths ignore it.
    reward_configs = framework_config.get("reward_configs", [])

    # Resolve env_type aliases/classes into canonical environment names used by
    # env_factory.  This is where user-facing names such as "PySimEnv" become
    # the registry keys that build_default_environment() understands.
    configured_env_names = resolve_environment_names(
        framework_config.get("env_type", py_sim_env_name)
    )

    # Keep the raw action string for warning messages.  The normalized `action`
    # above is already safe for branching.
    _raw_action = str(framework_config.get("action", "train")).strip().lower()
    if _raw_action not in {"train", "visualize", "evaluate_reward", "path_following"}:
        logger.warning("Unknown action '%s'. Falling back to 'train'.", framework_config.get("action"))

    def _run_airsim_extension(selected_action: str) -> bool:
        """
        Run the AirSim entrypoint as an ordered step.

        Args:
            selected_action: Framework action to forward to the AirSim
                entrypoint ("train", "visualize", or "evaluate_reward").

        Returns:
            True when the AirSim step completed successfully. False when
            it failed and the caller should continue with remaining steps.
        """

        try:
            os.environ["FW_MAIN_ACTION_OVERRIDE"] = selected_action
            runpy.run_module("fw.main_with_airsim", run_name="__main__")
            return True
        except SystemExit as ex:
            if ex.code in (None, 0):
                return True
            logger.warning(
                "AirSim extension exited with an error; "
                "falling back to the default framework entrypoint."
            )
            return False
        except (ConnectionError, OSError) as ex:
            logger.warning(
                "AirSim extension failed to connect to UE5 (%s); "
                "falling back to the default framework entrypoint.", ex
            )
            return False
        finally:
            os.environ.pop("FW_MAIN_ACTION_OVERRIDE", None)

    # Stop conditions are keyed by canonical environment name.  Trainer passes
    # this dict to SimpleScheduler; SimpleScheduler selects the matching entry
    # when it switches to each environment.
    #
    # max_time_steps here is not the same thing as PySimEnv.max_steps.  This
    # object defines when policy training should stop, while the environment's
    # max_steps defines the maximum length of one episode.
    configuration = {
        py_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=0.99,
            patience=4,
            max_time_steps=850,
        ),
        multi_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=1.75,
            patience=4,
            max_time_steps=1300,
        ),
        airsim_env_name: StopCondition(
            stop_criterion=Criterion.NBR_POLICY_UPDATES,
            max_time_steps=350,
        ),
        fh_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=3.80,
            patience=4,
            max_time_steps=1300,
        ),
        sync_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=3.80,
            patience=4,
            max_time_steps=1300,
        ),
        lunar_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD_AND_LOSS,
            target_reward=0.20,
            target_loss=0.8,
            patience=1,
            max_time_steps=300,
        ),
        py_sim_colav_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=1.75,
            patience=4,
            max_time_steps=500,
        ),
        multi_policy_colav_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=5.0,
            patience=4,
            max_time_steps=1150
        ),
    }

    if action not in {"evaluate_reward", "path_following"} and not configured_env_names:
        logger.error("No environments configured for %s. Exiting.", action)
        raise SystemExit(1)

    if action == "path_following":
        # Classical path following is kept as a separate baseline module.
        # main.py only dispatches to that module here; it does not run the
        # framework visualize post-processing chain for this action.
        from fw.control.path_following import main as run_path_following

        run_path_following()
        raise SystemExit(0)

    # Import/register all environment wrapper classes before the first factory
    # lookup.  Environment classes register themselves via BaseEnvironment's
    # __init_subclass__, so this step makes the registry complete.
    initialize_all_environments()

    def _select_agent_type(envs: list) -> str:
        """
        Choose the correct agent type based on the environments.

        If any environment has `include_scenario_one_hot` enabled, use
        MPSA (Multi-Policy Single Agent). If any environment is a
        MultiSimEnv, use MPPO. Otherwise, default to single-policy PPO.

        Args:
            envs: List of framework environment wrappers.

        Returns:
            The agent type string (e.g. "PPO", "MPPO", or "MPSA").
        """

        # MultiPolicySingleAgent requires an observation suffix that identifies
        # the COLREGs scenario.  If any selected env exposes that one-hot flag,
        # use MPSA even if the batch contains only one environment.
        for sub_env in envs:
            if getattr(sub_env.gym_env, "include_scenario_one_hot", False):
                return MULTI_POLICY_SINGLE_NAME

        # MultiSimEnv wraps multiple sub-environments and therefore needs the
        # multi-agent PPO implementation.
        if any(e.gym_env.training_name == multi_sim_env_name for e in envs):
            return MULTI_PPO_POLICY_NAME

        # Standard single-environment PySim/FH/Lunar runs use the regular PPO
        # policy unless another rule above applies.
        return PPO_POLICY_NAME

    def _run_standard_batch(envs: list) -> bool:
        """
        Run one contiguous batch of framework environments.

        Args:
            envs: Framework environment instances collected from the
                ordered `env_type` sequence up to the next AirSim step.

        Returns:
            True when a framework train/visualize step was executed,
            False when `envs` is empty.
        """

        if not envs:
            return False

        # A "standard batch" is a contiguous group of non-AirSim environments in
        # the ordered env_type sequence.  AirSim is handled as a separate step
        # because it needs its own UE5/CosysAirSim startup and fallback logic.
        batch_started = datetime.now(timezone.utc)
        batch_env_name = "+".join(sorted({e.gym_env.training_name for e in envs})) or env_name_for_log
        agent_type = PPO_POLICY_NAME

        try:
            for sim_env in envs:
                # Apply profile settings to each live Gym environment.  This is
                # where options such as initial_speed, enable_smoothing, and
                # rudder-angle-scaling reach the environment object.
                apply_profile(profile, env=sim_env.gym_env)

                # If the environment already owns a physical simulator, apply
                # simulator-level profile overrides too.  This is separate from
                # config application because actuator limits live on mutable
                # ShipSpecifications inside the simulator, not in the config dict.
                sim = getattr(sim_env.gym_env, "phys_sim", None) or getattr(sim_env.gym_env, "_phys_sim", None)
                if sim is not None:
                    apply_profile(profile, simulator=sim)

            # Select PPO/MPPO/MPSA after environments are built because the
            # decision depends on attributes exposed by those environment
            # instances.
            agent_type = _select_agent_type(envs)
            updated_log_path = update_log_file_agent_type(agent_type)
            if updated_log_path is not None:
                logger.info("Log file renamed to: %s", updated_log_path)

            # Trainer owns the AgentManager and Scheduler.  main.py stops here:
            # it does not directly call policy.learn(), env.step(), or render().
            # Those responsibilities are delegated through Trainer.
            std_trainer = Trainer(agent_type, envs, profile, "SimpleScheduler", config=framework_config)
            if action == "visualize":
                std_trainer.visualize(envs, airsim_bridge=None)
            else:
                metrics = std_trainer.train(configuration)
                if action == "evaluate_reward":
                    logger.info("Training metrics (grouped): %s", metrics)
                else:
                    filtered_metrics = {
                        fields: {
                            "status": env_metrics.get("status"),
                            "duration": env_metrics.get("duration"),
                        }
                        for fields, env_metrics in metrics.items()
                    }

                    logger.info("Training metrics (grouped): %s", filtered_metrics)
        finally:
            try:
                # Bundle metadata and outputs even when training/visualization
                # raises, so partial runs still leave enough information for
                # debugging.  In visualize mode the already-existing policy zip
                # is included as an additional model file for reproducibility.
                batch_hyperparameters = get_hyperparameters(agent_type)
                model_dir_value = batch_hyperparameters.get("model_dir", "models")
                log_dir_value = batch_hyperparameters.get("log_dir", "logs")
                existing_policy_files = []
                if action == "visualize":
                    batch_model_dir = resolve_path_relative_to_src(str(model_dir_value), _SRC_DIR)
                    if batch_model_dir.is_dir():
                        existing_policy_files = sorted(batch_model_dir.glob(f"*_{agent_type}_policy.zip"))
            except Exception as ex:
                logger.warning("Could not prepare bundle metadata for this batch: %s", ex)
                model_dir_value, log_dir_value, existing_policy_files = "models", "logs", []

            bundle_run_output(
                run_started=batch_started,
                src_dir=_SRC_DIR,
                model_dirs=model_dir_value,
                log_dirs=log_dir_value,
                input_files=[_CONFIG_INPUT_FILE],
                resolved_config=framework_config,
                additional_model_files=existing_policy_files,
                env_name=batch_env_name,
                agent_type=agent_type,
                run_type=action,
            )
        return True

    # Tracks whether train/visualize/evaluate_reward actually executed at least
    # one unit of work.  This catches misconfigurations where every requested
    # environment is skipped or fails before startup.
    executed_any_step = False

    if action == "evaluate_reward":
        # Reward evaluation is a separate branch because it trains multiple
        # independent reward configurations and exports a comparison table,
        # rather than training one policy over the configured env_type sequence.
        if not reward_configs:
            logger.error("action='evaluate_reward' requires 'reward_configs' in configuration.")
            raise SystemExit(1)

        eval_started = datetime.now(timezone.utc)
        eval_env_name = "reward_eval"
        eval_agent_type = PPO_POLICY_NAME

        try:
            # Build one representative environment to create the appropriate
            # Trainer/AgentManager.  Trainer.evaluate_reward() then creates
            # isolated fresh environments for each reward configuration.
            first_env_type = reward_configs[0].get("type", py_sim_env_name)
            first_env_name = resolve_environment_name(first_env_type)
            eval_env_name = first_env_name
            dummy_env = build_default_environment(first_env_name, framework_config)
            eval_policy_name = MULTI_PPO_POLICY_NAME if first_env_name == multi_sim_env_name else PPO_POLICY_NAME
            eval_agent_type = eval_policy_name
            renamed_log_path = update_log_file_agent_type(eval_policy_name)
            if renamed_log_path is not None:
                logger.info("Log file renamed to: %s", renamed_log_path)
            trainer = Trainer(eval_policy_name, [dummy_env], profile, "SimpleScheduler", config=framework_config)

            logger.info("Starting reward evaluation for %d configurations...", len(reward_configs))
            results = trainer.evaluate_reward(reward_configs, configuration, verbose=True)

            # Reward-evaluation CSV is separate from the run bundle.  The bundle
            # still captures config/log/model artifacts; this CSV is the
            # user-facing comparison of reward settings.
            results_file = str(framework_config.get("reward_results_file", "reward_evaluation_results.csv"))
            trainer.export_evaluation_results(results, results_file)
            logger.info("Reward evaluation complete. Results exported to '%s'.", results_file)
            executed_any_step = True
            raise SystemExit(0)

        except Exception as exc:
            logger.exception("Reward evaluation failed: %s", exc)
            raise SystemExit(1)

        finally:
            try:
                # As above, always try to create a run bundle for traceability.
                # Reward evaluation can train temporary policies in either the
                # configured model_dir or cwd/models, so both are considered.
                eval_hyperparameters = get_hyperparameters(eval_agent_type)
                model_dir_values = [
                    eval_hyperparameters.get("model_dir", "models"),
                    str(Path.cwd() / "models"),
                ]
                log_dir_values = eval_hyperparameters.get("log_dir", "logs")
            except Exception as exc:
                logger.warning("Could not prepare bundle metadata for reward evaluation: %s", exc)
                model_dir_values = ["models", str(Path.cwd() / "models")]
                log_dir_values = "logs"

            bundle_run_output(
                run_started=eval_started,
                src_dir=_SRC_DIR,
                model_dirs=model_dir_values,
                log_dirs=log_dir_values,
                input_files=[_CONFIG_INPUT_FILE],
                resolved_config=framework_config,
                env_name=eval_env_name,
                agent_type=eval_agent_type,
                run_type=action,
            )

    # Build and execute environments in the order requested by config.py.  A
    # non-AirSim run becomes one standard batch.  A mixed run such as
    # ["PySimEnv", "AirSimEnv", "FhSimEnv"] becomes:
    #   1. standard batch: PySimEnv
    #   2. AirSim extension
    #   3. standard batch: FhSimEnv
    standard_envs = []

    for env_name in configured_env_names:
        if env_name == airsim_env_name:
            # Flush any pending standard environments before entering AirSim.
            # This preserves user-specified ordering and avoids mixing a UE5
            # session into the regular SimpleScheduler batch.
            if _run_standard_batch(standard_envs):
                executed_any_step = True
            standard_envs = []

            if action == "train":
                logging.getLogger("airsim").setLevel(logging.INFO)
            if _run_airsim_extension(action):
                executed_any_step = True
            continue

        # Build the environment wrapper from the resolved framework config.
        # For PySimEnv this eventually constructs PySimEnv(...), creates the
        # ShipSimulator, loads the configured map/trajectory CSV files, and
        # initializes observation/action spaces.
        env = build_default_environment(env_name, framework_config)
        standard_envs.append(env)

    # Execute any trailing non-AirSim environments after the loop.
    if _run_standard_batch(standard_envs):
        executed_any_step = True

    if not executed_any_step:
        logger.error("No environments remaining after AirSim fallback. Exiting.")
        raise SystemExit(1)
