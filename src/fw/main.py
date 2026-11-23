if __name__ == "__main__" or __name__ == "fw.main":
    import os

    # Limit BLAS threads before NumPy/Torch load to avoid subprocess oversubscription.
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

    _SRC_DIR = Path(__file__).resolve().parents[1]
    _CONFIG_INPUT_FILE = Path(config_module.__file__)

    framework_config = get_config_parameters()

    action = str(framework_config.get("action", "train")).strip().lower()
    if action not in {"train", "visualize", "evaluate_reward"}:
        action = "train"

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

    # Profile object for the apply_profile(env/sim) calls below.
    profile_name = framework_config.get("profile", "default_profile")
    profile = ProfileRegistry.get(profile_name)

    reward_configs = framework_config.get("reward_configs", [])
    configured_env_names = resolve_environment_names(
        framework_config.get("env_type", py_sim_env_name)
    )
    _raw_action = str(framework_config.get("action", "train")).strip().lower()
    if _raw_action not in {"train", "visualize", "evaluate_reward"}:
        logger.warning("Unknown action '%s'. Falling back to 'train'.", framework_config.get("action"))

    def _run_airsim_extension(selected_action: str) -> bool:
        """
        Run fw.main_with_airsim as one step of the ordered env_type sequence.

        Args:
            selected_action: "train", "visualize" or "evaluate_reward".

        Returns:
            True on success; False on failure (the caller continues with the
            remaining steps).
        """

        # main_with_airsim adds a file handler to the root logger and lowers its
        # level on every run; snapshot both so they can be restored afterward.
        _root_logger = logging.getLogger()
        _handlers_before = list(_root_logger.handlers)
        _level_before = _root_logger.level

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
        except Exception:
            # Deliberate exception to fail-fast: a failing AirSim step is
            # skipped and the remaining environments still run.
            logger.exception(
                "AirSim step failed; skipping it and continuing with the remaining environments."
            )
            return False
        finally:
            os.environ.pop("FW_MAIN_ACTION_OVERRIDE", None)
            for _handler in list(_root_logger.handlers):
                if _handler not in _handlers_before:
                    _root_logger.removeHandler(_handler)
                    try:
                        _handler.close()
                    except Exception:
                        pass
            _root_logger.setLevel(_level_before)

    configuration = {
        py_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=0.99,
            patience=4,
            max_time_steps=1300,
        ),
        # MultiPPOAgent judges each agent on its own reward, hence the same
        # target as single-agent PySimEnv.
        multi_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=0.99,
            patience=4,
            max_time_steps=850,
        ),
        airsim_env_name: StopCondition(
            stop_criterion=Criterion.NBR_POLICY_UPDATES,
            max_time_steps=350,
        ),
        fh_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=0.99,
            patience=4,
            max_time_steps=1350,
        ),
        sync_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=0.99,
            patience=4,
            max_time_steps=1350,
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

    if action != "evaluate_reward" and not configured_env_names:
        logger.error("No environments configured for %s. Exiting.", action)
        raise SystemExit(1)

    initialize_all_environments()

    def _select_agent_type(envs: list) -> str:
        """
        Pick the agent type: MPSA if any env has include_scenario_one_hot, MPPO if
        any env is a MultiSimEnv, otherwise PPO.
        """

        for sub_env in envs:
            if getattr(sub_env.gym_env, "include_scenario_one_hot", False):
                return MULTI_POLICY_SINGLE_NAME

        if any(e.gym_env.training_name == multi_sim_env_name for e in envs):
            return MULTI_PPO_POLICY_NAME

        return PPO_POLICY_NAME

    def _run_standard_batch(envs: list) -> bool:
        """
        Train or visualize one batch of non-AirSim environments (those between
        AirSim steps in env_type), then bundle the run output.

        Returns:
            True if a step was executed, False if envs is empty.
        """

        if not envs:
            return False

        batch_started = datetime.now(timezone.utc)
        batch_env_name = "+".join(sorted({e.gym_env.training_name for e in envs})) or env_name_for_log
        agent_type = PPO_POLICY_NAME

        try:
            for sim_env in envs:
                apply_profile(profile, env=sim_env.gym_env)
                sim = getattr(sim_env.gym_env, "phys_sim", None) or getattr(sim_env.gym_env, "_phys_sim", None)
                if sim is not None:
                    apply_profile(profile, simulator=sim)

            agent_type = _select_agent_type(envs)
            updated_log_path = update_log_file_agent_type(agent_type)
            if updated_log_path is not None:
                logger.info("Log file renamed to: %s", updated_log_path)

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

    executed_any_step = False

    if action == "evaluate_reward":
        if not reward_configs:
            logger.error("action='evaluate_reward' requires 'reward_configs' in configuration.")
            raise SystemExit(1)

        eval_started = datetime.now(timezone.utc)
        eval_env_name = "reward_eval"
        eval_agent_type = PPO_POLICY_NAME

        try:
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

    standard_envs = []

    for env_name in configured_env_names:
        if env_name == airsim_env_name:
            if _run_standard_batch(standard_envs):
                executed_any_step = True
            standard_envs = []
            if action == "train":
                logging.getLogger("airsim").setLevel(logging.INFO)
            if _run_airsim_extension(action):
                executed_any_step = True
            continue

        env = build_default_environment(env_name, framework_config)
        standard_envs.append(env)

    if _run_standard_batch(standard_envs):
        executed_any_step = True

    if not executed_any_step:
        logger.error("No environments remaining after AirSim fallback. Exiting.")
        raise SystemExit(1)
