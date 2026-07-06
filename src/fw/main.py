if __name__ == "__main__" or __name__ == "fw.main":
    # MUST run before any import that pulls in torch/numpy, since both
    # BLAS thread pools read these once at library load time. With
    # SubprocVecEnv (num_envs=12) each child process inherits the env
    # and respects the same single-thread setting, eliminating the
    # 12-subproc x N-thread oversubscription that destabilizes PPO on
    # CPUs with high core counts
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import logging
    import multiprocessing

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # Suppress all airsim loggers by default; they are re-enabled once a live
    # connection is confirmed (see visualize / train blocks below).
    logging.getLogger("airsim").setLevel(logging.CRITICAL)
    logging.getLogger("sim").setLevel(logging.CRITICAL)


    multiprocessing.set_start_method("spawn", force=True)

    import os
    import runpy

    from fw.trainer import Trainer
    from fw.profile_applier import apply_profile
    from fw.config_profile import ProfileRegistry
    from fw.stop_condition import StopCondition, Criterion
    from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME, MULTI_POLICY_SINGLE_NAME, get_config_parameters
    from fw.env_factory import (
        build_default_environment,
        resolve_environment_name,
        py_sim_env_name,
        fh_sim_env_name,
        sync_sim_env_name,
        airsim_env_name,
        lunar_env_name,
        py_sim_polar_env_name,
        multi_sim_env_name,
        py_sim_colav_env_name,
        multi_policy_colav_env_name,
        resolve_environment_names,
        initialize_all_environments
    )

    framework_config = get_config_parameters()

    # retrieve the profile object here only for the later apply_profile(env/sim) calls.
    profile_name = framework_config.get("profile", "default_profile")
    profile = ProfileRegistry.get(profile_name)

    reward_configs = framework_config.get("reward_configs", [])
    configured_env_names = resolve_environment_names(
        framework_config.get("env_type", py_sim_env_name)
    )
    action = str(framework_config.get("action", "train")).strip().lower()
    if action not in {"train", "visualize", "evaluate_reward"}:
        print(f"[main] Unknown action '{action}'. Falling back to 'train'.")
        action = "train"

    def _run_airsim_extension(selected_action: str) -> bool:
        """
        Run the AirSim entrypoint as an ordered step.

        Args:
            selected_action: Framework action to forward to the AirSim
                entrypoint ("train" or "visualize" or "evaluate_reward").

        Returns:
            True when the AirSim step completed successfully. False when the
            AirSim step failed and the caller should continue with remaining
            framework steps.
        """

        try:
            os.environ["FW_MAIN_ACTION_OVERRIDE"] = selected_action
            runpy.run_module("fw.main_with_airsim", run_name="__main__")
            return True
        except SystemExit as ex:
            if ex.code in (None, 0):
                return True
            print(
                "[main] AirSim extension exited with an error; "
                "falling back to the default framework entrypoint."
            )
            return False
        except (ConnectionError, OSError) as ex:
            print(
                f"[main] AirSim extension failed to connect to UE5 ({ex}); "
                "falling back to the default framework entrypoint."
            )
            return False
        finally:
            os.environ.pop("FW_MAIN_ACTION_OVERRIDE", None)

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
        py_sim_polar_env_name: StopCondition(
            stop_criterion=Criterion.NBR_POLICY_UPDATES,
            max_time_steps=600,
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
        print(f"[main] No environments configured for {action}. Exiting.")
        raise SystemExit(1)

    initialize_all_environments()

    def _select_agent_type(envs: list) -> str:
        """
        Choose the correct agent type based on the environments.

        If any environment has ``include_scenario_one_hot`` enabled, use
        MPSA (Multi-Policy Single Agent).  If any environment is a
        MultiSimEnv, use MPPO.  Otherwise, default to single-policy PPO.

        Args:
            envs: List of framework environment wrappers.

        Returns:
            The agent type string (e.g. "PPO", "MPPO", or "MPSA").
        """

        for sub_env in envs:
            if getattr(sub_env.gym_env, "include_scenario_one_hot", False):
                return MULTI_POLICY_SINGLE_NAME

        # MultiSimEnv requires its own policy type; all other envs use PPO.
        # type_name equals ENVIRONMENT_NAME (e.g. "MultiSimEnv").
        if any(e.gym_env.type_name == multi_sim_env_name for e in envs):
            return MULTI_PPO_POLICY_NAME

        return PPO_POLICY_NAME

    def _run_standard_batch(envs: list) -> bool:
        """
        Run one contiguous batch of framework environments.

        Args:
            envs: Framework environment instances collected from the ordered
                `env_type` sequence up to the next AirSim step boundary.

        Returns:
            True when a framework train/visualize step was executed, otherwise
            False when `envs` is empty.
        """

        if not envs:
            return False

        for sim_env in envs:
            apply_profile(profile, env=sim_env.gym_env)
            sim = getattr(sim_env.gym_env, "phys_sim", None) or getattr(sim_env.gym_env, "_phys_sim", None)
            if sim is not None:
                apply_profile(profile, simulator=sim)

        agent_type = _select_agent_type(envs)
        std_trainer = Trainer(agent_type, envs, profile, "SimpleScheduler", config=framework_config)
        if action == "visualize":
            std_trainer.visualize(envs, airsim_bridge=None)
        else:
            metrics = std_trainer.train(configuration)
            if action == "evaluate_reward":
                print(f"Training metrics (grouped): {metrics}")
            else:
                filtered_metrics = {
                    fields: {
                        "status": env_metrics.get("status"),
                        "duration": env_metrics.get("duration"),
                    }
                    for fields, env_metrics in metrics.items()
                }

                print(f"Training metrics (grouped): {filtered_metrics}")
        return True

    # Both visualize and train now follow env_type order exactly (for airsim),
    # but standard environments are batched to preserve Trainer functionality.
    executed_any_step = False

    if action == "evaluate_reward":
        if not reward_configs:
            print("[main] action='evaluate_reward' requires 'reward_configs' in configuration.")
            raise SystemExit(1)

        try:
            first_env_type = reward_configs[0].get("type", py_sim_env_name)
            first_env_name = resolve_environment_name(first_env_type)
            dummy_env = build_default_environment(first_env_name, framework_config)
            eval_policy_name = MULTI_PPO_POLICY_NAME if first_env_name == multi_sim_env_name else PPO_POLICY_NAME
            trainer = Trainer(eval_policy_name, [dummy_env], profile, "SimpleScheduler", config=framework_config)

            print(f"[main] Starting reward evaluation for {len(reward_configs)} configurations...")
            results = trainer.evaluate_reward(reward_configs, configuration, verbose=True)

            results_file = str(framework_config.get("reward_results_file", "reward_evaluation_results.csv"))
            trainer.export_evaluation_results(results, results_file)
            print(f"[main] Reward evaluation complete. Results exported to '{results_file}'.")
            executed_any_step = True
            # evaluate_reward is a standalone mode; avoid running standard train/visualize afterward.
            raise SystemExit(0)
        except Exception as exc:
            print(f"[main] Reward evaluation failed: {exc}")
            raise SystemExit(1)

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
        print("[main] No environments remaining after AirSim fallback. Exiting.")
        raise SystemExit(1)
