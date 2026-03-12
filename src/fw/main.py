if __name__ == "__main__" or __name__ == "fw.main":
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

    from fw.trainer import Trainer
    from fw.stop_condition import StopCondition, Criterion
    from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, get_config_parameters
    from airsim.config.airsim_config import get_runtime_settings
    from fw.env_factory import (
        build_default_environment,
        get as get_environment,
        py_sim_env_name,
        fh_sim_env_name,
        airsim_env_name,
        lunar_env_name,
        py_sim_polar_env_name,
        resolve_environment_name,
        initialize_all_environments
    )

    def _should_delegate_to_airsim_extension() -> bool:
        """Return whether runtime settings explicitly select the AirSim entrypoint."""
        try:
            runtime_settings = get_runtime_settings()
        except Exception:
            return False

        if not isinstance(runtime_settings, dict):
            return False

        main_cfg = runtime_settings.get("main", {})
        selected_action = str(main_cfg.get("action", "")).strip().lower()
        selected_environment = str(main_cfg.get("train_environment", "")).strip()
        return (
            selected_environment == airsim_env_name
            and selected_action in {"train", "visualize"}
        )

    if _should_delegate_to_airsim_extension():
        import runpy

        try:
            runpy.run_module("fw.main_with_airsim", run_name="__main__")
            raise SystemExit(0)
        except SystemExit as exc:
            if exc.code in (None, 0):
                raise
            print(
                "[main] AirSim extension exited with an error; "
                "falling back to the default framework entrypoint."
            )

    action = "train"

    # Ensure all environments are registered once
    initialize_all_environments()

    # Define simulator inputs. Keep hardcoded behavior by default, allow runtime override.
    configured_env_name = resolve_environment_name(
        get_config_parameters().get("env_type", py_sim_env_name)
    )
    envs = [build_default_environment(configured_env_name)]
            # get_environment(fh_sim_env_name)]
            # [get_environment(airsim_env_name, configuration={"training_mode": False})]

    # Create a Trainer and pass on a list of agents that you want to train
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    if "train" == action:
        # Re-enable airsim logging only when the airsim environment is selected
        if any(env.__class__.__name__ == airsim_env_name for env in envs):
            logging.getLogger("airsim").setLevel(logging.INFO)

        configuration = {py_sim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=1.75, patience=4, max_time_steps=1300),
                         airsim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=1.75, patience=4, max_time_steps=1300),
                         fh_sim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=3.80, patience=4, max_time_steps=1300),
                         lunar_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD_AND_LOSS, target_reward=0.20, target_loss=0.8, patience=1, max_time_steps=300),
                         py_sim_polar_env_name: StopCondition(stop_criterion=Criterion.NBR_POLICY_UPDATES, max_time_steps=600)}
        metrics = trainer.train(configuration)

        print(f"Training metrics: {metrics}")
    elif "visualize" == action:
        trainer.visualize(envs, airsim_bridge=None)
    else:
        print(f"Unable to execute unknown action '{action}'")