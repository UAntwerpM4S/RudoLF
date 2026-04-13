if __name__ == "__main__" or __name__ == "fw.main":
    import logging
    import multiprocessing

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # Suppress all airsim loggers by de
    # fault; they are re-enabled once a live
    # connection is confirmed (see visualize / train blocks below).
    logging.getLogger("airsim").setLevel(logging.CRITICAL)
    logging.getLogger("sim").setLevel(logging.CRITICAL)


    multiprocessing.set_start_method("spawn", force=True)

    from fw.trainer import Trainer

    from fw.stop_condition import StopCondition, Criterion
    from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, get_config_parameters
    from fw.env_factory import (
        build_default_environment,
        get as get_environment,
        py_sim_env_name,
        fh_sim_env_name,
        airsim_env_name,
        lunar_env_name,
        py_sim_polar_env_name,
        resolve_environment_names,
        py_sim_colav_env_name,
        initialize_all_environments
    )

    framework_config = get_config_parameters()
    configured_env_names = resolve_environment_names(
        framework_config.get("env_type", py_sim_colav_env_name)
    )
    action = str(framework_config.get("action", "train")).strip().lower()
    if action not in {"train", "visualize"}:
        print(f"[main] Unknown action '{action}'. Falling back to 'train'.")
        action = "train"

    def _should_delegate_to_airsim_extension() -> bool:
        """Return whether any of the selected environments is AirSim."""
        return airsim_env_name in configured_env_names

    if _should_delegate_to_airsim_extension():
        import os
        import runpy

        try:
            os.environ["FW_MAIN_ACTION_OVERRIDE"] = action
            runpy.run_module("fw.main_with_airsim", run_name="__main__")
            raise SystemExit(0)
        except SystemExit as exc:
            if exc.code in (None, 0):
                raise
            print(
                "[main] AirSim extension exited with an error; "
                "falling back to the default framework entrypoint."
            )
            configured_env_names[:] = [n for n in configured_env_names if n != airsim_env_name]
        except (ConnectionError, OSError) as exc:
            print(
                f"[main] AirSim extension failed to connect to UE5 ({exc}); "
                "falling back to the default framework entrypoint."
            )
            configured_env_names[:] = [n for n in configured_env_names if n != airsim_env_name]
        finally:
            os.environ.pop("FW_MAIN_ACTION_OVERRIDE", None)

    if not configured_env_names:
        print("[main] No environments remaining after AirSim fallback. Exiting.")
        raise SystemExit(1)
    # Ensure all environments are registered once
    initialize_all_environments()

    # Define simulator inputs. Builds one environment per configured env_type entry.
    envs = [build_default_environment(name) for name in configured_env_names]

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
                         py_sim_polar_env_name: StopCondition(stop_criterion=Criterion.NBR_POLICY_UPDATES, max_time_steps=600),
                         py_sim_colav_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=0.8, patience=4, max_time_steps=1500)}
        metrics = trainer.train(configuration)

        print(f"Training metrics: {metrics}")
    elif "visualize" == action:
        trainer.visualize(envs, airsim_bridge=None)
    else:
        print(f"Unable to execute unknown action '{action}'")