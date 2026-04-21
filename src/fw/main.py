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

    import os
    import runpy

    from fw.trainer import Trainer
    from fw.stop_condition import StopCondition, Criterion
    from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME, MULTI_POLICY_SINGLE_NAME, get_config_parameters
    from fw.env_factory import (
        build_default_environment,
        py_sim_env_name,
        fh_sim_env_name,
        airsim_env_name,
        lunar_env_name,
        py_sim_polar_env_name,
        multi_sim_env_name,
        multi_policy_colav_env_name,
        py_sim_colav_env_name,
        resolve_environment_names,
        initialize_all_environments
    )

    framework_config = get_config_parameters()
    configured_env_names = resolve_environment_names(
        framework_config.get("env_type", py_sim_env_name)
    )
    action = str(framework_config.get("action", "train")).strip().lower()
    if action not in {"train", "visualize"}:
        print(f"[main] Unknown action '{action}'. Falling back to 'train'.")
        action = "train"

    def _run_airsim_extension(selected_action: str) -> bool:
        """
        Run the AirSim entrypoint as an ordered step.

        Args:
            selected_action: Framework action to forward to the AirSim
                entrypoint ("train" or "visualize").

        Returns:
            True when the AirSim step completed successfully. False when the
            AirSim step failed and the caller should continue with remaining
            framework steps.
        """

        try:
            os.environ["FW_MAIN_ACTION_OVERRIDE"] = selected_action
            runpy.run_module("fw.main_with_airsim", run_name="__main__")
            return True
        except SystemExit as exc:
            if exc.code in (None, 0):
                return True
            print(
                "[main] AirSim extension exited with an error; "
                "falling back to the default framework entrypoint."
            )
            return False
        except (ConnectionError, OSError) as exc:
            print(
                f"[main] AirSim extension failed to connect to UE5 ({exc}); "
                "falling back to the default framework entrypoint."
            )
            return False
        finally:
            os.environ.pop("FW_MAIN_ACTION_OVERRIDE", None)

    configuration = {
        py_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=1.75,
            patience=4,
            max_time_steps=1300,
        ),
        multi_sim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=1.75,
            patience=4,
            max_time_steps=1300,
        ),
        airsim_env_name: StopCondition(
            stop_criterion=Criterion.TARGET_REWARD,
            target_reward=1.75,
            patience=4,
            max_time_steps=1300,
        ),
        fh_sim_env_name: StopCondition(
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
            target_reward=0.75,
            patience=4,
            max_time_steps=200

        ),
    }

    if not configured_env_names:
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
        from fw.environments.multi_sim_env import MultiSimEnv
        for env in envs:
            if getattr(env.gym_env, "include_scenario_one_hot", False):
                return MULTI_POLICY_SINGLE_NAME
            if isinstance(env.gym_env, MultiSimEnv):
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

        agent_type = _select_agent_type(envs)
        trainer = Trainer(agent_type, envs, "SimpleScheduler")
        if action == "visualize":
            trainer.visualize(envs, airsim_bridge=None)
        else:
            metrics = trainer.train(configuration)
            print(f"Training metrics (grouped): {metrics}")
        return True

    # Both visualize and train now follow env_type order exactly (for airsim),
    # but standard environments are batched to preserve Trainer functionality.
    executed_any_step = False
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

        env = build_default_environment(env_name)
        standard_envs.append(env)

    if _run_standard_batch(standard_envs):
        executed_any_step = True

    if not executed_any_step:
        print("[main] No environments remaining after AirSim fallback. Exiting.")
        raise SystemExit(1)