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
    from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME
    from fw.env_factory import (
        get as get_environment,
        py_sim_env_name,
        fh_sim_env_name,
        airsim_env_name,
        lunar_env_name,
        py_sim_polar_env_name,
        initialize_all_environments
    )


    action = "train"

    # Ensure all environments are registered once
    initialize_all_environments()

    # Define simulator inputs
    envs = [get_environment(py_sim_env_name)]
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
        # Optionally create AirSim bridge for simultaneous vehicle control
        from airsim.config.airsim_config import get_airsim_config, AirSimConfig
        airsim_cfg = get_airsim_config()
        airsim_bridge = None
        if airsim_cfg.get("use_airsim", False):
            from airsim.communication.airsim_bridge import AirSimBridge
            bridge_cfg = AirSimConfig(
                ip=airsim_cfg.get("airsim_ip", ""),
                port=airsim_cfg.get("airsim_port", 41451)
            )
            temp_bridge = AirSimBridge(bridge_cfg)
            if temp_bridge.is_connected:
                airsim_bridge = temp_bridge
                logging.getLogger("airsim").setLevel(logging.INFO)  # connected — enable airsim logs
            else:
                print("[main] AirSim bridge not connected — running without.")

        trainer.visualize(envs, airsim_bridge=airsim_bridge)
    else:
        print(f"Unable to execute unknown action '{action}'")
