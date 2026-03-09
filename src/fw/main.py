if __name__ == "__main__" or __name__ == "fw.main":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    from fw.trainer import Trainer
    from fw.stop_condition import StopCondition, Criterion
    from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME
    from fw.env_factory import (
        get as get_environment,
        py_sim_env_name,
        fh_sim_env_name,
        lunar_env_name,
        py_sim_polar_env_name,
        py_sim_colav_env_name,
        initialize_all_environments
    )

    action = "visualize"

    # Ensure all environments are registered once
    initialize_all_environments()

    # Define simulator inputs
    envs = [get_environment(py_sim_colav_env_name)]
            # get_environment(fh_sim_env_name)]

    # Create a Trainer and pass on a list of agents that you want to train
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    if "train" == action:
        configuration = {py_sim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=100, patience=20, max_time_steps=1500),
                         fh_sim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=3.80, patience=4, max_time_steps=1300),
                         lunar_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD_AND_LOSS, target_reward=0.20, target_loss=0.8, patience=1, max_time_steps=300),
                         py_sim_polar_env_name: StopCondition(stop_criterion=Criterion.NBR_POLICY_UPDATES, max_time_steps=600),
                         py_sim_colav_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=3.9, patience=4, max_time_steps=1500)}
        metrics = trainer.train(configuration)

        print(f"Training metrics: {metrics}")
    elif "visualize" == action:
        # Optionally create AirSim bridge for simultaneous vehicle control
        from fw.config import get_config_parameters
        config = get_config_parameters()
        airsim_bridge = None
        if config.get("use_airsim", False):
            from fw.simulators.airsim_bridge import AirSimBridge
            airsim_bridge = AirSimBridge(
                ip=config.get("airsim_ip", ""),
                port=config.get("airsim_port", 41451),
                heading_kp=config.get("airsim_kp", 5.0),
                heading_ki=config.get("airsim_ki", 0.5),
                heading_ff=config.get("airsim_ff", 0.5),
                speed_scale=config.get("airsim_speed_scale", 1.0),
                throttle_kp=config.get("airsim_throttle_kp", 1.0),
                throttle_ki=config.get("airsim_throttle_ki", 0.05),
                dist_gain=config.get("airsim_dist_gain", 0.1),
                dist_ki=config.get("airsim_dist_ki", 0.0), # Default 0 if missing
                log_telemetry=config.get("airsim_log_telemetry", True),
            )

        trainer.visualize(envs, airsim_bridge=airsim_bridge)
    else:
        print(f"Unable to execute unknown action '{action}'")