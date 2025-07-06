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
        initialize_all_environments
    )


    action = "train"

    # Ensure all environments are registered once
    initialize_all_environments()

    # Define simulator inputs
    envs = [get_environment(py_sim_env_name)]
            # get_environment(fh_sim_env_name)]

    # Create a Trainer and pass on a list of agents that you want to train
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    if "train" == action:
        configuration = {py_sim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=0.552, patience=4, max_time_steps=800),
                         fh_sim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=0.90, patience=3, max_time_steps=1200),
                         lunar_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD_AND_LOSS, target_reward=0.20, target_loss=0.8, patience=1, max_time_steps=300)}
        metrics = trainer.train(configuration)

        print(f"Training metrics: {metrics}")
    elif "visualize" == action:
        trainer.visualize(envs[0])
    else:
        print(f"Unable to execute unknown action '{action}'")
