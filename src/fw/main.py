from fw.trainer import Trainer
from fw.env_factory import get as get_environment
from fw.stop_condition import StopCondition, Criterion
from fw.config import PPO_POLICY_NAME, SB3_PPO_POLICY_NAME
from fw.env_factory import py_sim_env_name, fh_sim_env_name, lunar_env_name


if __name__ == "__main__" or __name__ == "fw.main":
    action = "train"

    # Define simulator inputs
    envs = [get_environment(py_sim_env_name)]
            # get_environment(fh_sim_env_name)]

    # Create a Trainer and pass on a list of agents that you want to train
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    if "train" == action:
        configuration = {py_sim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=1.01, patience=3, max_time_steps=600),
                         fh_sim_env_name: StopCondition(stop_criterion=Criterion.NBR_POLICY_UPDATES, max_time_steps=2000),
                         lunar_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD_AND_LOSS, target_reward=0.20, target_loss=0.8, patience=1, max_time_steps=300)}
        metrics = trainer.train(configuration)

        print(f"Training metrics: {metrics}")
    elif "visualize" == action:
        trainer.visualize(envs[0])
    else:
        print(f"Unable to execute unknown action '{action}'")
