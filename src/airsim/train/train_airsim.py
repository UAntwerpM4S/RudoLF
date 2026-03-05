"""Train an RL agent inside UE5 via CosysAirSim.

Usage (from the ``src/`` directory):

    python -m airsim.train.train_airsim

Prerequisites:
    1. UE5 with CosysAirSim must be running and in Play mode.
    2. calibration.json must exist (run calibrate_coords.py once first).
"""

import logging
import multiprocessing

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    multiprocessing.set_start_method("spawn", force=True)

    from fw.trainer import Trainer
    from fw.stop_condition import StopCondition, Criterion
    from fw.config import PPO_POLICY_NAME
    from fw.env_factory import (
        get as get_environment,
        airsim_env_name,
    )

    # ── Configuration ──
    ACTION = "train"  # "train" or "visualize"

    # training_mode=True  → disables UE5 rendering + fast frame-stepping (optimal for training)
    # training_mode=False → full UE5 rendering + real-time stepping (required for visualize)
    env_config = {"training_mode": ACTION == "train"}

    # Register the AirSim environment (creates AirSimEnv, connects to UE5)
    env = get_environment(airsim_env_name, configuration=env_config)
    envs = [env]

    # Create trainer with PPO
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    if ACTION == "train":
        # Stop condition: train until target reward or give up
        configuration = {
            airsim_env_name: StopCondition(
                stop_criterion=Criterion.TARGET_REWARD,
                target_reward=1.75,
                patience=4,
                max_time_steps=22000,
            ),
        }

        print("\n" + "=" * 60)
        print("  STARTING AIRSIM RL TRAINING")
        print("=" * 60)
        print(f"  Environment: {airsim_env_name}")
        print(f"  Policy:      {PPO_POLICY_NAME}")
        print(f"  Target:      reward >= 1.75")
        print("=" * 60 + "\n")

        metrics = trainer.train(configuration)
        print(f"\nTraining metrics: {metrics}")

    elif ACTION == "visualize":
        print("\n" + "=" * 60)
        print("  VISUALIZING TRAINED POLICY IN AIRSIM")
        print("=" * 60 + "\n")

        trainer.visualize(envs)

    # Env is closed by the training pipeline, but just in case:
    try:
        env.gym_env.close()
    except Exception:
        pass
