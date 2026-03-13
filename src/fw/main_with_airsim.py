"""AirSim-focused runtime entrypoint for training and visualization.

This module centralizes AirSim orchestration from AirSim-specific runtime
settings:
- Training with AirSimEnv through the framework trainer.
- Visualization dispatch to either Fossen replay or UE5-direct inference.

It is intended to be called directly (``python -m fw.main_with_airsim``) or via
delegation from ``fw.main`` when ``fw.config.CONFIG_PARAMETERS['env_type']`` is
set to ``"AirSimEnv"``.
"""

if __name__ == "__main__" or __name__ == "fw.main_with_airsim":
    import os
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

    from airsim.config.airsim_config import (
        get_default_runtime_settings,
        get_runtime_settings,
        merge_runtime_settings,
    )
    from fw.env_factory import (
        airsim_env_name,
        fh_sim_env_name,
        lunar_env_name,
        py_sim_env_name,
        py_sim_polar_env_name,
    )

    # AirSim runtime only controls AirSim-specific behavior.
    # The selected framework environment is handled by fw.main / fw.config.
    _action_was_explicit = "FW_MAIN_ACTION_OVERRIDE" in os.environ
    action = str(os.environ.get("FW_MAIN_ACTION_OVERRIDE", "train")).strip().lower()
    if action not in {"train", "visualize"}:
        action = "train"
    selected_environment_name = airsim_env_name
    runtime_settings = get_default_runtime_settings()

    # Visualization flags switch the AirSim entrypoint into visualize mode.
    try:
        runtime_settings = merge_runtime_settings(get_runtime_settings())
    except Exception as exc:
        print(f"[main] Could not load runtime settings ({exc}). Using built-in defaults.")

    visualization_cfg = runtime_settings.get("visualization", {}) if isinstance(runtime_settings, dict) else {}
    training_cfg = runtime_settings.get("training", {}) if isinstance(runtime_settings, dict) else {}
    airsim_runtime_cfg = runtime_settings.get("airsim", {}) if isinstance(runtime_settings, dict) else {}

    training_speed_profiles = {
        "safe": {
            "simulation_speed": 20.0,
            "use_pause_stepping": True,
            "frames_per_step": 8,
            "time_step": 0.1,
            "physics_update_timeout": 1.0,
            "state_poll_interval": 0.005,
        },
        "fast": {
            "simulation_speed": 40.0,
            "use_pause_stepping": True,
            "frames_per_step": 10,
            "time_step": 0.1,
            "physics_update_timeout": 1.0,
            "state_poll_interval": 0.003,
        },
        "turbo": {
            "simulation_speed": 80.0,
            "use_pause_stepping": True,
            "frames_per_step": 12,
            "time_step": 0.1,
            "physics_update_timeout": 1.2,
            "state_poll_interval": 0.002,
        },
    }
    profile_name = str(training_cfg.get("speed_profile", "custom")).strip().lower()
    effective_training_cfg = dict(training_cfg)
    if profile_name in training_speed_profiles:
        effective_training_cfg.update(training_speed_profiles[profile_name])
        print(f"[main_with_airsim] Training speed profile: {profile_name}")
    elif profile_name != "custom":
        print(
            f"[main_with_airsim] Unknown training.speed_profile '{profile_name}'. "
            "Using explicit training settings."
        )
    # Only let runtime visualization flags drive the mode when main_with_airsim is
    # called directly (no explicit FW_MAIN_ACTION_OVERRIDE).  When delegated from
    # fw.main, the action set in fw.config takes precedence.
    if not _action_was_explicit and (
        bool(visualization_cfg.get("fossen_visualise", False)) or bool(
            visualization_cfg.get("inference_visualise", False)
        )
    ):
        action = "visualize"

    def _run_airsim_visualization_from_runtime() -> bool:
        """Dispatch AirSim-specific visualize runners driven by runtime settings.

        Returns:
            bool: True when a dedicated AirSim visualize runner was executed,
            False when caller should continue with generic trainer.visualize.
        """
        fossen_visualise = bool(visualization_cfg.get("fossen_visualise", False))
        inference_visualise = bool(visualization_cfg.get("inference_visualise", False))

        if not (fossen_visualise or inference_visualise):
            return False

        if fossen_visualise and inference_visualise:
            print("[main] Both fossen_visualise and inference_visualise are true. "
                  "Running fossen_visualise and skipping inference_visualise.")
            inference_visualise = False

        ip = airsim_runtime_cfg.get("ip", "")
        port = int(airsim_runtime_cfg.get("port", 41451))
        simulation_speed = float(airsim_runtime_cfg["simulation_speed"])
        save_slot = str(airsim_runtime_cfg["save_slot"])

        max_steps = int(visualization_cfg.get("max_steps", 14000))
        episodes = int(visualization_cfg.get("episodes", 1))
        device = str(visualization_cfg.get("device", "cpu"))
        log_every = int(visualization_cfg.get("log_every", 50))
        draw_ref = bool(visualization_cfg.get("draw_calibration_reference", True))

        if fossen_visualise:
            from airsim.inference.run_visualise_fossen import run_visualise_fossen

            logging.getLogger("airsim").setLevel(logging.INFO)
            run_visualise_fossen(
                model_path=str(visualization_cfg.get("fossen_model_path", "src/models/PySimEnv_PPO_policy.zip")),
                env_name=str(visualization_cfg.get("fossen_environment", "auto")),
                ip=ip,
                port=port,
                max_steps=max_steps,
                num_episodes=episodes,
                device=device,
                draw_calibration_reference=draw_ref,
                log_every=log_every,
                save_slot=save_slot,
                simulation_speed=simulation_speed,
                mirror_controls_in_ue5=bool(visualization_cfg.get("mirror_controls_in_ue5", False)),
                animate_steering_in_ue5=bool(visualization_cfg.get("animate_steering_in_ue5", True)),
                animate_throttle_in_ue5=bool(visualization_cfg.get("animate_throttle_in_ue5", True)),
                render_interpolation_steps=int(visualization_cfg.get("render_interpolation_steps", 8)),
                steering_visual_gain=float(visualization_cfg.get("steering_visual_gain", 0.35)),
                throttle_visual_gain=float(visualization_cfg.get("throttle_visual_gain", 1.0)),
                use_kinematics_in_ue5=bool(visualization_cfg.get("use_kinematics_in_ue5", False)),
                min_render_hz=float(visualization_cfg.get("min_render_hz", 90.0)),
                max_translation_step=float(visualization_cfg.get("max_translation_step", 0.15)),
                max_yaw_step_deg=float(visualization_cfg.get("max_yaw_step_deg", 1.5)),
                lock_path_in_ue5=bool(visualization_cfg.get("lock_path_in_ue5", False)),
                vehicle_object_name=str(visualization_cfg.get("vehicle_object_name", "")),
                show_minimap=bool(visualization_cfg.get("show_minimap", True)),
            )
            return True

        if inference_visualise:
            from airsim.inference.run_inference import run_inference

            logging.getLogger("airsim").setLevel(logging.INFO)
            run_inference(
                model_path=str(visualization_cfg.get("inference_model_path", "src/models/AirSimEnv_PPO_policy.zip")),
                ip=ip,
                port=port,
                max_steps=max_steps,
                num_episodes=episodes,
                device=device,
            )
            return True

        return False

    def _is_airsim_selected(env_wrappers) -> bool:
        """Check if any wrapped environment is AirSimEnv.

        Args:
            env_wrappers: Iterable of framework environment wrappers exposing
                ``gym_env``.

        Returns:
            bool: True if at least one wrapped environment has type
            ``AirSimEnv``.
        """
        return any(getattr(env.gym_env, "type_name", "") == airsim_env_name for env in env_wrappers)

    # AirSim visualize dispatch from runtime settings.
    # This makes fw.main_with_airsim a single entrypoint for both AirSim visualize modes.
    if action == "visualize":
        try:
            if _run_airsim_visualization_from_runtime():
                raise SystemExit(0)
        except ConnectionError as exc:
            print(f"[main_with_airsim] {exc}")
            print(
                "[main_with_airsim] UE5-dependent AirSim visualisation was requested, "
                "but UE5 is not reachable. Start UE5 in Play mode or disable the "
                "AirSim visualisation flags in runtime_settings.json."
            )
            raise SystemExit(1)

    # Ensure all environments are registered once
    from fw.trainer import Trainer
    from fw.stop_condition import StopCondition, Criterion
    from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME
    from fw.env_factory import (
        get as get_environment,
        initialize_all_environments,
    )

    initialize_all_environments()

    # Define simulator inputs. Keep hardcoded behavior by default, allow runtime override.
    if selected_environment_name == airsim_env_name:
        is_training = action == "train"
        airsim_env_cfg = dict(airsim_runtime_cfg)
        effective_max_steps = (
            int(effective_training_cfg.get("max_steps", 14000))
            if is_training
            else int(visualization_cfg.get("max_steps", 14000))
        )
        effective_simulation_speed = (
            float(effective_training_cfg.get("simulation_speed", airsim_runtime_cfg["simulation_speed"]))
            if is_training
            else float(airsim_runtime_cfg["simulation_speed"])
        )
        airsim_env_cfg.update({
            "training_mode": is_training,
            "use_pause_stepping": bool(effective_training_cfg.get("use_pause_stepping", is_training)),
            "simulation_speed": effective_simulation_speed,
            "max_steps": effective_max_steps,
            "frames_per_step": int(effective_training_cfg.get("frames_per_step", 10)),
            "time_step": float(effective_training_cfg.get("time_step", 0.1)),
            "physics_update_timeout": float(effective_training_cfg.get("physics_update_timeout", 1.0)),
            "state_poll_interval": float(effective_training_cfg.get("state_poll_interval", 0.005)),
        })
        if "latency_warning_threshold" in effective_training_cfg:
            airsim_env_cfg["latency_warning_threshold"] = float(effective_training_cfg["latency_warning_threshold"])
        envs = [get_environment(airsim_env_name, configuration=airsim_env_cfg)]
    else:
        envs = [get_environment(selected_environment_name)]

    # Create a Trainer and pass on a list of agents that you want to train
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    if "train" == action:
        # Re-enable airsim logging only when the airsim environment is selected
        if _is_airsim_selected(envs):
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
        if airsim_cfg.get("use_airsim", False) and not _is_airsim_selected(envs):
            from airsim.communication.airsim_bridge import AirSimBridge
            bridge_cfg = AirSimConfig(
                ip=airsim_cfg.get("ip", ""),
                port=airsim_cfg.get("port", 41451)
            )
            temp_bridge = AirSimBridge(bridge_cfg)
            if temp_bridge.is_connected:
                airsim_bridge = temp_bridge
                logging.getLogger("airsim").setLevel(logging.INFO)  # connected - enable airsim logs
            else:
                print("[main] AirSim bridge not connected - running without.")
        elif _is_airsim_selected(envs):
            logging.getLogger("airsim").setLevel(logging.INFO)

        trainer.visualize(envs, airsim_bridge=airsim_bridge)
    else:
        print(f"Unable to execute unknown action '{action}'")
