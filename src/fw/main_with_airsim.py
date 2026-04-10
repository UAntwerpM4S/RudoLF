"""AirSim-focused runtime entrypoint for training and visualization.

This module centralizes AirSim orchestration from AirSim-specific runtime
settings:
- Training with AirSimEnv through the framework trainer.
- Visualization dispatch to either modular framework-model replay or
  AirSimEnv inference.

It is intended to be called directly (``python -m fw.main_with_airsim``) or via
delegation from ``fw.main`` when ``fw.config.CONFIG_PARAMETERS['env_type']`` is
set to ``"AirSimEnv"``.
"""

if __name__ == "__main__" or __name__ == "fw.main_with_airsim":
    import logging
    import multiprocessing
    from pathlib import Path
    import os

    _verbose_airsim = str(os.environ.get("FW_AIRSIM_VERBOSE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _airsim_log_level = logging.INFO if _verbose_airsim else logging.WARNING
    _sim_log_level = logging.INFO if _verbose_airsim else logging.ERROR

    logging.basicConfig(
        level=_airsim_log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # Keep AirSim and simulator logs concise by default.
    logging.getLogger("airsim").setLevel(_airsim_log_level)
    logging.getLogger("sim").setLevel(_sim_log_level)


    multiprocessing.set_start_method("spawn", force=True)

    from airsim.config.airsim_config import (
        get_default_runtime_settings,
        get_runtime_settings,
        merge_runtime_settings,
    )
    from fw.common.tools.path_utils import resolve_path_relative_to_src
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
    _SRC_DIR = Path(__file__).resolve().parents[1]
    action = str(os.environ.get("FW_MAIN_ACTION_OVERRIDE", "train")).strip().lower()
    if action not in {"train", "visualize"}:
        action = "train"
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
        "normal": {
            "simulation_speed": 1.0,
            "use_pause_stepping": False,
            "frames_per_step": 1,
            "time_step": 0.1,
            "physics_update_timeout": 1.0,
            "state_poll_interval": 0.01,
        },
        "safe": {
            "simulation_speed": 20.0,
            "use_pause_stepping": True,
            "use_sim_time_stepping": True,
            "sim_time_per_action": 0.1,
            "frames_per_step": 8,
            "time_step": 0.1,
            "physics_update_timeout": 1.0,
            "state_poll_interval": 0.005,
        },
        "fast": {
            "simulation_speed": 40.0,
            "use_pause_stepping": True,
            "use_sim_time_stepping": True,
            "sim_time_per_action": 0.1,
            "frames_per_step": 10,
            "time_step": 0.1,
            "physics_update_timeout": 1.0,
            "state_poll_interval": 0.003,
        },
        "turbo": {
            "simulation_speed": 80.0,
            "use_pause_stepping": True,
            "use_sim_time_stepping": True,
            "sim_time_per_action": 0.1,
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
    if bool(effective_training_cfg.get("use_pause_stepping", False)):
        if bool(effective_training_cfg.get("use_sim_time_stepping", False)):
            print(
                "[main_with_airsim] Step mode: simContinueForTime("
                f"{float(effective_training_cfg.get('sim_time_per_action', effective_training_cfg.get('time_step', 0.1))):.3f}s)"
            )
        else:
            print(
                "[main_with_airsim] Step mode: simContinueForFrames("
                f"{int(effective_training_cfg.get('frames_per_step', 1))})"
            )
    # Runtime visualization flags may promote the AirSim entrypoint into
    # visualize mode only when fw.main did not pass an explicit action override.
    # This keeps ordered framework execution honest: a delegated AirSim train
    # step must remain a train step even though runpy executes this module with
    # __name__ == "__main__".
    if not _action_was_explicit and (
        bool(visualization_cfg.get("model_visualise", False))
        or bool(visualization_cfg.get("inference_visualise", False))
    ):
        action = "visualize"

    def _run_airsim_visualization_from_runtime() -> bool:
        """Dispatch AirSim-specific visualize runners driven by runtime settings.

        Returns:
            bool: True when a dedicated AirSim visualize runner was executed,
            False when caller should continue with generic trainer.visualize.
        """
        model_visualise = bool(visualization_cfg.get("model_visualise", False))
        inference_visualise = bool(visualization_cfg.get("inference_visualise", False))

        if not (model_visualise or inference_visualise):
            return False

        if model_visualise and inference_visualise:
            print("[main] Both model_visualise and inference_visualise are true. "
                  "Running model_visualise and skipping inference_visualise.")
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

        if model_visualise:
            import airsim.inference.run_visualise_models as run_visualise_models_module

            logging.getLogger("airsim").setLevel(_airsim_log_level)
            run_visualise_models_module.run_visualise_models(
                model_path=str(
                    resolve_path_relative_to_src(
                        str(visualization_cfg.get("model_path", "models/PySimEnv_PPO_policy.zip")),
                        _SRC_DIR,
                    )
                ),
                model_family=str(visualization_cfg.get("model_family", "auto")),
                env_name=str(visualization_cfg.get("environment_name", "auto")),
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
                save_matplotlib_minimap=bool(visualization_cfg.get("save_matplotlib_minimap", False)),
                use_ue5_camera_minimap=bool(visualization_cfg.get("use_ue5_camera_minimap", False)),
                ue5_camera_name=str(visualization_cfg.get("ue5_camera_name", "TopDownCam")),
                ue5_camera_fov_deg=float(visualization_cfg.get("ue5_camera_fov_deg", 38.0)),
                ue5_camera_height_m=float(visualization_cfg.get("ue5_camera_height_m", 120.0)),
                ue5_camera_follow_heading=bool(visualization_cfg.get("ue5_camera_follow_heading", False)),
                ue5_preview_window=bool(visualization_cfg.get("ue5_preview_window", True)),
                min_log_interval_s=float(visualization_cfg.get("min_log_interval_s", 0.25)),
                minimap_update_hz=float(visualization_cfg.get("minimap_update_hz", 20.0)),
                bridge_telemetry_hz=float(visualization_cfg.get("bridge_telemetry_hz", 12.0)),
                startup_settle_s=float(visualization_cfg.get("startup_settle_s", 0.4)),
                verbose=bool(visualization_cfg.get("verbose", _verbose_airsim)),
            )
            return True

        if inference_visualise:
            import airsim.inference.run_inference as run_inference_module

            logging.getLogger("airsim").setLevel(_airsim_log_level)
            run_inference_module.run_inference(
                model_path=str(resolve_path_relative_to_src(str(visualization_cfg.get("inference_model_path", "models/AirSimEnv_PPO_policy.zip")), _SRC_DIR)),
                ip=ip,
                port=port,
                max_steps=max_steps,
                num_episodes=episodes,
                device=device,
                draw_calibration_reference=draw_ref,
                verbose=bool(visualization_cfg.get("verbose", _verbose_airsim)),
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
    from fw.config import PPO_POLICY_NAME
    from fw.env_factory import (
        get as get_environment,
        initialize_all_environments,
    )

    initialize_all_environments()

    # This entrypoint always orchestrates the AirSim environment itself.
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
        "draw_calibration_reference": bool(
            effective_training_cfg.get(
                "draw_calibration_reference",
                visualization_cfg.get("draw_calibration_reference", True),
            )
        ),
        "no_rendering_mode": bool(effective_training_cfg.get("no_rendering_mode", False)),
        "episode_log_path": str(effective_training_cfg.get("episode_log_path", "")),
        "progress_log_interval_steps": int(effective_training_cfg.get("progress_log_interval_steps", 0)),
        "log_pause_watchdog_warnings": bool(effective_training_cfg.get("log_pause_watchdog_warnings", False)),
        "use_sim_time_stepping": bool(effective_training_cfg.get("use_sim_time_stepping", False)),
        "sim_time_per_action": float(effective_training_cfg.get("sim_time_per_action", effective_training_cfg.get("time_step", 0.1))),
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

    # When delegated from fw.main as an ordered AirSim step, fail fast if
    # AirSim resolved to a framework fallback environment (e.g. PySimEnv).
    # This prevents duplicate training/visualization of a non-Airsim env.
    if _action_was_explicit and not _is_airsim_selected(envs):
        print(
            "[main_with_airsim] AirSimEnv requested via fw.main, "
            "but UE5 is unavailable and env_factory returned a fallback env. "
            "Skipping this AirSim step."
        )
        raise SystemExit(1)

    # Create a Trainer and pass on a list of agents that you want to train
    trainer = Trainer(PPO_POLICY_NAME, envs, "SimpleScheduler")

    if "train" == action:
        # Re-enable airsim logging only when the airsim environment is selected
        if _is_airsim_selected(envs):
            logging.getLogger("airsim").setLevel(_airsim_log_level)

        configuration = {py_sim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=1.75, patience=4, max_time_steps=1300),
                         airsim_env_name: StopCondition(stop_criterion=Criterion.NBR_POLICY_UPDATES, max_time_steps=350),
                         fh_sim_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD, target_reward=3.80, patience=4, max_time_steps=1300),
                         lunar_env_name: StopCondition(stop_criterion=Criterion.TARGET_REWARD_AND_LOSS, target_reward=0.20, target_loss=0.8, patience=1, max_time_steps=300),
                         py_sim_polar_env_name: StopCondition(stop_criterion=Criterion.NBR_POLICY_UPDATES, max_time_steps=600)}
        metrics = trainer.train(configuration)

        print(f"Training metrics: {metrics}")
    else:
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
                logging.getLogger("airsim").setLevel(_airsim_log_level)
            else:
                print("[main] AirSim bridge not connected - running without.")
        elif _is_airsim_selected(envs):
            logging.getLogger("airsim").setLevel(_airsim_log_level)
        trainer.visualize(envs, airsim_bridge=airsim_bridge)
