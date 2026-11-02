"""
AirSim entry point for training and visualization, driven by the AirSim runtime settings.

- Training: AirSimEnv via train_airsim.run_training.
- Visualization: framework-model replay, AirSimEnv inference, or the generic Trainer.visualize.

Run directly (`python -m fw.main_with_airsim`) or via delegation from `fw.main` when
env_type is "AirSimEnv".
"""

if __name__ == "__main__" or __name__ == "fw.main_with_airsim":
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import logging
    import multiprocessing

    from pathlib import Path
    from fw.common.tools.logging_utils import create_logfile_handler
    from fw.common.airsim.classes.asyncio_win995_filter import AsyncioWin995Filter

    # Resolved before logging setup so the log filename can include the action.
    _action_was_explicit = "FW_MAIN_ACTION_OVERRIDE" in os.environ
    _SRC_DIR = Path(__file__).resolve().parents[1]
    action = str(os.environ.get("FW_MAIN_ACTION_OVERRIDE", "train")).strip().lower()
    if action not in {"train", "visualize"}:
        action = "train"

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
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    logging.getLogger("asyncio").addFilter(AsyncioWin995Filter())

    # Per-run log file. Console verbosity follows FW_AIRSIM_VERBOSE (AirSim/UE5 logs
    # are noisy); the file always captures at least INFO.
    from fw.common.tools.path_utils import resolve_path_relative_to_src as _resolve_log_dir
    from fw.configuration.config import PPO_POLICY_NAME as _PPO_POLICY_NAME
    from datetime import datetime, timezone as _timezone

    _file_level = min(_airsim_log_level, logging.INFO)
    _log_ts = datetime.now(_timezone.utc).strftime("%Y%m%d_%H%M%S")
    _log_dir = _resolve_log_dir("logs", Path(__file__).resolve().parents[1])
    _log_file_path = _log_dir / f"AirSimEnv_{_PPO_POLICY_NAME}_{action}_{_log_ts}.log"

    root_logger = logging.getLogger()
    # Keep this run's handler: the rename below must not touch fw.main's own log
    # handler when delegated.
    _own_file_handler = create_logfile_handler(_log_file_path, _file_level)
    root_logger.addHandler(_own_file_handler)
    root_logger.setLevel(min(root_logger.level, _file_level))

    logger = logging.getLogger("fw.main_with_airsim")
    logger.info("Logging to file: %s", _log_file_path)
    multiprocessing.set_start_method("spawn", force=True)

    from airsim.config.airsim_config import (
        DEFAULT_THROTTLE_VISUAL_GAIN,
        get_default_runtime_settings,
        get_runtime_settings,
        merge_runtime_settings,
    )
    from fw.common.tools.path_utils import resolve_path_relative_to_src
    from fw.common.tools.config_utils import get_validated_bool
    from fw.configuration.env_factory import (
        airsim_env_name,
    )
    from fw.configuration.config import get_config_parameters
    from fw.configuration.profile_applier import apply_profile
    from fw.configuration.config_profile import ProfileRegistry
    from airsim.train.train_airsim import TRAINING_SPEED_PROFILES

    runtime_settings = get_default_runtime_settings()

    try:
        runtime_settings = merge_runtime_settings(get_runtime_settings())
    except Exception as exc:
        logger.warning("Could not load runtime settings (%s). Using built-in defaults.", exc)

    visualization_cfg = runtime_settings.get("visualization", {}) if isinstance(runtime_settings, dict) else {}
    training_cfg = runtime_settings.get("training", {}) if isinstance(runtime_settings, dict) else {}
    airsim_runtime_cfg = runtime_settings.get("airsim", {}) if isinstance(runtime_settings, dict) else {}

    profile_name = str(training_cfg.get("speed_profile", "custom")).strip().lower()
    effective_training_cfg = dict(training_cfg)
    if profile_name in TRAINING_SPEED_PROFILES:
        effective_training_cfg.update(TRAINING_SPEED_PROFILES[profile_name])
        logger.info("Training speed profile: %s", profile_name)
    elif profile_name != "custom":
        logger.warning(
            "Unknown training.speed_profile '%s'. Using explicit training settings.",
            profile_name,
        )
    if get_validated_bool(effective_training_cfg, "use_pause_stepping", False) and get_validated_bool(
        effective_training_cfg, "use_sim_time_stepping", False
    ):
        effective_training_cfg["time_step"] = float(
            effective_training_cfg.get(
                "sim_time_per_action",
                effective_training_cfg.get("time_step", 0.1),
            )
        )
        logger.info(
            "MDP step: %.3fs simulated time per RL action",
            float(effective_training_cfg['time_step']),
        )

    if get_validated_bool(effective_training_cfg, "use_pause_stepping", False):
        if get_validated_bool(effective_training_cfg, "use_sim_time_stepping", False):
            logger.info(
                "Step mode: simContinueForTime(%.3fs)",
                float(effective_training_cfg.get('sim_time_per_action', effective_training_cfg.get('time_step', 0.1))),
            )
        else:
            logger.info(
                "Step mode: simContinueForFrames(%d)",
                int(effective_training_cfg.get('frames_per_step', 1)),
            )
    # Visualization flags switch to visualize mode only without an explicit action
    # override, so a train step delegated from fw.main stays a train step.
    if not _action_was_explicit and (
        get_validated_bool(visualization_cfg, "model_visualise", False)
        or get_validated_bool(visualization_cfg, "inference_visualise", False)
    ):
        if action != "visualize":
            action = "visualize"
            # Rename the log file, which was created with "train" in its name.
            _new_log_file_path = _log_dir / f"AirSimEnv_{_PPO_POLICY_NAME}_{action}_{_log_ts}.log"
            _root_logger = logging.getLogger()
            _old_file_handler = _own_file_handler if _own_file_handler in _root_logger.handlers else None
            if _old_file_handler is not None and _new_log_file_path != _log_file_path:
                _root_logger.removeHandler(_old_file_handler)
                _old_file_handler.close()
                try:
                    _log_file_path.rename(_new_log_file_path)
                    _log_file_path = _new_log_file_path
                except OSError:
                    pass  # Keep logging to the original filename rather than losing the log.
                _new_file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
                _new_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
                _new_file_handler.setLevel(_file_level)
                _root_logger.addHandler(_new_file_handler)
                _own_file_handler = _new_file_handler
                logger.info("Log file renamed to: %s", _log_file_path)

    def _run_airsim_visualization_from_runtime() -> bool:
        """
        Run the AirSim visualizer selected in the runtime settings.

        Returns:
            True if a dedicated runner was executed, False if the caller should use
            the generic trainer.visualize path.
        """

        model_visualise = get_validated_bool(visualization_cfg, "model_visualise", False)
        inference_visualise = get_validated_bool(visualization_cfg, "inference_visualise", False)

        if not (model_visualise or inference_visualise):
            return False

        if model_visualise and inference_visualise:
            logger.warning("Both model_visualise and inference_visualise are true. "
                            "Running model_visualise and skipping inference_visualise.")
            inference_visualise = False

        ip = airsim_runtime_cfg.get("ip", "")
        port = int(airsim_runtime_cfg.get("port", 41451))
        simulation_speed = float(airsim_runtime_cfg.get("simulation_speed", 1.0))
        save_slot = str(airsim_runtime_cfg.get("save_slot", ""))

        max_steps = int(visualization_cfg.get("max_steps", 14000))
        episodes = int(visualization_cfg.get("episodes", 1))
        device = str(visualization_cfg.get("device", "cpu"))
        log_every = int(visualization_cfg.get("log_every", 50))
        draw_ref = get_validated_bool(visualization_cfg, "draw_calibration_reference", True)

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
                mirror_controls_in_ue5=get_validated_bool(visualization_cfg, "mirror_controls_in_ue5", False),
                animate_steering_in_ue5=get_validated_bool(visualization_cfg, "animate_steering_in_ue5", True),
                animate_throttle_in_ue5=get_validated_bool(visualization_cfg, "animate_throttle_in_ue5", True),
                render_interpolation_steps=int(visualization_cfg.get("render_interpolation_steps", 8)),
                steering_visual_gain=float(visualization_cfg.get("steering_visual_gain", 0.35)),
                throttle_visual_gain=float(
                    visualization_cfg.get("throttle_visual_gain", DEFAULT_THROTTLE_VISUAL_GAIN)
                ),
                use_kinematics_in_ue5=get_validated_bool(visualization_cfg, "use_kinematics_in_ue5", False),
                min_render_hz=float(visualization_cfg.get("min_render_hz", 90.0)),
                max_translation_step=float(visualization_cfg.get("max_translation_step", 0.15)),
                max_yaw_step_deg=float(visualization_cfg.get("max_yaw_step_deg", 1.5)),
                lock_path_in_ue5=get_validated_bool(visualization_cfg, "lock_path_in_ue5", False),
                vehicle_object_name=str(visualization_cfg.get("vehicle_object_name", "")),
                min_log_interval_s=float(visualization_cfg.get("min_log_interval_s", 0.25)),
                bridge_telemetry_hz=float(visualization_cfg.get("bridge_telemetry_hz", 12.0)),
                startup_settle_s=float(visualization_cfg.get("startup_settle_s", 0.4)),
                verbose=get_validated_bool(visualization_cfg, "verbose", _verbose_airsim),
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
                verbose=get_validated_bool(visualization_cfg, "verbose", _verbose_airsim),
            )
            return True

        return False

    def _is_airsim_selected(env_wrappers) -> bool:
        """
        True if any of the environment wrappers (with `gym_env`) wraps an AirSimEnv.
        """

        return any(getattr(env.gym_env, "type_name", "") == airsim_env_name for env in env_wrappers)

    if action == "visualize":
        try:
            if _run_airsim_visualization_from_runtime():
                raise SystemExit(0)
            # No dedicated visualizer selected: fall through to trainer.visualize below.
        except ConnectionError as exc:
            print(f"[main_with_airsim] {exc}")
            print(
                "[main_with_airsim] UE5-dependent AirSim visualisation was requested, "
                "but UE5 is not reachable. Start UE5 in Play mode or disable the "
                "AirSim visualisation flags in runtime_settings.json."
            )
            logger.error(
                "%s. UE5-dependent AirSim visualisation was requested, but UE5 is "
                "not reachable. Start UE5 in Play mode or disable the AirSim "
                "visualisation flags in runtime_settings.json.",
                exc,
            )
            raise SystemExit(1)

    if action == "train":
        from airsim.train.train_airsim import run_training

        metrics = run_training(runtime_settings)
        if metrics:
            logger.info("Training metrics: %s", metrics)

        raise SystemExit(0)

    # --- Generic Trainer.visualize path (visualize without model/inference_visualise) ---
    from fw.scheduling.trainer import Trainer
    from fw.configuration.config import PPO_POLICY_NAME
    from fw.configuration.env_factory import (
        get as get_environment,
        initialize_all_environments,
    )
    from airsim.config.airsim_config import get_airsim_config, AirSimConfig

    initialize_all_environments()

    is_training = action == "train"
    airsim_env_cfg = dict(airsim_runtime_cfg)
    effective_max_steps = (
        int(effective_training_cfg.get("max_steps", 14000))
        if is_training
        else int(visualization_cfg.get("max_steps", 14000))
    )
    effective_simulation_speed = (
        float(effective_training_cfg.get("simulation_speed", airsim_runtime_cfg.get("simulation_speed", 1.0)))
        if is_training
        else float(airsim_runtime_cfg.get("simulation_speed", 1.0))
    )
    airsim_env_cfg.update({
        "training_mode": is_training,
        "draw_calibration_reference": get_validated_bool(
            effective_training_cfg,
            "draw_calibration_reference",
            get_validated_bool(visualization_cfg, "draw_calibration_reference", True),
        ),
        "no_rendering_mode": get_validated_bool(effective_training_cfg, "no_rendering_mode", False),
        "load_world_on_start": effective_training_cfg.get("load_world_on_start", None),
        "capture_live_start_pose": effective_training_cfg.get("capture_live_start_pose", None),
        "episode_log_path": str(effective_training_cfg.get("episode_log_path", "")),
        "normalize_reward": get_validated_bool(effective_training_cfg, "normalize_reward", True),
        "reward_normalization_scale": float(
            effective_training_cfg.get("reward_normalization_scale", 5.0)
        ),
        "reset_mode": str(effective_training_cfg.get("reset_mode", "rewind_pose")),
        "progress_log_interval_steps": int(effective_training_cfg.get("progress_log_interval_steps", 0)),
        "log_pause_watchdog_warnings": get_validated_bool(effective_training_cfg, "log_pause_watchdog_warnings", False),
        "use_sim_time_stepping": get_validated_bool(effective_training_cfg, "use_sim_time_stepping", False),
        "sim_time_per_action": float(effective_training_cfg.get("sim_time_per_action", effective_training_cfg.get("time_step", 0.1))),
        "use_pause_stepping": get_validated_bool(effective_training_cfg, "use_pause_stepping", is_training),
        "simulation_speed": effective_simulation_speed,
        "max_steps": effective_max_steps,
        "frames_per_step": int(effective_training_cfg.get("frames_per_step", 8)),
        "time_step": float(effective_training_cfg.get("time_step", 0.1)),
        "physics_update_timeout": float(effective_training_cfg.get("physics_update_timeout", 1.0)),
        "state_poll_interval": float(effective_training_cfg.get("state_poll_interval", 0.005)),
        "guidance_assist_enabled": get_validated_bool(effective_training_cfg, "guidance_assist_enabled", False),
        "guidance_residual_steering_scale": float(
            effective_training_cfg.get("guidance_residual_steering_scale", 0.50)
        ),
        "guidance_residual_throttle_scale": float(
            effective_training_cfg.get("guidance_residual_throttle_scale", 0.60)
        ),
        "control_interface": str(effective_training_cfg.get("control_interface", "car_physics")),
        "observation_profile": str(
            effective_training_cfg.get("observation_profile", "ppo_route_relative_car")
        ),
    })
    if "latency_warning_threshold" in effective_training_cfg:
        airsim_env_cfg["latency_warning_threshold"] = float(effective_training_cfg["latency_warning_threshold"])
    envs = [get_environment(airsim_env_name, configuration=airsim_env_cfg)]

    # When delegated from fw.main, stop if env_factory fell back to a non-AirSim env
    # (e.g. PySimEnv), to avoid running that env twice.
    if _action_was_explicit and not _is_airsim_selected(envs):
        logger.warning(
            "AirSimEnv requested via fw.main, but UE5 is unavailable and "
            "env_factory returned a fallback env. Skipping this AirSim step."
        )
        raise SystemExit(1)

    # Apply the configured profile and create the Trainer.
    _airsim_fw_config = get_config_parameters()
    _airsim_profile_name = _airsim_fw_config.get("profile", "default_airsim_profile")
    _airsim_profile = ProfileRegistry.get(_airsim_profile_name)
    _airsim_profile.validate()

    for _airsim_env_wrapper in envs:
        apply_profile(_airsim_profile, env=_airsim_env_wrapper.gym_env)
        _airsim_sim = (
            getattr(_airsim_env_wrapper.gym_env, "phys_sim", None)
            or getattr(_airsim_env_wrapper.gym_env, "_phys_sim", None)
        )
        if _airsim_sim is not None:
            apply_profile(_airsim_profile, simulator=_airsim_sim)

    trainer = Trainer(PPO_POLICY_NAME, envs, _airsim_profile, "SimpleScheduler")

    # Optional AirSim bridge to mirror a non-AirSim env in UE5.
    airsim_cfg = get_airsim_config()
    airsim_bridge = None
    if get_validated_bool(airsim_cfg, "use_airsim", False) and not _is_airsim_selected(envs):
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
            logger.warning("AirSim bridge not connected - running without.")
    elif _is_airsim_selected(envs):
        logging.getLogger("airsim").setLevel(_airsim_log_level)
    trainer.visualize(envs, airsim_bridge=airsim_bridge)
