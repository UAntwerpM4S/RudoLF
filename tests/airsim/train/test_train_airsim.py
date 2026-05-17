"""
Tests for the standalone AirSim training entrypoint helpers.
"""

from __future__ import annotations

from pathlib import Path


def test_build_training_env_config_uses_runtime_settings(tmp_path):
    """
    Standalone AirSim training config should come from runtime settings.
    """

    from airsim.train.train_airsim import _build_training_env_config

    runtime_settings = {
        "training": {
            "no_rendering_mode": True,
            "load_world_on_start": True,
            "capture_live_start_pose": False,
            "use_pause_stepping": True,
            "use_sim_time_stepping": True,
            "sim_time_per_action": 0.15,
            "frames_per_step": 12,
            "time_step": 0.2,
            "physics_update_timeout": 2.0,
            "state_poll_interval": 0.01,
            "draw_calibration_reference": False,
            "simulation_speed": 33.0,
            "episode_log_path": "logs/custom_episode_log.csv",
            "normalize_reward": False,
            "reward_normalization_scale": 95.0,
            "reset_mode": "bridge_reset",
            "progress_log_interval_steps": 321,
            "log_pause_watchdog_warnings": True,
            "max_steps": 9876,
            "curriculum_enabled": True,
            "curriculum_window_size": 5,
            "curriculum_frontier_increment": 3,
            "curriculum_window_increment": 4,
            "curriculum_success_patience": 6,
            "curriculum_randomization_scale": 7.5,
            "guidance_assist_enabled": True,
            "guidance_residual_steering_scale": 0.42,
            "guidance_residual_throttle_scale": 0.57,
            "steering_dynamics_enabled": False,
            "steering_smoothing_alpha": 0.8,
            "max_steering_rate_per_second": 1.75,
            "control_interface": "car_physics",
            "observation_profile": "ppo_route_relative_car",
            "latency_warning_threshold": 1.25,
        },
        "airsim": {
            "use_airsim": True,
            "ip": "127.0.0.1",
            "port": 41451,
            "timeout": 2.5,
            "max_retries": 7,
            "health_check_interval": 11.0,
            "csv_log_path": "bridge.csv",
            "save_slot": "slot_two",
            "simulation_speed": 99.0,
        },
    }

    config = _build_training_env_config(runtime_settings)

    assert config["training_mode"] is True
    assert config["use_airsim"] is True
    assert config["ip"] == "127.0.0.1"
    assert config["port"] == 41451
    assert config["timeout"] == 2.5
    assert config["max_retries"] == 7
    assert config["health_check_interval"] == 11.0
    assert config["csv_log_path"] == "bridge.csv"
    assert config["save_slot"] == "slot_two"
    assert config["load_world_on_start"] is True
    assert config["capture_live_start_pose"] is False
    assert config["no_rendering_mode"] is True
    assert config["use_pause_stepping"] is True
    assert config["use_sim_time_stepping"] is True
    assert config["sim_time_per_action"] == 0.15
    assert config["frames_per_step"] == 12
    assert config["time_step"] == 0.15
    assert config["physics_update_timeout"] == 2.0
    assert config["state_poll_interval"] == 0.01
    assert config["draw_calibration_reference"] is False
    assert config["simulation_speed"] == 33.0
    assert config["normalize_reward"] is False
    assert config["reward_normalization_scale"] == 95.0
    assert config["reset_mode"] == "bridge_reset"
    assert config["progress_log_interval_steps"] == 321
    assert config["log_pause_watchdog_warnings"] is True
    assert config["max_steps"] == 9876
    assert config["curriculum_enabled"] is True
    assert config["curriculum_window_size"] == 5
    assert config["curriculum_frontier_increment"] == 3
    assert config["curriculum_window_increment"] == 4
    assert config["curriculum_success_patience"] == 6
    assert config["curriculum_randomization_scale"] == 7.5
    assert config["guidance_assist_enabled"] is True
    assert config["guidance_residual_steering_scale"] == 0.42
    assert config["guidance_residual_throttle_scale"] == 0.57
    assert config["steering_dynamics_enabled"] is False
    assert config["steering_smoothing_alpha"] == 0.8
    assert config["max_steering_rate_per_second"] == 1.75
    assert config["control_interface"] == "car_physics"
    assert config["observation_profile"] == "ppo_route_relative_car"
    assert config["latency_warning_threshold"] == 1.25

    expected_suffix = Path("logs/custom_episode_log.csv")
    assert Path(config["episode_log_path"]).is_absolute()
    assert Path(config["episode_log_path"]).as_posix().endswith(expected_suffix.as_posix())


def test_build_visualize_env_config_uses_runtime_settings():
    """
    Visualization config should also honor runtime settings defaults.
    """

    from airsim.train.train_airsim import _build_visualize_env_config

    runtime_settings = {
        "visualization": {
            "draw_calibration_reference": False,
            "max_steps": 4321,
        },
        "airsim": {
            "use_airsim": True,
            "ip": "10.0.0.2",
            "port": 5000,
            "save_slot": "visual_slot",
        },
    }

    config = _build_visualize_env_config(runtime_settings)

    assert config["training_mode"] is False
    assert config["use_pause_stepping"] is False
    assert config["draw_calibration_reference"] is False
    assert config["simulation_speed"] == 1.0
    assert config["max_steps"] == 4321
    assert config["ip"] == "10.0.0.2"
    assert config["port"] == 5000
    assert config["save_slot"] == "visual_slot"
    assert config["load_world_on_start"] is True
    assert config["capture_live_start_pose"] is False


def test_build_training_env_config_defaults_to_direct_policy_control():
    """
    Standalone training defaults should avoid hidden guidance and use the car profile.
    """

    from airsim.train.train_airsim import _build_training_env_config

    config = _build_training_env_config({"training": {}, "airsim": {}})

    assert config["guidance_assist_enabled"] is False
    assert config["control_interface"] == "car_physics"
    assert config["observation_profile"] == "ppo_route_relative_car"
    assert config["normalize_reward"] is True
    assert config["reset_mode"] == "rewind_pose"
    assert config["curriculum_enabled"] is False
    assert config["load_world_on_start"] is True
    assert config["capture_live_start_pose"] is True
    assert config["steering_dynamics_enabled"] is True
    assert config["steering_smoothing_alpha"] == 0.65
    assert config["max_steering_rate_per_second"] == 3.2


def test_build_training_env_config_applies_speed_profile_defaults():
    """
    The standalone trainer should honor named runtime speed profiles too.
    """

    from airsim.train.train_airsim import _build_training_env_config

    config = _build_training_env_config(
        {
            "training": {
                "speed_profile": "safe",
            },
            "airsim": {},
        }
    )

    assert config["simulation_speed"] == 20.0
    assert config["use_pause_stepping"] is True
    assert config["use_sim_time_stepping"] is True
    assert config["sim_time_per_action"] == 0.05
    assert config["time_step"] == 0.05
    assert config["frames_per_step"] == 8
    assert config["physics_update_timeout"] == 1.2
    assert config["state_poll_interval"] == 0.003
    assert config["load_world_on_start"] is True
    assert config["capture_live_start_pose"] is True


def test_build_training_stop_condition_uses_runtime_settings():
    """
    Standalone AirSim training should source its update budget from runtime settings.
    """

    from airsim.train.train_airsim import _build_training_stop_condition
    from fw.stop_condition import Criterion

    stop_condition = _build_training_stop_condition(
        {
            "training": {
                "max_policy_updates": 1750,
            },
            "airsim": {},
        }
    )

    assert stop_condition.stop_criterion == Criterion.NBR_POLICY_UPDATES
    assert stop_condition.max_time_steps == 1750


def test_get_runtime_settings_falls_back_to_defaults_for_invalid_json(tmp_path, monkeypatch):
    """
    Malformed runtime settings should not break AirSim config consumers.
    """

    import airsim.config.airsim_config as airsim_config

    runtime_path = tmp_path / "runtime_settings.json"
    runtime_path.write_text("{ invalid json", encoding="utf-8")
    monkeypatch.setattr(airsim_config, "_RUNTIME_SETTINGS_PATH", runtime_path)

    runtime = airsim_config.get_runtime_settings()

    assert runtime["training"]["max_steps"] == airsim_config.get_default_runtime_settings()["training"]["max_steps"]
    assert runtime["airsim"]["save_slot"] == airsim_config.DEFAULT_AIRSIM_SAVE_SLOT
