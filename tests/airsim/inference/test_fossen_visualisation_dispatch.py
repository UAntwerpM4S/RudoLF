"""Tests for AirSim Fossen-visualisation dispatch through fw.main_with_airsim."""

import runpy

import pytest


def test_main_with_airsim_dispatches_fossen_visualisation(monkeypatch):
    """Run fw.main_with_airsim and verify it routes to run_visualise_fossen.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Verifies:
        - runtime settings select AirSim + visualize + fossen_visualise.
        - run_visualise_fossen receives expected arguments.
        - main_with_airsim exits early with code 0 after dispatch.
        - run_inference is not called in this mode.
    """
    runtime = {
        "main": {
            "action": "visualize",
            "train_environment": "AirSimEnv",
        },
        "visualization": {
            "fossen_visualise": True,
            "inference_visualise": False,
            "fossen_model_path": "src/models/PySimEnv_PPO_policy.zip",
            "fossen_environment": "auto",
            "episodes": 2,
            "max_steps": 123,
            "device": "cpu",
            "log_every": 7,
            "draw_calibration_reference": True,
            "mirror_controls_in_ue5": False,
            "animate_steering_in_ue5": True,
            "animate_throttle_in_ue5": False,
            "render_interpolation_steps": 8,
            "steering_visual_gain": 0.35,
            "throttle_visual_gain": 1.0,
            "use_kinematics_in_ue5": False,
            "min_render_hz": 90.0,
            "max_translation_step": 0.15,
            "max_yaw_step_deg": 1.5,
            "lock_path_in_ue5": False,
            "vehicle_object_name": "",
            "show_minimap": True,
        },
        "airsim": {
            "ip": "",
            "port": 41451,
            "save_slot": "simulator_1",
            "simulation_speed": 10.0,
        },
    }

    import airsim.config.airsim_config as airsim_config
    monkeypatch.setattr(airsim_config, "get_runtime_settings", lambda: runtime)

    called = {}

    import airsim.inference.run_visualise_fossen as run_visualise_fossen_module

    def _fake_run_visualise_fossen(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(run_visualise_fossen_module, "run_visualise_fossen", _fake_run_visualise_fossen)

    import airsim.inference.run_inference as run_inference_module

    def _fail_run_inference(**kwargs):
        raise AssertionError("run_inference should not be called when fossen_visualise=True")

    monkeypatch.setattr(run_inference_module, "run_inference", _fail_run_inference)

    import multiprocessing
    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("fw.main_with_airsim", run_name="__main__")

    assert exc.value.code == 0
    assert called["model_path"] == "src/models/PySimEnv_PPO_policy.zip"
    assert called["env_name"] == "auto"
    assert called["port"] == 41451
    assert called["num_episodes"] == 2
    assert called["max_steps"] == 123
    assert called["simulation_speed"] == 10.0
