"""Tests for fallback behavior in the generic framework entrypoint."""

import runpy
import sys
import os
from pathlib import Path

import pytest


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def test_main_falls_back_when_airsim_extension_exits(monkeypatch, capsys):
    """Verify fw.main exits cleanly when the AirSim extension fails and no
    other environments are configured.

    When fw.config selects only the AirSim environment but the extension
    exits with a non-zero status, the generic framework entrypoint should
    raise SystemExit(1) rather than continuing with an empty environment list.
    """
    import multiprocessing
    import runpy as stdlib_runpy

    import fw.config as fw_config
    import fw.env_factory as env_factory
    import fw.trainer as trainer_module

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {"env_type": env_factory.airsim_env_name, "action": "train"},
    )

    original_run_module = stdlib_runpy.run_module

    def _failing_run_module(module_name, run_name=None):
        if module_name == "fw.main_with_airsim":
            raise SystemExit(1)
        return original_run_module(module_name, run_name=run_name)

    monkeypatch.setattr(stdlib_runpy, "run_module", _failing_run_module)
    monkeypatch.setattr(env_factory, "initialize_all_environments", lambda: None)
    monkeypatch.setattr(env_factory, "get", lambda *args, **kwargs: object())
    monkeypatch.setattr(trainer_module, "Trainer", lambda *a, **kw: None)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("fw.main", run_name="__main__")

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "AirSim extension exited with an error" in captured.out
    assert "No environments remaining" in captured.out


def test_main_uses_framework_visualize_action(monkeypatch):
    """Verify fw.main honors framework config action='visualize'."""
    import multiprocessing

    import fw.config as fw_config
    import fw.env_factory as env_factory
    import fw.trainer as trainer_module

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {"env_type": env_factory.py_sim_env_name, "action": "visualize"},
    )
    monkeypatch.setattr(env_factory, "initialize_all_environments", lambda: None)
    monkeypatch.setattr(env_factory, "build_default_environment", lambda *args, **kwargs: object())

    trainer_calls = {}

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            trainer_calls["constructed"] = True

        def train(self, configuration):
            trainer_calls["train_called"] = True
            return {"status": "unexpected-train"}

        def visualize(self, envs, airsim_bridge=None):
            trainer_calls["visualize_called"] = True
            trainer_calls["airsim_bridge"] = airsim_bridge

    monkeypatch.setattr(trainer_module, "Trainer", _FakeTrainer)

    runpy.run_module("fw.main", run_name="__main__")

    assert trainer_calls["constructed"] is True
    assert trainer_calls.get("train_called") is not True
    assert trainer_calls["visualize_called"] is True
    assert trainer_calls["airsim_bridge"] is None


def test_main_passes_action_override_to_airsim_extension(monkeypatch):
    """Verify fw.main explicitly forwards the framework action to AirSim."""
    import multiprocessing
    import runpy as stdlib_runpy

    import fw.config as fw_config
    import fw.env_factory as env_factory

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {"env_type": env_factory.airsim_env_name, "action": "visualize"},
    )
    original_run_module = stdlib_runpy.run_module

    captured = {}

    def _capture_run_module(module_name, run_name=None):
        if module_name == "fw.main_with_airsim":
            captured["module_name"] = module_name
            captured["forwarded_action"] = os.environ.get("FW_MAIN_ACTION_OVERRIDE")
            raise SystemExit(0)
        return original_run_module(module_name, run_name=run_name)

    monkeypatch.setattr(stdlib_runpy, "run_module", _capture_run_module)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("fw.main", run_name="__main__")

    assert exc.value.code == 0
    assert captured["module_name"] == "fw.main_with_airsim"
    assert captured["forwarded_action"] == "visualize"
    assert "FW_MAIN_ACTION_OVERRIDE" not in os.environ