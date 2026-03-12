"""Tests for fallback behavior in the generic framework entrypoint."""

import runpy
import sys
from pathlib import Path


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def test_main_falls_back_when_airsim_extension_exits(monkeypatch, capsys):
    """Verify fw.main falls back to the default framework flow.

    When fw.config selects the AirSim extension but that extension exits
    with a non-zero status, the generic framework entrypoint should continue
    with its own hardcoded default flow instead of terminating early.
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

    trainer_calls = {}

    class _FakeTrainer:
        def __init__(self, *args, **kwargs):
            trainer_calls["constructed"] = True

        def train(self, configuration):
            trainer_calls["train_called"] = True
            trainer_calls["config_keys"] = set(configuration.keys())
            return {"status": "fallback-ok"}

    monkeypatch.setattr(trainer_module, "Trainer", _FakeTrainer)

    runpy.run_module("fw.main", run_name="__main__")

    captured = capsys.readouterr()
    assert "AirSim extension exited with an error" in captured.out
    assert "Training metrics" in captured.out
    assert trainer_calls["constructed"] is True
    assert trainer_calls["train_called"] is True
    assert env_factory.py_sim_env_name in trainer_calls["config_keys"]


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