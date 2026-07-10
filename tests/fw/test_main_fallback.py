"""
Tests for fallback behavior in the generic framework entrypoint.
"""

import runpy
import sys
import os
from pathlib import Path

import pytest
from types import SimpleNamespace


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def test_main_falls_back_when_airsim_extension_exits(monkeypatch, capsys):
    """
    Verify fw.main exits cleanly when the AirSim extension fails and no
    other environments are configured.

    When fw.config selects only the AirSim environment but the extension
    exits with a non-zero status, the generic framework entrypoint should
    raise SystemExit(1) rather than continuing with an empty environment list.
    """

    import multiprocessing
    import runpy as stdlib_runpy

    import fw.configuration.config as fw_config
    import fw.configuration.env_factory as env_factory
    import fw.scheduling.trainer as trainer_module

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {"env_type": env_factory.airsim_env_name, "action": "train"},
    )

    original_run_module = stdlib_runpy.run_module

    def _failing_run_module(module_name, run_name=None):
        """
        Raise a non-zero exit for fw.main_with_airsim to simulate failure.
        """

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
    """
    Verify fw.main honors framework config action='visualize'.
    """

    import multiprocessing

    import fw.configuration.config as fw_config
    import fw.configuration.env_factory as env_factory
    import fw.scheduling.trainer as trainer_module

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {"env_type": env_factory.py_sim_env_name, "action": "visualize"},
    )
    monkeypatch.setattr(env_factory, "initialize_all_environments", lambda: None)
    monkeypatch.setattr(
        env_factory,
        "build_default_environment",
        lambda name, *args, **kwargs: SimpleNamespace(gym_env=SimpleNamespace(type_name=name, training_name=name)),
    )

    trainer_calls = {}

    class _FakeTrainer:
        """
        Minimal trainer test double used to capture call flow.
        """

        def __init__(self, *args, **kwargs):
            """
            Record constructor usage.
            """

            trainer_calls["constructed"] = True

        def train(self, configuration):
            """
            Record unexpected train invocation in visualize mode.
            """

            trainer_calls["train_called"] = True
            return {"status": "unexpected-train"}

        def visualize(self, envs, airsim_bridge=None):
            """
            Record visualize invocation and forwarded bridge argument.
            """

            trainer_calls["visualize_called"] = True
            trainer_calls["airsim_bridge"] = airsim_bridge

    monkeypatch.setattr(trainer_module, "Trainer", _FakeTrainer)

    runpy.run_module("fw.main", run_name="__main__")

    assert trainer_calls["constructed"] is True
    assert trainer_calls.get("train_called") is not True
    assert trainer_calls["visualize_called"] is True
    assert trainer_calls["airsim_bridge"] is None


def test_main_passes_action_override_to_airsim_extension(monkeypatch):
    """
    Verify fw.main explicitly forwards the framework action to AirSim.
    """

    import multiprocessing
    import runpy as stdlib_runpy

    import fw.configuration.config as fw_config
    import fw.configuration.env_factory as env_factory

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {"env_type": env_factory.airsim_env_name, "action": "visualize"},
    )
    original_run_module = stdlib_runpy.run_module

    captured = {}

    def _capture_run_module(module_name, run_name=None):
        """
        Capture forwarded action when fw.main delegates to AirSim.
        """

        if module_name == "fw.main_with_airsim":
            captured["module_name"] = module_name
            captured["forwarded_action"] = os.environ.get("FW_MAIN_ACTION_OVERRIDE")
            raise SystemExit(0)
        return original_run_module(module_name, run_name=run_name)

    monkeypatch.setattr(stdlib_runpy, "run_module", _capture_run_module)

    runpy.run_module("fw.main", run_name="__main__")

    assert captured["module_name"] == "fw.main_with_airsim"
    assert captured["forwarded_action"] == "visualize"
    assert "FW_MAIN_ACTION_OVERRIDE" not in os.environ


def test_main_visualize_respects_environment_order(monkeypatch):
    """
    Verify visualize runs in env_type order and includes AirSim in-place.
    """

    import multiprocessing
    import runpy as stdlib_runpy

    import fw.configuration.config as fw_config
    import fw.configuration.env_factory as env_factory
    import fw.scheduling.trainer as trainer_module

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {
            "env_type": [env_factory.py_sim_env_name, env_factory.airsim_env_name],
            "action": "visualize",
        },
    )
    monkeypatch.setattr(env_factory, "initialize_all_environments", lambda: None)

    call_order = []

    def _build_env(name, framework_config=None):
        """
        Track framework environment construction order.
        """

        call_order.append(f"build:{name}")
        return SimpleNamespace(gym_env=SimpleNamespace(type_name=name, training_name=name))

    monkeypatch.setattr(env_factory, "build_default_environment", _build_env)

    class _FakeTrainer:
        """
        Test double for framework visualize execution.
        """

        def __init__(self, *args, **kwargs):
            """
            No-op constructor for parity with Trainer API.
            """

            pass

        def visualize(self, envs, airsim_bridge=None):
            """
            Record framework visualize call in ordered execution.
            """

            call_order.append("visualize:framework")

    monkeypatch.setattr(trainer_module, "Trainer", _FakeTrainer)

    original_run_module = stdlib_runpy.run_module

    def _capture_run_module(module_name, run_name=None):
        """
        Inject ordered AirSim visualize step signal.
        """

        if module_name == "fw.main_with_airsim":
            call_order.append("visualize:airsim")
            raise SystemExit(0)
        return original_run_module(module_name, run_name=run_name)

    monkeypatch.setattr(stdlib_runpy, "run_module", _capture_run_module)

    runpy.run_module("fw.main", run_name="__main__")

    assert call_order == [
        f"build:{env_factory.py_sim_env_name}",
        "visualize:framework",
        "visualize:airsim",
    ]


def test_main_train_respects_environment_order(monkeypatch):
    """
    Verify train runs in env_type order and keeps AirSim in sequence.
    """

    import multiprocessing
    import runpy as stdlib_runpy

    import fw.configuration.config as fw_config
    import fw.configuration.env_factory as env_factory
    import fw.scheduling.trainer as trainer_module

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {
            "env_type": [env_factory.py_sim_env_name, env_factory.airsim_env_name],
            "action": "train",
        },
    )
    monkeypatch.setattr(env_factory, "initialize_all_environments", lambda: None)

    call_order = []

    def _build_env(name, framework_config=None):
        """
        Track framework train environment construction order.
        """

        call_order.append(f"build:{name}")
        return SimpleNamespace(gym_env=SimpleNamespace(type_name=name, training_name=name))

    monkeypatch.setattr(env_factory, "build_default_environment", _build_env)

    class _FakeTrainer:
        """
        Test double for framework training execution.
        """

        def __init__(self, *args, **kwargs):
            """
            No-op constructor for parity with Trainer API.
            """

            pass

        def train(self, configuration):
            """
            Record framework train call in ordered execution.
            """

            call_order.append("train:framework")
            return {}

    monkeypatch.setattr(trainer_module, "Trainer", _FakeTrainer)

    original_run_module = stdlib_runpy.run_module

    def _capture_run_module(module_name, run_name=None):
        """
        Inject ordered AirSim train step signal.
        """

        if module_name == "fw.main_with_airsim":
            call_order.append("train:airsim")
            raise SystemExit(0)
        return original_run_module(module_name, run_name=run_name)

    monkeypatch.setattr(stdlib_runpy, "run_module", _capture_run_module)

    runpy.run_module("fw.main", run_name="__main__")

    assert call_order == [
        f"build:{env_factory.py_sim_env_name}",
        "train:framework",
        "train:airsim",
    ]


def test_main_train_mixed_list_skips_failed_airsim_step(monkeypatch):
    """
    Verify mixed env training continues without duplicating fallback env work.

    If the delegated AirSim step exits non-zero, fw.main should keep already
    completed framework training and not perform an extra fallback training pass.
    """

    import multiprocessing
    import runpy as stdlib_runpy

    import fw.configuration.config as fw_config
    import fw.configuration.env_factory as env_factory
    import fw.scheduling.trainer as trainer_module

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {
            "env_type": [env_factory.py_sim_env_name, env_factory.airsim_env_name],
            "action": "train",
        },
    )
    monkeypatch.setattr(env_factory, "initialize_all_environments", lambda: None)

    call_order = []

    def _build_env(name, framework_config=None):
        """
        Track framework train setup for mixed-list fallback scenario.
        """

        call_order.append(f"build:{name}")
        return SimpleNamespace(gym_env=SimpleNamespace(type_name=name, training_name=name))

    monkeypatch.setattr(env_factory, "build_default_environment", _build_env)

    class _FakeTrainer:
        """
        Test double for framework training in fallback scenario.
        """

        def __init__(self, *args, **kwargs):
            """
            No-op constructor for parity with Trainer API.
            """

            pass

        def train(self, configuration):
            """
            Record framework train call before AirSim step failure.
            """

            call_order.append("train:framework")
            return {}

    monkeypatch.setattr(trainer_module, "Trainer", _FakeTrainer)

    original_run_module = stdlib_runpy.run_module

    def _capture_run_module(module_name, run_name=None):
        """
        Simulate AirSim step failure without replaying framework training.
        """

        if module_name == "fw.main_with_airsim":
            call_order.append("train:airsim:failed")
            raise SystemExit(1)
        return original_run_module(module_name, run_name=run_name)

    monkeypatch.setattr(stdlib_runpy, "run_module", _capture_run_module)

    runpy.run_module("fw.main", run_name="__main__")

    assert call_order == [
        f"build:{env_factory.py_sim_env_name}",
        "train:framework",
        "train:airsim:failed",
    ]


def test_main_train_repeated_env_sequence_continues_after_failed_airsim_step(monkeypatch):
    """
    Verify repeated list stages are preserved around a failed AirSim step.

    Expected order for env_type=[PySimEnv, AirSimEnv, PySimEnv]:
    1) first PySim framework stage trains
    2) AirSim delegated stage fails and is skipped
    3) second PySim framework stage still trains
    """

    import multiprocessing
    import runpy as stdlib_runpy

    import fw.configuration.config as fw_config
    import fw.configuration.env_factory as env_factory
    import fw.scheduling.trainer as trainer_module

    monkeypatch.setattr(multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fw_config,
        "get_config_parameters",
        lambda: {
            "env_type": [
                env_factory.py_sim_env_name,
                env_factory.airsim_env_name,
                env_factory.py_sim_env_name,
            ],
            "action": "train",
        },
    )
    monkeypatch.setattr(env_factory, "initialize_all_environments", lambda: None)

    call_order = []

    def _build_env(name, framework_config=None):
        """
        Build a fake environment and log creation order.

        Args:
            name: Canonical environment name.
            framework_config: Framework configuration payload forwarded by main.
        """

        call_order.append(f"build:{name}")
        return SimpleNamespace(gym_env=SimpleNamespace(type_name=name, training_name=name))

    monkeypatch.setattr(env_factory, "build_default_environment", _build_env)

    class _FakeTrainer:
        """
        FakeTrainer.
        """

        def __init__(self, *args, **kwargs):
            """
            Create a no-op trainer test double.

            Args:
                *args: Positional constructor arguments.
                **kwargs: Keyword constructor arguments.
            """

            pass

        def train(self, configuration):
            """
            Record framework training invocation.

            Args:
                configuration: Scheduler configuration payload.
            """

            call_order.append("train:framework")
            return {}

    monkeypatch.setattr(trainer_module, "Trainer", _FakeTrainer)

    original_run_module = stdlib_runpy.run_module

    def _capture_run_module(module_name, run_name=None):
        """
        Simulate a failing AirSim delegated run.

        Args:
            module_name: Name of the module being executed.
            run_name: Run target name passed to run_module.
        """

        if module_name == "fw.main_with_airsim":
            call_order.append("train:airsim:failed")
            raise SystemExit(1)
        return original_run_module(module_name, run_name=run_name)

    monkeypatch.setattr(stdlib_runpy, "run_module", _capture_run_module)

    runpy.run_module("fw.main", run_name="__main__")

    assert call_order == [
        f"build:{env_factory.py_sim_env_name}",
        "train:framework",
        "train:airsim:failed",
        f"build:{env_factory.py_sim_env_name}",
        "train:framework",
    ]

