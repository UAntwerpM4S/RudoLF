import pytest

from fw.config import PPO_POLICY_NAME, get_hyperparameters
from fw.env_factory import (
    _Factory,
    EnvironmentRegistryError,
    airsim_env_name,
    build_default_environment,
    get as get_environment,
    py_sim_env_name,
    py_sim_polar_env_name,
    resolve_environment_name,
    resolve_environment_names,
)


def test_resolve_environment_name_defaults_to_pysim():
    """Test behavior for resolve environment name defaults to pysim.
    """
    assert resolve_environment_name(None) == py_sim_env_name
    assert resolve_environment_name("") == py_sim_env_name


def test_resolve_environment_name_accepts_aliases():
    """Test behavior for resolve environment name accepts aliases.
    """
    assert resolve_environment_name("pysim") == py_sim_env_name
    assert resolve_environment_name("AirSim") == airsim_env_name
    assert resolve_environment_name("polar") == py_sim_polar_env_name


def test_resolve_environment_name_rejects_unknown_alias():
    """Test behavior for resolve environment name rejects unknown alias.
    """
    with pytest.raises(EnvironmentRegistryError, match="Unknown env_type"):
        resolve_environment_name("not-a-real-env")


def test_resolve_environment_names_preserves_order():
    """Test behavior for resolve environment names preserves order.
    """
    names = resolve_environment_names(["PySimEnv", "AirSimEnv", "LunarLander"])
    assert names == ["PySimEnv", "AirSimEnv", "LunarLander"]


def test_resolve_environment_names_preserves_repeated_stages():
    """Test behavior for resolve environment names preserves repeated stages.
    """
    names = resolve_environment_names(["PySimEnv", "AirSimEnv", "PySimEnv"])
    assert names == ["PySimEnv", "AirSimEnv", "PySimEnv"]


def test_build_default_environment_accepts_legacy_empty_list_configurations():
    """Test behavior for build default environment accepts legacy empty list configurations.
    """
    get_environment(py_sim_env_name, configuration=[])

    environment = build_default_environment(py_sim_env_name)

    assert environment is get_environment(py_sim_env_name)
    assert dict(environment.configuration) == {}


def test_get_hyperparameters_keeps_shared_defaults_for_non_airsim_envs():
    """PySim/FhSim should keep the shared PPO defaults."""
    params = get_hyperparameters(PPO_POLICY_NAME)

    assert params["learning_rate"] == pytest.approx(1e-4)
    assert params["gamma"] == pytest.approx(0.95)
    assert params["clip_range"] == pytest.approx(0.05)


def test_get_hyperparameters_uses_airsim_specific_overrides():
    """AirSimEnv should get its own PPO tuning set."""
    params = get_hyperparameters(PPO_POLICY_NAME, env_name=airsim_env_name)

    assert params["learning_rate"] == pytest.approx(3e-4)
    assert params["gamma"] == pytest.approx(0.99)
    assert params["clip_range"] == pytest.approx(0.2)



def test_get_hyperparameters_uses_override_for_non_airsim_env_names():
    """Default PPO settings should be applied outside AirSim."""
    params = get_hyperparameters(PPO_POLICY_NAME, env_name=py_sim_env_name)

    assert params["learning_rate"] == pytest.approx(1e-4)
    assert params["gamma"] == pytest.approx(0.95)
    assert params["clip_range"] == pytest.approx(0.05)


def test_airsim_environment_is_not_singleton_cached(monkeypatch):
    """Test behavior for airsim environment is not singleton cached.

    Args:
        monkeypatch: Description.
    """
    created = []

    class _FakeAirSimEnvironment:
        """ FakeAirSimEnvironment."""
        _cache_instance = False

        def __init__(self, name, configuration):
            """Initialize the instance.

            Args:
                name: Description.
                configuration: Description.
            """
            self._name = name
            self._configuration = dict(configuration)
            self._env = object()
            created.append(self)

        @property
        def configuration(self):
            """Configuration.
            """
            return self._configuration

    monkeypatch.setitem(_Factory._environment_classes, airsim_env_name, _FakeAirSimEnvironment)
    _Factory._environment_instances.pop(airsim_env_name, None)

    first = get_environment(airsim_env_name, configuration={"save_slot": "slot_a"})
    second = get_environment(airsim_env_name, configuration={"save_slot": "slot_b"})

    assert first is not second
    assert len(created) == 2
    assert airsim_env_name not in _Factory._environment_instances


def test_airsim_environment_receives_full_runtime_configuration(monkeypatch):
    """Test behavior for airsim environment receives full runtime configuration.

    Args:
        monkeypatch: Description.
    """
    captured = {}

    class _FakeAirSimEnv:
        """ FakeAirSimEnv."""
        def __init__(self, **kwargs):
            """Initialize the instance.

            Args:
                **kwargs: Arbitrary runtime configuration arguments.
            """
            captured.update(kwargs)

    _Factory._environment_instances.pop(airsim_env_name, None)
    monkeypatch.setattr("fw.env_factory.AirSimEnv", _FakeAirSimEnv)

    config = {
        "training_mode": False,
        "use_pause_stepping": False,
        "use_airsim": True,
        "ip": "127.0.0.1",
        "port": 41451,
        "timeout": 2.5,
        "max_retries": 7,
        "health_check_interval": 12.0,
        "csv_log_path": "bridge.csv",
        "save_slot": "slot_two",
        "load_world_on_start": False,
        "capture_live_start_pose": True,
        "simulation_speed": 3.0,
        "max_steps": 321,
        "normalize_reward": False,
        "reward_normalization_scale": 90.0,
        "reset_mode": "bridge_reset",
    }

    get_environment(airsim_env_name, configuration=config)

    assert captured["use_airsim"] is True
    assert captured["ip"] == "127.0.0.1"
    assert captured["port"] == 41451
    assert captured["timeout"] == 2.5
    assert captured["max_retries"] == 7
    assert captured["health_check_interval"] == 12.0
    assert captured["csv_log_path"] == "bridge.csv"
    assert captured["save_slot"] == "slot_two"
    assert captured["load_world_on_start"] is False
    assert captured["capture_live_start_pose"] is True
    assert captured["simulation_speed"] == 3.0
    assert captured["max_steps"] == 321
    assert captured["normalize_reward"] is False
    assert captured["reward_normalization_scale"] == 90.0
    assert captured["reset_mode"] == "bridge_reset"

