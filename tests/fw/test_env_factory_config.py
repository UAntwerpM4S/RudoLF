import pytest

from fw.env_factory import (
    _Factory,
    EnvironmentRegistryError,
    airsim_env_name,
    build_default_environment,
    get as get_environment,
    py_sim_env_name,
    py_sim_polar_env_name,
    resolve_environment_name,
)


def test_resolve_environment_name_defaults_to_pysim():
    assert resolve_environment_name(None) == py_sim_env_name
    assert resolve_environment_name("") == py_sim_env_name


def test_resolve_environment_name_accepts_aliases():
    assert resolve_environment_name("pysim") == py_sim_env_name
    assert resolve_environment_name("AirSim") == airsim_env_name
    assert resolve_environment_name("polar") == py_sim_polar_env_name


def test_resolve_environment_name_rejects_unknown_alias():
    with pytest.raises(EnvironmentRegistryError, match="Unknown env_type"):
        resolve_environment_name("not-a-real-env")


def test_build_default_environment_accepts_legacy_empty_list_configurations():
    get_environment(py_sim_env_name, configuration=[])

    environment = build_default_environment(py_sim_env_name)

    assert environment is get_environment(py_sim_env_name)
    assert dict(environment.configuration) == {}


def test_airsim_environment_is_not_singleton_cached(monkeypatch):
    created = []

    class _FakeAirSimEnvironment:
        _cache_instance = False

        def __init__(self, name, configuration):
            self._name = name
            self._configuration = dict(configuration)
            self._env = object()
            created.append(self)

        @property
        def configuration(self):
            return self._configuration

    monkeypatch.setitem(_Factory._environment_classes, airsim_env_name, _FakeAirSimEnvironment)
    _Factory._environment_instances.pop(airsim_env_name, None)

    first = get_environment(airsim_env_name, configuration={"save_slot": "slot_a"})
    second = get_environment(airsim_env_name, configuration={"save_slot": "slot_b"})

    assert first is not second
    assert len(created) == 2
    assert airsim_env_name not in _Factory._environment_instances


def test_airsim_environment_receives_full_runtime_configuration(monkeypatch):
    captured = {}

    class _FakeAirSimEnv:
        def __init__(self, **kwargs):
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
        "simulation_speed": 3.0,
        "max_steps": 321,
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
    assert captured["simulation_speed"] == 3.0
    assert captured["max_steps"] == 321