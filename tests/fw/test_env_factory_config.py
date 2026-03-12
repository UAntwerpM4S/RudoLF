import pytest

from fw.env_factory import (
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