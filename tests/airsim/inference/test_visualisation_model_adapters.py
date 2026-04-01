"""Tests for modular model-visualisation adapters."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import airsim.inference.visualisation_models.model_adapters as module
from airsim.inference.visualisation_models.model_adapters import (
    FhSimVisualisationAdapter,
    PySimVisualisationAdapter,
    get_supported_environment_names,
    get_supported_model_families,
    resolve_visualisation_adapter,
)


class _FakeCoordTransform:
    """Small coordinate-transform stub for adapter motion-state tests."""

    def fossen_to_airsim(self, x: float, y: float, z: float):
        return x + 100.0, y - 50.0, z

    def fossen_heading_to_airsim_yaw(self, heading: float) -> float:
        return heading + 0.25

    def fossen_velocity_to_airsim(self, vx: float, vy: float):
        return vx + 1.0, vy - 2.0

    def fossen_yaw_rate_to_airsim(self, yaw_rate: float) -> float:
        return yaw_rate - 0.1


def _framework_env(
    *,
    ship_pos=None,
    heading: float = 0.3,
    u: float = 4.0,
    v: float = 1.5,
    r: float = 0.2,
    performed_action=None,
):
    """Build a lightweight framework-environment double for adapter tests."""
    return SimpleNamespace(
        ship_pos=np.array([12.0, 34.0], dtype=np.float32) if ship_pos is None else ship_pos,
        phys_sims=[SimpleNamespace(state=SimpleNamespace(heading=heading, u=u, v=v, r=r))],
        performed_action=performed_action,
        ships=[SimpleNamespace(specifications=SimpleNamespace(max_rudder_angle=1.0, min_thrust=-1.0, max_thrust=1.0))],
    )


def _bridge(*, connected: bool = True):
    """Build a lightweight bridge double for adapter tests."""
    return SimpleNamespace(
        coord_transform=_FakeCoordTransform(),
        render_z=-7.5,
        is_connected=connected,
        airsim=SimpleNamespace(CarControls=lambda: SimpleNamespace()),
        set_pose=lambda *args, **kwargs: None,
        set_kinematics=lambda *args, **kwargs: None,
        set_object_pose=lambda *args, **kwargs: None,
        set_controls=lambda *args, **kwargs: None,
        get_vehicle_pose=lambda: None,
    )


def test_supported_model_families_are_exposed():
    """The registry should expose the public adapter families."""
    assert get_supported_model_families() == ("pysim", "fhsim")


def test_supported_environment_names_are_exposed():
    """The registry should expose the registered framework environments."""
    assert get_supported_environment_names() == ("FhSimEnv", "PySimEnv")


def test_auto_detects_fhsim_adapter_from_model_name():
    """FHSim model names should select the FHSim adapter automatically."""
    adapter = resolve_visualisation_adapter(
        model_family="auto",
        model_path="models/FhSimEnv_PPO_policy.zip",
        env_name="auto",
    )
    assert isinstance(adapter, FhSimVisualisationAdapter)


def test_auto_detects_pysim_adapter_from_model_name():
    """PySim model names should select the PySim adapter automatically."""
    adapter = resolve_visualisation_adapter(
        model_family="auto",
        model_path="models/PySimEnv_PPO_policy.zip",
        env_name="auto",
    )
    assert isinstance(adapter, PySimVisualisationAdapter)


def test_explicit_environment_selects_matching_adapter():
    """A concrete framework environment should pick the owning adapter."""
    adapter = resolve_visualisation_adapter(
        model_family="auto",
        model_path="models/custom_policy.zip",
        env_name="FhSimEnv",
    )
    assert isinstance(adapter, FhSimVisualisationAdapter)


def test_fhsim_adapter_rejects_pysim_environment():
    """Conflicting family/environment combinations should fail loudly."""
    adapter = resolve_visualisation_adapter(
        model_family="fhsim",
        model_path="models/FhSimEnv_PPO_policy.zip",
        env_name="auto",
    )
    with pytest.raises(ValueError):
        adapter.resolve_environment_name("models/FhSimEnv_PPO_policy.zip", "PySimEnv")


def test_single_environment_adapter_resolves_auto_mode():
    """Adapters with one supported env should resolve ``auto`` themselves."""
    adapter = PySimVisualisationAdapter()
    assert adapter.resolve_environment_name("models/PySimEnv_PPO_policy.zip", "auto") == "PySimEnv"


def test_pysim_adapter_build_environment_uses_shared_defaults(monkeypatch):
    """Environment construction should use the adapter factory surface."""
    captured = {}

    def _fake_pysim_env(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name="fake-pysim-env")

    monkeypatch.setattr(module, "PySimEnv", _fake_pysim_env)

    adapter = PySimVisualisationAdapter()
    env = adapter.build_environment("PySimEnv", max_steps=321)

    assert env.name == "fake-pysim-env"
    assert captured["max_steps"] == 321
    assert captured["start_pos"] == [2245.0, -47.0]
    assert captured["target_pos"] == [-1559.0, 11990.0]


def test_motion_state_from_env_supports_flat_ship_position():
    """Adapter motion extraction should not require ``ship_pos[0]``."""
    adapter = PySimVisualisationAdapter()
    env = _framework_env(ship_pos=np.array([12.0, 34.0], dtype=np.float32))

    motion_state = adapter.motion_state_from_env(env, _bridge())

    expected_vx = 4.0 * np.cos(0.3) - 1.5 * np.sin(0.3)
    expected_vy = 4.0 * np.sin(0.3) + 1.5 * np.cos(0.3)
    np.testing.assert_allclose(
        motion_state,
        np.array(
            [
                112.0,
                -16.0,
                0.55,
                expected_vx + 1.0,
                expected_vy - 2.0,
                0.1,
            ],
            dtype=np.float64,
        ),
    )


def test_extract_actuator_state_supports_flat_action_vector():
    """Actuator extraction should not require ``performed_action[0]``."""
    adapter = PySimVisualisationAdapter()
    env = _framework_env(performed_action=np.array([0.15, -0.4], dtype=np.float32))

    actuator_state = adapter.extract_actuator_state(env)

    np.testing.assert_allclose(actuator_state, np.array([0.15, -0.4], dtype=np.float32))


def test_animate_state_returns_flat_action_vector_when_bridge_disconnects():
    """Disconnect-safe animation should still propagate the latest actuator state."""
    adapter = PySimVisualisationAdapter()
    env = _framework_env(performed_action=np.array([0.25, 0.75], dtype=np.float32))
    target_motion_state = np.array([1.0, 2.0, 0.3, 4.0, 5.0, 0.6], dtype=np.float64)

    motion_state, actuator_state = adapter.animate_state(
        env,
        _bridge(connected=False),
        target_motion_state=target_motion_state,
        previous_motion_state=None,
        total_duration=0.0,
        interpolation_steps=1,
        mirror_controls_in_ue5=False,
        animate_steering_in_ue5=False,
        animate_throttle_in_ue5=False,
        previous_actuator_state=None,
    )

    np.testing.assert_allclose(motion_state, target_motion_state)
    np.testing.assert_allclose(actuator_state, np.array([0.25, 0.75], dtype=np.float32))
