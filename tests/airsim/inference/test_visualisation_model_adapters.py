"""Tests for modular model-visualisation adapter resolution."""

import pytest

from airsim.inference.visualisation_models.model_adapters import (
    FhSimVisualisationAdapter,
    FossenVisualisationAdapter,
    resolve_visualisation_adapter,
)


def test_auto_detects_fhsim_adapter_from_model_name():
    """FHSim model names should select the FHSim adapter automatically."""
    adapter = resolve_visualisation_adapter(
        model_family="auto",
        model_path="models/FhSimEnv_PPO_policy.zip",
        env_name="auto",
    )
    assert isinstance(adapter, FhSimVisualisationAdapter)


def test_auto_detects_fossen_adapter_from_model_name():
    """PySim/Fossen model names should select the Fossen adapter automatically."""
    adapter = resolve_visualisation_adapter(
        model_family="auto",
        model_path="models/PySimEnv_PPO_policy.zip",
        env_name="auto",
    )
    assert isinstance(adapter, FossenVisualisationAdapter)


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
