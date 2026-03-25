"""Tests for AirSim-specific export helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_SRC_DIR = Path(__file__).resolve().parents[4] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fw.common.airsim.tools.airsim_export import AirSimSessionExporter, build_framework_visualisation_row


def test_airsim_session_exporter_writes_into_session_folder(tmp_path: Path) -> None:
    """AirSim session exporter should write telemetry and metadata to exports."""
    exporter = AirSimSessionExporter(
        session_name="demo_run",
        metadata={"model_path": "models/TestModel.zip", "environment_name": "AirSimEnv"},
        export_dir=tmp_path / "exports",
    )
    try:
        exporter.log_inference_step({"step": 1, "sim_time": 0.1, "x": 1.0, "y": 2.0})
        exporter.save()
        assert exporter.telemetry_path.exists()
        content = exporter.telemetry_path.read_text(encoding="utf-8")
        assert "step" in content
        assert "sim_time" in content
        assert exporter.bridge_log_path.parent == exporter.export_dir
        metadata = json.loads(exporter.metadata_path.read_text(encoding="utf-8"))
        assert metadata["session_name"].startswith("demo_run_")
        assert metadata["telemetry_file"] == exporter.telemetry_path.name
        assert metadata["airsim_bridge_log"] == exporter.bridge_log_path.name
    finally:
        exporter.close()


def test_airsim_visualisation_exporter_uses_visualisation_file_name(tmp_path: Path) -> None:
    """Visualisation sessions should write to a dedicated visualisation CSV."""
    exporter = AirSimSessionExporter(
        session_name="visualise_run",
        metadata={"model_path": "models/TestModel.zip", "environment_name": "PySimEnv"},
        export_dir=tmp_path / "exports",
        session_type="visualisation",
    )
    try:
        exporter.log_visualisation_step({"step": 1, "sim_time": 0.1, "x": 1.0, "y": 2.0})
        exporter.save()
        metadata = json.loads(exporter.metadata_path.read_text(encoding="utf-8"))
        assert metadata["session_type"] == "visualisation"
        assert metadata["telemetry_file"] == exporter.telemetry_path.name
    finally:
        exporter.close()


def test_airsim_session_exporter_context_manager(tmp_path: Path) -> None:
    """AirSimSessionExporter should cleanup when used as a context manager."""
    with AirSimSessionExporter(
        session_name="demo_run_context",
        metadata={"model_path": "models/TestModel.zip", "environment_name": "AirSimEnv"},
        export_dir=tmp_path / "exports",
    ) as exporter:
        exporter.log_inference_step({"step": 1, "sim_time": 0.1, "x": 1.0, "y": 2.0})
        # save() is called in __exit__, but can be explicit as well
        exporter.save()

    assert exporter.metadata_path.exists()
    assert exporter.telemetry_path.exists()


def test_build_framework_visualisation_row_handles_per_ship_lists() -> None:
    """Visualisation row builder should extract scalars from per-ship env lists."""
    env = SimpleNamespace(
        ship_pos=[[10.0, 20.0]],
        phys_sims=[SimpleNamespace(state=SimpleNamespace(u=3.5, heading=0.25))],
        time_step=0.1,
        checkpoint_index=[7],
        cross_error=[1.25],
    )
    bridge = SimpleNamespace(_controls=SimpleNamespace(brake=0.0))

    row = build_framework_visualisation_row(
        env=env,
        bridge=bridge,
        episode_index=1,
        step_index=5,
        reward=0.9,
        action=[0.1, 0.2],
        outcome="running",
    )

    assert row["checkpoint_index"] == 7
    assert row["cross_track_error"] == 1.25
    assert row["x"] == 10.0
    assert row["y"] == 20.0


def test_build_framework_visualisation_row_handles_empty_collections() -> None:
    """Safe extraction when env fields are empty or missing."""
    env = SimpleNamespace(
        ship_pos=[],
        phys_sims=[],
        time_step=0.1,
        checkpoint_index=[],
        cross_error=[],
    )
    bridge = SimpleNamespace(_controls=SimpleNamespace(brake=0.0))

    row = build_framework_visualisation_row(
        env=env,
        bridge=bridge,
        episode_index=1,
        step_index=1,
        reward=0.0,
        action=[0.0, 0.0],
        outcome="running",
    )

    assert row["x"] == 0.0
    assert row["y"] == 0.0
    assert row["checkpoint_index"] == 0
    assert row["cross_track_error"] == 0.0


def test_build_framework_visualisation_row_handles_mismatched_lengths() -> None:
    """Must handle length mismatch between ship_pos and phys_sims gracefully."""
    env = SimpleNamespace(
        ship_pos=[[10.0, 20.0]],
        phys_sims=[],
        time_step=0.1,
        checkpoint_index=[1],
        cross_error=[0.5],
    )
    bridge = SimpleNamespace(_controls=SimpleNamespace(brake=0.0))

    row = build_framework_visualisation_row(
        env=env,
        bridge=bridge,
        episode_index=1,
        step_index=1,
        reward=0.0,
        action=[0.0, 0.0],
        outcome="running",
    )

    assert row["x"] == 10.0
    assert row["y"] == 20.0
    assert row["checkpoint_index"] == 1
    assert row["cross_track_error"] == 0.5


def test_build_framework_visualisation_row_handles_missing_state_bits() -> None:
    """Missing state attrs are handled with defaults."""
    env = SimpleNamespace(
        ship_pos=[[10.0, 20.0]],
        phys_sims=[SimpleNamespace(state=SimpleNamespace())],
        time_step=0.1,
        checkpoint_index=[1],
        cross_error=[0.5],
    )
    bridge = SimpleNamespace(_controls=SimpleNamespace(brake=0.0))

    row = build_framework_visualisation_row(
        env=env,
        bridge=bridge,
        episode_index=1,
        step_index=1,
        reward=0.0,
        action=[0.0, 0.0],
        outcome="running",
    )

    assert row["x"] == 10.0
    assert row["y"] == 20.0
    assert row["heading_deg"] == 0.0
    assert row["speed"] == 0.0
