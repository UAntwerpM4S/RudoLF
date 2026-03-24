"""Unit tests for the common_tools TelemetryExporter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common_tools.telemetry_exporter import TelemetryExporter


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    """Provide a clean temporary export directory."""
    path = tmp_path / "exports"
    path.mkdir()
    return path


class TestTelemetryExporter:
    """Tests for TelemetryExporter CSV and metadata output."""

    def test_csv_created_with_header_on_init(self, export_dir: Path) -> None:
        """CSV file is created with a header row immediately on init."""
        exporter = TelemetryExporter(
            session_type="inference",
            export_dir=export_dir,
        )
        try:
            header = exporter.csv_path.read_text(encoding="utf-8").splitlines()[0]
            assert "step" in header
            assert "sim_time" in header
        finally:
            exporter.close()

    def test_log_step_and_flush(self, export_dir: Path) -> None:
        """Logged steps appear in the CSV after save()."""
        exporter = TelemetryExporter(
            session_type="visualise",
            export_dir=export_dir,
            flush_interval=2,
        )
        try:
            exporter.log_step(step=1, sim_time=0.1, x=1.0, y=2.0, reward=3.0)
            exporter.save()
            rows = exporter.csv_path.read_text(encoding="utf-8").splitlines()
            assert len(rows) == 2
            assert "1" in rows[1]
            assert "3.0" in rows[1]
        finally:
            exporter.close()

    def test_metadata_json_created(self, export_dir: Path) -> None:
        """Metadata JSON contains expected keys after save()."""
        exporter = TelemetryExporter(
            session_type="train",
            model_path="models/TestModel.zip",
            environment_name="PySimEnv",
            config_dict={"simulation_speed": 20.0},
            export_dir=export_dir,
        )
        try:
            exporter.log_step(step=1, sim_time=0.1)
            exporter.save()
            metadata = json.loads(exporter.metadata_path.read_text(encoding="utf-8"))
            assert metadata["session_type"] == "train"
            assert metadata["model_path"] == "models/TestModel.zip"
            assert metadata["environment_name"] == "PySimEnv"
            assert metadata["config"]["simulation_speed"] == 20.0
        finally:
            exporter.close()

    def test_save_is_idempotent(self, export_dir: Path) -> None:
        """Repeated save() calls should not fail or duplicate rows."""
        exporter = TelemetryExporter(session_type="inference", export_dir=export_dir)
        try:
            exporter.log_step(step=1, sim_time=0.1)
            exporter.save()
            first = exporter.csv_path.read_text(encoding="utf-8")
            exporter.save()
            second = exporter.csv_path.read_text(encoding="utf-8")
            assert first == second
        finally:
            exporter.close()

    def test_close_is_idempotent(self, export_dir: Path) -> None:
        """Repeated close() calls should not fail."""
        exporter = TelemetryExporter(session_type="inference", export_dir=export_dir)
        exporter.close()
        exporter.close()

    def test_csv_filename_matches_session_type(self, export_dir: Path) -> None:
        """Default CSV filename includes the normalized session type."""
        exporter = TelemetryExporter(session_type="Visualise", export_dir=export_dir)
        try:
            assert exporter.csv_path.name == "airsim_telemetry_visualise.csv"
        finally:
            exporter.close()
