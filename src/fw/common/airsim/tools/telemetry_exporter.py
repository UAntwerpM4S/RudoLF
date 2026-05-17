"""
Buffered telemetry CSV writer with sidecar metadata export.

The writer is intentionally lightweight during hot simulation loops:
- rows are queued in memory;
- a background thread writes batches to disk;
- metadata is flushed only on save/close.
"""

from __future__ import annotations

import atexit
import csv
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .util import _timestamp_utc

logger = logging.getLogger(__name__)

_DEFAULT_EXPORT_DIR = Path(__file__).resolve().parents[4] / "exports"
_STOP_SENTINEL = object()
_DEFAULT_FIELDNAMES = [
    "step",
    "sim_time",
    "episode",
    "episode_step",
    "x",
    "y",
    "pos_x",
    "pos_y",
    "heading_deg",
    "speed",
    "reward",
    "steering",
    "throttle",
    "brake",
    "checkpoint_index",
    "cross_track_error",
    "physics_wait_s",
    "step_latency_s",
    "ue5_timestamp",
    "outcome",
    "model_origin",
    "control_mode",
]


def _json_safe(value: Any) -> Any:
    """
    Recursively convert values into JSON-serializable structures.

    Args:
        value: Arbitrary Python value that may contain ``Path`` objects or
            nested containers.

    Returns:
        A JSON-safe equivalent structure.
    """

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


class TelemetryExporter:
    """
    Append-safe CSV telemetry writer with metadata export and crash guard.

    This class makes use of a background writer thread to flush buffered rows.
    Callers must call :meth:`close` explicitly to stop the worker thread cleanly
    (or use :func:`atexit` guard if relying on process shutdown behavior).
    """

    def __init__(
        self,
        session_type: str,
        model_path: str = "",
        environment_name: str = "",
        config_dict: Optional[Dict[str, Any]] = None,
        export_dir: Optional[str | Path] = None,
        flush_interval: int = 25,
        fieldnames: Optional[Iterable[str]] = None,
        csv_filename: Optional[str] = None,
        metadata_filename: Optional[str] = None,
    ) -> None:
        """
        Initialize the exporter and open its output files immediately.

        Args:
            session_type: Logical session category used for filenames and
                metadata.
            model_path: Source model path recorded in metadata.
            environment_name: Environment label recorded in metadata.
            config_dict: Serializable configuration payload recorded in
                metadata.
            export_dir: Directory where the CSV and metadata files are written.
            flush_interval: Number of queued rows to buffer before handing them
                to the background writer.
            fieldnames: Optional ordered CSV fieldnames. When omitted, default
                telemetry columns are used.
            csv_filename: Optional explicit CSV filename.
            metadata_filename: Optional explicit metadata JSON filename.
        """

        self._session_type = (session_type or "session").strip().lower()
        self._model_path = model_path
        self._environment_name = environment_name
        self._config_dict = dict(config_dict or {})
        self._flush_interval = max(1, int(flush_interval))

        self.export_dir = Path(export_dir) if export_dir is not None else _DEFAULT_EXPORT_DIR
        self.export_dir.mkdir(parents=True, exist_ok=True)

        csv_name = csv_filename or f"airsim_telemetry_{self._session_type}.csv"
        metadata_name = metadata_filename or f"metadata_{self._session_type}.json"
        self.csv_path = self.export_dir / csv_name
        self.metadata_path = self.export_dir / metadata_name

        requested_fields = list(fieldnames or _DEFAULT_FIELDNAMES)
        self._fieldnames = list(dict.fromkeys(requested_fields))

        self._queue: queue.Queue[Any] = queue.Queue()
        self._buffer: list[Dict[str, Any]] = []
        self._total_steps = 0
        self._saved = False
        self._closed = False
        self._lock = threading.Lock()
        self._start_wall_time = time.perf_counter()
        self._started_at_utc = _timestamp_utc()
        self._metadata: Dict[str, Any] = {}

        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames, extrasaction="ignore")
        if self.csv_path.stat().st_size == 0:
            self._writer.writeheader()
            self._csv_file.flush()

        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="telemetry-writer",
        )
        self._thread.start()
        atexit.register(self.close)

        logger.info(
            "TelemetryExporter ready -> %s (flush every %d rows)",
            self.csv_path,
            self._flush_interval,
        )

    def update_metadata(self, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Merge additional metadata into the JSON sidecar payload.

        Args:
            metadata: Optional mapping of metadata fields to merge.
            **kwargs: Additional metadata fields merged after ``metadata``.
        """

        payload = {}
        if metadata:
            payload.update(metadata)
        payload.update(kwargs)
        self._metadata.update(_json_safe(payload))

    def log_step(self, row: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Queue one telemetry row for asynchronous CSV writing.

        Args:
            row: Optional row mapping to serialize.
            **kwargs: Additional row fields merged after ``row``.
        """

        payload = {}
        if row:
            payload.update(row)
        payload.update(kwargs)
        serialized = {name: payload.get(name, "") for name in self._fieldnames}

        with self._lock:
            self._buffer.append(serialized)
            self._total_steps += 1
            if len(self._buffer) >= self._flush_interval:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """
        Hand buffered rows to the background writer thread.
        """

        if not self._buffer:
            return
        batch = list(self._buffer)
        self._buffer.clear()
        self._queue.put(batch)

    def _writer_loop(self) -> None:
        """
        Background thread that drains queued CSV batches to disk.
        """

        while True:
            batch = self._queue.get()
            try:
                if batch is _STOP_SENTINEL:
                    return
                self._writer.writerows(batch)
                self._csv_file.flush()
            except Exception as exc:
                logger.warning("Failed to flush telemetry buffer: %s", exc)
            finally:
                self._queue.task_done()

    def _write_metadata(self) -> None:
        """
        Write the metadata JSON sidecar.
        """

        metadata = {
            "session_type": self._session_type,
            "model_path": self._model_path,
            "environment_name": self._environment_name,
            "config": _json_safe(self._config_dict),
            "flush_interval": self._flush_interval,
            "csv_filename": self.csv_path.name,
            "started_at_utc": self._started_at_utc,
            "saved_at_utc": _timestamp_utc(),
            "steps_logged": self._total_steps,
            "duration_s": round(time.perf_counter() - self._start_wall_time, 6),
        }
        metadata.update(self._metadata)
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as fh:
                json.dump(_json_safe(metadata), fh, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to write metadata JSON: %s", exc)

    def save(self) -> None:
        """
        Flush all pending rows and update metadata on disk.
        """

        with self._lock:
            self._flush_buffer()
        self._queue.join()
        self._write_metadata()
        self._saved = True

    def close(self) -> None:
        """
        Flush pending rows, stop the worker thread, and close the CSV file.

        Important:
            This method must be called explicitly once the exporter is no longer
            needed, to ensure the background writer thread is stopped reliably
            and file handles are closed.
        """

        if self._closed:
            return

        self.save()
        try:
            self._queue.put(_STOP_SENTINEL)
            self._queue.join()
            if self._thread.is_alive():
                self._thread.join(timeout=5.0)
        finally:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            finally:
                self._closed = True
                try:
                    atexit.unregister(self.close)
                except Exception:
                    pass
                logger.info(
                    "TelemetryExporter closed - %d rows written to %s",
                    self._total_steps,
                    self.csv_path,
                )
