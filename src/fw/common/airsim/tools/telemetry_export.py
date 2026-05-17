"""
Legacy async CSV exporter used by older telemetry tests and tools.
"""

from __future__ import annotations

import csv
import queue
import signal
import logging
import threading

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


class AsyncTelemetryExporter:
    """
    Exports telemetry to CSV asynchronously using a background thread.
    """

    def __init__(
        self,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        export_dir: Optional[str | Path] = None,
    ) -> None:
        """
        Initialize the async exporter.

        Args:
            filename: Output CSV filename.
            metadata: Optional metadata lines written at the top of the file.
            export_dir: Optional export directory. When omitted, the workspace
                ``exports`` directory is used.
        """

        src_dir = Path(__file__).resolve().parents[1]
        self.export_dir = Path(export_dir) if export_dir is not None else (src_dir.parent / "exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.path = self.export_dir / filename
        self.metadata = dict(metadata or {})
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=2048)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file = None
        self._writer = None
        self._fieldnames: list[str] = []

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._signal_handler)
            except (ValueError, OSError):
                pass

    def start(self, fieldnames: Optional[Iterable[str]] = None) -> None:
        """
        Start the background writer and open the CSV file.

        Args:
            fieldnames: Optional ordered CSV fieldnames written as the header.
        """

        if self._thread is not None:
            return

        self._fieldnames = list(fieldnames or [])
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        for key, value in self.metadata.items():
            self._file.write(f"# {key}: {value}\n")
        if self._fieldnames:
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
            self._writer.writeheader()
            self._file.flush()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("AsyncTelemetryExporter started: %s", self.path)

    def log_row(self, row: Dict[str, Any]) -> None:
        """
        Queue one row for asynchronous export.

        Args:
            row: Flat telemetry payload to append to the CSV.
        """

        if self._thread is None:
            if not self._fieldnames:
                self.start(fieldnames=row.keys())
            else:
                self.start(fieldnames=self._fieldnames)
        try:
            self._queue.put(row, block=False)
        except queue.Full:
            logger.warning("Telemetry queue is full. Dropping row.")

    def _worker(self) -> None:
        """
        Background thread worker pulling from queue and writing to disk.
        """

        while not self._stop.is_set() or not self._queue.empty():
            try:
                row = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._writer is None:
                    self._fieldnames = list(row.keys())
                    self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
                    self._writer.writeheader()
                self._writer.writerow(row)
                self._file.flush()
            except Exception as exc:
                logger.error("Error writing telemetry row: %s", exc)
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        """
        Stop the background thread and close the file.
        """

        if self._thread is None:
            return
        logger.info("Stopping AsyncTelemetryExporter and flushing data...")
        self._stop.set()
        self._queue.join()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except OSError as exc:
                logger.warning("Error closing telemetry file: %s", exc)
        self._thread = None

    def close(self) -> None:
        """
        Alias for stop().
        """

        self.stop()

    def save(self) -> None:
        """
        Alias for stop() for backwards compatibility.
        """

        self.stop()

    def _signal_handler(self, signum, frame) -> None:
        """
        Handle termination signals to cleanly stop the exporter.

        Args:
            signum: Received signal number.
            frame: Current stack frame supplied by ``signal.signal``.
        """

        self.stop()
        raise KeyboardInterrupt(f"Signal {signum} received")
