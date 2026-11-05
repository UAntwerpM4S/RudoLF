"""
Central logging setup for the framework.

Every entry point (fw.main, fw.main_pipeline, fw.main_with_airsim,
airsim.train.train_airsim, ...) calls `setup_logging()` once, before importing
anything that logs. The level comes from the run configuration
(CONFIG_PARAMETERS['log_level'] or PipelineSpec.log_level), not from an
environment variable.

Log files follow the TrainingIterationLogger CSV naming; pass the same
`timestamp` to match a run's .log and .csv files by name.

SubprocVecEnv workers (spawn mode) do not inherit this configuration and call
`configure_worker_logging()` themselves (console only).
"""

from __future__ import annotations

import sys
import logging

from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from fw.common.tools.path_utils import resolve_path_relative_to_src

_CONSOLE_FORMAT = "%(levelname)s [%(name)s] %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_VALID_LEVEL_NAMES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Noisy third-party loggers, silenced up to CRITICAL.
_DEFAULT_SUPPRESSED_LOGGERS = ("airsim", "sim", "matplotlib")

# Filename parts of the active file-backed run, used by
# update_log_file_agent_type(); None if no log file is active.
_active_run_context: "dict | None" = None


def resolve_log_level(level: "int | str", *, where: str = "log_level") -> int:
    """
    Validate and resolve a log level given as either a logging int constant
    or one of the standard level names (case-insensitive).

    Raises:
        ValueError: If `level` is a bool, an unknown level name or another
            type, so config typos fail at load time.
    """

    if isinstance(level, bool):
        # bool is an int subclass.
        raise ValueError(f"{where}: expected a log level, got bool {level!r}.")

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        normalized = level.strip().upper()
        if normalized not in _VALID_LEVEL_NAMES:
            raise ValueError(
                f"{where}: unknown log level '{level}'. "
                f"Expected one of {_VALID_LEVEL_NAMES}."
            )
        return getattr(logging, normalized)

    raise ValueError(f"{where}: expected a log level (str or int), got {type(level)!r}.")


def setup_logging(
    *,
    run_type: str,
    env_name: str = "run",
    agent_type: str = "na",
    src_dir: Optional[Path] = None,
    log_dir: str = "logs",
    level: "int | str" = logging.INFO,
    timestamp: Optional[datetime] = None,
    to_file: bool = True,
) -> Optional[Path]:
    """
    Configure the root logger for this process with a console handler and,
    unless `to_file` is False, a per-run file handler.

    Args:
        run_type: Label for the run, e.g. "train", "visualize", "inference",
            "pipeline". Used in the log filename.
        env_name: Environment name, used in the log filename.
        agent_type: Agent type, used in the log filename.
        src_dir: Project src directory; required if `to_file` is True.
        log_dir: Log directory, relative to src_dir or absolute.
        level: Log level (int or level name); applies to both handlers.
        timestamp: Timestamp for the filename. Defaults to now (UTC).
        to_file: Whether to add the file handler.

    Returns:
        The resolved log file path, or None if `to_file` is False.

    Raises:
        ValueError: If `level` is not a recognized level.
        ValueError: If `to_file` is True but `src_dir` was not given.
    """

    resolved_level = resolve_log_level(level)

    global _active_run_context

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    # Avoid duplicate handlers on repeated calls (tests, REPL).
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    root_logger.addHandler(console_handler)

    for noisy_logger_name in _DEFAULT_SUPPRESSED_LOGGERS:
        logging.getLogger(noisy_logger_name).setLevel(logging.CRITICAL)

    if not to_file:
        _active_run_context = None
        return None

    if src_dir is None:
        raise ValueError("setup_logging: src_dir is required when to_file=True.")

    run_started = timestamp or datetime.now(timezone.utc)
    file_timestamp = run_started.strftime("%Y%m%d_%H%M%S")

    resolved_dir = resolve_path_relative_to_src(log_dir, src_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_dir / f"{env_name}_{agent_type}_{run_type}_{file_timestamp}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    root_logger.addHandler(file_handler)

    _active_run_context = {
        "dir": resolved_dir,
        "env_name": env_name,
        "run_type": run_type,
        "file_timestamp": file_timestamp,
    }

    return log_path


def update_log_file_agent_type(agent_type: str) -> Optional[Path]:
    """
    Rename the active log file to embed the agent type once it is known.

    Used when setup_logging() runs before the agent type (PPO/MPPO/MPSA) is
    resolved, e.g. in fw.main. Safe to call repeatedly; with several batches
    the filename reflects the last agent type. Failures are only logged as a
    warning, since a cosmetic rename must not crash a run.

    Returns:
        The new log file path, or None if no file-backed run is active, the
        name already matches, or the update failed.
    """

    if _active_run_context is None:
        return None

    root_logger = logging.getLogger()
    file_handler = next(
        (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)), None
    )
    if file_handler is None:
        return None

    ctx = _active_run_context
    target_path = ctx["dir"] / f"{ctx['env_name']}_{agent_type}_{ctx['run_type']}_{ctx['file_timestamp']}.log"
    old_path = Path(file_handler.baseFilename)

    if target_path == old_path:
        return None

    try:
        root_logger.removeHandler(file_handler)
        file_handler.close()

        final_path = target_path
        try:
            old_path.rename(target_path)
        except OSError:
            # Keep logging to the original file.
            final_path = old_path

        new_handler = logging.FileHandler(final_path, encoding="utf-8")
        new_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root_logger.addHandler(new_handler)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Could not update log filename for agent_type=%s: %s", agent_type, exc
        )
        return None

    return final_path if final_path != old_path else None


def configure_worker_logging(level: "int | str" = logging.INFO) -> None:
    """
    Console-only logging setup for SubprocVecEnv worker processes.

    Spawned workers do not inherit the main process's configuration, so each
    calls this once on startup. Output goes to the worker's stdout and is not
    written to the run's log file (a QueueHandler would be needed for that).
    """

    resolved_level = resolve_log_level(level)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured: only adjust the level.
        root_logger.setLevel(resolved_level)
        return

    root_logger.setLevel(resolved_level)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    root_logger.addHandler(console_handler)

    for noisy_logger_name in _DEFAULT_SUPPRESSED_LOGGERS:
        logging.getLogger(noisy_logger_name).setLevel(logging.CRITICAL)
