"""
Central logging setup for the framework.

Every entry point (fw.main, fw.main_pipeline, fw.main_with_airsim,
airsim.train.train_airsim, ...) should call `setup_logging()` exactly once,
before importing anything that logs. This replaces the ad-hoc
`logging.basicConfig(...)` calls that used to be duplicated in each entry
point.

Log level comes from the framework config (`CONFIG_PARAMETERS['log_level']`)
or a pipeline spec (`PipelineSpec.log_level`) - not from an environment
variable - so it is versioned alongside the rest of the run configuration.

File output uses the same naming convention as `TrainingIterationLogger`'s
CSVs, and - when a `timestamp` is passed in - shares that exact timestamp,
so the .log file and the per-run .csv file for a given run can be matched
up by filename.

Note on multiprocessing (spawn mode): this only configures the *current*
process. Worker processes started via `SubprocVecEnv` (spawn) do not
inherit this configuration and must call `configure_worker_logging()`
themselves (console-only by design - see module docstring section below).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fw.common.tools.path_utils import resolve_path_relative_to_src

_CONSOLE_FORMAT = "%(levelname)s [%(name)s] %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_VALID_LEVEL_NAMES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Third-party/verbose loggers that are noisy at INFO and below by default.
_DEFAULT_SUPPRESSED_LOGGERS = ("airsim", "sim", "matplotlib")

# Naming components of the most recent setup_logging(..., to_file=True)
# call in this process, kept so update_log_file_agent_type() can rebuild
# the filename without fragile parsing of the path itself. None if no
# file-backed run is active (to_file=False, or setup_logging not yet
# called).
_active_run_context: "dict | None" = None


def resolve_log_level(level: "int | str", *, where: str = "log_level") -> int:
    """
    Validate and resolve a log level given as either a logging int constant
    or one of the standard level names (case-insensitive).

    Raises:
        ValueError: If `level` is a string that isn't a recognized level
            name, so a typo in config/JSON fails at load time rather than
            silently falling back to WARNING (logging's own default when
            given an unrecognized name).
    """

    if isinstance(level, bool):
        # bool is an int subclass; guard against accidental True/False.
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
        run_type: Short label for what's running, e.g. "train", "visualize",
            "evaluate_reward", "inference", "pipeline". Used in the log
            filename.
        env_name: Environment name, used in the log filename to match the
            existing TrainingIterationLogger convention.
        agent_type: Agent type, used in the log filename.
        src_dir: Project src directory, required if `to_file` is True (same
            role as TrainingIterationLogger's `src_dir`).
        log_dir: Directory (relative to src_dir, or absolute) to write the
            log file under. Defaults to "logs", matching
            TrainingIterationLogger.
        level: Log level, as an int or standard level name string. Applies
            to both handlers.
        timestamp: Timestamp to embed in the filename. Defaults to now
            (UTC). Pass the same timestamp used for a run's
            TrainingIterationLogger/TrainingSession to keep filenames
            correlated.
        to_file: Whether to also write a file handler. Set False for
            short-lived tools that only want console output.

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
    # Avoid duplicate handlers if setup_logging is called more than once
    # in the same process (e.g. in tests, or a REPL).
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
    Rename the currently active per-run log file to embed a now-known
    agent_type, replacing the placeholder used when setup_logging() had to
    be called before the agent type could be resolved (e.g. fw.main, where
    the agent type - PPO/MPPO/MPSA - depends on which environments end up
    configured, and is only known once that resolution has happened).

    No-op (returns None) if setup_logging() wasn't called with
    to_file=True in this process, or if no FileHandler is currently
    attached to the root logger.

    Safe to call more than once (e.g. once per batch in a multi-batch
    run); each call renames from whatever the file's current name is.
    """

    global _active_run_context
    if _active_run_context is None:
        return None

    root_logger = logging.getLogger()
    file_handler = next(
        (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)), None
    )
    if file_handler is None:
        return None

    ctx = _active_run_context
    new_path = ctx["dir"] / f"{ctx['env_name']}_{agent_type}_{ctx['run_type']}_{ctx['file_timestamp']}.log"
    old_path = Path(file_handler.baseFilename)

    if new_path == old_path:
        return old_path

    root_logger.removeHandler(file_handler)
    file_handler.close()

    try:
        old_path.rename(new_path)
    except OSError:
        # Rename failed (e.g. cross-device, permissions) - keep logging to
        # the original file rather than losing the run's log entirely.
        new_path = old_path

    new_handler = logging.FileHandler(new_path, encoding="utf-8")
    new_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    root_logger.addHandler(new_handler)

    return new_path


def configure_worker_logging(level: "int | str" = logging.INFO) -> None:
    """
    Minimal console-only logging setup for SubprocVecEnv worker processes.

    Worker processes (multiprocessing "spawn" mode) do not inherit the main
    process's logging configuration, so each worker calls this once on
    startup. Output goes to the worker's own stdout only; per-worker output
    may interleave across processes and is not centralized into the main
    run's log file. If centralized worker logging is ever needed, this is
    the place to introduce a QueueHandler/QueueListener instead.
    """

    resolved_level = resolve_log_level(level)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured (e.g. re-entrant call) - just adjust the level.
        root_logger.setLevel(resolved_level)
        return

    root_logger.setLevel(resolved_level)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    root_logger.addHandler(console_handler)

    for noisy_logger_name in _DEFAULT_SUPPRESSED_LOGGERS:
        logging.getLogger(noisy_logger_name).setLevel(logging.CRITICAL)
