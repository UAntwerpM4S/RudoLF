"""
Bundle a run's inputs, resolved configuration, and newly created outputs into a ZIP archive.
"""

from __future__ import annotations

import json
import logging
import zipfile

from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from fw.common.tools.path_utils import resolve_path_relative_to_src

logger = logging.getLogger(__name__)


# Include a small margin for filesystems with coarse modification timestamps.
_MTIME_SAFETY_MARGIN_SECONDS = 2.0


def _json_default(obj: Any) -> Any:
    """
    Provide a best-effort fallback for JSON serialization.
    """

    if isinstance(obj, Enum):
        return obj.name
    return str(obj)


def _resolve_dir_list(
    value: "str | Path | Iterable[str | Path]", src_dir: Path
) -> list[Path]:
    """
    Normalize one or more directories into a de-duplicated list.
    """

    if isinstance(value, (str, Path)):
        value = [value]

    resolved: list[Path] = []
    for v in value:
        path = resolve_path_relative_to_src(str(v), src_dir)
        if path not in resolved:
            resolved.append(path)
    return resolved


def _collect_changed_files(root: Path, since_ts: float) -> list[Path]:
    """
    Collect files under root modified since the given timestamp.
    """

    if not root.exists():
        return []

    changed = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime >= since_ts:
                changed.append(path)
        except OSError:
            continue

    return changed


def _unique_labels(dirs: list[Path]) -> dict[Path, str]:
    """
    Create short, unique labels for directories.
    """

    used: set[str] = set()
    labels: dict[Path, str] = {}
    for d in dirs:
        base = d.name or "dir"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        labels[d] = candidate
    return labels


def _archive_dirs(
    dirs: list[Path],
    since_ts: float,
    arc_prefix: str,
    extra_files: Iterable[Path] = (),
) -> list[tuple[Path, str]]:
    """
    Return archive paths for changed files in the given directories.
    """

    labels = _unique_labels(dirs) if len(dirs) > 1 else None
    collected: dict[Path, str] = {}

    for d in dirs:
        prefix = f"{arc_prefix}/{labels[d]}" if labels else arc_prefix
        for f in _collect_changed_files(d, since_ts):
            collected[f.resolve()] = f"{prefix}/{f.relative_to(d).as_posix()}"

    for raw_extra in extra_files:
        extra_path = Path(raw_extra)
        if not extra_path.is_file():
            logger.warning(
                "bundle_run_output: extra file '%s' does not exist, skipping.",
                extra_path,
            )
            continue

        extra_path = extra_path.resolve()
        if extra_path in collected:
            continue

        owning_dir = next((d for d in dirs if extra_path.is_relative_to(d)), None)
        if owning_dir is not None:
            prefix = f"{arc_prefix}/{labels[owning_dir]}" if labels else arc_prefix
            collected[extra_path] = f"{prefix}/{extra_path.relative_to(owning_dir).as_posix()}"
            continue

        arcname = f"{arc_prefix}/{extra_path.name}"
        existing_arcnames = set(collected.values())
        if arcname in existing_arcnames:
            stem, suffix = extra_path.stem, extra_path.suffix
            disambiguated = arcname
            counter = 2
            while disambiguated in existing_arcnames:
                disambiguated = f"{arc_prefix}/{stem}_{counter}{suffix}"
                counter += 1
            logger.warning(
                "bundle_run_output: extra file '%s' collides with another "
                "bundled file named '%s'; writing it as '%s' instead so "
                "both are preserved.",
                extra_path, arcname, disambiguated,
            )
            arcname = disambiguated
        collected[extra_path] = arcname

    return list(collected.items())


def bundle_run_output(
    *,
    run_started: datetime,
    src_dir: Path,
    model_dirs: "str | Path | Iterable[str | Path]",
    log_dirs: "str | Path | Iterable[str | Path]",
    input_files: Iterable[Path],
    resolved_config: Optional[dict] = None,
    env_name: str,
    agent_type: str,
    run_type: str,
    bundle_dir: str = "run_archives",
    additional_model_files: Iterable[Path] = (),
) -> Optional[Path]:
    """
    Create a ZIP bundle containing this run's inputs and outputs.

    Args:
        run_started: Timestamp marking the start of the run/batch being
            bundled. Used both as the mtime cutoff for output files and to
            derive the bundle's filename timestamp (so it lines up with
            the run's log filename, which uses the same convention).
        src_dir: Project src directory (same root used to resolve
            model_dir/log_dir elsewhere in the framework).
        model_dirs: One or more model_dir values (as configured/resolved
            for the agent type(s) involved) to scan for changed output
            files.
        log_dirs: One or more log_dir values to scan for changed output
            files (typically just "logs").
        input_files: Paths to raw input file(s) to embed verbatim, e.g.
            config.py or a pipeline spec JSON file. Missing files are
            skipped with a warning, not an error.
        resolved_config: Optional dict of the fully-resolved effective
            configuration (profile applied) to embed as
            input/resolved_config.json. Enums are encoded by name.
        env_name: Used in the bundle filename (matches log naming).
        agent_type: Used in the bundle filename (matches log naming).
        run_type: Used in the bundle filename (matches log naming), e.g.
            "train", "visualize", "evaluate_reward", "pipeline".
        bundle_dir: Directory (relative to src_dir, or absolute) to write
            the bundle under. Defaults to "run_archives".
        additional_model_files: Specific files to include under
            output/models regardless of mtime, for output that exists
            but wasn't (re)written during this run - e.g. a "visualize"
            run/block only *loads* an existing policy checkpoint via
            `Trainer.visualize()`, so it would never be picked up by the
            mtime scan on its own. Missing files are skipped with a
            warning, not an error.

    Returns:
        Path to the created ZIP file, or None if bundling failed or
        produced nothing to write. Never raises.
    """

    try:
        if run_started.tzinfo is None:
            logger.warning(
                "bundle_run_output: run_started has no timezone info; assuming UTC."
            )
            run_started = run_started.replace(tzinfo=timezone.utc)

        resolved_bundle_dir = resolve_path_relative_to_src(bundle_dir, src_dir)
        resolved_model_dirs = _resolve_dir_list(model_dirs, src_dir)
        resolved_log_dirs = _resolve_dir_list(log_dirs, src_dir)

        since_ts = run_started.timestamp() - _MTIME_SAFETY_MARGIN_SECONDS
        timestamp = run_started.strftime("%Y%m%d_%H%M%S")
        bundle_path = resolved_bundle_dir / f"{env_name}_{agent_type}_{run_type}_{timestamp}.zip"

        entries: list[tuple[Path, str]] = []

        for input_file in input_files:
            input_path = Path(input_file)
            if input_path.is_file():
                entries.append((input_path, f"input/{input_path.name}"))
            else:
                logger.warning(
                    "bundle_run_output: input file '%s' does not exist, skipping.",
                    input_path,
                )

        entries.extend(
            _archive_dirs(
                resolved_model_dirs,
                since_ts,
                "output/models",
                extra_files=additional_model_files,
            )
        )
        entries.extend(_archive_dirs(resolved_log_dirs, since_ts, "output/logs"))

        if not entries and resolved_config is None:
            logger.info("bundle_run_output: nothing to bundle for this run, skipping.")
            return None

        resolved_bundle_dir.mkdir(parents=True, exist_ok=True)
        written_count = 0
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src, arcname in entries:
                try:
                    zf.write(src, arcname=arcname)
                    written_count += 1
                except OSError as exc:
                    logger.warning(
                        "bundle_run_output: could not add '%s' to the bundle, skipping: %s",
                        src, exc,
                    )
            if resolved_config is not None:
                zf.writestr(
                    "input/resolved_config.json",
                    json.dumps(resolved_config, indent=2, default=_json_default, sort_keys=True),
                )
                written_count += 1

        if written_count == 0:
            bundle_path.unlink(missing_ok=True)
            logger.info("bundle_run_output: nothing was actually written, skipping.")
            return None

        logger.info("Run bundle written to: %s", bundle_path)
        return bundle_path

    except Exception as exc:
        logger.warning("Could not create run bundle: %s", exc)
        return None
