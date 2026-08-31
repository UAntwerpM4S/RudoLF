"""
Bundles a run's input (config / pipeline spec, resolved effective config)
and output (newly-written files under model_dir/log_dir) into a single ZIP
archive under a top-level `run_archives` folder.

Used by `fw.main` (one bundle per standard batch / evaluate_reward action)
and `fw.main_pipeline` (one bundle for the whole pipeline run). Not wired
into `fw.main_with_airsim` or `fw.main2` - see their module docstrings /
CONFIG_PARAMETERS comments if that ever changes.

Design notes (see also the discussion that produced this module):
  - Policy checkpoints and plot images use static filenames and get
    overwritten across runs (`{training_name}_{agent_type}_policy.zip`,
    `image.png`, ...), so `model_dir`/`log_dir` can contain stale files
    from earlier runs sitting right next to this run's output. Files are
    therefore selected by mtime >= run_started, not "everything currently
    in the directory".
  - Bundling must never be able to crash an otherwise-successful (or
    already-failed) run - mirrors the existing pattern in
    `logging_config.update_log_file_agent_type()`. Any failure here is
    logged as a warning and the function returns None.
  - The bundle is written to a new sibling directory (`run_archives`, next
    to `models`/`logs`), never inside a directory being scanned, so a
    bundle never ends up containing itself.
  - When more than one model_dir/log_dir is scanned in a single call (e.g.
    a pipeline block overriding model_dir via hyperparameter_overrides),
    each directory's files are namespaced under its own labeled subfolder
    in the archive.
"""

from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from fw.common.tools.path_utils import resolve_path_relative_to_src

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """
    Best-effort JSON encoder fallback for the resolved-config snapshot.

    CONFIG_PARAMETERS / pipeline block dicts can contain framework enums
    (e.g. ShipModel, Model, VisualizationMode) that aren't natively JSON
    serializable. Enums are encoded as their member name; anything else
    unexpected falls back to str() rather than crashing the snapshot -
    this is a debugging aid, not a source of truth, so best-effort loss of
    fidelity here is preferable to losing the whole bundle.
    """

    if isinstance(obj, Enum):
        return obj.name
    return str(obj)


def _unique_label(base: str, used: set) -> str:
    """
    Return a label guaranteed not to already be in `used`, adding it to
    `used` as a side effect. Used to namespace multiple model_dirs/log_dirs
    in the archive so that, e.g., two distinct model_dirs that happen to
    both contain "policy.zip" don't collide under the same archive path
    (zipfile allows duplicate names silently; most extraction tools then
    just overwrite one file with the other).
    """

    label = base or "root"
    candidate = label
    suffix = 2
    while candidate in used:
        candidate = f"{label}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _collect_changed_files(root: Path, since_ts: float) -> list[Path]:
    """
    Recursively collect files under `root` modified at or after `since_ts`
    (a POSIX timestamp, comparable to Path.stat().st_mtime).

    Returns an empty list if `root` doesn't exist - a run that never wrote
    to a given directory (e.g. no log file, or log_dir not yet created)
    isn't an error here.
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
            # File vanished mid-scan (e.g. a temp file cleaned up by
            # another process) - skip it rather than fail the bundle.
            continue
    return changed


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
) -> Optional[Path]:
    """
    Create a ZIP bundle of this run's input and output files.

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
            config.py or a pipeline spec JSON file.
        resolved_config: Optional dict of the fully-resolved effective
            configuration (profile applied) to embed as
            input/resolved_config.json. Enums are encoded by name.
        env_name: Used in the bundle filename (matches log naming).
        agent_type: Used in the bundle filename (matches log naming).
        run_type: Used in the bundle filename (matches log naming), e.g.
            "train", "visualize", "evaluate_reward", "pipeline".
        bundle_dir: Directory (relative to src_dir, or absolute) to write
            the bundle under. Defaults to "run_archives".

    Returns:
        Path to the created ZIP file, or None if bundling failed or
        produced nothing to write. Never raises.
    """

    try:
        resolved_bundle_dir = resolve_path_relative_to_src(bundle_dir, src_dir)
        resolved_bundle_dir.mkdir(parents=True, exist_ok=True)

        def _as_dir_list(value: "str | Path | Iterable[str | Path]") -> list[Path]:
            if isinstance(value, (str, Path)):
                value = [value]
            resolved = [resolve_path_relative_to_src(str(v), src_dir) for v in value]
            # De-duplicate while preserving order (e.g. several pipeline
            # blocks sharing the same default model_dir).
            seen: list[Path] = []
            for path in resolved:
                if path not in seen:
                    seen.append(path)
            return seen

        resolved_model_dirs = _as_dir_list(model_dirs)
        resolved_log_dirs = _as_dir_list(log_dirs)

        since_ts = run_started.timestamp()

        timestamp = run_started.strftime("%Y%m%d_%H%M%S")
        bundle_path = resolved_bundle_dir / f"{env_name}_{agent_type}_{run_type}_{timestamp}.zip"

        wrote_anything = False

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for input_file in input_files:
                input_path = Path(input_file)
                if not input_path.is_file():
                    logger.warning(
                        "bundle_run_output: input file '%s' does not exist, skipping.",
                        input_path,
                    )
                    continue
                zf.write(input_path, arcname=f"input/{input_path.name}")
                wrote_anything = True

            if resolved_config is not None:
                zf.writestr(
                    "input/resolved_config.json",
                    json.dumps(resolved_config, indent=2, default=_json_default, sort_keys=True),
                )
                wrote_anything = True

            # Namespace by directory when there's more than one model_dir
            # (e.g. a pipeline block's hyperparameter_overrides pointing
            # at a different model_dir) so files with the same relative
            # path from different dirs can't collide in the archive. With
            # a single model_dir (the common case) the layout stays flat.
            used_model_labels: set = set()
            for model_dir in resolved_model_dirs:
                if len(resolved_model_dirs) > 1:
                    label = _unique_label(model_dir.name or "models", used_model_labels)
                    prefix = f"output/models/{label}"
                else:
                    prefix = "output/models"
                for file_path in _collect_changed_files(model_dir, since_ts):
                    arcname = f"{prefix}/{file_path.relative_to(model_dir).as_posix()}"
                    zf.write(file_path, arcname=arcname)
                    wrote_anything = True

            used_log_labels: set = set()
            for log_dir in resolved_log_dirs:
                if len(resolved_log_dirs) > 1:
                    label = _unique_label(log_dir.name or "logs", used_log_labels)
                    prefix = f"output/logs/{label}"
                else:
                    prefix = "output/logs"
                for file_path in _collect_changed_files(log_dir, since_ts):
                    arcname = f"{prefix}/{file_path.relative_to(log_dir).as_posix()}"
                    zf.write(file_path, arcname=arcname)
                    wrote_anything = True

        if not wrote_anything:
            # Nothing to bundle (e.g. run failed before writing/loading
            # anything) - remove the empty zip rather than leaving clutter.
            bundle_path.unlink(missing_ok=True)
            logger.info("bundle_run_output: nothing to bundle for this run, skipping.")
            return None

        logger.info("Run bundle written to: %s", bundle_path)
        return bundle_path

    except Exception as exc:
        # Bundling is a convenience/debugging feature - it must never be
        # able to take down an otherwise-successful (or already-failed)
        # training run.
        logger.warning("Could not create run bundle: %s", exc)
        return None
