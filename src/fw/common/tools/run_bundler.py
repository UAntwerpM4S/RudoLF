"""
run_bundler.py - Bundle a run's input (config/spec, resolved effective
config) and output (newly-written files under model_dir/log_dir) into a
single ZIP archive under a top-level `run_archives` folder.

Used by `fw.main` (one bundle per standard batch / evaluate_reward action -
main.py can execute several of each in one process, e.g. around an AirSim
step, and each is bundled separately) and `fw.main_pipeline` (one bundle
for the whole pipeline run, since main_pipeline.py always executes as a
single agent_type/run regardless of how many blocks it contains). Not
wired into `fw.main_with_airsim` or `fw.main2` - out of scope for now, see
the task that introduced this module.

Design notes:
  - Policy checkpoints and plot images use static filenames and get
    overwritten across runs (`{training_name}_{agent_type}_policy.zip`,
    `image.png`, ...), so `model_dir`/`log_dir` can contain stale files
    from earlier runs sitting right next to this run's output. Files are
    therefore selected by mtime >= run_started, not "everything currently
    in the directory".
  - Filesystems with coarse mtime resolution (some network filesystems,
    or FAT-family ones, truncate to whole seconds) can round a file
    written a few hundred ms after run_started down to a timestamp that
    looks earlier than run_started, causing a genuinely-new file to be
    missed. A small safety margin is subtracted from the cutoff to trade
    a little precision for the same "never silently miss this run's
    output" guarantee applied elsewhere in this module.
  - `run_started` is expected to be timezone-aware (every current caller
    passes `datetime.now(timezone.utc)`). A naive datetime's
    `.timestamp()` silently assumes *local* time, which would shift the
    mtime cutoff by the local UTC offset - treated as a caller mistake
    rather than intentional, so it's normalized to UTC with a warning
    instead of silently mis-filtering files.
  - `model_dir`/`log_dir` values are derived by callers from
    `get_hyperparameters(agent_type, env_name)` (the same source
    `Trainer`/`AgentManager` use to build agents) rather than hardcoded,
    so this module never goes stale relative to the framework's actual
    defaults.
  - `export_evaluation_results()` (see trainer.py) writes its CSV to
    `Path("models")` relative to the current working directory rather
    than through `resolve_path_relative_to_src` like every other output
    path in the framework. Callers bundling any run/block whose action is
    `evaluate_reward` should pass both the resolved model_dir *and*
    `Path.cwd() / "models"` in `model_dirs` to make sure that CSV isn't
    silently missed when the process wasn't launched from the project
    root. This applies both to `fw.main`'s standalone `evaluate_reward`
    action and to any `evaluate_reward`-action block inside a
    `fw.main_pipeline` run.
  - A "visualize" run/block only *loads* an existing policy checkpoint
    (`Trainer.visualize()` never calls `save_policy`), so it never shows
    up in the mtime scan on its own. Rather than reconstruct the exact
    filename `Trainer`/`AgentManager` would load - which depends on
    `visualize_best_trained` and, for pipeline blocks, on
    `PipelineScheduler._resolve_transfer()`'s block-chaining rules, logic
    this module intentionally does not duplicate - callers pass every
    `*_{agent_type}_policy.zip` currently present in the relevant
    model_dir(s) as `additional_model_files`. This can occasionally pull
    in an unrelated policy that happens to share the model_dir (broader
    net than strictly necessary), but never silently misses the one that
    was actually used, which a name-guessing approach could.

    Known residual gap (not fixed here, flagged rather than guessed at):
    MultiPPOAgent and MultiPolicySingleAgent save their own per-sub-agent
    "best" checkpoints under naming schemes that don't match
    `*_{agent_type}_policy.zip` at all (e.g.
    `Best_{training_name}_{i}_PPO_policy.zip` - literal "_PPO_", not
    "_MPPO_" - see those classes' train() methods), so with
    visualize_best_trained=True and a multi-agent type, this glob may not
    find the checkpoint actually in use. This is a pre-existing framework
    naming inconsistency, not something introduced or fixable here.
  - `additional_model_files` entries are matched against already-collected
    files by resolved source path, so an entry that lives inside one of
    the scanned dirs and happens to already qualify by mtime is only
    listed once. An entry that lives *outside* every scanned dir is filed
    directly under `arc_prefix` by basename; if two such external entries
    share a basename (a narrow case none of the current callers hit,
    since every current additional_model_files entry lives inside a
    scanned model_dir), the second is disambiguated with a numeric suffix
    and a warning rather than silently overwriting the first one's
    archive entry.
  - When more than one distinct `model_dir` (or `log_dir`) is scanned -
    e.g. several pipeline blocks with different `hyperparameter_overrides`
    - each directory's files are archived under their own labeled
    subfolder rather than flattened together, so two dirs that happen to
    both produce a same-named file (e.g. two blocks both writing
    `PySimEnv_PPO_policy.zip`) don't silently overwrite one another's
    entry inside the ZIP. With only one directory (the common case), no
    extra label folder is added, keeping the common-case archive layout
    flat and easy to browse.
  - A single file that vanishes or becomes unreadable between being
    scanned and being written (e.g. another process cleaning up a temp
    file concurrently) is skipped with a warning rather than aborting the
    whole bundle - one bad file shouldn't sink an otherwise-good archive.
  - Bundling must never be able to crash an otherwise-successful (or
    already-failed) run - mirrors the existing pattern in
    `logging_config.update_log_file_agent_type()`. Any failure here is
    logged as a warning and the function returns None.
  - The bundle is written to a new sibling directory (`run_archives`, next
    to `models`/`logs`), never inside a directory being scanned, so a
    bundle never ends up containing itself.
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

# See "Design notes" above re: coarse filesystem mtime resolution.
_MTIME_SAFETY_MARGIN_SECONDS = 2.0


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


def _resolve_dir_list(
    value: "str | Path | Iterable[str | Path]", src_dir: Path
) -> list[Path]:
    """
    Normalize a single dir / iterable of dirs into a de-duplicated list of
    resolved absolute paths (order preserved).
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


def _unique_labels(dirs: list[Path]) -> dict[Path, str]:
    """
    Assign each directory a short, collision-free label (its own name,
    disambiguated with a numeric suffix if two directories share a name)
    for namespacing archive entries when more than one directory is
    scanned together.
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
    Build (source_path, arcname) pairs for every changed file under each
    directory in `dirs` (mtime >= since_ts), plus any `extra_files` -
    specific files to include regardless of mtime, e.g. a pre-existing
    policy checkpoint that a "visualize" run only reads and never
    rewrites, so it would never show up in the mtime scan on its own.

    With a single directory, arcnames are flat: `{arc_prefix}/<relpath>`.
    With several, each directory's files are namespaced under their own
    label (see `_unique_labels`) to avoid two directories' same-named
    files silently overwriting each other's entry in the archive.

    An `extra_files` entry that lives under one of `dirs` is filed using
    that same scheme and de-duplicated against the mtime scan (so a file
    that happens to already qualify by mtime isn't listed twice); one
    that doesn't live under any of `dirs` is filed directly under
    `arc_prefix`. Entries that don't exist on disk are skipped with a
    warning rather than failing the whole bundle.
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
            continue  # already captured by the mtime scan above
        owning_dir = next((d for d in dirs if extra_path.is_relative_to(d)), None)
        if owning_dir is not None:
            prefix = f"{arc_prefix}/{labels[owning_dir]}" if labels else arc_prefix
            collected[extra_path] = f"{prefix}/{extra_path.relative_to(owning_dir).as_posix()}"
            continue

        # Doesn't live under any scanned dir - file it directly under
        # arc_prefix by basename. None of the current callers hit this
        # (every additional_model_files entry today is a glob result from
        # inside a scanned model_dir), but two such external entries could
        # in principle share a basename; since `collected` is keyed by
        # source path, that would otherwise produce two dict entries
        # mapping to the *same* arcname, which zipfile allows silently and
        # most extraction tools then resolve by overwriting one with the
        # other. Disambiguate instead so neither is lost.
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
            # .timestamp() on a naive datetime assumes *local* time,
            # silently shifting the mtime cutoff by the local UTC offset.
            # Every current caller passes an aware datetime.now(timezone.utc);
            # a naive one is almost certainly a mistake rather than
            # intentional, so assume UTC and say so instead of silently
            # mis-filtering files.
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
                    # A file can vanish (or become unreadable) between being
                    # scanned above and being written here - e.g. another
                    # process cleaning up a temp file concurrently. One
                    # missing file shouldn't sink an otherwise-good bundle.
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
            # Every entry failed to write (or there were none to begin
            # with beyond an empty resolved_config) - don't leave a
            # zero-content ZIP sitting around.
            bundle_path.unlink(missing_ok=True)
            logger.info("bundle_run_output: nothing was actually written, skipping.")
            return None

        logger.info("Run bundle written to: %s", bundle_path)
        return bundle_path

    except Exception as exc:
        # Bundling is a convenience/debugging feature - it must never be
        # able to take down an otherwise-successful (or already-failed)
        # training run.
        logger.warning("Could not create run bundle: %s", exc)
        return None
