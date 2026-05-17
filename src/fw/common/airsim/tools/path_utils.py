"""
Shared path utilities used across AirSim and framework entrypoints.
"""

from __future__ import annotations

from pathlib import Path

from fw.common.airsim.classes.path_resolution import PathResolutionContext


def resolve_model_path(model_path: str, src_dir: Path) -> Path:
    """
    Resolve model paths against common repository roots.

    Args:
        model_path: User-provided model path, absolute or relative.
        src_dir: Absolute path to the repository src directory.

    Returns:
        A resolved path to the model file. If no candidate exists, the
        repository-root candidate is returned unchanged but resolved.
    """

    context = PathResolutionContext.from_src_dir(src_dir)
    path = Path(model_path)

    if path.is_absolute() and path.exists():
        return path.resolve()

    for root in context.default_search_roots():
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()

    return (context.repo_root / path).resolve()
