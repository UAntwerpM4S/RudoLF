"""
Shared generic path utilities used across the framework.
"""

from __future__ import annotations

from pathlib import Path


def resolve_path_relative_to_src(raw_path: str, src_dir: Path) -> Path:
    """
    Resolve a path relative to src while supporting legacy src-prefixed input.
    """

    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "src":
        path = Path(*path.parts[1:])
    return (src_dir.resolve() / path).resolve()
