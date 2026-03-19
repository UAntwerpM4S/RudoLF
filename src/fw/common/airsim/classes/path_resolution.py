"""Path-resolution helpers shared by AirSim scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PathResolutionContext:
    """Search roots used to resolve model and artifact paths.

    Attributes:
        src_dir: Absolute path to the repository src directory.
        repo_root: Absolute path to the repository root.
    """

    src_dir: Path
    repo_root: Path

    @classmethod
    def from_src_dir(cls, src_dir: Path) -> "PathResolutionContext":
        """Build a context from the src directory."""
        resolved_src = src_dir.resolve()
        return cls(src_dir=resolved_src, repo_root=resolved_src.parent)

    def default_search_roots(self, cwd: Path | None = None) -> Iterable[Path]:
        """Return the default ordered roots used for relative-path lookup."""
        current = (cwd or Path.cwd()).resolve()
        return current, self.repo_root, self.src_dir
