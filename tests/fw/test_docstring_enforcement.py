"""Enforce docstring coverage for source and test modules."""

from __future__ import annotations

import ast
from pathlib import Path


def _is_excluded(path: Path, root: Path) -> bool:
    """Return True when the given file path is out of enforcement scope.

    Args:
        path: Python file path to evaluate.
        root: Repository root path.

    Returns:
        True when the file should be skipped by enforcement.
    """
    rel = path.relative_to(root)
    rel_str = str(rel).replace("\\", "/")
    return (
        "__pycache__" in rel_str
        or rel_str.endswith("/__init__.py")
        or rel_str.startswith("src/airsim/poc/")
        or rel_str.startswith("src/airsim/inference/visualisation_models/")
    )


def _iter_target_files(root: Path) -> list[Path]:
    """Collect all in-scope Python files in src/ and tests/.

    Args:
        root: Repository root path.

    Returns:
        Sorted list of in-scope Python file paths.
    """
    candidates = list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))
    return sorted(path for path in candidates if not _is_excluded(path, root))


def _missing_docstrings(path: Path) -> list[str]:
    """Find classes/functions without docstrings in one file.

    Args:
        path: Python module path.

    Returns:
        List of formatted `name@line` strings for missing docstrings.
    """
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing: list[str] = []

    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node) is None:
                missing.append(f"{node.name}@{node.lineno}")

    return missing


def test_all_functions_and_classes_have_docstrings():
    """Ensure all in-scope classes and callables define docstrings."""
    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []

    for path in _iter_target_files(root):
        missing = _missing_docstrings(path)
        if missing:
            rel = path.relative_to(root).as_posix()
            offenders.append(f"{rel}: {', '.join(missing)}")

    assert not offenders, "Missing docstrings:\n" + "\n".join(offenders)
