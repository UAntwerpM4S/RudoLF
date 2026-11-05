"""
Shared fixtures for the surrogate contract tests.

The tests need no FH simulator and no GPU. Run from the repository root
(pyproject.toml puts src on the path):

    python -m pytest tests/surrogate -q

Tiny synthetic datasets are written to a temporary folder and tiny models are
trained on them for one epoch (a few seconds on CPU): the tests check
contracts (column order, signs, file formats, integration), not model quality.
"""

import sys
import pytest

from pathlib import Path

# The shared helpers live next to this file; make them importable whatever
# pytest's import mode or the folder's package layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from surrogate_test_helpers import train_tiny  # noqa: E402


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """
    One-epoch open_water and confined_ukc Myzako models on synthetic data (trained once per session).
    """

    folder = tmp_path_factory.mktemp("tiny_models")
    clean, clean_splits = train_tiny(folder, "open_water")
    confined, confined_splits = train_tiny(folder, "confined_ukc")
    return {"open_water": clean, "confined_ukc": confined,
            "open_water_splits": clean_splits, "confined_ukc_splits": confined_splits}
