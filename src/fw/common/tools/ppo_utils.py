"""
Shared generic ppo utilities used across the framework.
"""

from __future__ import annotations

import re
import os

import numpy as np

WINDOWS_INVALID_FILENAME_CHARS = '<>:"/\\|?*'

def _sanitize_file_component(name: str) -> str:
    """
    Sanitize a single filename component for cross-platform safety.
    """

    # Remove control characters and trim whitespace/invalid trailing chars.
    cleaned = re.sub(r"[\x00-\x1f]", "", name).strip().rstrip(". ")
    if not cleaned:
        raise ValueError("Policy filename cannot be empty after sanitization.")

    return "".join("_" if ch in WINDOWS_INVALID_FILENAME_CHARS else ch for ch in cleaned)


def normalize_policy_zip_path(filename: str) -> str:
    """
    Normalize policy save/load path and ensure a .zip extension.
    """

    if filename is None:
        raise ValueError("Policy filename cannot be None.")

    raw = str(filename).strip()
    if not raw:
        raise ValueError("Policy filename cannot be empty.")

    raw = os.path.expanduser(raw)
    if raw.lower().endswith(".zip"):
        raw = raw[:-4]

    directory, base_name = os.path.split(raw)
    safe_base_name = _sanitize_file_component(base_name)

    zip_name = f"{safe_base_name}.zip"
    return os.path.normpath(os.path.join(directory, zip_name) if directory else zip_name)


def reseed_env_rng(env) -> None:
    """
    Give a deep-copied environment its own, independently seeded RNG.

    Vectorized workers are deep copies of one template environment, so
    without this every worker carries an identical copy of the template's
    `np.random.Generator` and samples identical start poses / route
    segments. Envs without a `_rng` attribute are left untouched; an
    explicit `env.reset(seed=...)` still re-seeds the generator.

    Args:
        env: Environment instance (possibly wrapped) to re-seed in place.
    """

    target = env
    while target is not None:
        rng = getattr(target, "_rng", None)
        if isinstance(rng, np.random.Generator):
            target._rng = np.random.default_rng()
            return
        target = getattr(target, "env", None)
