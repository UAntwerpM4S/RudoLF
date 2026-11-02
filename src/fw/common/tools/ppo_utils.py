"""
Generic PPO utilities: policy file paths and seeding.
"""

from __future__ import annotations

import re
import os

from typing import Optional, Union

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


def resolve_base_seed(seed: Optional[int]) -> int:
    """
    Validate the configured base seed, or draw and return a random one.

    A drawn seed is returned (and logged by the caller) so the run can be reproduced.

    Args:
        seed: Non-negative integer from the 'seed' hyperparameter, or None.

    Returns:
        int: The base seed to derive the worker seeds from.

    Raises:
        ValueError: If `seed` is not None and not a non-negative integer
            (bools are refused: `True` would silently become seed 1).
    """

    if seed is None:
        # 32-bit: short enough to copy from the log into the config.
        return int(np.random.default_rng().integers(0, 2**32))
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"'seed' must be a non-negative integer or None, got {seed!r}.")
    return int(seed)


def reseed_env_rng(env, seed: Union[None, int, np.random.SeedSequence] = None) -> None:
    """
    Give a deep-copied environment its own RNG, seeded with `seed`.

    Vectorized workers are deep copies of one template env and would otherwise
    share identical RNG state (identical start poses/route segments). Agents pass
    one SeedSequence child per worker, making workers independent and reproducible
    from the 'seed' hyperparameter. seed=None gives an unseeded generator. Envs
    without `_rng` (also when wrapped) are left untouched.

    Args:
        env: Environment instance (possibly wrapped) to re-seed in place.
        seed: Seed (int or SeedSequence) for the new generator, or None.
    """

    target = env
    while target is not None:
        rng = getattr(target, "_rng", None)
        if isinstance(rng, np.random.Generator):
            target._rng = np.random.default_rng(seed)
            return
        target = getattr(target, "env", None)
