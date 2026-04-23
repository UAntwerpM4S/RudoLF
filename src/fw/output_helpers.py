from __future__ import annotations

import numpy as np

from matplotlib.colors import to_rgb


# ---------------
# Shared constants
# ---------------

"""
List of base colors used for differentiating ships in visualizations.

Colors are cycled when the number of ships exceeds the available palette.
"""

SHIP_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf",  # cyan
]


# ---------------
# Output helpers
# ---------------

def compute_heading_error(ship_env) -> float:
    """
    Computes heading error relative to the next waypoint segment.

    This function replicates the heading error computation used in the agent
    manager for a single ship sub-environment.

    Args:
        ship_env: Ship environment containing:
            - checkpoints (list of dicts with "pos")
            - checkpoint_index (int)
            - phys_sim.state.heading (float)

    Returns:
        float: Angular error in radians, wrapped to [-pi, pi].

    Raises:
        AttributeError: If ship_env is missing required attributes.
        ValueError: If checkpoints list is empty.
        IndexError: If checkpoint_index is out of range.
    """

    checkpoints = ship_env.checkpoints
    if not checkpoints:
        raise ValueError("ship_env.checkpoints must not be empty.")

    idx = ship_env.checkpoint_index
    if idx < 0 or idx >= len(checkpoints):
        raise IndexError(
            f"checkpoint_index {idx} is out of range for checkpoints of length {len(checkpoints)}."
        )

    curr = checkpoints[idx]
    if idx + 1 < len(checkpoints):
        nxt = checkpoints[idx + 1]
    else:
        # At final checkpoint: reuse the last valid segment direction.
        nxt = checkpoints[idx]
        curr = checkpoints[max(idx - 1, 0)]

    desired = np.arctan2(
        nxt["pos"][1] - curr["pos"][1],
        nxt["pos"][0] - curr["pos"][0],
    )
    return (ship_env.phys_sim.state.heading - desired + np.pi) % (2 * np.pi) - np.pi


def normalized_time_axis(length: int) -> np.ndarray:
    """
    Creates a normalized time axis for a sequence.

    Maps sequence indices to percentage progress in [0, 100].

    Args:
        length: Number of timesteps. Must be a non-negative integer.

    Returns:
        np.ndarray: Array of shape (length,) with values in [0, 100].
            Returns an empty array when length is 0.

    Raises:
        ValueError: If length is negative.
    """

    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}.")
    if length == 0:
        return np.array([], dtype=float)
    if length == 1:
        return np.array([0.0])
    return np.linspace(0.0, 100.0, length)


def shade_color(base_color, n: int, lo: float = 0.45, hi: float = 1.0) -> list[tuple]:
    """
    Generates shaded variations of a base color.

    Creates a gradient of `n` colors by scaling the RGB intensity.

    Args:
        base_color: Matplotlib-compatible color string or RGB tuple.
        n: Number of shaded variants to generate. Must be non-negative.
        lo: Minimum brightness scaling factor. Must be in [0, 1].
        hi: Maximum brightness scaling factor. Must be in [0, 1].

    Returns:
        list[tuple]: List of RGB tuples representing shaded colors.
            Returns an empty list when n <= 0.

    Raises:
        ValueError: If lo or hi are outside [0, 1], or if lo >= hi.
    """

    if n <= 0:
        return []
    if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
        raise ValueError(f"lo and hi must be in [0, 1], got lo={lo}, hi={hi}.")
    if lo >= hi:
        raise ValueError(f"lo ({lo}) must be strictly less than hi ({hi}).")

    rgb = np.array(to_rgb(base_color))
    factors = np.linspace(hi, lo, n)
    return [tuple(np.clip(rgb * f, 0, 1)) for f in factors]


def resample_to_common_time(signal: np.ndarray, common_time: np.ndarray) -> np.ndarray:
    """
    Resamples a signal to a shared normalized time axis.

    Supports both 1D and multi-dimensional signals. Used to align episode
    data of varying lengths for averaging and comparison.

    Args:
        signal: Input array of shape (T,) or (T, ...).
        common_time: Target time axis in normalized coordinates [0, 1].
            Must be a 1-D array with at least one element.

    Returns:
        np.ndarray: Resampled signal with shape (len(common_time), ...).

    Raises:
        ValueError: If signal is empty (shape[0] == 0).
        ValueError: If common_time is empty or not 1-D.

    Notes:
        - Single-sample signals are repeated across the full time axis.
        - Multi-dimensional signals are flattened internally for interpolation
          and reshaped back after processing.
    """

    signal = np.asarray(signal)
    common_time = np.asarray(common_time)

    if common_time.ndim != 1 or common_time.size == 0:
        raise ValueError(
            f"common_time must be a non-empty 1-D array, "
            f"got shape {common_time.shape}."
        )

    if signal.shape[0] == 0:
        raise ValueError("Cannot resample an empty signal (shape[0] == 0).")

    if signal.shape[0] == 1:
        return np.repeat(signal, len(common_time), axis=0)

    original_time = np.linspace(0.0, 1.0, signal.shape[0])

    if signal.ndim == 1:
        return np.interp(common_time, original_time, signal)

    t = signal.shape[0]
    rest_shape = signal.shape[1:]
    flat_signal = signal.reshape(t, -1)
    resampled = np.stack(
        [np.interp(common_time, original_time, flat_signal[:, i])
         for i in range(flat_signal.shape[1])],
        axis=1,
    )
    return resampled.reshape(len(common_time), *rest_shape)
