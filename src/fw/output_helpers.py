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
    """Computes heading error relative to the next waypoint segment.

    This function replicates the heading error computation used in the agent
    manager for a single ship sub-environment.

    Args:
        ship_env: Ship environment containing:
            - checkpoints (list of dicts with "pos")
            - checkpoint_index (int)
            - phys_sim.state.heading (float)

    Returns:
        float: Angular error in radians, wrapped to [-pi, pi].
    """

    checkpoints = ship_env.checkpoints
    idx = ship_env.checkpoint_index
    curr = checkpoints[idx]
    nxt = checkpoints[min(idx + 1, len(checkpoints) - 1)]
    desired = np.arctan2(
        nxt["pos"][1] - curr["pos"][1],
        nxt["pos"][0] - curr["pos"][0],
    )
    return (ship_env.phys_sim.state.heading - desired + np.pi) % (2 * np.pi) - np.pi

def normalized_time_axis(length: int) -> np.ndarray:
    """Creates a normalized time axis for a sequence.

    Maps sequence indices to percentage progress in [0, 100].

    Args:
        length: Number of timesteps.

    Returns:
        np.ndarray: Array of shape (length,) with values in [0, 100].
    """

    if length <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 100.0, length)


def shade_color(base_color, n, lo=0.45, hi=1.0):
    """Generates shaded variations of a base color.

    Creates a gradient of `n` colors by scaling the RGB intensity.

    Args:
        base_color: Matplotlib-compatible color string or RGB tuple.
        n: Number of shaded variants to generate.
        lo: Minimum brightness scaling factor.
        hi: Maximum brightness scaling factor.

    Returns:
        list[tuple]: List of RGB tuples representing shaded colors.
    """

    rgb = np.array(to_rgb(base_color))
    factors = np.linspace(hi, lo, n)
    return [tuple(np.clip(rgb * f, 0, 1)) for f in factors]


def resample_to_common_time(signal: np.ndarray, common_time: np.ndarray) -> np.ndarray:
    """Resamples a signal to a shared normalized time axis.

    Supports both 1D and multi-dimensional signals. Used to align episode
    data of varying lengths for averaging and comparison.

    Args:
        signal: Input array of shape (T,) or (T, ...).
        common_time: Target time axis in normalized coordinates.

    Returns:
        np.ndarray: Resampled signal with shape (len(common_time), ...) .

    Notes:
        - Single-sample signals are repeated across the full time axis.
        - Multi-dimensional signals are flattened internally for interpolation
          and reshaped back after processing.
    """

    signal = np.asarray(signal)
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
