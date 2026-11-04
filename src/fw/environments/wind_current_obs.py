"""
Wind and current features of the RL observation (PySimEnv with wind=current=True).

The four features are the "body_components" of the surrogate dataset
(fw.surrogate.schema.env_body_components), from DynamicsBase.observed_wind_current:

    wind_from_ahead, wind_from_stbd    apparent wind / WIND_OBS_SCALE
    current_to_ahead, current_to_stbd  current / CURRENT_OBS_SCALE

each clipped to [-1, 1]. Models without wind and current give zeros.
"""

import numpy as np

WIND_CURRENT_OBS_NAMES = ("wind_from_ahead", "wind_from_stbd", "current_to_ahead", "current_to_stbd")
WIND_CURRENT_OBS_DIM = len(WIND_CURRENT_OBS_NAMES)

# Scales [m/s]: p99.9 of max(|ahead|, |stbd|) from fw.surrogate.compute_wind_current_obs_scales
# on the tripoli_full and myzako_full datasets (19.2 M rows, 0.1 % clipped; maxima
# 36.8 m/s wind, 2.71 m/s current).
WIND_OBS_SCALE = 26.6
CURRENT_OBS_SCALE = 2.12


def normalize_wind_current(components: np.ndarray) -> np.ndarray:
    """
    Scale [wind_from_ahead, wind_from_stbd, current_to_ahead, current_to_stbd]
    [m/s] to the observation range and clip to [-1, 1].

    Raises:
        ValueError: If `components` does not have shape (4,).
    """

    components = np.asarray(components, dtype=np.float64)
    if components.shape != (WIND_CURRENT_OBS_DIM,):
        raise ValueError(f"Expected {WIND_CURRENT_OBS_DIM} wind/current components, got shape {components.shape}.")
    scales = np.array([WIND_OBS_SCALE, WIND_OBS_SCALE, CURRENT_OBS_SCALE, CURRENT_OBS_SCALE])
    return np.clip(components / scales, -1.0, 1.0).astype(np.float32)
