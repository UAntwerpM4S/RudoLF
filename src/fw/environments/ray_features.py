"""
Ray-cast proximity features around a ship -- the single implementation shared
by the dataset generator (sim_fh_dataset_generator.py) and the surrogate
dynamics models (e.g. UAntwerpSurrogateTripoliFull). Training data and
inference MUST compute these features with this one function; never copy it.

Conventions
-----------
* World frame: x = East, y = North (metres).
* Heading: PySimEnv convention -- radians, 0 = East, counter-clockwise
  positive (heading = arctan2(dy, dx) of the bow direction).
* Ray layout: ray_k points k degrees CLOCKWISE from the bow, in the ship's
  own frame, independent of the heading:
      ray_0   = straight ahead
      ray_90  = starboard beam
      ray_180 = astern
      ray_270 = port beam
  (relative bearing, as used on board. Note: sway in the framework and the
  datasets is positive to PORT, see fw.ships.vessel_state.)
* Feature value: (RAY_MAX_DISTANCE - d) / RAY_MAX_DISTANCE, with d the
  distance to the nearest river-polygon boundary along the ray, clipped to
  [0, RAY_MAX_DISTANCE]: 1.0 = boundary at the ship center, 0.0 = nothing
  within range.

History: the previous (duplicated) implementation assumed a compass heading
(0 = North, clockwise) while the heading passed in is counter-clockwise from
East, so the rays rotated opposite to the ship and the bow ray changed with
heading (ray_90 at heading 0, ray_0 at 45 deg, ray_270 at 90 deg). Datasets
and surrogate checkpoints made with that version are NOT compatible with
this one and must be regenerated / retrained.
"""

from typing import Sequence

import numpy as np

# Version tag of the ray layout; stored with generated datasets/checkpoints
# so data made with a different layout is never mixed in.
RAY_LAYOUT_VERSION = "bow0_clockwise_v2"

RAY_MAX_DISTANCE = 200.0                       # metres; beyond this the feature is 0
RAY_ANGLES_DEG = np.arange(0, 360, 15)         # 24 rays, relative bearings from the bow
NBR_RAY_FEATURES = len(RAY_ANGLES_DEG)         # 24
_RAY_ANGLES_RAD = np.radians(RAY_ANGLES_DEG)   # shape (24,)


def ray_world_angles(heading_rad: float) -> np.ndarray:
    """
    World-frame direction of every ray (radians, 0 = East, counter-clockwise).

    Clockwise relative bearing in a counter-clockwise world frame means
    subtracting: world angle = heading - relative bearing.
    """

    return float(heading_rad) - _RAY_ANGLES_RAD


def ray_end_points(ship_pos: Sequence[float], heading_rad: float,
                   max_distance: float = RAY_MAX_DISTANCE) -> np.ndarray:
    """End point of every ray at `max_distance` from `ship_pos`; shape (24, 2)."""

    ang = ray_world_angles(heading_rad)
    origin = np.asarray(ship_pos, dtype=np.float64)[:2]
    return origin + max_distance * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def compute_ray_features(env, ship_pos: Sequence[float], heading_rad: float) -> np.ndarray:
    """
    Cast NBR_RAY_FEATURES rays around the ship and return normalized
    proximity features (see the module docstring for all conventions).

    Uses `env.ray_cast_distances_batch` when available (one vectorized GEOS
    call), otherwise falls back to sequential `env.ray_cast_distance` calls.
    `env` is any object with those methods (normally a PySimEnv); it is not
    imported here to avoid circular imports.

    Args:
        env:         Object providing the ray-cast method(s) against the river polygon.
        ship_pos:    [x, y] world-frame ship position (m).
        heading_rad: Ship heading in radians, PySimEnv convention
                     (0 = East, counter-clockwise).

    Returns:
        float32 array of shape (NBR_RAY_FEATURES,), values in [0, 1];
        index k corresponds to relative bearing RAY_ANGLES_DEG[k].
    """

    end_points = ray_end_points(ship_pos, heading_rad, RAY_MAX_DISTANCE)
    origin = (float(ship_pos[0]), float(ship_pos[1]))

    if hasattr(env, "ray_cast_distances_batch"):
        distances = np.asarray(
            env.ray_cast_distances_batch(origin, end_points, RAY_MAX_DISTANCE),
            dtype=np.float32,
        )
    else:
        distances = np.array(
            [env.ray_cast_distance(origin, (float(e[0]), float(e[1])), max_distance=RAY_MAX_DISTANCE)
             for e in end_points],
            dtype=np.float32,
        )
    np.clip(distances, 0.0, RAY_MAX_DISTANCE, out=distances)

    return ((RAY_MAX_DISTANCE - distances) / RAY_MAX_DISTANCE).astype(np.float32)
