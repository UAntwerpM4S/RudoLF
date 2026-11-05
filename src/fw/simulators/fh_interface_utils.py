"""
FH Simulator conventions and helpers shared by the RL bridge (FhSimBridge,
FhTabularModel), the dataset generator and the model-comparison tests.

  * Heading: FH degrees <-> framework radians (0 = East, CCW).
  * Wind and current as the dataset generator stores them: apparent wind
    (FH WindState, relative to the bow, positive from starboard) and the
    earth-fixed current (over ground minus through water, flowing toward).
  * Wind-field randomization: the per-episode wind-field transform
    distribution and the warm-up FH needs before WindState shows it.
  * Exercise names: the scenario base name per wind/current setting.
"""

import numpy as np

from typing import Tuple
from fw.ships.ship_factory import ShipModel
from fw.surrogate.schema import apparent_to_true_wind

# FH heading convention: "math" (0 = East, CCW) or "compass" (0 = North, CW).
FH_HEADING_CONVENTION = "math"

# FH's WindState lags the wind-field transform by about 1 s (it first reports
# the exercise's default wind), so this warm-up is simulated after setting it.
WIND_WARMUP_TIME = 1.0  # s

# Wind-field transform per episode: Rotation uniform in [0, 360) deg, Scale
# |N(0, WIND_SCALE_SIGMA)| clipped to WIND_SCALE_MAX, Offset 0.
WIND_SCALE_SIGMA = 7.8   # about 80% of the wind scales between 0 and 10
WIND_SCALE_MAX = 25.0

# Exercise base names with wind and current (the no-wind bases are the
# configured 'scenario' and, for the Tripoli, TRIPOLI_NO_WIND_CURRENT_EXERCISE).
WIND_CURRENT_SCENARIO = "DDShip_scenario3_windcurrent"
TRIPOLI_WIND_CURRENT_EXERCISE = "DDShip_Tripoli_scenario_windcurrent"
TRIPOLI_NO_WIND_CURRENT_EXERCISE = "DDShip_Tripoli_scenario_nowindnocurrent"


def _wrap_pi(angle: float) -> float:
    """
    Angle [rad] wrapped to [-pi, pi).
    """

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def fh_heading_to_env_rad(fh_heading_deg: float) -> float:
    """
    FH heading (degrees, FH_HEADING_CONVENTION) -> radians, 0 = East, counter-clockwise, [-pi, pi).
    """

    if FH_HEADING_CONVENTION == "math":
        heading_deg = fh_heading_deg
    elif FH_HEADING_CONVENTION == "compass":
        heading_deg = 90.0 - fh_heading_deg
    else:
        raise ValueError(f"Unknown FH_HEADING_CONVENTION {FH_HEADING_CONVENTION!r}")
    return _wrap_pi(np.radians(heading_deg))


def env_rad_to_fh_heading_deg(heading_rad: float) -> float:
    """
    Inverse of fh_heading_to_env_rad: radians (0 = East, CCW) -> FH degrees in [0, 360).
    """

    deg = float(np.degrees(heading_rad))
    if FH_HEADING_CONVENTION == "math":
        fh = deg
    elif FH_HEADING_CONVENTION == "compass":
        fh = 90.0 - deg
    else:
        raise ValueError(f"Unknown FH_HEADING_CONVENTION {FH_HEADING_CONVENTION!r}")
    return fh % 360.0


def get_current_velocity(_ship_interface):
    """
    Current speed (m/s) and direction (deg in [0, 360), 0 = East, CCW, flowing
    toward), from FH's body-frame velocities over ground and through water.
    FH's body frame (x forward, y port) is rotated directly to the world frame.
    """

    psi = fh_heading_to_env_rad(_ship_interface.getShipHeading())

    r = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi),  np.cos(psi)]
    ])

    vog = _ship_interface.getShipVelocityOverGround()
    vtw = _ship_interface.getShipVelocityThroughWater()

    vog_earth = r @ np.array([vog.x, vog.y])
    vtw_earth = r @ np.array([vtw.x, vtw.y])

    current = vog_earth - vtw_earth

    current_speed = np.linalg.norm(current)
    current_direction_deg = np.degrees(np.arctan2(current[1], current[0]))
    if current_direction_deg < 0:
        current_direction_deg += 360

    return current_speed, current_direction_deg


def fh_wind_current_state(ship_interface) -> Tuple[float, float, float, float]:
    """
    Wind and current as the dataset generator stores them.

    Returns:
        (wind speed [m/s], wind direction [rad, FH WindState relative to the
        bow, positive from starboard], current speed [m/s], current direction
        [rad, 0 = East, CCW, flowing toward]); directions in [-pi, pi).
    """

    wind_state = ship_interface.getWindState()
    current_speed, current_direction_deg = get_current_velocity(ship_interface)
    return (float(wind_state.Speed),
            _wrap_pi(np.radians(wind_state.Direction)),
            float(current_speed),
            _wrap_pi(np.radians(current_direction_deg)))


def fh_true_wind_speed(ship_interface) -> float:
    """
    True wind speed [m/s] at FH's ship: its apparent wind minus its own motion.
    """

    wind_state = ship_interface.getWindState()
    vog = ship_interface.getShipVelocityOverGround()
    ahead, stbd = apparent_to_true_wind(float(wind_state.Speed), float(np.radians(wind_state.Direction)),
                                        float(vog.x), float(vog.y))
    return float(np.hypot(ahead, stbd))


def draw_wind_field_transform(rng: np.random.Generator) -> Tuple[float, float]:
    """
    Random wind-field transform (Rotation [deg], Scale); the draw order is fixed
    (Rotation, then Scale), so seeded runs stay reproducible.
    """

    rotation = float(rng.uniform(0.0, 360.0))
    scale = float(np.clip(np.abs(rng.normal(loc=0.0, scale=WIND_SCALE_SIGMA)), 0.0, WIND_SCALE_MAX))
    return rotation, scale


def fh_exercise_base(ship_model: ShipModel, scenario: str, wind_current: bool) -> str:
    """
    Exercise base name of a ship: the wind/current exercise with `wind_current`,
    otherwise `scenario` (the Tripoli has fixed names). FhSimBridge adds the
    ship suffix for the Myzako presets.
    """

    if ship_model == ShipModel.TRIPOLI:
        return TRIPOLI_WIND_CURRENT_EXERCISE if wind_current else TRIPOLI_NO_WIND_CURRENT_EXERCISE
    return WIND_CURRENT_SCENARIO if wind_current else scenario
