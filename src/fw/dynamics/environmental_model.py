import numpy as np

from typing import Tuple


class EnvironmentModel:
    """
    Simplified wind and current disturbance model.

    Wind and current are constant-magnitude vectors with fixed global directions
    (current 180 deg, wind 90 deg, both 0.35), rotated into the body frame with the
    vessel heading and returned as surge/sway accelerations. Yaw moments and
    higher-order effects are neglected.
    """

    def __init__(self, wind: bool, current: bool):
        """
        Args:
            wind: Include wind-induced disturbances.
            current: Include current-induced disturbances.
        """

        self.wind = wind
        self.current = current

        self.radians_current = np.radians(180.0)
        self.current_direction = np.array([np.cos(self.radians_current), np.sin(self.radians_current)],
                                          dtype=np.float32)
        self.current_strength = 0.35

        self.radians_wind = np.radians(90.0)
        self.wind_direction = np.array([np.cos(self.radians_wind), np.sin(self.radians_wind)],
                                       dtype=np.float32)
        self.wind_strength = 0.35

    def accelerations(self, psi: float) -> Tuple[float, float]:
        """
        Compute body-fixed surge/sway accelerations from wind and/or current.

        Args:
            psi: Vessel heading (yaw) in radians.

        Returns:
            (ax, ay): surge and sway accelerations (positive surge forward; the sway
            sign follows the adopted body-frame convention). Independent of vessel
            velocity; no yaw moment.
        """

        ax = ay = 0.0

        if self.wind:
            relative_wind_angle = self.radians_wind - psi
            ax += self.wind_strength * np.cos(relative_wind_angle)
            ay += self.wind_strength * np.sin(relative_wind_angle)

        if self.current:
            relative_current_angle = self.radians_current - psi
            ax += self.current_strength * np.cos(relative_current_angle)
            ay += self.current_strength * np.sin(relative_current_angle)

        return ax, ay
