import numpy as np

from typing import Tuple


class EnvironmentModel:
    """
    Environmental disturbance model for wind and current effects.

    This class provides a simplified representation of environmental
    disturbances acting on a vessel in the horizontal plane. Wind and
    current are modeled as constant-magnitude vectors with fixed global
    directions, projected into the vessel body-fixed frame based on the
    current vessel heading.

    The resulting effects are expressed as equivalent accelerations in
    surge and sway. Yaw moments and higher-order aerodynamic or
    hydrodynamic effects are intentionally neglected.
    """

    def __init__(self, wind: bool, current: bool):
        """
        Initialize the environmental model.

        Args:
            wind: If `True`, include wind-induced disturbances.
            current: If `True`, include current-induced disturbances.
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
        Compute body-fixed environmental accelerations.

        Computes the surge and sway accelerations induced by wind and/or
        current, expressed in the vessel body-fixed frame. The global
        wind and current directions are rotated into the body frame
        using the vessel heading.

        The disturbances are modeled as constant accelerations, scaled
        by predefined strength parameters.

        Args:
            psi: Vessel heading (yaw angle) in radians, measured from the
                inertial frame to the body-fixed frame.

        Returns:
            A tuple `(ax, ay)` where:
                - ax: Surge acceleration due to environmental effects.
                - ay: Sway acceleration due to environmental effects.

        Notes:
            - Positive surge is forward; positive sway is to port/starboard
              depending on the adopted body-frame convention.
            - This model does not include yaw moments, turbulence,
              shielding effects, or dependence on vessel velocity.
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
