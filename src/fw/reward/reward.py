from typing import Callable, Dict, Optional


class Reward:
    """
    Configurable reward calculator for PySimEnv: a profile selects the components,
    the reward is their weighted sum. Weights can be overridden per instance.
    """

    # Default per-component weights (negative values act as penalties).
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "forward": -1.25,
        "heading_alignment": -0.5,
        "cross_track": -0.5,
        "rudder_change": -0.4,
        "rudder_angle": -0.05,
        "thrust_change": 0.0,
    }

    # Profile name to the ordered components it evaluates.
    PROFILE_COMPONENTS: Dict[str, tuple] = {
        "default": (
            "forward",
            "heading_alignment",
            "cross_track",
            "rudder_change",
            "rudder_angle",
            "thrust_change",
        ),
    }

    def __init__(self, reward_name: str = "default", reward_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Args:
            reward_name: Reward profile, a key of PROFILE_COMPONENTS.
            reward_weights: Optional weight overrides; missing keys use DEFAULT_WEIGHTS.

        Raises:
            ValueError: If reward_name is not in PROFILE_COMPONENTS.
        """

        self.reward_name = reward_name
        self.reward_weights: Dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

        if self.reward_name not in self.PROFILE_COMPONENTS:
            supported = ", ".join(sorted(self.PROFILE_COMPONENTS.keys()))
            raise ValueError(f"Unknown reward_name '{self.reward_name}'. Supported reward profiles: {supported}")

        # Component name to its function.
        self._component_functions: Dict[str, Callable[..., float]] = {
            "forward": self.r_forward,
            "heading_alignment": self.r_heading_alignment,
            "cross_track": self.r_cross_track,
            "rudder_change": self.r_rudder_angle_change,
            "rudder_angle": self.r_rudder_angle,
            "thrust_change": self.r_thrust_change,
        }

    @staticmethod
    def r_forward(velocity: float, target_velocity: float, **_: float) -> float:
        """
        Normalized speed deviation |target - velocity| / target (0 if target is 0).

        Extra keyword arguments (**_) are ignored by all components; they share one input dict.
        """

        if target_velocity == 0:
            return 0.0

        return abs(target_velocity - velocity) / target_velocity

    @staticmethod
    def r_heading_alignment(heading_error: float, **_: float) -> float:
        """
        Absolute heading error.
        """

        return abs(heading_error)

    @staticmethod
    def r_cross_track(cross_track_error: float, **_: float) -> float:
        """
        Cross-track error, returned as passed in (not abs'd here).
        """

        return cross_track_error

    def r_rudder_angle_change(self, rudder_change: float, max_rudder_change: float, **_: float) -> float:
        """
        Rudder change relative to max_rudder_change (per step). At or above the
        maximum it returns -1/weight, so the weighted contribution is exactly -1.
        Returns 0 when the component weight is 0.
        """

        weight = self.reward_weights.get("rudder_change", 1.0)
        if weight == 0.0:
            return 0.0
        max_rudder_change = max(float(max_rudder_change), 1e-6)

        return (rudder_change / max_rudder_change) if rudder_change < max_rudder_change else (-1/weight)
    
    @staticmethod
    def r_rudder_angle(rudder_angle: float, **_: float) -> float:
        """
        Rudder angle penalty (|angle| / 0.2 rad) ** p, with p = 2 below 0.2 rad
        and p = 4 above.
        """

        return (abs(rudder_angle) / 0.2) ** (2 if abs(rudder_angle) < 0.2 else 4)

    def r_thrust_change(self, thrust_change: float, max_thrust_change: float, **_: float) -> float:
        """
        Thrust change counterpart of r_rudder_angle_change (same formula and limits).
        """

        weight = self.reward_weights.get("thrust_change", 1.0)
        if weight == 0.0:
            return 0.0
        max_thrust_change = max(float(max_thrust_change), 1e-6)

        return (thrust_change / max_thrust_change) if thrust_change < max_thrust_change else (-1/weight)

    def r_thrust_change_quadratic(self, thrust_change: float, max_thrust_change: float, **_: float) -> float:
        """
        Smoother alternative to r_thrust_change (not used by any profile): piecewise
        quadratic in the normalized change, -1/weight at or above the maximum,
        0 when the weight is 0.
        """

        weight = self.reward_weights.get("thrust_change", 1.0)
        if weight == 0.0:
            return 0.0

        max_thrust_change = max(float(max_thrust_change), 1e-6)
        normalized_change = abs(thrust_change) / max_thrust_change
        
        if normalized_change < 0.5:
            return normalized_change ** 2
        elif normalized_change < 1.0:
            return 0.25 + 2 * (normalized_change - 0.5) ** 2
        else:
            return -1 / weight

    def calculate_reward(
        self,
        velocity: float,
        target_velocity: float,
        heading_error: float,
        cross_track_error: float,
        rudder_change: float,
        max_rudder_change: float,
        rudder_angle: float,
        thrust_change: float,
        max_thrust_change: float
    ) -> float:
        """
        Return the weighted sum of the active profile's components for one step.
        Arguments are passed to every component (see the r_* methods).
        """

        reward = 0.0
        component_inputs = {
            "velocity": velocity,
            "target_velocity": target_velocity,
            "heading_error": heading_error,
            "cross_track_error": cross_track_error,
            "rudder_change": rudder_change,
            "max_rudder_change": max_rudder_change,
            "rudder_angle": rudder_angle,
            "thrust_change": thrust_change,
            "max_thrust_change": max_thrust_change,
        }

        for component_name in self.PROFILE_COMPONENTS[self.reward_name]:
            component_fn = self._component_functions[component_name]
            weight = self.reward_weights.get(component_name, 1.0)
            component_value = component_fn(**component_inputs)
            #print(f"Component '{component_name}': value={component_value:.4f}, weight={weight:.4f}, contribution={weight * component_value:.4f}")
            reward += weight * component_value

        return reward
