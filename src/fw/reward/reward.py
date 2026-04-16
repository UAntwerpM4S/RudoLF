from typing import Callable, Dict, Optional


class Reward:
    """Configurable reward calculator for PySimEnv.

    Reward profiles select which component functions are combined.
    Component weights can be overridden at runtime.
    """

    # Default per-component weights (negative values act as penalties).
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "forward": -0.25,
        "heading_alignment": -0.5,
        "cross_track": -0.05,
        "rudder_change": -0.4,
        "rudder_angle": -0.05,
        "thrust_change": 0.0,
    }

    # Mapping from profile name to the ordered list of components to evaluate.
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
        self.reward_name = reward_name
        # Start from defaults and allow caller overrides for quick experimentation.
        self.reward_weights: Dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

        # Fail fast if an unknown profile is requested.
        if self.reward_name not in self.PROFILE_COMPONENTS:
            supported = ", ".join(sorted(self.PROFILE_COMPONENTS.keys()))
            raise ValueError(f"Unknown reward_name '{self.reward_name}'. Supported reward profiles: {supported}")

        # Registry of component name -> computation function.
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
        # Penalize deviation from target speed (normalized by target).
        return abs(target_velocity - velocity) / target_velocity

    @staticmethod
    def r_heading_alignment(heading_error: float, **_: float) -> float:
        # Penalize absolute heading error.
        return abs(heading_error)

    @staticmethod
    def r_cross_track(cross_track_error: float, **_: float) -> float:
        # Penalize lateral deviation from the intended path.
        return cross_track_error

    def r_rudder_angle_change(self, rudder_change: float, max_rudder_change: float, **_: float) -> float:
        # Penalize sudden rudder changes, saturating at -1 when over the max.
        weight = self.reward_weights.get("rudder_change", 1.0)
        if weight == 0.0:
            return 0.0  # No penalty if the component is disabled, even if the function is called.
        return (rudder_change / max_rudder_change) ** 1 if rudder_change < max_rudder_change else (-1/weight)
    
    @staticmethod
    def r_rudder_angle(rudder_angle: float, **_: float) -> float:
        # Penalize large rudder angles with a steeper curve beyond the threshold.
        return (abs(rudder_angle) / 0.2) ** (2 if abs(rudder_angle) < 0.2 else 4)

    def r_thrust_change(self, thrust_change: float, max_thrust_change: float, **_: float) -> float:
        # Optional penalty for thrust changes (currently pass-through).
        weight = self.reward_weights.get("thrust_change", 1.0)
        if weight == 0.0:
            return 0.0  # No penalty if the component is disabled, even if the function is called.
        return (thrust_change / max_thrust_change) ** 1 if thrust_change < max_thrust_change else (-1/weight)

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
        # Aggregate the weighted sum of enabled components.
        reward = 0.0
        # Shared inputs for all component functions.
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

        # Compute each component in order and apply its weight.
        for component_name in self.PROFILE_COMPONENTS[self.reward_name]:
            component_fn = self._component_functions[component_name]
            weight = self.reward_weights.get(component_name, 1.0)
            component_value = component_fn(**component_inputs)
            #print(f"Component '{component_name}': value={component_value:.4f}, weight={weight:.4f}, contribution={weight * component_value:.4f}")
            reward += weight * component_value

        return reward
