from typing import Callable, Dict, Optional


class Reward:
    """
    Configurable reward calculator for PySimEnv.

    Reward profiles select which component functions are combined.
    Component weights can be overridden at runtime.
    """

    # Default per-component weights (negative values act as penalties).
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "forward": -1.25,
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
        """
        Initialize the Reward calculator with a specified profile and optional weight overrides.
        
        Args:
            reward_name: Name of the reward profile to use. Must be a key in PROFILE_COMPONENTS.
            reward_weights: Optional dictionary of component weights to override defaults.
                           Missing keys default to DEFAULT_WEIGHTS values.
        
        Raises:
            ValueError: If reward_name is not in PROFILE_COMPONENTS.
        """

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
        """
        Calculate penalty for deviation from target speed.
        
        Penalizes the normalized difference between actual and target velocity.
        
        Args:
            velocity: Current velocity of the vessel.
            target_velocity: Desired velocity of the vessel.
            **_: Unused keyword arguments (for compatibility with component interface).
        
        Returns:
            float: Normalized speed deviation penalty (non-negative).
        """

        if target_velocity == 0:
            return 0.0

        return abs(target_velocity - velocity) / target_velocity

    @staticmethod
    def r_heading_alignment(heading_error: float, **_: float) -> float:
        """
        Calculate penalty for heading error.
        
        Penalizes the absolute difference between desired and actual heading.
        
        Args:
            heading_error: Angular difference from desired heading.
            **_: Unused keyword arguments (for compatibility with component interface).
        
        Returns:
            float: Absolute heading error penalty.
        """

        return abs(heading_error)

    @staticmethod
    def r_cross_track(cross_track_error: float, **_: float) -> float:
        """
        Calculate penalty for lateral path deviation.
        
        Penalizes deviation from the intended trajectory path.
        
        Args:
            cross_track_error: Lateral distance from the desired path.
            **_: Unused keyword arguments (for compatibility with component interface).
        
        Returns:
            float: Cross-track error penalty.
        """

        return cross_track_error

    def r_rudder_angle_change(self, rudder_change: float, max_rudder_change: float, **_: float) -> float:
        """
        Calculate penalty for sudden rudder angle changes.
        
        Penalizes large changes in rudder angle, with hard penalty for exceeding max change.
        Respects weight setting to allow component disable via zero weight.
        
        Args:
            rudder_change: Magnitude of change in rudder angle.
            max_rudder_change: Maximum allowed rudder change per timestep.
            **_: Unused keyword arguments (for compatibility with component interface).
        
        Returns:
            float: Rudder change penalty, or 0 if component is disabled (weight=0).
        """

        weight = self.reward_weights.get("rudder_change", 1.0)
        if weight == 0.0:
            return 0.0  # No penalty if the component is disabled, even if the function is called.
        max_rudder_change = max(float(max_rudder_change), 1e-6)

        return (rudder_change / max_rudder_change) if rudder_change < max_rudder_change else (-1/weight)
    
    @staticmethod
    def r_rudder_angle(rudder_angle: float, **_: float) -> float:
        """
        Calculate penalty for large rudder angles.
        
        Uses a piecewise function with threshold at 0.2 radians to penalize large angles more steeply.
        
        Args:
            rudder_angle: Current rudder angle in radians.
            **_: Unused keyword arguments (for compatibility with component interface).
        
        Returns:
            float: Rudder angle penalty with quadratic growth below 0.2 and quartic growth above.
        """

        return (abs(rudder_angle) / 0.2) ** (2 if abs(rudder_angle) < 0.2 else 4)

    def r_thrust_change(self, thrust_change: float, max_thrust_change: float, **_: float) -> float:
        """
        Calculate penalty for thrust changes.
        
        Penalizes rapid changes in thrust, with hard penalty for exceeding max change.
        Respects weight setting to allow component disable via zero weight.
        
        Args:
            thrust_change: Magnitude of change in thrust.
            max_thrust_change: Maximum allowed thrust change per timestep.
            **_: Unused keyword arguments (for compatibility with component interface).
        
        Returns:
            float: Thrust change penalty, or 0 if component is disabled (weight=0).
        """

        weight = self.reward_weights.get("thrust_change", 1.0)
        if weight == 0.0:
            return 0.0
        max_thrust_change = max(float(max_thrust_change), 1e-6)

        return (thrust_change / max_thrust_change) if thrust_change < max_thrust_change else (-1/weight)

    def r_thrust_change_quadratic(self, thrust_change: float, max_thrust_change: float, **_: float) -> float:
        """
        Calculate smooth quadratic penalty for thrust changes.
        
        Uses piecewise quadratic function that increases in steepness as thrust change approaches max.
        Provides smoother penalty curve compared to r_thrust_change.
        
        Args:
            thrust_change: Magnitude of change in thrust.
            max_thrust_change: Maximum allowed thrust change per timestep.
            **_: Unused keyword arguments (for compatibility with component interface).
        
        Returns:
            float: Smooth quadratic penalty, or 0 if component is disabled (weight=0).
        """

        weight = self.reward_weights.get("thrust_change", 1.0)
        if weight == 0.0:
            return 0.0

        max_thrust_change = max(float(max_thrust_change), 1e-6)
        normalized_change = abs(thrust_change) / max_thrust_change
        
        # Smooth quadratic penalty that steepens near the limit
        if normalized_change < 0.5:
            return normalized_change ** 2  # Gentle for small changes
        elif normalized_change < 1.0:
            return 0.25 + 2 * (normalized_change - 0.5) ** 2  # Steeper
        else:
            return -1 / weight  # Hard violation penalty

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
        Calculate total reward as weighted sum of component penalties.
        
        Evaluates all components in the active reward profile, applies learned weights,
        and returns aggregate reward value. Components are processed in profile order.
        
        Args:
            velocity: Current vessel velocity.
            target_velocity: Desired vessel velocity.
            heading_error: Angular error from desired heading.
            cross_track_error: Lateral deviation from intended path.
            rudder_change: Change in rudder angle this timestep.
            max_rudder_change: Maximum allowed rudder change.
            rudder_angle: Current rudder angle.
            thrust_change: Change in thrust this timestep.
            max_thrust_change: Maximum allowed thrust change.
        
        Returns:
            float: Weighted sum of all enabled component penalties for this timestep.
        """

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
