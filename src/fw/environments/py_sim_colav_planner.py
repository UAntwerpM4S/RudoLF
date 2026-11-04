import numpy as np

from typing import Optional, Tuple
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.environments.py_sim_colav_base import PySimColavBaseEnv


class PySimColavPlannerEnv(PySimColavBaseEnv):
    """
    Collision avoidance environment for replanning approaches: PySimColavBaseEnv
    with obstacles regenerated every step by default.
    """

    def __init__(
            self,
            render_mode: Optional[str] = None,
            time_step: float = 1,
            max_steps: int = 1500,
            verbose: Optional[bool] = None,
            ship_pos: Optional[np.ndarray] = None,
            target_pos: Optional[np.ndarray] = None,
            overall_contour: Optional[str] = None,
            depth_contour: Optional[str] = None,
            trajectory: Optional[str] = None,
            dynamics_model: Optional[Model] = Model.FOSSEN,
            ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
            wind: bool = False,
            current: bool = False,
            ugent_surrogate_config: Optional[dict] = None,
            ugent_b01_surrogate_config: Optional[dict] = None,
            enable_obstacles: bool = True,
            encounter_range: float = 500.0,
            evaluate: bool = True,
            # Scenario configuration (passed to base)
            scenario_types: Optional[list] = None,
            scenario_probs: Optional[list] = None,
            obstacle_population: Optional[list] = None,
            obstacle_weights: Optional[list] = None,
            scenario_obstacle_level: Optional[str] = None,
    ) -> None:
        """
        Forwards all arguments to the base env; only the default obstacle level
        differs ("step" instead of "episode").

        Args:
            render_mode: Rendering mode for the environment.
            time_step: Simulation time step in seconds.
            max_steps: Maximum number of simulation steps per episode.
            verbose: Enables verbose logging if True.
            ship_pos: Initial ship position. Randomized if None.
            target_pos: Target position. Randomized if None.
            overall_contour: Overall contour file.
            depth_contour: Depth contour file.
            trajectory: Trajectory file.
            dynamics_model: Vessel dynamics model.
            ship_model: Ship specification model.
            wind: Add 2 zero wind observation slots (no model here has wind).
            current: Add 2 zero current observation slots (as wind).
            ugent_surrogate_config: Configuration for the UGent surrogate
                (Model.UGENT_SURROGATE), e.g. {"ukc_percent": 363}.
            ugent_b01_surrogate_config: Configuration for the UGent B01 model-scale
                surrogate (Model.UGENT_B01_SURROGATE), e.g. {"ukc_percent": 379}.
            enable_obstacles: Enables obstacle generation.
            encounter_range: Maximum obstacle encounter distance in meters.
            evaluate: Enables evaluation mode.
            scenario_types: Available encounter scenario types.
            scenario_probs: Sampling probabilities for scenario types.
            obstacle_population: Available obstacle density levels.
            obstacle_weights: Sampling weights for obstacle densities.
            scenario_obstacle_level: Obstacle generation level. Defaults to
                "step" when not specified.
        """

        scenario_obstacle_level = str(scenario_obstacle_level) if scenario_obstacle_level is not None else "step"

        super().__init__(
            render_mode=render_mode,
            time_step=time_step,
            max_steps=max_steps,
            verbose=verbose,
            ship_pos=ship_pos,
            target_pos=target_pos,
            overall_contour=overall_contour,
            depth_contour=depth_contour,
            trajectory=trajectory,
            dynamics_model=dynamics_model,
            ship_model=ship_model,
            wind=wind,
            current=current,
            ugent_surrogate_config=ugent_surrogate_config,
            ugent_b01_surrogate_config=ugent_b01_surrogate_config,
            enable_obstacles=enable_obstacles,
            encounter_range=encounter_range,
            evaluate=evaluate,
            scenario_types=scenario_types,
            scenario_probs=scenario_probs,
            obstacle_population=obstacle_population,
            obstacle_weights=obstacle_weights,
            scenario_obstacle_level=scenario_obstacle_level
        )

    def _calculate_colav_reward(self) -> Tuple[float, bool]:
        """
        Delegates to the base reward (no planner-specific shaping); collision
        detection and failure penalty are handled by the base step().

        Returns:
            (reward for the current step, whether the reward logic terminates the episode)
        """

        return super()._calculate_colav_reward()
