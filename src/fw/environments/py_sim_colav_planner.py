import numpy as np

from typing import Optional, Tuple
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.environments.py_sim_colav_base import PySimColavBaseEnv


class PySimColavPlannerEnv(PySimColavBaseEnv):
    """Collision avoidance environment using a replanner approach.

    Inherits scenario construction, entity advancement, and collision
    detection from PySimColavBaseEnv. Subclass-specific additions:
    """

    def __init__(
            self,
            render_mode: Optional[str] = None,
            time_step: float = 1,
            max_steps: int = 1500,
            verbose: Optional[bool] = None,
            ship_pos: Optional[np.ndarray] = None,
            target_pos: Optional[np.ndarray] = None,
            dynamics_model: Optional[Model] = Model.FOSSEN,
            ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
            wind: bool = False,
            current: bool = False,
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

        # Any of the parameters in the below super().__init(...)
        # can be set in this environment for different scenario configurations
        # For example: use obstacle generation at step level as opposed to the default "episode" level
        scenario_obstacle_level = str(scenario_obstacle_level) if scenario_obstacle_level is not None else "step"

        super().__init__(
            render_mode=render_mode,
            time_step=time_step,
            max_steps=max_steps,
            verbose=verbose,
            ship_pos=ship_pos,
            target_pos=target_pos,
            dynamics_model=dynamics_model,
            ship_model=ship_model,
            wind=wind,
            current=current,
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
        """Replanner-specific reward. Override with domain-specific shaping."""
        # For now, use only the base path-following reward.
        # Collision detection and failure penalty are handled by base step().
        return super()._calculate_colav_reward()
