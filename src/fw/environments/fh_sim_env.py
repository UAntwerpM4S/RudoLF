import copy
import importlib
import numpy as np

from typing import Optional
from fw.ships.ship_factory import ShipModel
from fw.environments.py_sim_env import PySimEnv
from fw.simulators.fh_sim_adapter import FhSimAdapter


class FhSimEnv(PySimEnv):
    """
    An environment for ship navigation using the FH Simulator.

    This class extends the `PySimEnv` class to integrate with the FH Simulator,
    providing realistic ship dynamics and control. It allows for simulation of ship
    navigation, including path-following, and environmental effects.

    Attributes:
        _fh_sim: An instance of the FH Simulator used for ship dynamics simulation.
    """

    def __init__(self,
                 render_mode: Optional[str] = None,
                 time_step: float = 1.0,
                 max_steps: int = 15000,
                 verbose: Optional[bool] = None,
                 start_pos: Optional[np.ndarray] = None,
                 target_pos: Optional[np.ndarray] = None,
                 ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
                 scenario: str = "DDShip_scenario1_nowindnocurrent"):
        """
        Initialize the FhSimEnv environment.

        Args:
            render_mode (str, optional): The rendering mode. Defaults to None.
            time_step (float, optional): Simulation time step in seconds. Defaults to 1.0.
            max_steps (int, optional): The maximum number of steps per episode. Defaults to 15000.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to None.
            start_pos (tuple, optional): The initial position (x, y) of the ship. Defaults to None.
            target_pos (tuple, optional): The target position (x, y) for the ship. Defaults to None.
            ship_model (ShipModel, optional): The ship model. Defaults to ShipModel.MYZAKO_DEFAULT.
            scenario (str, optional): The scenario name. Defaults to "DDShip_scenario1_nowindnocurrent".
        """
        try:
            fh_wrapper_module = importlib.import_module('fw.simulators.fh_sim_wrapper')
            fh_wrapper_class = getattr(fh_wrapper_module, "FhSimWrapper")
            self._fh_sim = fh_wrapper_class(ship_model, scenario)
        except ModuleNotFoundError:
            self._fh_sim = None

        super().__init__(render_mode=render_mode, time_step=time_step, max_steps=max_steps, verbose=verbose,
                         start_pos=start_pos, target_pos=target_pos, ship_model=ship_model)

        if self._fh_sim is not None:
            self._fh_sim.initial_vessel_state = self.phys_sims[0].state


    def create_simulator(self):
        """Create and return a configured FH-based ship simulator.

        This factory method instantiates an `FhSimulator` using the current
        ship model, FH simulation backend, environmental conditions, and
        integration time step stored on this object.

        Returns:
            FhSimulator: A fully initialized simulator instance configured
            with the specified initial state and the current model parameters.
        """

        return FhSimAdapter(self.ships[0].specifications, self._fh_sim, self.time_step)


    def __deepcopy__(self, memo):
        """
        Create a deep copy of the FhSimEnv instance, excluding the FH Simulator.

        This method ensures that the FH Simulator instance (`_fh_sim`) is not deep-copied,
        as it may contain non-picklable or shared resources.

        Args:
            memo (dict): A dictionary used to track already copied objects.

        Returns:
            FhSimEnv: A deep copy of the current instance.
        """
        new_instance = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            if key == '_fh_sim':
                setattr(new_instance, key, value)   # Retain the original reference
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))  # Deep copy all other attributes

        return new_instance


    @property
    def fh_sim(self):
        """
        Get the FH Simulator instance.

        Returns:
            object: The FH Simulator instance.
        """
        return self._fh_sim


    def randomize(self, randomization_scale: Optional[float] = None):
        """
        Randomize the environment's initial conditions.

        This method can be overridden to introduce randomization in the ship's initial state,
        environmental conditions, or other parameters.

        Args:
            randomization_scale (float, optional): The scale of randomization. Defaults to None.
        """
        pass


    def env_specific_reset(self):
        """
        Reset the FH Simulator to its initial state.

        This method is called during environment reset to ensure the FH Simulator
        is properly reinitialized.
        """
        self._fh_sim.reset()


    def get_keel_clearance(self) -> float:
        return self._fh_sim.ship_interface.getKeelClearance()


    def _has_enough_keel_clearance(self, depth_threshold=0.5):
        return self._fh_sim.ship_interface.getKeelClearance() >= depth_threshold
