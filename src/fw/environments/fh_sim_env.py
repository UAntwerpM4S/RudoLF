import copy
import importlib
import numpy as np

from typing import Optional
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.environments.py_sim_env import PySimEnv
from fw.simulators.ship_simulator import ShipSimulator
from fw.dynamics.fh_tabular_model import FhTabularModel


class FhSimEnv(PySimEnv):
    """
    An environment for ship navigation using the FH Simulator.

    This class extends PySimEnv to integrate with the FH Simulator, providing
    realistic ship dynamics and control including accurate physics, wind, and
    current effects.

    The FH Simulator is treated as a black box: interaction is limited to
    setting normalized rudder and thrust commands in [-1, 1] via the SimAPI
    Python layer.

    Attributes:
        _fh_sim: FhSimBridge instance managing the FH Simulator math model.
            Is None only if the fh_sim_bridge module cannot be imported, in
            which case instantiation will fail at create_simulator().
    """

    def __init__(self,
                 render_mode: Optional[str] = None,
                 time_step: float = 1.0,
                 max_steps: int = 15000,
                 verbose: Optional[bool] = None,
                 start_pos: Optional[np.ndarray] = None,
                 target_pos: Optional[np.ndarray] = None,
                 dynamics_model: Optional[Model] = Model.FH_TABULAR,
                 ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
                 scenario: str = "DDShip_scenario1_nowindnocurrent"):
        """
        Initialize the FhSimEnv environment.

        Args:
            render_mode (str, optional): The rendering mode. Defaults to None.
            time_step (float, optional): Simulation time step in seconds. Defaults to 1.0.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 15000.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to None.
            start_pos (np.ndarray, optional): Initial ship position (x, y). Defaults to None.
            target_pos (np.ndarray, optional): Target position (x, y). Defaults to None.
            ship_model (ShipModel, optional): Ship model preset. Defaults to MYZAKO_DEFAULT.
            scenario (str, optional): FH Simulator scenario name.

        Raises:
            RuntimeError: If the fh_sim_bridge module cannot be imported. FhSimEnv
                requires the FH Simulator to be installed — use PySimEnv for
                pure-Python simulation without external dependencies.
        """

        try:
            fh_bridge_module = importlib.import_module('fw.simulators.fh_sim_bridge')
            fh_bridge_class = getattr(fh_bridge_module, "FhSimBridge")
            self._fh_sim = fh_bridge_class(ship_model, scenario)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "FH Simulator bridge module 'fw.simulators.fh_sim_bridge' could not be "
                "imported. Ensure the FH Simulator package is installed and the "
                "SIMEXE_REPO_FOLDER environment variable is set correctly. "
                "Use PySimEnv for pure-Python simulation without external dependencies."
            ) from e

        super().__init__(
            render_mode=render_mode,
            time_step=time_step,
            max_steps=max_steps,
            verbose=verbose,
            start_pos=start_pos,
            target_pos=target_pos,
            dynamics_model=dynamics_model,
            ship_model=ship_model,
        )

        # Set the initial vessel state on the bridge so the math model can be
        # initialized with the correct position, heading, and velocities on
        # the first reset() call.
        self._fh_sim.initial_vessel_state = self.phys_sim.state


    def create_simulator(self):
        """
        Create and return a configured FH-based ship simulator.

        Called by PySimEnv.__init__ via the simulator factory pattern.

        Returns:
            ShipSimulator: Initialized wrapper around the FH engine.

        Raises:
            RuntimeError: If _fh_sim is None (should not occur after __init__ fix).
        """

        # Explicit guard so any future code path that bypasses
        # __init__ does not silently pass None to ShipSimulator.
        if self._fh_sim is None:
            raise RuntimeError(
                "FH Simulator engine (_fh_sim) is not available. "
                "FhSimEnv requires a working FH Simulator installation."
            )

        if self.dynamics_model == Model.FH_TABULAR:
            self.dynamics = FhTabularModel(self.ship.specifications, self._fh_sim)
        else:
            raise RuntimeError(f"Unknown dynamics model {self.dynamics_model} specified.")

        return ShipSimulator(self.ship.specifications, self.dynamics, self.time_step, self.wind, self.current)


    def __deepcopy__(self, memo):
        """
        Deep copy this instance, sharing the FH Simulator reference.

        The FH Simulator bridge (_fh_sim) is not deep-copied because it holds
        non-picklable external resources (native library handles, COM objects).
        All other attributes are deep-copied normally.

        Args:
            memo (dict): Memoization dict used by copy.deepcopy.

        Returns:
            FhSimEnv: Deep copy with shared _fh_sim reference (including all
            other references to the same bridge in the object graph).
        """

        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self._fh_sim)] = self._fh_sim

        for key, value in self.__dict__.items():
            if key == '_fh_sim':
                setattr(new_instance, key, value)   # Retain the original reference
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))  # Deep copy all other attributes

        return new_instance


    @property
    def fh_sim(self):
        """
        The FH Simulator bridge instance.

        Returns:
            FhSimBridge: Active bridge to the FH Simulator math model.
        """

        return self._fh_sim


    def randomize(self, randomization_scale: Optional[float] = None):
        """
        Intentionally no-op for FhSimEnv.

        The FH Simulator always rewinds to the fixed initial state baked into
        the exercise file at math model initialization time. The initial position,
        heading, and velocities are set once via FhSimBridge.initial_vessel_state
        and cannot be changed between episodes without reinitializing the math
        model from scratch.

        If per-episode starting position variation is needed, it must be
        implemented by loading a different exercise file or by extending this
        method to call dispose() + reset() with a new initial_vessel_state.

        Args:
            randomization_scale (float, optional): Unused. Retained for API
                compatibility with PySimEnv.randomize().
        """

        pass  # Intentional no-op, documented explicitly


    def env_specific_reset(self):
        """
        Reset the FH Simulator to its initial state.

        Called by PySimEnv.reset() before _initialize_simulation(). Rewinds
        the math model to t=0.25s (the earliest supported rewind point).

        Raises:
            RuntimeError: If _fh_sim is None or the math model fails to reset.
        """

        # Guard against None so the error is clear, not AttributeError
        if self._fh_sim is None:
            raise RuntimeError(
                "FH Simulator is not available - cannot reset. "
                "This should not occur if FhSimEnv was constructed normally."
            )

        self._fh_sim.reset()


    def get_keel_clearance(self) -> float:
        """
        Query the current under-keel clearance from the FH Simulator.

        Returns:
            float: Keel clearance in metres as reported by the FH physics engine.

        Raises:
            RuntimeError: If _fh_sim is None or the ship interface is unavailable.
        """

        # Guard against None
        if self._fh_sim is None:
            raise RuntimeError(
                "FH Simulator is not available - cannot query keel clearance."
            )

        return self._fh_sim.ship_interface.getKeelClearance()


    def _has_enough_keel_clearance(self, depth_threshold: float = 0.5) -> bool:
        """
        Check whether the vessel has sufficient keel clearance.

        Args:
            depth_threshold (float): Minimum acceptable keel clearance in
                metres. Defaults to 0.5 m.

        Returns:
            bool: True if keel clearance >= depth_threshold.

        Raises:
            RuntimeError: If _fh_sim is None or the ship interface is unavailable.
        """

        # Guard against None
        if self._fh_sim is None:
            raise RuntimeError(
                "FH Simulator is not available - cannot query keel clearance."
            )

        return self._fh_sim.ship_interface.getKeelClearance() >= depth_threshold
