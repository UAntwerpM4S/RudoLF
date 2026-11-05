import copy
import logging
import importlib
import numpy as np

from typing import Optional, Dict
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.environments.py_sim_env import PySimEnv
from fw.simulators.ship_simulator import ShipSimulator
from fw.dynamics.fh_tabular_model import FhTabularModel

logger = logging.getLogger(__name__)


class FhSimEnv(PySimEnv):
    """
    Deprecated: PySimEnv variant driven by the FH Simulator.

    The FH Simulator is a black box: only normalized rudder and thrust
    commands in [-1, 1] are set, via the SimAPI Python layer.

    Attributes:
        _fh_sim: FhSimBridge managing the FH Simulator math model.
    """

    def __init__(self,
                 render_mode: Optional[str] = None,
                 time_step: float = 1.0,
                 max_steps: int = 15000,
                 verbose: Optional[bool] = None,
                 start_pos: Optional[np.ndarray] = None,
                 target_pos: Optional[np.ndarray] = None,
                 overall_contour: Optional[str] = None,
                 depth_contour: Optional[str] = None,
                 trajectory: Optional[str] = None,
                 dynamics_model: Optional[Model] = Model.FH_TABULAR,
                 ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
                 scenario: str = "DDShip_scenario1_nowindnocurrent"):
        """
        Deprecated; use PySimEnv instead.

        Args:
            render_mode: Rendering mode.
            time_step: Simulation time step (s).
            max_steps: Maximum steps per episode.
            verbose: Verbose logging.
            start_pos: Initial ship position (x, y).
            target_pos: Target position (x, y).
            overall_contour, depth_contour, trajectory: Map/path CSV files.
            dynamics_model: Dynamics model (only Model.FH_TABULAR is supported).
            ship_model: Ship model preset.
            scenario: FH Simulator scenario name.

        Raises:
            RuntimeError: If fw.simulators.fh_sim_bridge cannot be imported
                (FH Simulator not installed).
        """

        logger.warning(
            "FhSimEnv is deprecated and will be removed in a future version; "
            "use PySimEnv instead."
        )

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
            overall_contour=overall_contour,
            depth_contour=depth_contour,
            trajectory=trajectory,
            dynamics_model=dynamics_model,
            ship_model=ship_model,
        )

        # Initial state used by the bridge when the math model is first reset
        self._fh_sim.initial_vessel_state = self.phys_sim.state


    def create_simulator(self, ua_surrogate_config: Optional[Dict]=None,
                         ugent_surrogate_config: Optional[Dict]=None,
                         ugent_b01_surrogate_config: Optional[Dict]=None):
        """
        Create the ShipSimulator around FhTabularModel (called by PySimEnv.__init__).

        Args:
            ua_surrogate_config: Configuration for the UA surrogate model.
            ugent_surrogate_config: Configuration for the UGent surrogate model.
            ugent_b01_surrogate_config: Configuration for the UGent B01 surrogate model.

        Returns:
            ShipSimulator using this env's ship and time step.

        Raises:
            RuntimeError: If _fh_sim is None or the dynamics model is not
                Model.FH_TABULAR.
        """

        if self._fh_sim is None:
            raise RuntimeError(
                "FH Simulator engine (_fh_sim) is not available. "
                "FhSimEnv requires a working FH Simulator installation."
            )

        if self.dynamics_model == Model.FH_TABULAR:
            self.dynamics = FhTabularModel(self.ship.specifications, self._fh_sim)
        else:
            raise RuntimeError(f"Unknown dynamics model {self.dynamics_model} specified.")

        return ShipSimulator(self.ship.specifications, self.dynamics, self.time_step)


    def __deepcopy__(self, memo):
        """
        Deep copy everything except the FH bridge, which is shared (it holds
        non-picklable native resources), including references to it elsewhere
        in the object graph.
        """

        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self._fh_sim)] = self._fh_sim

        for key, value in self.__dict__.items():
            if key == '_fh_sim':
                setattr(new_instance, key, value)
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))

        return new_instance


    @property
    def fh_sim(self):
        """
        The FhSimBridge instance.
        """

        return self._fh_sim


    def randomize(self, randomization_scale: Optional[float] = None):
        """
        No-op: the FH Simulator always rewinds to the initial state set once at
        math-model initialization; changing it requires reinitializing the model.
        """

        pass


    def env_specific_reset(self):
        """
        Rewind the FH math model to t = 0.25 s, the earliest supported rewind
        point (called by reset() before _initialize_simulation()).

        Raises:
            RuntimeError: If _fh_sim is None or the reset fails.
        """

        if self._fh_sim is None:
            raise RuntimeError(
                "FH Simulator is not available - cannot reset. "
                "This should not occur if FhSimEnv was constructed normally."
            )

        self._fh_sim.reset()


    def get_keel_clearance(self, pos: Optional[np.ndarray] = None) -> float:
        """
        Under-keel clearance (m) reported by the FH Simulator; pos is unused.

        Raises:
            RuntimeError: If _fh_sim is None.
        """

        if self._fh_sim is None:
            raise RuntimeError(
                "FH Simulator is not available - cannot query keel clearance."
            )

        return self._fh_sim.ship_interface.getKeelClearance()


    def _has_enough_keel_clearance(self, depth_threshold: float = 0.5) -> bool:
        """
        True if the FH keel clearance is >= depth_threshold (m).

        Raises:
            RuntimeError: If _fh_sim is None.
        """

        if self._fh_sim is None:
            raise RuntimeError(
                "FH Simulator is not available - cannot query keel clearance."
            )

        return self._fh_sim.ship_interface.getKeelClearance() >= depth_threshold
