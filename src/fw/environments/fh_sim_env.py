import logging
import importlib
import numpy as np

from typing import Optional
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.environments.py_sim_env import PySimEnv

logger = logging.getLogger(__name__)


class FhSimEnv(PySimEnv):
    """
    Deprecated convenience subclass: a PySimEnv driven by the FH Simulator.

    It only constructs the `FhSimBridge` and hands it to `PySimEnv` via
    `fh_bridge`. Everything FH-specific (FH_TABULAR dynamics, initial vessel
    state on the bridge, rewind on reset, keel clearance, the no-op
    `randomize`, and sharing the bridge on deep copy) is implemented once, in
    `PySimEnv`.

    Previously this class stored the bridge in `self._fh_sim` and then called
    `PySimEnv.__init__` without `fh_bridge`, which reset `_fh_sim` to None and
    raised; its `create_simulator` override also had the wrong signature.

    Attributes:
        _fh_sim: FhSimBridge instance managing the FH Simulator math model.
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
        Initialize the FhSimEnv environment.

        .. deprecated::
            FhSimEnv is deprecated; use PySimEnv(..., fh_bridge=FhSimBridge(...))
            instead (as `FhSimEnvironment` in env_factory does).

        Args:
            render_mode (str, optional): The rendering mode. Defaults to None.
            time_step (float, optional): Simulation time step in seconds. Defaults to 1.0.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 15000.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to None.
            start_pos (np.ndarray, optional): Initial ship position (x, y). Defaults to None.
            target_pos (np.ndarray, optional): Target position (x, y). Defaults to None.
            overall_contour (str, optional): Overall contour file.
            depth_contour (str, optional): Depth contour file.
            trajectory (str, optional): Trajectory file.
            dynamics_model (Model, optional): Must be Model.FH_TABULAR.
            ship_model (ShipModel, optional): Ship model preset. Defaults to MYZAKO_DEFAULT.
            scenario (str, optional): FH Simulator scenario name.

        Raises:
            ValueError: If dynamics_model is not Model.FH_TABULAR.
            RuntimeError: If the fh_sim_bridge module cannot be imported.
        """

        logger.warning(
            "FhSimEnv is deprecated and will be removed in a future version; "
            "use PySimEnv(..., fh_bridge=FhSimBridge(...)) instead."
        )

        if dynamics_model != Model.FH_TABULAR:
            raise ValueError(
                f"FhSimEnv requires dynamics_model=Model.FH_TABULAR, got {dynamics_model}."
            )

        try:
            fh_bridge_module = importlib.import_module('fw.simulators.fh_sim_bridge')
            fh_bridge_class = getattr(fh_bridge_module, "FhSimBridge")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "FH Simulator bridge module 'fw.simulators.fh_sim_bridge' could not be "
                "imported. Ensure the FH Simulator package is installed and the "
                "SIMEXE_REPO_FOLDER environment variable is set correctly. "
                "Use PySimEnv for pure-Python simulation without external dependencies."
            ) from e

        fh_bridge = fh_bridge_class(ship_model, scenario)

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
            scenario=scenario,
            fh_bridge=fh_bridge,
        )


    @property
    def fh_sim(self):
        """
        The FH Simulator bridge instance.

        Returns:
            FhSimBridge: Active bridge to the FH Simulator math model.
        """

        return self._fh_sim
