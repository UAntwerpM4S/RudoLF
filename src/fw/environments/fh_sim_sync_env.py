import copy
import numpy as np

from typing import Optional
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.environments.py_sim_env import PySimEnv
from fw.simulators.fh_sim_wrapper import FhSimWrapper
from fw.simulators.sync_sim_bridge import SimSyncBridge


class FhSimSyncEnv(PySimEnv):
    """
    Realtime sibling of FhSimEnv. Connects to a running simSync instance
    over UDP instead of initializing a local FH math model.

    Same shape as FhSimEnv from the agent's point of view: phys_sim.step(),
    phys_sim.state, env_specific_reset(), get_keel_clearance() all work
    identically. The differences are:
      - No scenario / exercise file: simSync is already running with its own.
      - initial_vessel_state is accepted but ignored — the running sim
        already has one.
      - env_specific_reset() connects on first call; rewind between episodes
        is NOT supported in realtime.
      - Each phys_sim.step() blocks for ~time_step wall-clock seconds while
        sim time advances by time_step (at 40 Hz tick rate, dt=2.0s → 80
        events per step, ~2 s wall-clock).
    """

    def __init__(self,
                 render_mode: Optional[str] = None,
                 time_step: float = 2.0,
                 max_steps: int = 15000,
                 verbose: Optional[bool] = None,
                 start_pos: Optional[np.ndarray] = None,
                 target_pos: Optional[np.ndarray] = None,
                 scenario: Optional[str] = "DDShip_scenario1_nowindnocurrent",
                 dynamics_model: Optional[Model] = Model.FH_TABULAR,
                 overall_contour: Optional[str] = None,
                 depth_contour: Optional[str] = None,
                 trajectory: Optional[str] = None,
                 ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
                 remote_ip: str = "127.0.0.1", remote_port: int = 5000,
                 local_ip: str  = "127.0.0.1", local_port: int  = 5001,
                 connect_timeout: float = 10.0,
                 step_timeout_per_tick: float = 1.0):
        """
        Initialize the SimSyncEnv environment.

        Args:
            render_mode, time_step, max_steps, verbose, start_pos, target_pos,
            ship_model: Same as FhSimEnv. start_pos / target_pos affect the
                reward/observation construction, not the actual sim state
                (which is owned by the remote simSync).
            scenario: Same as FhSimEnv. start_pos / target_pos affect
            dynamics_model: Same as FhSimEnv. start_pos / target_pos affect
            remote_ip, remote_port: simSync host endpoint to connect to.
            local_ip, local_port: local UDP endpoint to bind to.
            connect_timeout: Max seconds to wait for the first state event
                after Connect()+Start() before giving up.
            step_timeout_per_tick: Max seconds to wait for each sim tick
                during simulateSeconds(); raises if exceeded (paused sim,
                disconnect, tick rate dropped).

        Raises:
            RuntimeError: If SimSyncBridge cannot connect to simSync.
        """

        # Construct bridge BEFORE super().__init__() so create_simulator()
        # (called from PySimEnv.__init__) has self._fh_sim available.
        sim_sync_bridge = SimSyncBridge(
            remote_ip=remote_ip, remote_port=remote_port,
            local_ip=local_ip,   local_port=local_port,
            connect_timeout=connect_timeout,
            step_timeout_per_tick=step_timeout_per_tick,
        )

        super().__init__(
            render_mode=render_mode,
            time_step=time_step,
            max_steps=max_steps,
            verbose=verbose,
            start_pos=start_pos,
            target_pos=target_pos,
            scenario=scenario,
            dynamics_model=dynamics_model,
            overall_contour=overall_contour,
            depth_contour=depth_contour,
            trajectory=trajectory,
            ship_model=ship_model,
            sim_sync_bridge=sim_sync_bridge,
        )
