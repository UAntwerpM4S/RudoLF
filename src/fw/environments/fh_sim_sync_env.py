import copy
import numpy as np

from typing import Optional
from fw.ships.ship_factory import ShipModel
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
        self._fh_sim = SimSyncBridge(
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
            ship_model=ship_model,
        )

        # Parity with FhSimEnv. SimSyncBridge stores but ignores this.
        self._fh_sim.initial_vessel_state = self.phys_sim.state


    def create_simulator(self):
        """
        Create and return the FhSimWrapper bound to the realtime bridge.

        FhSimWrapper is unchanged — it duck-types on .ship_interface and
        .math_model.simulateSeconds(), both of which SimSyncBridge provides.
        """

        if self._fh_sim is None:
            raise RuntimeError(
                "SimSyncBridge is not available. SimSyncEnv requires a "
                "reachable simSync instance."
            )

        return FhSimWrapper(self.ship.specifications, self._fh_sim, self.time_step)


    def __deepcopy__(self, memo):
        """
        Deep copy this instance, sharing the bridge reference.

        The SimSyncBridge holds a SyncApp with .NET event handlers and an
        open UDP connection — non-picklable, and copying it would give two
        objects fighting over one connection. Share the original reference.
        """

        new_instance = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key == '_fh_sim':
                setattr(new_instance, key, value)
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))
        return new_instance


    @property
    def fh_sim(self):
        """
        The realtime bridge. Named `fh_sim` for API parity with FhSimEnv —
        agent code that calls `env.fh_sim` works against either env.
        """

        return self._fh_sim


    def randomize(self, randomization_scale: Optional[float] = None):
        """
        No-op for SimSyncEnv.

        The running simSync owns the world state — we can't randomize it
        from here. Initial conditions, wind, current are set in simSync's
        own configuration before it starts.
        """

        pass


    def env_specific_reset(self):
        """
        Reset hook called by PySimEnv.reset().

        First call: connects to simSync and waits for the first state event,
        so phys_sim.state is valid immediately after reset() returns.
        Subsequent calls: no-op — realtime has no episode boundary to reset
        to. If the agent's reward logic terminates an episode, the loop ends
        but the sim keeps running.
        """

        if self._fh_sim is None:
            raise RuntimeError(
                "SimSyncBridge is not available — cannot reset."
            )

        self._fh_sim.reset()


    def get_keel_clearance(self, pos: Optional[np.ndarray] = None) -> float:
        """
        Query under-keel clearance from the live ship interface.

        Works identically to FhSimEnv: the IShip interface exposed by
        evSyncState is the same managed type, so getKeelClearance() exists.

        Args:
            pos (np.array, optional): Unused. Retained for API
                compatibility with PySimEnv.get_keel_clearance().
        """

        if self._fh_sim is None:
            raise RuntimeError(
                "SimSyncBridge is not available — cannot query keel clearance."
            )

        return self._fh_sim.ship_interface.getKeelClearance()


    def _has_enough_keel_clearance(self, depth_threshold: float = 0.5) -> bool:
        """
        Check whether the vessel has sufficient keel clearance.

        Args:
            depth_threshold: Minimum acceptable keel clearance in metres.
        """

        if self._fh_sim is None:
            raise RuntimeError(
                "SimSyncBridge is not available — cannot query keel clearance."
            )

        return self._fh_sim.ship_interface.getKeelClearance() >= depth_threshold


    def close(self):
        """
        Disconnect from simSync and release event subscriptions.

        Safe to call multiple times. Idempotent on the bridge side.
        """

        if self._fh_sim is not None:
            self._fh_sim.dispose()
        if hasattr(super(), 'close'):
            super().close()