import os
import logging
import threading

from typing import Optional
from fw.ships.ship_factory import ShipModel


try:
    _simexe_path = os.environ["SIMEXE_REPO_FOLDER"]
except KeyError:
    _simexe_path = ""

# Resolved from SIMEXE_REPO_FOLDER (as in FhSimBridge). clr.AddReference() is
# deferred to _connect(), so a missing DLL does not break the import.
_MANAGED_SIM_SYNC_DLL = os.path.join(
    _simexe_path,
    "exe/installed/ManagedSimSync.v143.x64.dll",
)

_SyncApp = None  # populated on first _connect() call


def _load_sync_app():
    """
    Import clr and load ManagedSimSync on first use.
    """

    global _SyncApp
    if _SyncApp is not None:
        return
    import clr  # pythonnet: only needed at connect time
    clr.AddReference(_MANAGED_SIM_SYNC_DLL)
    from Managed.SimSync import SyncApp  # noqa: PLC0415
    _SyncApp = SyncApp


class _SyncMathModelProxy:
    """
    Stand-in for FH MathModel in realtime mode.

    simulateSeconds(dt) blocks until dt seconds of simulation time have passed
    (counted from state-update events), with the previous step()'s controls in
    effect. Same semantics as fasttime, at 1x wall-clock (e.g. 40 Hz ticks,
    dt=2.0 s: ~80 events over ~2 s).
    """

    def __init__(self, bridge: "SimSyncBridge", timeout_per_tick: float = 1.0):
        """
        Args:
            bridge: Connected SimSyncBridge.
            timeout_per_tick: Maximum wait [s] for a single state update.
        """

        self._bridge = bridge
        self._timeout = timeout_per_tick

    def simulateSeconds(self, dt: float) -> None:
        """
        Block until dt seconds of simulation time have elapsed (simSync timestamps).

        Raises:
            RuntimeError: If no state update arrives within the per-tick timeout.
        """

        if dt <= 0.0:
            return

        start_t = self._bridge._current_sim_time()
        target_t = start_t + dt

        while self._bridge._current_sim_time() < target_t:
            if not self._bridge._wait_for_next_state(self._timeout):
                elapsed = self._bridge._current_sim_time() - start_t
                raise RuntimeError(
                    f"No state update within {self._timeout}s while waiting "
                    f"for {dt}s of sim time (got {elapsed:.3f}s). "
                    "Sim is paused, disconnected, or below expected tick rate."
                )

    def getSimInterface(self):
        """
        Simulator interface of the latest state.

        Raises:
            RuntimeError: If no state has been received yet.
        """

        return self._bridge._require_state().SimItf

    def SimulateRewindTo(self, time_point: float) -> None:
        """
        Not supported in realtime mode.

        Raises:
            RuntimeError: Always.
        """

        raise RuntimeError("Rewind is not supported in realtime mode.")


class SimSyncBridge:
    """
    Realtime counterpart of FhSimBridge.

    Connects to a running simSync instance over UDP and exposes the same
    interface as FhSimBridge (ship_model, ship_interface, math_model,
    initial_vessel_state, reset, dispose), so FhTabularModel can drive either.
    """

    def __init__(self,
                 ship_model: Optional[ShipModel] = ShipModel.MYZAKO_37,
                 remote_ip: str = "127.0.0.1", remote_port: int = 5000,
                 local_ip: str = "127.0.0.1",  local_port: int = 5001,
                 connect_timeout: float = 10.0,
                 step_timeout_per_tick: float = 1.0):
        """
        Args:
            ship_model: Ship already loaded in the remote simSync instance. Must
                match it, because FhTabularModel.step() branches on it (e.g.
                TRIPOLI uses azipod controls); it does not configure the remote ship.
            remote_ip: IP address of the simSync instance.
            remote_port: UDP port of the simSync instance.
            local_ip: Local IP address for the UDP connection.
            local_port: Local UDP port.
            connect_timeout: Maximum wait [s] for the first state after connecting.
            step_timeout_per_tick: Maximum wait [s] per state update while advancing.
        """

        self._ship_model = ship_model
        self._remote = (remote_ip, remote_port)
        self._local  = (local_ip, local_port)
        self._connect_timeout = connect_timeout

        # Created lazily in _connect(), after the DLL is loaded.
        self._app = None

        # Shared between .NET event thread and agent thread.
        self._latest_state = None
        self._state_lock = threading.Lock()
        self._new_state = threading.Event()
        self._disconnected = threading.Event()

        self._connected = False
        self._math_model = _SyncMathModelProxy(self, timeout_per_tick=step_timeout_per_tick)
        self._logger = logging.getLogger(f"SimSyncBridge.{id(self)}")

        # API parity with FhSimBridge
        self._initial_vessel_state = None
        self.math_model_initialized = False
        self.initialize_math_model = True

    # ---- Public surface mirroring FhSimBridge -------------------------------

    @property
    def ship_model(self):
        """
        ShipModel given at construction (the ship running in simSync).
        """

        return self._ship_model

    @property
    def initial_vessel_state(self):
        """
        Stored for API parity only; ignored in realtime mode.
        """

        return self._initial_vessel_state

    @initial_vessel_state.setter
    def initial_vessel_state(self, value):
        """
        Store the value (ignored: the running simulator owns the vessel state).
        """

        self._initial_vessel_state = value
        if value is not None:
            self._logger.info("initial_vessel_state ignored in realtime mode.")

    @property
    def math_model(self):
        """
        The realtime math model proxy.

        Raises:
            RuntimeError: If the bridge is not connected.
        """

        if not self._connected:
            raise RuntimeError("Not connected. Call reset() first.")
        return self._math_model

    @property
    def ship_interface(self):
        """
        Own-ship interface of the latest state.

        Raises:
            RuntimeError: If no state has been received yet.
        """

        return self._require_state().OwnShipItf

    def reset(self, time_point: float = 0.25) -> None:
        """
        Connect on first call and block until the first state arrives.

        Args:
            time_point: Ignored (rewind is not supported in realtime mode).

        Raises:
            RuntimeError: If connecting fails or no state arrives in time.
        """

        if not self._connected:
            self._connect()
        elif time_point != 0.25:
            self._logger.warning(
                "Rewind not supported in realtime mode; ignoring time_point=%s",
                time_point,
            )

    def dispose(self) -> None:
        """
        Disconnect and unsubscribe event handlers (idempotent).

        The bridge is marked disconnected even if Disconnect() fails.
        """

        if self._connected:
            try:
                self._app.Disconnect()
                self._disconnected.wait(timeout=5.0)
            finally:
                self._connected = False
                self.math_model_initialized = False
        try:
            self._app.onUpdate       -= self._on_update
            self._app.onInitialized  -= self._on_initialized
            self._app.onDisconnected -= self._on_disconnected
        except Exception:
            pass

    def __repr__(self):
        """
        Return "SimSync".
        """

        return "SimSync"

    # ---- Internals used by the math-model proxy -----------------------------

    def _current_sim_time(self) -> float:
        """
        Current simulation time [s]; 0.0 before the first state.
        """

        with self._state_lock:
            if self._latest_state is None:
                return 0.0
            sim_itf = self._latest_state.SimItf
            return float(sim_itf.getSimulationTime()) if sim_itf is not None else 0.0

    def _wait_for_next_state(self, timeout: float) -> bool:
        """
        Wait up to timeout [s] for a new state; False on timeout or disconnect.
        """

        self._new_state.clear()
        if self._disconnected.is_set():
            return False
        return self._new_state.wait(timeout=timeout)

    def _require_state(self):
        """
        Latest simulator state.

        Raises:
            RuntimeError: If no state has been received yet.
        """

        with self._state_lock:
            if self._latest_state is None:
                raise RuntimeError(
                    "No state received yet. Call reset() first. It blocks "
                    "until the first state arrives."
                )
            return self._latest_state

    def _connect(self) -> None:
        """
        Load ManagedSimSync, connect, subscribe handlers and wait for the first state.

        Raises:
            RuntimeError: If connecting fails or no state arrives within connect_timeout.
        """

        _load_sync_app()
        self._app = _SyncApp()
        self._app.onUpdate       += self._on_update
        self._app.onInitialized  += self._on_initialized
        self._app.onDisconnected += self._on_disconnected
        ok = self._app.Connect(self._remote[0], self._remote[1],
                               self._local[0],  self._local[1], True)
        if not ok:
            raise RuntimeError(f"Connect to {self._remote} failed.")
        self._app.Start()

        if not self._wait_for_next_state(self._connect_timeout):
            self._app.Disconnect()
            raise RuntimeError(
                f"Connected but no state received within {self._connect_timeout}s."
            )
        self._connected = True
        self.math_model_initialized = True
        self.initialize_math_model = False
        self._logger.info("Connected to simSync at %s:%d", *self._remote)

    # ---- Event handlers (fire on .NET threads) ------------------------------

    def _on_update(self, sender, state):
        """
        Store the latest state and wake waiters.
        """

        with self._state_lock:
            self._latest_state = state
        self._new_state.set()

    def _on_initialized(self, sender, state):
        """
        Store the initial state and wake waiters.
        """

        with self._state_lock:
            self._latest_state = state
        self._new_state.set()

    def _on_disconnected(self, sender):
        """
        Mark disconnected and wake waiters so they see it.
        """

        self._disconnected.set()
        self._new_state.set()
