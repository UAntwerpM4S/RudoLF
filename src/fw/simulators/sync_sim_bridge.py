import os
import logging
import threading

from typing import Optional
from fw.ships.ship_factory import ShipModel


try:
    _simexe_path = os.environ["SIMEXE_REPO_FOLDER"]
except KeyError:
    _simexe_path = ""

# DLL path mirrors the pattern used in FhSimBridge for library_path.
# Resolved once at import time from SIMEXE_REPO_FOLDER so no path is
# hardcoded. The clr.AddReference() call itself is deferred to _connect()
# so a missing DLL does not crash the process on import (e.g. when running
# without a simSync instance, or on a machine without the DLL).
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
    import clr  # pythonnet — only needed at connect time
    clr.AddReference(_MANAGED_SIM_SYNC_DLL)
    from Managed.SimSync import SyncApp  # noqa: PLC0415
    _SyncApp = SyncApp


class _SyncMathModelProxy:
    """
    Stand-in for FH MathModel in realtime mode.

    simulateSeconds(dt) blocks until `dt` seconds of *simulation time* have
    elapsed by accumulating state-update events. The controls written by the
    previous step() remain in effect throughout the interval — same semantics
    as fasttime, just running at 1x wall-clock.

    At 40 Hz sim tick rate with dt=2.0 s, simulateSeconds blocks for ~80
    events over ~2 s wall-clock.
    """

    def __init__(self, bridge: "SimSyncBridge", timeout_per_tick: float = 1.0):
        """
        Initialize the realtime math model proxy.

        Args:
            bridge: Connected realtime simSync bridge instance.
            timeout_per_tick: Maximum number of seconds to wait for a single
                simulator state update.
        """

        self._bridge = bridge
        self._timeout = timeout_per_tick

    def simulateSeconds(self, dt: float) -> None:
        """
        Advance realtime simulation by a given duration.

        This method blocks until the requested amount of simulation time has
        elapsed according to simulator timestamps received from simSync.

        Args:
            dt: Amount of simulation time, in seconds, to advance.

        Raises:
            RuntimeError: If no simulator state update is received within the
                configured timeout while waiting for simulation progression.
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
        Retrieve the simulator interface from the latest simulator state.

        Returns:
            Simulator interface object associated with the latest state.

        Raises:
            RuntimeError: If no simulator state has been received yet.
        """

        return self._bridge._require_state().SimItf

    def SimulateRewindTo(self, time_point: float) -> None:
        """
        Attempt to rewind realtime simulation.

        Rewind operations are unsupported in realtime mode.

        Args:
            time_point: Requested rewind target time in seconds.

        Raises:
            RuntimeError: Always raised because realtime rewind is not
                supported.
        """

        raise RuntimeError("Rewind is not supported in realtime mode.")


class SimSyncBridge:
    """
    Realtime counterpart of FhSimBridge.

    Connects to a running simSync instance over UDP and exposes the same
    interface as FhSimBridge (ship_model, ship_interface, math_model,
    initial_vessel_state, reset, dispose), so FhTabularModel (via
    ShipSimulator) can drive either bridge unchanged.
    """

    def __init__(self,
                 ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
                 remote_ip: str = "127.0.0.1", remote_port: int = 5000,
                 local_ip: str = "127.0.0.1",  local_port: int = 5001,
                 connect_timeout: float = 10.0,
                 step_timeout_per_tick: float = 1.0):
        """
        Initialize the realtime simSync bridge.

        Args:
            ship_model: Ship model of the vessel already running in the
                remote simSync instance. Required for API parity with
                FhSimBridge.ship_model - FhTabularModel.step() branches on
                this (e.g. ShipModel.TRIPOLI uses azipod controls instead
                of rudder/propeller controls) regardless of which bridge
                is in use, so it must be set correctly here to match
                whatever ship the running simSync instance actually has
                loaded. This does not configure or change the remote
                simulator's ship in any way; simSync already owns that.
            remote_ip: IP address of the remote simSync instance.
            remote_port: UDP port of the remote simSync instance.
            local_ip: Local IP address used for the UDP connection.
            local_port: Local UDP port used for the connection.
            connect_timeout: Maximum number of seconds to wait for the first
                simulator state after connecting.
            step_timeout_per_tick: Maximum number of seconds to wait for a
                realtime state update while advancing simulation time.
        """

        self._ship_model = ship_model
        self._remote = (remote_ip, remote_port)
        self._local  = (local_ip, local_port)
        self._connect_timeout = connect_timeout

        # self._app is created lazily in _connect() after the DLL is loaded.
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
        Model type of the ship already running in the remote simSync instance.

        Returns:
            ShipModel: Ship model enum value (e.g. ShipModel.MYZAKO_DEFAULT,
                ShipModel.TRIPOLI), as provided at construction time.
        """

        return self._ship_model

    @property
    def initial_vessel_state(self):
        """
        Placeholder initial vessel state for API compatibility.

        In realtime mode this value is ignored because the running simulator
        already owns the vessel state.

        Returns:
            Optional initial vessel state object.
        """

        return self._initial_vessel_state

    @initial_vessel_state.setter
    def initial_vessel_state(self, value):
        """
        Store the initial vessel state placeholder.

        The value is retained for compatibility with `FhSimBridge` but is
        ignored in realtime mode.

        Args:
            value: Initial vessel state object.
        """

        # Stored for API parity but ignored in realtime — the running sim
        # already has a vessel state.
        self._initial_vessel_state = value
        if value is not None:
            self._logger.info("initial_vessel_state ignored in realtime mode.")

    @property
    def math_model(self):
        """
        Realtime math model proxy.

        Returns:
            _SyncMathModelProxy instance used to advance realtime simulation.

        Raises:
            RuntimeError: If the bridge is not connected.
        """

        if not self._connected:
            raise RuntimeError("Not connected. Call reset() first.")
        return self._math_model

    @property
    def ship_interface(self):
        """
        Ship interface from the latest simulator state.

        Returns:
            Simulator own-ship interface object.

        Raises:
            RuntimeError: If no simulator state has been received yet.
        """

        return self._require_state().OwnShipItf

    def reset(self, time_point: float = 0.25) -> None:
        """
        Connect to the realtime simulator and wait for the first state.

        The first invocation establishes the UDP connection and blocks until
        an initial simulator state is received. Rewind operations are not
        supported in realtime mode.

        Args:
            time_point: Ignored rewind target retained for API compatibility.

        Raises:
            RuntimeError: If the connection fails or no state is received
                within the configured timeout.
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
        Disconnect from the realtime simulator and release resources.

        This method unsubscribes all event handlers and marks the bridge as
        disconnected regardless of whether the underlying disconnect call
        succeeds.
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
        Return the bridge representation string.

        Returns:
            String identifier for the realtime bridge.
        """

        return "SimSync"

    # ---- Internals used by the math-model proxy -----------------------------

    def _current_sim_time(self) -> float:
        """
        Retrieve the current simulator time.

        Returns:
            Current simulation time in seconds. Returns `0.0` if no state
            has been received yet.
        """

        with self._state_lock:
            if self._latest_state is None:
                return 0.0
            sim_itf = self._latest_state.SimItf
            return float(sim_itf.getSimulationTime()) if sim_itf is not None else 0.0

    def _wait_for_next_state(self, timeout: float) -> bool:
        """
        Wait for the next simulator state update.

        Args:
            timeout: Maximum number of seconds to wait.

        Returns:
            `True` if a new state was received before the timeout,
            otherwise `False`.
        """

        self._new_state.clear()
        if self._disconnected.is_set():
            return False
        return self._new_state.wait(timeout=timeout)

    def _require_state(self):
        """
        Retrieve the latest simulator state.

        Returns:
            Latest simulator state object.

        Raises:
            RuntimeError: If no simulator state has been received yet.
        """

        with self._state_lock:
            if self._latest_state is None:
                raise RuntimeError(
                    "No state received yet. Call reset() first — it blocks "
                    "until the first state arrives."
                )
            return self._latest_state

    def _connect(self) -> None:
        """
        Establish the realtime simSync connection.

        Loads the ManagedSimSync assembly, creates the underlying SyncApp,
        subscribes event handlers, and waits for the initial simulator state.

        Raises:
            RuntimeError: If the connection attempt fails.
            RuntimeError: If no simulator state is received within the
                configured timeout.
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
        Handle realtime simulator update events.

        Args:
            sender: Event sender object.
            state: Latest simulator state object.
        """

        with self._state_lock:
            self._latest_state = state
        self._new_state.set()

    def _on_initialized(self, sender, state):
        """
        Handle simulator initialization events.

        Args:
            sender: Event sender object.
            state: Initial simulator state object.
        """

        with self._state_lock:
            self._latest_state = state
        self._new_state.set()

    def _on_disconnected(self, sender):
        """
        Handle simulator disconnect events.

        Args:
            sender: Event sender object.
        """

        self._disconnected.set()
        self._new_state.set()  # unblock any waiter so they observe disconnect
