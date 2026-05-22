import threading
import logging
import clr

# Adjust the path/name to wherever your SyncApp assembly lives
clr.AddReference("C:/dev/simsourcesnew/Sim/msvc143/output/x64/Release/ManagedSimSync.v143.x64.dll")        # or clr.AddReference(r"C:\path\to\SimSync.dll")
from Managed.SimSync import SyncApp


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
        self._bridge = bridge
        self._timeout = timeout_per_tick

    def simulateSeconds(self, dt: float) -> None:
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
        return self._bridge._require_state().SimItf

    def SimulateRewindTo(self, time_point: float) -> None:
        raise RuntimeError("Rewind is not supported in realtime mode.")


class SimSyncBridge:
    """
    Realtime counterpart of FhSimBridge.

    Connects to a running simSync instance over UDP and exposes the same
    surface as FhSimBridge (ship_interface, math_model, reset, dispose),
    so FhSimWrapper can be reused unchanged.
    """

    def __init__(self,
                 remote_ip: str = "127.0.0.1", remote_port: int = 5000,
                 local_ip: str = "127.0.0.1",  local_port: int = 5001,
                 connect_timeout: float = 10.0,
                 step_timeout_per_tick: float = 1.0):
        self._remote = (remote_ip, remote_port)
        self._local  = (local_ip, local_port)
        self._connect_timeout = connect_timeout

        self._app = SyncApp()
        self._app.onUpdate       += self._on_update
        self._app.onInitialized  += self._on_initialized
        self._app.onDisconnected += self._on_disconnected

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
    def initial_vessel_state(self):
        return self._initial_vessel_state

    @initial_vessel_state.setter
    def initial_vessel_state(self, value):
        # Stored for API parity but ignored in realtime — the running sim
        # already has a vessel state.
        self._initial_vessel_state = value
        if value is not None:
            self._logger.info("initial_vessel_state ignored in realtime mode.")

    @property
    def math_model(self):
        if not self._connected:
            raise RuntimeError("Not connected. Call reset() first.")
        return self._math_model

    @property
    def ship_interface(self):
        return self._require_state().OwnShipItf

    def reset(self, time_point: float = 0.25) -> None:
        """First call connects and waits for the first state. No rewind."""
        if not self._connected:
            self._connect()
        elif time_point != 0.25:
            self._logger.warning(
                "Rewind not supported in realtime mode; ignoring time_point=%s",
                time_point,
            )

    def dispose(self) -> None:
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
        return "SimSync"

    # ---- Internals used by the math-model proxy -----------------------------

    def _current_sim_time(self) -> float:
        with self._state_lock:
            if self._latest_state is None:
                return 0.0
            sim_itf = self._latest_state.SimItf
            return float(sim_itf.getSimulationTime()) if sim_itf is not None else 0.0

    def _wait_for_next_state(self, timeout: float) -> bool:
        """Block until the next state event arrives. Returns False on timeout."""
        self._new_state.clear()
        if self._disconnected.is_set():
            return False
        return self._new_state.wait(timeout=timeout)

    def _require_state(self):
        with self._state_lock:
            if self._latest_state is None:
                raise RuntimeError(
                    "No state received yet. Call reset() first — it blocks "
                    "until the first state arrives."
                )
            return self._latest_state

    def _connect(self) -> None:
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
        with self._state_lock:
            self._latest_state = state
        self._new_state.set()

    def _on_initialized(self, sender, state):
        with self._state_lock:
            self._latest_state = state
        self._new_state.set()

    def _on_disconnected(self, sender):
        self._disconnected.set()
        self._new_state.set()  # unblock any waiter so they observe disconnect