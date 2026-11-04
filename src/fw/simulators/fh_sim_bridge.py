import os
import sys
import logging
import numpy as np

from typing import Optional
from fw.ships.ship_factory import ShipModel, require_fh_supported
from fw.simulators.fh_interface_utils import (
    WIND_WARMUP_TIME,
    draw_wind_field_transform,
    fh_exercise_base,
    fh_true_wind_speed,
)

_logger = logging.getLogger(__name__)

# FH installation root and username for the default paths
try:
    simexe_path = os.environ['SIMEXE_REPO_FOLDER']
except KeyError:
    simexe_path = ""

try:
    user_login = os.getlogin()
except OSError:
    user_login = ""


class FhSimBridge:
    """
    Bridge to the FH Simulator math model and ship interface.

    Manages initialization, per-episode rewind and disposal of the black-box
    simulator: initial conditions are set before init, the model is rewound
    between episodes, and the state is read after each step.

    With wind_current, the exercise with wind and current is loaded and every
    reset() draws a new wind-field transform (draw_wind_field_transform) that
    holds for the whole episode; the current and tide stay the exercise's.
    The first rewind verifies once that FH applies a transform set after
    SimulateRewindTo (WIND_PROBE_SCALE); otherwise reset() raises.

    Attributes:
        initialize_math_model (bool): True when the math model needs (re)initialization.
        math_model_initialized (bool): True only after a successful Initialize() call.
        _math_model: MathModel instance. None until successfully initialized.
        _ship_interface: ShipInterface instance. None until successfully initialized.
    """

    SNAPSHOT_WINDOW = 100000  # Maximum rewind window [s] (~27 h)
    # Rewind check: true wind at Scale WIND_PROBE_SCALE must exceed the one at
    # Scale 0 by WIND_PROBE_MIN_DIFF (about 1.1 m/s per unit Scale was measured
    # for the Tripoli exercise, see model_comparison_wind).
    WIND_PROBE_SCALE = 10.0
    WIND_PROBE_MIN_DIFF = 2.0  # m/s

    def __init__(self,
                 ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
                 scenario: str = "DDShip_scenario1_nowindnocurrent",
                 library_path=None,
                 config_path=None,
                 exercise_path=None,
                 output_path=None,
                 rewind_enabled: bool = True,
                 wind_current: bool = False):
        """
        Resolve file paths and set up logging. The math model itself is
        initialized lazily on the first reset().

        Args:
            ship_model (ShipModel): Determines the scenario file suffix.
            scenario (str): Base scenario name without wind and current,
                matched against .tab files (not used for the Tripoli).
            library_path (str, optional): Path to the SimAPI native library.
            config_path (str, optional): Path to simXdrive.config.xml.
            exercise_path (str, optional): Explicit path to the .tab exercise file.
            output_path (str, optional): Directory for simulation output / logs.
            rewind_enabled (bool): Keep FH snapshots so reset() can rewind
                between episodes. If False (as in sim_fh_dataset_generator),
                reset() works only once.
            wind_current (bool): Load the exercise with wind and current
                (fh_exercise_base) and randomize the wind field per episode;
                reset() then needs a wind_rng.

        Initial-condition attributes (set before the first reset(); None keeps
        the exercise's value):
            initial_propeller_rps (float): Initial propeller rps.
            initial_rudder_angle (float): Initial rudder angle [deg].
            initial_wind_field_transform (tuple): (rotation_deg, scale, offset)
                of the exercise's wind field, applied after enableBridge() and
                before the first simulateSeconds() (FH ignores it later).
        """

        require_fh_supported(ship_model, "FhSimBridge")
        self._ship_model = ship_model
        self.exercise_path_exists = True

        base_dir = os.path.join(
            simexe_path,
            "database/areas/ScheldeSaeftinge_23_002/invoer/tra"
        )

        self.wind_current = bool(wind_current)
        base = fh_exercise_base(ship_model, scenario, self.wind_current)
        if base != scenario and ship_model != ShipModel.TRIPOLI:
            _logger.info("[FhSimBridge] wind_current: exercise base '%s' replaces the configured scenario '%s'.",
                         base, scenario)
        scenario = base
        if ship_model == ShipModel.MYZAKO_DEFAULT:
            scenario_suffix = f"{scenario}_default"
        elif ship_model == ShipModel.TRIPOLI:
            scenario_suffix = scenario
        else:
            scenario_suffix = f"{scenario}_{ship_model.name.lower()}"

        self._initial_vessel_state = None
        self.rewind_enabled = bool(rewind_enabled)
        # Initial-condition hooks (see __init__ docstring); None = exercise default.
        self.initial_propeller_rps: Optional[float] = None
        self.initial_rudder_angle: Optional[float] = None          # degrees
        self.initial_wind_field_transform: Optional[tuple] = None  # (rotation_deg, scale, offset)
        # Wind-field transform of the current episode (wind_current only) and
        # whether a transform set after a rewind was verified to take effect.
        self.episode_wind_field_transform: Optional[tuple] = None
        self._rewind_wind_verified = False
        # True once reset() set initial_wind_field_transform itself (wind_current).
        self._owns_initial_wind_transform = False
        self.library_path  = library_path  or os.path.join(simexe_path, 'exe/installed')
        self.config_path   = config_path   or os.path.join(simexe_path, 'config/simXdrive.config.xml')
        self.output_path   = output_path   or os.path.join(
            'C:/Users', user_login, 'Desktop', 'testExercise', 'results'
        )

        self.exercise_path = exercise_path or os.path.join(
            base_dir, f"{scenario_suffix}.tab"
        )
        if not os.path.exists(self.exercise_path):
            if ship_model not in (ShipModel.MYZAKO_29, ShipModel.MYZAKO_37):
                self.exercise_path = exercise_path or os.path.join(
                    base_dir, f"{scenario}.tab"
                )

            if not os.path.exists(self.exercise_path):
                _logger.warning("[FhSimBridge] Exercise file not found: '%s'", self.exercise_path)
                self.exercise_path_exists = False

        self.initialize_math_model = True
        self.math_model_initialized = False
        self._ship_interface = None
        self._math_model = None

        # Per-instance logger, so instances with different output_path do not
        # interfere and the root logger is untouched.
        self._logger = logging.getLogger(f"FhSimBridge.{id(self)}")
        self._logger.setLevel(logging.INFO)

        if not self._logger.handlers:
            log_path = os.path.join(self.output_path, 'math_model.log')
            try:
                os.makedirs(self.output_path, exist_ok=True)
                handler = logging.FileHandler(log_path)
                handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
                self._logger.addHandler(handler)

                # Route SimAPI's own logger to the same file
                sim_logger = logging.getLogger("sim")
                sim_logger.setLevel(logging.INFO)
                sim_logger.addHandler(handler)
                sim_logger.propagate = False  # keep it out of the root handler
            except (OSError, FileNotFoundError) as exc:
                # Fallback to console if the log directory cannot be created.
                self._logger.addHandler(logging.StreamHandler())
                self._logger.warning(
                    "Could not open log file '%s': %s - logging to console.", log_path, exc
                )


    @property
    def ship_model(self):
        """
        ShipModel of the simulated ship.
        """

        return self._ship_model


    @property
    def initial_vessel_state(self):
        """
        Initial vessel state used to seed the math model on first reset().
        """

        return self._initial_vessel_state


    @initial_vessel_state.setter
    def initial_vessel_state(self, value):
        """
        Set the initial vessel state; required before the first reset().

        Args:
            value (VesselState): Initial state to set.  Positions (x, y) in
                metres, heading in radians (counter-clockwise), velocities
                (u, v) in m/s (v port positive), yaw rate r in rad/s
                (counter-clockwise positive).
        """

        self._initial_vessel_state = value


    def __repr__(self):
        """
        Return "FH", the bridge identifier used in logs.
        """

        return "FH"


    def _initialize_math_model(self, sample_frequency=10.0, snapshot_window=10000, snapshot_frequency=5):
        """
        Initialize the FH Simulator math model.

        Sets up the SimAPI config, loads the exercise file, applies the
        initial vessel state, and calls MathModel.Initialize().

        Args:
            sample_frequency (float): Math model update rate [Hz].
            snapshot_window (int): Maximum rewind window [s].
            snapshot_frequency (int): Snapshot interval [s].

        Raises:
            RuntimeError: If the exercise file is missing, the initial vessel
                state has not been set, or Initialize() returns False.
        """

        if not self.exercise_path_exists:
            raise RuntimeError(
                f"Exercise file does not exist: '{self.exercise_path}'. "
                "Check SIMEXE_REPO_FOLDER and the scenario name."
            )

        if self._initial_vessel_state is None:
            raise RuntimeError(
                "Initial vessel state has not been set. "
                "Assign FhSimBridge.initial_vessel_state before calling reset()."
            )

        original_sys_path = sys.path.copy()

        try:
            sys.path.append(self.library_path)
            from sim import Config, Exercise, MathModel, SimException

            config = Config(self.config_path)
            config.SetOutputDir(self.output_path)
            if self.rewind_enabled:
                config.SetRewindEnabled(True)
                config.SetRewindConfig(snapshot_window, snapshot_frequency)
            else:
                config.SetRewindEnabled(False)
            config.SetMathModelFrequency(sample_frequency)
            config.ClearConnections()

            exercise = Exercise(str(self.exercise_path), config)  # .NET ctor needs a str, not a Path
            ship_config = exercise.getShipConfig()

            ship_config.setInitialPosition(float(self._initial_vessel_state.x), float(self._initial_vessel_state.y))
            ship_config.setInitialHeading(float(np.degrees(self._initial_vessel_state.heading)))
            ship_config.setInitialSurgeVelocity(float(self._initial_vessel_state.u))
            # Sway is port positive in both FH and the framework (fw.ships.vessel_state).
            ship_config.setInitialSwayVelocity(float(self._initial_vessel_state.v))
            ship_config.setInitialYawVelocity(float(np.degrees(self._initial_vessel_state.r)))
            if self.initial_propeller_rps is not None:
                ship_config.setInitialPropellerRps(float(self.initial_propeller_rps))
            if self.initial_rudder_angle is not None:
                ship_config.setInitialRudderAngle(float(self.initial_rudder_angle))

            self._math_model = MathModel()

            # Release a model whose Initialize() failed, so no native model leaks.
            try:
                initialized = self._math_model.Initialize(config, exercise)
            except BaseException:
                self._release_uninitialized_model()
                raise
            if not initialized:
                self._release_uninitialized_model()
                raise RuntimeError(
                    "Math model initialisation failed: MathModel.Initialize() returned False. "
                    "Check the exercise file, config, and SimAPI installation."
                )

            # From here on, any failure disposes the native model before raising.
            self.math_model_initialized = True
            self.initialize_math_model = False
            try:
                self._math_model.enableBridge()

                self._ship_interface = self._math_model.getShipInterface(0)
                self._logger.info("Math model initialised successfully.")

                # After enableBridge(), before the first simulateSeconds().
                if self.initial_wind_field_transform is not None:
                    self._apply_wind_field_transform(*self.initial_wind_field_transform)

                self.math_model.simulateSeconds(0.125)  # initialize velocities
            except BaseException:
                # Keep the log handlers, so the error and a retry are still logged.
                self._dispose_math_model()
                raise

        except SimException as e:
            self.math_model_initialized = False
            self._math_model = None
            self._ship_interface = None
            raise RuntimeError(f"SimException during math model initialisation: {e.ToString()}") from e
        finally:
            sys.path = original_sys_path


    def _release_uninitialized_model(self) -> None:
        """
        Best-effort Terminate/Dispose of a MathModel whose Initialize() failed.
        """

        model, self._math_model = self._math_model, None
        if model is not None:
            self._terminate_and_dispose(model, "releasing a failed math model")


    def _terminate_and_dispose(self, model, context: str) -> None:
        """
        Call Terminate() and Dispose() on `model` independently; failures are
        logged, not raised, so cleanup never masks the original error.
        """

        for step in ("Terminate", "Dispose"):
            try:
                getattr(model, step)()
            except Exception as exc:
                self._logger.warning("%s() failed while %s: %s", step, context, exc)


    def _apply_wind_field_transform(self, rotation: float, scale: float, offset: float) -> None:
        """
        Set the exercise's wind-field transform (Rotation [deg], Scale, Offset).

        Raises:
            RuntimeError: If the exercise has no wind field control.
        """

        control = self._math_model.getEnvironmentInterface().getWindFieldControl()
        if control is None:
            raise RuntimeError("A wind-field transform is requested, but the exercise "
                               f"'{self.exercise_path}' has no wind field control.")
        transform = control.getFieldTransform()
        transform.Rotation = float(rotation)
        transform.Scale = float(scale)
        transform.Offset = float(offset)
        control.setFieldTransform(transform)


    @property
    def math_model(self):
        """
        The active MathModel instance.

        Raises:
            RuntimeError: If the math model has not been initialized yet.
        """

        if self._math_model is None:
            raise RuntimeError(
                "Math model is not initialised. Call reset() first."
            )
        return self._math_model


    @property
    def ship_interface(self):
        """
        The active ShipInterface instance.

        Raises:
            RuntimeError: If the ship interface has not been initialized yet.
        """

        if self._ship_interface is None:
            raise RuntimeError(
                "Ship interface is not initialised. Call reset() first."
            )
        return self._ship_interface


    def reset(self, time_point: float = 0.25, wind_rng: Optional[np.random.Generator] = None):
        """
        Reset the simulation.

        The first call initializes the math model from initial_vessel_state;
        later calls rewind to time_point using the snapshots. With wind_current,
        a new wind-field transform is drawn from wind_rng and WIND_WARMUP_TIME
        is simulated, so the episode starts WIND_WARMUP_TIME after the reset point.

        Args:
            time_point (float): Simulation time [s] to rewind to, at least
                0.25 s (SimAPI cannot rewind to t=0) and at most the current
                simulation time (not checked here).
            wind_rng: Random generator for the wind field; required with
                wind_current, ignored otherwise.

        Raises:
            ValueError: wind_current without wind_rng, or with a user-set
                initial_wind_field_transform.
            RuntimeError: If time_point < 0.25 s, rewind is disabled,
                initialization fails, or the wind-field transform has no effect.
        """

        if self.wind_current and wind_rng is None:
            raise ValueError("FhSimBridge with wind_current=True needs a wind_rng in reset().")
        if self.wind_current and self.initial_wind_field_transform is not None \
                and not self._owns_initial_wind_transform:
            raise ValueError("initial_wind_field_transform must not be set with wind_current=True: "
                             "reset() draws the wind field per episode.")
        transform = draw_wind_field_transform(wind_rng) if self.wind_current else None

        if self.initialize_math_model:
            if transform is not None:
                # Applied in _initialize_math_model before the first simulateSeconds().
                self.initial_wind_field_transform = (transform[0], transform[1], 0.0)
                self._owns_initial_wind_transform = True
            self._initialize_math_model(snapshot_window=self.SNAPSHOT_WINDOW, snapshot_frequency=5000)
            if transform is not None:
                try:
                    self._warm_up_wind()
                except BaseException:
                    # As for other init failures: the next reset() re-initializes.
                    self._dispose_math_model()
                    raise
                self.episode_wind_field_transform = self.initial_wind_field_transform
            return

        if not self.rewind_enabled:
            raise RuntimeError(
                "FhSimBridge was created with rewind_enabled=False, so it cannot rewind; "
                "dispose() it and create a new bridge for the next episode."
            )
        else:
            if time_point < 0.25:
                raise RuntimeError(
                    f"time_point must be >= 0.25 s (got {time_point:.3f} s). "
                    "The SimAPI rewind mechanism does not support rewinding to t=0."
                )

            sim_time = self._math_model.getSimInterface().getSimulationTime()

            if sim_time > self.SNAPSHOT_WINDOW:
                self._logger.warning(
                    "Simulation time %.1f s exceeds snapshot window %d s. "
                    "Full rewind to t=%.2f s may not be possible.",
                    sim_time, self.SNAPSHOT_WINDOW, time_point,
                )

            self._math_model.SimulateRewindTo(time_point)

        if transform is not None:
            if not self._rewind_wind_verified:
                self._verify_rewind_wind_transform(time_point)
            self._apply_wind_field_transform(transform[0], transform[1], 0.0)
            self._warm_up_wind()
            self.episode_wind_field_transform = (transform[0], transform[1], 0.0)


    def _warm_up_wind(self) -> None:
        """
        Simulate WIND_WARMUP_TIME after a wind-field transform was set.

        Raises:
            RuntimeError: If WindState is bit-identical before and after (it
                still shows the previous wind, as in sim_fh_dataset_generator).
        """

        before = self._wind_state_tuple()
        self.math_model.simulateSeconds(WIND_WARMUP_TIME)
        after = self._wind_state_tuple()
        if after == before:
            raise RuntimeError(
                f"FH WindState unchanged after WIND_WARMUP_TIME={WIND_WARMUP_TIME} s ({after}): "
                f"the wind-field transform did not take effect (or needs a longer warm-up)."
            )


    def _wind_state_tuple(self) -> tuple:
        """
        (Speed, Direction) of FH's current WindState.
        """

        wind_state = self.ship_interface.getWindState()
        return float(wind_state.Speed), float(wind_state.Direction)


    def _verify_rewind_wind_transform(self, time_point: float) -> None:
        """
        One-time check that a wind-field transform set after SimulateRewindTo
        takes effect: from the same rewound state, Scale WIND_PROBE_SCALE must
        give a clearly stronger true wind than Scale 0. Leaves the model rewound
        to time_point.

        Raises:
            RuntimeError: If the difference is below WIND_PROBE_MIN_DIFF or not finite.
        """

        speeds = []
        for i, scale in enumerate((0.0, self.WIND_PROBE_SCALE)):
            if i > 0:
                self._math_model.SimulateRewindTo(time_point)
            self._apply_wind_field_transform(0.0, scale, 0.0)
            self.math_model.simulateSeconds(WIND_WARMUP_TIME)
            speeds.append(fh_true_wind_speed(self.ship_interface))
        self._math_model.SimulateRewindTo(time_point)

        diff = speeds[1] - speeds[0]
        if not np.isfinite(diff) or diff < self.WIND_PROBE_MIN_DIFF:
            raise RuntimeError(
                f"FH ignores a wind-field transform set after SimulateRewindTo: true wind "
                f"{speeds[0]:.2f} m/s at Scale 0 and {speeds[1]:.2f} m/s at Scale "
                f"{self.WIND_PROBE_SCALE:g} (expected a difference of at least "
                f"{self.WIND_PROBE_MIN_DIFF:g} m/s) in '{self.exercise_path}'. Per-episode wind "
                f"randomization with rewind does not work for this exercise."
            )
        self._logger.info("Wind-field transform after rewind verified: true wind %.2f m/s at Scale 0, "
                          "%.2f m/s at Scale %g.", speeds[0], speeds[1], self.WIND_PROBE_SCALE)
        self._rewind_wind_verified = True


    def dispose(self):
        """
        Release all SimAPI resources and close this bridge's log handlers.

        A later reset() re-initializes from scratch. Idempotent.
        """

        self._dispose_math_model()

        # Also remove the handler from SimAPI's global "sim" logger.
        sim_logger = logging.getLogger("sim")
        for handler in self._logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
            self._logger.removeHandler(handler)
            sim_logger.removeHandler(handler)


    def _dispose_math_model(self) -> None:
        """
        Terminate and dispose the math model (best effort) and reset the
        bridge so the next reset() re-initializes; the log handlers are kept.
        No-op when no model is initialized.
        """

        if self.math_model_initialized:
            try:
                self._terminate_and_dispose(self._math_model, "disposing")
            finally:
                # Clear dead references; the next reset() re-initializes.
                self._math_model = None
                self._ship_interface = None
                self.math_model_initialized = False
                self.initialize_math_model = True
