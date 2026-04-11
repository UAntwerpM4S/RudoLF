import os
import sys
import logging
import numpy as np

from typing import Optional
from fw.ships.ship_factory import ShipModel

# Define paths for simulation environment
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

    Manages the lifecycle of the FH Simulator: initialization, per-episode
    rewind, and final disposal. Interaction with the black-box simulator is
    limited to:
        - Setting initial position, heading, and velocities before init.
        - Rewinding to a fixed time point between episodes.
        - Reading position, velocity, heading, and keel clearance after each step.

    Attributes:
        initialize_math_model (bool): True when the math model needs (re)initialization.
        math_model_initialized (bool): True only after a successful Initialize() call.
        _math_model: MathModel instance. None until successfully initialized.
        _ship_interface: ShipInterface instance. None until successfully initialized.
    """

    SNAPSHOT_WINDOW = 100000  # Maximum rewind window [s] (~27 h)

    def __init__(self,
                 ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
                 scenario: str = "DDShip_scenario1_nowindnocurrent",
                 library_path=None,
                 config_path=None,
                 exercise_path=None,
                 output_path=None):
        """
        Initialize the FhSimBridge.

        Resolves all file paths and sets up instance logging. The math model
        itself is not initialized here — it is initialized lazily on the first
        call to reset().

        Args:
            ship_model (ShipModel): Determines the scenario file suffix.
            scenario (str): Base scenario name, matched against .tab files.
            library_path (str, optional): Path to the SimAPI native library.
            config_path (str, optional): Path to simXdrive.config.xml.
            exercise_path (str, optional): Explicit path to the .tab exercise file.
            output_path (str, optional): Directory for simulation output / logs.
        """

        self.exercise_path_exists = True

        base_dir = os.path.join(
            simexe_path,
            "database/areas/ScheldeSaeftinge_23_002/invoer/tra"
        )

        # Determine scenario suffix
        if ship_model == ShipModel.MYZAKO_DEFAULT:
            scenario_suffix = f"{scenario}_default"
        else:
            scenario_suffix = f"{scenario}_{ship_model.name.lower()}"

        self._initial_vessel_state = None
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
                print(f"[FhSimBridge] Exercise file not found: '{self.exercise_path}'")
                self.exercise_path_exists = False

        self.initialize_math_model = True
        self.math_model_initialized = False
        self._ship_interface = None
        self._math_model = None

        # Per-instance named logger instead of root logger.
        # Each instance gets a unique logger so output_path changes between
        # instances don't interfere, and the root logger is never touched.
        self._logger = logging.getLogger(f"FhSimBridge.{id(self)}")
        self._logger.setLevel(logging.INFO)

        if not self._logger.handlers:
            log_path = os.path.join(self.output_path, 'math_model.log')
            try:
                os.makedirs(self.output_path, exist_ok=True)
                handler = logging.FileHandler(log_path)
                handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
                self._logger.addHandler(handler)
            except (OSError, FileNotFoundError) as exc:
                # Fallback to console if the log directory cannot be created.
                self._logger.addHandler(logging.StreamHandler())
                self._logger.warning(
                    "Could not open log file '%s': %s — logging to console.", log_path, exc
                )


    @property
    def initial_vessel_state(self):
        """
        Initial vessel state used to seed the math model on first reset().
        """

        return self._initial_vessel_state


    @initial_vessel_state.setter
    def initial_vessel_state(self, value):
        """
        Initial vessel state.

        Args:
            value: Description.
        """

        self._initial_vessel_state = value


    def __repr__(self):
        """
        Returns the FH wrapper's user-friendly name.

        Returns:
            str: The name of the FH wrapper.
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

        # Save the original sys.path
        original_sys_path = sys.path.copy()

        try:
            sys.path.append(self.library_path)
            from sim import Config, Exercise, MathModel, SimException

            # Configure the math model
            config = Config(self.config_path)
            config.SetOutputDir(self.output_path)
            config.SetRewindEnabled(True)
            config.SetRewindConfig(snapshot_window, snapshot_frequency)
            config.SetMathModelFrequency(sample_frequency)
            config.ClearConnections()

            exercise = Exercise(self.exercise_path, config)
            ship_config = exercise.getShipConfig()

            # Set initial vessel state
            ship_config.setInitialPosition(float(self._initial_vessel_state.x), float(self._initial_vessel_state.y))
            ship_config.setInitialHeading(float(np.degrees(self._initial_vessel_state.heading)))
            ship_config.setInitialSurgeVelocity(float(self._initial_vessel_state.u))
            ship_config.setInitialSwayVelocity(float(-self._initial_vessel_state.v))
            ship_config.setInitialYawVelocity(float(np.degrees(self._initial_vessel_state.r)))

            # Initialize the math model and enable the bridge
            self._math_model = MathModel()

            if self._math_model.Initialize(config, exercise):
                self._math_model.enableBridge()

                # Get the ship interface
                self._ship_interface = self._math_model.getShipInterface(0)
                self.math_model_initialized = True
                self.initialize_math_model = False
                self._logger.info("Math model initialised successfully.")
                print("Math model initialized!")
            else:
                self._math_model = None
                raise RuntimeError(
                    "Math model initialisation failed: MathModel.Initialize() returned False. "
                    "Check the exercise file, config, and SimAPI installation."
                )

        except SimException as e:
            self.math_model_initialized = False
            self._math_model = None
            self._ship_interface = None
            raise RuntimeError(f"SimException during math model initialisation: {e.ToString()}") from e
        finally:
            # Restore the original sys.path
            sys.path = original_sys_path


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


    def reset(self, time_point: float = 0.25):
        """
        Reset the simulation to a specific time point.

        On the first call, the math model is fully initialized using the
        initial_vessel_state. On subsequent calls, the simulation is rewound
        to time_point using the snapshot mechanism.

        Args:
            time_point (float): Target simulation time [s] to rewind to.
                Must be in [0.25, current_simulation_time]. Defaults to 0.25 s
                (the earliest supported rewind point; t=0 is not supported by
                the SimAPI rewind mechanism).

        Raises:
            RuntimeError: If time_point is out of bounds or the rewind fails.
        """

        if self.initialize_math_model:
            self._initialize_math_model(snapshot_window=self.SNAPSHOT_WINDOW, snapshot_frequency=5000)
        else:
            # Validate time_point against both lower and upper bounds.
            # Previously only the lower bound was checked; passing a time_point beyond
            # the current simulation time would silently fail or crash inside the SimAPI.
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


    def dispose(self):
        """
        Terminate and dispose the math model, releasing all SimAPI resources.

        After this call the bridge is in a clean state: math_model_initialized
        is False, _math_model and _ship_interface are None, and initialize_math_model
        is True so a subsequent reset() will re-initialize from scratch.

        Safe to call multiple times (idempotent).
        """

        if self.math_model_initialized:
            try:
                self._math_model.Terminate()
                self._math_model.Dispose()
            except Exception as exc:
                self._logger.warning("Exception during dispose: %s", exc)
            finally:
                # Always clear state after dispose so the dead references
                # are not accessible via math_model / ship_interface properties,
                # and a subsequent reset() correctly re-initialises.
                self._math_model = None
                self._ship_interface = None
                self.math_model_initialized = False
                self.initialize_math_model = True

        for handler in self._logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
            self._logger.removeHandler(handler)
