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

# Configure logging
try:
    logfile = os.path.join('C:/Users', user_login, 'Desktop', 'testExercise', 'results', 'math_model.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(levelname)s: %(message)s')
except FileNotFoundError:
    logfile = None


class FhSimWrapper:
    """
    A wrapper class for interacting with the FH Sim math model and ship interface.

    This class provides methods to initialize, reset, and dispose the simulation environment,
    as well as access to the math model and ship interface.

    Attributes:
        initialize_math_model (bool): Flag to determine if the math model needs initialization.
        math_model_initialized (bool): Flag to track if the math model has been initialized.
        _math_model: The math model instance.
        _ship_interface: The ship interface instance.
    """

    SNAPSHOT_WINDOW = 100000

    def __init__(self,
                 ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
                 scenario: str = "DDShip_scenario1_nowindnocurrent",
                 library_path=None,
                 config_path=None,
                 exercise_path=None,
                 output_path=None):
        """
        Initialize the FhSimWrapper instance.

        Sets flags for math model initialization and tracks the initialization state.
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
        self.library_path = library_path or os.path.join(simexe_path, 'exe/installed')
        self.config_path = config_path or os.path.join(simexe_path, 'config/simXdrive.config.xml')
        self.exercise_path = exercise_path or os.path.join(
            base_dir, f"{scenario_suffix}.tab"
        )
        if not os.path.exists(self.exercise_path):
            if ship_model != ShipModel.MYZAKO_29 and ship_model != ShipModel.MYZAKO_37:
                self.exercise_path = exercise_path or os.path.join(
                    base_dir, f"{scenario}.tab"
                )

            if not os.path.exists(self.exercise_path):
                print(f"File '{self.exercise_path}' does not exists!")
                self.exercise_path_exists = False
        self.output_path = output_path or os.path.join('C:/Users', user_login, 'Desktop', 'testExercise', 'results')
        self.initialize_math_model = True
        self.math_model_initialized = False
        self._ship_interface = None
        self._math_model = None


    @property
    def initial_vessel_state(self):
        return self._initial_vessel_state


    @initial_vessel_state.setter
    def initial_vessel_state(self, value):
        self._initial_vessel_state = value


    def __repr__(self):
        """Returns the FH wrapper's user-friendly name.

        Returns:
            str: The name of the FH wrapper.
        """
        return "FH"


    def _initialize_math_model(self, sample_frequency=10.0, snapshot_window=10000, snapshot_frequency=5):
        """
        Initialize the math model for the simulation.

        Args:
            sample_frequency (float): The frequency at which the math model runs, in Hz.
            snapshot_window (int): The maximum snapshot window for rewinding, in seconds.
            snapshot_frequency (int): The frequency at which snapshots are taken, in seconds.

        Raises:
            SimAPI.SystemException: If the math model initialization fails.
        """
        if not self.exercise_path_exists:
            raise RuntimeError(f"Exercise path {self.exercise_path} does not exist!")

        self.initialize_math_model = False

        # Save the original sys.path
        original_sys_path = sys.path.copy()

        try:
            sys.path.append(self.library_path)
            from sim import Config, Exercise, MathModel, SimException       # sim.py is now part of the release package

            # Configure the math model
            config = Config(self.config_path)
            config.SetOutputDir(self.output_path)
            config.SetRewindEnabled(True)
            config.SetRewindConfig(snapshot_window, snapshot_frequency)  # First parameter is the maximum snapshot window in [s], second parameter is the snapshot frequency in [s]
            config.SetMathModelFrequency(sample_frequency)  # Math model frequency in Hz
            config.ClearConnections()

            exercise = Exercise(self.exercise_path, config)
            ship_config = exercise.getShipConfig()

            # Set initial vessel state
            if self._initial_vessel_state is not None:
                ship_config.setInitialPosition(float(self._initial_vessel_state.x), float(self._initial_vessel_state.y))
                ship_config.setInitialHeading(float(np.degrees(self._initial_vessel_state.heading)))
                ship_config.setInitialSurgeVelocity(float(self._initial_vessel_state.u))
                ship_config.setInitialSwayVelocity(float(-self._initial_vessel_state.v))
                ship_config.setInitialYawVelocity(float(np.degrees(self._initial_vessel_state.r)))
            else:
                raise RuntimeError("Initial vessel state not set.")

            # Initialize the math model and enable the bridge
            self._math_model = MathModel()
            if self._math_model.Initialize(config, exercise):
                print("Math model initialized!")
                self._math_model.enableBridge()

                # Get the ship interface
                self._ship_interface = self._math_model.getShipInterface(0)
                self.math_model_initialized = True
            else:
                print("Math model initialization failed!")
        except SimException as e:
            self.math_model_initialized = False
            raise RuntimeError(f"{e.ToString()}")
        finally:
            # Restore the original sys.path
            sys.path = original_sys_path


    @property
    def math_model(self):
        """
        Get the math model instance.

        Returns:
            The math model instance.
        """
        if self._math_model is None:
            raise RuntimeError("Math model is not initialized.")

        return self._math_model


    @property
    def ship_interface(self):
        """
        Get the ship interface instance.

        Returns:
            The ship interface instance.
        """
        if self._ship_interface is None:
            raise RuntimeError("Ship interface is not initialized.")

        return self._ship_interface


    def reset(self, time_point=0.25):
        """
        Reset the simulation environment to its initial state or a specific time point.

        Args:
            time_point (float): The time point to rewind to, in seconds. Defaults to 0.25.
        """
        if self.initialize_math_model:
            self._initialize_math_model(snapshot_window=self.SNAPSHOT_WINDOW, snapshot_frequency=5000)
        else:
            if time_point < 0.25:
                raise RuntimeError("The time point should be larger than or equal to 0.25")

            simulation_time = self._math_model.getSimInterface().getSimulationTime()
            if simulation_time > self.SNAPSHOT_WINDOW:
                print(f"Completely rewinding to 0 will not be possible, total simulation time = {simulation_time}")

            self._math_model.SimulateRewindTo(time_point)


    def dispose(self):
        """
        Dispose the simulation environment and clean up resources.

        Terminates the math model, disposes of it, and cleans up the SimAPI.
        """
        if self.math_model_initialized:
            self.initialize_math_model = True
            self._math_model.Terminate()
            self._math_model.Dispose()
            #Sim.MathModel = None               # should not be required
        #Sim.disconnect_logger()                # is handled by the module's at-exit event