import copy
import importlib
import numpy as np

from fw.simulators.py_sim_env import PySimEnv


class FhSimEnv(PySimEnv):
    """
    An environment for ship navigation using the FH Simulator.

    This class extends the `PySimEnv` class to integrate with the FH Simulator,
    providing realistic ship dynamics and control. It allows for simulation of ship
    navigation, including path-following, and environmental effects.

    Attributes:
        _fh_sim: An instance of the FH Simulator used for ship dynamics simulation.
    """

    def __init__(self, render_mode=None, time_step=1.0, max_steps=15000, verbose=None, target_pos=None, ship_pos=None, wind=False, current=False):
        """
        Initialize the FhSimEnv environment.

        Args:
            render_mode (str, optional): The rendering mode. Defaults to None.
            max_steps (int, optional): The maximum number of steps per episode. Defaults to 200.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to None.
            target_pos (tuple, optional): The target position (x, y) for the ship. Defaults to None.
            ship_pos (tuple, optional): The initial position (x, y) of the ship. Defaults to None.
            wind (bool, optional): Whether to enable wind effects. Defaults to False.
            current (bool, optional): Whether to enable current effects. Defaults to False.
        """
        super().__init__(render_mode, time_step, max_steps, verbose, target_pos, ship_pos, wind, current)

        try:
            fh_wrapper_module = importlib.import_module('fw.simulators.fh_sim_wrapper')
            fh_wrapper_class = getattr(fh_wrapper_module, "FhSimWrapper")
            self._fh_sim = fh_wrapper_class(self.ship_pos)
        except ModuleNotFoundError:
            self._fh_sim = None


    def __deepcopy__(self, memo):
        """
        Create a deep copy of the FhSimEnv instance, excluding the FH Simulator.

        This method ensures that the FH Simulator instance (`_fh_sim`) is not deep-copied,
        as it may contain non-picklable or shared resources.

        Args:
            memo (dict): A dictionary used to track already copied objects.

        Returns:
            FhSimEnv: A deep copy of the current instance.
        """
        new_instance = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            if key == '_fh_sim':
                setattr(new_instance, key, value)   # Retain the original reference
            else:
                setattr(new_instance, key, copy.deepcopy(value, memo))  # Deep copy all other attributes

        return new_instance


    def _reduce_path(self, path, start_pos):
        """Reduces the path to start from the point closest to the given start position.

        This method finds the point in the path that is closest to `start_pos`, then slices
        the path so that it begins just after that point and continues to the original end.
        The `start_pos` itself is not added to the returned path.

        Args:
            path (list of np.ndarray or array-like): The original path as a list of coordinate points.
            start_pos (np.ndarray or array-like): The new starting position to align the path with.

        Returns:
            list of np.ndarray: The reduced path starting after the closest point to `start_pos`.

        Note:
            - If `start_pos` is not on the path, the method finds the closest match and starts from there.
            - The point closest to `start_pos` is **excluded** from the result.
        """
        # Find index of the point on the path closest to start_pos
        distances = [np.linalg.norm(p - start_pos) for p in path]
        closest_index = np.argmin(distances)

        # Slice the path from following point to the end
        reduced_path = path[closest_index+1:]

        return reduced_path


    @property
    def fh_sim(self):
        """
        Get the FH Simulator instance.

        Returns:
            object: The FH Simulator instance.
        """
        return self._fh_sim


    def randomize(self, randomization_scale=None):
        """
        Randomize the environment's initial conditions.

        This method can be overridden to introduce randomization in the ship's initial state,
        environmental conditions, or other parameters.

        Args:
            randomization_scale (float, optional): The scale of randomization. Defaults to None.
        """
        pass


    def env_specific_reset(self):
        """
        Reset the FH Simulator to its initial state.

        This method is called during environment reset to ensure the FH Simulator
        is properly reinitialized.
        """
        self._fh_sim.reset()


    def _update_ship_dynamics(self, action, alpha=0.3):
        """
        Update the ship's dynamics based on the provided action.

        This method applies the action to the FH Simulator, updates the ship's state,
        and ensures the ship remains within the environment's bounds.

        Args:
            action (tuple): A tuple containing two values (turning, thrust) that determine
                            the rudder angle and thrust power, respectively.
            alpha (float, optional): Smoothing factor for action application. Defaults to 0.3.
        """
        # Clip the action values to ensure they're within the valid range [-1, 1].
        turning_smooth = alpha * action[0] + (1 - alpha) * self.current_action[0]
        thrust_smooth = alpha * abs(action[1]) + (1 - alpha) * self.current_action[1]
        self.current_action = [turning_smooth, thrust_smooth]

        # Save the current ship position as the previous position for tracking.
        self.previous_ship_pos = copy.deepcopy(self.ship_pos)
        self.previous_heading = np.radians(self._fh_sim.ship_interface.getShipHeading())

        # Update rudder controls based on the turning action.
        for rudder in self._fh_sim.ship_interface.getRudderControls():
            rudder.setControlValue(float(-1.0*turning_smooth)) # this is to compensate for opposite behaviour of the Python environment
                                                        # in Python: -1 is turn right ; 1 is turn left
                                                        # FH sim: -1 is turn left ; 1 is turn right
        # Update propeller controls based on the thrust action.
        for propeller in self._fh_sim.ship_interface.getPropellerControls():
            propeller.setEngineLeverValue(float(thrust_smooth))

        # Simulate the ship's dynamics for a fixed period.
        self._fh_sim.math_model.simulateSeconds(self.time_step)

        # Retrieve the updated ship position from the ship interface.
        new_ship_pos = self._fh_sim.ship_interface.getShipPosition()
        velocity_over_ground = self._fh_sim.ship_interface.getShipVelocityOverGround()
        self.state[0] = new_ship_pos.x
        self.state[1] = new_ship_pos.y
        self.state[2] = np.radians(self._fh_sim.ship_interface.getShipHeading())
        self.state[3] = velocity_over_ground.x
        self.state[4] = velocity_over_ground.y
        self.state[5] = self._fh_sim.ship_interface.getShipYawRate()

        # Clip the ship's position to ensure it's within the environment's defined bounds.
        self.ship_pos = self.state[:2]
