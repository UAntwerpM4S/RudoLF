import os
import sys
import time
import shutil
import logging
import traceback
import concurrent
import numpy as np
import multiprocessing

from pathlib import Path
from functools import partial
from typing import Optional, Tuple
from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from fw.simulators.py_sim_env import PySimEnv
from concurrent.futures import ThreadPoolExecutor

# Constants for magic numbers
DEFAULT_MATH_MODEL_FREQUENCY = 10.0  # Hz
MIN_KEEL_CLEARANCE = 2.0
TIMEOUT_SECONDS = 60.0

# Try to import simulation modules early
SIM_AVAILABLE = False
Config = Exercise = MathModel = SimException = None

try:
    from sim import Config, Exercise, MathModel, SimException

    SIM_AVAILABLE = True
except ImportError:
    # Will be imported later with sys.path modification
    pass

try:
    simexe_path = os.environ['SIMEXE_REPO_FOLDER']
except KeyError:
    simexe_path = ""

try:
    user_login = os.getlogin()
except OSError:
    user_login = ""

# Configure global logging
try:
    logfile = Path('C:/Users') / user_login / 'Desktop' / 'testExercise' / 'results' / 'math_model.log'
    logging.basicConfig(filename=str(logfile), level=logging.INFO, format='%(levelname)s: %(message)s')
except (FileNotFoundError, OSError):
    logfile = None


@dataclass
class SimConfig:
    """Configuration for simulation environment."""
    lib_path: str = ""
    config_path: str = ""
    timeout_seconds: float = TIMEOUT_SECONDS


@dataclass
class SimOutput:
    """Container for simulation results."""
    status: str = ""
    time_seconds: float = 0.0
    x_input: Optional[np.ndarray] = None
    y_output: Optional[np.ndarray] = None
    trajectory: Optional[np.ndarray] = None
    info: Optional[dict] = field(default_factory=dict)


@dataclass
class SimSetup:
    """Setup parameters for individual simulation run."""
    exercise_path: str = ""
    output_path: str = ""
    x_pos: float = 0.0
    y_pos: float = 0.0
    dt: float = 0.1
    horizon_seconds: float = 3.0
    output: SimOutput = field(default_factory=SimOutput)


def plot_xy_points_and_trajectories(env: PySimEnv, xy: np.ndarray, trajectories: list) -> None:
    """
    Plot river polygon with start points and trajectories.

    Assumes env has:
    - polygon_shape attribute with exterior.xy
    - checkpoints attribute (optional)

    Args:
        env: PySimEnv instance with river polygon
        xy: Array of start positions (N, 2)
        trajectories: List of trajectory arrays
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 10))

        # --- River polygon ---
        river_poly = env.polygon_shape
        x_poly, y_poly = river_poly.exterior.xy
        ax.plot(x_poly, y_poly, "k-", linewidth=1.0)
        ax.fill(x_poly, y_poly, color="lightblue", alpha=0.30)

        # --- Sampled points (start positions) ---
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=20,
            c="darkred",
            alpha=0.9,
            edgecolors="white",
            linewidth=0.4,
            label="Start positions"
        )

        # --- Trajectories ---
        for traj in trajectories:
            ax.plot(
                traj[:, 0], traj[:, 1],
                linestyle="-",
                linewidth=1.0,
                alpha=0.35,
                color="green",
            )

        # --- Optional checkpoints ---
        if hasattr(env, "checkpoints"):
            for cp in env.checkpoints:
                if "pos" in cp:
                    ax.plot(cp["pos"][0], cp["pos"][1], "go", markersize=4)

        ax.set_aspect("equal", "box")
        ax.set_title("Sampled Points + Trajectories")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend()

        # Save the figure
        plt.savefig("generated_points.png", bbox_inches='tight', dpi=300)
        print("\nPath visualization saved as 'generated_points.png'.")

        plt.show()
    finally:
        plt.close('all')


def load_supervised_dataset(file_name: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load the supervised dataset from CSV instead of running simulations.

    The CSV layout is assumed to be:
        col 0:  start_x
        col 1:  start_y
        col 2:  heading
        col 3:  surge
        col 4:  sway
        col 5:  yaw
        col 6:  rudder_action
        col 7:  thrust_action
        col 8:  delta_x
        col 9:  delta_y
        col10:  out_heading
        col11:  out_surge
        col12:  out_sway
        col13:  out_yaw

    Returns:
        Tuple of (x_inputs, y_outputs, counter)
    """
    data = np.loadtxt(file_name, delimiter=",")

    # --- Extract components from the CSV ---
    xy = data[:, 0:2]  # (N,2) start positions
    initial_states = data[:, 2:8]  # (N,6) heading, surge, sway, yaw, rudder_action, thrust_action
    actions = data[:, 6:8]  # last 2 of the above
    y_outputs = data[:, 8:14]  # (N,6)

    # X = [initial_state(6) + actions(2)] = (N,8)
    x_inputs = np.hstack([initial_states[:, 0:6], actions])

    x_values = x_inputs.astype(np.float32)
    y_values = y_outputs.astype(np.float32)
    counter = x_values.shape[0]

    # --- Build dummy trajectories for plotting ---
    trajectories = []
    for i in range(counter):
        start = xy[i].astype(float)
        delta = data[i, 8:10].astype(float)
        end = start + delta
        trajectories.append(np.vstack([start, end]))

    # Use a dummy env with polygon to reuse the plotting function
    env = PySimEnv(render_mode=None, time_step=0.1, wind=False, current=False)
    plot_xy_points_and_trajectories(env, xy, trajectories)

    return x_values, y_values, counter


def run_simulation(sim_config: SimConfig, sim_setup: SimSetup) -> SimSetup:
    """
    Run a single simulation with given configuration and setup.

    Note: Each simulation runs in its own subprocess to avoid
    simulator concurrency issues.
    """
    # Constants
    initialization_time = 0.125
    position_format_string = "[{:10.3f} {:10.3f}]"

    # Lazy import of simulator modules
    if not SIM_AVAILABLE:
        sys.path.append(sim_config.lib_path)
        from sim import Config, Exercise, MathModel, SimException

    output_path = Path(sim_setup.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logfile_path = output_path / "math_model.log"
    logging.basicConfig(
        filename=str(logfile_path),
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    env = PySimEnv(
        render_mode=None,
        time_step=sim_setup.dt,
        wind=False,
        current=False,
    )

    sim_setup.output.status = "PASSED"
    sim_setup.output.time_seconds = -1.0

    _math_model = None
    _ship_interface = None
    time_started = time.time()

    try:
        # -----------------------------
        # Configure math model
        # -----------------------------
        config = Config(sim_config.config_path)
        config.SetOutputDir(str(sim_setup.output_path))
        config.SetMathModelFrequency(DEFAULT_MATH_MODEL_FREQUENCY)
        config.SetRewindEnabled(False)
        config.ClearConnections()

        exercise = Exercise(sim_setup.exercise_path, config)
        ship_config = exercise.getShipConfig()
        rng = np.random.default_rng()

        # Set initial conditions with explicit parameter names
        ship_config.setInitialPosition(sim_setup.x_pos, sim_setup.y_pos)
        ship_config.setInitialPropellerRps(rng.uniform(low=[0.3], high=[5.5])[0])
        ship_config.setInitialRudderAngle(rng.uniform(low=[-30.0], high=[30.0])[0])
        ship_config.setInitialHeading(rng.uniform(low=[0.0], high=[360.0])[0])

        # -----------------------------
        # Initialize math model
        # -----------------------------
        _math_model = MathModel()
        if not _math_model.Initialize(config, exercise):
            error_msg = "MathModel initialization failed!"
            print(error_msg)
            sim_setup.output.status = "FAILED"
            sim_setup.output.info["error"] = error_msg
            return sim_setup

        # Log initialization success
        pos_info = position_format_string.format(sim_setup.x_pos, sim_setup.y_pos)
        print(f"Math model initialized! Starting simulation at {pos_info}")

        _math_model.enableBridge()
        _math_model.simulateSeconds(initialization_time)  # initialize velocities

        # -----------------------------
        # Retrieve ship interface
        # -----------------------------
        _ship_interface = _math_model.getShipInterface(0)

        # Get initial state
        initial_heading = _ship_interface.getShipHeading()
        initial_vog = _ship_interface.getShipVelocityOverGround()
        initial_yaw_rate = _ship_interface.getShipYawRate()

        input_state = np.array(
            [
                sim_setup.x_pos,
                sim_setup.y_pos,
                0.0,
                np.radians(initial_heading),
                initial_vog.x,
                initial_vog.y,
                np.radians(initial_yaw_rate),
            ],
            dtype=np.float64,  # More precise than float
        )

        # Generate random action and duration
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        duration = rng.uniform(low=[5.0 * sim_setup.dt], high=[sim_setup.horizon_seconds])[0]
        steps_per_episode = max(1, int(round(duration / sim_setup.dt)))

        # -----------------------------
        # Trajectory storage
        # -----------------------------
        new_ship_pos = np.array([sim_setup.x_pos, sim_setup.y_pos], dtype=np.float32)

        input_states = []
        output_states = []
        traj = [new_ship_pos.copy()]

        # Get controls once for efficiency
        propellers = _ship_interface.getPropellerControls()
        rudders = _ship_interface.getRudderControls()
        has_propellers = len(propellers) > 0
        has_rudders = len(rudders) > 0

        # Simulation loop
        for step in range(steps_per_episode):
            # Check termination conditions
            is_in_open_water = env._ship_in_open_water(new_ship_pos)
            has_sufficient_keel_clearance = _ship_interface.getKeelClearance() >= MIN_KEEL_CLEARANCE

            if not (is_in_open_water and has_sufficient_keel_clearance):
                break

            # Store input state
            input_states.append(np.concatenate([input_state.copy(), action.copy()], axis=0))

            # Apply controls with safety checks
            if has_rudders:
                # Rudder control (sign inversion is intentional)
                rudder_value = float(-1.0 * action[0])
                for rudder in rudders:
                    rudder.setControlValue(rudder_value)

            if has_propellers:
                # Propeller control
                propeller_value = float(action[1])
                for propeller in propellers:
                    propeller.setEngineLeverValue(propeller_value)

            # Step simulation
            _math_model.simulateSeconds(sim_setup.dt)

            # Update position
            pos = _ship_interface.getShipPosition()
            new_ship_pos = np.array([pos.x, pos.y], dtype=np.float32)
            traj.append(new_ship_pos.copy())

            # Calculate deltas
            delta_x = pos.x - input_state[0]
            delta_y = pos.y - input_state[1]
            current_vog = _ship_interface.getShipVelocityOverGround()
            current_heading = _ship_interface.getShipHeading()
            current_yaw_rate = _ship_interface.getShipYawRate()

            # Store output state
            output_states.append(
                np.array(
                    [
                        delta_x,
                        delta_y,
                        sim_setup.dt,
                        np.radians(current_heading) - input_state[3],
                        current_vog.x - input_state[4],
                        current_vog.y - input_state[5],
                        np.radians(current_yaw_rate) - input_state[6],
                    ],
                    dtype=np.float64,
                )
            )

            # Update initial state for next iteration
            input_state = np.array(
                [
                    pos.x,
                    pos.y,
                    input_state[2] + sim_setup.dt,
                    np.radians(current_heading),
                    current_vog.x,
                    current_vog.y,
                    np.radians(current_yaw_rate),
                ],
                dtype=np.float64,
            )

        # Store trajectory
        sim_setup.output.trajectory = np.array(traj, dtype=np.float32)

        # Final validation
        final_keel_clearance = _ship_interface.getKeelClearance()
        final_is_open_water = env._ship_in_open_water(new_ship_pos)

        if not final_is_open_water or final_keel_clearance < 1.0:
            error_msg = "Ship not in open water!"
            sim_setup.output.status = "FAILED"
            sim_setup.output.info["error"] = error_msg
            return sim_setup

        # Store results if simulation succeeded
        sim_setup.output.x_input = input_states
        sim_setup.output.y_output = output_states

    except SimException as e:
        # Handle simulator-specific exceptions
        error_msg = f"Simulator exception: {e}"
        print(error_msg)
        _handle_exception(sim_setup, e, "SimException")

    except Exception as e:
        # Handle all other exceptions
        error_msg = f"Unexpected error: {e}"
        print(error_msg)
        _handle_exception(sim_setup, e, type(e).__name__)

    finally:
        # Cleanup resources
        if _math_model is not None:
            try:
                _math_model.Terminate()
                _math_model.Dispose()
            except Exception as e:
                print(f"Warning: Error during math model cleanup: {e}")

            # Cleanup output directory, preserving math_model.log
            try:
                for item in output_path.iterdir():
                    if item.name == "math_model.log":
                        continue
                    try:
                        if item.is_file():
                            item.unlink()
                    except (PermissionError, OSError) as e:
                        print(f"Warning: Could not delete {item}: {e}")
            except (PermissionError, OSError) as e:
                print(f"Warning: Could not list directory {output_path}: {e}")

    # Record execution time
    sim_setup.output.time_seconds = time.time() - time_started
    return sim_setup


def _handle_exception(sim_setup: SimSetup, exception: Exception, exception_type: str) -> None:
    """Handle exceptions consistently."""
    sim_setup.output.status = "FAILED"
    sim_setup.output.info["message"] = str(exception)
    sim_setup.output.info["exception_type"] = exception_type

    # Extract traceback information
    tb = traceback.extract_tb(exception.__traceback__)
    if tb:
        filename, line_number, function_name, _ = tb[-1]
        sim_setup.output.info["traceback"] = (
            f"Exception occurred at {filename}, "
            f"line {line_number} in {function_name}"
        )
    else:
        sim_setup.output.info["traceback"] = "No traceback available"


def run_in_process(sim_setup: SimSetup, sim_config: SimConfig) -> SimSetup:
    """
    Run simulation in a separate subprocess.

    Note: Each simulation runs in its own process to avoid
    simulator concurrency issues. This is intentional.
    """
    p = multiprocessing.Pool(1)
    res = p.apply_async(run_simulation, args=[sim_config, sim_setup])

    try:
        return res.get(sim_config.timeout_seconds)
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        p.terminate()
        sim_setup.output.status = "TIMEOUT"
        sim_setup.output.time = sim_config.timeout_seconds
        return sim_setup
    finally:
        p.close()
        p.join()


def collect_supervised_dataset(
        nbr_samples: int = 5000,
        horizon_seconds: float = 3.0,
        dt: float = 0.1,
        workers: int = 4,
        output_file: str = "ship_dynamics_dataset.csv",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Collect supervised dataset by running multiple simulations.

    Args:
        nbr_samples: Number of samples to generate
        horizon_seconds: Maximum simulation time
        dt: Time step for simulation
        workers: Number of worker threads (for compatibility, but simulations run sequentially)
        output_file: Output CSV file path

    Returns:
        Tuple of (x_inputs, y_outputs, successful_count)
    """
    # Input validation
    if nbr_samples <= 0:
        raise ValueError("nbr_samples must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")

    # Create SimConfig with default values (will be updated below)
    sim_config = SimConfig()
    sim_config.lib_path = str(Path(simexe_path) / 'exe' / 'installed')
    sim_config.config_path = str(Path(simexe_path) / 'config' / 'simXdrive.config.xml')
    sim_config.timeout_seconds = TIMEOUT_SECONDS

    output_path = Path('C:/Users') / user_login / 'Desktop' / 'testExercise' / 'results'

    # Clear output folder
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    exercise_path = Path(
        simexe_path) / 'database' / 'areas' / 'ScheldeSaeftinge_23_002' / 'invoer' / 'tra' / 'DDShip_scenario2_windnocurrent.tab'

    # Create array of SimSetup test items
    env = PySimEnv(render_mode=None, time_step=dt, wind=False, current=False)

    setups = []
    x_inputs = []
    y_outputs = []

    xy = env.sample_point_in_river(nbr_samples)
    x0 = xy[:, 0]
    y0 = xy[:, 1]

    nbr_samples = len(xy)
    trajectories = []
    counter = 0

    for i in range(nbr_samples):
        setup = SimSetup(
            exercise_path=str(exercise_path),
            output_path=str(output_path / str(i)),
            x_pos=float(x0[i]),
            y_pos=float(y0[i]),
            dt=dt,
            horizon_seconds=horizon_seconds,
            output=SimOutput(status="", time_seconds=0.0)
        )
        setups.append(setup)

    # Setup thread pool for parallel test execution
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Prepare test execution function with fixed math_model_path parameter
        execute_with_param = partial(run_in_process, sim_config=sim_config)

        # Submit all tasks to the executor
        future_to_test = {
            executor.submit(execute_with_param, setup): (i, setup)
            for i, setup in enumerate(setups)
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_test):
            try:
                # Get the completed test result
                sim_setup = future.result()
                test_item = future_to_test[future]
                index, _ = test_item

                if sim_setup.output.status == "PASSED":
                    # Update test config with results
                    for i in range(len(sim_setup.output.x_input)):
                        x_inputs.append(sim_setup.output.x_input[i])
                        y_outputs.append(sim_setup.output.y_output[i])
                    trajectories.append(sim_setup.output.trajectory)
                    print(f"Simulation {index} succeeded: {sim_setup.output.status}")
                    counter += 1
                else:
                    error_info = sim_setup.output.info.get('error', 'Unknown error')
                    print(f"Simulation {index} failed: {error_info}")
            except Exception as exc:
                print(f"Simulation {index} generated an exception: {exc}")

    print(f"Successful simulations: {counter}")

    if counter == 0:
        print("Warning: No successful simulations!")
        return np.array([]), np.array([]), 0

    x_values = np.vstack(x_inputs).astype(np.float32)
    y_values = np.vstack(y_outputs).astype(np.float32)

    store_data = np.hstack([x_values, y_values])  # shape: (nbr_samples, 8 + 6 = 14)
    np.savetxt(output_file, store_data, delimiter=",", fmt="%.6f")
    plot_xy_points_and_trajectories(env, xy, trajectories)

    return x_values, y_values, counter


# Example: Generate dataset
if __name__ == "__main__":
    x_vals, y_vals, successful_count = collect_supervised_dataset(
        nbr_samples=10000,
        horizon_seconds=5.0,
        dt=0.1,
        workers=max(1, multiprocessing.cpu_count() - 1)
    )

    print(f"Dataset generation complete. Successful samples: {successful_count}")
    print("Dataset shapes:")
    print(f"X: {x_vals.shape}")  # [N, 8] → (state(6) + action(2))
    print(f"Y: {y_vals.shape}")  # [N, 6] → final state

    if successful_count > 0:
        np.save("ship_dynamics_inputs.npy", x_vals)
        np.save("ship_dynamics_outputs.npy", y_vals)
        print("Dataset saved to ship_dynamics_inputs.npy and ship_dynamics_outputs.npy")
