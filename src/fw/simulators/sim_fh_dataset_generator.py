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
    # Import simulation modules if not already available
    if not SIM_AVAILABLE:
        sys.path.append(sim_config.lib_path)
        from sim import Config, Exercise, MathModel, SimException

    output_path = Path(sim_setup.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logfile_path = output_path / 'math_model.log'
    logging.basicConfig(filename=str(logfile_path), level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    env = PySimEnv(render_mode=None, time_step=sim_setup.dt, wind=False, current=False)

    sim_setup.output.status = "PASSED"
    sim_setup.output.time_seconds = -1.0

    time_started = time.time()

    _math_model = None
    _ship_interface = None

    try:
        # Configure the math model
        config = Config(sim_config.config_path)
        config.SetOutputDir(str(sim_setup.output_path))
        config.SetRewindEnabled(False)
        config.SetMathModelFrequency(DEFAULT_MATH_MODEL_FREQUENCY)
        config.ClearConnections()

        exercise = Exercise(sim_setup.exercise_path, config)
        ship_config = exercise.getShipConfig()
        rng = np.random.default_rng()

        ship_config.setInitialPosition(sim_setup.x_pos, sim_setup.y_pos)
        ship_config.setInitialPropellerRps(rng.uniform(low=[0.3], high=[5.5])[0])
        ship_config.setInitialRudderAngle(rng.uniform(low=[-30.0], high=[30.0])[0])

        # Initialize the math model and enable the bridge
        _math_model = MathModel()
        if not _math_model.Initialize(config, exercise):
            error_msg = "MathModel initialization failed!"
            print(error_msg)
            sim_setup.output.status = "FAILED"
            sim_setup.output.info['error'] = error_msg
            return sim_setup

        pos_info = f"[{sim_setup.x_pos:10.3f} {sim_setup.y_pos:10.3f}]"
        print(f"Math model initialized! Starting simulation at {pos_info}")

        _math_model.enableBridge()
        _math_model.simulateSeconds(0.125)  # initialize velocities (surge, sway, yaw)

        # Get the ship interface
        _ship_interface = _math_model.getShipInterface(0)
        initial_velocity_over_ground = _ship_interface.getShipVelocityOverGround()

        initial_state = np.array([
            sim_setup.x_pos,
            sim_setup.y_pos,
            np.radians(_ship_interface.getShipHeading()),
            initial_velocity_over_ground.x,
            initial_velocity_over_ground.y,
            _ship_interface.getShipYawRate()
        ], dtype=float)

        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        duration = rng.uniform(low=[5.0 * sim_setup.dt], high=[sim_setup.horizon_seconds])[0]
        steps_per_episode = max(1, int(duration / sim_setup.dt))

        # --- Start storing trajectory ---
        new_ship_pos = np.array([sim_setup.x_pos, sim_setup.y_pos], dtype=np.float32)
        traj = [new_ship_pos.copy()]

        for _ in range(steps_per_episode):
            if env._ship_in_open_water(new_ship_pos) and _ship_interface.getKeelClearance() >= MIN_KEEL_CLEARANCE:
                # Update rudder controls based on the turning action
                for rudder in _ship_interface.getRudderControls():
                    # Compensate for opposite behaviour between Python env and FH sim
                    rudder.setControlValue(float(-1.0 * action[0]))

                # Update propeller controls based on the thrust action
                for propeller in _ship_interface.getPropellerControls():
                    propeller.setEngineLeverValue(float(action[1]))

                # Simulate the ship's dynamics for a fixed period
                _math_model.simulateSeconds(sim_setup.dt)

                # Retrieve the updated ship position
                new_ship_pos = np.array([
                    _ship_interface.getShipPosition().x,
                    _ship_interface.getShipPosition().y
                ], dtype=np.float32)
                traj.append(new_ship_pos.copy())
            else:
                break

        sim_setup.output.trajectory = np.array(traj)  # shape (T, 2)

        if not env._ship_in_open_water(new_ship_pos) or _ship_interface.getKeelClearance() < 1.0:
            error_msg = "Ship not in open water!"
            sim_setup.output.status = "FAILED"
            sim_setup.output.info['error'] = error_msg
            return sim_setup

        # Calculate delta X and delta Y
        delta_x = _ship_interface.getShipPosition().x - sim_setup.x_pos
        delta_y = _ship_interface.getShipPosition().y - sim_setup.y_pos
        velocity_over_ground = _ship_interface.getShipVelocityOverGround()

        final_state = np.array([
            delta_x,
            delta_y,
            np.radians(_ship_interface.getShipHeading()),
            velocity_over_ground.x,
            velocity_over_ground.y,
            _ship_interface.getShipYawRate(),
        ], dtype=float)

        sim_setup.output.x_input = np.concatenate([initial_state, action], axis=0)
        sim_setup.output.y_output = final_state

    except (SimException, Exception) as e:
        error_msg = f"Simulation failed: {e}"
        print(error_msg)
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            filename, line_number, function_name, text = tb[-1]
            traceback_info = f"Exception occurred at {filename}, line {line_number} in {function_name}"
        else:
            traceback_info = "No traceback available"

        sim_setup.output.status = "FAILED"
        sim_setup.output.info['message'] = str(error_msg)
        sim_setup.output.info['exception_type'] = type(e).__name__
        sim_setup.output.info['traceback'] = traceback_info

    finally:
        if _math_model is not None:
            _math_model.Terminate()
            _math_model.Dispose()

            # Clean up output files (keep math_model.log)
            for item in os.listdir(sim_setup.output_path):
                if item == "math_model.log":
                    continue
                item_path = Path(sim_setup.output_path) / item
                if item_path.is_file():
                    item_path.unlink()

    sim_setup.output.time_seconds = time.time() - time_started
    return sim_setup


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
                    x_inputs.append(sim_setup.output.x_input)
                    y_outputs.append(sim_setup.output.y_output)
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
