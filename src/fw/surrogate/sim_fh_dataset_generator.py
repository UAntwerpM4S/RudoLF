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
from fw.environments.py_sim_env import PySimEnv
from concurrent.futures import ThreadPoolExecutor

# Constants for magic numbers
DEFAULT_MATH_MODEL_FREQUENCY = 10.0  # Hz
MIN_KEEL_CLEARANCE = 2.0
TIMEOUT_SECONDS = 60.0
CACHE_CLEAR_INTERVAL = 100        # Clear C:\cache every N process spawns
CACHE_FOLDER = Path(r"C:\cache")  # Windows crash-dump folder to periodically wipe

# Try to import simulation modules early
SIM_AVAILABLE = False
Config = Exercise = MathModel = SimException = None

try:
    from sim import Config, Exercise, MathModel, SimException

    SIM_AVAILABLE = True
except ImportError:
    # Will be imported later with sys.path modification
    pass

def clear_cache_folder(cache_path: Path = CACHE_FOLDER) -> None:
    """
    Remove all files and sub-directories inside *cache_path* without deleting
    the folder itself.  Silently skips items that cannot be deleted (e.g. still
    locked by another process) and logs a warning instead of crashing.

    Intended to be called every CACHE_CLEAR_INTERVAL process spawns so that
    crash-dumps written to C:\\cache by failed Python subprocesses do not
    accumulate and exhaust swap space.
    """
    if not cache_path.exists():
        return

    deleted, skipped = 0, 0
    for item in cache_path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            deleted += 1
        except (PermissionError, OSError) as exc:
            print(f"Warning: Could not remove cache item {item}: {exc}")
            skipped += 1

    print(f"Cache folder '{cache_path}' cleared — {deleted} item(s) removed"
          + (f", {skipped} skipped (locked)" if skipped else "."))


try:
    simexe_path = os.environ['SIMEXE_REPO_FOLDER']
except KeyError:
    simexe_path = ""

try:
    user_login = os.getlogin()
except OSError:
    user_login = ""

# Logging is configured per-simulation inside run_simulation so that each
# subprocess writes to its own scratch directory.
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

        plt.savefig("generated_points.png", bbox_inches='tight', dpi=300)
        print("\nPath visualization saved as 'generated_points.png'.")

        plt.show()
    finally:
        plt.close('all')


def load_supervised_dataset(file_name: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load the supervised dataset from CSV instead of running simulations.

    CSV column layout (11 columns, matching collect_supervised_dataset output):
        X – inputs (8 columns)
          col 0:  x                (m, world-frame position)
          col 1:  y                (m, world-frame position)
          col 2:  heading          (rad, normalized to [-pi, pi])
          col 3:  surge            (m/s, body-frame longitudinal velocity)
          col 4:  sway             (m/s, body-frame lateral velocity)
          col 5:  yaw_rate         (rad/s)
          col 6:  rudder_action    (normalized [-1, 1])
          col 7:  thrust_action    (normalized [-1, 1])

        Y – one-step delta outputs (3 columns)
          col  8: delta_surge      (m/s)
          col  9: delta_sway       (m/s)
          col 10: delta_yaw_rate   (rad/s)

    Returns:
        Tuple of (x_inputs, y_outputs, counter)
    """
    data = np.loadtxt(file_name, delimiter=",")

    expected_cols = 11
    if data.ndim != 2 or data.shape[1] != expected_cols:
        raise ValueError(
            f"Expected CSV with {expected_cols} columns, got shape {data.shape}. "
            "Re-generate the dataset or check the file path."
        )

    x_values = data[:, 0:8].astype(np.float32)   # x, y, heading, surge, sway, yaw_rate, rudder, thrust
    y_values = data[:, 8:11].astype(np.float32)  # delta_surge, delta_sway, delta_yaw_rate
    counter = x_values.shape[0]

    # Reconstruct start positions and dummy trajectories for the diagnostic plot.
    # Use the absolute x/y from X cols 0-1 as start positions.
    start_xy = x_values[:, 0:2].astype(np.float64)
    env = PySimEnv(render_mode=None, time_step=0.1, wind=False, current=False)
    # No world-frame deltas in Y, so trajectories are single-point stubs.
    trajectories = [np.vstack([start_xy[i], start_xy[i]]) for i in range(counter)]
    plot_xy_points_and_trajectories(env, start_xy, trajectories)

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

        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

        # Set initial conditions with explicit parameter names
        ship_config.setInitialPosition(sim_setup.x_pos, sim_setup.y_pos)
        ship_config.setInitialPropellerRps(float(5.5*action[1]))
        ship_config.setInitialRudderAngle(float(75.0*action[0]))
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
        initial_position = _ship_interface.getShipPosition()
        initial_heading = _ship_interface.getShipHeading()
        initial_vog = _ship_interface.getShipVelocityOverGround()
        initial_yaw_rate = _ship_interface.getShipYawRate()

        # Full state vector including absolute position:
        # [x, y, heading, surge, sway, yaw_rate]
        input_state = np.array(
            [
                initial_position.x,
                initial_position.y,
                env.normalize_angle(np.radians(initial_heading)),
                initial_vog.x,
                -initial_vog.y,
                np.radians(initial_yaw_rate),
            ],
            dtype=np.float64,
        )

        # Generate random action and duration
        duration = sim_setup.horizon_seconds # rng.uniform(low=[5.0 * sim_setup.dt], high=[sim_setup.horizon_seconds])[0]
        steps_per_episode = max(1, int(round(duration / sim_setup.dt)))

        # -----------------------------
        # Trajectory storage
        # -----------------------------
        curr_ship_pos = np.array([sim_setup.x_pos, sim_setup.y_pos], dtype=np.float32)

        input_states = []
        output_states = []
        traj = [curr_ship_pos.copy()]

        # Get controls once for efficiency
        propellers = _ship_interface.getPropellerControls()
        rudders = _ship_interface.getRudderControls()
        has_propellers = len(propellers) > 0
        has_rudders = len(rudders) > 0

        # Simulation loop
        for step in range(steps_per_episode):
            # Check termination conditions
            is_in_open_water = env._ship_in_open_water(curr_ship_pos)
            has_sufficient_keel_clearance = _ship_interface.getKeelClearance() >= MIN_KEEL_CLEARANCE

            if not (is_in_open_water and has_sufficient_keel_clearance):
                break

            # Store input state.
            # X layout (8-D): [x, y, heading, surge, sway, yaw_rate, rudder_action, thrust_action]
            if step > (5.0 / sim_setup.dt):
                input_states.append(np.concatenate([input_state.copy(), action.copy()], axis=0))

            # Apply controls with safety checks
            if has_rudders:
                # Rudder control (sign inversion is intentional)
                rudder_value = float(-1.0 * action[0])
                for rudder in rudders:
                    rudder.setControlValue(rudder_value)

            if has_propellers:
                # Propeller / thrust control
                propeller_value = float(action[1])
                for propeller in propellers:
                    propeller.engage()
                    propeller.setEngineLeverValue(propeller_value)

            # Step simulation
            _math_model.simulateSeconds(sim_setup.dt)

            # Update position
            pos = _ship_interface.getShipPosition()
            curr_ship_pos = np.array([pos.x, pos.y], dtype=np.float32)
            if step > (5.0 / sim_setup.dt):
                traj.append(curr_ship_pos.copy())

            # Retrieve new velocities
            current_vog = _ship_interface.getShipVelocityOverGround()
            current_heading = _ship_interface.getShipHeading()
            current_yaw_rate = _ship_interface.getShipYawRate()

            new_surge = current_vog.x
            new_sway = -current_vog.y
            new_yaw_rate = np.radians(current_yaw_rate)

            # Store output state.
            # Y layout (3-D): [delta_surge, delta_sway, delta_yaw_rate]
            # Body-frame velocity deltas only – world-frame position and heading
            # deltas are excluded because the surrogate is used to propagate
            # body-frame dynamics; position integration is handled externally.
            if step > (5.0 / sim_setup.dt):
                output_states.append(
                    np.array(
                        [
                            new_surge    - input_state[3],   # delta_surge
                            new_sway     - input_state[4],   # delta_sway
                            new_yaw_rate - input_state[5],   # delta_yaw_rate
                        ],
                        dtype=np.float64,
                    )
                )

            # Update state for next iteration (including new absolute position)
            input_state = np.array(
                [
                    pos.x,
                    pos.y,
                    env.normalize_angle(np.radians(current_heading)),
                    new_surge,
                    new_sway,
                    new_yaw_rate,
                ],
                dtype=np.float64,
            )

        # Store trajectory
        sim_setup.output.trajectory = np.array(traj, dtype=np.float32)

        # Final validation
        final_keel_clearance = _ship_interface.getKeelClearance()
        final_is_open_water = env._ship_in_open_water(curr_ship_pos)

        if not final_is_open_water or final_keel_clearance < 1.0:
            error_msg = "Ship not in open water!"
            sim_setup.output.status = "FAILED"
            sim_setup.output.info["error"] = error_msg
            return sim_setup

        # Store results if simulation succeeded
        sim_setup.output.x_input = input_states
        sim_setup.output.y_output = output_states

    except SimException as e:
        error_msg = f"Simulator exception: {e}"
        print(error_msg)
        _handle_exception(sim_setup, e, "SimException")

    except Exception as e:
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

    CSV column layout (11 columns):
        X (8): x, y, heading, surge, sway, yaw_rate, rudder_action, thrust_action
        Y (3): delta_surge, delta_sway, delta_yaw_rate

    Args:
        nbr_samples: Number of samples to generate
        horizon_seconds: Maximum simulation time
        dt: Time step for simulation
        workers: Number of worker threads
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

    # Create SimConfig
    sim_config = SimConfig()
    sim_config.lib_path = str(Path(simexe_path) / 'exe' / 'installed')
    sim_config.config_path = str(Path(simexe_path) / 'config' / 'simXdrive.config.xml')
    sim_config.timeout_seconds = TIMEOUT_SECONDS

    output_path = Path(output_file).resolve().parent / "_sim_scratch"

    # Clear output folder
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    exercise_path = Path(
        simexe_path) / 'database' / 'areas' / 'ScheldeSaeftinge_23_002' / 'invoer' / 'tra' / 'DDShip_scenario2_windnocurrent.tab'

    # Create environment and sample start positions
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
    spawn_counter = 0  # Tracks total process spawns for periodic cache clearing

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

    # Setup thread pool for parallel execution
    with ThreadPoolExecutor(max_workers=workers) as executor:
        execute_with_param = partial(run_in_process, sim_config=sim_config)

        future_to_test = {
            executor.submit(execute_with_param, setup): (i, setup)
            for i, setup in enumerate(setups)
        }

        for future in concurrent.futures.as_completed(future_to_test):
            spawn_counter += 1

            # Periodically wipe C:\cache to reclaim space consumed by crash dumps
            # from Python subprocesses that terminated unexpectedly.
            if spawn_counter % CACHE_CLEAR_INTERVAL == 0:
                print(f"[Cache cleanup] {spawn_counter} processes spawned — clearing {CACHE_FOLDER} …")
                clear_cache_folder(CACHE_FOLDER)

            try:
                sim_setup = future.result()
                test_item = future_to_test[future]
                index, _ = test_item

                if sim_setup.output.status == "PASSED":
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

    store_data = np.hstack([x_values, y_values])
    # CSV column layout (11 columns):
    #   X (8): x, y, heading, surge, sway, yaw_rate, rudder_action, thrust_action
    #   Y (3): delta_surge, delta_sway, delta_yaw_rate
    np.savetxt(output_file, store_data, delimiter=",", fmt="%.8f")
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
    print(f"X: {x_vals.shape}")  # [N, 8] -> (x, y, heading, surge, sway, yaw_rate, rudder, thrust)
    print(f"Y: {y_vals.shape}")  # [N, 3] -> (delta_surge, delta_sway, delta_yaw_rate)

    if successful_count > 0:
        np.save("ship_dynamics_inputs.npy", x_vals)
        np.save("ship_dynamics_outputs.npy", y_vals)
        print("Dataset saved to ship_dynamics_inputs.npy and ship_dynamics_outputs.npy")
