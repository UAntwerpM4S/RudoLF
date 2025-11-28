import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

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

library_path = os.path.join(simexe_path, 'exe/installed')
config_path = os.path.join(simexe_path, 'config/simXdrive.config.xml')
exercise_path = os.path.join(simexe_path, 'database/areas/ScheldeSaeftinge_23_002/invoer/tra/DDShip_scenario2_windnocurrent.tab')
output_path = os.path.join('C:/Users', user_login, 'Desktop', 'testExercise', 'results')
sys.path.append(library_path)

logfile = os.path.join(output_path, 'math_model.log')
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(levelname)s: %(message)s')

from py_sim_env import PySimEnv
from sim import Config, Exercise, MathModel, SimException


def plot_xy_points_and_trajectories(env, xy, trajectories):
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
                color="green",       # ← VERY DIFFERENT COLOR
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

def collect_supervised_dataset(
    N_samples=5000,
    horizon_seconds=3.0,
    dt=0.1,
    output_file="ship_dynamics_dataset.csv"
):
    env = PySimEnv(render_mode=None, time_step=dt, wind=False, current=False)
    env.time_step = dt

    X_inputs = []
    Y_outputs = []

    xy = env.sample_point_in_river(N_samples)
    x0 = xy[:, 0]
    y0 = xy[:, 1]

    N_samples = len(xy)
    trajectories = []
    counter = 0

    for n in range(N_samples):
        _math_model = None
        _ship_interface = None

        try:
            # Configure the math model
            config = Config(config_path)
            config.SetOutputDir(output_path)
            config.SetRewindEnabled(True)
            config.SetRewindConfig(10000,
                                   5)  # First parameter is the maximum snapshot window in [s], second parameter is the snapshot frequency in [s]
            config.SetMathModelFrequency(10.0)  # Math model frequency in Hz
            config.ClearConnections()

            exercise = Exercise(exercise_path, config)
            ship_config = exercise.getShipConfig()

            start_pos = np.array([x0[n], y0[n]], dtype=np.float32)

            x_pos = float(start_pos[0])
            y_pos = float(start_pos[1])
            ship_config.setInitialPosition(x_pos, y_pos)

            # Initialize the math model and enable the bridge
            _math_model = MathModel()
            if _math_model.Initialize(config, exercise):
                print(f"Math model initialized!  --  starting simulation at {start_pos}")
                _math_model.enableBridge()

                # Get the ship interface
                _ship_interface = _math_model.getShipInterface(0)

                env.initial_ship_pos = start_pos
                obs, _ = env.reset()

                rng = np.random.default_rng()
                env.state[-4:] = rng.uniform(low=[-np.pi, 0.0, -2.0, -0.5], high=[np.pi, 5.0, 2.0, 0.5])

                initial_state = env.state.copy()
                initial_pos = env.ship_pos.copy()
                new_ship_pos = env.ship_pos.copy()

                action = rng.uniform(-1, 1, size=2).astype(np.float32)

                duration = rng.uniform(low=[dt], high=[horizon_seconds])[0]
                steps_per_episode = int(duration / dt)

                # --- Start storing trajectory ---
                traj = [env.ship_pos.copy()]

                for _ in range(steps_per_episode):
                    if env._ship_in_open_water(new_ship_pos) and _ship_interface.getKeelClearance() >= 5.0:
                        # Update rudder controls based on the turning action.
                        for rudder in _ship_interface.getRudderControls():
                            rudder.setControlValue(float(
                                -1.0 * action[0]))  # this is to compensate for opposite behaviour of the Python environment
                            # in Python: -1 is turn right ; 1 is turn left
                            # FH sim: -1 is turn left ; 1 is turn right
                        # Update propeller controls based on the thrust action.
                        for propeller in _ship_interface.getPropellerControls():
                            propeller.setEngineLeverValue(float(action[1]))

                        # Simulate the ship's dynamics for a fixed period.
                        _math_model.simulateSeconds(dt)

                        # Retrieve the updated ship position from the ship interface.
                        new_ship_pos = (_ship_interface.getShipPosition().x, _ship_interface.getShipPosition().y)
                        traj.append(new_ship_pos)
                    else:
                        break

                traj = np.array(traj)  # shape (T, 2)
                trajectories.append(traj)

                if env._ship_in_open_water(new_ship_pos) and _ship_interface.getKeelClearance() >= 5.0:
                    # Calculate delta X and delta Y
                    delta_x = env.state[0] - initial_pos[0]
                    delta_y = env.state[1] - initial_pos[1]

                    # Create modified final state with delta X and delta Y instead of absolute positions
                    # Assuming state structure: [x, y, heading, velocity, angular_velocity, ...]
                    # Replace the first two elements (x, y) with delta_x, delta_y
                    modified_final_state = env.state.copy()
                    modified_final_state[0] = delta_x  # Replace x with delta_x
                    modified_final_state[1] = delta_y  # Replace y with delta_y

                    counter += 1
                    X_inputs.append(np.concatenate([initial_state, action], axis=0))
                    Y_outputs.append(modified_final_state)
            else:
                print("MathModel initialization failed!")
        except SimException as e:
            print('Simulation failed: ' + e.ToString())
        except Exception as exc:
            print(f'Simulation failed: {exc}')
        finally:
            if not _math_model is None:
                _math_model.Terminate()
                _math_model.Dispose()


    X = np.vstack(X_inputs).astype(np.float32)
    Y = np.vstack(Y_outputs).astype(np.float32)

    store_data = np.hstack([X_inputs, Y_outputs])  # shape: (N_samples, 8 + 6 = 14)
    np.savetxt(output_file, store_data, delimiter=",", fmt="%.6f")

    plot_xy_points_and_trajectories(env, xy, trajectories)

    return X, Y, counter


# Example: Generate dataset
if __name__ == "__main__":
    X, Y, _ = collect_supervised_dataset(
        N_samples=10000,
        horizon_seconds=5.0,
        dt=0.1,
    )

    print("Dataset shapes:")
    print("X:", X.shape)  # [N, 8] → (state(6) + action(2))
    print("Y:", Y.shape)  # [N, 6] → final state
    np.save("ship_dynamics_inputs.npy", X)
    np.save("ship_dynamics_outputs.npy", Y)
