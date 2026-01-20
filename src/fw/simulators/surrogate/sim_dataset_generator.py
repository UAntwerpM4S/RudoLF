import numpy as np
import matplotlib.pyplot as plt

from py_sim_env import PySimEnv


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
        env.ship_pos = np.array([x0[n], y0[n]], dtype=np.float32)
        obs, _ = env.reset()

        rng = np.random.default_rng()
        env.state[-4:] = rng.uniform(low=[-np.pi, 0.0, -2.0, -0.5], high=[np.pi, 5.0, 2.0, 0.5])

        initial_state = env.state.copy()
        initial_pos = env.ship_pos.copy()

        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        duration = rng.uniform(low=[dt], high=[horizon_seconds])[0]
        steps_per_episode = int(duration / dt)

        # --- Start storing trajectory ---
        traj = [env.ship_pos.copy()]

        for _ in range(steps_per_episode):
            if env._ship_in_open_water(env.ship_pos):
                env._update_ship_dynamics(action)
                traj.append(env.ship_pos.copy())
            else:
                break

        traj = np.array(traj)  # shape (T, 2)
        trajectories.append(traj)

        if env._ship_in_open_water(env.ship_pos):
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
