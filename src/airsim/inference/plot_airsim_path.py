"""
Live path-reconstruction plotter for AirSim telemetry.

Reads `airsim_telemetry.csv` produced by the AirSim bridge, reconstructs
ship and vehicle paths via dead reckoning, and displays them in a
continuously updating matplotlib window.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import csv
import os

# Configuration
DT = 0.1  # Simulation time step
POSSIBLE_PATHS = [
    "airsim_telemetry.csv",
    "src/airsim/config/airsim_telemetry.csv",
    "src/fw/air_sim/airsim_telemetry.csv",
    "../src/airsim/config/airsim_telemetry.csv",
]


def find_telemetry_file():
    """
    Search for the telemetry CSV in a list of candidate locations.
    Returns:
        The first existing path from `POSSIBLE_PATHS`, or `None`
        if no file is found.
    """

    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    return None


def reconstruct_path(speeds, headings_deg, start_pos=(0, 0)):
    """
    Reconstruct a 2D path from speed and heading arrays via dead reckoning.

    Assumes headings are north-referenced (0 = North, 90 = East)::

        x_{t+1} = x_t + v_t * cos(theta_t) * dt   (North)
        y_{t+1} = y_t + v_t * sin(theta_t) * dt   (East)

    Args:
        speeds: Sequence of scalar speeds (m/s) at each time step.
        headings_deg: Sequence of headings in degrees
            (0 = North, 90 = East).
        start_pos: `(x, y)` starting position. Defaults to `(0, 0)`.

    Returns:
        A tuple `(x_list, y_list)` of reconstructed positions.
        Each list has `len(speeds) + 1` elements (start + one per step).
    """

    x, y = [start_pos[0]], [start_pos[1]]
    for v, h_deg in zip(speeds, headings_deg):
        h_rad = math.radians(h_deg)
        dx = v * math.cos(h_rad) * DT
        dy = v * math.sin(h_rad) * DT
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    return x, y


class LivePlotter:
    """
    Continuously updating matplotlib plot of reference-ship vs. AirSim paths.

    Reads `airsim_telemetry.csv` every second, reconstructs both paths
    with dead reckoning, aligns their initial headings, and re-draws the
    figure.

    Attributes:
        telemetry_file: Path to the CSV file, or `None` while waiting.
        is_running: Whether the animation loop is still active.
    """

    def __init__(self):
        """
        Initialize the figure, axes, line artists, and event handlers.
        """

        self.telemetry_file = find_telemetry_file()
        if not self.telemetry_file:
            print(f"Warning: airsim_telemetry.csv not found in {POSSIBLE_PATHS}. Waiting for file...")

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Initialize plots with empty data
        self.ship_line, = self.ax.plot([], [], label='Reference Ship', linewidth=2, linestyle='--', color='blue')
        self.car_line, = self.ax.plot([], [], label='AirSim Vehicle (Aligned Start)', linewidth=2, alpha=0.8, color='orange')

        self.ship_start = self.ax.scatter([], [], c='green', marker='o', label='Start')
        self.ship_end = self.ax.scatter([], [], c='blue', marker='x', label='Ship End')
        self.car_end = self.ax.scatter([], [], c='red', marker='x', label='Car End')

        self.ax.set_title("Live Path Reconstruction (Waiting for data...)")
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.axis('equal')

        self.ani = None
        self.is_running = True

        # Connect events
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        print("Live Plotting Started.")
        print("Press 'q' or close the window to stop and save the plot.")

    def update(self, frame):
        """
        Read the latest telemetry and refresh the plot.

        Called automatically by `FuncAnimation` each interval.

        Args:
            frame: Frame index supplied by the animation timer
                (unused).
        """

        if not self.is_running:
            return

        # Re-check for file if not found yet
        if not self.telemetry_file:
            self.telemetry_file = find_telemetry_file()
            if not self.telemetry_file:
                return  # Still waiting

        try:
            ship_speeds = []
            car_speeds = []
            ship_headings = []
            car_headings = []

            with open(self.telemetry_file, 'r') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return

                for row in reader:
                    try:
                        ship_speeds.append(float(row['ship_speed']))
                        car_speeds.append(float(row['car_speed']))
                        ship_headings.append(90.0 - float(row['ship_heading_deg']))
                        car_headings.append(float(row['car_yaw_deg']))
                    except ValueError:
                        continue

            if not ship_speeds:
                return

            # --- ALIGNMENT LOGIC ---
            n_avg = min(10, len(ship_headings))
            if n_avg > 0:
                start_ship_h = sum(ship_headings[:n_avg]) / n_avg
                start_car_h = sum(car_headings[:n_avg]) / n_avg
                heading_offset = start_ship_h - start_car_h
                car_headings_aligned = [h + heading_offset for h in car_headings]
                title = f"Live Path Reconstruction (Aligned)\nOffset Removed: {heading_offset:.1f} deg | Steps: {len(ship_speeds)}"
            else:
                car_headings_aligned = car_headings
                title = f"Live Path Reconstruction | Steps: {len(ship_speeds)}"

            self.ax.set_title(title)

            # Reconstruct paths with aligned headings
            ship_x, ship_y = reconstruct_path(ship_speeds, ship_headings)
            car_x, car_y = reconstruct_path(car_speeds, car_headings_aligned)

            # Update Plot Data
            self.ship_line.set_data(ship_y, ship_x)
            self.car_line.set_data(car_y, car_x)

            if ship_x:
                self.ship_start.set_offsets([[ship_y[0], ship_x[0]]])
                self.ship_end.set_offsets([[ship_y[-1], ship_x[-1]]])
                self.car_end.set_offsets([[car_y[-1], car_x[-1]]])

                self.ax.relim()
                self.ax.autoscale_view()
                self.ax.axis('equal')

        except Exception as e:
            print(f"Update error: {e}")

    def save_plot(self):
        """
        Save the current figure to `airsim_path_reconstruction.png`.
        """

        if self.telemetry_file:
            output_file = os.path.join(os.path.dirname(self.telemetry_file), "airsim_path_reconstruction.png")
            if not os.path.dirname(output_file):
                output_file = "airsim_path_reconstruction.png"

            self.fig.savefig(output_file)
            print(f"Plot saved to {os.path.abspath(output_file)}")

    def on_close(self, event):
        """
        Handle the matplotlib window-close event.

        Args:
            event: The matplotlib `CloseEvent`.
        """

        if self.is_running:
            print("Window closed. Saving plot...")
            self.save_plot()
            self.is_running = False

    def on_key(self, event):
        """
        Handle keyboard input; pressing *q* stops the plot.

        Args:
            event: The matplotlib `KeyEvent`.
        """

        if event.key == 'q':
            print("Stop requested. Closing...")
            plt.close(self.fig)

    def start(self):
        """
        Start the live animation loop (updates every 1 second).
        """

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1000, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    plotter = LivePlotter()
    plotter.start()
