"""
Headless check of the AirSim path-reconstruction plot: forces the Agg backend,
runs one LivePlotter update and saves the figure (for CI or display-less servers).
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')

# Allow running from the src directory.
sys.path.append(os.getcwd())

try:
    import airsim.inference.plot_airsim_path as airsim_plotter
except ImportError:
    print("Could not import plot_airsim_path. Please ensure the airsim package is on the path.")
    airsim_plotter = None
    sys.exit(1)


def main():
    """
    Run a single headless update of the live plotter and save the figure.
    """

    print("Initializing LivePlotter (Headless)...")

    if airsim_plotter is not None:
        plotter = airsim_plotter.LivePlotter()

        print("Processing telemetry data...")
        plotter.update(0)

        print("Saving plot...")
        plotter.save_plot()
        print("Done.")


if __name__ == "__main__":
    main()
