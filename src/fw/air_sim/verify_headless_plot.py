"""Headless verification script for the AirSim path-reconstruction plot.

Imports ``plot_airsim_path``, forces the ``Agg`` (non-interactive) backend,
runs a single update cycle, and saves the resulting figure to disk.
This is useful for CI pipelines or servers without a display.
"""

import matplotlib
matplotlib.use('Agg')
import sys
import os

# Add src to path just in case
sys.path.append(os.getcwd())

try:
    import fw.air_sim.plot_airsim_path as airsim_plotter
except ImportError:
    print("Could not import plot_airsim_path. Please ensure it is in the current directory.")
    airsim_plotter = None
    sys.exit(1)

def main():
    """Run a single headless update of the live plotter and save the figure."""
    print("Initializing LivePlotter (Headless)...")

    if airsim_plotter is not None:
        plotter = airsim_plotter.LivePlotter()

        # Force single update to process data
        print("Processing telemetry data...")
        plotter.update(0)

        # Save
        print("Saving plot...")
        plotter.save_plot()
        print("Done.")

if __name__ == "__main__":
    main()
