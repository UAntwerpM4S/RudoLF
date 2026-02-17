
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Add src to path just in case
sys.path.append(os.getcwd())

try:
    import plot_airsim_path
except ImportError:
    print("Could not import plot_airsim_path. Please ensure it is in the current directory.")
    sys.exit(1)

def main():
    print("Initializing LivePlotter (Headless)...")
    plotter = plot_airsim_path.LivePlotter()
    
    # Force single update to process data
    print("Processing telemetry data...")
    plotter.update(0)
    
    # Save
    print("Saving plot...")
    plotter.save_plot()
    print("Done.")

if __name__ == "__main__":
    main()
