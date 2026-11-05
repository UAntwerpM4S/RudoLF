"""
Offline analysis of AirSim bridge telemetry.

Reads airsim_telemetry.csv and prints mean/max distance and heading errors
and derived accuracy percentages.

Usage:
    python -m fw.common.airsim.tools.analyze_telemetry
"""

import os
import csv

from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[4]
_CONFIG_DIR = _SRC_DIR / "airsim" / "config"

# Candidate telemetry locations, searched in order
POSSIBLE_PATHS = [
    str(_CONFIG_DIR / "airsim_telemetry.csv"),
    "airsim_telemetry.csv",
    "src/fw/air_sim/airsim_telemetry.csv",
]


def find_telemetry_file() -> str | None:
    """
    Return the first existing path in POSSIBLE_PATHS, or None.
    """

    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    return None


def analyze() -> None:
    """
    Print distance/heading error statistics and accuracy percentages.

    Errors (missing file, bad rows) are printed, not raised.
    """

    filename = find_telemetry_file()
    if filename is None:
        print(f"Telemetry file not found in: {POSSIBLE_PATHS}")
        return

    steps = 0
    total_dist_err = 0.0
    total_head_err = 0.0
    max_dist_err = 0.0
    max_head_err = 0.0
    total_distance = 0.0

    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps += 1
                dist_err = float(row["dist_error"])
                head_err = float(row["heading_error_deg"])
                ship_spd = float(row["ship_speed"])

                total_dist_err += abs(dist_err)
                total_head_err += abs(head_err)

                max_dist_err = max(max_dist_err, abs(dist_err))
                max_head_err = max(max_head_err, abs(head_err))

                total_distance += ship_spd * 0.1

        if steps > 0:
            mae_dist = total_dist_err / steps
            mae_head = total_head_err / steps

            dist_acc = 100.0 * (1.0 - (mae_dist / max(total_distance, 1.0)))
            head_acc_technical = 100.0 * (1.0 - (mae_head / 180.0))
            head_acc_practical = 100.0 * (1.0 - (mae_head / 45.0))

            print(f"--- Telemetry Analysis ({steps} steps) ---")
            print(f"File: {filename}")
            print(f"Total Distance Traveled: {total_distance:.2f} meters")
            print(f"Mean Absolute Distance Error: {mae_dist:.4f} meters")
            print(f"Max Distance Error: {max_dist_err:.4f} meters")
            print(f"Positional Accuracy: {dist_acc:.4f}%")
            print("-" * 30)
            print(f"Mean Absolute Heading Error: {mae_head:.4f} degrees")
            print(f"Max Heading Error: {max_head_err:.4f} degrees")
            print(f"Heading Control Accuracy (vs 180°): {head_acc_technical:.4f}%")
            print(f"Heading Control Accuracy (vs 45°): {head_acc_practical:.4f}%")
        else:
            print("No telemetry data found.")

    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"Error analyzing telemetry: {e}")


if __name__ == "__main__":
    analyze()
