
import csv
import math

filename = "src/airsim_telemetry.csv"

steps = 0
total_dist_err = 0.0
total_head_err = 0.0
max_dist_err = 0.0
max_head_err = 0.0
total_distance = 0.0
last_ship_speed = 0.0

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
            
            # Approx distance traveled (assuming 0.1s step)
            total_distance += ship_spd * 0.1

    if steps > 0:
        mae_dist = total_dist_err / steps
        mae_head = total_head_err / steps
        
        # Accuracy Calculation
        # Distance Accuracy: 1 - Error / Total Distance
        # (Alternatively relative to ship length e.g. 5m? But total distance is better for tracking)
        dist_acc = 100.0 * (1.0 - (mae_dist / max(total_distance, 1.0)))
        
        # Heading Accuracy: 1 - Error / 180 (max possible error)
        # Or relative to "control loss" ~45 deg
        head_acc_technical = 100.0 * (1.0 - (mae_head / 180.0))
        head_acc_practical = 100.0 * (1.0 - (mae_head / 45.0)) # If error > 45, it's 0%
        
        print(f"--- Telemetry Analysis ({steps} steps) ---")
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
