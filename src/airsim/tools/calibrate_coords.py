"""Coordinate calibration tool for Fossen <-> AirSim mapping.

Run this script while UE5 + CosysAirSim is running. It will:
1. Ask you to drive the vehicle to 2 known reference points
2. Compute the transform (scale, rotation, mirror, offset)
3. Save calibration as ``calibration.json``
4. Visually verify by drawing the Fossen polygon in UE5

Usage:
    python -m airsim.tools.calibrate_coords
"""

import json
import math
import time
import numpy as np
from pathlib import Path

try:
    from airsim.communication.coord_transform import CoordTransform
except ImportError:
    from communication.coord_transform import CoordTransform

_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _THIS_DIR.parent / "config"
_ENVIRONMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "fw" / "environments"


def _load_trajectory_endpoints() -> tuple:
    """Load the first and last waypoints from the trajectory CSV.

    Returns:
        A tuple ``(start, end)`` where each element is a NumPy array
        of shape ``(2,)`` containing ``[x_east, y_north]``.
    """
    csv_path = _ENVIRONMENTS_DIR / "trajectory_points_no_scale.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    start = data[0]
    end = data[-1]
    return start, end


def _detect_origin_drift(client, saved_config: dict) -> np.ndarray:
    """Detect and report AirSim coordinate-origin drift.

    Compares the vehicle's current spawn position against the position
    recorded at calibration time.  If the delta exceeds 0.5 m, the
    returned drift vector should be added to all AirSim coordinates.

    Args:
        client: An active ``cosysairsim.CarClient`` instance.
        saved_config: The ``calibration.json`` dictionary.  If it
            contains a ``"spawn_position"`` key the drift is computed;
            otherwise a zero vector is returned.

    Returns:
        A 2-element NumPy array ``[dx, dy]`` representing the drift.
    """
    if "spawn_position" not in saved_config:
        return np.array([0.0, 0.0])

    saved_spawn = np.array(saved_config["spawn_position"])
    pose = client.simGetVehiclePose()
    current_spawn = np.array([pose.position.x_val, pose.position.y_val])
    drift = current_spawn - saved_spawn

    mag = np.linalg.norm(drift)
    if mag > 0.5:
        print(f"\n  WARNING: AirSim origin drift detected: dx={drift[0]:.2f}, dy={drift[1]:.2f}  ({mag:.1f} m)")
        print("      -> Auto-correcting all coordinates.")
    else:
        print(f"\n  OK: No origin drift ({mag:.2f} m).")

    return drift


def _collect_reference_points(client, traj_start, traj_end, has_existing, choice):
    """Gather two Fossen ↔ AirSim reference-point pairs interactively.

    Depending on the user's ``choice`` the method either re-uses a
    saved calibration, drives to points in UE5, or accepts manual
    coordinates from stdin.

    Args:
        client: An active ``cosysairsim.CarClient`` instance.
        traj_start: Start of the reference trajectory (Fossen frame).
        traj_end: End of the reference trajectory (Fossen frame).
        has_existing: Whether a ``calibration.json`` already exists.
        choice: User-selected calibration mode (``"1"``–``"4"``).

    Returns:
        An 8-tuple ``(fossen_A, fossen_B, ue5_A, ue5_B,
        mirror_x, mirror_y, offset_x, offset_y)``.
    """
    if choice in ["1", "4"] and has_existing:
        cal_path = _CONFIG_DIR / "calibration.json"
        with open(cal_path, "r") as f:
            saved = json.load(f)

        drift = _detect_origin_drift(client, saved)

        fossen_A = np.array(saved["fossen_origin"])
        ue5_A = np.array(saved["airsim_origin"]) + drift
        mirror_x = saved.get("mirror_x", False)
        mirror_y = saved.get("mirror_y", False)

        if "fossen_endpoint" in saved and "airsim_endpoint" in saved:
            fossen_B = np.array(saved["fossen_endpoint"])
            ue5_B = np.array(saved["airsim_endpoint"]) + drift
        else:
            fossen_B = traj_end
            corrected = dict(saved)
            corrected["airsim_origin"] = ue5_A.tolist()
            coord = CoordTransform(corrected)
            ux, uy = coord.fossen_to_airsim_2d(fossen_B[0], fossen_B[1])
            ue5_B = np.array([ux, uy])

        print(f"\n  Re-using saved calibration from: {cal_path}")
        return fossen_A, fossen_B, ue5_A, ue5_B, mirror_x, mirror_y, 0.0, 0.0

    # Ask for Fossen reference points
    print()
    custom_fossen = input("Use trajectory start/end as Fossen points? [Y/n]: ").strip().lower()
    if custom_fossen == "n":
        ax = float(input("Fossen point A — x: "))
        ay = float(input("Fossen point A — y: "))
        bx = float(input("Fossen point B — x: "))
        by = float(input("Fossen point B — y: "))
        fossen_A = np.array([ax, ay])
        fossen_B = np.array([bx, by])
    else:
        fossen_A = traj_start
        fossen_B = traj_end

    # Get AirSim points
    if choice == "3":
        print()
        uax = float(input("AirSim point A — x: "))
        uay = float(input("AirSim point A — y: "))
        ubx = float(input("AirSim point B — x: "))
        uby = float(input("AirSim point B — y: "))
        ue5_A = np.array([uax, uay])
        ue5_B = np.array([ubx, uby])
    else:
        print()
        print("-" * 40)
        print("Drive/place the vehicle at the START of the route in UE5.")
        input("Press Enter when ready...")
        pose_A = client.simGetVehiclePose()
        ue5_A = np.array([pose_A.position.x_val, pose_A.position.y_val])
        print(f"  UE5 point A measured: x={ue5_A[0]:.2f}, y={ue5_A[1]:.2f}")

        print()
        print("Now drive/place the vehicle at the END of the route in UE5.")
        input("Press Enter when ready...")
        pose_B = client.simGetVehiclePose()
        ue5_B = np.array([pose_B.position.x_val, pose_B.position.y_val])
        print(f"  UE5 point B measured: x={ue5_B[0]:.2f}, y={ue5_B[1]:.2f}")

    # Mirror and offset
    print()
    flip_x = input("Is the UE5 X-axis flipped relative to Fossen? [y/N]: ").strip().lower()
    flip_y = input("Is the UE5 Y-axis flipped relative to Fossen? (Usually Yes for UE5) [Y/n]: ").strip().lower()
    mirror_x = (flip_x == "y")
    mirror_y = (flip_y != "n")

    print()
    print("Fine-tuning: you can add a manual offset in meters to shift the Fossen data in UE5.")
    offset_x = float(input("Manual X offset (e.g. 0.0): ") or 0.0)
    offset_y = float(input("Manual Y offset (e.g. 0.0): ") or 0.0)

    return fossen_A, fossen_B, ue5_A, ue5_B, mirror_x, mirror_y, offset_x, offset_y


def _compute_calibration(fossen_A, fossen_B, ue5_A, ue5_B,
                          mirror_x, mirror_y, offset_x, offset_y,
                          spawn_position) -> dict:
    """Compute the calibration dictionary from two reference-point pairs.

    Args:
        fossen_A: First reference point in Fossen coordinates.
        fossen_B: Second reference point in Fossen coordinates.
        ue5_A: First reference point in AirSim / UE5 coordinates.
        ue5_B: Second reference point in AirSim / UE5 coordinates.
        mirror_x: Whether the X-axis is flipped.
        mirror_y: Whether the Y-axis is flipped.
        offset_x: Manual X offset in metres.
        offset_y: Manual Y offset in metres.
        spawn_position: Current vehicle spawn position in UE5.

    Returns:
        A dictionary suitable for writing to ``calibration.json``.
    """
    fossen_dist = np.linalg.norm(fossen_B - fossen_A)
    ue5_dist = np.linalg.norm(ue5_B - ue5_A)
    scale = ue5_dist / fossen_dist

    dx_f = fossen_B[0] - fossen_A[0]
    dy_f = fossen_B[1] - fossen_A[1]
    if mirror_x:
        dx_f = -dx_f
    if mirror_y:
        dy_f = -dy_f

    fossen_angle = np.arctan2(dy_f, dx_f)
    ue5_angle = np.arctan2(ue5_B[1] - ue5_A[1], ue5_B[0] - ue5_A[0])
    rotation = ue5_angle - fossen_angle

    ue5_A_shifted = ue5_A + np.array([offset_x, offset_y])

    return {
        "scale": float(scale),
        "rotation": float(rotation),
        "mirror_x": mirror_x,
        "mirror_y": mirror_y,
        "fossen_origin": fossen_A.tolist(),
        "airsim_origin": ue5_A_shifted.tolist(),
        "fossen_endpoint": fossen_B.tolist(),
        "airsim_endpoint": ue5_B.tolist(),
        "spawn_position": spawn_position.tolist(),
    }


def _draw_visualization(client, coord, fossen_A, fossen_B, ue5_A, ue5_B) -> None:
    """Draw polygon, trajectory, and reference dots in UE5.

    Uses the AirSim ``simPlotLineStrip`` and ``simPlotPoints`` APIs to
    render persistent markers so the user can visually verify the
    calibration inside Unreal Engine.

    Args:
        client: An active ``cosysairsim.CarClient`` instance.
        coord: A ``CoordTransform`` configured with the current
            calibration.
        fossen_A: First Fossen reference point.
        fossen_B: Second Fossen reference point.
        ue5_A: First AirSim reference point.
        ue5_B: Second AirSim reference point.
    """
    import cosysairsim as airsim

    client.simFlushPersistentMarkers()
    time.sleep(0.3)

    check_A = coord.fossen_to_airsim_2d(*fossen_A)
    check_B = coord.fossen_to_airsim_2d(*fossen_B)
    err_A = np.linalg.norm(np.array(check_A) - ue5_A)
    err_B = np.linalg.norm(np.array(check_B) - ue5_B)
    print(f"\n  Verification A: ({check_A[0]:.2f}, {check_A[1]:.2f}) vs ({ue5_A[0]:.2f}, {ue5_A[1]:.2f}) — error: {err_A:.4f}")
    print(f"  Verification B: ({check_B[0]:.2f}, {check_B[1]:.2f}) vs ({ue5_B[0]:.2f}, {ue5_B[1]:.2f}) — error: {err_B:.4f}")

    # Draw polygon (green)
    print("\nDrawing Fossen polygon in UE5 for visual verification...")
    csv_path = _ENVIRONMENTS_DIR / "env_Sche_no_scale.csv"
    polygon = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    points_3d = []
    for p in polygon[::2]:
        ux, uy = coord.fossen_to_airsim_2d(p[0], p[1])
        points_3d.append(airsim.Vector3r(ux, uy, -2.0))

    chunk_size = 500
    for i in range(0, len(points_3d), chunk_size - 1):
        chunk = points_3d[i:i + chunk_size]
        if len(chunk) > 1:
            client.simPlotLineStrip(
                chunk,
                color_rgba=[0, 1, 0, 0.8],
                thickness=5.0,
                is_persistent=True,
            )

    # Draw trajectory (yellow)
    ship_pos = np.array([2245.0, -47.0])
    traj_path = _ENVIRONMENTS_DIR / "trajectory_points_no_scale.csv"
    traj_data = np.loadtxt(traj_path, delimiter=",", skiprows=1)

    distances = [np.linalg.norm(p - ship_pos) for p in traj_data]
    closest_idx = int(np.argmin(distances))
    traj_data = traj_data[closest_idx:]

    traj_points = []
    for p in traj_data:
        ux, uy = coord.fossen_to_airsim_2d(p[0], p[1])
        traj_points.append(airsim.Vector3r(ux, uy, -3.0))

    for i in range(0, len(traj_points), chunk_size - 1):
        chunk = traj_points[i:i + chunk_size]
        if len(chunk) > 1:
            client.simPlotLineStrip(
                chunk,
                color_rgba=[1, 1, 0, 1.0],
                thickness=8.0,
                is_persistent=True,
            )

    # Draw start (blue) and target (red) dots
    start_ux, start_uy = coord.fossen_to_airsim_2d(ship_pos[0], ship_pos[1])
    target_ux, target_uy = coord.fossen_to_airsim_2d(*fossen_B)

    client.simPlotPoints(
        [airsim.Vector3r(start_ux, start_uy, -3.0)],
        color_rgba=[0, 0, 1, 1],
        size=30.0,
        is_persistent=True,
    )
    client.simPlotPoints(
        [airsim.Vector3r(target_ux, target_uy, -3.0)],
        color_rgba=[1, 0, 0, 1],
        size=30.0,
        is_persistent=True,
    )

    print()
    print("  Green line  = Fossen polygon (should overlay kades)")
    print("  Yellow line = Trajectory (ship route)")
    print("  Blue dot    = Start position")
    print("  Red dot     = End position")
    print()


def load_saved_calibration_reference(
    client,
    calibration_path: Path | None = None,
    draw_reference: bool = True,
) -> tuple[CoordTransform, dict]:
    """Load saved calibration, apply drift correction, and optionally redraw it.

    Args:
        client: Active ``cosysairsim.CarClient`` instance.
        calibration_path: Optional override for ``calibration.json``.
        draw_reference: Whether to redraw the saved reference overlay in UE5.

    Returns:
        A tuple ``(coord, config)`` containing the corrected transform and
        calibration dictionary after any drift adjustment.
    """
    cal_path = Path(calibration_path) if calibration_path else _CONFIG_DIR / "calibration.json"
    with open(cal_path, "r") as f:
        saved = json.load(f)

    drift = _detect_origin_drift(client, saved)
    config = dict(saved)

    ue5_A = np.array(saved["airsim_origin"], dtype=np.float64) + drift
    fossen_A = np.array(saved["fossen_origin"], dtype=np.float64)
    config["airsim_origin"] = ue5_A.tolist()

    _, traj_end = _load_trajectory_endpoints()
    fossen_B = np.array(saved.get("fossen_endpoint", traj_end), dtype=np.float64)
    if "airsim_endpoint" in saved:
        ue5_B = np.array(saved["airsim_endpoint"], dtype=np.float64) + drift
    else:
        preview = CoordTransform(config)
        ux, uy = preview.fossen_to_airsim_2d(fossen_B[0], fossen_B[1])
        ue5_B = np.array([ux, uy], dtype=np.float64)
    config["airsim_endpoint"] = ue5_B.tolist()

    coord = CoordTransform(config)
    if draw_reference:
        _draw_visualization(client, coord, fossen_A, fossen_B, ue5_A, ue5_B)

    return coord, config


def run_calibration() -> None:
    """Run the interactive Fossen ↔ AirSim calibration wizard.

    Connects to AirSim, collects two reference-point pairs (either
    from an existing calibration, interactive driving, or manual
    entry), computes the transform, writes ``calibration.json``,
    and draws a visual verification overlay in UE5.
    """
    import cosysairsim as airsim

    client = airsim.CarClient()
    client.confirmConnection()

    spawn_pose = client.simGetVehiclePose()
    spawn_position = np.array([spawn_pose.position.x_val, spawn_pose.position.y_val])

    traj_start, traj_end = _load_trajectory_endpoints()

    print("=" * 60)
    print("  FOSSEN <-> AIRSIM COORDINATE CALIBRATION")
    print("=" * 60)
    print()
    print(f"Trajectory START (Fossen): ({traj_start[0]:.1f}, {traj_start[1]:.1f})")
    print(f"Trajectory END   (Fossen): ({traj_end[0]:.1f}, {traj_end[1]:.1f})")
    print()

    existing_cal = _CONFIG_DIR / "calibration.json"
    has_existing = existing_cal.exists()

    print("How do you want to provide AirSim coordinates?")
    print()
    if has_existing:
        print("  [1] Re-use saved calibration (just redraw visualization)")
        print("  [4] Live tune existing calibration offsets (Interactive)")
    print("  [2] Drive to points interactively in UE5")
    print("  [3] Enter AirSim coordinates manually")
    print()

    choice = input(f"Choose [{'1/4/' if has_existing else ''}2/3]: ").strip()

    fossen_A, fossen_B, ue5_A, ue5_B, mirror_x, mirror_y, offset_x, offset_y = (
        _collect_reference_points(client, traj_start, traj_end, has_existing, choice)
    )

    if choice in ["1", "4"] and has_existing:
        with open(existing_cal, "r") as f:
            config = json.load(f)
        drift = ue5_A - np.array(config["airsim_origin"])
        if np.linalg.norm(drift) > 0.01:
            config["airsim_origin"] = ue5_A.tolist()
            if "airsim_endpoint" in config:
                config["airsim_endpoint"] = ue5_B.tolist()
            config["spawn_position"] = spawn_position.tolist()
    else:
        config = _compute_calibration(
            fossen_A, fossen_B, ue5_A, ue5_B,
            mirror_x, mirror_y, offset_x, offset_y,
            spawn_position,
        )

    print()
    print("=" * 40)
    print(f"  Scale:    {config['scale']:.6f}")
    print(f"  Rotation: {math.degrees(config['rotation']):.2f}°")
    print(f"  Fossen origin: {config['fossen_origin']}")
    print(f"  AirSim origin: {config['airsim_origin']}")
    print("=" * 40)

    while True:
        out_path = _CONFIG_DIR / "calibration.json"
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n  Calibration saved to: {out_path}")

        coord = CoordTransform(config)
        _draw_visualization(client, coord, fossen_A, fossen_B, ue5_A, ue5_B)

        if choice != "4":
            print("To clear markers: client.simFlushPersistentMarkers()")
            print("Done!")
            break

        print("=" * 40)
        print("  LIVE TUNING MODE")
        print("  Shift the polygon in UE5 by entering an X and Y offset in meters.")
        print("  Leave blank and press Enter to save and exit.")
        print("=" * 40)
        offset_val = input("Enter shift as 'X, Y' (e.g. '10.5, -5.0') or leave empty: ").strip()

        if not offset_val:
            print("\nTo clear markers: client.simFlushPersistentMarkers()")
            print("Done!")
            break

        try:
            parts = offset_val.split(",")
            dx = float(parts[0].strip())
            dy = float(parts[1].strip())
            config["airsim_origin"][0] += dx
            config["airsim_origin"][1] += dy
            print(f"\nApplied shift: +{dx} X, +{dy} Y. Redrawing...")
        except Exception:
            print("\nInvalid format. Please enter 'X, Y' like '10.5, -3.2'")


if __name__ == "__main__":
    run_calibration()
