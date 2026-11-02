"""
Interactive calibration of the Framework <-> AirSim coordinate mapping.

Run while UE5 + CosysAirSim is running. It collects two reference points,
computes the transform (scale, rotation, mirror, offset), saves it to
calibration.json and draws the Framework polygons in UE5 for visual checking.

Usage:
    python -m fw.common.airsim.tools.calibrate_coords
"""

import json
import math
import sys
import time
import numpy as np
from pathlib import Path
from typing import cast

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[4]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from airsim.communication.coord_transform import CoordTransform

_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _SRC_DIR / "airsim" / "config"
_ENVIRONMENTS_DIR = _SRC_DIR / "fw" / "environments"
_PLOT_CHUNK_SIZE = 500   # points per simPlotLineStrip call (chunks overlap by one point)


def _load_trajectory_endpoints() -> tuple:
    """
    First and last trajectory waypoints, each a (2,) array [x_east, y_north].
    """

    csv_path = _ENVIRONMENTS_DIR / "trajectory_points_no_scale.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    start = data[0]
    end = data[-1]
    return start, end


def _detect_origin_drift(client, saved_config: dict) -> np.ndarray:
    """
    Detect and report AirSim origin drift: the current spawn position minus the
    one stored at calibration time ("spawn_position"; zero if absent). A drift
    above 0.5 m should be added to all AirSim coordinates.

    Args:
        client: Active cosysairsim.CarClient.
        saved_config: The calibration.json dictionary.

    Returns:
        Drift [dx, dy] as a NumPy array.
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
    """
    Gather two Framework <-> AirSim reference-point pairs: reuse a saved
    calibration, drive to points in UE5, or enter coordinates manually.

    Args:
        client: Active cosysairsim.CarClient.
        traj_start: Start of the reference trajectory (Framework frame).
        traj_end: End of the reference trajectory (Framework frame).
        has_existing: Whether calibration.json already exists.
        choice: User-selected calibration mode ("1"-"4").

    Returns:
        (framework_a, framework_b, ue5_a, ue5_b, mirror_x, mirror_y, offset_x, offset_y).
    """

    if choice in ["1", "4", "5"] and has_existing:
        cal_path = _CONFIG_DIR / "calibration.json"
        with open(cal_path, "r") as f:
            saved = json.load(f)

        drift = _detect_origin_drift(client, saved)

        framework_a = np.array(saved["framework_origin"])
        ue5_a = np.array(saved["airsim_origin"]) + drift
        mirror_x = saved.get("mirror_x", False)
        mirror_y = saved.get("mirror_y", False)

        if "framework_endpoint" in saved and "airsim_endpoint" in saved:
            framework_b = np.array(saved["framework_endpoint"])
            ue5_b = np.array(saved["airsim_endpoint"]) + drift
        else:
            framework_b = traj_end
            corrected = dict(saved)
            corrected["airsim_origin"] = ue5_a.tolist()
            coord = CoordTransform(corrected)
            ux, uy = coord.framework_to_airsim_2d(framework_b[0], framework_b[1])
            ue5_b = np.array([ux, uy])

        print(f"\n  Re-using saved calibration from: {cal_path}")
        return framework_a, framework_b, ue5_a, ue5_b, mirror_x, mirror_y, 0.0, 0.0

    # Ask for Framework reference points
    print()
    custom_framework = input("Use trajectory start/end as Framework points? [Y/n]: ").strip().lower()
    if custom_framework == "n":
        ax = float(input("Framework point A - x: "))
        ay = float(input("Framework point A - y: "))
        bx = float(input("Framework point B - x: "))
        by = float(input("Framework point B - y: "))
        framework_a = np.array([ax, ay])
        framework_b = np.array([bx, by])
    else:
        framework_a = traj_start
        framework_b = traj_end

    # Get AirSim points
    if choice == "3":
        print()
        uax = float(input("AirSim point A - x: "))
        uay = float(input("AirSim point A - y: "))
        ubx = float(input("AirSim point B - x: "))
        uby = float(input("AirSim point B - y: "))
        ue5_a = np.array([uax, uay])
        ue5_b = np.array([ubx, uby])
    else:
        print()
        print("-" * 40)
        print("Drive/place the vehicle at the START of the route in UE5.")
        input("Press Enter when ready...")
        pose_a = client.simGetVehiclePose()
        ue5_a = np.array([pose_a.position.x_val, pose_a.position.y_val])
        print(f"  UE5 point A measured: x={ue5_a[0]:.2f}, y={ue5_a[1]:.2f}")

        print()
        print("Now drive/place the vehicle at the END of the route in UE5.")
        input("Press Enter when ready...")
        pose_b = client.simGetVehiclePose()
        ue5_b = np.array([pose_b.position.x_val, pose_b.position.y_val])
        print(f"  UE5 point B measured: x={ue5_b[0]:.2f}, y={ue5_b[1]:.2f}")

    # Mirror and offset
    print()
    flip_x = input("Is the UE5 X-axis flipped relative to Framework? [y/N]: ").strip().lower()
    flip_y = input("Is the UE5 Y-axis flipped relative to Framework? (Usually Yes for UE5) [Y/n]: ").strip().lower()
    mirror_x = (flip_x == "y")
    mirror_y = (flip_y != "n")

    print()
    print("Fine-tuning: you can add a manual offset in meters to shift the Framework data in UE5.")
    offset_x = float(input("Manual X offset (e.g. 0.0): ") or 0.0)
    offset_y = float(input("Manual Y offset (e.g. 0.0): ") or 0.0)

    return framework_a, framework_b, ue5_a, ue5_b, mirror_x, mirror_y, offset_x, offset_y


def _compute_calibration(framework_a, framework_b, ue5_a, ue5_b,
                          mirror_x, mirror_y, offset_x, offset_y,
                          spawn_position) -> dict:
    """
    Compute the calibration dictionary from two reference-point pairs.

    Args:
        framework_a: First reference point in Framework coordinates.
        framework_b: Second reference point in Framework coordinates.
        ue5_a: First reference point in AirSim / UE5 coordinates.
        ue5_b: Second reference point in AirSim / UE5 coordinates.
        mirror_x: Whether the X-axis is flipped.
        mirror_y: Whether the Y-axis is flipped.
        offset_x: Manual X offset in metres.
        offset_y: Manual Y offset in metres.
        spawn_position: Current vehicle spawn position in UE5.

    Returns:
        Dictionary to write to calibration.json.
    """

    framework_dist = np.linalg.norm(framework_b - framework_a)
    ue5_dist = np.linalg.norm(ue5_b - ue5_a)
    scale = ue5_dist / framework_dist

    dx_f = framework_b[0] - framework_a[0]
    dy_f = framework_b[1] - framework_a[1]
    if mirror_x:
        dx_f = -dx_f
    if mirror_y:
        dy_f = -dy_f

    framework_angle = np.arctan2(dy_f, dx_f)
    ue5_angle = np.arctan2(ue5_b[1] - ue5_a[1], ue5_b[0] - ue5_a[0])
    rotation = ue5_angle - framework_angle

    ue5_a_shifted = ue5_a + np.array([offset_x, offset_y])

    return {
        "scale": float(scale),
        "rotation": float(rotation),
        "mirror_x": mirror_x,
        "mirror_y": mirror_y,
        "framework_origin": framework_a.tolist(),
        "airsim_origin": ue5_a_shifted.tolist(),
        "framework_endpoint": framework_b.tolist(),
        "airsim_endpoint": ue5_b.tolist(),
        "spawn_position": spawn_position.tolist(),
    }


def _draw_visualization(
    client,
    coord,
    framework_a,
    framework_b,
    ue5_a,
    ue5_b,
    ridge_overlay_offset: tuple[float, float] = (0.0, 0.0),
    ridge_overlay_rot_z_deg: float = 0.0,
    ship_pos: tuple[float, float] | None = None,
    target_pos: tuple[float, float] | None = None,
    duration: float = 7200.0,
) -> None:
    """
    Draw polygons, trajectory and reference dots in UE5 as persistent markers
    (simPlotLineStrip / simPlotPoints) to verify the calibration visually.

    Args:
        client: Active cosysairsim.CarClient.
        coord: CoordTransform with the current calibration.
        framework_a: First Framework reference point.
        framework_b: Second Framework reference point.
        ue5_a: First AirSim reference point.
        ue5_b: Second AirSim reference point.
        ridge_overlay_offset: (dx, dy) shift [m] applied to the ridge overlay only.
        ridge_overlay_rot_z_deg: Z rotation [deg] of the ridge overlay only, about ue5_a.
        ship_pos: Framework start position (blue marker, trajectory slicing); default (2245, -47).
        target_pos: Framework target position (red marker); default framework_b.
        duration: Marker lifetime [s].
    """

    import cosysairsim as airsim

    # Draw overlays high above the water so they remain visible in top-down cameras.
    overlay_z = -110.0
    ridge_dx, ridge_dy = ridge_overlay_offset
    ridge_rot_z_rad = math.radians(ridge_overlay_rot_z_deg)
    cos_rz = math.cos(ridge_rot_z_rad)
    sin_rz = math.sin(ridge_rot_z_rad)
    ridge_pivot_x = float(ue5_a[0])
    ridge_pivot_y = float(ue5_a[1])

    client.simFlushPersistentMarkers()
    time.sleep(0.3)

    check_a = coord.framework_to_airsim_2d(*framework_a)
    check_b = coord.framework_to_airsim_2d(*framework_b)
    err_a = np.linalg.norm(np.array(check_a) - ue5_a)
    err_b = np.linalg.norm(np.array(check_b) - ue5_b)
    print(f"\n  Verification A: ({check_a[0]:.2f}, {check_a[1]:.2f}) vs ({ue5_a[0]:.2f}, {ue5_a[1]:.2f}) - error: {err_a:.4f}")
    print(f"  Verification B: ({check_b[0]:.2f}, {check_b[1]:.2f}) vs ({ue5_b[0]:.2f}, {ue5_b[1]:.2f}) - error: {err_b:.4f}")

    # Ridges: river banks (overall map) in green, navigation channel in red.
    print("\nDrawing Framework polygons in UE5 for visual verification...")

    def _plot_polygon_ridges(csv_name: str, color_rgba: list[float], thickness: float, z: float = overlay_z) -> None:
        """
        Render a polygon CSV as persistent line-strip chunks in UE5.
        """

        csv_path = _ENVIRONMENTS_DIR / csv_name
        polygon = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        points_3d = []
        for p in polygon[::2]:
            rx, ry = coord.framework_to_airsim_2d(p[0], p[1])
            rx += ridge_dx
            ry += ridge_dy
            # Optional overlay-only Z-axis rotation for ridge finetuning (top-down orientation).
            dx_rel = rx - ridge_pivot_x
            dy_rel = ry - ridge_pivot_y
            ux_rot = ridge_pivot_x + (cos_rz * dx_rel - sin_rz * dy_rel)
            uy_rot = ridge_pivot_y + (sin_rz * dx_rel + cos_rz * dy_rel)
            points_3d.append(airsim.Vector3r(ux_rot, uy_rot, z))

        for start in range(0, len(points_3d), _PLOT_CHUNK_SIZE - 1):
            ridge_chunk = points_3d[start:start + _PLOT_CHUNK_SIZE]
            if len(ridge_chunk) > 1:
                client.simPlotLineStrip(
                    ridge_chunk,
                    color_rgba=color_rgba,
                    thickness=thickness,
                    duration=duration,
                    is_persistent=True,
                )

    _plot_polygon_ridges("env_Sche_no_scale.csv", color_rgba=[0.75, 1.0, 0.75, 1.0], thickness=56.0)
    _plot_polygon_ridges("env_Sche_250cm_no_scale.csv", color_rgba=[1.0, 0.3, 0.3, 1.0], thickness=56.0)

    # Draw trajectory (yellow)
    route_start = np.array(
        ship_pos if ship_pos is not None else [2245.0, -47.0],
        dtype=np.float64,
    )
    traj_path = _ENVIRONMENTS_DIR / "trajectory_points_no_scale.csv"
    traj_data = np.loadtxt(traj_path, delimiter=",", skiprows=1)

    distances = [np.linalg.norm(p - route_start) for p in traj_data]
    closest_idx = int(np.argmin(distances))
    traj_data = traj_data[closest_idx:]

    traj_points = []
    for p in traj_data:
        ux, uy = coord.framework_to_airsim_2d(p[0], p[1])
        traj_points.append(airsim.Vector3r(ux, uy, overlay_z))

    for i in range(0, len(traj_points), _PLOT_CHUNK_SIZE - 1):
        chunk = traj_points[i:i + _PLOT_CHUNK_SIZE]
        if len(chunk) > 1:
            client.simPlotLineStrip(
                chunk,
                color_rgba=[1.0, 1.0, 0.55, 1.0],
                thickness=56.0,
                duration=duration,
                is_persistent=True,
            )

    # Draw start (blue) and target (red) dots
    marker_start = route_start
    marker_target = np.array(
        target_pos if target_pos is not None else framework_b,
        dtype=np.float64,
    )
    start_ux, start_uy = coord.framework_to_airsim_2d(marker_start[0], marker_start[1])
    target_ux, target_uy = coord.framework_to_airsim_2d(marker_target[0], marker_target[1])

    client.simPlotPoints(
        [airsim.Vector3r(start_ux, start_uy, -3.0)],
        color_rgba=[0, 0, 1, 1],
        size=30.0,
        duration=duration,
        is_persistent=True,
    )
    client.simPlotPoints(
        [airsim.Vector3r(target_ux, target_uy, -3.0)],
        color_rgba=[1, 0, 0, 1],
        size=30.0,
        duration=duration,
        is_persistent=True,
    )

    print()
    print("  Green line  = River ridges / overall polygon (should overlay kades)")
    print("  Red line    = Navigation channel / waterway ridges")
    print("  Yellow line = Trajectory (ship route)")
    print("  Blue dot    = Start position")
    print("  Red dot     = End position")
    print(f"  Overlay Z   = {abs(overlay_z):.0f} m above water for top-down visibility")
    print(f"  Ridge shift = dx={ridge_dx:.2f} m, dy={ridge_dy:.2f} m (overlay only)")
    print(f"  Ridge rotZ  = {ridge_overlay_rot_z_deg:.2f} deg (overlay only)")
    print()


def load_saved_calibration_reference(
    client,
    calibration_path: Path | None = None,
    draw_reference: bool = True,
    ship_pos: tuple[float, float] | None = None,
    target_pos: tuple[float, float] | None = None,
    duration: float = 7200.0,
) -> tuple[CoordTransform, dict]:
    """
    Load saved calibration, apply drift correction, and optionally redraw it.

    Args:
        client: Active cosysairsim.CarClient.
        calibration_path: Optional override for calibration.json.
        draw_reference: Whether to redraw the saved reference overlay in UE5.
        ship_pos: Optional Framework start position used for the blue marker and
            trajectory slicing.
        target_pos: Optional Framework target position used for the red marker.
        duration: Marker lifetime [s] when draw_reference is set.

    Returns:
        (coord, config): drift-corrected transform and calibration dictionary.
    """

    cal_path = Path(calibration_path) if calibration_path else _CONFIG_DIR / "calibration.json"
    with open(cal_path, "r") as f:
        saved = json.load(f)

    drift = _detect_origin_drift(client, saved)
    config = dict(saved)

    ue5_a = np.array(saved["airsim_origin"], dtype=np.float64) + drift
    framework_a = np.array(saved["framework_origin"], dtype=np.float64)
    config["airsim_origin"] = ue5_a.tolist()

    _, traj_end = _load_trajectory_endpoints()
    framework_b = np.array(saved.get("framework_endpoint", traj_end), dtype=np.float64)
    if "airsim_endpoint" in saved:
        ue5_b = np.array(saved["airsim_endpoint"], dtype=np.float64) + drift
    else:
        preview = CoordTransform(config)
        ux, uy = preview.framework_to_airsim_2d(framework_b[0], framework_b[1])
        ue5_b = np.array([ux, uy], dtype=np.float64)
    config["airsim_endpoint"] = ue5_b.tolist()

    coord = CoordTransform(config)
    if draw_reference:
        ridge_overlay_offset = cast(tuple[float, float], tuple(config.get("ridge_overlay_offset", [0.0, 0.0])))
        ridge_overlay_rot_z_deg = float(config.get("ridge_overlay_rot_z_deg", 0.0))
        _draw_visualization(
            client,
            coord,
            framework_a,
            framework_b,
            ue5_a,
            ue5_b,
            ridge_overlay_offset=ridge_overlay_offset,
            ridge_overlay_rot_z_deg=ridge_overlay_rot_z_deg,
            ship_pos=ship_pos,
            target_pos=target_pos,
            duration=duration,
        )

    return coord, config


def run_calibration() -> None:
    """
    Interactive calibration wizard: connect to AirSim, collect two reference
    pairs, compute the transform, write calibration.json and draw the overlay.
    """

    import cosysairsim as airsim

    client = airsim.CarClient()
    client.confirmConnection()

    spawn_pose = client.simGetVehiclePose()
    spawn_position = np.array([spawn_pose.position.x_val, spawn_pose.position.y_val])

    traj_start, traj_end = _load_trajectory_endpoints()

    print("=" * 60)
    print("  FRAMEWORK <-> AIRSIM COORDINATE CALIBRATION")
    print("=" * 60)
    print()
    print(f"Trajectory START (Framework): ({traj_start[0]:.1f}, {traj_start[1]:.1f})")
    print(f"Trajectory END   (Framework): ({traj_end[0]:.1f}, {traj_end[1]:.1f})")
    print()

    existing_cal = _CONFIG_DIR / "calibration.json"
    has_existing = existing_cal.exists()

    print("How do you want to provide AirSim coordinates?")
    print()
    if has_existing:
        print("  [1] Re-use saved calibration (just redraw visualization)")
        print("  [4] Live tune existing calibration offsets (Interactive)")
        print("  [5] Live tune river/waterway ridge overlay only (keeps ideal path fixed)")
    print("  [2] Drive to points interactively in UE5")
    print("  [3] Enter AirSim coordinates manually")
    print()

    choice = input(f"Choose [{'1/4/5/' if has_existing else ''}2/3]: ").strip()

    framework_a, framework_b, ue5_a, ue5_b, mirror_x, mirror_y, offset_x, offset_y = (
        _collect_reference_points(client, traj_start, traj_end, has_existing, choice)
    )

    if choice in ["1", "4", "5"] and has_existing:
        with open(existing_cal, "r") as f:
            config = json.load(f)
        drift = ue5_a - np.array(config["airsim_origin"])
        if np.linalg.norm(drift) > 0.01:
            config["airsim_origin"] = ue5_a.tolist()
            if "airsim_endpoint" in config:
                config["airsim_endpoint"] = ue5_b.tolist()
            config["spawn_position"] = spawn_position.tolist()
    else:
        config = _compute_calibration(
            framework_a, framework_b, ue5_a, ue5_b,
            mirror_x, mirror_y, offset_x, offset_y,
            spawn_position,
        )

    if "ridge_overlay_offset" not in config:
        config["ridge_overlay_offset"] = [0.0, 0.0]
    if "ridge_overlay_rot_z_deg" not in config:
        config["ridge_overlay_rot_z_deg"] = 0.0

    print()
    print("=" * 40)
    print(f"  Scale:    {config['scale']:.6f}")
    print(f"  Rotation: {math.degrees(config['rotation']):.2f} deg")
    print(f"  Framework origin: {config['framework_origin']}")
    print(f"  AirSim origin: {config['airsim_origin']}")
    print("=" * 40)

    while True:
        out_path = _CONFIG_DIR / "calibration.json"
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n  Calibration saved to: {out_path}")

        coord = CoordTransform(config)
        _draw_visualization(
            client,
            coord,
            framework_a,
            framework_b,
            ue5_a,
            ue5_b,
            ridge_overlay_offset=cast(tuple[float, float], tuple(config.get("ridge_overlay_offset", [0.0, 0.0]))),
            ridge_overlay_rot_z_deg=float(config.get("ridge_overlay_rot_z_deg", 0.0)),
        )

        if choice not in ["4", "5"]:
            print("To clear markers: client.simFlushPersistentMarkers()")
            print("Done!")
            break

        print("=" * 40)
        if choice == "4":
            print("  LIVE TUNING MODE (FULL CALIBRATION)")
            print("  Shift the full Framework overlay in UE5 by entering an X and Y offset in meters.")
            print("  This updates airsim_origin and affects all transformed data.")
        else:
            print("  LIVE TUNING MODE (RIDGE OVERLAY ONLY)")
            print("  Shift ridges:   X, Y")
            print("  Rotate ridges Z-axis only: RZ, degrees")
            print("  This updates ridge_overlay_offset / ridge_overlay_rot_z_deg only and does NOT move the ideal path.")
        print("  Leave blank and press Enter to save and exit.")
        print("=" * 40)
        offset_val = input("Enter 'X, Y' or 'RZ, deg' (e.g. '10.5, -5.0' or 'RZ, 2.5'), or leave empty: ").strip()

        if not offset_val:
            # Always save the latest config before exit
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)
            print("\nCalibration saved to:", out_path)
            print("To clear markers: client.simFlushPersistentMarkers()")
            print("Done!")
            break

        # Any parse or update error re-prompts.
        # noinspection PyBroadException
        try:
            parts = offset_val.split(",")
            if len(parts) != 2:
                raise ValueError("Expected 2 comma-separated values")

            cmd = parts[0].strip().lower()
            val = float(parts[1].strip())

            if choice == "4":
                dx = float(parts[0].strip())
                dy = val
                config["airsim_origin"][0] += dx
                config["airsim_origin"][1] += dy
                print(f"\nApplied full calibration shift: +{dx} X, +{dy} Y. Redrawing...")
            else:
                if cmd == "rz":
                    config["ridge_overlay_rot_z_deg"] += val
                    print(
                        f"\nApplied ridge-only Z-axis rotation: +{val} deg "
                        f"(total {config['ridge_overlay_rot_z_deg']:.2f} deg). "
                        "Ideal path remains fixed. Redrawing..."
                    )
                else:
                    dx = float(parts[0].strip())
                    dy = val
                    config["ridge_overlay_offset"][0] += dx
                    config["ridge_overlay_offset"][1] += dy
                    print(f"\nApplied ridge-only shift: +{dx} X, +{dy} Y. Ideal path remains fixed. Redrawing...")
        except Exception:
            print("\nInvalid format. Use 'X, Y' (e.g. '10.5, -3.2') or 'RZ, deg' (e.g. 'RZ, 2.5')")


if __name__ == "__main__":
    run_calibration()
