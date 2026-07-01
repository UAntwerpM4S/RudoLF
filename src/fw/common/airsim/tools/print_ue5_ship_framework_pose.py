"""
Print the current UE5 AirSim ship pose in both raw and framework coordinates.

Run from the repository root:

    python print_ue5_ship_framework_pose.py

Optional:

    python print_ue5_ship_framework_pose.py --json
    python print_ue5_ship_framework_pose.py --draw-reference
    python print_ue5_ship_framework_pose.py --ip 127.0.0.1 --port 41451
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from airsim.communication.airsim_bridge import AirSimBridge
from airsim.config.airsim_config import AirSimConfig, get_airsim_config
from fw.common.airsim.tools.angles import yaw_from_quaternion
from fw.common.airsim.tools.calibrate_coords import load_saved_calibration_reference


def _build_parser() -> argparse.ArgumentParser:
    """
    Create the CLI parser for the framework pose reader.
    """

    parser = argparse.ArgumentParser(
        description="Read the current AirSim ship pose and convert it to framework coordinates."
    )
    parser.add_argument("--ip", default=None, help="AirSim RPC IP override.")
    parser.add_argument("--port", type=int, default=None, help="AirSim RPC port override.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the pose data as one JSON object.",
    )
    parser.add_argument(
        "--draw-reference",
        action="store_true",
        help="Redraw the saved calibration overlay in UE5 before printing.",
    )
    return parser


def main() -> int:
    """
    Connect to AirSim, apply calibration drift correction, and print the pose.
    """

    args = _build_parser().parse_args()

    runtime_cfg = get_airsim_config()
    bridge_cfg = AirSimConfig(
        use_airsim=True,
        ip=runtime_cfg.get("ip", "") if args.ip is None else args.ip,
        port=int(runtime_cfg.get("port", 41451)) if args.port is None else int(args.port),
        timeout=float(runtime_cfg.get("timeout", 10.0)),
        max_retries=int(runtime_cfg.get("max_retries", 3)),
        health_check_interval=float(runtime_cfg.get("health_check_interval", 30.0)),
        csv_log_path="",
    )

    bridge = AirSimBridge(bridge_cfg)
    if not bridge.is_connected:
        print(
            f"Could not connect to AirSim at {bridge_cfg.resolved_ip}:{bridge_cfg.port}.",
            file=sys.stderr,
        )
        return 1

    pose = bridge.get_vehicle_pose()
    raw_x = float(pose.position.x_val)
    raw_y = float(pose.position.y_val)
    raw_z = float(pose.position.z_val)
    raw_yaw_rad = float(yaw_from_quaternion(pose.orientation))
    raw_heading_deg = float(math.degrees(raw_yaw_rad))

    coord_transform, corrected_config = load_saved_calibration_reference(
        bridge.client,
        draw_reference=bool(args.draw_reference),
    )
    framework_x, framework_y = coord_transform.airsim_to_framework(raw_x, raw_y)
    framework_heading_deg = float(
        math.degrees(coord_transform.airsim_yaw_to_framework_heading(raw_yaw_rad))
    )

    saved_spawn = corrected_config.get("spawn_position")
    drift_x = None
    drift_y = None
    drift_m = None
    if isinstance(saved_spawn, list) and len(saved_spawn) >= 2:
        drift_x = float(raw_x - float(saved_spawn[0]))
        drift_y = float(raw_y - float(saved_spawn[1]))
        drift_m = float(math.hypot(drift_x, drift_y))

    result = {
        "raw_airsim_x": raw_x,
        "raw_airsim_y": raw_y,
        "raw_airsim_z": raw_z,
        "raw_airsim_heading_deg": raw_heading_deg,
        "framework_ship_x": float(framework_x),
        "framework_ship_y": float(framework_y),
        "framework_heading_deg": framework_heading_deg,
        "drift_x": drift_x,
        "drift_y": drift_y,
        "drift_m": drift_m,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Raw UE5 / AirSim pose")
        print(f"  x = {result['raw_airsim_x']:.6f}")
        print(f"  y = {result['raw_airsim_y']:.6f}")
        print(f"  z = {result['raw_airsim_z']:.6f}")
        print(f"  heading_deg = {result['raw_airsim_heading_deg']:.6f}")
        print()
        print("Drift-corrected framework pose")
        print(f"  ship_x = {result['framework_ship_x']:.6f}")
        print(f"  ship_y = {result['framework_ship_y']:.6f}")
        print(f"  heading_deg = {result['framework_heading_deg']:.6f}")
        if result["drift_x"] is not None:
            print()
            print("Detected spawn drift vs calibration")
            print(f"  dx = {result['drift_x']:.6f}")
            print(f"  dy = {result['drift_y']:.6f}")
            print(f"  drift_m = {result['drift_m']:.6f}")

    try:
        bridge.disconnect()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
