"""
Print the current UE5 AirSim ship position and heading.

Usage (from the repository root):

    python src/fw/common/airsim/tools/print_ue5_ship_pose.py [--ip 127.0.0.1] [--port 41451] [--json]
"""

from __future__ import annotations

import sys
import math
import json
import argparse

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from airsim.communication.airsim_bridge import AirSimBridge
from fw.common.airsim.tools.angles import yaw_from_quaternion
from airsim.config.airsim_config import AirSimConfig, get_airsim_config


def _build_parser() -> argparse.ArgumentParser:
    """
    Create the CLI parser.
    """

    parser = argparse.ArgumentParser(
        description="Read the current AirSim ship pose from UE5."
    )
    parser.add_argument("--ip", default=None, help="AirSim RPC IP override.")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="AirSim RPC port override.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the pose as one JSON object.",
    )
    return parser


def main() -> int:
    """
    Connect to AirSim, read the live pose and print it. Returns the exit code.
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
    heading_rad = float(yaw_from_quaternion(pose.orientation))
    result = {
        "x": float(pose.position.x_val),
        "y": float(pose.position.y_val),
        "z": float(pose.position.z_val),
        "heading_deg": float(math.degrees(heading_rad)),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"x = {result['x']:.6f}")
        print(f"y = {result['y']:.6f}")
        print(f"z = {result['z']:.6f}")
        print(f"heading_deg = {result['heading_deg']:.6f}")

    try:
        bridge.disconnect()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
