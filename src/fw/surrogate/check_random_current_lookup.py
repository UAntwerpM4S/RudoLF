"""
Check how the "random" current of the UAntwerp full_env surrogate
(ua_surrogate_config["wind_current"] = "random") follows a real lookup CSV.

The ship is moved along the env's route (checkpoints from start to target) in
small steps; at each point the nearest lookup row gives the current line and
fw.dynamics.uantwerp_surrogate.follow_current_line the direction, exactly as
during training, for both flood/ebb signs. Reported per sign:

  * rows below CURRENT_AXIS_MIN_SPEED (slack) and rows whose line turned more
    than CURRENT_AXIS_MAX_TURN_DEG from the reference (rejected as cross flow);
    both keep the reference direction,
  * the largest step-to-step direction change and the number above --jump-deg,
  * reversals: sign changes of the current's component along the route
    (only where it flows within 45 deg of the route), which should be 0,
  * the share of points where the current is within 45 deg of the route,
  * the lookup distance (nearest row) along the route.

Usage (from src):

    python -m fw.surrogate.check_random_current_lookup --preset myzako_full
    python -m fw.surrogate.check_random_current_lookup --preset tripoli_full --step 0.5
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from typing import Dict, List, Sequence
from fw.common.default_positions import DEFAULT_FRAMEWORK_SHIP_POS, DEFAULT_FRAMEWORK_TARGET_POS
from fw.dynamics.uantwerp_surrogate import (
    CURRENT_AXIS_MAX_TURN_DEG,
    CURRENT_AXIS_MIN_SPEED,
    follow_current_line,
    make_scenario_lookup,
)
from fw.surrogate.presets import get_preset, resolve_lookup_csv

_ALONG = np.cos(np.radians(45.0))


def route_points(start, target, step: float) -> np.ndarray:
    """
    Points every `step` metres along the PySimEnv route (its checkpoints) from `start` to `target`.
    """

    from fw.dynamics.dynamics_model import Model
    from fw.environments.py_sim_env import PySimEnv
    from fw.ships.ship_factory import ShipModel

    env = PySimEnv(render_mode=None, dynamics_model=Model.FOSSEN, ship_model=ShipModel.MYZAKO_37,
                   start_pos=np.array(start, dtype=np.float32), target_pos=np.array(target, dtype=np.float32))
    corners = np.array([np.asarray(c["pos"], dtype=np.float64) for c in env.checkpoints])
    points = [corners[0]]
    for a, b in zip(corners[:-1], corners[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        points.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return np.array(points)


def follow_route(lookup, points: np.ndarray, sign: float, jump_deg: float) -> Dict[str, object]:
    """
    Run follow_current_line along `points` for one sign; returns the statistics (see module docstring).
    """

    tangents = np.gradient(points, axis=0)
    prev = ref = None
    along_signs: List[int] = []
    stats = {"points": len(points), "slack": 0, "turn_rejected": 0, "max_change_deg": 0.0, "jumps": 0,
             "reversals": 0, "along_share": 0.0, "worst": []}
    distances, along = [], 0
    for p, t in zip(points, tangents):
        sample = lookup.get_sample(float(p[0]), float(p[1]))
        axis, speed = float(sample.env.current_direction), float(sample.env.current_speed)
        distances.append(sample.distance)
        if not speed >= CURRENT_AXIS_MIN_SPEED:
            stats["slack"] += 1
        elif ref is not None and abs(np.cos(axis - ref)) < np.cos(np.radians(CURRENT_AXIS_MAX_TURN_DEG)):
            stats["turn_rejected"] += 1
        d, ref = follow_current_line(axis, speed, ref, sign)
        if prev is not None:
            change = abs(np.degrees((d - prev + np.pi) % (2.0 * np.pi) - np.pi))
            stats["max_change_deg"] = max(stats["max_change_deg"], change)
            stats["jumps"] += int(change > jump_deg)
            stats["worst"].append((change, float(p[0]), float(p[1])))
        c = np.cos(d - np.arctan2(t[1], t[0]))
        if abs(c) >= _ALONG:
            along += 1
            along_signs.append(1 if c > 0 else -1)
        prev = d
    stats["reversals"] = int(np.sum(np.diff(along_signs) != 0)) if along_signs else 0
    stats["along_share"] = along / len(points)
    stats["distance_mean"] = float(np.mean(distances))
    stats["distance_max"] = float(np.max(distances))
    stats["worst"] = sorted(stats["worst"], reverse=True)[:5]
    return stats


def main(argv: Sequence[str] = None) -> int:
    """
    Command-line entry point; prints the statistics per sign.
    """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True, help="full_env preset, e.g. myzako_full or tripoli_full.")
    parser.add_argument("--lookup-csv", default=None, help="Override of the preset's lookup CSV.")
    parser.add_argument("--step", type=float, default=0.5, help="Distance between route points [m] (default 0.5).")
    parser.add_argument("--jump-deg", type=float, default=10.0, help="Report direction changes above this [deg].")
    args = parser.parse_args(argv)

    preset = get_preset(args.preset)
    if not preset.scenario.includes_wind_current:
        parser.error(f"Preset '{preset.name}' has no wind/current (not full_env).")
    csv = resolve_lookup_csv(args.lookup_csv if args.lookup_csv is not None else preset.lookup_csv)
    if csv is None or not csv.exists():
        parser.error(f"Lookup CSV not found: {csv}")
    lookup = make_scenario_lookup(csv, preset.scenario, preset.name)
    points = route_points(DEFAULT_FRAMEWORK_SHIP_POS, DEFAULT_FRAMEWORK_TARGET_POS, args.step)
    print(f"{csv}: {len(points)} route points every {args.step:g} m")

    for sign, label in ((1.0, "sign +1"), (-1.0, "sign -1")):
        s = follow_route(lookup, points, sign, args.jump_deg)
        print(f"{label}: slack rows {s['slack']}, rejected turns {s['turn_rejected']}, "
              f"max step change {s['max_change_deg']:.1f} deg, changes > {args.jump_deg:g} deg: {s['jumps']}, "
              f"reversals along the route: {s['reversals']}, within 45 deg of the route: "
              f"{100 * s['along_share']:.1f} %, lookup distance mean {s['distance_mean']:.1f} m / "
              f"max {s['distance_max']:.1f} m")
        for change, x, y in s["worst"]:
            print(f"    {change:6.1f} deg at ({x:.1f}, {y:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
