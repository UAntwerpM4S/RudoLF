"""
Check how the "random" current of the UAntwerp full_env surrogate
(ua_surrogate_config["wind_current"] = "random") follows a real lookup CSV.

The ship is moved along the env's route (checkpoints from start to target),
and along copies shifted sideways by --offsets (the ship wanders in
training), in small steps at --speed; at each point the nearest lookup row gives the current
line and fw.dynamics.uantwerp_surrogate.CurrentLineFollower the direction,
exactly as during training (line average --tau, as env_smoothing_tau), for
both flood/ebb signs. Reported per path and sign:

  * rows below CURRENT_AXIS_MIN_SPEED (slack, not averaged into the line),
    steps with zero current before the first reliable row, and steps where
    the averaged line was weaker than WEAK_LINE_STRENGTH (direction barely moves),
  * the largest step-to-step direction change and the number above --jump-deg,
  * reversals: sign changes of the current's component along the route
    (only where it flows within 45 deg of the route), with their positions;
    a reversal can also be the current gradually following a flow line that
    turns away from the route (check the largest step changes),
  * the share of points where the current is within 45 deg of the route,
  * the lookup distance (nearest row) along the route.

Usage (from src):

    python -m fw.surrogate.check_random_current_lookup --preset myzako_full
    python -m fw.surrogate.check_random_current_lookup --preset tripoli_full --offsets 0 25 -25 50 -50 100 -100
"""

from __future__ import annotations

import sys
import argparse
import numpy as np

from typing import Dict, List, Optional, Sequence, Tuple
from fw.surrogate.presets import get_preset, resolve_lookup_csv
from fw.common.default_positions import DEFAULT_FRAMEWORK_SHIP_POS, DEFAULT_FRAMEWORK_TARGET_POS
from fw.dynamics.uantwerp_surrogate import (
    CURRENT_AXIS_MIN_SPEED,
    CurrentLineFollower,
    make_scenario_lookup,
)

_ALONG = np.cos(np.radians(45.0))
WEAK_LINE_STRENGTH = 0.2   # reporting only: line strength below this counts as a weak-line step


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


def offset_path(points: np.ndarray, offset: float) -> np.ndarray:
    """
    `points` shifted sideways by `offset` metres (positive to port of the route direction).
    """

    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    return points + offset * np.column_stack([-tangents[:, 1], tangents[:, 0]])


def follow_route(lookup, points: np.ndarray, sign: float, jump_deg: float, tau: Optional[float] = None,
                 speed: float = 4.0, tangents: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    Run CurrentLineFollower along `points` for one sign, sailing at `speed`
    [m/s] (time step = point spacing / speed); returns the statistics (see
    module docstring). `tangents`: route direction per point (default: of
    `points`; pass the centerline's for offset paths, which fold back at
    inner corners).
    """

    if tangents is None:
        tangents = np.gradient(points, axis=0)
    follower = CurrentLineFollower(sign, tau)
    prev = None
    along_signs: List[int] = []
    along_points: List[Tuple[float, float]] = []
    stats = {"points": len(points), "slack": 0, "zero_current": 0, "weak_line": 0, "max_change_deg": 0.0,
             "jumps": 0, "reversals": 0, "reversal_positions": [], "along_share": 0.0, "worst": []}
    distances, along = [], 0
    for k, (p, t) in enumerate(zip(points, tangents)):
        dt = float(np.linalg.norm(points[k] - points[k - 1])) / speed if k > 0 else 0.0
        sample = lookup.get_sample(float(p[0]), float(p[1]))
        axis, axis_speed = float(sample.env.current_direction), float(sample.env.current_speed)
        distances.append(sample.distance)
        stats["slack"] += int(not axis_speed >= CURRENT_AXIS_MIN_SPEED)
        d = follower.step(axis, axis_speed, dt)
        if d is None:
            stats["zero_current"] += 1
            continue
        stats["weak_line"] += int(follower.line_strength < WEAK_LINE_STRENGTH)
        if prev is not None:
            change = abs(np.degrees((d - prev + np.pi) % (2.0 * np.pi) - np.pi))
            stats["max_change_deg"] = max(stats["max_change_deg"], change)
            stats["jumps"] += int(change > jump_deg)
            stats["worst"].append((change, float(p[0]), float(p[1])))
        c = np.cos(d - np.arctan2(t[1], t[0]))
        if abs(c) >= _ALONG:
            along += 1
            along_signs.append(1 if c > 0 else -1)
            along_points.append((float(p[0]), float(p[1])))
        prev = d
    flips = np.flatnonzero(np.diff(along_signs) != 0) if along_signs else np.array([], dtype=int)
    stats["reversals"] = int(len(flips))
    stats["reversal_positions"] = [along_points[i + 1] for i in flips]
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
    parser.add_argument("--tau", type=float, default=10.0,
                        help="River-line average time constant [s] as env_smoothing_tau (default 10; "
                             "0 = CURRENT_LINE_TAU, as without smoothing).")
    parser.add_argument("--speed", type=float, default=4.0, help="Ship speed along the route [m/s] (default 4).")
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.0],
                        help="Sideways offsets of the route [m], positive to port (default 0).")
    args = parser.parse_args(argv)

    preset = get_preset(args.preset)
    if not preset.scenario.includes_wind_current:
        parser.error(f"Preset '{preset.name}' has no wind/current (not full_env).")
    csv = resolve_lookup_csv(args.lookup_csv if args.lookup_csv is not None else preset.lookup_csv)
    if csv is None or not csv.exists():
        parser.error(f"Lookup CSV not found: {csv}")
    if args.speed <= 0:
        parser.error("--speed must be positive.")
    if args.step <= 0:
        parser.error("--step must be positive.")
    if args.tau < 0:
        parser.error("--tau must be >= 0 (0 = CURRENT_LINE_TAU).")
    lookup = make_scenario_lookup(csv, preset.scenario, preset.name)
    points = route_points(DEFAULT_FRAMEWORK_SHIP_POS, DEFAULT_FRAMEWORK_TARGET_POS, args.step)
    tau = args.tau if args.tau > 0 else None
    print(f"{csv}: {len(points)} route points every {args.step:g} m at {args.speed:g} m/s, "
          f"line tau {'CURRENT_LINE_TAU' if tau is None else f'{tau:g} s'}")

    route_tangents = np.gradient(points, axis=0)
    for offset in args.offsets:
        path = offset_path(points, offset) if offset else points
        for sign, label in ((1.0, "sign +1"), (-1.0, "sign -1")):
            stats = follow_route(lookup, path, sign, args.jump_deg, tau, args.speed, route_tangents)
            _report(f"offset {offset:g} m, {label}", stats, args.jump_deg)
    return 0


def _report(label: str, s: Dict[str, object], jump_deg: float) -> None:
    """
    Print the statistics of one path and sign.
    """

    print(f"{label}: slack rows {s['slack']}, zero-current steps at the start {s['zero_current']}, "
          f"weak-line steps {s['weak_line']}, "
          f"max step change {s['max_change_deg']:.1f} deg, changes > {jump_deg:g} deg: {s['jumps']}, "
          f"reversals along the route: {s['reversals']}, within 45 deg of the route: "
          f"{100 * s['along_share']:.1f} %, lookup distance mean {s['distance_mean']:.1f} m / "
          f"max {s['distance_max']:.1f} m")
    print("    largest step changes:")
    for change, x, y in s["worst"]:
        print(f"    {change:6.1f} deg at ({x:.1f}, {y:.1f})")
    for x, y in s["reversal_positions"]:
        print(f"    reversal at ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    sys.exit(main())
