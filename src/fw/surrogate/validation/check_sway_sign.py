"""
Check the sign of FH's sway (and yaw rate) from existing test outputs.

Uses the outputs of the wind and current comparison tests (no FH needed):
FH's track (trajectory_*.csv) and FH's sway as the tests read it (env_*.csv:
fh_sway from FhSession.state(), the value the dataset generator stores).

Per FH step, fh_sway is regressed on the sideways velocity of FH's own track,
projected on the starboard direction (ray_90):

    slope ~ -1: fh_sway is port positive (framework convention, fw.ships.vessel_state)
    slope ~ +1: fh_sway is starboard positive (old outputs, or a sign flip)

Surge must give slope ~ +1 (sanity check). For the yaw rate, slope ~ +1 means
counter-clockwise (port) positive, as ShipSimulator.integrate assumes.

This matters because ShipSimulator.integrate treats v as port positive;
starboard-positive sway makes every non-FH model drift to the wrong side.

Usage (from src):
    python -m fw.surrogate.validation.check_sway_sign model_comparison_output/wind_trajectories
    python -m fw.surrogate.validation.check_sway_sign model_comparison_output/current_trajectories
"""

import sys
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from fw.environments.ray_features import RAY_ANGLES_DEG, ray_world_angles

# Allow running by file path (src on the import path).
_SRC_DIR = str(Path(__file__).resolve().parents[3])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_IDX_BOW = int(np.flatnonzero(np.isclose(np.asarray(RAY_ANGLES_DEG, dtype=float), 0.0))[0])
_IDX_STBD = int(np.flatnonzero(np.isclose(np.asarray(RAY_ANGLES_DEG, dtype=float), 90.0))[0])
MIN_ABS_SWAY = 1e-3   # m/s; steps with smaller sway carry no sign information


def _unit(angle: float) -> np.ndarray:
    """
    Unit vector (cos, sin) for a world-frame angle [rad].
    """

    return np.array([np.cos(angle), np.sin(angle)])


def _slope(observed: np.ndarray, reported: np.ndarray) -> float:
    """
    Least-squares slope through the origin of observed vs reported.
    """

    return float(observed @ reported / max(float(reported @ reported), 1e-30))


def check_pair(traj_path: Path, env_path: Path):
    """
    Compare FH's reported sway, surge and yaw rate with the motion seen in FH's own track.

    Args:
        traj_path: trajectory_*.csv of one run (FH rows: step, t, x, y, heading, u).
        env_path: Matching env_*.csv (fh_sway, optionally fh_yaw_rate, per step).

    Returns:
        Dict with n, sway_slope, sway_corr, surge_slope and yaw_slope (NaN when there are
        too few steps with a yaw rate), or None when the pair has too little usable data.
    """

    traj = pd.read_csv(traj_path)
    fh = traj[traj["model"] == "FH"].sort_values("step")
    env = pd.read_csv(env_path).set_index("step")
    if len(fh) < 3 or "fh_sway" not in env.columns:
        return None
    steps = fh["step"].to_numpy()
    xy = fh[["x", "y"]].to_numpy(dtype=float)
    hdg = fh["heading"].to_numpy(dtype=float)
    t = fh["t"].to_numpy(dtype=float)
    u = fh["u"].to_numpy(dtype=float)

    has_r = "fh_yaw_rate" in env.columns
    v_obs, v_rep, u_obs, u_rep, r_obs, r_rep = [], [], [], [], [], []
    for i in range(len(fh) - 1):
        s0, s1 = int(steps[i]), int(steps[i + 1])
        if s0 not in env.index:
            continue
        dt = t[i + 1] - t[i]
        if dt <= 0:
            continue
        vel = (xy[i + 1] - xy[i]) / dt
        # mid-step heading, wrapped difference
        dpsi = (hdg[i + 1] - hdg[i] + np.pi) % (2 * np.pi) - np.pi
        angles = ray_world_angles(hdg[i] + 0.5 * dpsi)
        bow, stbd = _unit(angles[_IDX_BOW]), _unit(angles[_IDX_STBD])
        sway = float(env.loc[s0, "fh_sway"])
        if s1 in env.index:
            sway = 0.5 * (sway + float(env.loc[s1, "fh_sway"]))
        v_obs.append(vel @ stbd)
        v_rep.append(sway)
        u_obs.append(vel @ bow)
        u_rep.append(0.5 * (u[i] + u[i + 1]))
        if has_r:
            r = float(env.loc[s0, "fh_yaw_rate"])
            if s1 in env.index:
                r = 0.5 * (r + float(env.loc[s1, "fh_yaw_rate"]))
            r_obs.append(dpsi / dt)
            r_rep.append(r)
    v_obs, v_rep = np.asarray(v_obs), np.asarray(v_rep)
    keep = np.abs(v_rep) > MIN_ABS_SWAY
    if keep.sum() < 5:
        return None
    r_obs, r_rep = np.asarray(r_obs), np.asarray(r_rep)
    keep_r = np.abs(r_rep) > 1e-4
    yaw_slope = _slope(r_obs[keep_r], r_rep[keep_r]) if keep_r.sum() >= 5 else float("nan")
    return dict(n=int(keep.sum()), sway_slope=_slope(v_obs[keep], v_rep[keep]), yaw_slope=yaw_slope,
                sway_corr=float(np.corrcoef(v_obs[keep], v_rep[keep])[0, 1]),
                surge_slope=_slope(np.asarray(u_obs), np.asarray(u_rep)))


def main(argv=None):
    """
    CLI entry point: run check_pair on every trajectory/env CSV pair in the output folder and
    print the step-weighted slopes with a verdict on the sway sign.
    """

    ap = argparse.ArgumentParser(description="Sign of FH's sway, from existing comparison-test outputs.")
    ap.add_argument("out_dir", help="wind_trajectories or current_trajectories output folder")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)

    rows = []
    for traj in sorted(out.glob("trajectory_*.csv")):
        env = traj.with_name("env_" + traj.name[len("trajectory_"):])
        if not env.is_file():
            continue
        res = check_pair(traj, env)
        if res is not None:
            rows.append((traj.name, res))
            print(f"{traj.name:55s} n={res['n']:5d}  sway slope {res['sway_slope']:+.3f}  "
                  f"corr {res['sway_corr']:+.3f}  | surge slope {res['surge_slope']:+.3f}  "
                  f"| yaw-rate slope {res['yaw_slope']:+.3f}")
    if not rows:
        raise SystemExit(f"No usable trajectory_*.csv / env_*.csv pairs with FH sway in {out}.")

    n = np.array([r["n"] for _, r in rows], dtype=float)
    sway = float(np.average([r["sway_slope"] for _, r in rows], weights=n))
    surge = float(np.average([r["surge_slope"] for _, r in rows], weights=n))
    yaw = [(r["yaw_slope"], r["n"]) for _, r in rows if np.isfinite(r["yaw_slope"])]
    yaw_txt = (f"{np.average([y for y, _ in yaw], weights=[w for _, w in yaw]):+.3f}" if yaw else "n/a")
    print(f"\nWeighted over {len(rows)} runs: sway slope {sway:+.3f}, surge slope {surge:+.3f} (must be ~ +1), "
          f"yaw-rate slope {yaw_txt} (+1 = counter-clockwise positive, as ShipSimulator assumes)")
    if abs(surge - 1.0) > 0.1:
        print("Surge check failed: the track/heading convention itself is off -- the sway result is not reliable.")
    elif sway < -0.5:
        print("=> fh_sway is PORT positive: consistent with ShipSimulator.integrate and fw.ships.vessel_state. OK.")
    elif sway > 0.5:
        print("=> fh_sway is STARBOARD positive: these outputs predate the 2026-10-19 sway fix, or a sign flip "
              "crept back in (FhSession.state() / the generator). Non-FH models would drift to the wrong side.")
    else:
        print("=> Inconclusive (too little sideways motion); try runs with cross wind/current.")


if __name__ == "__main__":
    main()
