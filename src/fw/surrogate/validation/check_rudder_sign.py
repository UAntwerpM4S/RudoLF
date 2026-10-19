"""
check_rudder_sign.py -- which way does FH turn for a positive rudder command?

Reads a dataset CSV written by a sim_fh_dataset_generator variant (raw X row,
3 targets, episode_id) and relates the stored rudder_action to the yaw rate
FH produced. Yaw rate is counter-clockwise (to port) positive -- verified with
check_sway_sign.py.

    slope/corr > 0  -> a positive rudder_action turns the ship to PORT
    slope/corr < 0  -> a positive rudder_action turns the ship to STARBOARD

The rudder_action is what the generator stored, NOT necessarily what it sent
to FH: the Tripoli generator sends +action (azipod setDesiredDirection), the
Myzako generators send -action (rudder setControlValue). Pass --negated if the
generator negated the action before sending it, to also get the answer for
FH's own command sign.

Usage (from src):
    python -m fw.surrogate.check_rudder_sign data/ship_dynamics_dataset.csv
    python -m fw.surrogate.check_rudder_sign data/myzako.csv --negated
"""

import argparse
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

from fw.surrogate.schema import scenario_for_raw_width

MIN_SURGE = 1.0          # m/s; the rudder has little effect below this
MIN_ABS_ACTION = 0.1     # rows with a clearly non-zero rudder


def main(argv=None):
    """
    CLI entry point: correlate the stored rudder_action with the yaw rate (and delta yaw rate)
    in the dataset CSV and print which way a positive command turns the ship.
    """

    ap = argparse.ArgumentParser(description="Sign of FH's turn for a positive rudder command.")
    ap.add_argument("csv")
    ap.add_argument("--negated", action="store_true",
                    help="the generator sent -rudder_action to FH (Myzako generators)")
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    args = ap.parse_args(argv)

    n_cols = len(pd.read_csv(args.csv, header=None, nrows=1).columns)
    scenario = scenario_for_raw_width(n_cols - 4)          # raw X + 3 targets + episode_id
    c_u, c_r, c_rud = scenario.col("surge"), scenario.col("yaw_rate"), scenario.col("rudder_action")
    c_dr = scenario.raw_input_dim + 2
    df = pd.read_csv(args.csv, header=None, usecols=[c_u, c_r, c_rud, c_dr], nrows=args.max_rows)
    u, r, a, dr = (df[c].to_numpy(dtype=float) for c in (c_u, c_r, c_rud, c_dr))

    m = (u > MIN_SURGE) & (np.abs(a) > MIN_ABS_ACTION)
    if m.sum() < 100:
        raise SystemExit(f"Only {int(m.sum())} usable rows (surge > {MIN_SURGE}, |rudder| > {MIN_ABS_ACTION}).")
    corr_r = float(np.corrcoef(a[m], r[m])[0, 1])
    corr_dr = float(np.corrcoef(a[m], dr[m])[0, 1])
    mean_r_pos = float(r[m & (a > 0)].mean())
    mean_r_neg = float(r[m & (a < 0)].mean())

    print(f"{args.csv}: scenario {scenario.name}, {int(m.sum()):,} rows with surge > {MIN_SURGE} m/s and "
          f"|rudder_action| > {MIN_ABS_ACTION}")
    print(f"  corr(rudder_action, yaw_rate)       = {corr_r:+.3f}")
    print(f"  corr(rudder_action, delta_yaw_rate) = {corr_dr:+.3f}")
    print(f"  mean yaw rate: rudder_action > 0 -> {np.degrees(mean_r_pos):+.3f} deg/s, "
          f"< 0 -> {np.degrees(mean_r_neg):+.3f} deg/s")
    stored = "PORT" if corr_r > 0 else "STARBOARD"
    print(f"=> a positive stored rudder_action turns the ship to {stored}.")
    sent = stored if not args.negated else ("STARBOARD" if stored == "PORT" else "PORT")
    print(f"=> a positive command SENT TO FH turns the ship to {sent}"
          + (" (the generator negated the action)." if args.negated else "."))
    if abs(corr_r) < 0.2:
        print("   (weak correlation -- treat with care)")


if __name__ == "__main__":
    main()
