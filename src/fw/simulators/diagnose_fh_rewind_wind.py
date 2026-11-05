"""
Diagnose when FH applies a wind-field transform around SimulateRewindTo.

FhSimBridge's rewind probe failed with "true wind 3.60 m/s at Scale 0 and
0.03 m/s at Scale 10": each reading matched the transform set one step
earlier, not the current one. This script tells apart the candidate causes on
one FH session, by logging FH's wind every LOG_STEP seconds after:

  A  rewind, then set the transform  (what FhSimBridge does now)
  B  set the transform, then rewind
  C  rewind, set the transform, rewind again

each for alternating Scales (SCALES), so a one-setting lag shows up. It also
reads the transform back (getFieldTransform) right after setting it and after
the rewind, to see whether the rewind restores an older transform.

Usage (from src, with SIMEXE_REPO_FOLDER set):

    python -m fw.simulators.diagnose_fh_rewind_wind --ship MYZAKO_37 --scenario <config scenario>
"""

from __future__ import annotations

import sys
import argparse
import numpy as np

from typing import Dict, List, Sequence
from fw.ships.ship_factory import ShipModel
from fw.ships.vessel_state import VesselState
from fw.simulators.fh_sim_bridge import FhSimBridge
from fw.simulators.fh_interface_utils import fh_true_wind_speed
from fw.common.default_positions import DEFAULT_FRAMEWORK_SHIP_POS

REWIND_TIME = 0.25        # s, as FhSimBridge.reset()
LOG_STEP = 0.25           # s between readings
LOG_DURATION = 5.0        # s logged after each setting
SCALES = (10.0, 0.0, 10.0, 0.0)
FIRST_RUN_SECONDS = 20.0  # s simulated before the first rewind (as a short episode)


def _sim_time(bridge: FhSimBridge) -> float:
    """
    FH's simulation time [s].
    """

    return float(bridge.math_model.getSimInterface().getSimulationTime())


def _transform_scale(bridge: FhSimBridge) -> float:
    """
    Scale of the wind-field transform as FH reports it.
    """

    return float(bridge.math_model.getEnvironmentInterface().getWindFieldControl().getFieldTransform().Scale)


def _log(bridge: FhSimBridge) -> List[Dict[str, float]]:
    """
    Simulate LOG_DURATION in LOG_STEP steps, recording time, WindState and true wind.
    """

    rows = []
    for _ in range(int(round(LOG_DURATION / LOG_STEP))):
        bridge.math_model.simulateSeconds(LOG_STEP)
        ws = bridge.ship_interface.getWindState()
        rows.append({"t": _sim_time(bridge), "app_speed": float(ws.Speed), "app_dir": float(ws.Direction),
                     "true": fh_true_wind_speed(bridge.ship_interface)})
    return rows


def _rewind(bridge: FhSimBridge) -> None:
    """
    Rewind FH to REWIND_TIME.
    """

    bridge.math_model.SimulateRewindTo(REWIND_TIME)


def _run_mode(bridge: FhSimBridge, mode: str) -> None:
    """
    Run one mode (A, B or C) for every Scale in SCALES and print the readings.
    """

    print(f"\n===== Mode {mode} =====")
    for scale in SCALES:
        if mode == "B":
            bridge._apply_wind_field_transform(0.0, scale, 0.0)
            set_readback = _transform_scale(bridge)
            _rewind(bridge)
        else:
            _rewind(bridge)
            bridge._apply_wind_field_transform(0.0, scale, 0.0)
            set_readback = _transform_scale(bridge)
            if mode == "C":
                _rewind(bridge)
        after_rewind = _transform_scale(bridge)
        t0 = _sim_time(bridge)
        rows = _log(bridge)
        print(f"Scale {scale:g}: readback after set {set_readback:g}, before logging {after_rewind:g}, "
              f"logging from t = {t0:.2f} s")
        print("    t [s]   true wind [m/s]   WindState speed [m/s]  dir [deg]")
        for r in rows:
            print(f"    {r['t']:6.2f}  {r['true']:8.2f}          {r['app_speed']:8.2f}           {r['app_dir']:7.1f}")


def main(argv: Sequence[str] = None) -> int:
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ship", default="MYZAKO_37", help="ShipModel name (default MYZAKO_37).")
    parser.add_argument("--scenario", default=None, help="Scenario as in the config (default: FhSimBridge's).")
    parser.add_argument("--heading-deg", type=float, default=55.3, help="Start heading [deg], CCW from East.")
    parser.add_argument("--modes", default="ABC", help="Modes to run, e.g. A or AB (default ABC).")
    args = parser.parse_args(argv)

    kwargs = {"ship_model": ShipModel[args.ship], "wind_current": True}
    if args.scenario is not None:
        kwargs["scenario"] = args.scenario
    bridge = FhSimBridge(**kwargs)
    print(f"Exercise: {bridge.exercise_path}")
    x, y = DEFAULT_FRAMEWORK_SHIP_POS
    bridge.initial_vessel_state = VesselState(terminated=False, truncated=False, x=x, y=y,
                                              heading=float(np.radians(args.heading_deg)), u=0.0, v=0.0, r=0.0)
    try:
        # First episode as FhSimBridge.reset() starts it (transform before the first simulateSeconds).
        bridge.initial_wind_field_transform = (0.0, 3.0, 0.0)
        bridge._initialize_math_model(snapshot_window=bridge.SNAPSHOT_WINDOW, snapshot_frequency=5000)
        print(f"Initialized with Scale 3; transform reads {_transform_scale(bridge):g}")
        bridge.math_model.simulateSeconds(FIRST_RUN_SECONDS)
        print(f"After {FIRST_RUN_SECONDS:g} s: true wind {fh_true_wind_speed(bridge.ship_interface):.2f} m/s "
              f"at t = {_sim_time(bridge):.2f} s")
        for mode in args.modes.upper():
            if mode not in "ABC":
                parser.error(f"Unknown mode {mode!r}.")
            _run_mode(bridge, mode)
    finally:
        bridge.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
