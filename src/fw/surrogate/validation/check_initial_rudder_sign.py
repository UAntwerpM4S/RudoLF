"""
check_initial_rudder_sign.py -- does FH's setInitialRudderAngle use the same
sign as the per-step rudder command?

The Myzako generators and FhSession pass
    setInitialRudderAngle(fh_rudder_sign(ship) * deg)
i.e. they ASSUME the initial angle has the per-step command's sign
(check_rudder_sign.py only measured the per-step command). This script
measures it (needs FH):

  * two short runs, each with a fresh FhSimBridge (no rewind);
  * run "+": initial_rudder_angle = fh_rudder_sign(ship) * +ANGLE, which
    under that assumption is a rudder that turns the ship to PORT;
    run "-": the opposite;
  * after initialization the rudder is commanded back to 0, so only the
    initial angle (while FH moves the rudder back) turns the ship;
  * the heading change over the first seconds says which way it turned
    (heading counter-clockwise positive: > 0 = to port).

    run "+" turns to port and run "-" to starboard -> the assumption holds
    the other way round                             -> negate the initial angle
    both ~0                                         -> inconclusive (FH ignores
                                                       the initial angle, or
                                                       the rudder snaps back)

It also prints FH's own maximum rudder angle (getMaximumAngle) and the
angle-related methods of the rudder control, to compare with the
generator's RUDDER_MAX_DEG (the per-step setControlValue is normalized to
FH's maximum, the initial angle uses RUDDER_MAX_DEG degrees).

Usage (from src; position/heading must be in open water of the exercise):
    python -m fw.surrogate.check_initial_rudder_sign --ship MYZAKO_37
    python -m fw.surrogate.check_initial_rudder_sign --ship TRIPOLI --x -1559 --y 11990 --heading -45
"""

import argparse
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np

from fw.ships.ship_factory import ShipModel
from fw.ships.vessel_state import VesselState
from fw.dynamics.fh_tabular_model import fh_rudder_sign
from fw.simulators.fh_sim_bridge import FhSimBridge

ANGLE_DEG = 20.0          # initial rudder angle magnitude
SURGE = 3.0               # m/s initial surge
THRUST = 0.5              # engine lever during the run
PROPELLER_RPS = 2.75      # initial propeller rps (~ THRUST * generators' 5.5)
DURATION = 4.0            # s
STEP = 0.5                # s
MIN_TURN_DEG = 0.05       # |heading change| below this = no clear turn


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _controls(si, ship_model):
    if ship_model == ShipModel.TRIPOLI:
        return list(si.getAzipodControls()), []
    return list(si.getRudderControls()), list(si.getPropellerControls())


def run_once(ship_model: ShipModel, initial_deg: float, x: float, y: float, heading_deg: float,
             show_api: bool = False) -> float:
    """Heading change [deg] after DURATION s, rudder commanded to 0 after init."""

    bridge = FhSimBridge(ship_model=ship_model, rewind_enabled=False)
    bridge.initial_vessel_state = VesselState(terminated=False, truncated=False, x=x, y=y,
                                              heading=float(np.radians(heading_deg)),
                                              u=SURGE, v=0.0, r=0.0)
    bridge.initial_rudder_angle = float(initial_deg)
    bridge.initial_propeller_rps = PROPELLER_RPS
    try:
        bridge.reset()
        si = bridge.ship_interface
        rudders, propellers = _controls(si, ship_model)
        if not rudders:
            raise RuntimeError(f"No rudder/azipod controls on the exercise's ship for {ship_model.name}.")
        if show_api:
            r0 = rudders[0]
            angle_api = sorted(m for m in dir(r0) if "ngle" in m or "irection" in m)
            print(f"  rudder control: {type(r0).__name__}; angle-related methods: {angle_api}")
            if hasattr(r0, "getMaximumAngle"):
                print(f"  FH getMaximumAngle() = {r0.getMaximumAngle()} deg")
        # Direct read-back of the actuator right after initialisation (FH's own
        # getter: rudders getActualAngle, azipods getActualDirection), before
        # any command is sent.
        getter = "getActualDirection" if ship_model == ShipModel.TRIPOLI else "getActualAngle"
        if hasattr(rudders[0], getter):
            try:
                actual = f"{float(getattr(rudders[0], getter)()):+.2f} deg"
            except Exception as exc:  # diagnostic only
                actual = f"unreadable ({exc})"
            print(f"    after init: {getter}() = {actual} (setInitialRudderAngle({initial_deg:+.1f}))")
        h0 = float(si.getShipHeading())
        t = 0.0
        while t < DURATION - 1e-9:
            for c in rudders:
                if ship_model == ShipModel.TRIPOLI:
                    c.setDesiredDirection(0.0)
                    c.setEngineLeverValue(THRUST)
                else:
                    c.setControlValue(0.0)
            for p in propellers:
                p.engage()
                p.setEngineLeverValue(THRUST)
            bridge.math_model.simulateSeconds(STEP)
            t += STEP
        return _wrap_deg(float(si.getShipHeading()) - h0)
    finally:
        bridge.dispose()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sign of FH's setInitialRudderAngle.")
    ap.add_argument("--ship", choices=[m.name for m in ShipModel], default=ShipModel.MYZAKO_37.name)
    ap.add_argument("--x", type=float, default=-1559.0)
    ap.add_argument("--y", type=float, default=11990.0)
    ap.add_argument("--heading", type=float, default=-45.0,
                    help="deg, counter-clockwise from East (framework/FH convention)")
    args = ap.parse_args(argv)

    ship = ShipModel[args.ship]
    sign = fh_rudder_sign(ship)
    print(f"{ship.name}: fh_rudder_sign = {sign:+.0f}; initial angle +-{ANGLE_DEG} deg, rudder then "
          f"commanded to 0, {DURATION} s at {SURGE} m/s")

    turns = {}
    for label, framework_sign in (("+", 1.0), ("-", -1.0)):
        initial = sign * framework_sign * ANGLE_DEG
        dh = run_once(ship, initial, args.x, args.y, args.heading, show_api=(label == "+"))
        turns[label] = dh
        side = "PORT" if dh > MIN_TURN_DEG else "STARBOARD" if dh < -MIN_TURN_DEG else "none"
        print(f"  run {label}: setInitialRudderAngle({initial:+.1f}) -> heading change {dh:+.3f} deg ({side})")

    if abs(turns["+"]) < MIN_TURN_DEG and abs(turns["-"]) < MIN_TURN_DEG:
        print("=> INCONCLUSIVE: no clear turn in either run.")
    elif turns["+"] > 0 > turns["-"]:
        print("=> OK: the initial angle has the per-step command's sign (fh_rudder_sign is right for it).")
    elif turns["+"] < 0 < turns["-"]:
        print("=> OPPOSITE: setInitialRudderAngle needs -fh_rudder_sign(ship) * deg "
              "(Myzako generators, FhSession).")
    else:
        print("=> UNCLEAR: both runs turned the same way (check the start position: bank effects?).")


if __name__ == "__main__":
    main()
