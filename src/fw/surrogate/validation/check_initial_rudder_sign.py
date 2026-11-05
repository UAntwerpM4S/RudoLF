"""
Check whether FH's setInitialRudderAngle uses the same sign as the per-step
rudder command (the Myzako generators and FhSession assume it does and pass
fh_rudder_sign(ship) * deg). Needs FH.

Two short runs with a fresh FhSimBridge each: "+" sets an initial angle that
should turn the ship to port under that assumption, "-" the opposite. After
initialization the rudder is commanded to 0, so only the initial angle turns
the ship. The heading change (counter-clockwise positive) gives the verdict:
"+" to port and "-" to starboard means the assumption holds; the reverse means
the initial angle must be negated; both near 0 is inconclusive.

Also prints FH's getMaximumAngle and the rudder control's angle-related
methods, for comparison with the generators' RUDDER_MAX_DEG.

Usage (from src; position/heading must be in open water of the exercise):
    python -m fw.surrogate.validation.check_initial_rudder_sign --ship MYZAKO_37
    python -m fw.surrogate.validation.check_initial_rudder_sign --ship TRIPOLI --x -1559 --y 11990 --heading -45
"""

import sys
import argparse
import numpy as np

from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[3])     # src (this file: src/fw/surrogate/validation/)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fw.ships.ship_factory import ShipModel
from fw.ships.vessel_state import VesselState
from fw.simulators.fh_sim_bridge import FhSimBridge
from fw.dynamics.fh_tabular_model import fh_rudder_sign

ANGLE_DEG = 20.0          # initial rudder angle magnitude
SURGE = 3.0               # m/s initial surge
THRUST = 0.5              # engine lever during the run
PROPELLER_RPS = 2.75      # initial propeller rps (~ THRUST * generators' 5.5)
DURATION = 4.0            # s
STEP = 0.5                # s
MIN_TURN_DEG = 0.05       # |heading change| below this = no clear turn


def _wrap_deg(a: float) -> float:
    """
    Wrap an angle in degrees to [-180, 180).
    """

    return (a + 180.0) % 360.0 - 180.0


def _controls(si, ship_model):
    """
    (steering, propeller) controls: azipods and no propellers for TRIPOLI, else rudders and propellers.
    """

    if ship_model == ShipModel.TRIPOLI:
        return list(si.getAzipodControls()), []
    return list(si.getRudderControls()), list(si.getPropellerControls())


def run_once(ship_model: ShipModel, initial_deg: float, x: float, y: float, heading_deg: float,
             show_api: bool = False) -> float:
    """
    Heading change [deg] after DURATION s, rudder commanded to 0 after init.
    """

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
        # Read back the actuator angle after initialization, before any command.
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
    """
    CLI: run the '+' and '-' runs and print the sign convention of setInitialRudderAngle.
    """

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
