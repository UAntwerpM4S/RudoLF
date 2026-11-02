"""
Probe FH's actuator response at the start of a dataset episode.

Do FH's actuators (azipod direction, propeller rpm, rudder angle) start at the
commanded value, or do they first have to get there? The surrogate only sees
the command, so a lag makes the first seconds of every episode unexplainable
(see time_error_diagnostic.py).

FH is initialized as in sim_fh_dataset_generator.run_simulation (initial
pose, speeds, rudder angle and propeller rps from the action, same per-step
commands). Every --every seconds the ship state and all parameterless
get*/is* methods of the ship interface and the control objects (found by
introspection) are printed and written to a CSV.

Usage (from src; runs FH in this process, takes about as long as the
simulated time):

    python -m fw.surrogate.validation.probe_actuator_lag --ship-model TRIPOLI ^
        --rudder 0.5 --thrust 0.8 --surge 2.0 --seconds 40

Default position/heading: mid-river at the model_comparison seed point,
heading along the river.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile

import numpy as np

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fw.dynamics.fh_tabular_model import fh_rudder_sign
from fw.ships.ship_factory import ShipModel
import fw.surrogate.sim_fh_dataset_generator as gen


def _getters(obj) -> List[str]:
    """
    Names of the public get*/is* methods of a (pythonnet) control object.
    """

    names = []
    for name in dir(obj):
        if (name.startswith("get") or name.startswith("Get") or name.startswith("is")) and callable(getattr(obj, name, None)):
            names.append(name)
    return sorted(names)


def _read(obj, names: Sequence[str]) -> Dict[str, str]:
    """
    Call every parameterless getter; values as short strings (errors are skipped).
    """

    out = {}
    for name in names:
        try:
            value = getattr(obj, name)()
        except Exception:
            continue                      # needs arguments, or not readable now
        try:
            out[name] = f"{float(value):.4g}"
        except (TypeError, ValueError):
            text = str(value)
            if len(text) <= 40:
                out[name] = text
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Command-line entry point; returns the exit code.
    """

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ship-model", choices=[m.name for m in gen.SHIP_PROFILES], default="TRIPOLI")
    p.add_argument("--configuration", choices=sorted(gen.CONFIGURATIONS), default="confined")
    p.add_argument("--rudder", type=float, default=0.5, help="rudder_action in [-1, 1] (default 0.5)")
    p.add_argument("--thrust", type=float, default=0.8, help="thrust_action in [-1, 1] (default 0.8)")
    p.add_argument("--surge", type=float, default=2.0, help="initial surge [m/s] (default 2.0)")
    p.add_argument("--x", type=float, default=2272.5)
    p.add_argument("--y", type=float, default=-64.3)
    p.add_argument("--heading-deg", type=float, default=55.287,
                   help="initial heading, framework convention (0 = East, CCW) [deg]")
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--every", type=float, default=0.5, help="print interval [s] (default 0.5)")
    p.add_argument("--csv", type=Path, default=Path("actuator_lag.csv"))
    args = p.parse_args(argv)

    ship_model = ShipModel[args.ship_model]
    ship = gen.ship_profile(ship_model)
    rudder_max = gen.resolve_rudder_max_deg(ship_model, args.configuration)
    sign = fh_rudder_sign(ship_model)
    action = np.array([args.rudder, args.thrust], dtype=np.float64)

    sim_config = gen.default_sim_config()
    if sim_config.lib_path not in sys.path:
        sys.path.append(sim_config.lib_path)
    from sim import Config, Exercise, MathModel

    exercise_path, _ = gen.exercise_paths(ship_model, args.configuration)
    out_dir = Path(tempfile.mkdtemp(prefix="actuator_probe_"))
    config = Config(sim_config.config_path)
    config.SetOutputDir(str(out_dir))
    config.SetMathModelFrequency(gen.DEFAULT_MATH_MODEL_FREQUENCY)
    config.SetRewindEnabled(False)
    config.ClearConnections()
    exercise = Exercise(str(exercise_path), config)
    sc = exercise.getShipConfig()
    setters = sorted(n for n in dir(sc) if n.lower().startswith("set") and callable(getattr(sc, n, None)))
    print(f"ship config setters: {', '.join(setters)}")
    # As run_simulation:
    sc.setInitialPosition(args.x, args.y)
    sc.setInitialPropellerRps(float(gen._PROPELLER_RPS_SCALE * action[1]))
    sc.setInitialRudderAngle(float(sign * rudder_max * action[0]))
    sc.setInitialHeading(float(gen.env_rad_to_fh_heading_deg(np.radians(args.heading_deg))))
    sc.setInitialSurgeVelocity(float(args.surge))
    sc.setInitialSwayVelocity(0.0)
    sc.setInitialYawVelocity(0.0)

    mm = MathModel()
    try:
        if not mm.Initialize(config, exercise):
            print("MathModel initialization failed.")
            return 1
        mm.enableBridge()
        mm.simulateSeconds(gen.INITIALIZATION_TIME)
        si = mm.getShipInterface(0)
        if ship.steering == "azipod":
            controls = {"azipod": list(si.getAzipodControls())}
            command = f"setDesiredDirection({sign * action[0] * rudder_max:+.2f} deg) + setEngineLeverValue({action[1]:+.2f})"
        else:
            controls = {"rudder": list(si.getRudderControls()), "propeller": list(si.getPropellerControls())}
            command = f"rudder setControlValue({sign * action[0]:+.3f}) + propeller setEngineLeverValue({action[1]:+.2f})"
        getter_names = {kind: _getters(objs[0]) for kind, objs in controls.items() if objs}
        # Ship-interface getters that return a number (roll, pitch, heave, ... if FH offers them).
        ship_getters = [n for n in _getters(si) if n in _read(si, [n])]
        print(f"ship interface numeric getters: {', '.join(ship_getters)}")
        print(f"{ship_model.name}: rudder_action {action[0]:+.3f} (= {action[0] * rudder_max:+.1f} deg), "
              f"thrust_action {action[1]:+.2f}; initial rps {gen._PROPELLER_RPS_SCALE * action[1]:+.2f}, "
              f"initial rudder angle {sign * rudder_max * action[0]:+.2f} (FH sign); command each step: {command}")
        for kind, names in getter_names.items():
            print(f"{kind} getters ({len(controls[kind])} control(s)): {', '.join(names)}")

        rows = []
        steps = int(round(args.seconds / 0.1))
        every = max(1, int(round(args.every / 0.1)))
        for k in range(steps + 1):
            if k % every == 0:
                vog = si.getShipVelocityOverGround()
                row = {"t": f"{k * 0.1:.1f}", "surge": f"{vog.x:.4f}", "sway": f"{vog.y:.4f}",
                       "yaw_rate_deg_s": f"{si.getShipYawRate():.4f}"}
                for name, value in _read(si, ship_getters).items():
                    row[f"ship.{name}"] = value
                for kind, objs in controls.items():
                    for i, obj in enumerate(objs):
                        for name, value in _read(obj, getter_names[kind]).items():
                            row[f"{kind}{i}.{name}"] = value
                rows.append(row)
                print("  ".join(f"{key}={value}" for key, value in row.items()), flush=True)
            if k == steps:
                break
            if ship.steering == "azipod":
                for az in controls["azipod"]:
                    az.setDesiredDirection(float(sign * action[0] * rudder_max))
                    az.setEngineLeverValue(float(action[1]))
            else:
                for r in controls["rudder"]:
                    r.setControlValue(float(sign * action[0]))
                for pr in controls["propeller"]:
                    pr.engage()
                    pr.setEngineLeverValue(float(action[1]))
            mm.simulateSeconds(0.1)
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")
    finally:
        for method in ("Terminate", "Dispose"):
            try:
                getattr(mm, method)()
            except Exception as exc:
                print(f"Warning: {method}() failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
