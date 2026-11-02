"""
check_model_conventions.py -- do all dynamics models follow the same conventions?

Every model (Fossen, Nomoto, UGent, the UA_* surrogates and FH) gets the SAME
framework commands from the SAME start state, in open water and calm
conditions:

    rudder +R, 0, -R    (framework rudder command [rad]; + = turn to PORT)
    thrust LOW and HIGH (rudder 0)

and each run is checked against the conventions in fw/CONVENTIONS.md:

  TURN       rudder +R turns the ship to PORT (heading increases), -R to
             STARBOARD                                              -> FAIL if not
  YAW RATE   the reported yaw rate r has the sign of the heading change
             (counter-clockwise positive)                           -> FAIL if not
  SWAY       the reported sway v matches the sideways motion of the track
             (port positive): slope of track-derived v on reported v ~ +1
                                                                    -> FAIL if < 0
  THRUST     more thrust gives a larger surge gain (first 2 s)      -> FAIL if not
  and, for information only (no pass/fail):
  STRAIGHT   heading change with rudder 0, relative to the turns
  SYMMETRY   |turn(+R)| / |turn(-R)|
  DRIFT      side of the turn the ship drifts to (a real ship: outside; the
             toy models Fossen/Nomoto: inside, see CONVENTIONS.md)

Non-FH models run through ShipSimulator.integrate (as the RL environment
does), so TURN/THRUST also check the simulator path; for them YAW RATE and
SWAY agree by construction (heading and track are built from r and v). FH
runs through FhTabularModel.step (including fh_rudder_sign) on a fresh
FhSimBridge per run, with the state read as the framework reads it.

UGent and UA_* are Tripoli-only; for B01 only UGentB01 runs (model scale).
UA_* need their checkpoints and lookup CSVs (missing ones are skipped);
UA_full's lookup wind and current are zeroed unless --keep-lookup-env. The
start position must be in open water of the FH exercise.

Usage (from src):
    python -m fw.surrogate.validation.check_model_conventions                      # Tripoli, all models
    python -m fw.surrogate.validation.check_model_conventions --ship MYZAKO_37
    python -m fw.surrogate.validation.check_model_conventions --no-fh             # without FH
    python -m fw.surrogate.validation.check_model_conventions --plot conventions.png
"""

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

_SRC_DIR = str(Path(__file__).resolve().parents[3])     # src (this file: src/fw/surrogate/validation/)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np

from fw.ships.ship_factory import B01_MODEL_SCALE, MODEL_SCALE_SHIP_MODELS, ShipModel, create_ship
from fw.ships.vessel_state import VesselState
from fw.ships.actuator_state import ActuatorState
from fw.surrogate.schema import SURROGATE_DT

logger = logging.getLogger(__name__)

# Defaults
RUDDER_DEG = 15.0             # +- rudder command [deg]
THRUST = 0.5                  # thrust for the rudder runs (normalized, as FhTabularModel / UA input)
THRUST_LOW, THRUST_HIGH = 0.2, 0.8
U0 = 3.0                      # initial surge [m/s] (full scale; model-scale ships: U0 / sqrt(lambda))
DURATION = 30.0               # [s] per run
PROPELLER_RPS_SCALE = 5.5     # FH initial rps = scale * thrust (as the generators / FhSession)
THRUST_CHECK_TIME = 2.0      # [s] surge gain compared after this time
MIN_TURN_DEG = 0.05           # |heading change| below this = "no clear turn"
MIN_SWAY = 0.005              # [m/s] sway std below this: SWAY check not meaningful
UA_PRESETS = {"UA_clean": "tripoli_clean", "UA_confined": "tripoli_confined", "UA_full": "tripoli_full"}


@dataclasses.dataclass
class Track:
    """
    One run sampled every SURROGATE_DT: t [s], x, y [m], unwrapped h [rad], u, v [m/s], r [rad/s].
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    h: np.ndarray     # unwrapped heading [rad]
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray

    @property
    def dh_deg(self) -> float:
        """
        Heading change over the whole run [deg], counter-clockwise positive.
        """

        return float(np.degrees(self.h[-1] - self.h[0]))


def _track(rows: List[tuple]) -> Track:
    """
    Track from (t, x, y, heading, u, v, r) rows; the heading is unwrapped.
    """

    a = np.asarray(rows, dtype=float)
    return Track(t=a[:, 0], x=a[:, 1], y=a[:, 2], h=np.unwrap(a[:, 3]), u=a[:, 4], v=a[:, 5], r=a[:, 6])


# ---------------------------------------------------------------- models

def build_models(ship: ShipModel, spec, keep_lookup_env: bool) -> Dict[str, Callable[[], object]]:
    """
    label -> factory of a FRESH model (each run starts from a clean model).
    """

    from fw.dynamics.fossen import Fossen
    from fw.dynamics.nomoto import Nomoto

    factories: Dict[str, Callable[[], object]] = {"Fossen": lambda: Fossen(spec), "Nomoto": lambda: Nomoto(spec)}
    if ship == ShipModel.B01:
        from fw.dynamics.ugent_b01_surrogate import UGentB01Surrogate
        # Model scale, open water (no bank provider). Fossen/Nomoto's heuristic
        # full-scale coefficients are meaningless here (Nomoto hits the 1.0 m/s
        # surge limit within 2 s at any thrust), so only the B01 model runs.
        print("B01: Fossen/Nomoto (heuristic full-scale coefficients), UA_* and UGent (Tripoli) -- skipped.")
        return {"UGentB01": lambda: UGentB01Surrogate(spec, {"ukc_percent": 379})}
    if ship != ShipModel.TRIPOLI:
        print(f"{ship.name}: UGent and UA_* are Tripoli models -- skipped.")
        return factories

    from fw.dynamics.ugent_surrogate import UGentSurrogate
    factories["UGent"] = lambda: UGentSurrogate(spec, {"ukc_percent": 363})   # no bank provider: open water

    from fw.dynamics.uantwerp_surrogate import UAntwerpSurrogate
    for label, preset in UA_PRESETS.items():
        try:
            UAntwerpSurrogate(spec, {"preset": preset}, ship_model=ship)     # loads checkpoint + lookup once
        except Exception as exc:  # missing checkpoint / lookup CSV: skip, don't stop the check
            print(f"{label}: skipped ({type(exc).__name__}: {exc})")
            continue

        def make(preset=preset):
            """
            Fresh UAntwerpSurrogate for `preset`; zero wind/current unless keep_lookup_env.
            """

            model = UAntwerpSurrogate(spec, {"preset": preset}, ship_model=ship)
            if not keep_lookup_env:
                looked_up = model.get_env

                def calm_env(pos, state=None, _get=looked_up):
                    """
                    The looked-up environment with wind_speed and current_speed set to 0.
                    """

                    env = _get(pos, state)
                    return dataclasses.replace(env, wind_speed=0.0, current_speed=0.0)

                model.get_env = calm_env
            return model

        factories[label] = make
    return factories


def run_sim_model(model, spec, start: VesselState, rudder: float, thrust: float, duration: float) -> Track:
    """
    Through ShipSimulator.integrate, as the RL environment and the comparison tests.
    """

    from fw.simulators.ship_simulator import ShipSimulator

    sim = ShipSimulator(specs=spec, dynamics=model, dt=SURROGATE_DT, wind=False, current=False)
    sim.reset(dataclasses.replace(start))
    s = sim.state
    rows = [(0.0, s.x, s.y, s.heading, s.u, s.v, s.r)]
    actuator = ActuatorState(rudder, thrust)
    for k in range(int(round(duration / SURROGATE_DT))):
        sim.integrate(actuator, SURROGATE_DT)
        s = sim.state
        rows.append(((k + 1) * SURROGATE_DT, s.x, s.y, s.heading, s.u, s.v, s.r))
    return _track(rows)


def run_fh(ship: ShipModel, spec, start: VesselState, rudder: float, thrust: float, duration: float,
           exercise_path: Optional[str]) -> Track:
    """
    FH via FhTabularModel.step on a fresh bridge; state read as the framework reads it.
    """

    from fw.dynamics.fh_tabular_model import FhTabularModel, fh_rudder_sign
    from fw.simulators.fh_sim_bridge import FhSimBridge

    kwargs = {"exercise_path": exercise_path} if exercise_path else {}
    bridge = FhSimBridge(ship_model=ship, rewind_enabled=False, **kwargs)
    bridge.initial_vessel_state = dataclasses.replace(start)
    bridge.initial_propeller_rps = float(PROPELLER_RPS_SCALE * thrust)
    bridge.initial_rudder_angle = float(fh_rudder_sign(ship) * np.degrees(rudder))
    try:
        bridge.reset()
        fh = FhTabularModel(spec, bridge)
        si = bridge.ship_interface

        def read(t):
            """
            FH state at time `t` as a Track row: heading and yaw rate converted from degrees to radians.
            """

            pos, vog = si.getShipPosition(), si.getShipVelocityOverGround()
            return (t, float(pos.x), float(pos.y), float(np.radians(si.getShipHeading())),
                    float(vog.x), float(vog.y), float(np.radians(si.getShipYawRate())))

        rows = [read(0.0)]
        for k in range(int(round(duration / SURROGATE_DT))):
            fh.step(None, rudder, thrust, SURROGATE_DT)
            rows.append(read((k + 1) * SURROGATE_DT))
        return _track(rows)
    finally:
        bridge.dispose()


# ---------------------------------------------------------------- analysis

def sway_slope(tr: Track) -> Optional[float]:
    """
    Slope of the track's sideways (port +) velocity on the reported sway; None if v barely varies.
    """

    vx, vy = np.gradient(tr.x, tr.t), np.gradient(tr.y, tr.t)
    v_track = -np.sin(tr.h) * vx + np.cos(tr.h) * vy          # port unit vector = (-sin h, cos h)
    if np.std(tr.v) < MIN_SWAY and abs(np.mean(tr.v)) < MIN_SWAY:
        return None
    a = np.vstack([tr.v, np.ones_like(tr.v)]).T
    return float(np.linalg.lstsq(a, v_track, rcond=None)[0][0])


def check_model(label: str, runs: Dict[str, Track], is_fh: bool) -> List[str]:
    """
    Returns the failures of one model and prints its lines.
    """

    fails = []
    dp, d0, dm = runs["+R"].dh_deg, runs["0"].dh_deg, runs["-R"].dh_deg

    def side(d):
        """
        "PORT", "STARBOARD" or "none" for a heading change [deg], using MIN_TURN_DEG.
        """

        return "PORT" if d > MIN_TURN_DEG else "STARBOARD" if d < -MIN_TURN_DEG else "none"

    ok_turn = dp > MIN_TURN_DEG and dm < -MIN_TURN_DEG
    print(f"  TURN      +R -> {dp:+8.2f} deg ({side(dp)}), -R -> {dm:+8.2f} deg ({side(dm)})"
          f"   {'PASS' if ok_turn else 'FAIL'}")
    if not ok_turn:
        fails.append("TURN")
        if max(abs(dp), abs(dm)) < MIN_TURN_DEG:
            print(f"            (no clear turn within the run: try a longer --duration or a larger --rudder)")

    for key in ("+R", "-R"):
        tr = runs[key]
        r_mean = float(np.mean(tr.r[1:]))
        ok = np.sign(r_mean) == np.sign(tr.dh_deg) or abs(tr.dh_deg) < MIN_TURN_DEG
        note = "" if is_fh else " (by construction)"
        print(f"  YAW RATE  {key}: mean r {np.degrees(r_mean):+7.3f} deg/s vs heading change "
              f"{tr.dh_deg:+7.2f} deg   {'PASS' if ok else 'FAIL'}{note}")
        if not ok:
            fails.append(f"YAW RATE {key}")

    slopes = {k: sway_slope(runs[k]) for k in ("+R", "-R")}
    known = [s for s in slopes.values() if s is not None]
    if known:
        ok = all(s > 0 for s in known)
        txt = ", ".join(f"{k}: {'n/a' if s is None else f'{s:+.2f}'}" for k, s in slopes.items())
        print(f"  SWAY      track vs reported v, slope {txt} (expect ~+1)   {'PASS' if ok else 'FAIL'}"
              f"{'' if is_fh else ' (by construction)'}")
        if not ok:
            fails.append("SWAY")
    else:
        print("  SWAY      sway too small to check")

    # Surge gain after THRUST_CHECK_TIME (not at the end: a fast model can
    # reach the simulator's surge limit with both thrusts).
    lo, hi = runs["thrust_low"], runs["thrust_high"]
    k = min(int(round(THRUST_CHECK_TIME / SURROGATE_DT)), len(lo.u) - 1)
    g_lo, g_hi = lo.u[k] - lo.u[0], hi.u[k] - hi.u[0]
    ok = g_hi > g_lo
    print(f"  THRUST    surge change after {THRUST_CHECK_TIME:g} s: {g_lo:+.3f} m/s (thrust {THRUST_LOW}) -> "
          f"{g_hi:+.3f} m/s (thrust {THRUST_HIGH}); after the run {lo.u[-1]:.2f} -> {hi.u[-1]:.2f} m/s"
          f"   {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("THRUST")

    turn = max(abs(dp), abs(dm))
    print(f"  STRAIGHT  rudder 0 -> {d0:+.2f} deg"
          + (f" ({abs(d0) / turn:.0%} of the turn)" if turn > MIN_TURN_DEG else ""))
    if abs(dm) > MIN_TURN_DEG:
        print(f"  SYMMETRY  |+R| / |-R| = {abs(dp) / abs(dm):.2f}")
    vp = float(np.mean(runs["+R"].v[len(runs["+R"].v) // 2:]))
    drift = "outside (starboard)" if vp < 0 else "inside (port)"
    print(f"  DRIFT     +R (port turn): mean sway in 2nd half {vp:+.3f} m/s -> {drift} of the turn")
    return fails


def plot(results: Dict[str, Dict[str, Track]], path: str) -> None:
    """
    Save every model's +R, 0 and -R tracks in the start frame (port to the left) to `path`.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    styles = {"+R": "-", "0": ":", "-R": "--"}
    for i, (label, runs) in enumerate(results.items()):
        color = f"C{i}"
        for key, ls in styles.items():
            tr = runs[key]
            # in the start frame: forward = x', port = y'
            h0 = tr.h[0]
            dx, dy = tr.x - tr.x[0], tr.y - tr.y[0]
            fwd = dx * np.cos(h0) + dy * np.sin(h0)
            port = -dx * np.sin(h0) + dy * np.cos(h0)
            ax.plot(port, fwd, ls, color=color, label=f"{label} {key}" if key == "+R" else None)
    ax.set_xlabel("to port [m] (start frame)")
    ax.set_ylabel("forward [m]")
    ax.invert_xaxis()                    # port on the LEFT, as seen from the ship
    ax.set_aspect("equal", "datalim")
    ax.grid(True)
    ax.legend(title="solid +R, dotted 0, dashed -R")
    ax.set_title("Same commands, every model (port to the left)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {path}")


def main(argv=None) -> int:
    """
    CLI entry point: run every model, print the checks and a summary, optionally plot.

    Returns:
        0 if every model passes, 1 otherwise (used as the exit code).
    """

    ap = argparse.ArgumentParser(description="Check that every dynamics model follows the framework conventions.")
    ap.add_argument("--ship", choices=[m.name for m in ShipModel], default=ShipModel.TRIPOLI.name)
    ap.add_argument("--x", type=float, default=-1559.0)
    ap.add_argument("--y", type=float, default=11990.0)
    ap.add_argument("--heading", type=float, default=-45.0, help="deg, counter-clockwise from East")
    ap.add_argument("--u0", type=float, default=None,
                    help=f"initial surge [m/s] (default {U0:g}; model-scale ships {U0 / B01_MODEL_SCALE ** 0.5:g})")
    ap.add_argument("--rudder", type=float, default=RUDDER_DEG, help="rudder command magnitude [deg]")
    ap.add_argument("--thrust", type=float, default=THRUST)
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--no-fh", action="store_true", help="skip FH (no simexe needed)")
    ap.add_argument("--exercise", default=None, help="FH .tab exercise (default: FhSimBridge's for the ship)")
    ap.add_argument("--keep-lookup-env", action="store_true", help="UA_full: keep wind/current from the lookup CSV")
    ap.add_argument("--models", nargs="*", default=None, help="only these labels (e.g. Fossen UGent FH)")
    ap.add_argument("--plot", default=None, help="save the tracks to this .png")
    args = ap.parse_args(argv)

    ship = ShipModel[args.ship]
    spec = create_ship(ship).specifications
    model_scale = ship in MODEL_SCALE_SHIP_MODELS
    if args.u0 is None:
        args.u0 = U0 / B01_MODEL_SCALE ** 0.5 if model_scale else U0
    if model_scale and not args.no_fh:
        print(f"{ship.name}: model-scale preset without an FH ship model -- FH skipped.")
        args.no_fh = True
    rudder = float(np.radians(args.rudder))
    start = VesselState(terminated=False, truncated=False, x=args.x, y=args.y,
                        heading=float(np.radians(args.heading)), u=args.u0, v=0.0, r=0.0)
    commands = {"+R": (rudder, args.thrust), "0": (0.0, args.thrust), "-R": (-rudder, args.thrust),
                "thrust_low": (0.0, THRUST_LOW), "thrust_high": (0.0, THRUST_HIGH)}

    runners: Dict[str, Callable[[float, float], Track]] = {}
    for label, factory in build_models(ship, spec, args.keep_lookup_env).items():
        runners[label] = (lambda f: lambda rud, thr: run_sim_model(f(), spec, start, rud, thr, args.duration))(factory)
    if not args.no_fh:
        runners["FH"] = lambda rud, thr: run_fh(ship, spec, start, rud, thr, args.duration, args.exercise)
    if args.models:
        runners = {k: v for k, v in runners.items() if k in args.models}

    print(f"\n{ship.name}: rudder +-{args.rudder:g} deg / 0, thrust {args.thrust:g} "
          f"(and {THRUST_LOW}/{THRUST_HIGH} at rudder 0), u0 {args.u0:g} m/s, {args.duration:g} s, "
          f"start ({args.x:g}, {args.y:g}) heading {args.heading:g} deg")

    results: Dict[str, Dict[str, Track]] = {}
    failures: Dict[str, List[str]] = {}
    for label, run in runners.items():
        print(f"\n{label}")
        try:
            results[label] = {key: run(rud, thr) for key, (rud, thr) in commands.items()}
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            failures[label] = ["ERROR"]
            continue
        failures[label] = check_model(label, results[label], is_fh=(label == "FH"))

    # One-glance cross-model comparison.
    print("\nSummary (heading change [deg] after the run, +R / 0 / -R; final surge at +R):")
    for label, runs in results.items():
        print(f"  {label:12s} {runs['+R'].dh_deg:+8.2f} {runs['0'].dh_deg:+8.2f} {runs['-R'].dh_deg:+8.2f}"
              f"   u {runs['+R'].u[-1]:.2f} m/s   {'OK' if not failures[label] else 'FAIL: ' + ', '.join(failures[label])}")

    if args.plot and results:
        plot(results, args.plot)

    bad = [k for k, v in failures.items() if v]
    print("\n=> " + ("ALL MODELS FOLLOW THE CONVENTIONS." if not bad else f"FAILED: {', '.join(bad)}"))
    return 1 if bad else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    # FhSimBridge loggers are INFO and propagate; show only warnings and errors.
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.WARNING)
    sys.exit(main())
