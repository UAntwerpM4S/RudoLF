"""
Standalone FH wind-field diagnostic -- NO FhSimBridge, NO FhSession.

Replicates sim_fh_dataset_generator.run_simulation()'s FH calls line for line
(Config -> Exercise -> ship_config -> MathModel.Initialize -> enableBridge ->
wind-field transform -> simulateSeconds), each probe in its own fresh worker
process like the generator's pool, and prints per probe:
  - the transform FH reports back right after setFieldTransform()
  - the transform FH reports back after the first simulateSeconds()
  - WindState at the ship after the settle, after 1 s and after 10 s

One run answers the open question:
  A) wind changes here            -> the generator's FH calls work at this
                                     start pose; the difference is in
                                     FhSimBridge/FhSession (I'll diff further)
  B) read-back == requested, wind
     never changes                -> the transform is stored but does not
                                     drive the wind at this pose / with this
                                     exercise (also with the generator's code)
  C) read-back != requested       -> FH does not accept the transform at all

Usage: set START_* below (your wind-test start pose), run with the RudoLF venv.
SIMEXE_REPO_FOLDER must be set, as for the generator.
"""

import multiprocessing
import os
import sys
from pathlib import Path

START_X = -1559.0
START_Y = 11990.0
START_HEADING_DEG = -45.0      # passed to setInitialHeading as-is (FH degrees)
START_SURGE = 5.0
PROPELLER_RPS = 0.0            # 0 = leave as in generator with action[1]=0; only wind matters here

TABS = [
    "DDShip_Tripoli_scenario_windnocurrent.tab",
    "DDShip_Tripoli_scenario_windcurrent.tab",     # the generator's exercise
]
PROBES = [(0.0, 1.0), (90.0, 1.0), (0.0, 2.0), (0.0, 0.0)]   # (Rotation, Scale)

MATH_MODEL_FREQUENCY = 10.0    # generator's DEFAULT_MATH_MODEL_FREQUENCY
INITIALIZATION_TIME = 0.1      # generator's INITIALIZATION_TIME

SIMEXE = os.environ.get("SIMEXE_REPO_FOLDER", "")
LIB_PATH = str(Path(SIMEXE) / "exe" / "installed")
CONFIG_PATH = str(Path(SIMEXE) / "config" / "simXdrive.config.xml")
TRA_DIR = Path(SIMEXE) / "database" / "areas" / "ScheldeSaeftinge_23_002" / "invoer" / "tra"
OUTPUT_ROOT = Path.home() / "Desktop" / "testExercise" / "wind_diag"


def _fmt_transform(tr) -> str:
    """One-line text of an FH wind-field transform (Rotation, Scale, Offset)."""
    return f"R={tr.Rotation:g} S={tr.Scale:g} O={tr.Offset:g}"


def probe(tab: str, rotation: float, scale: float) -> str:
    """One FH session, generator-style. Runs in a worker process; returns a report line."""
    if LIB_PATH not in sys.path:
        sys.path.append(LIB_PATH)
    from sim import Config, Exercise, MathModel  # noqa: E402

    out_dir = OUTPUT_ROOT / f"{Path(tab).stem}_R{rotation:g}_S{scale:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config(CONFIG_PATH)
    config.SetOutputDir(str(out_dir))
    config.SetMathModelFrequency(MATH_MODEL_FREQUENCY)
    config.SetRewindEnabled(False)
    config.ClearConnections()

    exercise = Exercise(str(TRA_DIR / tab), config)
    ship_config = exercise.getShipConfig()
    ship_config.setInitialPosition(START_X, START_Y)
    ship_config.setInitialPropellerRps(float(PROPELLER_RPS))
    ship_config.setInitialRudderAngle(0.0)
    ship_config.setInitialHeading(float(START_HEADING_DEG))
    ship_config.setInitialSurgeVelocity(float(START_SURGE))
    ship_config.setInitialSwayVelocity(0.0)
    ship_config.setInitialYawVelocity(0.0)

    math_model = MathModel()
    if not math_model.Initialize(config, exercise):
        return f"{tab:45s} R={rotation:5g} S={scale:3g}: MathModel.Initialize() FAILED"
    try:
        math_model.enableBridge()
        environment = math_model.getEnvironmentInterface()
        control = environment.getWindFieldControl()
        if control is None:
            return f"{tab:45s} R={rotation:5g} S={scale:3g}: getWindFieldControl() is None"

        before = _fmt_transform(control.getFieldTransform())
        tr = control.getFieldTransform()
        tr.Rotation = float(rotation)
        tr.Offset = 0.0
        tr.Scale = float(scale)
        control.setFieldTransform(tr)
        after_set = _fmt_transform(control.getFieldTransform())

        math_model.simulateSeconds(INITIALIZATION_TIME)
        after_settle = _fmt_transform(control.getFieldTransform())
        ship = math_model.getShipInterface(0)
        w0 = ship.getWindState()
        math_model.simulateSeconds(1.0)
        w1 = ship.getWindState()
        math_model.simulateSeconds(9.0)
        w10 = ship.getWindState()
        return (f"{tab:45s} req R={rotation:5g} S={scale:3g} | default [{before}] | "
                f"after set [{after_set}] | after settle [{after_settle}] | wind "
                f"t0={w0.Speed:.3f}@{w0.Direction:.2f}  t1={w1.Speed:.3f}@{w1.Direction:.2f}  "
                f"t10={w10.Speed:.3f}@{w10.Direction:.2f}")
    finally:
        try:
            math_model.Terminate()
            math_model.Dispose()
        except Exception:
            pass


def main() -> None:
    """
    Entry point: run every (exercise, rotation, scale) probe in its own spawned process and print
    one result line per probe.
    """

    if not SIMEXE:
        raise SystemExit("SIMEXE_REPO_FOLDER is not set.")
    jobs = [(tab, r, s) for tab in TABS for (r, s) in PROBES]
    # maxtasksperchild=1: every probe in a fresh process, like the generator's pool.
    with multiprocessing.get_context("spawn").Pool(processes=1, maxtasksperchild=1) as pool:
        for line in pool.starmap(probe, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
