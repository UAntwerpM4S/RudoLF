"""
One-step accuracy of UA_full ON ITS OWN DATASET, with the same metric as section
4b of the wind report (RMS error / RMS FH signal, and the mean yaw bias).

Question it answers: is UA_full's large one-step error in the wind test
(e.g. 132 % RMS yaw-acceleration error, -0.0052 deg/s^2 yaw bias without wind)
  - also present on the training data in the same kind of situation
    (rudder ~0, small yaw rate, open water, low wind, low current)
    -> the model itself is not accurate enough there (capacity / training /
       loss weighting), or
  - absent on the training data
    -> something in the wind test differs from the training rows
       (an input that is outside the training distribution, or fed differently).

The dataset contains the training split too, so these numbers are a LOWER bound
for the model's real error (optimistic). If they are already bad, that settles it.

Rows are fed to SurrogatePredictor exactly as stored (cols 0..36 = the raw
37-column full_env input vector, the same thing UAntwerpSurrogate builds for the
tripoli_full preset);
targets are cols 37..39 (delta u, v, r over 0.1 s).

Usage (from src/fw/surrogate, with the RudoLF venv):
    python check_ua_full_onestep_on_data.py <dataset.csv> [--checkpoint PATH] [--sample N]
"""

import argparse
import math
import sys
import time
from pathlib import Path

# Make `import fw...` work when run as a plain script from src/fw/surrogate
# (this file lives in src/fw/surrogate -> src is two levels up).
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

DT = 0.1
N_IN = 37
COL_R, COL_RUDDER, COL_THRUST = 5, 11, 12
COL_U, COL_V, COL_WS, COL_WD, COL_CS = 3, 4, 7, 8, 9
RAY_COLS = list(range(13, 37))
TARGET_COLS = [37, 38, 39]
COL_EP = 40

# "Test-like" = what the wind test looks like: rudder ~0, nearly straight, open water.
RUDDER_TOL = 0.02          # normalized rudder action (|.| < 0.02 = < 0.7 deg)
YAW_RATE_TOL = 0.005       # rad/s (0.29 deg/s)
MAX_RAY = 0.05             # all ray proximities below this = no bank nearby
LOW_WIND = 2.0             # m/s true wind
LOW_CURRENT = 0.1          # m/s


def _true_wind_speed(a):
    # true = apparent - (u, -v) in (ahead, starboard) components: WindState angle
    # positive = from starboard, sway v PORT positive (dataset convention since
    # 2026-10-19; model_comparison_wind: TRUE_WIND_SWAY_SIGN = -1;
    # analyze_wind_training_data uses the same formula).
    ws, wd, u, v = a[:, COL_WS], a[:, COL_WD], a[:, COL_U], a[:, COL_V]
    return np.hypot(ws * np.cos(wd) - u, ws * np.sin(wd) + v)


def _load(path: Path, sample: int, seed: int):
    """Random row sample + ALL test-like rows (they are rare), chunked to save memory."""
    rng = np.random.default_rng(seed)
    keep_rand, keep_like, n_total = [], [], 0
    for chunk in pd.read_csv(path, header=None, dtype=np.float32, chunksize=500_000):
        a = chunk.to_numpy()
        n_total += len(a)
        keep_rand.append(a[rng.random(len(a)) < sample / 6.5e6])  # ~`sample` rows for a ~6.5 M-row dataset
        like = ((np.abs(a[:, COL_RUDDER]) < RUDDER_TOL) & (np.abs(a[:, COL_R]) < YAW_RATE_TOL)
                & (a[:, RAY_COLS].max(axis=1) < MAX_RAY))
        keep_like.append(a[like])
        print(f"  read {n_total:,} rows", end="\r", flush=True)
    print()
    return np.concatenate(keep_rand), np.concatenate(keep_like), n_total


def _predict(predictor, x: np.ndarray) -> np.ndarray:
    """Batch predict if SurrogatePredictor supports it (verified against predict_single), else loop."""
    singles = np.array([predictor.predict_single(x[i]) for i in range(min(5, len(x)))], dtype=np.float64)
    try:
        batch = np.asarray(predictor.predict(x), dtype=np.float64)
        if batch.shape == (len(x), 3) and np.allclose(batch[:len(singles)], singles, rtol=1e-4, atol=1e-7):
            return batch
        print(f"  predict() returned shape {batch.shape} or values != predict_single -> using predict_single loop")
    except Exception as exc:  # noqa: BLE001 - fall back to the known-good per-row path
        print(f"  predict() on a batch failed ({type(exc).__name__}: {exc}) -> using predict_single loop")
    out = np.empty((len(x), 3))
    t0 = time.time()
    for i in range(len(x)):
        out[i] = predictor.predict_single(x[i])
        if i % 5000 == 0 and i:
            print(f"  {i:,}/{len(x):,} rows ({(time.time() - t0) / i * (len(x) - i):.0f} s left)", end="\r", flush=True)
    print()
    return out


def _report(name: str, a: np.ndarray, pred: np.ndarray) -> None:
    n = len(a)
    if n < 50:
        print(f"{name:52s} n={n:>8,}  (too few rows)")
        return
    y = a[:, TARGET_COLS].astype(np.float64)
    err = pred - y
    rms_err = np.sqrt(np.mean(err ** 2, axis=0))
    rms_sig = np.sqrt(np.mean(y ** 2, axis=0))
    ratio = np.where(rms_sig > 0, 100.0 * rms_err / np.maximum(rms_sig, 1e-30), np.nan)
    bias_dr = math.degrees(float(np.mean(err[:, 2])) / DT)
    eps = len(np.unique(a[:, COL_EP]))
    print(f"{name:52s} n={n:>8,} ({eps:>5,} eps) | RMS err/RMS FH: du {ratio[0]:6.1f}%  dv {ratio[1]:6.1f}%  "
          f"dr {ratio[2]:6.1f}% | mean dr bias {bias_dr:+.4f} deg/s^2 | RMS FH dr {math.degrees(rms_sig[2] / DT):.4f} deg/s^2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--checkpoint", default="checkpoints/best_surrogate_tripoli_full.pth")
    ap.add_argument("--sample", type=int, default=50_000, help="random rows for the 'all rows' line")
    ap.add_argument("--max-like", type=int, default=100_000, help="cap on test-like rows to predict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thrust", type=float, default=5.0 / 6.0,
                    help="thrust action of the wind test (model_comparison_wind: min(1, 5/6) = 0.833)")
    ap.add_argument("--thrust-tol", type=float, default=0.1)
    args = ap.parse_args()

    from fw.surrogate.inference_surrogate import SurrogatePredictor
    from fw.surrogate.schema import FULL_ENV
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute() and not ckpt.exists() and (Path(__file__).resolve().parent / ckpt).exists():
        ckpt = Path(__file__).resolve().parent / ckpt
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt} (cwd {Path.cwd()}); pass --checkpoint <path>.")
    if not Path(args.csv).exists():
        raise SystemExit(f"Dataset not found: {args.csv}")
    print(f"Checkpoint: {ckpt.resolve()}")
    predictor = SurrogatePredictor(str(ckpt))
    if predictor.scenario != FULL_ENV:
        raise SystemExit(f"{ckpt} is a '{predictor.scenario.name}' checkpoint; this check needs a "
                         f"'{FULL_ENV.name}' (UA_full) checkpoint.")

    print(f"Reading {args.csv} ...")
    rand, like, n_total = _load(Path(args.csv), args.sample, args.seed)
    if len(like) > args.max_like:
        like = like[np.random.default_rng(args.seed).choice(len(like), args.max_like, replace=False)]
    print(f"Rows: {n_total:,}; random sample {len(rand):,}; test-like rows used {len(like):,} "
          f"(rudder |a|<{RUDDER_TOL}, |r|<{YAW_RATE_TOL} rad/s, all rays <{MAX_RAY})")

    print("Predicting random sample ...")
    p_rand = _predict(predictor, rand[:, :N_IN])
    print("Predicting test-like rows ...")
    p_like = _predict(predictor, like[:, :N_IN]) if len(like) else np.empty((0, 3))

    tw, cs = _true_wind_speed(like), like[:, COL_CS]
    print("\nSame metric as section 4b of the wind report (wind test, UA_full, no wind: dv 69 %, dr 132 %, "
          "dr bias -0.0052 deg/s^2):\n")
    _report("all rows (random sample)", rand, p_rand)
    _report("test-like", like, p_like)
    _report(f"test-like, true wind < {LOW_WIND:g} m/s", like[tw < LOW_WIND], p_like[tw < LOW_WIND])
    _report(f"test-like, current < {LOW_CURRENT:g} m/s", like[cs < LOW_CURRENT], p_like[cs < LOW_CURRENT])
    m = (tw < LOW_WIND) & (cs < LOW_CURRENT)
    _report(f"test-like, wind < {LOW_WIND:g} AND current < {LOW_CURRENT:g}", like[m], p_like[m])
    for lo, hi in ((5, 10), (10, 15), (15, 25)):
        mm = (tw >= lo) & (tw < hi)
        _report(f"test-like, true wind {lo}-{hi} m/s", like[mm], p_like[mm])
    tm = np.abs(like[:, COL_THRUST] - args.thrust) < args.thrust_tol
    _report(f"test-like, thrust {args.thrust:.2f}+-{args.thrust_tol:g}", like[tm], p_like[tm])
    _report(f"test-like, thrust {args.thrust:.2f}+-{args.thrust_tol:g}, wind < {LOW_WIND:g}",
            like[tm & (tw < LOW_WIND)], p_like[tm & (tw < LOW_WIND)])
    th_all = rand[:, COL_THRUST]
    print(f"\nThrust {args.thrust:.2f}+-{args.thrust_tol:g}: {100.0 * np.mean(np.abs(th_all - args.thrust) < args.thrust_tol):.2f}% "
          f"of all rows (random sample), {int(tm.sum()):,} test-like rows in {len(np.unique(like[tm, COL_EP])):,} episodes.")
    print("Thrust action, all rows (random sample): " + ", ".join(
        f"p{q}={np.percentile(th_all, q):.2f}" for q in (5, 25, 50, 75, 95)))
    if len(like):
        th = like[:, COL_THRUST]
        print(f"\nThrust action in the test-like rows: p5 {np.percentile(th, 5):.2f}, median {np.median(th):.2f}, "
              f"p95 {np.percentile(th, 95):.2f} (compare with the wind test's thrust).")


if __name__ == "__main__":
    main()
