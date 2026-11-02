"""
One-step accuracy of UA_full on its own dataset, with the metric of section 4b
of the wind report (RMS error / RMS FH signal, mean yaw bias).

Is UA_full's large one-step error in the wind test (e.g. 132 % RMS yaw
acceleration error, -0.0052 deg/s^2 yaw bias without wind) also present on the
training data in similar situations (rudder ~0, small yaw rate, open water,
low wind and current)? If yes, the model itself is not accurate enough there;
if not, the wind test feeds inputs differently or outside the training
distribution. The dataset includes the training split, so these numbers are
an optimistic lower bound.

Rows are fed to SurrogatePredictor as stored (cols 0-36: the raw 37-column
full_env input, as UAntwerpSurrogate builds it for tripoli_full); targets are
cols 37-39 (delta u, v, r over 0.1 s).

Usage (from src):
    python -m fw.surrogate.validation.check_ua_full_onestep_on_data <dataset.csv> [--checkpoint PATH] [--sample N]

--checkpoint: an existing path is used as given, otherwise resolved like the
presets (relative to fw/checkpoints). Default: the tripoli_full preset's
checkpoint. The CSV has no ship identity, so use the dataset of the
checkpoint's ship (printed).
"""

import argparse
import math
import sys
import time
from pathlib import Path

# Make `import fw...` work when run as a plain script.
_SRC = Path(__file__).resolve().parents[3]     # src (this file: src/fw/surrogate/validation/)
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
    """
    True wind speed [m/s] per row: apparent wind minus (u, -v), with sway v port positive.
    """

    # (ahead, starboard) components: wind angle positive from starboard, v port
    # positive (same as model_comparison_wind and analyze_wind_training_data).
    ws, wd, u, v = a[:, COL_WS], a[:, COL_WD], a[:, COL_U], a[:, COL_V]
    return np.hypot(ws * np.cos(wd) - u, ws * np.sin(wd) + v)


def _count_rows(path: Path) -> int:
    """
    Number of non-empty lines of the CSV (one cheap pass, for the sampling rate).
    """

    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _load(path: Path, sample: int, seed: int):
    """
    Random row sample (about `sample` rows) + ALL test-like rows (they are rare), chunked to save memory.
    """

    rng = np.random.default_rng(seed)
    n_cols = len(pd.read_csv(path, header=None, nrows=1).columns)
    if n_cols != N_IN + len(TARGET_COLS) + 1:
        raise SystemExit(f"{path} has {n_cols} columns; a full_env dataset CSV has "
                         f"{N_IN + len(TARGET_COLS) + 1} (raw row, 3 targets, episode_id).")
    rate = min(1.0, sample / max(1, _count_rows(path)))
    keep_rand, keep_like, n_total = [], [], 0
    for chunk in pd.read_csv(path, header=None, dtype=np.float32, chunksize=500_000):
        a = chunk.to_numpy()
        n_total += len(a)
        keep_rand.append(a[rng.random(len(a)) < rate])
        like = ((np.abs(a[:, COL_RUDDER]) < RUDDER_TOL) & (np.abs(a[:, COL_R]) < YAW_RATE_TOL)
                & (a[:, RAY_COLS].max(axis=1) < MAX_RAY))
        keep_like.append(a[like])
        print(f"  read {n_total:,} rows", end="\r", flush=True)
    print()
    return np.concatenate(keep_rand), np.concatenate(keep_like), n_total


def _predict(predictor, x: np.ndarray) -> np.ndarray:
    """
    Batch predict if SurrogatePredictor supports it (verified against predict_single), else loop.
    """

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
    """
    Print RMS error as % of RMS FH signal per target, and the mean yaw-acceleration bias.
    """

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
    """
    CLI entry point: print the metrics for a random sample and for test-like row subsets.
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint path (existing path as given, else relative to fw/checkpoints); "
                         "default: the tripoli_full preset's checkpoint")
    ap.add_argument("--sample", type=int, default=50_000, help="random rows for the 'all rows' line")
    ap.add_argument("--max-like", type=int, default=100_000, help="cap on test-like rows to predict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thrust", type=float, default=5.0 / 6.0,
                    help="thrust action of the wind test (model_comparison_wind: min(1, 5/6) = 0.833)")
    ap.add_argument("--thrust-tol", type=float, default=0.1)
    args = ap.parse_args()

    from fw.surrogate.inference_surrogate import SurrogatePredictor
    from fw.surrogate.presets import get_preset, resolve_checkpoint
    from fw.surrogate.schema import FULL_ENV
    if args.checkpoint is None:
        ckpt = resolve_checkpoint(get_preset("tripoli_full").checkpoint)
    else:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            ckpt = resolve_checkpoint(args.checkpoint)
        if not ckpt.exists():      # e.g. "checkpoints/<name>.pth": try the bare name
            ckpt = resolve_checkpoint(Path(args.checkpoint).name)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt} (cwd {Path.cwd()}); pass --checkpoint <path>.")
    if not Path(args.csv).exists():
        raise SystemExit(f"Dataset not found: {args.csv}")
    print(f"Checkpoint: {ckpt.resolve()}")
    predictor = SurrogatePredictor(str(ckpt))
    print(f"Checkpoint ship: {predictor.ship_model} -- the dataset must be of the same ship.")
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
