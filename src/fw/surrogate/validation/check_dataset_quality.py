#!/usr/bin/env python3
"""
Quality checks of a generated surrogate dataset.

Reads the .npz splits written by generate_ship_data.py (X, Y, episode_id,
identity) and reports, for any ship / configuration:

  1. Identity: same scenario, ship, rudder scale, sway convention and ray
     layout in all splits (schema.read_npz_identity).
  2. Split hygiene: rows and episodes per split, no episode leakage between
     splits, all values finite, no (near-)constant input column.
  3. Design: one constant (rudder, thrust) action per episode within [-1, 1];
     at most horizon / dt rows per episode.
  4. Coverage of rudder, thrust, surge, sway, yaw rate, UKC, heading, the
     share of straight episodes and of rows with a bank in ray range.
  5. Targets: per-step deltas and robust outliers (|delta - median| > K * MAD).
  6. Rudder sign: corr(rudder_action, delta_yaw_rate) must be > 0 for every
     ship (positive rudder turns to port), on rows with forward speed and thrust.
  7. Exact duplicate rows.

Writes <prefix>_quality.png (histograms) next to the files. Needs numpy and
matplotlib only (no torch, no FH). Exit code 1 if a hard check fails.

Usage (from src):
    python -m fw.surrogate.validation.check_dataset_quality data/myzako_confined/ship_dynamics_dataset
    python -m fw.surrogate.validation.check_dataset_quality <prefix> --dt 0.1 --horizon 80
"""

import sys
import argparse
import numpy as np

from pathlib import Path
from fw.surrogate.schema import TARGET_NAMES, get_scenario, read_npz_identity

SPLITS = ("train", "val", "test")
STD_FLOOR = 1e-6           # input std below this = constant column
OUTLIER_K = 12.0           # robust outlier threshold, in MADs
MIN_SURGE_FOR_SIGN = 1.0   # m/s; the rudder has little effect below this
MIN_ABS_RUDDER = 0.1       # rows with a clearly non-zero rudder for the sign check
STRAIGHT_ABS_RUDDER = 0.03  # |rudder_action| below this: a "straight" episode (rudder ~0)


def _load(prefix: str):
    """
    Load the three splits.

    Returns:
        {split: (X, Y, episode_id, identity)} for the splits that exist.
    """

    out = {}
    for split in SPLITS:
        path = Path(f"{prefix}_{split}.npz")
        if not path.is_file():
            print(f"!! {path} not found")
            continue
        with np.load(path, allow_pickle=False) as z:
            ident = read_npz_identity(z, str(path))
            out[split] = (np.asarray(z["X"]), np.asarray(z["Y"]), np.asarray(z["episode_id"]).astype(np.int64), ident)
    return out


def _pct(values: np.ndarray, qs=(0, 1, 50, 99, 100)) -> str:
    """
    Percentiles of `values` as one compact text.
    """

    return " ".join(f"p{q}={v:.4g}" for q, v in zip(qs, np.atleast_1d(np.percentile(values, qs))))


def _share_pct(mask) -> float:
    """
    Share of True values in `mask` [%] (as a Python float).
    """

    return 100.0 * float(np.mean(mask))


def _hist_text(values: np.ndarray, edges) -> str:
    """
    Share of `values` per bin of `edges`, as percentages (np.histogram: the last bin is closed, [lo, hi]).
    """

    counts, _ = np.histogram(values, bins=edges)
    total = max(len(values), 1)
    last = len(counts) - 1
    return "  ".join(f"[{lo:g},{hi:g}{']' if i == last else ')'}:{100 * c / total:.1f}%"
                     for i, (lo, hi, c) in enumerate(zip(edges[:-1], edges[1:], counts)))


def main(argv=None) -> int:
    """
    CLI entry point: run all checks on `<prefix>_{train,val,test}.npz`, print
    the report and save the histogram PNG.

    Returns:
        0 if every hard check passes, 1 otherwise (used as the exit code).
    """

    ap = argparse.ArgumentParser(description="Quality checks of a generated surrogate dataset (.npz splits).")
    ap.add_argument("prefix", help="path prefix of the splits, e.g. data/myzako_confined/ship_dynamics_dataset")
    ap.add_argument("--dt", type=float, default=0.1, help="time step of the data [s]")
    ap.add_argument("--horizon", type=float, default=80.0, help="--horizon used for generation [s]")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    data = _load(args.prefix)
    if not data:
        print("No splits found.")
        return 1
    failures = []

    # ---- 1. identity ---------------------------------------------------------
    idents = {s: d[3] for s, d in data.items()}
    first = next(iter(idents.values()))
    print(f"== 1. Identity: {first}")
    for s, ident in idents.items():
        if ident != first:
            failures.append(f"identity of {s} differs: {ident}")
    scenario = get_scenario(first.scenario)
    cols = list(scenario.raw_columns)
    c = {name: i for i, name in enumerate(cols)}

    # ---- 2. split hygiene ------------------------------------------------------
    print("\n== 2. Splits")
    eps = {}
    for s, (x, y, ep, _) in data.items():
        eps[s] = set(np.unique(ep).tolist())
        print(f"  {s:5s}: {len(x):>10,} rows  {len(eps[s]):>7,} episodes  X{tuple(x.shape)} Y{tuple(y.shape)}")
        if x.shape[1] != scenario.raw_input_dim or y.shape[1] != len(TARGET_NAMES):
            failures.append(f"{s}: X/Y width {x.shape[1]}/{y.shape[1]}, expected {scenario.raw_input_dim}/3")
        bad = int((~np.isfinite(x)).sum() + (~np.isfinite(y)).sum())
        if bad:
            failures.append(f"{s}: {bad} NaN/inf values")
        if len(x) == 0:
            failures.append(f"{s}: empty split")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in eps and b in eps and eps[a] & eps[b]:
            failures.append(f"episode leakage: {len(eps[a] & eps[b])} episode(s) in both {a} and {b}")
    n_eps = sum(len(e) for e in eps.values())
    print("  episode fractions: " + ", ".join(f"{s} {100 * len(e) / max(n_eps, 1):.1f}%" for s, e in eps.items()))

    # float32 as stored (a full run has millions of rows); statistics per column.
    x_all = np.concatenate([d[0] for d in data.values()]).astype(np.float32, copy=False)
    y_all = np.concatenate([d[1] for d in data.values()]).astype(np.float32, copy=False)
    ep_all = np.concatenate([d[2] for d in data.values()])
    if "train" in data:
        x_train = data["train"][0]
        const = [cols[i] for i in scenario.input_col_indices()
                 if np.std(x_train[:, i], dtype=np.float64) < STD_FLOOR]
        if const:
            failures.append(f"(near-)constant input column(s) in train: {const}")

    # ---- 3. design: one action per episode, lengths --------------------------
    print("\n== 3. Episode design")
    order = np.argsort(ep_all, kind="stable")
    ep_sorted = ep_all[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ep_sorted)) + 1]
    lengths = np.diff(np.r_[starts, len(ep_sorted)])
    for name in ("rudder_action", "thrust_action"):
        v = x_all[order, c[name]]
        spread = np.maximum.reduceat(v, starts) - np.minimum.reduceat(v, starts)
        n_var = int((spread > 1e-6).sum())
        print(f"  {name}: constant within every episode: {'yes' if n_var == 0 else f'NO ({n_var} episodes vary)'}; "
              f"range [{v.min():+.4f}, {v.max():+.4f}]")
        if n_var:
            failures.append(f"{name} not constant within {n_var} episode(s)")
        if v.min() < -1 - 1e-6 or v.max() > 1 + 1e-6:
            failures.append(f"{name} outside [-1, 1]")
    max_rows = int(round(args.horizon / args.dt))
    print(f"  rows per episode: {_pct(lengths)}  (max possible {max_rows}); "
          f"full length: {_share_pct(lengths >= max_rows):.1f}% of the episodes")
    if lengths.max() > max_rows:
        failures.append(f"an episode has {lengths.max()} rows > horizon/dt = {max_rows} (wrong --horizon/--dt?)")

    ep_rudder = x_all[order[starts], c["rudder_action"]]
    ep_thrust = x_all[order[starts], c["thrust_action"]]
    straight = (np.abs(ep_rudder) < STRAIGHT_ABS_RUDDER) & (ep_thrust >= 0.2 - 1e-6)
    print(f"  straight-like episodes (|rudder| < {STRAIGHT_ABS_RUDDER}, thrust >= 0.2): "
          f"{_share_pct(straight):.1f}% of the episodes, {100 * lengths[straight].sum() / len(x_all):.1f}% of the rows")

    # ---- 4. coverage -----------------------------------------------------------
    print(f"\n== 4. Coverage (all rows, {first.ship_model}, rudder_action +-1 = +-{first.rudder_max_deg:g} deg)")
    print("  rudder_action per episode: " + _hist_text(ep_rudder, np.linspace(-1, 1, 9)))
    print("  thrust_action per episode: " + _hist_text(ep_thrust, np.linspace(-1, 1, 9)))
    for name, unit in (("surge", "m/s"), ("sway", "m/s"), ("yaw_rate", "rad/s"), ("ukc", "m")):
        if name in c:
            print(f"  {name:9s} [{unit}]: {_pct(x_all[:, c[name]])}")
    if "ukc" in c:
        u = x_all[:, c["ukc"]]
        print(f"  UKC: {_share_pct(u < 2):.1f}% of rows < 2 m, {_share_pct(u < 5):.1f}% < 5 m")
    heading_deg = np.degrees(x_all[:, c["heading"]])
    hc, _ = np.histogram(heading_deg, bins=np.linspace(-180, 180, 13))
    print(f"  heading (30 deg bins): min {100 * hc.min() / len(x_all):.1f}% / max {100 * hc.max() / len(x_all):.1f}% of rows")
    ray_cols = [i for i, n in enumerate(cols) if n.startswith("ray_")]
    if ray_cols:
        rays = x_all[:, ray_cols]
        hit = rays.max(1)
        print(f"  rays: {_share_pct(hit > 1e-6):.1f}% of rows see a bank (any ray), "
              f"{_share_pct(hit > 0.75):.1f}% a close bank (feature > 0.75)")
    if scenario.includes_wind_current:
        print(f"  wind speed [m/s]: {_pct(x_all[:, c['wind_speed']])}")
        print(f"  current speed [m/s]: {_pct(x_all[:, c['current_speed']])}")

    # ---- 5. targets ------------------------------------------------------------
    print(f"\n== 5. Targets (per {args.dt:g} s step)")
    for k, name in enumerate(TARGET_NAMES):
        v = y_all[:, k]
        med = np.median(v)
        mad = float(np.median(np.abs(v - med)) * 1.4826)   # product in float32, as stored
        if mad == 0.0:          # constant target: avoid division by zero
            mad = 1e-12
        z = np.abs(v - med) / mad
        n_out = int((z > OUTLIER_K).sum())
        print(f"  {name:15s}: {_pct(v)}  outliers (> {OUTLIER_K:g} MAD): {n_out} rows "
              f"({100 * n_out / len(v):.3f}%)")
        if n_out and ray_cols:
            # Outliers near a bank are plausible (bank effects); in open water suspicious.
            out_rows = z > OUTLIER_K
            near = x_all[:, ray_cols].max(1)
            for label, sel in (("all rows", np.ones(len(v), dtype=bool)), ("outlier rows", out_rows)):
                print(f"      {label:12s}: close bank (max ray > 0.75) {_share_pct(near[sel] > 0.75):5.1f}%, "
                      f"no bank in range {_share_pct(near[sel] <= 1e-6):5.1f}%, "
                      f"mean |rudder| {np.mean(np.abs(x_all[sel, c['rudder_action']])):.2f}, "
                      f"mean surge {np.mean(x_all[sel, c['surge']]):+.2f} m/s")
            open_out = out_rows & (near <= 1e-6)
            if open_out.any():
                print(f"      outliers with NO bank in ray range: {int(open_out.sum()):,} rows in "
                      f"{len(np.unique(ep_all[open_out])):,} episodes (the most suspicious ones)")
        if n_out:
            worst = np.argsort(z)[-3:][::-1]
            for i in worst:
                bank = f"  max_ray={x_all[i, ray_cols].max():.2f}" if ray_cols else ""
                print(f"      episode {ep_all[i]}: {name}={v[i]:+.4g}  surge={x_all[i, c['surge']]:+.3f}  "
                      f"rudder={x_all[i, c['rudder_action']]:+.3f}  thrust={x_all[i, c['thrust_action']]:+.3f}{bank}")

    # ---- 6. rudder sign ----------------------------------------------------------
    print("\n== 6. Rudder sign (framework: + rudder_action -> turn to PORT -> yaw rate increases)")
    m = ((x_all[:, c["surge"]] > MIN_SURGE_FOR_SIGN) & (x_all[:, c["thrust_action"]] > 0)
         & (np.abs(x_all[:, c["rudder_action"]]) > MIN_ABS_RUDDER))
    if m.sum() > 100:
        corr = float(np.corrcoef(x_all[m, c["rudder_action"]].astype(np.float64), y_all[m, 2].astype(np.float64))[0, 1])
        pos = float(y_all[m & (x_all[:, c["rudder_action"]] > 0), 2].mean())
        neg = float(y_all[m & (x_all[:, c["rudder_action"]] < 0), 2].mean())
        print(f"  {int(m.sum()):,} rows: corr(rudder_action, delta_yaw_rate) = {corr:+.3f}; mean delta_yaw_rate "
              f"for rudder > 0: {np.degrees(pos):+.5f} deg/s, < 0: {np.degrees(neg):+.5f} deg/s")
        if corr <= 0:
            failures.append(f"rudder sign: corr(rudder_action, delta_yaw_rate) = {corr:+.3f} <= 0")
    else:
        print("  too few rows with forward speed, forward thrust and a clear rudder")

    # ---- 7. duplicates -------------------------------------------------------------
    rows = np.ascontiguousarray(np.hstack([x_all, y_all]))
    n_dup = len(rows) - len(np.unique(rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1])))))
    print(f"\n== 7. Exact duplicate rows: {n_dup}")
    if n_dup > 0.001 * len(rows):
        failures.append(f"{n_dup} exact duplicate rows (> 0.1%)")

    # ---- plot --------------------------------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            panels = [("rudder_action (per episode)", ep_rudder), ("thrust_action (per episode)", ep_thrust),
                      ("surge [m/s]", x_all[:, c["surge"]]), ("sway [m/s]", x_all[:, c["sway"]]),
                      ("yaw_rate [deg/s]", np.degrees(x_all[:, c["yaw_rate"]])), ("heading [deg]", heading_deg),
                      ("rows per episode", lengths)]
            if "ukc" in c:
                panels.append(("UKC [m]", x_all[:, c["ukc"]]))
            if ray_cols:
                panels.append(("max ray feature", x_all[:, ray_cols].max(1)))
            panels += [(f"{n}", y_all[:, k]) for k, n in enumerate(TARGET_NAMES)]
            ncol = 4
            fig, axes = plt.subplots(-(-len(panels) // ncol), ncol, figsize=(4.2 * ncol, 3.0 * -(-len(panels) // ncol)))
            for ax, (title, v) in zip(axes.flat, panels):
                lo, hi = np.percentile(v, [0.1, 99.9])
                ax.hist(v, bins=80, range=(lo, hi) if hi > lo else None, color="#3b6ea5")
                ax.set_title(title, fontsize=9)
            for ax in list(axes.flat)[len(panels):]:
                ax.axis("off")
            fig.suptitle(f"{Path(args.prefix).name}: {first.ship_model} / {first.scenario}, {len(x_all):,} rows")
            fig.tight_layout()
            out = Path(f"{args.prefix}_quality.png")
            fig.savefig(out, dpi=110)
            print(f"\nWrote {out}")
        except Exception as exc:
            print(f"(plot skipped: {exc})")

    print("\n=> " + ("ALL HARD CHECKS PASSED" if not failures else "FAILED:\n   - " + "\n   - ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
