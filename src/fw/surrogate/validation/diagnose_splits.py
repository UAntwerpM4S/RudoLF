"""
diagnose_splits.py -- check train/val/test .npz splits for problems that
make val loss diverge from train loss.

Reports, per encoded input feature and per target:
  * features that are (near-)constant in TRAIN (std < STD_FLOOR)
  * how far val/test drift from train, in train-std units
    (|mean shift| and the max |z| any val/test sample reaches)
  * the fraction of val/test samples outside train's [min, max] range

Features are encoded the way training encodes them: for --scenario
(default: the files' own scenario) with --env-encoding (default: the
training default). Use --env-encoding angles to inspect the angle encoding.

Usage:
    python -m fw.surrogate.validation.diagnose_splits                # default data/ files
    python -m fw.surrogate.validation.diagnose_splits --train-file ... --val-file ... --test-file ...
    python -m fw.surrogate.validation.diagnose_splits --scenario open_water ...   # as UA_clean sees it
"""

import argparse

import numpy as np

from fw.surrogate.dataset import STD_FLOOR, check_same_identity, load_encoded, read_identity
from fw.surrogate.schema import (
    DEFAULT_ENV_ENCODING, ENV_ENCODINGS, SCENARIOS, TARGET_NAMES, encoded_feature_names, get_scenario,
)
from fw.surrogate.utils import data_path


def report(kind, names, tr, splits):
    """
    Print, per column, the train std and each split's mean shift, max |z| and out-of-range share,
    sorted by severity.

    Args:
        kind: Section title (e.g. "INPUT FEATURES").
        names: Column names, one per column of `tr`.
        tr: Train array, shape (N, len(names)).
        splits: {split name: array} to compare with train; empty splits are skipped.
    """

    splits = {k: v for k, v in splits.items() if len(v)}  # empty splits reported in main()
    w = max(16, max(len(n) for n in names) + 2)
    mean = tr.mean(0)
    raw_std = tr.std(0)
    std = np.where(raw_std < STD_FLOOR, 1.0, raw_std)
    lo, hi = tr.min(0), tr.max(0)

    rows = []
    for i, name in enumerate(names):
        row = {"name": name, "train_std": raw_std[i], "const": raw_std[i] < STD_FLOOR}
        for sname, arr in splits.items():
            col = arr[:, i]
            row[f"{sname}_shift"] = abs(col.mean() - mean[i]) / std[i]
            row[f"{sname}_maxz"] = np.abs((col - mean[i]) / std[i]).max()
            row[f"{sname}_out"] = np.mean((col < lo[i]) | (col > hi[i]))
        rows.append(row)

    score = lambda r: max((r[f"{s}_maxz"] for s in splits), default=0.0) + 100 * r["const"]
    rows.sort(key=score, reverse=True)

    snames = list(splits)
    print(f"\n=== {kind} (sorted by severity; z in train-std units) ===")
    hdr = f"{'feature':<{w}}{'train_std':>11}  " + "".join(
        f"{s + ' shift':>11}{s + ' max|z|':>12}{s + ' out%':>10}" for s in snames)
    print(hdr)
    for r in rows:
        line = f"{r['name']:<{w}}{r['train_std']:>11.3g}  " + "".join(
            f"{r[f'{s}_shift']:>11.2f}{r[f'{s}_maxz']:>12.1f}{100 * r[f'{s}_out']:>9.1f}%"
            for s in snames)
        flag = "  <-- CONSTANT IN TRAIN" if r["const"] else (
            "  <-- outside train range" if any(r[f"{s}_out"] > 0.01 for s in snames) else "")
        print(line + flag)


def main(argv=None):
    """
    CLI entry point: load and encode the three splits, report NaN/inf values, then print the
    input-feature and target reports.
    """

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train-file", default=data_path("ship_dynamics_dataset_train.npz"))
    p.add_argument("--val-file", default=data_path("ship_dynamics_dataset_val.npz"))
    p.add_argument("--test-file", default=data_path("ship_dynamics_dataset_test.npz"))
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default=None,
                   help="Model scenario to encode for (default: the files' own scenario)")
    p.add_argument("--env-encoding", choices=list(ENV_ENCODINGS), default=DEFAULT_ENV_ENCODING,
                   help="Wind/current encoding to inspect (default: the training default)")
    a = p.parse_args(argv)

    files = [str(a.train_file), str(a.val_file), str(a.test_file)]
    identity = check_same_identity(files, [read_identity(f) for f in files])
    scenario = get_scenario(a.scenario or identity.scenario)
    (xtr, ytr, _), (xva, yva, _), (xte, yte, _) = (load_encoded(f, scenario, a.env_encoding) for f in files)
    print(f"samples: train={len(xtr)}  val={len(xva)}  test={len(xte)}  "
          f"(data: {identity.scenario}, {identity.ship_model}; model scenario={scenario.name}, "
          f"env_encoding={a.env_encoding})")
    if len(xtr) == 0:
        raise SystemExit("The train split is empty -- nothing to compare against.")
    for name, arr in (("val", xva), ("test", xte)):
        if len(arr) == 0:
            print(f"!! the {name} split is EMPTY -- training will fail on it; it is skipped below.")

    for name, arr in (("train X", xtr), ("val X", xva), ("test X", xte),
                      ("train Y", ytr), ("val Y", yva), ("test Y", yte)):
        bad = int((~np.isfinite(arr)).sum())
        if bad:
            print(f"!! {name} contains {bad} NaN/inf values")

    report("INPUT FEATURES", list(encoded_feature_names(scenario, a.env_encoding)), xtr, {"val": xva, "test": xte})
    report("TARGETS", list(TARGET_NAMES), ytr, {"val": yva, "test": yte})

    print("\nHow to read this:")
    print(" * CONSTANT IN TRAIN: the old code divided this by ~1e-8. Any value")
    print("   different from the train constant in val/test exploded the loss.")
    print(" * max|z| >> 5 or out% > a few %: val/test cover conditions train never")
    print("   saw (distribution shift). No regularisation fixes that -- the data")
    print("   generation / episode split needs to cover those conditions.")


if __name__ == "__main__":
    main()
