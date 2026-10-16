"""
plot_val_vs_simulator.py
------------------------
Compares surrogate model predictions on the validation set against the
ground-truth values recorded by the FH simulator.

Layout (3 × 3 grid):
  Row 1 – delta_surge   : scatter pred vs true  |  residual histogram  |  sorted true vs pred
  Row 2 – delta_sway    : same
  Row 3 – delta_yaw_rate: same

Usage
-----
    python plot_val_vs_simulator.py

Paths (edit the constants below if needed):
    VAL_NPZ   – validation split produced by generate_ship_data.py
    CKPT_PATH – trained surrogate checkpoint
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Paths – adjust as needed
# ──────────────────────────────────────────────────────────────────────────────
VAL_NPZ   = Path("data/ship_dynamics_dataset_val.npz")
CKPT_PATH = Path("checkpoints/best_surrogate.pth")
OUT_PNG   = Path("val_vs_simulator.png")

# ──────────────────────────────────────────────────────────────────────────────
# Output labels / units
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_NAMES  = ["Δ Surge",    "Δ Sway",     "Δ Yaw rate"]
OUTPUT_UNITS  = ["m/s",        "m/s",         "rad/s"]
OUTPUT_KEYS   = ["delta_surge","delta_sway",  "delta_yaw_rate"]

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load surrogate (its checkpoint says which scenario it expects)
# ──────────────────────────────────────────────────────────────────────────────
from fw.surrogate.inference_surrogate import SurrogatePredictor   # noqa: E402
from fw.surrogate.dataset import load_raw                         # noqa: E402

predictor = SurrogatePredictor(str(CKPT_PATH))
print(f"Surrogate: scenario={predictor.scenario.name}, env_encoding={predictor.env_encoding}, "
      f"ship_model={predictor.ship_model}")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Load validation split (X used for inference, Y_sim = ground truth) and
#     run inference
# ──────────────────────────────────────────────────────────────────────────────
print("Loading validation split …")
# RAW rows in the checkpoint scenario's layout (projected from the file's own
# scenario when allowed, e.g. open_water from confined_ukc data); load_raw
# checks the file's identity. SurrogatePredictor.predict() selects and
# encodes the columns itself -- never hand-slice X here.
X_val, Y_sim, val_identity = load_raw(VAL_NPZ, predictor.scenario)
if val_identity.ship_model != predictor.ship_model:
    raise SystemExit(f"{VAL_NPZ} is {val_identity.ship_model} data, but the surrogate was trained "
                     f"on {predictor.ship_model} data.")

N = X_val.shape[0]
print(f"  {N:,} validation samples")

print("Running surrogate inference …")
Y_pred    = predictor.predict(X_val)   # batched, avoids one Python-level
                                        # model call per row on what can be
                                        # a 100k+ row validation split
print(f"  Predictions shape: {Y_pred.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Per-output metrics
# ──────────────────────────────────────────────────────────────────────────────
residuals = Y_pred - Y_sim       # (N, 3)

def rmse(a, b):
    """
    Root mean squared error between two arrays.

    Args:
        a: First array.
        b: Second array, same shape as `a`.

    Returns:
        float: RMSE of `a - b`.
    """

    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b):
    """
    Mean absolute error between two arrays.

    Args:
        a: First array.
        b: Second array, same shape as `a`.

    Returns:
        float: MAE of `a - b`.
    """

    return float(np.mean(np.abs(a - b)))

def r2(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Coefficient of determination (R²) of predictions against ground truth.

    Args:
        true: Ground-truth values.
        pred: Predicted values, same shape as `true`.

    Returns:
        float: R², or NaN if `true` has zero variance.
    """

    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))

    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

metrics = []
for i in range(3):
    metrics.append({
        "rmse": rmse(Y_sim[:, i], Y_pred[:, i]),
        "mae":  mae (Y_sim[:, i], Y_pred[:, i]),
        "r2":   r2  (Y_sim[:, i], Y_pred[:, i]),
    })

print("\nMetrics on validation set:")
print(f"  {'Output':<18}  {'RMSE':>12}  {'MAE':>12}  {'R²':>8}")
print(f"  {'-'*18}  {'-'*12}  {'-'*12}  {'-'*8}")
for i, name in enumerate(OUTPUT_NAMES):
    m = metrics[i]
    print(f"  {name + ' (' + OUTPUT_UNITS[i] + ')':<18}  "
          f"{m['rmse']:12.6f}  {m['mae']:12.6f}  {m['r2']:8.5f}")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Plot
# ──────────────────────────────────────────────────────────────────────────────
ALPHA_SCATTER = 0.15
N_SORTED      = min(2000, N)   # subset for the sorted-overlay panel (speed)
SUBSAMPLE     = max(1, N // 4000)   # thin scatter so it renders quickly

matplotlib.rcParams.update({
    "font.size":       10,
    "axes.titlesize":  10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":     120,
})

COLOR_TRUE = "#2563EB"   # blue
COLOR_PRED = "#DC2626"   # red
COLOR_RESID= "#059669"   # green

fig = plt.figure(figsize=(16, 13))
fig.suptitle("Surrogate vs FH Simulator - Validation Set", fontsize=13, fontweight="bold", y=0.98)

outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.48)

for row, i in enumerate(range(3)):
    name  = OUTPUT_NAMES[i]
    unit  = OUTPUT_UNITS[i]
    y_t   = Y_sim[:, i]
    y_p   = Y_pred[:, i]
    resid = residuals[:, i]
    m     = metrics[i]

    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[row],
                                             wspace=0.38)

    # ── Panel A: scatter true vs predicted ────────────────────────────────
    ax_sc = fig.add_subplot(inner[0])
    idx   = np.arange(0, N, SUBSAMPLE)
    ax_sc.scatter(y_t[idx], y_p[idx],
                  s=6, alpha=ALPHA_SCATTER, color=COLOR_PRED, linewidths=0)
    lims = (min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max()))
    pad  = (lims[1] - lims[0]) * 0.05
    lims = (lims[0] - pad, lims[1] + pad)
    ax_sc.plot(lims, lims, "k--", linewidth=0.9, label="Ideal (y = x)")
    ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)
    ax_sc.set_xlabel(f"Simulator  [{unit}]")
    ax_sc.set_ylabel(f"Surrogate  [{unit}]")
    ax_sc.set_title(
        f"{name}\n"
        f"RMSE={m['rmse']:.4g}  MAE={m['mae']:.4g}  R²={m['r2']:.4f}"
    )
    ax_sc.legend(fontsize=7, loc="upper left")
    ax_sc.set_aspect("equal", "box")
    ax_sc.grid(True, linewidth=0.4, alpha=0.5)

    # ── Panel B: residual histogram ────────────────────────────────────────
    ax_h = fig.add_subplot(inner[1])
    ax_h.hist(resid, bins=80, color=COLOR_RESID, alpha=0.75, edgecolor="none")
    ax_h.axvline(0, color="k", linewidth=0.9, linestyle="--")
    ax_h.axvline(resid.mean(), color="orange", linewidth=1.2,
                 linestyle="-", label=f"Mean={resid.mean():.3g}")
    ax_h.set_xlabel(f"Residual (pred − sim)  [{unit}]")
    ax_h.set_ylabel("Count")
    ax_h.set_title(f"{name} - Residuals\n"
                   f"μ={resid.mean():.3g}  σ={resid.std():.3g}")
    ax_h.legend(fontsize=7)
    ax_h.grid(True, linewidth=0.4, alpha=0.5)

    # ── Panel C: sorted true vs predicted (random subset) ─────────────────
    ax_s = fig.add_subplot(inner[2])
    rng  = np.random.default_rng(42)
    idx2 = rng.choice(N, size=N_SORTED, replace=False)
    order = np.argsort(y_t[idx2])
    xs    = np.arange(N_SORTED)
    ax_s.plot(xs, y_t[idx2][order], color=COLOR_TRUE,  linewidth=0.9,
              alpha=0.85, label="Simulator (ground truth)")
    ax_s.plot(xs, y_p[idx2][order], color=COLOR_PRED,  linewidth=0.7,
              alpha=0.65, linestyle="--", label="Surrogate prediction")
    ax_s.set_xlabel("Sample index (sorted by simulator value)")
    ax_s.set_ylabel(f"Value  [{unit}]")
    ax_s.set_title(f"{name} - Sorted overlay\n(random {N_SORTED:,} samples)")
    ax_s.legend(fontsize=7)
    ax_s.grid(True, linewidth=0.4, alpha=0.5)

plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"\nPlot saved -> {OUT_PNG}")
plt.show()
