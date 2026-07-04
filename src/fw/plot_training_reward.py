"""
plot_training_reward.py
-----------------------
Plots average reward vs. training iteration in the style typical of
Deep Reinforcement Learning publications:

  - Raw per-step reward shown as a faint shaded area (or scatter).
  - A running/rolling mean overlaid as a solid line (the "smooth" curve).
  - A shaded ±1 std-dev band around the running mean to convey variance.

Usage
-----
    python plot_training_reward.py                       # uses default CSV path
    python plot_training_reward.py --csv path/to/file.csv
    python plot_training_reward.py --window 50 --out reward_curve.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CSV    = "training_metrics.csv"
DEFAULT_WINDOW = 50          # rolling-window size (iterations)
DEFAULT_OUT    = "reward_curve.pdf"

# Colour palette — a single accent colour keeps the figure clean and readable
# in both colour and greyscale print.
COLOR_RAW    = "#a8c7e8"     # light blue  → raw reward
COLOR_SMOOTH = "#1a5fa8"     # dark blue   → rolling mean
COLOR_BAND   = "#a8c7e8"     # same as raw → ±1 σ band

FONT_SIZE_LABEL  = 14
FONT_SIZE_TICK   = 12
FONT_SIZE_LEGEND = 12
FONT_SIZE_TITLE  = 15

FIG_W, FIG_H = 8, 4.5       # inches — fits a single-column thesis layout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and return a DataFrame with a zero-based 'iteration' column."""
    df = pd.read_csv(csv_path)

    required = {"reward"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(
            f"CSV must contain a 'reward' column. Found: {list(df.columns)}"
        )

    # Normalise column names to lowercase
    df.columns = df.columns.str.lower()

    # Use explicit iteration column if present, else use row index
    if "iteration" not in df.columns and "step" not in df.columns:
        df.insert(0, "iteration", np.arange(len(df)))
    elif "step" in df.columns:
        df = df.rename(columns={"step": "iteration"})

    return df


def rolling_stats(
    series: pd.Series, window: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (rolling_mean, rolling_mean - std, rolling_mean + std)."""
    rm   = series.rolling(window, min_periods=1).mean()
    rstd = series.rolling(window, min_periods=1).std().fillna(0)
    return rm, rm - rstd, rm + rstd


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reward(df: pd.DataFrame, window: int, out_path: str) -> None:
    iterations = df["iteration"].values
    reward     = df["reward"].values

    rm, rm_lo, rm_hi = rolling_stats(df["reward"], window)

    # --- Figure & axes ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # 1. Raw reward — semi-transparent to keep focus on the smooth curve
    ax.plot(
        iterations, reward,
        color=COLOR_RAW, alpha=0.35, linewidth=0.7,
        label="Reward (per iteration)",
        zorder=1,
    )

    # 2. ±1 σ shaded band
    ax.fill_between(
        iterations, rm_lo, rm_hi,
        color=COLOR_BAND, alpha=0.25,
        label=r"$\pm 1\,\sigma$ (rolling)",
        zorder=2,
    )

    # 3. Rolling mean — the star of the show
    ax.plot(
        iterations, rm,
        color=COLOR_SMOOTH, linewidth=2.0,
        label=f"Rolling mean (w = {window})",
        zorder=3,
    )

    # --- Labels & formatting --------------------------------------------
    ax.set_xlabel("Iteration", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Average Reward", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Training Reward Curve", fontsize=FONT_SIZE_TITLE, pad=10)

    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    # Tight x-limits — no unnecessary whitespace
    ax.set_xlim(iterations[0], iterations[-1])

    # Horizontal guide line at y = 0 for reference
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5, zorder=0)

    # Legend inside the plot, upper-left (common DRL convention)
    ax.legend(
        fontsize=FONT_SIZE_LEGEND,
        loc="best",
        framealpha=0.85,
        edgecolor="lightgrey",
    )

    # Clean spine style (no top / right frame) — widely used in ML papers
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # --- Save ------------------------------------------------------------
    out = Path(out_path)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[✓] Figure saved → {out.resolve()}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DRL training reward curve (thesis-quality)."
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help=f"Path to the training CSV file (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=f"Rolling-window size in iterations (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Output file path; format inferred from extension (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    print(
        f"[i] Loaded {len(df):,} iterations · "
        f"reward ∈ [{df['reward'].min():.3f}, {df['reward'].max():.3f}]"
    )

    plot_reward(df, window=args.window, out_path=args.out)


if __name__ == "__main__":
    main()
