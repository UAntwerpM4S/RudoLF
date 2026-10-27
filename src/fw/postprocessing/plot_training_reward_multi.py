"""
plot_training_reward_multi.py
-----------------------------
Plots average reward vs. training iteration per policy from a
multi_policy_training_metrics.csv file.

  - One line (rolling mean) per policy, each with its own color.
  - Faint raw reward and ±1 σ band per policy.
  - Policies whose reward is NaN for every row are skipped.

Usage
-----
    python plot_training_reward_multi.py
    python plot_training_reward_multi.py --csv path/to/file.csv
    python plot_training_reward_multi.py --window 50 --out reward_curve.pdf
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path


DEFAULT_CSV    = "models/multi_policy_training_metrics.csv"
DEFAULT_WINDOW = 50
DEFAULT_OUT    = "reward_curve_multi.pdf"

# Distinct, print-friendly colors per policy.
POLICY_COLORS = {
    "path-following": "#1a5fa8",   # blue
    "head-on":        "#d1495b",   # red
    "crossing-right": "#2e8b57",   # green
    "crossing-left":  "#e0a800",   # amber
    "overtaking":     "#7d3cae",   # purple
}
FALLBACK_CYCLE = ["#1a5fa8", "#d1495b", "#2e8b57", "#e0a800",
                  "#7d3cae", "#00a3a3", "#c45a00"]

FONT_SIZE_LABEL  = 14
FONT_SIZE_TICK   = 12
FONT_SIZE_LEGEND = 11
FONT_SIZE_TITLE  = 15

FIG_W, FIG_H = 9, 5


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load and normalize training metrics from a CSV file.

    The function ensures required columns are present, normalizes column names
    to lowercase, and constructs a consistent 'iteration' column if missing.

    Args:
        csv_path (str): Path to the input CSV file containing training metrics.

    Returns:
        pd.DataFrame: Normalized dataframe containing at least:
            - policy (str)
            - reward (float)
            - iteration (int)

    Raises:
        ValueError: If required columns ('policy', 'reward') are missing.
    """

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    if "policy" not in df.columns or "reward" not in df.columns:
        raise ValueError(
            f"CSV must contain 'policy' and 'reward' columns. Found: {list(df.columns)}"
        )

    if "iteration" not in df.columns and "step" not in df.columns:
        df["iteration"] = df.groupby("policy").cumcount()
    elif "step" in df.columns and "iteration" not in df.columns:
        df = df.rename(columns={"step": "iteration"})

    return df


def rolling_stats(series: pd.Series, window: int):
    """
    Compute rolling mean and rolling standard deviation bounds.

    This function computes a rolling mean over the input series and derives
    upper and lower confidence bands using ±1 standard deviation.

    Args:
        series (pd.Series): Input numeric series (e.g., rewards).
        window (int): Rolling window size.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]:
            - rolling mean
            - lower bound (mean - std)
            - upper bound (mean + std)
    """

    rm   = series.rolling(window, min_periods=1).mean()
    rstd = series.rolling(window, min_periods=1).std().fillna(0)
    return rm, rm - rstd, rm + rstd


def plot_reward(df: pd.DataFrame, window: int, out_path: str) -> None:
    """
    Plot per-policy training reward curves with rolling statistics.

    For each policy in the dataframe, this function plots:
        - Raw reward trajectory (low opacity)
        - Rolling mean reward
        - ±1 standard deviation band around the rolling mean

    Policies with entirely NaN rewards are skipped.

    Args:
        df (pd.DataFrame): Input dataframe containing columns:
            - policy
            - reward
            - iteration
        window (int): Rolling window size for smoothing.
        out_path (str): Output file path for saving the figure.

    Raises:
        RuntimeError: If no valid (non-NaN) policies are available to plot.
    """

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    policies = list(df["policy"].unique())
    plotted_any = False

    for idx, policy in enumerate(policies):
        sub = df[df["policy"] == policy].sort_values("iteration")
        reward = sub["reward"]

        if reward.isna().all():
            print(f"[skip] '{policy}' - all reward values are NaN")
            continue

        color = POLICY_COLORS.get(policy, FALLBACK_CYCLE[idx % len(FALLBACK_CYCLE)])
        iterations = sub["iteration"].values
        rm, rm_lo, rm_hi = rolling_stats(reward, window)

        ax.plot(iterations, reward.values,
                color=color, alpha=0.20, linewidth=0.7, zorder=1)
        ax.fill_between(iterations, rm_lo, rm_hi,
                        color=color, alpha=0.15, zorder=2)
        ax.plot(iterations, rm,
                color=color, linewidth=2.0,
                label=policy, zorder=3)
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("No policies had any non-NaN reward values to plot.")

    ax.set_xlabel("Iteration", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Average Reward", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Training Reward Curve per Policy", fontsize=FONT_SIZE_TITLE, pad=10)

    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5, zorder=0)

    ax.legend(
        title="Policy (rolling mean)",
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_LEGEND,
        loc="best",
        framealpha=0.85,
        edgecolor="lightgrey",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out = Path(out_path)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[ok] Figure saved -> {out.resolve()}")
    plt.show()


def main() -> None:
    """
    CLI entry point for plotting multi-policy training reward curves.

    This function:
        1. Parses command-line arguments
        2. Loads and normalizes the CSV dataset
        3. Generates and saves a reward curve visualization

    CLI Arguments:
        --csv (str): Path to input CSV file.
        --window (int): Rolling window size for smoothing.
        --out (str): Output file path for saved figure.

    Returns:
        None
    """

    parser = argparse.ArgumentParser(
        description="Plot per-policy DRL training reward curves."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    df = load_data(args.csv)
    print(f"[i] Loaded {len(df):,} rows across {df['policy'].nunique()} policies")

    plot_reward(df, window=args.window, out_path=args.out)


if __name__ == "__main__":
    main()
