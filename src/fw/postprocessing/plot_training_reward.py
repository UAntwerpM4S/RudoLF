"""
plot_training_reward.py
-----------------------
Plots average reward vs. training iteration in the style typical of
Deep Reinforcement Learning publications:

  - Raw per-step reward shown as a faint shaded area (or scatter).
  - A running/rolling mean overlaid as a solid line (the "smooth" curve).
  - A shaded ±1 std-dev band around the running mean to convey variance.

Multi-agent CSVs (MultiPPOAgent, with an `agent_id` column) get one curve per
agent, each with its own rolling statistics, and a dashed vertical line where
an agent met its stop condition and was frozen (`frozen` column). Before,
the rows of all agents were interleaved into one curve.

Usage
-----
    python plot_training_reward.py                       # uses default CSV path
    python plot_training_reward.py --csv path/to/file.csv
    python plot_training_reward.py --window 50 --out reward_curve.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CSV    = "training_metrics.csv"
DEFAULT_WINDOW = 50          # rolling-window size (iterations)
DEFAULT_OUT    = "reward_curve.pdf"

# Color palette: a single accent color keeps the figure clean and readable
# in both color and greyscale print.
COLOR_RAW    = "#a8c7e8"     # light blue  -> raw reward
COLOR_SMOOTH = "#1a5fa8"     # dark blue   -> rolling mean
COLOR_BAND   = "#a8c7e8"     # same as raw -> +/- 1 σ band

# Per-agent (light, dark) color pairs for multi-agent CSVs.
AGENT_COLORS = [
    ("#a8c7e8", "#1a5fa8"),     # blue
    ("#f5c28b", "#c25e00"),     # orange
    ("#a9dba4", "#2a7a22"),     # green
    ("#e8a8a8", "#a81a1a"),     # red
    ("#cdb5e3", "#5b2a8a"),     # purple
]

FONT_SIZE_LABEL  = 14
FONT_SIZE_TICK   = 12
FONT_SIZE_LEGEND = 12
FONT_SIZE_TITLE  = 15

FIG_W, FIG_H = 8, 4.5       # inches, fits a single-column thesis layout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load training reward data from a CSV file and normalize its structure.

    The function ensures a consistent format by:
    - Lowercasing column names
    - Ensuring an 'iteration' column exists
    - Creating a synthetic iteration index if necessary

    Args:
        csv_path (str): Path to the CSV file containing training data.

    Returns:
        pd.DataFrame: DataFrame containing at least:
            - iteration (int)
            - reward (float)

    Raises:
        ValueError: If the required 'reward' column is missing.
    """

    df = pd.read_csv(csv_path)

    required = {"reward"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(
            f"CSV must contain a 'reward' column. Found: {list(df.columns)}"
        )

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Use explicit iteration column if present, else use row index (per
    # agent for a multi-agent CSV, whose rows interleave the agents)
    if "iteration" not in df.columns and "step" not in df.columns:
        if "agent_id" in df.columns:
            df.insert(0, "iteration", df.groupby("agent_id").cumcount())
        else:
            df.insert(0, "iteration", np.arange(len(df)))
    elif "step" in df.columns:
        df = df.rename(columns={"step": "iteration"})

    return df


def rolling_stats(
    series: pd.Series, window: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute rolling mean and ±1 standard deviation envelope.

    Args:
        series (pd.Series): Input numeric series (e.g., rewards).
        window (int): Rolling window size.

    Returns:
        tuple:
            - rolling mean (pd.Series)
            - lower bound (mean - std)
            - upper bound (mean + std)
    """

    rm   = series.rolling(window, min_periods=1).mean()
    rstd = series.rolling(window, min_periods=1).std().fillna(0)
    return rm, rm - rstd, rm + rstd


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_curve(ax, iterations, reward: pd.Series, window: int,
                color_raw: str, color_smooth: str, label: str = "") -> None:
    """
    Draw one reward curve: raw signal, ±1 σ rolling band and rolling mean.

    Args:
        ax: Matplotlib axes to draw on.
        iterations: X values (training iterations).
        reward (pd.Series): Reward per iteration.
        window (int): Rolling window size for smoothing.
        color_raw (str): Color of the raw signal and band.
        color_smooth (str): Color of the rolling mean.
        label (str): Prefix for the legend entries ("" for a single curve).
    """

    prefix = f"{label}: " if label else ""
    rm, rm_lo, rm_hi = rolling_stats(reward.reset_index(drop=True), window)

    # 1. Raw reward: semi-transparent to keep focus on the smooth curve
    ax.plot(
        iterations, reward.values,
        color=color_raw, alpha=0.35, linewidth=0.7,
        label=f"{prefix}Reward (per iteration)",
        zorder=1,
    )

    # 2. ±1 σ shaded band
    ax.fill_between(
        iterations, rm_lo, rm_hi,
        color=color_raw, alpha=0.25,
        label=f"{prefix}" + r"$\pm 1\,\sigma$ (rolling)",
        zorder=2,
    )

    # 3. Rolling mean: the star of the show
    ax.plot(
        iterations, rm,
        color=color_smooth, linewidth=2.0,
        label=f"{prefix}Rolling mean (w = {window})",
        zorder=3,
    )


def plot_reward(df: pd.DataFrame, window: int, out_path: str):
    """
    Plot DRL training reward curve with smoothing and variance visualization.

    The plot includes:
        - Raw reward signal (low-opacity line)
        - Rolling mean reward (main curve)
        - ±1 standard deviation band around rolling mean

    With an `agent_id` column (multi-agent CSV), every agent gets its own
    curve and rolling statistics, plus a dashed vertical line at the first
    iteration it was frozen (`frozen` column), if any.

    Args:
        df (pd.DataFrame): Input dataframe containing:
            - iteration (int)
            - reward (float)
            - agent_id (int, optional), frozen (bool, optional)
        window (int): Rolling window size for smoothing.
        out_path (str): Output file path for saving the figure.

    Returns:
        matplotlib.figure.Figure: The figure (also saved to `out_path`).
    """

    # --- Figure & axes ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    if "agent_id" in df.columns:
        for k, (agent_id, agent_df) in enumerate(df.groupby("agent_id", sort=True)):
            agent_df = agent_df.sort_values("iteration")
            color_raw, color_smooth = AGENT_COLORS[k % len(AGENT_COLORS)]
            _plot_curve(ax, agent_df["iteration"].values, agent_df["reward"], window,
                        color_raw, color_smooth, label=f"Agent {agent_id}")
            if "frozen" in agent_df.columns:
                frozen = agent_df["frozen"].astype(str).str.lower() == "true"
                if frozen.any():
                    ax.axvline(agent_df.loc[frozen, "iteration"].iloc[0],
                               color=color_smooth, linewidth=1.0, linestyle="--", alpha=0.8,
                               label=f"Agent {agent_id}: frozen", zorder=4)
        iterations = np.sort(df["iteration"].unique())
    else:
        iterations = df["iteration"].values
        _plot_curve(ax, iterations, df["reward"], window, COLOR_RAW, COLOR_SMOOTH)

    # --- Labels & formatting --------------------------------------------
    ax.set_xlabel("Iteration", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Average Reward", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Training Reward Curve", fontsize=FONT_SIZE_TITLE, pad=10)

    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    # Tight x-limits: no unnecessary whitespace
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

    # Clean spine style (no top / right frame), widely used in ML papers
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # --- Save ------------------------------------------------------------
    out = Path(out_path)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[✓] Figure saved -> {out.resolve()}")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point for DRL reward curve visualization.

    Workflow:
        1. Parse CLI arguments
        2. Load and normalize training CSV
        3. Generate and save reward plot

    CLI Arguments:
        --csv (str): Path to input CSV file.
        --window (int): Rolling window size for smoothing.
        --out (str): Output file path.

    Returns:
        None
    """

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
    n_agents = df["agent_id"].nunique() if "agent_id" in df.columns else 1
    print(
        f"[i] Loaded {len(df):,} rows ({n_agents} agent(s)) · "
        f"reward ∈ [{df['reward'].min():.3f}, {df['reward'].max():.3f}]"
    )

    plot_reward(df, window=args.window, out_path=args.out)


if __name__ == "__main__":
    main()
