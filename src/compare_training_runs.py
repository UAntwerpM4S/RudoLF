"""
Compare the train blocks of one or more pipeline specs from their iteration logs
(logs/<block>_<agent>_iterations_<timestamp>.csv, the latest one per block).

Per block: batch_size, num_envs and seed (the block's hyperparameter_overrides on
top of the config defaults of the agent named in its iteration log, as the
pipeline resolves them), iterations and episodes run,
training time, time per iteration, whether and when the block's stop condition
was reached, and the best rolling-mean reward. The stop condition follows
StopCondition: TARGET_REWARD means mean_reward >= target_reward for `patience`
consecutive iterations (patience missing/None -> 1, minimum 1); TARGET_REWARD_AND_LOSS
also needs loss <= target_loss; NBR_POLICY_UPDATES has no target (no hit is
reported). --target / --patience replace the specs' values for every block (the
criterion's loss condition stays), so runs from specs with different targets can be
compared on one target; NBR_POLICY_UPDATES blocks are then scored on the reward
rule. A figure shows the
rolling-mean reward per block against iterations, episodes and training time. With
num_envs > 1 an iteration holds num_envs episodes, so compare such runs by
episodes and training time, not by iterations.

The times are those of the learn() loop (elapsed_seconds), without the pipeline
start-up.

Run from src/ after the pipeline:
    python compare_training_runs.py --spec fw/batch_size_comparison_pipeline.json
    python compare_training_runs.py --spec fw/batch_size_comparison_pipeline.json --window 20 --out cmp.png
    python compare_training_runs.py --spec fw/batch_size_comparison_pipeline.json \
        fw/envs_and_bs32_comparison_pipeline.json --blocks "bs64_*" "bs32_*" "envs*" --target 0.93
"""

import argparse
import json
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent


def latest_iteration_log(log_dir: Path, block_name: str) -> Optional[Path]:
    """
    Newest iteration log of a block, or None. Matches
    <block>_<agent>_iterations_<YYYYmmdd_HHMMSS>.csv exactly, so 'bs64_seed1' does
    not pick up the logs of 'bs64_seed10'; the timestamp sorts chronologically.
    """

    pattern = re.compile(rf"^{re.escape(block_name)}_[A-Za-z0-9]+_iterations_\d{{8}}_\d{{6}}\.csv$")
    candidates = [p for p in log_dir.glob(f"{block_name}_*_iterations_*.csv") if pattern.match(p.name)]
    return max(candidates, key=lambda p: p.name) if candidates else None


def log_agent_type(log: Path, block_name: str) -> str:
    """
    Agent type in an iteration log's name: <block>_<agent>_iterations_<timestamp>.csv.
    """

    return log.name[len(block_name) + 1:].split("_iterations_")[0]


def block_hyperparameters(block: dict, agent_type: str) -> dict:
    """
    Hyperparameters a block trained with: the config defaults of agent_type (with the
    env-specific overrides, as get_hyperparameters applies them) and the block's
    hyperparameter_overrides on top. Unknown agent type or no fw package: the
    overrides only.
    """

    try:
        from fw.configuration.config import get_hyperparameters
        defaults = get_hyperparameters(agent_type, env_name=block.get("env_type"))
    except (ImportError, ValueError):
        defaults = {}
    return {**defaults, **block.get("hyperparameter_overrides", {})}


def normalized_patience(patience) -> int:
    """
    Patience as StopCondition uses it: None -> 1, otherwise at least 1.
    """

    return max(1, int(patience)) if patience is not None else 1


def first_target_hit(rewards: pd.Series, target: float, patience, losses: Optional[pd.Series] = None,
                     target_loss: Optional[float] = None) -> Optional[int]:
    """
    Row index at which `patience` consecutive successes are complete (where
    StopCondition stops), or None. A success is reward >= target, and with
    target_loss also loss <= target_loss (TARGET_REWARD_AND_LOSS).
    """

    patience = normalized_patience(patience)
    loss_values = list(losses) if target_loss is not None else None
    run = 0
    for i, reward in enumerate(rewards):
        ok = reward >= target and (loss_values is None or loss_values[i] <= target_loss)
        run = run + 1 if ok else 0
        if run >= patience:
            return i
    return None


def block_target(stop: dict, target_override: Optional[float],
                 patience_override: Optional[int]) -> Tuple[Optional[float], Optional[float], int]:
    """
    (target_reward, target_loss, patience) to score a block on: from its
    stop_condition per criterion, with the overrides on top. target_reward None
    means no target (NBR_POLICY_UPDATES without --target).
    """

    criterion = str(stop.get("criterion", "NBR_POLICY_UPDATES")).upper()
    target = stop.get("target_reward") if criterion in ("TARGET_REWARD", "TARGET_REWARD_AND_LOSS") else None
    target_loss = stop.get("target_loss") if criterion == "TARGET_REWARD_AND_LOSS" else None
    if target_override is not None:
        target = target_override
    patience = normalized_patience(patience_override if patience_override is not None else stop.get("patience"))
    return target, target_loss, patience


def main() -> None:
    """
    Read the spec and the logs, print the table and save the figure.
    """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, nargs="+", help="Pipeline JSON(s) whose train blocks to compare.")
    parser.add_argument("--blocks", nargs="+", default=None,
                        help="Only blocks whose name matches one of these patterns (e.g. \"bs64_*\").")
    parser.add_argument("--log-dir", default=str(SRC_DIR / "logs"))
    parser.add_argument("--window", type=int, default=20, help="Rolling-mean window [iterations].")
    parser.add_argument("--out", default="training_comparison.png")
    parser.add_argument("--target", type=float, default=None,
                        help="Score every block on this target_reward instead of its spec's.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Score every block with this patience instead of its spec's.")
    args = parser.parse_args()

    blocks = [block for spec_path in args.spec
              for block in json.loads(Path(spec_path).read_text(encoding="utf-8"))["blocks"]]
    rows, curves = [], []
    for block in blocks:
        if block.get("action") != "train":
            continue
        if args.blocks and not any(fnmatch(block["name"], pattern) for pattern in args.blocks):
            continue
        name = block["name"]
        overrides = block.get("hyperparameter_overrides", {})
        stop = block.get("stop_condition", {})
        target, target_loss, patience = block_target(stop, args.target, args.patience)
        log = latest_iteration_log(Path(args.log_dir), name)
        if log is None:
            rows.append({"block": name, "batch_size": overrides.get("batch_size"),
                         "num_envs": overrides.get("num_envs"), "seed": overrides.get("seed"),
                         "iterations": 0, "note": "no iteration log"})
            continue
        params = block_hyperparameters(block, log_agent_type(log, name))

        df = pd.read_csv(log)
        reward = df["mean_reward"].astype(float)
        rolling = reward.rolling(args.window, min_periods=1).mean()
        hit = (first_target_hit(reward, target, patience, df["loss"].astype(float), target_loss)
               if target is not None else None)
        total = float(df["elapsed_seconds"].iloc[-1])
        episodes = df["num_episode_batches"].astype(int).cumsum()
        rows.append({
            "block": name,
            "batch_size": params.get("batch_size"),
            "num_envs": params.get("num_envs"),
            "seed": params.get("seed"),
            "target": target,
            "iterations": len(df),
            "episodes": int(episodes.iloc[-1]),
            "train_time_min": round(total / 60, 2),
            "sec_per_iter": round(total / len(df), 2),
            "target_reached_at_iter": None if hit is None else int(df["iteration"].iloc[hit]),
            "episodes_to_target": None if hit is None else int(episodes.iloc[hit]),
            "time_to_target_min": None if hit is None else round(float(df["elapsed_seconds"].iloc[hit]) / 60, 2),
            "best_rolling_reward": round(float(rolling.max()), 3),
            "last_rolling_reward": round(float(rolling.iloc[-1]), 3),
            "log": log.name,
        })
        curves.append((name, df["iteration"], episodes, df["elapsed_seconds"] / 60, rolling, target,
                       (params.get("batch_size"), params.get("num_envs")), params.get("seed")))

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.drop(columns=["log"], errors="ignore").to_string(index=False))

    if not curves:
        print("No iteration logs found: nothing to plot.")
        return
    fig, (ax_iter, ax_episodes, ax_time) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    axes = (ax_iter, ax_episodes, ax_time)
    # One colour per (batch_size, num_envs) setting, one line style per seed.
    colors = {setting: f"C{i}" for i, setting in enumerate(dict.fromkeys(c[6] for c in curves))}
    styles = {seed: ("-", "--", ":", "-.")[i % 4] for i, seed in enumerate(dict.fromkeys(c[7] for c in curves))}
    for name, iterations, episodes, minutes, rolling, _, setting, seed in curves:
        style = dict(label=name, color=colors[setting], linestyle=styles[seed])
        ax_iter.plot(iterations, rolling, **style)
        ax_episodes.plot(episodes, rolling, **style)
        ax_time.plot(minutes, rolling, **style)
    targets = {c[5] for c in curves if c[5] is not None}
    for target in targets:
        for ax in axes:
            ax.axhline(target, color="grey", linestyle="--", linewidth=1)
    ax_iter.set_xlabel("iteration (policy update)")
    ax_episodes.set_xlabel("episodes")
    ax_time.set_xlabel("training time [min]")
    ax_iter.set_ylabel(f"mean reward, rolling mean over {args.window} iterations")
    for ax in axes:
        ax.grid(alpha=0.3)
    ax_time.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Figure: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
