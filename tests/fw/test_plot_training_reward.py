"""
plot_training_reward.py: a single-agent CSV still gives one curve; a
multi-agent (MultiPPOAgent) CSV gives one curve per agent, with its own
rolling statistics, and a marker where an agent was frozen.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fw.postprocessing.plot_training_reward import load_data, plot_reward


def _labels(fig):
    """
    Legend labels of all lines on the figure's axes.
    """

    return [line.get_label() for line in fig.axes[0].get_lines()]


def test_single_agent_csv_gives_one_curve(tmp_path):
    """
    No agent_id column: unchanged single curve, synthetic iteration index.
    """

    csv = tmp_path / "training_metrics.csv"
    pd.DataFrame({"loss": np.ones(20), "advantage": np.zeros(20), "reward": np.linspace(0, 1, 20)}).to_csv(csv, index=False)
    fig = plot_reward(load_data(str(csv)), window=5, out_path=str(tmp_path / "single.png"))
    assert [l for l in _labels(fig) if not l.startswith("_")] == ["Reward (per iteration)", "Rolling mean (w = 5)"]
    assert (tmp_path / "single.png").exists()
    plt.close(fig)


def test_multi_agent_csv_gives_one_curve_per_agent_and_freeze_marker(tmp_path):
    """
    Interleaved agent rows are split per agent; agent 0's freeze is marked, agent 1 never froze.
    """

    rows = []
    for it in range(10):
        rows.append({"agent_id": 0, "iteration": it, "loss": 0.1, "advantage": 0.0, "reward": 1.0, "frozen": it >= 4})
        rows.append({"agent_id": 1, "iteration": it, "loss": 0.1, "advantage": 0.0, "reward": -1.0, "frozen": False})
    csv = tmp_path / "training_metrics.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    fig = plot_reward(load_data(str(csv)), window=3, out_path=str(tmp_path / "multi.png"))
    labels = [l for l in _labels(fig) if not l.startswith("_")]
    assert "Agent 0: Rolling mean (w = 3)" in labels and "Agent 1: Rolling mean (w = 3)" in labels
    assert "Agent 0: frozen" in labels and "Agent 1: frozen" not in labels

    lines = {line.get_label(): line for line in fig.axes[0].get_lines()}
    assert np.allclose(lines["Agent 0: Rolling mean (w = 3)"].get_ydata(), 1.0)    # not mixed with agent 1
    assert np.allclose(lines["Agent 1: Rolling mean (w = 3)"].get_ydata(), -1.0)
    assert lines["Agent 0: frozen"].get_xdata()[0] == 4
    plt.close(fig)
