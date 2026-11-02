"""
Evaluation plots for collision-avoidance environments.

Functions take the env as their first argument and read encounter / geometry
state directly from it. Kept out of the env module to keep that file focused
on simulation logic.
"""

from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from fw.collision_avoidance.scenario_classifier import EXTENDED_SCENARIO_LABELS


def plot_min_distance_stats(
    env,
    pdf_save_path: str = "min_distance_PDF.png",
    boxplot_save_path: str = "min_distance_boxplots.png",
    include_banks: bool = False,
) -> None:
    """
    Plot encounter-level min-distance statistics.

    Writes a histogram + KDE of the min center-to-center distance over all
    encounters (`pdf_save_path`) and boxplots of the same values per
    ground-truth scenario (`boxplot_save_path`). With `include_banks`, the
    per-encounter distance also accounts for the closest bank distance.

    Only encounters that resolved normally (TS passed and left the forget
    range) without collision are used; partial encounters would skew the
    distribution.

    Args:
        env: Collision-avoidance environment.
        pdf_save_path: Output path for the PDF plot.
        boxplot_save_path: Output path for the boxplot figure.
        include_banks: Whether to include bank clearance when computing the
            minimum distance.
    """

    # print(f"banks included: {include_banks}")

    # Finalize a still-active encounter; it did not resolve normally.
    env._finalize_active_encounter(resolved_normally=False)

    encounters = list(env._all_encounters)
    if env._episode_encounters:
        encounters.extend(env._episode_encounters)

    per_policy: dict[str, list[float]] = {}
    all_distances: list[float] = []
    for enc in encounters:
        if not enc.get("resolved", False) or enc.get("collided", False):
            continue
        md = enc.get("min_dist", float("inf"))
        if include_banks:
            md = min(md, enc.get("min_dist_bank", float("inf")))
        if md is None or md == float("inf"):
            continue
        gt = enc.get("gt") or "unknown"
        per_policy.setdefault(gt, []).append(float(md))
        all_distances.append(float(md))

    if not all_distances:
        print("No valid encounters to plot min-distance stats for.")
        return

    os_r = float(env.own_ship_radius)
    ts_r_min = float(env.static_ship_radius_min)
    ts_r_max = float(env.static_ship_radius_max)
    variable_ts = ts_r_min != ts_r_max
    collision_dist_min = os_r + ts_r_min
    collision_dist_max = os_r + ts_r_max

    _plot_min_distance_pdf(
        np.asarray(all_distances, dtype=float),
        os_r, ts_r_min, ts_r_max, variable_ts,
        collision_dist_min, collision_dist_max,
        pdf_save_path,
        include_banks=include_banks,
    )
    _plot_min_distance_boxplots(
        per_policy,
        collision_dist_min, collision_dist_max, variable_ts,
        boxplot_save_path,
        include_banks=include_banks,
    )


def _plot_min_distance_pdf(
    distances: np.ndarray,
    os_r: float,
    ts_r_min: float,
    ts_r_max: float,
    variable_ts: bool,
    collision_dist_min: float,
    collision_dist_max: float,
    save_path: str,
    include_banks: bool = False,
) -> None:
    """
    Plots the minimum-distance probability density.

    Args:
        distances: Minimum encounter distances.
        os_r: Own-ship radius.
        ts_r_min: Minimum target-ship radius.
        ts_r_max: Maximum target-ship radius.
        variable_ts: Whether target-ship radii vary.
        collision_dist_min: Minimum collision boundary.
        collision_dist_max: Maximum collision boundary.
        save_path: Output image path.
        include_banks: Whether bank distances are included.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    n_bins = max(10, len(distances) // 3)
    ax.hist(distances, bins=n_bins, density=True, alpha=0.6, color="steelblue",
            edgecolor="white", label="Histogram (density)")

    kde = gaussian_kde(distances)
    x_grid = np.linspace(max(0, distances.min() - 5), distances.max() + 10, 300)
    ax.plot(x_grid, kde(x_grid), color="darkblue", linewidth=2, label="KDE")

    arrow_coords = ("data", "axes fraction")
    arrow_kwargs = dict(arrowstyle="<->", lw=1.5)
    text_kwargs = dict(ha="center", va="top", fontsize=8, fontweight="bold")

    row1_y = -0.06
    if os_r > 0:
        ax.annotate("", xy=(os_r, row1_y), xytext=(0, row1_y),
                     arrowprops=dict(**arrow_kwargs, color="red"),
                     xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
        ax.text(os_r / 2, row1_y - 0.02, f"OS radius = {os_r:.0f}m",
                transform=ax.get_xaxis_transform(), color="red",
                clip_on=False, **text_kwargs)

    row2_y = row1_y - 0.08
    if variable_ts:
        ax.annotate("", xy=(os_r + ts_r_min, row2_y), xytext=(os_r, row2_y),
                     arrowprops=dict(**arrow_kwargs, color="orange"),
                     xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
        ax.text(os_r + ts_r_min / 2, row2_y - 0.02,
                f"min TS radius = {ts_r_min:.0f}m",
                transform=ax.get_xaxis_transform(), color="orange",
                clip_on=False, **text_kwargs)
        row3_y = row2_y - 0.08
        ax.annotate("", xy=(os_r + ts_r_max, row3_y), xytext=(os_r, row3_y),
                     arrowprops=dict(**arrow_kwargs, color="darkorange"),
                     xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
        ax.text(os_r + ts_r_max / 2, row3_y - 0.02,
                f"max TS radius = {ts_r_max:.0f}m",
                transform=ax.get_xaxis_transform(), color="darkorange",
                clip_on=False, **text_kwargs)
    else:
        ax.annotate("", xy=(os_r + ts_r_min, row2_y), xytext=(os_r, row2_y),
                     arrowprops=dict(**arrow_kwargs, color="orange"),
                     xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
        ax.text(os_r + ts_r_min / 2, row2_y - 0.02,
                f"TS radius = {ts_r_min:.0f}m",
                transform=ax.get_xaxis_transform(), color="orange",
                clip_on=False, **text_kwargs)

    ax.axvline(collision_dist_min, color="red", linestyle="--", alpha=0.6,
               label=f"Collision boundary (min) = {collision_dist_min:.0f}m")
    if variable_ts:
        ax.axvline(collision_dist_max, color="darkorange", linestyle="--", alpha=0.6,
                   label=f"Collision boundary (max) = {collision_dist_max:.0f}m")

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    xlabel_target = "ship or bank" if include_banks else "collision entity"
    ax.set_xlabel(f"Minimum center-to-center distance to {xlabel_target} [m]")
    ax.set_ylabel("Probability density")
    title_suffix = " incl. banks" if include_banks else ""
    ax.set_title(
        f"Min-distance PDF{title_suffix} "
        f"({len(distances)} encounters)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Min-distance PDF saved as '{save_path}'")
    plt.close(fig)


def _plot_min_distance_boxplots(
    per_policy: dict,
    collision_dist_min: float,
    collision_dist_max: float,
    variable_ts: bool,
    save_path: str,
    include_banks: bool = False,
) -> None:
    """
    Plots minimum-distance boxplots by scenario.

    Also exports the underlying encounter samples as a CSV file.

    Args:
        per_policy: Mapping from scenario labels to distance samples.
        collision_dist_min: Minimum collision boundary.
        collision_dist_max: Maximum collision boundary.
        variable_ts: Whether target-ship radii vary.
        save_path: Output image path.
        include_banks: Whether bank distances are included.
    """

    # Canonical scenario order first, then extras such as "unknown".
    ordered = [lbl for lbl in EXTENDED_SCENARIO_LABELS if per_policy.get(lbl)]
    ordered += [lbl for lbl in per_policy.keys()
                if lbl not in EXTENDED_SCENARIO_LABELS and per_policy[lbl]]
    if not ordered:
        print("No valid encounters with a scenario tag for boxplots.")
        return

    data = [per_policy[lbl] for lbl in ordered]
    counts = [len(v) for v in data]

    # Long-form CSV next to the image: one row per encounter (scenario, min_distance_m).
    csv_path = os.path.splitext(save_path)[0] + ".csv"
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "min_distance_m"])
            for lbl in ordered:
                for dist in per_policy[lbl]:
                    writer.writerow([lbl, f"{float(dist):.6f}"])
        print(f"Min-distance per-encounter samples saved as '{csv_path}'")
    except OSError as e:
        print(f"Warning: failed to write min-distance CSV '{csv_path}': {e}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        data,
        tick_labels=[f"{lbl}\n(n={c})" for lbl, c in zip(ordered, counts)],
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=12),
        medianprops=dict(color="darkblue", linewidth=4),
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        patch_artist=True,
    )

    ax.axhline(collision_dist_min, color="red", linestyle="--", alpha=0.6,
               label=f"Collision boundary (min) = {collision_dist_min:.0f}m")
    if variable_ts:
        ax.axhline(collision_dist_max, color="darkorange", linestyle="--", alpha=0.6,
                   label=f"Collision boundary (max) = {collision_dist_max:.0f}m")

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Minimum distance to TS or bank [m]", fontsize=16)
    #ax.set_xlabel("Policy", fontsize=18)
    title_suffix = " incl. banks" if include_banks else ""
    ax.set_title(
        f"Min-distance per policy{title_suffix} "
        f"({int(np.sum(counts))} encounters)",
        fontsize=20,
    )
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Min-distance boxplots saved as '{save_path}'")
    plt.close(fig)


def plot_scenario_confusion_matrix(
    env,
    save_path: str = "scenario_confusion_matrix.png",
) -> None:
    """
    Plot a confusion matrix over encounters aggregated across episodes.

    Rows: builder ground-truth scenario; columns: geometric classifier
    prediction at encounter entry. One entry per encounter.
    """

    # Include the current episode's encounters (not yet flushed by a reset).
    encounters = list(env._all_encounters)
    if env._episode_encounters:
        encounters.extend(env._episode_encounters)

    if not encounters:
        print("No encounters to plot scenario confusion matrix for.")
        return

    labels = list(EXTENDED_SCENARIO_LABELS)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(labels)

    matrix = np.zeros((n, n), dtype=np.int32)
    for enc in encounters:
        gt = enc.get("gt")
        pred = enc.get("pred")
        if gt not in label_to_idx or pred not in label_to_idx:
            continue
        matrix[label_to_idx[gt], label_to_idx[pred]] += 1

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = (correct / total * 100.0) if total else 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Geometric prediction (at encounter entry)")
    ax.set_ylabel("Ground truth (builder scenario)")
    ax.set_title(
        f"Scenario confusion matrix - {total} encounters, "
        f"accuracy {accuracy:.1f}% ({correct}/{total})"
    )

    # Overlay counts; white text on dark cells.
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            color = "white" if val > thresh else "black"
            ax.text(
                j, i, str(val),
                ha="center", va="center",
                color=color, fontsize=10, fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Encounter count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Scenario confusion matrix saved as '{save_path}'")
    plt.close(fig)
