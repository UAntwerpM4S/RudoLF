"""Evaluation plots for collision-avoidance environments.

Functions take the env as their first argument and read encounter / geometry
state directly from it. Kept out of the env module to keep that file focused
on simulation logic.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from fw.collision_avoidance.scenario_classifier import EXTENDED_SCENARIO_LABELS


def plot_min_distance_stats(
    env,
    pdf_save_path: str = "min_distance_PDF.png",
    boxplot_save_path: str = "min_distance_boxplots.png",
    paired_boxplot_save_path: str = "min_distance_paired_boxplots.png",
    include_banks: bool = False,
) -> None:
    """Plot encounter-level min-distance statistics.

    Generates two images:
      * ``pdf_save_path`` — histogram + KDE of min center-to-center distance
        aggregated across all non-collision encounters (no per-policy split).
      * ``boxplot_save_path`` — per-policy/scenario boxplots of the same
        min-distance values, bucketed by the encounter's ground-truth label.

    When ``include_banks`` is True, the per-encounter distance is the minimum
    of the closest target-ship distance and the closest distance to the bank
    (polygon boundary) observed during the encounter; otherwise only
    target-ship distance is used.

    Only encounters that did not end in collision are included.
    """
    # Make sure any encounter still active at call time is finalized.
    env._finalize_active_encounter()

    encounters = list(env._all_encounters)
    if env._episode_encounters:
        encounters.extend(env._episode_encounters)

    per_policy: dict[str, list[float]] = {}
    per_policy_avail: dict[str, list[float]] = {}
    all_distances: list[float] = []
    for enc in encounters:
        md = enc.get("min_dist", float("inf"))
        if include_banks:
            md = min(md, enc.get("min_dist_bank", float("inf")))
        if enc.get("collided", False) or md is None or md == float("inf"):
            continue
        gt = enc.get("gt") or "unknown"
        per_policy.setdefault(gt, []).append(float(md))
        all_distances.append(float(md))

        # Pair this encounter's min_dist with the tightest corridor width
        # the OS path passed through during the encounter. Use right-only
        # for COLREG-pinned starboard scenarios; full corridor where the
        # policy is free to pick either side.
        if gt in ("head-on", "crossing-right"):
            avail = enc.get("min_avail_right", float("inf"))
        elif gt == "overtaking":
            avail = enc.get("min_avail_full", float("inf"))
        else:
            avail = float("inf")
        per_policy_avail.setdefault(gt, []).append(float(avail))

    if not all_distances:
        print("No non-collision encounters to plot min-distance stats for.")
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
    )
    _plot_min_distance_paired_boxplots(
        per_policy,
        per_policy_avail,
        collision_dist_min, collision_dist_max, variable_ts,
        paired_boxplot_save_path,
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
        f"({len(distances)} non-collision encounters)"
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
) -> None:
    # Order known labels first (canonical scenario order), then any
    # extras encountered (e.g. "unknown") tacked on at the end.
    ordered = [lbl for lbl in EXTENDED_SCENARIO_LABELS if per_policy.get(lbl)]
    ordered += [lbl for lbl in per_policy.keys()
                if lbl not in EXTENDED_SCENARIO_LABELS and per_policy[lbl]]
    if not ordered:
        print("No non-collision encounters with a scenario tag for boxplots.")
        return

    data = [per_policy[lbl] for lbl in ordered]
    counts = [len(v) for v in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        data,
        labels=[f"{lbl}\n(n={c})" for lbl, c in zip(ordered, counts)],
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=6),
        medianprops=dict(color="darkblue", linewidth=2),
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        patch_artist=True,
    )

    ax.axhline(collision_dist_min, color="red", linestyle="--", alpha=0.6,
               label=f"Collision boundary (min) = {collision_dist_min:.0f}m")
    if variable_ts:
        ax.axhline(collision_dist_max, color="darkorange", linestyle="--", alpha=0.6,
                   label=f"Collision boundary (max) = {collision_dist_max:.0f}m")

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Minimum center-to-center distance to collision entity [m]")
    ax.set_xlabel("Scenario / policy")
    ax.set_title(
        f"Min-distance per policy ({int(np.sum(counts))} non-collision encounters)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Min-distance boxplots saved as '{save_path}'")
    plt.close(fig)


def _plot_min_distance_paired_boxplots(
    per_policy: dict,
    per_policy_avail: dict,
    collision_dist_min: float,
    collision_dist_max: float,
    variable_ts: bool,
    save_path: str,
) -> None:
    """Side-by-side boxes per scenario: achieved min-distance vs. available
    corridor width. Right-bank-only width for head-on / crossing-right (COLREG
    pins starboard); full corridor for overtaking (policy free to pick a side).
    The gap between the two boxes is the headroom the policy left unused."""
    ordered = [lbl for lbl in EXTENDED_SCENARIO_LABELS if per_policy.get(lbl)]
    ordered += [lbl for lbl in per_policy.keys()
                if lbl not in EXTENDED_SCENARIO_LABELS and per_policy[lbl]]
    if not ordered:
        print("No non-collision encounters with a scenario tag for paired boxplots.")
        return

    width_kind = {
        "head-on": "right-bank",
        "crossing-right": "right-bank",
        "overtaking": "full corridor",
    }

    md_data = []
    aw_data = []
    labels = []
    for lbl in ordered:
        mds = per_policy.get(lbl, [])
        aws_raw = per_policy_avail.get(lbl, [])
        aws = [v for v in aws_raw if v != float("inf")]
        if not mds:
            continue
        md_data.append(mds)
        aw_data.append(aws)
        kind = width_kind.get(lbl, "n/a")
        labels.append(f"{lbl}\n(n={len(mds)})\nwidth: {kind}")

    n_groups = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, 3 * n_groups), 7))

    group_centers = np.arange(1, n_groups + 1, dtype=float)
    box_offset = 0.22
    box_width = 0.34

    md_positions = group_centers - box_offset
    aw_positions = group_centers + box_offset

    md_bp = ax.boxplot(
        md_data,
        positions=md_positions,
        widths=box_width,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=6),
        medianprops=dict(color="darkblue", linewidth=2),
        boxprops=dict(facecolor="steelblue", alpha=0.7),
        patch_artist=True,
    )
    aw_bp = ax.boxplot(
        [aws if aws else [np.nan] for aws in aw_data],
        positions=aw_positions,
        widths=box_width,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=6),
        medianprops=dict(color="darkgreen", linewidth=2),
        boxprops=dict(facecolor="seagreen", alpha=0.5),
        patch_artist=True,
    )

    ax.axhline(collision_dist_min, color="red", linestyle="--", alpha=0.6,
               label=f"Collision boundary (min) = {collision_dist_min:.0f}m")
    if variable_ts:
        ax.axhline(collision_dist_max, color="darkorange", linestyle="--", alpha=0.6,
                   label=f"Collision boundary (max) = {collision_dist_max:.0f}m")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.4, n_groups + 0.6)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Distance [m]")
    ax.set_xlabel("Scenario / policy")

    total_n = int(sum(len(v) for v in md_data))
    ax.set_title(
        f"Min-distance vs. available corridor width "
        f"({total_n} non-collision encounters)"
    )
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="steelblue", alpha=0.7,
              label="Min center-to-center distance to collision entity"),
        Patch(facecolor="seagreen", alpha=0.5,
              label="Min available corridor width at OS path-foot"),
    ]
    existing_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=legend_handles + existing_handles, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Min-distance paired boxplots saved as '{save_path}'")
    plt.close(fig)


def plot_scenario_confusion_matrix(
    env,
    save_path: str = "scenario_confusion_matrix.png",
) -> None:
    """Plot a confusion matrix over encounters aggregated across episodes.

    Rows = builder ground-truth scenario, columns = geometric classifier
    prediction at encounter entry. One entry per encounter (multiple
    encounters may occur per episode in step-level generation).
    """
    # Flush current episode's encounters if the run ended without another
    # env_specific_reset() being called.
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
        f"Scenario confusion matrix — {total} encounters, "
        f"accuracy {accuracy:.1f}% ({correct}/{total})"
    )

    # Overlay counts. Use white text on dark cells for readability.
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
