from __future__ import annotations

import numpy as np

from collections import Counter
from dataclasses import dataclass, field
from fw.ship_episode_data import RunData, ShipEpisodeData


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ShipMetrics:
    """
    Aggregated performance metrics for a single ship over multiple episodes.

    This dataclass stores summary statistics computed from a set of episodes,
    including tracking performance, control effort, motion behavior, and reward
    statistics.

    Attributes:
        ship_id: Identifier of the ship.

        mTEI: Mean Track Error Integral (average integrated cross-track error).
        MTE: Maximum Track Error (peak absolute cross-track error).

        mRA: Mean Rudder Angle (average absolute rudder command).
        MRA: Maximum Rudder Angle.

        mRAC: Mean Rudder Angle Change (control smoothness indicator).

        mHE: Mean Heading Error (average absolute heading deviation).

        mu: Mean surge velocity magnitude.
        Mu: Maximum surge velocity magnitude.

        avg_reward: Mean per-step reward averaged across episodes.
        best_reward: Best (highest) normalized episode reward.
        worst_reward: Worst (lowest) normalized episode reward.
    """

    ship_id: str
    mTEI: float = 0.0   # Mean Track Error Integral
    MTE: float = 0.0    # Maximum Track Error
    mRA: float = 0.0    # Mean Rudder Angle
    MRA: float = 0.0    # Max Rudder Angle
    mRAC: float = 0.0   # Mean Rudder Angle Change
    mHE: float = 0.0    # Mean Heading Error
    mu: float = 0.0     # Mean surge velocity
    Mu: float = 0.0     # Max surge velocity
    avg_reward: float = 0.0
    best_reward: float = 0.0
    worst_reward: float = 0.0
    # Env-reported collision / COLREG metrics, aggregated across episodes.
    num_episodes: int = 0
    collision_count: int = 0
    collision_type_count: dict = field(default_factory=dict)
    collision_side_count: dict = field(default_factory=dict)
    colreg_violation_count: int = 0
    # Scenario classification diagnostics, aggregated per-encounter across
    # all episodes (5-way extended label space). In step-level generation an
    # episode may contain multiple encounters; each contributes one record.
    num_encounters: int = 0
    encounter_misclassification_count: int = 0
    encounter_confusion: dict = field(default_factory=dict)   # {(gt, pred): count}


class MetricsComputer:
    """
    Computes aggregated per-ship metrics from evaluation run data.

    This class processes `RunData` (containing multiple episodes per ship)
    and computes summary statistics such as tracking error, control effort,
    velocity statistics, and reward performance.

    Typical usage:
        metrics = MetricsComputer.compute(run_data)
    """

    @staticmethod
    def compute(run_data: RunData) -> dict[str, ShipMetrics]:
        """
        Computes metrics for all ships in a run.

        Args:
            run_data: Evaluation output containing per-ship episode data.

        Returns:
            Dictionary mapping ship IDs to computed `ShipMetrics`.
        """

        result = {}
        for ship_id, episodes in run_data.ships.items():
            result[ship_id] = MetricsComputer._compute_ship(ship_id, episodes)
        return result

    @staticmethod
    def _compute_ship(ship_id: str, episodes: list[ShipEpisodeData]) -> ShipMetrics:
        """
        Computes aggregate metrics for a single ship.

        Args:
            ship_id: Identifier of the ship.
            episodes: List of `ShipEpisodeData` objects for that ship.

        Returns:
            ShipMetrics containing aggregated statistics.

        Notes:
            - Metrics are averaged across episodes where applicable.
            - Cross-track error is averaged per episode (nanmean).
            - mTEI is a mean absolute cross-track error.
            - Reward metrics are normalized by number of steps per episode.
            - Episodes with zero steps are excluded from reward normalisation.
        """

        m = ShipMetrics(ship_id=ship_id)
        if not episodes:
            return m

        rewards = np.array([ep.reward for ep in episodes], dtype=float)

        # Use rudder_actions length as step count proxy; fall back to path length.
        step_counts = np.array([
            ep.rudder_actions.shape[0] if ep.rudder_actions.size > 0
            else max(ep.path.shape[0], 1)
            for ep in episodes
        ], dtype=float)

        # Guard against zero-length episodes to avoid division by zero.
        safe_steps = np.where(step_counts > 0, step_counts, 1.0)
        normalized_rewards = rewards / safe_steps

        m.avg_reward   = float(np.mean(normalized_rewards))
        m.best_reward  = float(np.max(normalized_rewards))
        m.worst_reward = float(np.min(normalized_rewards))

        per_ep_mTEI, per_ep_MTE = [], []
        per_ep_mRA, per_ep_MRA, per_ep_mRAC = [], [], []
        per_ep_mHE, per_ep_mu, per_ep_Mu = [], [], []

        for ep in episodes:
            if ep.cross_errors.size:
                abs_cte = np.abs(ep.cross_errors)
                per_ep_mTEI.append(float(np.nanmean(abs_cte)))   # correct mean
                per_ep_MTE.append(float(np.nanmax(abs_cte)))
            if ep.rudder_actions.size:
                ra = np.abs(ep.rudder_actions)
                per_ep_mRA.append(float(np.nanmean(ra)))
                per_ep_MRA.append(float(np.nanmax(ra)))
            if ep.rudder_changes.size:
                per_ep_mRAC.append(float(np.nanmean(np.abs(ep.rudder_changes))))
            if ep.heading_errors.size:
                per_ep_mHE.append(float(np.nanmean(np.abs(ep.heading_errors))))
            if ep.velocities.size:
                v = np.abs(ep.velocities).ravel()
                per_ep_mu.append(float(np.nanmean(v)))
                per_ep_Mu.append(float(np.nanmax(v)))

        def safe_mean(lst: list) -> float:
            return float(np.nanmean(lst)) if lst else 0.0

        def safe_max(lst: list) -> float:
            return float(np.nanmax(lst)) if lst else 0.0

        m.mTEI = safe_mean(per_ep_mTEI)
        m.MTE  = safe_max(per_ep_MTE)
        m.mRA  = safe_mean(per_ep_mRA)
        m.MRA  = safe_max(per_ep_MRA)
        m.mRAC = safe_mean(per_ep_mRAC)
        m.mHE  = safe_mean(per_ep_mHE)
        m.mu   = safe_mean(per_ep_mu)
        m.Mu   = safe_max(per_ep_Mu)

        # Aggregate env-reported collision / COLREG info.
        m.num_episodes = len(episodes)
        m.collision_count = sum(1 for ep in episodes if ep.collision)
        m.collision_type_count = dict(
            Counter(ep.collision_type for ep in episodes if ep.collision and ep.collision_type)
        )
        m.collision_side_count = dict(
            Counter(ep.collision_side for ep in episodes if ep.collision and ep.collision_side)
        )
        m.colreg_violation_count = sum(int(ep.colreg_violations) for ep in episodes)

        # Classification diagnostics — one record per encounter.
        confusion: dict[tuple[str, str], int] = {}
        mismatches = 0
        total = 0
        for ep in episodes:
            for enc in ep.scenario_encounters:
                gt = enc.get("gt")
                pred = enc.get("pred")
                if gt is None or pred is None:
                    continue
                total += 1
                key = (gt, pred)
                confusion[key] = confusion.get(key, 0) + 1
                if gt != pred:
                    mismatches += 1

        m.num_encounters = total
        m.encounter_misclassification_count = mismatches
        m.encounter_confusion = confusion

        return m
