from __future__ import annotations

import math
import numpy as np

from collections import Counter
from dataclasses import dataclass, field
from fw.postprocessing.ship_episode_data import RunData, ShipEpisodeData


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ShipMetrics:
    """
    Metrics for one ship, aggregated over episodes.

    Per-episode values are averaged over episodes; maxima (MTE, MRA, MT, Mu)
    are the maximum over all episodes. Rewards are per step.

    Attributes:
        mTEI: Mean absolute cross-track error ("track error integral").
        MTE: Maximum absolute cross-track error.
        mRA / MRA: Mean / maximum absolute rudder command.
        mRAC: Mean absolute rudder change (smoothness).
        mT / MT: Mean / maximum absolute thrust command.
        mTC: Mean absolute thrust change.
        vRA: Variance of the rudder command.
        sCF: Rudder sign-change frequency (oscillation).
        mHE: Mean absolute heading error.
        mu / Mu: Mean / maximum absolute surge velocity.
        avg_reward / best_reward / worst_reward: Mean, best and worst
            per-step episode reward.

    The remaining fields are COLAV metrics. "Per policy" means per
    ground-truth scenario (gt). A valid encounter resolved normally (OS passed
    the TS and the TS left forget_range) without collision; CRI, CTE and
    time-to-first-action (TFA) aggregates use valid encounters only.
    """

    ship_id: str
    mTEI: float = 0.0   # Mean Track Error Integral
    MTE: float = 0.0    # Maximum Track Error
    mRA: float = 0.0    # Mean Rudder Angle
    MRA: float = 0.0    # Max Rudder Angle
    mRAC: float = 0.0   # Mean Rudder Angle Change
    mT: float = 0.0     # Mean thrust
    MT: float = 0.0     # Max thrust
    mTC: float = 0.0    # Mean thrust change
    vRA: float = 0.0    # Steering Variance
    sCF: float = 0.0    # Steering Sign-Change Frequency
    mHE: float = 0.0    # Mean Heading Error
    mu: float = 0.0     # Mean surge velocity
    Mu: float = 0.0     # Max surge velocity
    avg_reward: float = 0.0
    best_reward: float = 0.0
    worst_reward: float = 0.0
    # Env-reported collision / COLREG counts
    num_episodes: int = 0
    collision_count: int = 0
    collision_type_count: dict = field(default_factory=dict)
    collision_side_count: dict = field(default_factory=dict)
    colreg_violation_count: int = 0
    # Scenario classification, one record per encounter (5-way extended labels)
    num_encounters: int = 0
    encounter_misclassification_count: int = 0
    encounter_confusion: dict = field(default_factory=dict)   # {(gt, pred): count}
    # Collision-rate denominators. encounters_per_policy counts encounters with
    # a determined outcome (collided, or passed and tracked_lost); encounters
    # cut off by episode end are excluded. Collisions outside encounters (e.g.
    # groundings while path-following) are counted separately.
    encounters_per_policy: dict = field(default_factory=dict)         # {gt: valid_outcome_count}
    total_encounters_per_policy: dict = field(default_factory=dict)   # {gt: all_encounter_starts}
    collisions_per_policy: dict = field(default_factory=dict)         # {gt: collided_encounter_count}
    collisions_outside_encounters: int = 0
    # Outcome categories: "collided", "resolved", "passed_not_tracked_lost",
    # "tracked_lost_not_passed", "neither"; sums to total_encounters_per_policy.
    encounter_outcomes_per_policy: dict = field(default_factory=dict) # {gt: {category: count}}
    # CRI over valid encounters
    mean_cri_per_policy: dict = field(default_factory=dict)           # {gt: mean of per-encounter mean CRI}
    median_cri_per_policy: dict = field(default_factory=dict)         # {gt: median of per-encounter mean CRI}
    max_cri_per_policy: dict = field(default_factory=dict)            # {gt: max of per-encounter max CRI}
    valid_encounters_per_policy: dict = field(default_factory=dict)   # {gt: valid encounter count}
    # Cross-track error over valid encounters
    mean_cte_per_policy: dict = field(default_factory=dict)           # {gt: mean of per-encounter mean CTE [m]}
    median_cte_per_policy: dict = field(default_factory=dict)         # {gt: median of per-encounter mean CTE [m]}
    max_cte_per_policy: dict = field(default_factory=dict)            # {gt: max of per-encounter max CTE [m]}
    # Time to first rudder command above the threshold (default 10 deg), over
    # valid encounters. Encounters without such a command are only counted
    # in non_reactive_encounters_per_policy.
    mean_tfa_per_policy: dict = field(default_factory=dict)           # {gt: mean time-to-first-action [s]}
    median_tfa_per_policy: dict = field(default_factory=dict)         # {gt: median time-to-first-action [s]}
    reactive_encounters_per_policy: dict = field(default_factory=dict)     # {gt: count with reaction}
    non_reactive_encounters_per_policy: dict = field(default_factory=dict) # {gt: count without reaction}
    # Baseline CRI floor for reading the per-policy values. Empirical: mean
    # and min of per-encounter minimum CRI over all valid encounters (all
    # policies). Theoretical: CRI of a TS at encounter_range with TCPA = 0,
    # filled in by the caller (agent_manager).
    baseline_cri_empirical_mean: float = 0.0
    baseline_cri_empirical_min: float = 0.0
    baseline_cri_theoretical: float = 0.0
    baseline_cri_sample_count: int = 0


class MetricsComputer:
    """
    Compute per-ship metrics from evaluation RunData.

    Usage:
        metrics = MetricsComputer.compute(run_data)
    """

    @staticmethod
    def compute(run_data: RunData) -> dict[str, ShipMetrics]:
        """
        Return {ship_id: ShipMetrics} for all ships in run_data.
        """

        result = {}
        for ship_id, episodes in run_data.ships.items():
            result[ship_id] = MetricsComputer._compute_ship(ship_id, episodes)
        return result

    @staticmethod
    def _compute_ship(ship_id: str, episodes: list[ShipEpisodeData]) -> ShipMetrics:
        """
        Aggregate the metrics of one ship over its episodes (see ShipMetrics).
        """

        m = ShipMetrics(ship_id=ship_id)
        if not episodes:
            return m

        rewards = np.array([ep.reward for ep in episodes], dtype=float)

        # Step count: number of rudder actions, else path length (at least 1)
        step_counts = np.array([
            ep.rudder_actions.shape[0] if ep.rudder_actions.size > 0
            else max(ep.path.shape[0], 1)
            for ep in episodes
        ], dtype=float)

        safe_steps = np.where(step_counts > 0, step_counts, 1.0)
        normalized_rewards = rewards / safe_steps

        m.avg_reward   = float(np.mean(normalized_rewards))
        m.best_reward  = float(np.max(normalized_rewards))
        m.worst_reward = float(np.min(normalized_rewards))

        per_ep_mTEI, per_ep_MTE = [], []
        per_ep_mRA, per_ep_MRA, per_ep_mRAC = [], [], []
        per_ep_mT, per_ep_MT, per_ep_mTC = [], [], []
        per_ep_vRA, per_ep_sCF = [], []
        per_ep_mHE, per_ep_mu, per_ep_Mu = [], [], []

        for ep in episodes:
            if ep.cross_errors.size:
                abs_cte = np.abs(ep.cross_errors)
                per_ep_mTEI.append(float(np.nanmean(abs_cte)))
                per_ep_MTE.append(float(np.nanmax(abs_cte)))
            if ep.rudder_actions.size:
                ra_raw = ep.rudder_actions
                ra_abs = np.abs(ra_raw)
                per_ep_mRA.append(float(np.nanmean(ra_abs)))
                per_ep_MRA.append(float(np.nanmax(ra_abs)))
                per_ep_vRA.append(float(np.nanvar(ra_raw)))
                
                # Sign-change frequency
                if ra_raw.size > 1:
                    signs = np.sign(ra_raw)
                    # Ignore zeros so passing through 0 counts once
                    nz_signs = signs[signs != 0]
                    if nz_signs.size > 1:
                        sign_changes = np.sum(nz_signs[:-1] != nz_signs[1:])
                        per_ep_sCF.append(float(sign_changes / (ra_raw.size - 1)))
                    else:
                        per_ep_sCF.append(0.0)
                else:
                    per_ep_sCF.append(0.0)

            if ep.rudder_changes.size:
                per_ep_mRAC.append(float(np.nanmean(np.abs(ep.rudder_changes))))
            if ep.thrust_actions.size:
                thrust = np.abs(ep.thrust_actions)
                per_ep_mT.append(float(np.nanmean(thrust)))
                per_ep_MT.append(float(np.nanmax(thrust)))
            if ep.thrust_changes.size:
                per_ep_mTC.append(float(np.nanmean(np.abs(ep.thrust_changes))))
            if ep.heading_errors.size:
                per_ep_mHE.append(float(np.nanmean(np.abs(ep.heading_errors))))
            if ep.velocities.size:
                v = np.abs(ep.velocities).ravel()
                per_ep_mu.append(float(np.nanmean(v)))
                per_ep_Mu.append(float(np.nanmax(v)))

        def safe_mean(lst: list) -> float:
            """
            NaN-ignoring mean; 0.0 for an empty list.
            """

            return float(np.nanmean(lst)) if lst else 0.0

        def safe_max(lst: list) -> float:
            """
            NaN-ignoring max; 0.0 for an empty list.
            """

            return float(np.nanmax(lst)) if lst else 0.0

        m.mTEI = safe_mean(per_ep_mTEI)
        m.MTE  = safe_max(per_ep_MTE)
        m.mRA  = safe_mean(per_ep_mRA)
        m.MRA  = safe_max(per_ep_MRA)
        m.mRAC = safe_mean(per_ep_mRAC)
        m.mT   = safe_mean(per_ep_mT)
        m.MT   = safe_max(per_ep_MT)
        m.mTC  = safe_mean(per_ep_mTC)
        m.vRA  = safe_mean(per_ep_vRA)
        m.sCF  = safe_mean(per_ep_sCF)
        m.mHE  = safe_mean(per_ep_mHE)
        m.mu   = safe_mean(per_ep_mu)
        m.Mu   = safe_max(per_ep_Mu)

        m.num_episodes = len(episodes)
        m.collision_type_count = dict(
            Counter(ep.collision_type for ep in episodes if ep.collision and ep.collision_type)
        )
        m.collision_side_count = dict(
            Counter(ep.collision_side for ep in episodes if ep.collision and ep.collision_side)
        )
        m.colreg_violation_count = sum(int(ep.colreg_violations) for ep in episodes)

        # Per-encounter records. num_encounters counts all encounter starts;
        # classification metrics use labeled ones only. collision_count also
        # includes collisions outside encounters, which have no gt scenario.
        confusion: dict[tuple[str, str], int] = {}
        mismatches = 0
        total = 0
        classified = 0
        collided_encounters = 0
        valid_encounters_per_policy: dict[str, int] = {}
        total_encounters_per_policy: dict[str, int] = {}
        collisions_per_policy: dict[str, int] = {}
        outcomes_per_policy: dict[str, dict[str, int]] = {}
        collisions_outside_encounters = 0
        mean_cri_samples_per_policy: dict[str, list[float]] = {}
        max_cri_per_policy: dict[str, float] = {}
        mean_cte_samples_per_policy: dict[str, list[float]] = {}
        max_cte_per_policy: dict[str, float] = {}
        tfa_samples_per_policy: dict[str, list[float]] = {}
        non_reactive_per_policy: dict[str, int] = {}
        # Per-encounter min CRI over all policies (empirical baseline)
        min_cri_samples_all_policies: list[float] = []
        for ep in episodes:
            encounter_collision_in_ep = False
            for enc in ep.scenario_encounters:
                total += 1
                gt = enc.get("gt")
                collided = bool(enc.get("collided", False))
                passed = bool(enc.get("passed", False))
                tracked_lost = bool(enc.get("tracked_lost", False))
                # Determined outcome: collision, or fully resolved
                valid_outcome = collided or (passed and tracked_lost)
                if gt is not None:
                    total_encounters_per_policy[gt] = total_encounters_per_policy.get(gt, 0) + 1
                    if valid_outcome:
                        valid_encounters_per_policy[gt] = valid_encounters_per_policy.get(gt, 0) + 1
                    if collided:
                        cat = "collided"
                    elif passed and tracked_lost:
                        cat = "resolved"
                    elif passed and not tracked_lost:
                        cat = "passed_not_tracked_lost"
                    elif tracked_lost and not passed:
                        cat = "tracked_lost_not_passed"
                    else:
                        cat = "neither"
                    bucket = outcomes_per_policy.setdefault(gt, {})
                    bucket[cat] = bucket.get(cat, 0) + 1
                if collided:
                    collided_encounters += 1
                    encounter_collision_in_ep = True
                    if gt is not None:
                        collisions_per_policy[gt] = collisions_per_policy.get(gt, 0) + 1
                # CRI, CTE and TFA samples from valid encounters only
                if (
                    gt is not None
                    and enc.get("resolved", False)
                    and not enc.get("collided", False)
                ):
                    enc_mean_cri = float(enc.get("mean_cri", 0.0))
                    enc_max_cri = float(enc.get("max_cri", 0.0))
                    mean_cri_samples_per_policy.setdefault(gt, []).append(enc_mean_cri)
                    if enc_max_cri > max_cri_per_policy.get(gt, 0.0):
                        max_cri_per_policy[gt] = enc_max_cri
                    enc_min_cri = float(enc.get("min_cri", 0.0))
                    if math.isfinite(enc_min_cri):
                        min_cri_samples_all_policies.append(enc_min_cri)
                    enc_mean_cte = float(enc.get("mean_cte", 0.0))
                    enc_max_cte = float(enc.get("max_cte", 0.0))
                    mean_cte_samples_per_policy.setdefault(gt, []).append(enc_mean_cte)
                    if enc_max_cte > max_cte_per_policy.get(gt, 0.0):
                        max_cte_per_policy[gt] = enc_max_cte
                    # None: no rudder command above the threshold
                    tfa = enc.get("time_to_first_action", None)
                    if tfa is None:
                        non_reactive_per_policy[gt] = non_reactive_per_policy.get(gt, 0) + 1
                    else:
                        tfa_samples_per_policy.setdefault(gt, []).append(float(tfa))
                pred = enc.get("pred")
                if gt is None or pred is None:
                    continue
                classified += 1
                key = (gt, pred)
                confusion[key] = confusion.get(key, 0) + 1
                if gt != pred:
                    mismatches += 1
            # Collision outside any encounter (e.g. grounding while path-following)
            if ep.collision and not encounter_collision_in_ep:
                collisions_outside_encounters += 1

        m.num_encounters = total
        m.collision_count = collided_encounters + collisions_outside_encounters
        m.encounters_per_policy = valid_encounters_per_policy
        m.total_encounters_per_policy = total_encounters_per_policy
        m.collisions_per_policy = collisions_per_policy
        m.encounter_outcomes_per_policy = outcomes_per_policy
        m.collisions_outside_encounters = collisions_outside_encounters
        m.encounter_misclassification_count = mismatches
        m.encounter_confusion = confusion
        m.mean_cri_per_policy = {
            gt: float(np.mean(samples)) for gt, samples in mean_cri_samples_per_policy.items()
        }
        m.median_cri_per_policy = {
            gt: float(np.median(samples)) for gt, samples in mean_cri_samples_per_policy.items()
        }
        m.max_cri_per_policy = max_cri_per_policy
        m.valid_encounters_per_policy = {
            gt: len(samples) for gt, samples in mean_cri_samples_per_policy.items()
        }
        m.mean_cte_per_policy = {
            gt: float(np.mean(samples)) for gt, samples in mean_cte_samples_per_policy.items()
        }
        m.median_cte_per_policy = {
            gt: float(np.median(samples)) for gt, samples in mean_cte_samples_per_policy.items()
        }
        m.max_cte_per_policy = max_cte_per_policy
        m.mean_tfa_per_policy = {
            gt: float(np.mean(samples)) for gt, samples in tfa_samples_per_policy.items()
        }
        m.median_tfa_per_policy = {
            gt: float(np.median(samples)) for gt, samples in tfa_samples_per_policy.items()
        }
        m.reactive_encounters_per_policy = {
            gt: len(samples) for gt, samples in tfa_samples_per_policy.items()
        }
        m.non_reactive_encounters_per_policy = non_reactive_per_policy
        if min_cri_samples_all_policies:
            m.baseline_cri_empirical_mean = float(np.mean(min_cri_samples_all_policies))
            m.baseline_cri_empirical_min = float(np.min(min_cri_samples_all_policies))
            m.baseline_cri_sample_count = len(min_cri_samples_all_policies)
        # baseline_cri_theoretical needs env constants; agent_manager sets it.

        return m
