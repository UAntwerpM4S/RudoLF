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

        mT: Mean Thrust (average absolute thrust command).
        MT: Maximum Thrust.
        mTC: Mean Thrust Change (thrust smoothness indicator).

        vRA: Steering Variance (spread of rudder commands).
        sCF: Steering Sign-Change Frequency (oscillation indicator).

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
    # Per-policy (gt scenario) collision counts and encounter denominators.
    # Outside-encounter collisions (e.g. bank groundings while path-following)
    # are tracked separately since they have no encounter denominator.
    # encounters_per_policy is the *valid-outcome* denominator used by the
    # collision rate: collided OR (passed AND tracked_lost). Encounters cut
    # off by episode termination/truncation are excluded since they weren't
    # exposed to the full range of dangers. total_encounters_per_policy
    # retains the raw count of all encounter starts for diagnostics.
    encounters_per_policy: dict = field(default_factory=dict)         # {gt: valid_outcome_count}
    total_encounters_per_policy: dict = field(default_factory=dict)   # {gt: all_encounter_starts}
    collisions_per_policy: dict = field(default_factory=dict)         # {gt: collided_encounter_count}
    collisions_outside_encounters: int = 0
    # Per-policy outcome breakdown over all encounter starts. Categories:
    # "collided", "resolved", "passed_not_tracked_lost",
    # "tracked_lost_not_passed", "neither". Sums to total_encounters_per_policy.
    encounter_outcomes_per_policy: dict = field(default_factory=dict) # {gt: {category: count}}
    # Per-policy CRI aggregates over *valid* encounters only. An encounter
    # is "valid" when it resolved normally (OS passed the TS and the TS
    # subsequently left the forget range) and did not end in collision.
    # mean_cri_per_policy is the average of per-encounter mean-CRI samples,
    # median_cri_per_policy the median of those same samples, and
    # max_cri_per_policy the maximum of per-encounter max-CRI samples.
    # valid_encounters_per_policy gives the sample count behind each entry.
    mean_cri_per_policy: dict = field(default_factory=dict)           # {gt: mean of per-encounter mean CRI}
    median_cri_per_policy: dict = field(default_factory=dict)         # {gt: median of per-encounter mean CRI}
    max_cri_per_policy: dict = field(default_factory=dict)            # {gt: max of per-encounter max CRI}
    valid_encounters_per_policy: dict = field(default_factory=dict)   # {gt: valid encounter count}
    # Per-policy cross-track error aggregates over the same *valid*
    # encounters as the CRI metrics. mean_cte_per_policy / median_cte_per_policy
    # are computed from per-encounter mean-CTE samples; max_cte_per_policy
    # takes the maximum of per-encounter max-CTE samples.
    mean_cte_per_policy: dict = field(default_factory=dict)           # {gt: mean of per-encounter mean CTE [m]}
    median_cte_per_policy: dict = field(default_factory=dict)         # {gt: median of per-encounter mean CTE [m]}
    max_cte_per_policy: dict = field(default_factory=dict)            # {gt: max of per-encounter max CTE [m]}
    # Per-policy time-to-first-action aggregates, in seconds. Restricted to
    # valid encounters where the OS commanded a rudder above the configured
    # threshold (default 10 deg). reactive_encounters_per_policy gives the
    # sample count; non_reactive_encounters_per_policy counts valid encounters
    # in which no rudder command crossed the threshold (excluded from the
    # mean/median aggregates to avoid skewing them with a fall-back value).
    mean_tfa_per_policy: dict = field(default_factory=dict)           # {gt: mean time-to-first-action [s]}
    median_tfa_per_policy: dict = field(default_factory=dict)         # {gt: median time-to-first-action [s]}
    reactive_encounters_per_policy: dict = field(default_factory=dict)     # {gt: count with reaction}
    non_reactive_encounters_per_policy: dict = field(default_factory=dict) # {gt: count without reaction}
    # Baseline CRI: a "floor" against which the per-policy CRI numbers can
    # be read. Empirical baseline is the mean (and absolute min) of the
    # per-encounter MIN-CRI samples across all valid encounters from every
    # policy - i.e. "the lowest CRI value samples typically hit in any
    # encounter, regardless of policy". Theoretical baseline is what the
    # CRI formula yields when a TS sits exactly at encounter_range with
    # TCPA = 0 (parting). Together they bound where the policy mean CRI
    # *could* possibly land if the OS perfectly disengaged.
    baseline_cri_empirical_mean: float = 0.0
    baseline_cri_empirical_min: float = 0.0
    baseline_cri_theoretical: float = 0.0
    baseline_cri_sample_count: int = 0


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
            - Episodes with zero steps are excluded from reward normalization.
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
        per_ep_mT, per_ep_MT, per_ep_mTC = [], [], []
        per_ep_vRA, per_ep_sCF = [], []
        per_ep_mHE, per_ep_mu, per_ep_Mu = [], [], []

        for ep in episodes:
            if ep.cross_errors.size:
                abs_cte = np.abs(ep.cross_errors)
                per_ep_mTEI.append(float(np.nanmean(abs_cte)))   # correct mean
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
                    # Filter out zeros to avoid double counting zero-crossings
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
            Compute mean of list, returning 0.0 for empty lists.
            
            Uses np.nanmean to ignore NaN values. Returns 0.0 if list is empty
            to avoid warnings and ensure consistent behavior across episodes.
            
            Args:
                lst: List of numeric values (may contain NaN).
            
            Returns:
                float: Mean of non-NaN values, or 0.0 if list is empty.
            """

            return float(np.nanmean(lst)) if lst else 0.0

        def safe_max(lst: list) -> float:
            """
            Compute maximum of list, returning 0.0 for empty lists.
            
            Uses np.nanmax to ignore NaN values. Returns 0.0 if list is empty
            to avoid warnings and ensure consistent behavior across episodes.
            
            Args:
                lst: List of numeric values (may contain NaN).
            
            Returns:
                float: Maximum of non-NaN values, or 0.0 if list is empty.
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

        # Aggregate env-reported collision / COLREG info.
        m.num_episodes = len(episodes)
        m.collision_type_count = dict(
            Counter(ep.collision_type for ep in episodes if ep.collision and ep.collision_type)
        )
        m.collision_side_count = dict(
            Counter(ep.collision_side for ep in episodes if ep.collision and ep.collision_side)
        )
        m.colreg_violation_count = sum(int(ep.colreg_violations) for ep in episodes)

        # Classification diagnostics - one record per encounter.
        # num_encounters counts all encounter starts (including those without
        # a gt label); classification accuracy metrics only use labeled ones.
        # collision_count is the total: collided encounters + episode-terminating
        # collisions that happened outside any encounter (e.g. bank groundings
        # while path-following). Per-policy counts use only encounter-level
        # collisions since outside-encounter collisions have no gt scenario.
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
        # Pool of per-encounter min-CRI samples across *all* policies - used
        # to compute the empirical baseline CRI floor reported globally.
        min_cri_samples_all_policies: list[float] = []
        for ep in episodes:
            encounter_collision_in_ep = False
            for enc in ep.scenario_encounters:
                total += 1
                gt = enc.get("gt")
                collided = bool(enc.get("collided", False))
                passed = bool(enc.get("passed", False))
                tracked_lost = bool(enc.get("tracked_lost", False))
                # An encounter has a fully determined outcome when it either
                # ended in collision or fully resolved (OS passed and TS left
                # forget range). Truncated / mid-encounter records are skipped
                # since they weren't exposed to the full set of dangers.
                valid_outcome = collided or (passed and tracked_lost)
                if gt is not None:
                    total_encounters_per_policy[gt] = total_encounters_per_policy.get(gt, 0) + 1
                    if valid_outcome:
                        valid_encounters_per_policy[gt] = valid_encounters_per_policy.get(gt, 0) + 1
                    # Outcome category for the diagnostic breakdown.
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
                # CRI samples are restricted to *valid* encounters: those
                # that resolved (OS passed and TS left forget range) and
                # did not end in collision. Unresolved encounters give a
                # truncated view of CRI and would skew the aggregate.
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
                    # Per-encounter min CRI contributes to the global baseline
                    # (pooled across all policies - that's the user-requested
                    # "lowest CRI value typically attainable" reference).
                    enc_min_cri = float(enc.get("min_cri", 0.0))
                    if math.isfinite(enc_min_cri):
                        min_cri_samples_all_policies.append(enc_min_cri)
                    # Same valid-encounter gate for per-policy CTE samples.
                    enc_mean_cte = float(enc.get("mean_cte", 0.0))
                    enc_max_cte = float(enc.get("max_cte", 0.0))
                    mean_cte_samples_per_policy.setdefault(gt, []).append(enc_mean_cte)
                    if enc_max_cte > max_cte_per_policy.get(gt, 0.0):
                        max_cte_per_policy[gt] = enc_max_cte
                    # Time-to-first-action: None means the OS never commanded
                    # a rudder above threshold during the encounter - bucket
                    # those separately so they don't skew the mean/median.
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
            # Episode-terminating collisions that didn't occur in any encounter
            # (e.g. bank groundings while path-following) are counted separately
            # so total collisions aren't underreported.
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
        # Empirical baseline CRI - pooled across all policies.
        if min_cri_samples_all_policies:
            m.baseline_cri_empirical_mean = float(np.mean(min_cri_samples_all_policies))
            m.baseline_cri_empirical_min = float(np.min(min_cri_samples_all_policies))
            m.baseline_cri_sample_count = len(min_cri_samples_all_policies)
        # Theoretical baseline is left at 0.0 here - it depends on env-side
        # CRI/encounter-range constants the metric layer doesn't see. The
        # caller (agent_manager) fills it in before printing.

        return m
