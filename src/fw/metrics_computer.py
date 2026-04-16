from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from fw.ship_episode_data import RunData, ShipEpisodeData


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ShipMetrics:
    """Per-ship, per-run aggregate metrics."""
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


class MetricsComputer:
    """
    Computes per-ship aggregate metrics from ShipData.

    Usage
    -----
        metrics: dict[str, ShipMetrics] = MetricsComputer.compute(ship_data)
    """

    @staticmethod
    def compute(run_data: RunData) -> dict[str, ShipMetrics]:
        result = {}
        for ship_id, episodes in run_data.ships.items():
            result[ship_id] = MetricsComputer._compute_ship(ship_id, episodes)
        return result

    @staticmethod
    def _compute_ship(ship_id: str, episodes: list[ShipEpisodeData]) -> ShipMetrics:
        m = ShipMetrics(ship_id=ship_id)
        if not episodes:
            return m

        rewards = [ep.reward for ep in episodes]
        m.avg_reward = float(np.mean(rewards))
        m.best_reward = float(np.max(rewards))
        m.worst_reward = float(np.min(rewards))

        per_ep_mTEI, per_ep_MTE = [], []
        per_ep_mRA, per_ep_MRA, per_ep_mRAC = [], [], []
        per_ep_mHE, per_ep_mu, per_ep_Mu = [], [], []

        for ep in episodes:
            if ep.cross_errors.size:
                # Proper integral (normalized by length)
                integral = np.trapezoid(np.abs(ep.cross_errors))
                normalized = integral / len(ep.cross_errors)
                per_ep_mTEI.append(normalized)
                per_ep_MTE.append(np.max(np.abs(ep.cross_errors)))
            if ep.rudder_actions.size:
                per_ep_mRA.append(np.mean(np.abs(ep.rudder_actions)))
                per_ep_MRA.append(np.max(np.abs(ep.rudder_actions)))
            if ep.rudder_changes.size:
                per_ep_mRAC.append(np.mean(ep.rudder_changes))
            if ep.heading_errors.size:
                per_ep_mHE.append(np.mean(np.abs(ep.heading_errors)))
            if ep.velocities.size:
                per_ep_mu.append(np.mean(np.abs(ep.velocities)))
                per_ep_Mu.append(np.max(np.abs(ep.velocities)))

        def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
        def safe_max(lst):  return float(np.max(lst))  if lst else 0.0

        m.mTEI = safe_mean(per_ep_mTEI)
        m.MTE  = safe_max(per_ep_MTE)
        m.mRA  = safe_mean(per_ep_mRA)
        m.MRA  = safe_max(per_ep_MRA)
        m.mRAC = safe_mean(per_ep_mRAC)
        m.mHE  = safe_mean(per_ep_mHE)
        m.mu   = safe_mean(per_ep_mu)
        m.Mu   = safe_max(per_ep_Mu)

        return m
