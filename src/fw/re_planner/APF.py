import math
from typing import Tuple

import numpy as np


class ArtificialPotentialField:
    """
    Attractive-only Artificial Potential Field helper used by the hybrid
    RRT-APF replanner.

    Design note
    -----------
    This implementation deliberately has NO repulsive term. Obstacle
    avoidance is already handled upstream by RRTStarReplanner:
      - sampling is restricted to the annulus [search_inner, search_outer]
        around the obstacle (see RRTStarReplanner._sample), and
      - every candidate edge is rejected if it crosses search_inner
        (see RRTStarReplanner._edge_free).
    Adding a repulsive force here would double-count obstacle avoidance and
    was the source of the previous "bad path" behaviour (the two repulsive
    fields fought each other / pushed nodes out of the valid annulus).

    So this class only ever does one thing: produce a force that pulls a
    candidate node toward the goal, and a (alpha, beta) blend weight that
    decides how much of "go toward goal" vs. "go toward random RRT sample"
    should be used at the current node, based on how close that node already
    is to the goal.
    """

    def __init__(
        self,
        attractive_gain: float = 0.1,
        attractive_shape: float = 0.002,
        beta_min: float = 0.05,
        beta_max: float = 0.8,
    ) -> None:
        """
        Parameters
        ----------
        attractive_gain : float
            Saturation magnitude of the attractive force (Eq. 5 style,
            exponential approach to `attractive_gain` as distance grows).
        attractive_shape : float
            Controls how quickly the attractive force saturates with
            distance. Larger -> saturates faster (closer to goal).
        beta_min : float
            APF weight when the node is at/beyond search_outer distance
            from the goal (far away -> mostly random exploration).
        beta_max : float
            APF weight when the node is essentially at the goal (close ->
            mostly direct homing toward the goal).
        """
        self.attractive_gain = float(attractive_gain)
        self.attractive_shape = float(attractive_shape)

        if not (0.0 <= beta_min <= 1.0) or not (0.0 <= beta_max <= 1.0):
            raise ValueError("beta_min and beta_max must be in [0, 1].")
        if beta_min > beta_max:
            raise ValueError("beta_min must be <= beta_max.")

        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    @staticmethod
    def _safe_unit(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return np.zeros_like(vec, dtype=float)
        return vec / norm

    def attractive_force(self, q_near: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Exponential attractive force, saturating to `attractive_gain` as
        distance to the goal grows, and vanishing at the goal itself.
        """
        d_att = np.asarray(goal, dtype=float) - np.asarray(q_near, dtype=float)
        dist = float(np.linalg.norm(d_att))
        if dist < 1e-12:
            return np.zeros_like(d_att, dtype=float)

        magnitude = self.attractive_gain * (1.0 - math.exp(-dist * self.attractive_shape))
        return magnitude * (d_att / dist)

    def total_force(self, q_near: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Kept as a thin wrapper (rather than removed outright) so external
        callers that already depend on `total_force(...)` as the "APF
        force at this node" entry point keep working unchanged. With the
        repulsive term removed, total force == attractive force.
        """
        return self.attractive_force(q_near, goal)

    def compute_rrt_apf_weights(
        self,
        q_near: np.ndarray,
        goal: np.ndarray,
        search_outer: float,
    ) -> Tuple[float, float]:
        """
        Compute (alpha, beta) = (RRT/random weight, APF/attractive weight)
        from how close `q_near` already is to the goal, normalized by
        `search_outer` (the outer radius of the RRT sampling annulus, which
        is already a problem-scaled distance the planner knows about).

        beta is LOW far from the goal (favor random exploration so the
        tree can still find its way around/through the annulus) and HIGH
        near the goal (favor direct homing so the tree converges cleanly
        instead of continuing to wander once it's basically arrived).
        alpha = 1 - beta.

        ratio = clip(dist_to_goal / search_outer, 0, 1)
        beta  = beta_min + (beta_max - beta_min) * (1 - ratio)
        """
        q_near = np.asarray(q_near, dtype=float)
        goal = np.asarray(goal, dtype=float)
        search_outer = max(float(search_outer), 1e-6)

        dist_to_goal = float(np.linalg.norm(goal - q_near))
        ratio = float(np.clip(dist_to_goal / search_outer, 0.0, 1.0))

        beta = self.beta_min + (self.beta_max - self.beta_min) * (1.0 - ratio)
        alpha = 1.0 - beta
        return alpha, beta
