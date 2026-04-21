import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


class ArtificialPotentialField:
    """
    Artificial Potential Field helper used by the hybrid RRT-APF replanner.

    The implementation follows the structure of the paper the user provided:
      - total force = attractive(goal) + repulsive(static) + repulsive(dynamic)
      - weights between RRT and APF are adapted from the local obstacle density

    The class is intentionally lightweight so it can be plugged into the
    existing RRT code with minimal changes.
    """

    def __init__(
        self,
        attractive_gain: float = 1.0,
        attractive_shape: float = 0.02,
        repulsive_gain: float = 1.0,
        repulsive_shape: float = 0.08,
        density_radius: float = 60.0,
        max_obstacle_ratio: float = 0.95,
    ) -> None:
        self.attractive_gain = float(attractive_gain)
        self.attractive_shape = float(attractive_shape)
        self.repulsive_gain = float(repulsive_gain)
        self.repulsive_shape = float(repulsive_shape)
        self.density_radius = float(density_radius)
        self.max_obstacle_ratio = float(max_obstacle_ratio)

    @staticmethod
    def _safe_unit(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return np.zeros_like(vec, dtype=float)
        return vec / norm

    def attractive_force(self, q_near: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Exponential attractive force similar to Eq. (5) in the paper.
        """
        d_att = np.asarray(goal, dtype=float) - np.asarray(q_near, dtype=float)
        dist = float(np.linalg.norm(d_att))
        if dist < 1e-12:
            return np.zeros_like(d_att, dtype=float)

        magnitude = self.attractive_gain * (1.0 - math.exp(-dist * self.attractive_shape))
        return magnitude * (d_att / dist)

    def repulsive_force(
        self,
        q_near: np.ndarray,
        obstacle_center: np.ndarray,
        safe_distance: float,
    ) -> np.ndarray:
        """
        Smooth repulsive force similar to Eq. (6) in the paper.
        The direction points away from the obstacle.
        """
        q_near = np.asarray(q_near, dtype=float)
        obstacle_center = np.asarray(obstacle_center, dtype=float)

        d_rep = q_near - obstacle_center
        dist = float(np.linalg.norm(d_rep))
        if dist < 1e-12:
            # Strong deterministic push if we land exactly on the obstacle center.
            out = np.zeros_like(d_rep, dtype=float)
            out[0] = self.repulsive_gain
            return out

        # Logistic-like shaping around safe_distance. Near the obstacle the force is high,
        # farther away it decays smoothly.
        magnitude = self.repulsive_gain / (1.0 + math.exp((dist - safe_distance) * self.repulsive_shape))
        return magnitude * (d_rep / dist)

    def total_force(
        self,
        q_near: np.ndarray,
        goal: np.ndarray,
        static_obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
        dynamic_obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
        safe_distance: float = 0.0,
    ) -> np.ndarray:
        total = self.attractive_force(q_near, goal)

        for obstacles in (static_obstacles or [], dynamic_obstacles or []):
            center, radius = obstacles
            total = total + self.repulsive_force(
                q_near=q_near,
                obstacle_center=np.asarray(center, dtype=float),
                safe_distance=float(safe_distance) + float(radius),
            )

        return total

    def compute_rrt_apf_weights(
        self,
        q_near: np.ndarray,
        search_radius: float,
        static_obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
        dynamic_obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
    ) -> Tuple[float, float]:
        """
        Compute (alpha, beta) = (RRT weight, APF weight).

        The paper estimates alpha from the occupancy of a local search space around
        the current node. We do not have obstacle point clouds in the current ship
        code, so we use a minimal geometric proxy: the local obstacle occupancy is
        approximated from the overlap between the search sphere centered at q_near
        and the obstacle discs. More nearby / larger obstacles => larger alpha.
        """
        q_near = np.asarray(q_near, dtype=float)
        search_radius = max(float(search_radius), 1e-6)

        occupancy = 0.0
        obstacles: List[Tuple[np.ndarray, float]] = []
        obstacles.extend(static_obstacles or [])
        obstacles.extend(dynamic_obstacles or [])

        for center, radius in obstacles:
            center = np.asarray(center, dtype=float)
            radius = float(radius)
            dist = float(np.linalg.norm(q_near - center))
            influence = max(0.0, 1.0 - dist / (search_radius + radius + 1e-6))
            size_factor = min(1.0, radius / search_radius)
            occupancy += influence * (0.5 + 0.5 * size_factor)

        alpha = float(np.clip(occupancy, 0.0, self.max_obstacle_ratio))
        beta = float(1.0 - alpha)
        return alpha, beta
