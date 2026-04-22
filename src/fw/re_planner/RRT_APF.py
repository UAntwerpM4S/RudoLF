"""
Hybrid RRT-APF local replanner.

This file is designed to be a drop-in alternative to the existing
RRTStarReplanner. The public API and the returned result type stay the same,
so the environment can switch planner with a minimal import change.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import KDTree

from fw.re_planner.APF import ArtificialPotentialField
from fw.re_planner.RRT import Node, RRTStarReplanner


class RRTAPFReplanner(RRTStarReplanner):
    """
    Hybrid RRT-APF replanner.

    It keeps the same public `replan(...)` interface and the same output format
    as `RRTStarReplanner`, but changes the tree expansion rule from pure RRT*
    steering to the weighted combination described in the paper:

        q_new = q_near + step * ( alpha * u_rand + beta * u_apf )

    where:
      - alpha is the RRT weight
      - beta is the APF weight
      - u_rand is the unit direction toward the random sample
      - u_apf is the unit APF direction
    """

    def __init__(
        self,
        danger_radius: float = 40.0,
        inner_margin: float = 10.0,
        outer_margin: float = 300.0,
        max_iter: int = 1_000,
        refinement_iter: int = 5_000,
        step_size: float = 30.0,
        goal_bias: float = 0.10,
        goal_tolerance: float = 2.0,
        seed: Optional[int] = None,
        apf: Optional[ArtificialPotentialField] = None,
        static_obstacles: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
    ) -> None:
        super().__init__(
            danger_radius=danger_radius,
            inner_margin=inner_margin,
            outer_margin=outer_margin,
            max_iter=max_iter,
            refinement_iter=refinement_iter,
            step_size=step_size,
            goal_bias=goal_bias,
            goal_tolerance=goal_tolerance,
            seed=seed,
        )
        self.apf = apf if apf is not None else ArtificialPotentialField(
            attractive_gain=1.0,
            attractive_shape=0.02,
            repulsive_gain=1.0,
            repulsive_shape=0.08,
            density_radius=max(self.step_size * 2.0, 1.0),
        )
        self.static_obstacles = list(static_obstacles or [])

    @staticmethod
    def _safe_unit(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return np.zeros_like(vec, dtype=float)
        return vec / norm

    def _hybrid_steer(
        self,
        q_near: np.ndarray,
        q_rand: np.ndarray,
        goal: np.ndarray,
        dynamic_obstacles: Sequence[Tuple[np.ndarray, float]],
        search_inner: float,
    ) -> np.ndarray:
        rand_dir = self._safe_unit(np.asarray(q_rand, dtype=float) - np.asarray(q_near, dtype=float))
        f_total = self.apf.total_force(
            q_near=q_near,
            goal=goal,
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            safe_distance=search_inner,
        )
        apf_dir = self._safe_unit(f_total)

        alpha, beta = self.apf.compute_rrt_apf_weights(
            q_near=q_near,
            search_radius=max(self.apf.density_radius, self.step_size),
            static_obstacles=self.static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
        )

        direction = alpha * rand_dir + beta * apf_dir
        direction = self._safe_unit(direction)

        if float(np.linalg.norm(direction)) < 1e-12:
            direction = rand_dir if float(np.linalg.norm(rand_dir)) > 1e-12 else apf_dir

        if float(np.linalg.norm(direction)) < 1e-12:
            return np.asarray(q_near, dtype=float).copy()

        return np.asarray(q_near, dtype=float) + direction * self.step_size

    def _run_rrt_star(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_center: np.ndarray,
        obstacle_radius: float,
        search_inner: float,
        search_outer: float,
        dim: int,
    ) -> Tuple[Optional[List[np.ndarray]], int]:
        """
        Same structure as the original `_run_rrt_star`, but node expansion uses
        the hybrid RRT + APF direction.
        """
        rewire_r = self.step_size * 3.5

        nodes: List[Node] = [Node(pos=start.copy(), cost=0.0)]
        goal_node: Optional[Node] = None
        refinement_budget = 0

        kd: Optional[KDTree] = None
        kd_dirty = True

        dynamic_obstacles = [(np.asarray(obstacle_center, dtype=float), float(obstacle_radius))]

        def rebuild_kd() -> None:
            nonlocal kd, kd_dirty
            positions = np.array([n.pos for n in nodes])
            kd = KDTree(positions)
            kd_dirty = False

        iters = 0
        for iters in range(1, self.max_iter + 1):
            if goal_node is not None:
                if refinement_budget <= 0:
                    break
                refinement_budget -= 1

            q_rand = self._sample(goal, obstacle_center, dim, search_inner, search_outer)

            if kd_dirty:
                rebuild_kd()
            _, nn_idx = kd.query(q_rand)
            q_near = nodes[nn_idx]

            q_new_pos = self._hybrid_steer(
                q_near=q_near.pos,
                q_rand=q_rand,
                goal=goal,
                dynamic_obstacles=dynamic_obstacles,
                search_inner=search_inner,
            )

            if np.allclose(q_new_pos, q_near.pos):
                continue

            # Keep the new node in the annulus used by the original planner.
            radial_dist = float(np.linalg.norm(q_new_pos - obstacle_center))
            if radial_dist < search_inner or radial_dist > search_outer:
                continue

            if not self._edge_free(q_near.pos, q_new_pos, obstacle_center, search_inner):
                continue

            if kd_dirty:
                rebuild_kd()
            neighbour_idxs = kd.query_ball_point(q_new_pos, rewire_r)
            neighbours = [nodes[i] for i in neighbour_idxs]

            best_parent = q_near
            best_cost = q_near.cost + float(np.linalg.norm(q_new_pos - q_near.pos))
            for nb in neighbours:
                c = nb.cost + float(np.linalg.norm(q_new_pos - nb.pos))
                if c < best_cost and self._edge_free(nb.pos, q_new_pos, obstacle_center, search_inner):
                    best_parent, best_cost = nb, c

            q_new = Node(pos=q_new_pos, parent=best_parent, cost=best_cost)
            best_parent.children.append(q_new)
            nodes.append(q_new)
            kd_dirty = True

            for nb in neighbours:
                if nb is best_parent:
                    continue
                new_cost = best_cost + float(np.linalg.norm(nb.pos - q_new_pos))
                if new_cost < nb.cost and self._edge_free(q_new_pos, nb.pos, obstacle_center, search_inner):
                    if nb.parent is not None:
                        try:
                            nb.parent.children.remove(nb)
                        except ValueError:
                            pass
                    nb.parent = q_new
                    q_new.children.append(nb)
                    self._propagate_cost(nb, new_cost)

            dist_to_goal = float(np.linalg.norm(q_new_pos - goal))
            if dist_to_goal <= self.goal_tolerance:
                goal_cost = best_cost + dist_to_goal
                if goal_node is None or goal_cost < goal_node.cost:
                    gn = Node(pos=goal.copy(), parent=q_new, cost=goal_cost)
                    q_new.children.append(gn)
                    nodes.append(gn)
                    kd_dirty = True
                    goal_node = gn
                    refinement_budget = self.refinement_iter

        if goal_node is None:
            return None, iters

        return self._extract_path(goal_node), iters
