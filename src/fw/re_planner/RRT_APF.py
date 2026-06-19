"""
Hybrid RRT-APF local replanner.

This file is designed to be a drop-in alternative to the existing
RRTStarReplanner. The public API and the returned result type stay the same,
so the environment can switch planner with a minimal import change.

Design note on obstacle avoidance
----------------------------------
RRTStarReplanner already guarantees obstacle clearance two ways:
  1. `_sample` only draws candidates inside the annulus [search_inner,
     search_outer] around the obstacle.
  2. `_edge_free` rejects any candidate edge whose closest approach to the
     obstacle center is < search_inner.
The APF used here therefore carries NO repulsive term -- it only ever pulls
the new sample toward the goal. All obstacle-keep-out logic remains exactly
as implemented in the base RRTStarReplanner, untouched by this subclass.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from fw.re_planner.APF import ArtificialPotentialField
from fw.re_planner.RRT import Node, RRTStarReplanner


class RRTAPFReplanner(RRTStarReplanner):
    """
    Hybrid RRT-APF replanner.

    It keeps the same public `replan(...)` interface and the same output
    format as `RRTStarReplanner`, but changes the tree expansion rule from
    pure RRT* steering to the weighted combination:

        q_new = q_near + step * unit( alpha * u_rand + beta * u_apf )

    where:
      - alpha is the RRT (random-direction) weight
      - beta is the APF (goal-attraction) weight
      - u_rand is the unit direction toward the random sample
      - u_apf is the unit attractive-force direction toward the goal
      - (alpha, beta) are computed from how close q_near already is to the
        goal (see ArtificialPotentialField.compute_rrt_apf_weights):
        far from goal -> more alpha (explore), close to goal -> more beta
        (home in directly).

    Everything else (sampling annulus, edge collision checks, rewiring,
    goal connection, path extraction) is inherited unchanged from
    RRTStarReplanner.
    """

    def __init__(
        self,
        danger_radius: float = 15.0,
        inner_margin: float = 5.0,
        outer_margin: float = 500.0,
        max_iter: int = 50_000,
        refinement_iter: int = 1_000,
        step_size: float = 30.0,
        goal_bias: float = 0.10,
        goal_tolerance: float = 2.0,
        seed: Optional[int] = None,
        apf: Optional[ArtificialPotentialField] = None,
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
            attractive_gain=0.01,
            attractive_shape=0.002,
        )

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
        search_outer: float,
    ) -> np.ndarray:
        """
        Blend the random-sample direction with the goal-attraction
        direction, weighted by (alpha, beta) from the current distance to
        the goal, then take one `step_size` step in that blended direction.
        No obstacle term is involved here by design (see module docstring).
        """
        rand_dir = self._safe_unit(np.asarray(q_rand, dtype=float) - np.asarray(q_near, dtype=float))

        f_att = self.apf.attractive_force(q_near=q_near, goal=goal)
        apf_dir = self._safe_unit(f_att)

        alpha, beta = self.apf.compute_rrt_apf_weights(
            q_near=q_near,
            goal=goal,
            search_outer=search_outer,
        )

        direction = alpha * rand_dir + beta * apf_dir
        direction = self._safe_unit(direction)

        if float(np.linalg.norm(direction)) < 1e-12:
            # Degenerate blend (e.g. rand_dir and apf_dir cancel out exactly).
            # Fall back to whichever single direction is non-zero.
            direction = rand_dir if float(np.linalg.norm(rand_dir)) > 1e-12 else apf_dir

        if float(np.linalg.norm(direction)) < 1e-12:
            # q_near == q_rand == goal: nothing to do.
            return np.asarray(q_near, dtype=float).copy()

        # Clamp the step length exactly like the base class's `_steer`:
        # never overshoot the nearer of (q_rand, goal). Without this clamp
        # the hybrid always strides a full `step_size`, which can
        # repeatedly overshoot past a tight `goal_tolerance` and prevent
        # the tree from ever landing inside it even though it keeps
        # passing close by.
        q_near_arr = np.asarray(q_near, dtype=float)
        dist_to_rand = float(np.linalg.norm(np.asarray(q_rand, dtype=float) - q_near_arr))
        dist_to_goal = float(np.linalg.norm(np.asarray(goal, dtype=float) - q_near_arr))
        step_len = min(self.step_size, dist_to_rand, dist_to_goal)
        step_len = max(step_len, 1e-9)

        return q_near_arr + direction * step_len

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
        Same structure and same obstacle-avoidance guarantees as the base
        class `_run_rrt_star` (annulus sampling + `_edge_free`), but node
        expansion uses the hybrid RRT + attractive-APF direction instead of
        plain steering toward q_rand.
        """
        rewire_r = self.step_size * 3.5

        nodes: List[Node] = [Node(pos=start.copy(), cost=0.0)]
        goal_node: Optional[Node] = None
        refinement_budget = 0

        kd: Optional[KDTree] = None
        kd_dirty = True

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

            # 1. Sample from search_inner to search_outer annulus (inherited,
            #    unchanged obstacle-aware sampling).
            q_rand = self._sample(goal, obstacle_center, dim, search_inner, search_outer)

            # 2. Nearest node in tree.
            if kd_dirty:
                rebuild_kd()
            _, nn_idx = kd.query(q_rand)
            q_near = nodes[nn_idx]

            # 3. Steer using the hybrid RRT + attractive-APF blend.
            q_new_pos = self._hybrid_steer(
                q_near=q_near.pos,
                q_rand=q_rand,
                goal=goal,
                search_outer=search_outer,
            )

            if np.allclose(q_new_pos, q_near.pos):
                continue

            # Keep the new node inside the same annulus the base planner
            # uses (defense in depth -- normally satisfied automatically
            # since both rand_dir and apf_dir point "inward/along", but the
            # blend could in principle push slightly outside near the
            # boundary).
            radial_dist = float(np.linalg.norm(q_new_pos - obstacle_center))
            if radial_dist < search_inner or radial_dist > search_outer:
                continue

            # 4. Collision check on the edge (q_near -> q_new). This is the
            #    sole obstacle-clearance gate, identical to the base class.
            if not self._edge_free(q_near.pos, q_new_pos, obstacle_center, search_inner):
                continue

            # 5. Find nodes within rewiring radius.
            if kd_dirty:
                rebuild_kd()
            neighbour_idxs = kd.query_ball_point(q_new_pos, rewire_r)
            neighbours = [nodes[i] for i in neighbour_idxs]

            # 6. Choose best parent.
            best_parent = q_near
            best_cost = q_near.cost + float(np.linalg.norm(q_new_pos - q_near.pos))
            for nb in neighbours:
                c = nb.cost + float(np.linalg.norm(q_new_pos - nb.pos))
                if c < best_cost and self._edge_free(nb.pos, q_new_pos, obstacle_center, search_inner):
                    best_parent, best_cost = nb, c

            # 7. Add new node.
            q_new = Node(pos=q_new_pos, parent=best_parent, cost=best_cost)
            best_parent.children.append(q_new)
            nodes.append(q_new)
            kd_dirty = True

            # 8. Rewire neighbours through q_new.
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

            # 9. Goal check.
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
