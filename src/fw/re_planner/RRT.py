"""
rrt_star_replanner.py
=====================
RRT* local path replanner triggered by a collision detection event.

Terminology
-----------
obstacle_center : position of the detected obstacle / CPA point (2-D or 3-D).
danger_radius   : clearance from the obstacle border (e.g. 40 m).
search_inner    : inner boundary of the RRT* sampling annulus (minimum safe distance)
                  = obstacle_radius + danger_radius + inner_margin
search_outer    : outer boundary of the sampling annulus
                  = search_inner + outer_margin

The algorithm
-------------
1. Given the full ship trajectory and the obstacle center, find:
   - entry_wp : point in which the search_outer circle intersects the trajectory for the first time (closest to the "approaching" side).
   - exit_wp  : point in which the search_outer circle intersects the trajectory for the last time (closest to the "leaving" side).

2. Run RRT* in 2-D (or 3-D) with:
   - start = entry_wp
   - goal  = exit_wp
   - obstacles = a single disc (sphere) of radius
                 obstacle_radius + danger_radius centered on obstacle_center.
   - sampling domain = annulus [search_inner, search_outer] around obstacle_center.
     All RRT nodes and paths stay >= search_inner distance from obstacle center.
- Early termination: as soon as the goal is reached the algorithm keeps
  improving the path for `refinement_iter` more iterations, then stops.
"""

import math
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Node:
    """A node in the RRT* tree."""

    __slots__ = ("pos", "parent", "cost", "children")

    def __init__(self, pos: np.ndarray, parent: "Optional[Node]" = None, cost: float = 0.0):
        self.pos: np.ndarray = pos
        self.parent: Optional[Node] = parent
        self.cost: float = cost
        self.children: List[Node] = []

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class RRTStarResult:
    """Output of a single RRT* planning call."""

    def __init__(
        self,
        success: bool,
        rrt_path: "List[np.ndarray]",
        spliced_trajectory: "List[np.ndarray]",
        entry_idx: int,
        exit_idx: int,
        entry_wp: np.ndarray,
        exit_wp: np.ndarray,
        iterations_used: int,
        planning_time_s: float,
        cost: float,
    ):
        self.success = success
        self.rrt_path = rrt_path                        # waypoints entry_wp → exit_wp
        self.spliced_trajectory = spliced_trajectory    # full patched trajectory
        self.entry_idx = entry_idx                      # index in original trajectory
        self.exit_idx = exit_idx                        # index in original trajectory
        self.entry_wp = entry_wp
        self.exit_wp = exit_wp
        self.iterations_used = iterations_used
        self.planning_time_s = planning_time_s
        self.cost = cost                                # arc-length of rrt_path

    def __repr__(self) -> str:
        return (
            f"RRTStarResult(success={self.success}, cost={self.cost:.2f}, "
            f"entry_idx={self.entry_idx}, exit_idx={self.exit_idx}, "
            f"waypoints={len(self.rrt_path)}, "
            f"time={self.planning_time_s*1000:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RRTStarReplanner:
    """
    RRT* local replanner for collision avoidance.

    Parameters
    ----------
    danger_radius : float
        Clearance from the obstacle border [same units as coordinates].
    inner_margin : float
        Gap between the effective keep-out radius and the inner edge of the
        sampling annulus.
        Default 10.
    outer_margin : float
        Width of the sampling annulus beyond search_inner.
        Default 40.
    max_iter : int
        Maximum RRT* iterations.
    refinement_iter : int
        Extra iterations after first goal connection (path refinement).
    step_size : float
        Maximum extension length per RRT* step.
    goal_bias : float
        Probability [0, 1] of sampling the goal directly instead of randomly.
    goal_tolerance : float
        Distance within which a node is considered to have reached the goal.
    seed : int | None
        Random seed for reproducibility.
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
    ):
        if danger_radius <= 0:
            raise ValueError("danger_radius must be positive.")
        if inner_margin < 0 or outer_margin <= 0:
            raise ValueError("Margins must be non-negative (inner) / positive (outer).")

        self.danger_radius = float(danger_radius)
        self.inner_margin = float(inner_margin)
        self.outer_margin = float(outer_margin)
        self.search_inner = self.danger_radius + self.inner_margin
        self.search_outer = self.search_inner + self.outer_margin
        self.max_iter = int(max_iter)
        self.refinement_iter = int(refinement_iter)
        self.step_size = float(step_size)
        self.goal_bias = float(goal_bias)
        self.goal_tolerance = float(goal_tolerance)
        self.rng = np.random.default_rng(seed)

        # Rewiring neighbourhood radius (RRT* asymptotically optimal formula).
        # Will be updated at runtime once we know the dimension.
        self._rewire_radius: Optional[float] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def replan(
        self,
        obstacle_center: np.ndarray,
        ship_trajectory: np.ndarray,
        obstacle_radius: float = 0.0,
    ) -> RRTStarResult:
        """
        Compute a collision-free local path around obstacle_center.

        Parameters
        ----------
        obstacle_center : array-like, shape (D,)
            Center of the obstacle / CPA point.
        ship_trajectory : array-like, shape (N, D)
            Original ship waypoints.  Must have at least 2 points.
        obstacle_radius : float, optional
            Radius of the obstacle itself. The planner keeps a clearance of
            ``danger_radius`` from the obstacle border.

        Returns
        -------
        RRTStarResult
            .success            – True if a path was found.
            .rrt_path           – list of np.ndarray waypoints (entry → exit).
            .spliced_trajectory – full trajectory with the replanned segment.
            .entry_idx / .exit_idx – indices into original trajectory.
            .entry_wp / .exit_wp   – boundary waypoints used as start/goal.
            .iterations_used    – actual RRT* iterations run.
            .planning_time_s    – wall-clock seconds.
            .cost               – arc-length of rrt_path.
        """
        t0 = time.perf_counter()

        obstacle_center = np.asarray(obstacle_center, dtype=float)
        ship_trajectory = np.asarray(ship_trajectory, dtype=float)
        obstacle_radius = float(obstacle_radius)
        dim = obstacle_center.shape[0]

        if obstacle_radius < 0.0:
            raise ValueError("obstacle_radius must be non-negative.")

        effective_danger_radius = obstacle_radius + self.danger_radius
        search_inner = effective_danger_radius + self.inner_margin
        search_outer = search_inner + self.outer_margin

        if ship_trajectory.ndim != 2 or ship_trajectory.shape[1] != dim:
            raise ValueError(
                f"ship_trajectory must have shape (N, {dim}), "
                f"got {ship_trajectory.shape}."
            )
        if len(ship_trajectory) < 2:
            raise ValueError("ship_trajectory must have at least 2 waypoints.")

        # 1. Find entry and exit waypoints
        entry_idx, exit_idx, entry_wp, exit_wp = self._find_boundary_waypoints(
            obstacle_center,
            ship_trajectory,
            search_outer,
        )

        # 2. Run RRT*
        rrt_path, iters = self._run_rrt_star(
            start=entry_wp,
            goal=exit_wp,
            obstacle_center=obstacle_center,
            search_inner=search_inner,
            search_outer=search_outer,
            dim=dim,
        )

        dt = time.perf_counter() - t0
        success = rrt_path is not None and len(rrt_path) >= 2

        if success:
            cost = float(sum(
                np.linalg.norm(rrt_path[i + 1] - rrt_path[i])
                for i in range(len(rrt_path) - 1)
            ))
            spliced = self._splice(ship_trajectory, rrt_path, entry_idx, exit_idx)
        else:
            cost = float("inf")
            spliced = ship_trajectory.tolist()

        return RRTStarResult(
            success=success,
            rrt_path=rrt_path or [],
            spliced_trajectory=spliced,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            entry_wp=entry_wp,
            exit_wp=exit_wp,
            iterations_used=iters,
            planning_time_s=dt,
            cost=cost,
        )

    # ------------------------------------------------------------------
    # Boundary waypoint detection
    # ------------------------------------------------------------------

    def _find_boundary_waypoints(
        self,
        obstacle_center: np.ndarray,
        trajectory: np.ndarray,
        search_outer: float,
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
        """
        Identify entry/exit points where trajectory intersects the search_outer sphere.

        Definitions
        -----------
        entry_wp  : first intersection point of the trajectory polyline with the
                    search_outer sphere when traversing from start to end.
        exit_wp   : last intersection point of the trajectory polyline with the
                    search_outer sphere when traversing from start to end.

        entry_idx : index of waypoint before entry_wp (used for splicing prefix).
        exit_idx  : index of waypoint after exit_wp (used for splicing suffix).

        Robustness
        ----------
        If fewer than two intersections are found, we fall back to the closest
        waypoint split as before.
        """
        N = len(trajectory)
        dists = np.linalg.norm(trajectory - obstacle_center, axis=1)  # (N,)

        intersections: List[Tuple[float, int, float, np.ndarray]] = []
        for i in range(N - 1):
            a = trajectory[i]
            b = trajectory[i + 1]
            ts = self._segment_sphere_intersections(a, b, obstacle_center, search_outer)
            for t in ts:
                pt = a + t * (b - a)
                intersections.append((i + t, i, t, pt))

        # Keep unique intersections along path order.
        intersections.sort(key=lambda x: x[0])
        uniq: List[Tuple[float, int, float, np.ndarray]] = []
        for item in intersections:
            if not uniq or abs(item[0] - uniq[-1][0]) > 1e-7:
                uniq.append(item)

        if len(uniq) >= 2:
            _, entry_seg_idx, _, entry_wp = uniq[0]
            _, exit_seg_idx, _, exit_wp = uniq[-1]

            entry_idx = int(np.clip(entry_seg_idx, 0, N - 2))
            exit_idx = int(np.clip(exit_seg_idx + 1, 1, N - 1))

            if entry_idx >= exit_idx:
                exit_idx = min(N - 1, entry_idx + 1)

            return entry_idx, exit_idx, entry_wp.copy(), exit_wp.copy()

        # Fallback: obstacle far from path or degenerate intersections.
        mid = int(np.argmin(dists))
        entry_idx = max(0, mid - 1)
        exit_idx = min(N - 1, mid + 1)
        if entry_idx >= exit_idx:
            exit_idx = min(N - 1, entry_idx + 1)

        return (
            entry_idx,
            exit_idx,
            trajectory[entry_idx].copy(),
            trajectory[exit_idx].copy(),
        )

    @staticmethod
    def _segment_sphere_intersections(
        a: np.ndarray,
        b: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> List[float]:
        """Return sorted t values in [0,1] where segment a->b intersects sphere."""
        d = b - a
        f = a - center

        A = float(d @ d)
        if A < 1e-12:
            return []

        B = float(2.0 * (f @ d))
        C = float((f @ f) - radius * radius)

        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return []

        if disc < 1e-12:
            t = -B / (2.0 * A)
            return [float(np.clip(t, 0.0, 1.0))] if 0.0 <= t <= 1.0 else []

        sqrt_disc = math.sqrt(disc)
        t1 = (-B - sqrt_disc) / (2.0 * A)
        t2 = (-B + sqrt_disc) / (2.0 * A)
        out = []
        if 0.0 <= t1 <= 1.0:
            out.append(float(t1))
        if 0.0 <= t2 <= 1.0:
            out.append(float(t2))
        out.sort()
        return out

    # ------------------------------------------------------------------
    # RRT* core
    # ------------------------------------------------------------------

    def _run_rrt_star(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_center: np.ndarray,
        search_inner: float,
        search_outer: float,
        dim: int,
    ) -> Tuple[Optional[List[np.ndarray]], int]:
        """
        Run RRT* and return (path, iterations_used).

        Returns None as path if no solution was found within max_iter.
        """
        # Rewire radius: RRT* formula γ*(log n / n)^(1/d), simplified to a
        # fixed multiple of step_size that works well in practice.
        rewire_r = self.step_size * 3.5

        # Tree
        nodes: List[Node] = [Node(pos=start.copy(), cost=0.0)]
        goal_node: Optional[Node] = None
        refinement_budget = 0

        # KD-tree (rebuilt lazily each iteration after insert)
        kd: Optional[KDTree] = None
        kd_dirty = True

        def rebuild_kd():
            nonlocal kd, kd_dirty
            positions = np.array([n.pos for n in nodes])
            kd = KDTree(positions)
            kd_dirty = False

        iters = 0
        for iters in range(1, self.max_iter + 1):

            # ── early exit once we have a goal + refinement budget spent ──
            if goal_node is not None:
                if refinement_budget <= 0:
                    break
                refinement_budget -= 1

            # 1. Sample from search_inner to search_outer annulus
            q_rand = self._sample(goal, obstacle_center, dim, search_inner, search_outer)

            # 2. Nearest node in tree
            if kd_dirty:
                rebuild_kd()
            dist_to_nn, nn_idx = kd.query(q_rand)
            q_near = nodes[nn_idx]

            # 3. Steer toward q_rand
            q_new_pos = self._steer(q_near.pos, q_rand)

            # 4. Collision check on the edge (q_near → q_new)
            if not self._edge_free(q_near.pos, q_new_pos, obstacle_center, search_inner):
                continue

            # 5. Find nodes within rewiring radius
            if kd_dirty:
                rebuild_kd()
            neighbour_idxs = kd.query_ball_point(q_new_pos, rewire_r)
            neighbours = [nodes[i] for i in neighbour_idxs]

            # 6. Choose best parent
            best_parent, best_cost = q_near, q_near.cost + np.linalg.norm(q_new_pos - q_near.pos)
            for nb in neighbours:
                c = nb.cost + np.linalg.norm(q_new_pos - nb.pos)
                if c < best_cost and self._edge_free(nb.pos, q_new_pos, obstacle_center, search_inner):
                    best_parent, best_cost = nb, c

            # 7. Add new node
            q_new = Node(pos=q_new_pos, parent=best_parent, cost=best_cost)
            best_parent.children.append(q_new)
            nodes.append(q_new)
            kd_dirty = True

            # 8. Rewire neighbours through q_new
            for nb in neighbours:
                if nb is best_parent:
                    continue
                new_cost = best_cost + np.linalg.norm(nb.pos - q_new_pos)
                if new_cost < nb.cost and self._edge_free(q_new_pos, nb.pos, obstacle_center, search_inner):
                    # Detach nb from old parent
                    if nb.parent is not None:
                        try:
                            nb.parent.children.remove(nb)
                        except ValueError:
                            pass
                    nb.parent = q_new
                    q_new.children.append(nb)
                    self._propagate_cost(nb, new_cost)

            # 9. Goal check
            dist_to_goal = float(np.linalg.norm(q_new_pos - goal))
            if dist_to_goal <= self.goal_tolerance:
                goal_cost = best_cost + dist_to_goal
                if goal_node is None or goal_cost < goal_node.cost:
                    # Connect to goal
                    gn = Node(pos=goal.copy(), parent=q_new, cost=goal_cost)
                    q_new.children.append(gn)
                    nodes.append(gn)
                    kd_dirty = True
                    goal_node = gn
                    refinement_budget = self.refinement_iter

        if goal_node is None:
            return None, iters

        path = self._extract_path(goal_node)
        return path, iters

    # ------------------------------------------------------------------
    # Helpers: sampling, steering, collision, path extraction
    # ------------------------------------------------------------------

    def _sample(
        self,
        goal: np.ndarray,
        obstacle_center: np.ndarray,
        dim: int,
        search_inner: float,
        search_outer: float,
    ) -> np.ndarray:
        """
        Biased sampler:
          - goal_bias  → sample exactly at goal.
          - remainder  → uniform sample in annulus [search_inner, search_outer]
                         around obstacle_center.
        """
        if self.rng.random() < self.goal_bias:
            return goal.copy()

        # Random direction on unit sphere in R^dim.
        direction = self.rng.normal(size=dim)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.zeros(dim, dtype=float)
            direction[0] = 1.0
        else:
            direction = direction / norm

        # Uniform radius in D-dimensional annulus volume.
        # Sample from search_inner to search_outer to ensure all nodes
        # stay outside the inner margin boundary.
        p = float(self.rng.random())
        inner_d = search_inner ** dim
        outer_d = search_outer ** dim
        radius = (inner_d + p * (outer_d - inner_d)) ** (1.0 / dim)
        return obstacle_center + direction * radius

    def _point_near_segment(
        self,
        pt: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        threshold: float,
    ) -> bool:
        """True if pt is within threshold of the line segment a-b."""
        ab = b - a
        ab_sq = ab @ ab
        if ab_sq < 1e-12:
            return bool(np.linalg.norm(pt - a) <= threshold)
        t = float(np.clip((pt - a) @ ab / ab_sq, 0.0, 1.0))
        closest = a + t * ab
        return bool(np.linalg.norm(pt - closest) <= threshold)

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Move from from_pos toward to_pos by at most step_size."""
        delta = to_pos - from_pos
        dist = float(np.linalg.norm(delta))
        if dist <= self.step_size:
            return to_pos.copy()
        return from_pos + (delta / dist) * self.step_size

    def _edge_free(
        self,
        a: np.ndarray,
        b: np.ndarray,
        obstacle_center: np.ndarray,
        search_inner: float,
    ) -> bool:
        """
        True if the segment a→b stays outside the search_inner boundary.
        
        Ensures the path avoids the region within search_inner distance
        from the obstacle center (i.e., obstacle + clearance + margin).

        Closest distance from a point on the parametric segment
            p(t) = a + t*(b-a),  t ∈ [0,1]
        to obstacle_center is minimised at:
            t* = -((a - obs) · d) / (d · d),  d = b - a
        clamped to [0, 1].
        """
        d = b - a
        dd = d @ d
        if dd < 1e-12:
            return bool(np.linalg.norm(a - obstacle_center) > search_inner)
        t_star = float(np.clip(-((a - obstacle_center) @ d) / dd, 0.0, 1.0))
        closest = a + t_star * d
        return bool(np.linalg.norm(closest - obstacle_center) > search_inner)

    def _propagate_cost(self, node: Node, new_cost: float) -> None:
        """Recursively update costs after a rewire (DFS)."""
        node.cost = new_cost
        for child in node.children:
            self._propagate_cost(
                child,
                new_cost + float(np.linalg.norm(child.pos - node.pos)),
            )

    def _extract_path(self, goal_node: Node) -> List[np.ndarray]:
        """Walk parent pointers from goal back to root; reverse."""
        path = []
        node: Optional[Node] = goal_node
        while node is not None:
            path.append(node.pos.copy())
            node = node.parent
        path.reverse()
        return path

    @staticmethod
    def _splice(
        trajectory: np.ndarray,
        rrt_path: List[np.ndarray],
        entry_idx: int,
        exit_idx: int,
    ) -> List[np.ndarray]:
        """
        Build the full spliced trajectory:
            original[0 .. entry_idx]  +  rrt_path  +  original[exit_idx ..]
        """
        before = [trajectory[i].copy() for i in range(entry_idx + 1)]
        after  = [trajectory[i].copy() for i in range(exit_idx, len(trajectory))]
        return before + rrt_path + after