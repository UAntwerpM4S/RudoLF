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
from scipy.spatial import cKDTree


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
    max_turn_angle_deg : float | None
        Maximum allowed relative angle [deg] between a new edge and the
        immediately preceding edge along the same branch (deviation from
        straight-ahead; 0 = no turn, 180 = unrestricted). Edges that would
        exceed this threshold are discarded (treated like a collision).
        ``None`` disables the check. Default None.
    smooth_rrt_path : bool
        If True, apply shortcut smoothing only to the local RRT* segment
        between entry_wp and exit_wp before splicing it into the original
        trajectory. The original trajectory outside the replanned segment is
        not smoothed or modified. Default True.
    smoothing_iterations : int
        Number of random shortcut attempts used to simplify/smooth the local
        RRT* path. Default 200.
    full_circle : bool
        If True, the whole sampling annulus is available. If False, the
        points in Sector 3 (backward-left relative to the directed vector
        ``entry_wp -> obstacle_center``) are forbidden. Edges are validated
        only at their endpoints. Default True.
    spatial_index_rebuild_interval : int
        Number of newly inserted nodes retained in the exact recent-node
        buffer before rebuilding the spatial index. Larger values reduce
        rebuild cost; the default 128 is a balanced setting. Default 128.
    direct_path_fastpath : bool
        Return immediately when the direct entry-to-exit segment satisfies
        every clearance, sector, and boundary-turn constraint. Default True.
    """

    def __init__(
        self,
        danger_radius: float = 15.0,
        inner_margin: float = 5.0,
        outer_margin: float = 500.0,
        max_iter: int = 50_000,
        refinement_iter: int = 5_000,
        step_size: float = 30.0,
        goal_bias: float = 0.1,
        goal_tolerance: float = 2.0,
        seed: Optional[int] = None,
        max_turn_angle_deg: Optional[float] = 5.0,
        smooth_rrt_path: bool = None,
        smoothing_iterations: int = 3,
        full_circle: bool = True,
        spatial_index_rebuild_interval: int = 128,
        direct_path_fastpath: bool = False,
    ):
        if danger_radius <= 0:
            raise ValueError("danger_radius must be positive.")
        if inner_margin < 0 or outer_margin <= 0:
            raise ValueError("Margins must be non-negative (inner) / positive (outer).")
        if max_turn_angle_deg is not None and not (0.0 < max_turn_angle_deg <= 180.0):
            raise ValueError("max_turn_angle_deg must be in (0, 180].")
        if smoothing_iterations < 0:
            raise ValueError("smoothing_iterations must be non-negative.")
        if spatial_index_rebuild_interval < 1:
            raise ValueError("spatial_index_rebuild_interval must be at least 1.")

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
        self.max_turn_angle_deg = float(max_turn_angle_deg) if max_turn_angle_deg is not None else None
        self.smooth_rrt_path = bool(smooth_rrt_path)
        self.smoothing_iterations = int(smoothing_iterations)
        self.full_circle = bool(full_circle)
        self.spatial_index_rebuild_interval = int(spatial_index_rebuild_interval)
        self.direct_path_fastpath = bool(direct_path_fastpath)

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

        # Checkpoints immediately outside the search area, used for the
        # turn-angle check on the first/last RRT* segments.
        point_before, point_after = self._find_outer_checkpoints(
            obstacle_center,
            ship_trajectory,
            search_outer,
            entry_idx,
            exit_idx,
        )

        # 2. Run RRT*
        rrt_path, iters = self._run_rrt_star(
            start=entry_wp,
            goal=exit_wp,
            obstacle_center=obstacle_center,
            obstacle_radius=obstacle_radius,
            search_inner=search_inner,
            search_outer=search_outer,
            dim=dim,
            point_before=point_before,
            point_after=point_after,
        )

        dt = time.perf_counter() - t0
        success = rrt_path is not None and len(rrt_path) >= 2

        if success:
            # Smooth only the local RRT* segment. The original trajectory
            # before entry_idx and after exit_idx is left unchanged.
            if self.smooth_rrt_path:
                rrt_path = self._smooth_rrt_path(
                    rrt_path=rrt_path,
                    obstacle_center=obstacle_center,
                    search_inner=search_inner,
                    point_before=point_before,
                    point_after=point_after,
                )

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

            # If entry_wp is too close to entry_idx checkpoint,
            # move entry_idx one waypoint backward.
            if np.linalg.norm(entry_wp - trajectory[entry_idx]) < 50:
                entry_idx = max(0, entry_idx - 1)

            # If exit_wp is too close to exit_idx checkpoint,
            # move exit_idx one waypoint forward.
            if np.linalg.norm(exit_wp - trajectory[exit_idx]) < 50:
                exit_idx = min(N - 1, exit_idx + 1)

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

    def _find_outer_checkpoints(
        self,
        obstacle_center: np.ndarray,
        ship_trajectory: np.ndarray,
        search_outer: float,
        entry_idx: int,
        exit_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the trajectory waypoints immediately outside the
        search_outer boundary: the one right before the entry point and the
        one right after the exit point.

        Given the trajectory, the obstacle position/dimensions (folded into
        ``search_outer``) and the boundary indices found by
        ``_find_boundary_waypoints``, these are simply the original
        waypoints ``ship_trajectory[entry_idx]`` (last waypoint outside the
        annulus before entering it) and ``ship_trajectory[exit_idx]`` (first
        waypoint outside the annulus after leaving it), since by
        construction neither lies within ``search_outer`` of
        ``obstacle_center``.
        """
        point_before = ship_trajectory[entry_idx].copy()
        point_after = ship_trajectory[exit_idx].copy()
        return point_before, point_after

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
        obstacle_radius: float,
        search_inner: float,
        search_outer: float,
        dim: int,
        point_before: np.ndarray,
        point_after: np.ndarray,
    ) -> Tuple[Optional[List[np.ndarray]], int]:
        """
        Run RRT* and return (path, iterations_used).

        Returns None as path if no solution was found within max_iter.

        point_before / point_after are the waypoints immediately outside the
        search area (before `start` and after `goal` respectively) and are
        used solely for the turn-angle check on the first/last segments.

        Turn-angle policy
        -----------------
        The normal RRT* turn-angle check is applied only when the middle point
        of the tested turn lies in the outer 10% of the search area radius.
        Example: if search_outer = 500 m, only nodes with distance from
        obstacle_center >= 450 m are protected by the turn-angle limit.

        The two fixed boundary turns are always checked:
            point_before -> start -> first RRT node
            last RRT node -> goal -> point_after

        If ``full_circle`` is False, every newly sampled or generated point
        must remain outside Sector 3 (backward-left relative to the directed
        vector ``start -> obstacle_center``). Only points are checked; the
        interior of an edge is not checked against the sector.
        """
        if not self.full_circle and dim != 2:
            raise ValueError("full_circle=False is supported only for 2-D paths.")

        # Rewire radius: RRT* formula gamma*(log n / n)^(1/d), simplified to a
        # fixed multiple of step_size that works well in practice.
        rewire_r = self.step_size * 3.5

        # Only the farthest 10% of the search radius is angle-protected.
        # Example: search_outer = 500 -> check band starts at 450.
        turn_check_band_start = 0.9 * search_outer

        # Tree.  Rebuilding a KD-tree after every insertion dominated the
        # original runtime (quadratic work for a long planning attempt).
        # Keep a frozen cKDTree plus a small, exact linear buffer of recently
        # inserted nodes instead.  Queries use both sets, so this is exact --
        # the rebuild interval changes speed only, never feasibility.
        nodes: List[Node] = [Node(pos=start.copy(), cost=0.0)]
        goal_node: Optional[Node] = None
        refinement_budget = 0

        kd: Optional[cKDTree] = None
        indexed_count = 0
        # At most `interval + 1` nodes can be added between loop-start
        # rebuilds (a newly connected goal may add the extra one).
        recent_positions = np.empty((self.spatial_index_rebuild_interval + 1, dim))
        recent_indices = np.empty(self.spatial_index_rebuild_interval + 1, dtype=int)
        recent_count = 0

        def rebuild_kd():
            nonlocal kd, indexed_count, recent_count
            positions = np.array([n.pos for n in nodes])
            kd = cKDTree(positions)
            indexed_count = len(nodes)
            recent_count = 0

        def append_to_spatial_buffer(node_index: int) -> None:
            nonlocal recent_count
            recent_positions[recent_count] = nodes[node_index].pos
            recent_indices[recent_count] = node_index
            recent_count += 1

        def nearest_node(point: np.ndarray) -> Node:
            """Exact nearest node across the frozen tree and recent buffer."""
            assert kd is not None
            best_dist, best_idx = kd.query(point)
            if recent_count:
                delta = recent_positions[:recent_count] - point
                dist_sq = np.einsum("ij,ij->i", delta, delta)
                local_idx = int(np.argmin(dist_sq))
                if float(dist_sq[local_idx]) < best_dist * best_dist:
                    best_idx = int(recent_indices[local_idx])
            return nodes[int(best_idx)]

        def nearby_nodes(point: np.ndarray, radius: float) -> List[Node]:
            """Exact radius neighbours across the frozen tree and buffer."""
            assert kd is not None
            indices = kd.query_ball_point(point, radius)
            result = [nodes[int(i)] for i in indices]
            radius_sq = radius * radius
            if recent_count:
                delta = recent_positions[:recent_count] - point
                dist_sq = np.einsum("ij,ij->i", delta, delta)
                result.extend(
                    nodes[int(idx)]
                    for idx in recent_indices[:recent_count][dist_sq <= radius_sq]
                )
            return result

        # Build once for the start node.  Afterwards it is rebuilt only when
        # the recent-node buffer reaches the configured bounded size.
        rebuild_kd()

        def _prev_pos(node: Node) -> np.ndarray:
            return node.parent.pos if node.parent is not None else point_before

        def _inside_turn_check_band(pos: np.ndarray) -> bool:
            """True if pos is in the outer 10% of the search area radius."""
            return bool(np.linalg.norm(pos - obstacle_center) >= turn_check_band_start)

        def _guarded_turn_angle_ok(
            prev_pos: np.ndarray,
            mid_pos: np.ndarray,
            new_pos: np.ndarray,
            *,
            force: bool = False,
        ) -> bool:
            """
            Apply the turn-angle check only in the protected outer band.

            force=True is used for fixed boundary transitions, which must
            always respect the turn-angle limit regardless of radius.
            """
            if force or _inside_turn_check_band(mid_pos):
                return self._turn_angle_ok(prev_pos, mid_pos, new_pos)
            return True

        if self.full_circle:
            def _sector_point_ok(
                point: np.ndarray,
                *,
                print_result: bool = False,
            ) -> bool:
                return True
        else:
            def _sector_point_ok(
                point: np.ndarray,
                *,
                print_result: bool = False,
            ) -> bool:
                return self._point_in_allowed_sector(
                    point,
                    start,
                    obstacle_center,
                    print_result=print_result,
                )

            # The goal is part of the generated path. If it is in Sector 3,
            # no valid point-only solution can connect to it.
            if not _sector_point_ok(goal, print_result=True):
                print("RRT* goal is in prohibited Sector 3; no solution possible.")
                return None, 0

        def _children_turns_ok_after_reparent(
            node: Node,
            new_parent_pos: np.ndarray,
        ) -> bool:
            """
            Rewiring changes the incoming edge of `node`.

            If `node` is in the protected outer band, all existing turns
            through `node` must remain valid after the parent change. Middle
            nodes outside the protected band intentionally ignore this check.
            """
            if not _inside_turn_check_band(node.pos):
                return True
            return all(
                self._turn_angle_ok(new_parent_pos, node.pos, child.pos)
                for child in node.children
            )

        # The straight segment is the globally shortest possible solution.
        # Testing it up front avoids all sampling/indexing work for the common
        # case where the local trajectory can safely stay direct.
        if self.direct_path_fastpath:
            direct_ok = (
                self._edge_free(start, goal, obstacle_center, search_inner)
                and self._turn_angle_ok(point_before, start, goal)
                and self._turn_angle_ok(start, goal, point_after)
            )
            if direct_ok:
                return [start.copy(), goal.copy()], 0

        iters = 0
        for iters in range(1, self.max_iter + 1):

            # Early exit once we have a goal + refinement budget spent.
            if goal_node is not None:
                if refinement_budget <= 0:
                    break
                refinement_budget -= 1

            if len(nodes) - indexed_count >= self.spatial_index_rebuild_interval:
                rebuild_kd()

            # 1. Sample from search_inner to search_outer annulus
            # Do not spend roughly one quarter of the samples in a region
            # that is guaranteed to be rejected in restricted-circle mode.
            # Goal samples are safe here because an invalid goal was handled
            # above.
            q_rand = self._sample(goal, obstacle_center, dim, search_inner, search_outer)
            while not _sector_point_ok(q_rand):
                q_rand = self._sample(goal, obstacle_center, dim, search_inner, search_outer)

            # 2. Nearest node in tree
            q_near = nearest_node(q_rand)

            # 3. Steer toward q_rand
            q_new_pos = self._steer(q_near.pos, q_rand)

            # Reject every newly generated node inside prohibited Sector 3.
            if not _sector_point_ok(q_new_pos):
                continue

            # 4. Collision check on the edge (q_near -> q_new)
            if not self._edge_free(q_near.pos, q_new_pos, obstacle_center, search_inner):
                continue

            # 5. Find nodes within rewiring radius
            neighbours = nearby_nodes(q_new_pos, rewire_r)

            # 6. Choose best parent.
            #
            # The angle is checked only if the candidate parent is in the
            # outer 10% radius band. The root/start boundary is always checked
            # because it connects the replanned path to point_before.
            best_parent, best_cost = None, math.inf
            if _guarded_turn_angle_ok(
                _prev_pos(q_near),
                q_near.pos,
                q_new_pos,
                force=(q_near.parent is None),
            ):
                best_parent = q_near
                best_cost = q_near.cost + np.linalg.norm(q_new_pos - q_near.pos)

            for nb in neighbours:
                c = nb.cost + np.linalg.norm(q_new_pos - nb.pos)
                if (
                    c < best_cost
                    and self._edge_free(nb.pos, q_new_pos, obstacle_center, search_inner)
                    and _guarded_turn_angle_ok(
                        _prev_pos(nb),
                        nb.pos,
                        q_new_pos,
                        force=(nb.parent is None),
                    )
                ):
                    best_parent, best_cost = nb, c

            if best_parent is None:
                # No candidate parent satisfies the required angle limit.
                continue

            # 7. Add new node
            q_new = Node(pos=q_new_pos, parent=best_parent, cost=best_cost)
            best_parent.children.append(q_new)
            nodes.append(q_new)
            append_to_spatial_buffer(len(nodes) - 1)

            # 8. Rewire neighbours through q_new
            for nb in neighbours:
                if nb is best_parent:
                    continue

                new_cost = best_cost + np.linalg.norm(nb.pos - q_new_pos)

                # Rewiring creates/changes these turns:
                #   best_parent -> q_new -> nb
                #   q_new -> nb -> each existing child of nb
                # Middle nodes outside the outer 10% band ignore these checks.
                angle_ok = _guarded_turn_angle_ok(best_parent.pos, q_new_pos, nb.pos)
                angle_ok = angle_ok and _children_turns_ok_after_reparent(nb, q_new_pos)

                if nb is goal_node:
                    # The fixed boundary turn goal -> point_after is always
                    # checked, regardless of the goal radius.
                    angle_ok = angle_ok and self._turn_angle_ok(q_new_pos, nb.pos, point_after)

                if (
                    new_cost < nb.cost
                    and self._edge_free(q_new_pos, nb.pos, obstacle_center, search_inner)
                    and angle_ok
                ):
                    # Detach nb from old parent
                    if nb.parent is not None:
                        try:
                            nb.parent.children.remove(nb)
                        except ValueError:
                            pass
                    nb.parent = q_new
                    q_new.children.append(nb)
                    self._propagate_cost(nb, new_cost)

            # 9. Goal check.
            #
            # Two angle checks may matter here:
            #   best_parent -> q_new -> goal      only if q_new is protected
            #   q_new -> goal -> point_after      always, fixed boundary
            dist_to_goal = float(np.linalg.norm(q_new_pos - goal))
            if dist_to_goal <= self.goal_tolerance:
                goal_cost = best_cost + dist_to_goal
                if (
                    (goal_node is None or goal_cost < goal_node.cost)
                    and _guarded_turn_angle_ok(best_parent.pos, q_new_pos, goal)
                    and self._turn_angle_ok(q_new_pos, goal, point_after)
                ):
                    # Connect to goal
                    gn = Node(pos=goal.copy(), parent=q_new, cost=goal_cost)
                    q_new.children.append(gn)
                    nodes.append(gn)
                    append_to_spatial_buffer(len(nodes) - 1)
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

    @staticmethod
    def _point_in_allowed_sector(
        point: np.ndarray,
        entry_waypoint: np.ndarray,
        obstacle_center: np.ndarray,
        *,
        print_result: bool = False,
    ) -> bool:
        """
        Classify a point into one of four sectors around the obstacle center.

        Motion direction:
            entry_waypoint -> obstacle_center

        Sector numbering, proceeding anti-clockwise:
            Sector 1: forward-right
            Sector 2: forward-left
            Sector 3: backward-left  <-- prohibited
            Sector 4: backward-right

        Points on either dividing line are allowed.

        A single Cartesian convention is used throughout the planner:
            positive X points right;
            positive Y points up.
        """
        point = np.asarray(point, dtype=float)
        entry_waypoint = np.asarray(entry_waypoint, dtype=float)
        obstacle_center = np.asarray(obstacle_center, dtype=float)

        if (
            point.shape != (2,)
            or entry_waypoint.shape != (2,)
            or obstacle_center.shape != (2,)
        ):
            raise ValueError("The sector check requires 2-D points.")

        # Motion direction: entry waypoint -> obstacle center
        direction = obstacle_center - entry_waypoint
        direction_norm = float(np.linalg.norm(direction))

        if direction_norm < 1e-12:
            raise ValueError(
                "entry_waypoint must differ from obstacle_center."
            )

        forward_unit = direction / direction_norm

        # Fixed Cartesian Y-up convention. Rotating a vector 90 degrees
        # anti-clockwise produces its left-facing perpendicular.
        left_unit = np.array([
            -forward_unit[1],
            forward_unit[0],
        ])

        relative_point = point - obstacle_center

        forward_component = float(relative_point @ forward_unit)
        left_component = float(relative_point @ left_unit)

        tolerance = 1e-10 * max(
            1.0,
            float(np.linalg.norm(relative_point)),
        )

        # Points on either dividing line are allowed.
        if (
            abs(forward_component) <= tolerance
            or abs(left_component) <= tolerance
        ):
            sector = 0
            sector_description = "boundary"
            is_allowed = True

        elif forward_component > 0.0 and left_component < 0.0:
            sector = 1
            sector_description = "forward-right"
            is_allowed = True

        elif forward_component > 0.0 and left_component > 0.0:
            sector = 2
            sector_description = "forward-left"
            is_allowed = True

        elif forward_component < 0.0 and left_component > 0.0:
            sector = 3
            sector_description = "backward-left"
            is_allowed = False  # Sector 3 is prohibited

        else:
            sector = 4
            sector_description = "backward-right"
            is_allowed = True

        if print_result:
            print(
                f"Point: {point}\n"
                f"Entry waypoint: {entry_waypoint}\n"
                f"Obstacle center: {obstacle_center}\n"
                f"Coordinate system: Cartesian Y-up\n"
                f"Motion direction: {forward_unit}\n"
                f"Forward component: {forward_component:.6f}\n"
                f"Left component: {left_component:.6f}\n"
                f"Sector: {sector} ({sector_description})\n"
                f"Allowed: {is_allowed}"
            )

        return is_allowed

    def _turn_angle_ok(
        self,
        prev_pos: np.ndarray,
        mid_pos: np.ndarray,
        new_pos: np.ndarray,
    ) -> bool:
        """
        True if the relative angle between segment (prev_pos -> mid_pos)
        and segment (mid_pos -> new_pos) does not exceed
        ``self.max_turn_angle_deg``.

        The angle is measured as deviation from straight-ahead
        (0 deg = continuing straight, 180 deg = doubling back).
        If ``max_turn_angle_deg`` is None the check is disabled.
        """
        if self.max_turn_angle_deg is None:
            return True
        v1 = mid_pos - prev_pos
        v2 = new_pos - mid_pos
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-12 or n2 < 1e-12:
            return True
        cos_angle = float(np.clip((v1 @ v2) / (n1 * n2), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(cos_angle))
        return angle_deg <= self.max_turn_angle_deg

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

    def _smooth_rrt_path(
        self,
        rrt_path: List[np.ndarray],
        obstacle_center: np.ndarray,
        search_inner: float,
        point_before: np.ndarray,
        point_after: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Shortcut-smooth only the RRT* segment from entry_wp to exit_wp.

        This does not touch the original trajectory outside the replanned
        interval. A shortcut is accepted only if:
          - the replacement edge stays outside search_inner, and
          - the turn-angle constraint is still valid at the shortcut endpoints.

        Sector validation is point-only. Because smoothing reuses existing
        validated path points, it does not perform a sector check on the
        interior of the replacement segment.
        """
        if len(rrt_path) <= 2 or self.smoothing_iterations <= 0:
            return [p.copy() for p in rrt_path]

        smoothed = [p.copy() for p in rrt_path]

        for _ in range(self.smoothing_iterations):
            if len(smoothed) <= 2:
                break

            # Pick two non-consecutive points and try to replace the
            # intermediate polyline with one direct segment.
            i = int(self.rng.integers(0, len(smoothed) - 2))
            j = int(self.rng.integers(i + 2, len(smoothed)))

            if not self._edge_free(smoothed[i], smoothed[j], obstacle_center, search_inner):
                continue

            prev_i = point_before if i == 0 else smoothed[i - 1]
            next_j = point_after if j == len(smoothed) - 1 else smoothed[j + 1]

            if not self._turn_angle_ok(prev_i, smoothed[i], smoothed[j]):
                continue
            if not self._turn_angle_ok(smoothed[i], smoothed[j], next_j):
                continue

            smoothed = smoothed[:i + 1] + smoothed[j:]

        return smoothed

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
