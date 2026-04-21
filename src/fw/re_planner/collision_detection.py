import numpy as np
from typing import Tuple, Optional, Union


class CollisionDetector:
    """
    Detect potential collisions between a ship and an obstacle.

    The ship follows a known polyline trajectory at constant speed magnitude
    derived from ship_velocity, but starts from its current ship_position,
    which is projected onto the path.

    Returns:
    - collision_detected
    - cpa_point
    - tcpa
    - dcpa
    """

    def __init__(self, min_distance: float = 40.0, time_horizon: float = 50.0):
        self.min_distance = float(min_distance)
        self.time_horizon = float(time_horizon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_collision(
        self,
        obstacle: Union[np.ndarray, list],
        ship_position: Union[np.ndarray, list],
        ship_trajectory: Union[np.ndarray, list],
        ship_velocity: Union[np.ndarray, list],
    ) -> Tuple[list, list, list, list]:
        """
        Predict whether the ship will come within min_distance of the obstacle
        within time_horizon.

        Args:
            obstacle: Obstacle object.
            ship_position: Current ship position [x, y] or [x, y, z].
            ship_trajectory: Ship path checkpoints, shape (N, D) or (D,).
            ship_velocity: Current ship velocity vector (magnitude used as speed).

        Returns:
            (collision_detected, cpa_point, tcpa, dcpa)
        """
        dcpas, cpa_points, tcpas = self._calculate_dcpa_tcpa(
            obstacle=obstacle,
            ship_position=ship_position,
            ship_trajectory=ship_trajectory,
            ship_velocity=ship_velocity,
        )

        collisions_detected = []

        for i in range(len(dcpas)):
            if dcpas[i] is None or tcpas[i] is None:
                collisions_detected.append(False)
            elif 0.0 <= tcpas[i] <= self.time_horizon and dcpas[i] < self.min_distance:
                collisions_detected.append(True)
            else:
                collisions_detected.append(False)

        return collisions_detected, cpa_points, tcpas, dcpas

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _calculate_dcpa_tcpa(
        self,
        obstacle,
        ship_position,
        ship_trajectory,
        ship_velocity,
    ) -> Tuple[list, list, list]:

        obstacles = self._normalize_obstacles(obstacle)
        ship_position = np.asarray(ship_position, dtype=float)
        ship_trajectory = np.asarray(ship_trajectory, dtype=float)
        ship_velocity = np.asarray(ship_velocity, dtype=float)

        cpa_points = []
        tcpas = []
        dcpas = []

        remaining_path = self._build_remaining_path_from_ship_position(ship_position, ship_trajectory)

        ship_speed = float(np.linalg.norm(ship_velocity))
        if ship_speed <= 0.0:
            ship_speed = 1e-6  # Avoid division by zero; treat as very slow movement

        for entity in obstacles:
            obstacle_pos = np.asarray(entity.get_position(), dtype=float)
            obstacle_vel = np.asarray(entity.get_velocity(), dtype=float)
            obstacle_type = str(entity.get_entity_type()).lower()

            if obstacle_type == "static":
                dcpa, cpa_point, tcpa = self._cpa_static_obstacle(
                    obstacle_pos=obstacle_pos,
                    ship_trajectory=remaining_path,
                    ship_speed=ship_speed,
                )
            else:
                dcpa, cpa_point, tcpa = self._cpa_moving_obstacle_analytic(
                    obstacle_pos=obstacle_pos,
                    obstacle_vel=obstacle_vel,
                    ship_trajectory=remaining_path,
                    ship_speed=ship_speed,
                )

            cpa_points.append(cpa_point)
            tcpas.append(tcpa)
            dcpas.append(dcpa)

        return dcpas, cpa_points, tcpas

    def _normalize_obstacles(self, obstacle: Union[object, list, tuple]) -> list:
        """
        Normalize input to a flat list of obstacle entities.
        """
        if obstacle is None:
            return []

        if isinstance(obstacle, (list, tuple)):
            normalized = []
            for item in obstacle:
                if isinstance(item, (list, tuple)):
                    normalized.extend(item)
                else:
                    normalized.append(item)
            return normalized

        return [obstacle]

    # ------------------------------------------------------------------
    # Path preprocessing
    # ------------------------------------------------------------------

    def _build_remaining_path_from_ship_position(
        self,
        ship_position: np.ndarray,
        ship_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Project ship_position onto the segment between the first two checkpoints
        of ship_trajectory, then return a new path made of:
            [projected_point, checkpoint_1, checkpoint_2, ...]

        In other words:
        - only the first segment (pts[0] -> pts[1]) is considered
        - the result contains the projected point
        - all original checkpoints are kept except the first one

        Example:
            input path: [A, B, C, D]
            output: [proj_on_AB, B, C, D]
        """
        pts = np.asarray(ship_trajectory, dtype=np.float32)

        if len(pts) == 0:
            raise ValueError("ship_trajectory cannot be empty.")

        if len(pts) == 1:
            return np.vstack([ship_position.astype(np.float32), pts[0].copy()])

        p0 = pts[0]
        p1 = pts[1]
        seg = p1 - p0
        seg_len_sq = float(seg @ seg)

        if seg_len_sq <= 1e-12:
            proj = p0.copy()
        else:
            alpha = float(np.clip(((ship_position - p0) @ seg) / seg_len_sq, 0.0, 1.0))
            proj = p0 + alpha * seg

        return np.vstack([proj.astype(np.float32), pts[1:].copy()])

    # ------------------------------------------------------------------
    # Static obstacle
    # ------------------------------------------------------------------

    def _cpa_static_obstacle(
        self,
        obstacle_pos: np.ndarray,
        ship_trajectory: np.ndarray,
        ship_speed: float,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Exact DCPA/TCPA for a static obstacle along the remaining path.
        """
        pts = ship_trajectory

        min_dist = np.inf
        best_ship_pt = None
        best_path_dist = 0.0
        cumulative = 0.0

        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]

            seg = p1 - p0
            seg_len_sq = float(seg @ seg)
            seg_len = float(np.sqrt(seg_len_sq)) if seg_len_sq > 0.0 else 0.0

            if seg_len_sq <= 1e-12:
                alpha = 0.0
                closest = p0.copy()
            else:
                alpha = float(np.clip(((obstacle_pos - p0) @ seg) / seg_len_sq, 0.0, 1.0))
                closest = p0 + alpha * seg

            dist = float(np.linalg.norm(obstacle_pos - closest))

            if dist < min_dist:
                min_dist = dist
                best_ship_pt = closest
                best_path_dist = cumulative + alpha * seg_len

            cumulative += seg_len

        if best_ship_pt is None:
            best_ship_pt = pts[0].copy()
            min_dist = float(np.linalg.norm(best_ship_pt - obstacle_pos))
            best_path_dist = 0.0

        tcpa = min(best_path_dist / ship_speed, self.time_horizon)
        cpa_point = obstacle_pos.copy()
        return min_dist, cpa_point, tcpa

    # ------------------------------------------------------------------
    # Moving obstacle
    # ------------------------------------------------------------------

    def _cpa_moving_obstacle_analytic(
        self,
        obstacle_pos: np.ndarray,
        obstacle_vel: np.ndarray,
        ship_trajectory: np.ndarray,
        ship_speed: float,
    ) -> Tuple[Optional[float], Optional[np.ndarray], Optional[float]]:
        """
        Analytical CPA for a moving obstacle against a piecewise-linear ship path.
        """
        trajectory_times = self._build_trajectory_times(ship_trajectory, ship_speed)
        max_time = min(self.time_horizon, trajectory_times[-1])

        min_dist_sq = np.inf
        best_t = None
        best_cpa = None

        pts = ship_trajectory

        for i in range(len(pts) - 1):
            t_i = trajectory_times[i]
            t_next = trajectory_times[i + 1]

            if t_i > max_time:
                break

            t_exit = min(t_next, max_time)
            seg_time = t_exit - t_i
            if seg_time <= 0.0:
                continue

            full_seg_time = t_next - t_i
            frac = seg_time / full_seg_time if full_seg_time > 0.0 else 1.0

            p0 = pts[i]
            p1 = pts[i] + (pts[i + 1] - pts[i]) * frac

            v_ship = (p1 - p0) / seg_time if seg_time > 0.0 else np.zeros_like(p0)
            obs_at_ti = obstacle_pos + obstacle_vel * t_i
            r0 = p0 - obs_at_ti
            v_rel = v_ship - obstacle_vel

            # Lower-bound pruning
            r0_norm = float(np.linalg.norm(r0))
            v_rel_norm = float(np.linalg.norm(v_rel))
            lb = max(0.0, r0_norm - v_rel_norm * seg_time)

            if lb * lb >= min_dist_sq:
                continue

            vv = float(v_rel @ v_rel)
            if vv < 1e-12:
                tau_star = 0.0
            else:
                tau_star = float(np.clip(-(r0 @ v_rel) / vv, 0.0, seg_time))

            r_tau = r0 + v_rel * tau_star
            d_sq = float(r_tau @ r_tau)

            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_t = t_i + tau_star

                ship_pt = p0 + v_ship * tau_star
                obs_pt = obs_at_ti + obstacle_vel * tau_star
                best_cpa = (ship_pt + obs_pt) / 2.0

        if best_t is None:
            return None, None, None

        return float(np.sqrt(min_dist_sq)), best_cpa, float(best_t)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_trajectory_times(self, trajectory: np.ndarray, speed: float) -> np.ndarray:
        """
        Build cumulative timestamps for constant-speed traversal of a polyline.
        """
        diffs = np.diff(trajectory, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)

        times = np.empty(len(trajectory), dtype=float)
        times[0] = 0.0
        np.cumsum(seg_lengths / speed, out=times[1:])
        return times

    def _validate_inputs(
        self,
        obstacle_pos: np.ndarray,
        obstacle_vel: np.ndarray,
        ship_position: np.ndarray,
        ship_trajectory: np.ndarray,
        ship_velocity: np.ndarray,
    ) -> None:
        if obstacle_pos.ndim != 1:
            raise ValueError("obstacle_pos must be a 1-D vector.")
        if obstacle_vel.ndim != 1:
            raise ValueError("obstacle_vel must be a 1-D vector.")
        if ship_position.ndim != 1:
            raise ValueError("ship_position must be a 1-D vector.")
        if ship_velocity.ndim != 1:
            raise ValueError("ship_velocity must be a 1-D vector.")

        if obstacle_pos.shape != obstacle_vel.shape:
            raise ValueError("obstacle_pos and obstacle_vel must have the same dimensions.")
        if obstacle_pos.shape != ship_position.shape:
            raise ValueError("obstacle_pos and ship_position must have the same dimensions.")
        if obstacle_pos.shape != ship_velocity.shape:
            raise ValueError("obstacle_pos and ship_velocity must have the same dimensions.")

        if ship_trajectory.ndim == 1:
            if ship_trajectory.shape != obstacle_pos.shape:
                raise ValueError("Single trajectory point must match obstacle dimensions.")
        elif ship_trajectory.ndim == 2:
            if ship_trajectory.shape[1] != obstacle_pos.shape[0]:
                raise ValueError("Each trajectory point must match obstacle dimensions.")
            if len(ship_trajectory) == 0:
                raise ValueError("ship_trajectory cannot be empty.")
        else:
            raise ValueError("ship_trajectory must be a 1-D point or 2-D array of points.")

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_min_distance(self, min_distance: float) -> None:
        if min_distance <= 0.0:
            raise ValueError("min_distance must be positive.")
        self.min_distance = float(min_distance)

    def set_time_horizon(self, time_horizon: float) -> None:
        if time_horizon <= 0.0:
            raise ValueError("time_horizon must be positive.")
        self.time_horizon = float(time_horizon)