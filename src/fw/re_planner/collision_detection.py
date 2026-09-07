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

    def __init__(self, min_distance: float = 140.0, time_horizon: float = 500.0):
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
    ) -> Tuple[list, list, list, list, list]:
        """
        Predict whether the ship will come within min_distance of the obstacle
        within time_horizon.

        Args:
            obstacle: Obstacle object.
            ship_position: Current ship position [x, y] or [x, y, z].
            ship_trajectory: Ship path checkpoints, shape (N, D) or (D,).
            ship_velocity: Current ship velocity vector (magnitude used as speed).

        Returns:
            (collision_detected, cpa_point, tcpa, dcpa, relative_heading)
        """
        dcpas, cpa_points, tcpas, relative_headings = self._calculate_dcpa_tcpa(
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

        return collisions_detected, cpa_points, tcpas, dcpas, relative_headings

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _calculate_dcpa_tcpa(
        self,
        obstacle,
        ship_position,
        ship_trajectory,
        ship_velocity,
    ) -> Tuple[list, list, list, list]:

        obstacles = self._normalize_obstacles(obstacle)
        ship_position = np.asarray(ship_position, dtype=float)
        ship_trajectory = np.asarray(ship_trajectory, dtype=float)
        ship_velocity = np.asarray(ship_velocity, dtype=float)

        cpa_points = []
        tcpas = []
        dcpas = []
        relative_headings = []

        remaining_path = self._build_remaining_path_from_ship_position(ship_position, ship_trajectory)

        ship_speed = float(np.linalg.norm(ship_velocity))
        if ship_speed <= 0.0:
            ship_speed = 1e-6  # Avoid division by zero; treat as very slow movement

        for entity in obstacles:
            obstacle_pos = np.asarray(entity.get_position(), dtype=float)
            obstacle_vel = np.asarray(entity.get_velocity(), dtype=float)
            obstacle_heading = float(entity.get_heading())
            obstacle_type = str(entity.get_entity_type()).lower()
            print(f"TS speed: {np.linalg.norm(obstacle_vel):.2f} m/s, OS speed: {ship_speed:.2f} m/s,")

            if obstacle_type == "static":
                dcpa, cpa_point, tcpa, relative_heading = self._cpa_static_obstacle(
                    obstacle_pos=obstacle_pos,
                    ship_trajectory=remaining_path,
                    ship_speed=ship_speed,
                )
            else:
                dcpa, cpa_point, tcpa, relative_heading = self._cpa_moving_obstacle(
                    obstacle_pos=obstacle_pos,
                    obstacle_velocity=obstacle_vel,
                    ship_trajectory=remaining_path,
                    ship_speed=ship_speed,
                )

            cpa_points.append(cpa_point)
            tcpas.append(tcpa)
            dcpas.append(dcpa)
            relative_headings.append(relative_heading)

        return dcpas, cpa_points, tcpas, relative_headings

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
    ) -> Tuple[float, np.ndarray, float, float]:
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
        return min_dist, cpa_point, tcpa, 180.0

    # ------------------------------------------------------------------
    # Moving obstacle
    # ------------------------------------------------------------------

    def _cpa_moving_obstacle(
        self,
        obstacle_pos: np.ndarray,
        obstacle_velocity: np.ndarray,
        ship_trajectory: np.ndarray,
        ship_speed: float,
    ) -> Tuple[float, np.ndarray, float, float]:
        """
        Calculate DCPA and TCPA for an obstacle moving with a constant
        two-dimensional velocity vector.

        Coordinate convention:
            x positive = East
            y positive = North

        Args:
            obstacle_pos:
                Current obstacle position [x, y].

            obstacle_velocity:
                Constant obstacle velocity [vx, vy].

                Examples:
                    [5, 0]   -> East
                    [0, 5]   -> North
                    [-5, 0]  -> West
                    [0, -5]  -> South

            ship_trajectory:
                Piecewise-linear ship trajectory with shape (N, 2).

            ship_speed:
                Constant scalar speed of the ship.

        Returns:
            min_dist:
                Minimum distance between the ship and obstacle.

            cpa_point:
                Obstacle position at CPA.

            tcpa:
                Time from the current instant to CPA.
        """
        pts = np.asarray(ship_trajectory, dtype=float)
        obstacle_pos = np.asarray(
            obstacle_pos,
            dtype=float,
        ).reshape(-1)

        obstacle_velocity = np.asarray(
            obstacle_velocity,
            dtype=float,
        ).reshape(-1)

        ship_speed = float(
            np.asarray(ship_speed, dtype=float).squeeze()
        )

        if len(pts) == 0:
            raise ValueError(
                "ship_trajectory must contain at least one point."
            )

        if obstacle_pos.size != 2:
            raise ValueError(
                "obstacle_pos must contain exactly two values [x, y]. "
                f"Received shape {np.shape(obstacle_pos)}."
            )

        if obstacle_velocity.size != 2:
            raise ValueError(
                "obstacle_velocity must contain exactly two values [vx, vy]. "
                f"Received shape {np.shape(obstacle_velocity)}."
            )

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                "ship_trajectory must have shape (N, 2). "
                f"Received shape {pts.shape}."
            )

        if ship_speed < 0.0:
            raise ValueError("ship_speed cannot be negative.")

        min_dist = np.inf
        best_tcpa = 0.0
        best_obstacle_point = obstacle_pos.copy()

        def evaluate_interval(
            ship_start: np.ndarray,
            ship_velocity: np.ndarray,
            interval_start: float,
            interval_duration: float,
        ) -> None:
            """
            Calculate the minimum distance during one interval in which
            both the ship and obstacle have constant velocities.
            """
            nonlocal min_dist, best_tcpa, best_obstacle_point

            if interval_duration < 0.0:
                return

            # Obstacle position when the current ship segment begins.
            obstacle_start = (
                obstacle_pos
                + obstacle_velocity * interval_start
            )

            # Relative state: ship relative to obstacle.
            relative_position = ship_start - obstacle_start
            relative_velocity = ship_velocity - obstacle_velocity

            relative_speed_sq = float(
                relative_velocity @ relative_velocity
            )

            if relative_speed_sq <= 1e-12:
                # No relative movement: distance is constant.
                local_tcpa = 0.0
            else:
                # Unconstrained CPA time for this segment.
                local_tcpa = -float(
                    relative_position @ relative_velocity
                ) / relative_speed_sq

                # CPA must occur while the ship is on this segment.
                local_tcpa = float(
                    np.clip(
                        local_tcpa,
                        0.0,
                        interval_duration,
                    )
                )

            absolute_tcpa = interval_start + local_tcpa

            ship_cpa = (
                ship_start
                + ship_velocity * local_tcpa
            )

            obstacle_cpa = (
                obstacle_start
                + obstacle_velocity * local_tcpa
            )

            distance = float(
                np.linalg.norm(ship_cpa - obstacle_cpa)
            )

            if distance < min_dist:
                min_dist = distance
                best_tcpa = absolute_tcpa
                best_obstacle_point = obstacle_cpa.copy()

        # The ship is stationary if it has no path or its speed is zero.
        if len(pts) == 1 or ship_speed <= 1e-12:
            evaluate_interval(
                ship_start=pts[0],
                ship_velocity=np.zeros(2),
                interval_start=0.0,
                interval_duration=self.time_horizon,
            )

            return min_dist, best_obstacle_point, best_tcpa, 0.0

        current_time = 0.0

        for i in range(len(pts) - 1):
            if current_time >= self.time_horizon:
                break

            p0 = pts[i]
            p1 = pts[i + 1]

            segment = p1 - p0
            segment_length = float(np.linalg.norm(segment))

            # Skip consecutive identical trajectory points.
            if segment_length <= 1e-12:
                continue

            segment_direction = segment / segment_length

            # Ship velocity vector on this trajectory segment.
            ship_velocity = ship_speed * segment_direction

            segment_duration = segment_length / ship_speed

            # Only analyze the part inside the prediction horizon.
            available_duration = min(
                segment_duration,
                self.time_horizon - current_time,
            )

            evaluate_interval(
                ship_start=p0,
                ship_velocity=ship_velocity,
                interval_start=current_time,
                interval_duration=available_duration,
            )

            current_time += segment_duration

        # If the ship reaches its destination before the time horizon,
        # assume it remains stationary at the final trajectory point.
        if current_time < self.time_horizon:
            evaluate_interval(
                ship_start=pts[-1],
                ship_velocity=np.zeros(2),
                interval_start=current_time,
                interval_duration=self.time_horizon - current_time,
            )

        relative_heading = self._calculate_relative_heading(
        target_ship_velocity=obstacle_velocity,
        ship_trajectory=pts,
        cpa_point=best_obstacle_point,
        )

        return min_dist, best_obstacle_point, best_tcpa, relative_heading

    def _calculate_relative_heading(
        self,
        target_ship_velocity: np.ndarray,
        ship_trajectory: np.ndarray,
        cpa_point: np.ndarray,
    ) -> float:
        """
        Calculate the relative heading between the OS and TS at CPA.

        The OS heading is obtained from the trajectory segment closest to
        cpa_point. The CPA point does not need to lie exactly on the segment.

        If cpa_point is at the junction of two trajectory segments, the
        outgoing segment is used.

        Relative-heading convention:
            0 degrees   -> overtaking / same direction
            90 degrees  -> right crossing
            180 degrees -> head-on
            270 degrees -> left crossing

        The angle is measured clockwise from the OS heading to the TS heading.

        Args:
            target_ship_velocity:
                Constant TS velocity vector [vx, vy].

            ship_trajectory:
                Piecewise-linear OS trajectory with shape (N, 2).

            cpa_point:
                TS position at CPA [x, y].

        Returns:
            Relative heading in degrees in the range [0, 360).

            Returns np.nan if the TS is stationary or if the OS trajectory
            does not contain a valid segment.
        """
        target_ship_velocity = np.asarray(
            target_ship_velocity,
            dtype=float,
        ).reshape(-1)

        ship_trajectory = np.asarray(
            ship_trajectory,
            dtype=float,
        )

        cpa_point = np.asarray(
            cpa_point,
            dtype=float,
        ).reshape(-1)

        if target_ship_velocity.size != 2:
            raise ValueError(
                "target_ship_velocity must contain exactly two values [vx, vy]. "
                f"Received shape {target_ship_velocity.shape}."
            )

        if (
            ship_trajectory.ndim != 2
            or ship_trajectory.shape[1] != 2
        ):
            raise ValueError(
                "ship_trajectory must have shape (N, 2). "
                f"Received shape {ship_trajectory.shape}."
            )

        if cpa_point.size != 2:
            raise ValueError(
                "cpa_point must contain exactly two values [x, y]. "
                f"Received shape {cpa_point.shape}."
            )

        if np.linalg.norm(target_ship_velocity) <= 1e-12:
            # The TS heading is undefined.
            return float("nan")

        if len(ship_trajectory) < 2:
            # The OS trajectory has no segment from which to determine
            # its heading.
            return float("nan")

        closest_segment_direction = None
        minimum_distance = np.inf
        comparison_tolerance = 1e-10

        for i in range(len(ship_trajectory) - 1):
            segment_start = ship_trajectory[i]
            segment_end = ship_trajectory[i + 1]

            segment = segment_end - segment_start
            segment_length_squared = float(segment @ segment)

            # Ignore consecutive identical trajectory points.
            if segment_length_squared <= 1e-12:
                continue

            # Project cpa_point onto the infinite line containing the segment.
            projection_parameter = float(
                (cpa_point - segment_start) @ segment
                / segment_length_squared
            )

            # Restrict the projection to the finite trajectory segment.
            projection_parameter = float(
                np.clip(projection_parameter, 0.0, 1.0)
            )

            closest_point_on_segment = (
                segment_start
                + projection_parameter * segment
            )

            distance_to_segment = float(
                np.linalg.norm(
                    cpa_point - closest_point_on_segment
                )
            )

            is_closer = (
                distance_to_segment
                < minimum_distance - comparison_tolerance
            )

            is_equal_distance = np.isclose(
                distance_to_segment,
                minimum_distance,
                rtol=0.0,
                atol=comparison_tolerance,
            )

            # In case of equal distance, select the later segment.
            # Therefore, at a junction, the outgoing segment is used.
            if is_closer or is_equal_distance:
                minimum_distance = distance_to_segment
                closest_segment_direction = (
                    segment / np.sqrt(segment_length_squared)
                )

        if closest_segment_direction is None:
            # All trajectory points are identical.
            return float("nan")

        os_direction = closest_segment_direction
        ts_direction = (
            target_ship_velocity
            / np.linalg.norm(target_ship_velocity)
        )

        dot_product = float(
            np.clip(os_direction @ ts_direction, -1.0, 1.0)
        )

        cross_product = float(
            os_direction[0] * ts_direction[1]
            - os_direction[1] * ts_direction[0]
        )

        # Negative cross product converts the standard counterclockwise
        # mathematical angle into a clockwise angle.
        relative_heading = np.degrees(
            np.arctan2(
                -cross_product,
                dot_product,
            )
        ) % 360.0

        if np.isclose(relative_heading, 360.0, atol=1e-10):
            relative_heading = 0.0

        relative_heading = float(relative_heading) + 180.0
        if relative_heading >= 360.0:
            relative_heading -= 360.0

        return float(relative_heading)

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