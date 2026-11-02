import os
import json
import heapq
import numpy as np


def create_checkpoints_from_simple_path(points, max_distance):
    """
    Densify a path so consecutive checkpoints are at most `max_distance` apart.

    Args:
        points (list/array): Points defining the path.
        max_distance (float): Maximum distance between consecutive checkpoints.

    Returns:
        list of tuples: Checkpoint coordinates (original points plus evenly
        spaced intermediate ones).
    """

    points = np.array(points)

    if points.size == 0:
        return []

    final_checkpoints = [tuple(points[0])]

    for i in range(1, len(points)):
        current_point = np.array(final_checkpoints[-1])
        next_point = points[i]

        segment_vector = next_point - current_point
        segment_length = np.linalg.norm(segment_vector)

        if segment_length > max_distance:
            num_segments = int(np.ceil(float(segment_length) / max_distance))

            for j in range(1, num_segments):
                fraction = j / num_segments
                intermediate_point = current_point + fraction * segment_vector
                final_checkpoints.append(tuple(intermediate_point))

        final_checkpoints.append(tuple(next_point))

    return final_checkpoints


def is_ship_heading_correct(checkpoint, state):
    """
    Check whether the ship's heading ray crosses the checkpoint's perpendicular line ahead.

    `state` starts with [x, y, heading (rad)].
    """

    ship_position = np.array(state[:2])
    heading = state[2]

    heading_vector = np.array([np.cos(heading), np.sin(heading)])

    # Approximate the heading ray by a 10 km segment.
    ray_start = ship_position
    ray_end = ray_start + heading_vector * 10000

    line_point1 = np.array(checkpoint['perpendicular_line'][0])
    line_point2 = np.array(checkpoint['perpendicular_line'][1])

    def line_intersection(p1, p2, q1, q2):
        """
        Return the intersection point of two line segments, or None.
        """

        def _cross(a, b):
            """
            z-component of the 2-D cross product (np.cross on 2-D input is deprecated).
            """

            return a[0] * b[1] - a[1] * b[0]

        r = p2 - p1
        s = q2 - q1

        r_cross_s = _cross(r, s)
        q_minus_p = q1 - p1

        if r_cross_s == 0:
            return None  # parallel

        t = _cross(q_minus_p, s) / r_cross_s
        u = _cross(q_minus_p, r) / r_cross_s

        if 0 <= t <= 1 and 0 <= u <= 1:
            return p1 + t * r
        return None

    intersection = line_intersection(ray_start, ray_end, line_point1, line_point2)

    if intersection is not None:
        intersection_vector = intersection - ray_start
        if np.dot(intersection_vector, heading_vector) > 0:
            return True
    return False


def is_within_obstacle(point, obstacles, safety_margin=3):
    """
    Check if a point is within any circular obstacle with an added safety margin.
    """

    return any(np.linalg.norm(point - obs['pos']) < obs['radius'] + safety_margin for obs in obstacles)


def distance_point_to_line(p1, p2, point):
    """
    Approximate distance from a point to the segment p1-p2.

    The projection is clamped per coordinate to the segment's bounding box.
    """

    p1, p2, point = map(np.asarray, (p1, p2, point))

    line_len = np.linalg.norm(p2 - p1)
    if line_len == 0:
        return np.linalg.norm(point - p1)

    line_unit_vec = (p2 - p1) / line_len
    projection = np.dot(point - p1, line_unit_vec)
    closest_point = p1 + projection * line_unit_vec

    closest_point = np.clip(closest_point, np.minimum(p1, p2), np.maximum(p1, p2))

    return np.linalg.norm(closest_point - point)


def is_line_intersecting_obstacle(p1, p2, obstacles, radius=3):
    """
    Check whether the segment p1-p2 passes within `radius` of any obstacle point.
    """

    for obstacle in obstacles:
        if distance_point_to_line(p1, p2, obstacle) <= radius:
            return True
    return False


def get_n_closest_obstacles(point, obstacles, n=8):
    """
    Return the `n` obstacles (array of points) closest to `point`, nearest first.
    """

    distances = np.linalg.norm(obstacles - point, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_obstacles = obstacles[sorted_indices]
    return sorted_obstacles[:n]


def astar(start, goal, obstacles, min_grid, max_grid, boundary_threshold=2):
    """
    A* search on a unit grid (4-connected) avoiding obstacle points.

    Returns:
        list: Path from start to goal, or [] if none is found.
    """

    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))  # (f_score, node)

    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): np.linalg.norm(np.array(start) - np.array(goal))}

    while open_set:
        _, current = heapq.heappop(open_set)

        if np.allclose(current, goal, atol=1e-2):
            path = [goal]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        closest_obstacles = get_n_closest_obstacles(current, obstacles)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Stay boundary_threshold inside the grid limits.
            if not (
                min_grid + boundary_threshold <= neighbor[0] <= max_grid - boundary_threshold and
                min_grid + boundary_threshold <= neighbor[1] <= max_grid - boundary_threshold
            ):
                continue

            tentative_g_score = g_score[current] + np.hypot(dx, dy)

            # if is_within_obstacle(neighbor_pos, obstacles):
            if is_line_intersecting_obstacle(current, neighbor, closest_obstacles, radius=3):
                continue

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + np.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])

                if (f_score[neighbor], neighbor) not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []


def calculate_cumulative_distances(path):
    """
    Calculate cumulative distances along the path.
    """

    return np.hstack(([0], np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()))


def is_safe_from_obstacles(point, obstacles, safe_distance):
    """
    Check if a checkpoint is a safe distance away from any obstacle.
    """

    return all(np.linalg.norm(point - obs['pos']) >= (obs['radius'] + safe_distance) for obs in obstacles)


def generate_even_checkpoints(path, num_checkpoints, obstacles, safe_distance=5.0):
    """
    Return `num_checkpoints` evenly spaced points along the path.

    Points too close to an obstacle are skipped; the list is padded with the
    last path point. Paths with at most `num_checkpoints` points are returned as is.
    """

    if len(path) <= num_checkpoints:
        return path

    cumulative_distances = calculate_cumulative_distances(path)
    total_length = cumulative_distances[-1]
    checkpoint_distance = total_length / (num_checkpoints - 1)

    checkpoints = [path[0]]
    current_distance = checkpoint_distance

    for i in range(1, len(path)):
        while current_distance <= cumulative_distances[i] and len(checkpoints) < num_checkpoints:
            ratio = ((current_distance - cumulative_distances[i - 1]) /
                     (cumulative_distances[i] - cumulative_distances[i - 1]))
            checkpoint = (1 - ratio) * path[i - 1] + ratio * path[i]

            if is_safe_from_obstacles(checkpoint, obstacles, safe_distance):
                checkpoints.append(checkpoint)

            current_distance += checkpoint_distance

    while len(checkpoints) < num_checkpoints:
        checkpoints.append(path[-1])

    return checkpoints


def save_hyperparameters(params, filename="best_hyperparameters.json"):
    """
    Save hyperparameters to a JSON file.
    """

    with open(filename, 'w') as f:
        json.dump(params, f)


def load_hyperparameters(filename="best_hyperparameters.json"):
    """
    Load hyperparameters from a JSON file; None if missing or empty.
    """

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as f:
            return json.load(f)
    return None
