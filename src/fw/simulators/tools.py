import os
import json
import heapq
import numpy as np

from shapely.geometry import Point


def create_checkpoints_from_simple_path(points, max_distance):
    """
    Create checkpoints from a path ensuring no two consecutive points are further than max_distance apart.

    Args:
        points (list/array): List or numpy array of points defining the path.
        max_distance (float): Maximum allowed distance between consecutive checkpoints.

    Returns:
        list of tuples: A list of checkpoint coordinates with appropriate spacing.
    """
    # Convert to numpy array if not already
    points = np.array(points)
    
    # Check if array is empty
    if points.size == 0:
        return []
    
    final_checkpoints = [tuple(points[0])]  # Start with the first point
    
    for i in range(1, len(points)):
        current_point = np.array(final_checkpoints[-1])
        next_point = points[i]
        
        # Calculate vector between points
        segment_vector = next_point - current_point
        segment_length = np.linalg.norm(segment_vector)
        
        if segment_length > max_distance:
            # Calculate how many intermediate points we need
            num_segments = int(np.ceil(segment_length / max_distance))
            
            # Create evenly spaced intermediate points
            for j in range(1, num_segments):
                fraction = j / num_segments
                intermediate_point = current_point + fraction * segment_vector
                final_checkpoints.append(tuple(intermediate_point))
        
        # Add the next point
        final_checkpoints.append(tuple(next_point))
    
    return final_checkpoints


def is_ship_heading_correct(checkpoint, state):
    """
    Check if the ship's heading will cross the line segment formed by the perpendicular line.
    """
    # Ship's position and heading
    ship_position = np.array(state[:2])  # Assuming state contains [x, y, heading]
    heading = state[2]  # Heading in radians

    # Heading direction vector
    heading_vector = np.array([np.cos(heading), np.sin(heading)])

    # Extend the ship's trajectory into a ray
    ray_start = ship_position
    ray_end = ray_start + heading_vector * 10000  # Large number to create a long ray

    # Line segment points
    line_point1 = np.array(checkpoint['perpendicular_line'][0])
    line_point2 = np.array(checkpoint['perpendicular_line'][1])

    # Function to compute intersection of two line segments
    def line_intersection(p1, p2, q1, q2):
        """Find the intersection of two line segments if it exists."""
        r = p2 - p1
        s = q2 - q1

        r_cross_s = np.cross(r, s)
        q_minus_p = q1 - p1

        if r_cross_s == 0:
            return None  # Lines are parallel and non-intersecting

        t = np.cross(q_minus_p, s) / r_cross_s
        u = np.cross(q_minus_p, r) / r_cross_s

        if 0 <= t <= 1 and 0 <= u <= 1:
            return p1 + t * r
        return None

    # Check for intersection
    intersection = line_intersection(ray_start, ray_end, line_point1, line_point2)

    if intersection is not None:
        # Check if the intersection is in the forward direction of the ship
        intersection_vector = intersection - ray_start
        if np.dot(intersection_vector, heading_vector) > 0:
            return True  # Heading crosses the line in the forward direction
    return False  # No intersection or in the wrong direction


def check_collision_ship(ship_position, polygon):
    """
    Check if the ship makes contact with any edge of the polygon.
    """
    x, y = ship_position
    min_x, min_y, max_x, max_y = polygon.bounds

    # Combine bounding box checks
    if not (min_x <= x <= max_x and min_y <= y <= max_y):
        return False

    return polygon.contains(Point(ship_position))


def is_within_obstacle(point, obstacles, safety_margin=3):
    """Check if a point is within any circular obstacle with an added safety margin."""
    return any(np.linalg.norm(point - obs['pos']) < obs['radius'] + safety_margin for obs in obstacles)


def distance_point_to_line(p1, p2, point):
    """Calculate the distance from a point to a line segment defined by p1 and p2."""
    p1, p2, point = map(np.asarray, (p1, p2, point))

    line_len = np.linalg.norm(p2 - p1)
    if line_len == 0:
        return np.linalg.norm(point - p1)

    line_unit_vec = (p2 - p1) / line_len
    projection = np.dot(point - p1, line_unit_vec)
    closest_point = p1 + projection * line_unit_vec

    # Clamp the closest point to the line segment boundaries
    closest_point = np.clip(closest_point, np.minimum(p1, p2), np.maximum(p1, p2))

    return np.linalg.norm(closest_point - point)


def is_line_intersecting_obstacle(p1, p2, obstacles, radius=3):
    """Check if a line segment intersects any circular obstacle."""
    for obstacle in obstacles:
        if distance_point_to_line(p1, p2, obstacle) <= radius:
            return True
    return False


def get_n_closest_obstacles(point, obstacles, n=8):
    """Order the obstacles based on their distance to the reference point."""

    # Calculate the distance from each obstacle to the reference point
    distances = np.linalg.norm(obstacles - point, axis=1)

    # Sort the obstacles based on distances
    sorted_indices = np.argsort(distances)

    # Return the n closest obstacles ordered by distance
    sorted_obstacles = obstacles[sorted_indices]
    return sorted_obstacles[:n]


def astar(start, goal, obstacles, min_grid, max_grid, boundary_threshold=2):
    """Perform A* search avoiding circular obstacles."""
    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))  # Priority queue with (f_score, node)

    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): np.linalg.norm(np.array(start) - np.array(goal))}  # Start with heuristic to goal

    while open_set:
        _, current = heapq.heappop(open_set)

        if np.allclose(current, goal, atol=1e-2):  # Goal reached
            path = [goal]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]   # Return the path as a list of points

        closest_obstacles = get_n_closest_obstacles(current, obstacles)

        # Generate neighbors (assuming grid-like movement)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Enforce grid boundary constraints and threshold distance
            if not (
                min_grid + boundary_threshold <= neighbor[0] <= max_grid - boundary_threshold and
                min_grid + boundary_threshold <= neighbor[1] <= max_grid - boundary_threshold
            ):
                continue  # Skip this neighbor if it’s outside the allowed boundary region

            tentative_g_score = g_score[current] + np.hypot(dx, dy)

            # Check if the path to this neighbor crosses any obstacles
            # if is_within_obstacle(neighbor_pos, obstacles):
            if is_line_intersecting_obstacle(current, neighbor, closest_obstacles, radius=3):
                continue  # Skip if it crosses an obstacle

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + np.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])

                # Only add neighbor to open set if it's not already in it
                if (f_score[neighbor], neighbor) not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # Return an empty path if no valid path is found


def calculate_cumulative_distances(path):
    """Calculate cumulative distances along the path."""
    return np.hstack(([0], np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()))


def is_safe_from_obstacles(point, obstacles, safe_distance):
    """Check if a checkpoint is a safe distance away from any obstacle."""
    return all(np.linalg.norm(point - obs['pos']) >= (obs['radius'] + safe_distance) for obs in obstacles)


def generate_even_checkpoints(path, num_checkpoints, obstacles, safe_distance=5.0):
    """Generate evenly spaced checkpoints along the path."""
    if len(path) <= num_checkpoints:
        return path

    cumulative_distances = calculate_cumulative_distances(path)
    total_length = cumulative_distances[-1]
    checkpoint_distance = total_length / (num_checkpoints - 1)

    checkpoints = [path[0]]  # Start with the first point
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
    """Save hyperparameters to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(params, f)


def load_hyperparameters(filename="best_hyperparameters.json"):
    """Load hyperparameters from a JSON file."""
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as f:
            return json.load(f)
    return None
