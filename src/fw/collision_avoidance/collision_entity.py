import math
import numpy as np

from shapely.geometry import Point, Polygon

EPSILON = 1e-9

class CollisionEntity:
    """
    Represents a collision entity in the simulation.

    A collision entity may be static or dynamic and stores its position,
    velocity, heading, and collision geometry. Dynamic entities support
    simple kinematic motion and additional metadata used by encounter
    scenarios.
    """

    def __init__(
        self,
        velocity=None,
        radius=0.0,
        x=0.0,
        y=0.0,
        entity_type="static",
        heading=None,
        speed=None,
        is_dynamic=False,
    ):
        """
        Initializes a collision entity.

        Args:
            velocity: Initial velocity vector.
            radius: Collision radius.
            x: Initial x-coordinate.
            y: Initial y-coordinate.
            entity_type: Entity category.
            heading: Initial heading in radians.
            speed: Initial speed.
            is_dynamic: Whether the entity moves during simulation.
        """

        self.radius = float(radius)
        self.x = float(x)
        self.y = float(y)
        self.entity_type = entity_type
        self.is_dynamic = bool(is_dynamic)
        self.lateral_speed = 0.0

        if velocity is not None:
            vel = np.asarray(velocity, dtype=np.float32)
            vx = float(vel[0])
            vy = float(vel[1])
            inferred_speed = float(np.linalg.norm(vel))
            inferred_heading = float(math.atan2(vy, vx)) if inferred_speed > 1e-9 else 0.0

            self.speed = inferred_speed if speed is None else float(speed)
            self.heading = inferred_heading if heading is None else float(heading)
        else:
            self.speed = 0.0 if speed is None else float(speed)
            self.heading = 0.0 if heading is None else float(heading)

    def get_position(self) -> np.ndarray:
        """
        Returns the entity position.

        Returns:
            Position as a 2D NumPy array.
        """

        return np.array([self.x, self.y], dtype=np.float32)

    def set_position(self, x: float, y: float) -> None:
        """
        Sets the entity position.

        Args:
            x: X-coordinate.
            y: Y-coordinate.
        """

        self.x = float(x)
        self.y = float(y)

    def update_position(self, x: float, y: float) -> None:
        """
        Updates the entity position.

        Args:
            x: X-coordinate.
            y: Y-coordinate.
        """

        self.set_position(x, y)

    def set_heading(self, heading: float) -> None:
        """
        Sets the entity heading.

        Args:
            heading: Heading in radians.
        """

        self.heading = float(heading)

    def get_heading(self) -> float:
        """
        Returns the entity heading.

        Returns:
            Heading in radians.
        """

        return float(self.heading)

    def set_speed(self, speed: float) -> None:
        """
        Sets the forward speed.

        Args:
            speed: Forward speed.
        """

        self.speed = float(speed)

    def get_speed(self) -> float:
        """
        Returns the forward speed.

        Returns:
            Forward speed.
        """

        return float(self.speed)

    def set_velocity(self, velocity: np.ndarray) -> None:
        """
        Sets the velocity vector.

        Updates both the speed and heading to match the supplied velocity.

        Args:
            velocity: Velocity vector.
        """

        vel = np.asarray(velocity, dtype=np.float32)
        vx = float(vel[0])
        vy = float(vel[1])
        self.speed = float(np.linalg.norm(vel))
        self.heading = float(math.atan2(vy, vx)) if self.speed > 1e-9 else self.heading

    def get_velocity(self) -> np.ndarray:
        """
        Returns the velocity vector.

        Returns:
            Velocity as a 2D NumPy array.
        """

        return np.array(
            [
                self.speed * math.cos(self.heading),
                self.speed * math.sin(self.heading),
            ],
            dtype=np.float32,
        )

    def set_lateral_speed(self, lateral_speed: float) -> None:
        """
        Sets the lateral speed.

        Args:
            lateral_speed: Lateral speed relative to the entity heading.
        """

        self.lateral_speed = float(lateral_speed)

    def get_lateral_speed(self) -> float:
        """
        Returns the lateral speed.

        Returns:
            Lateral speed.
        """

        return float(self.lateral_speed)

    def step(self, dt: float) -> None:
        """
        Advances the entity state by one simulation step.

        Moves the entity according to its forward and lateral speeds.

        Args:
            dt: Simulation time step.
        """

        forward_x = math.cos(self.heading)
        forward_y = math.sin(self.heading)

        # Starboard = 90 deg clockwise from heading
        starboard_x = math.sin(self.heading)
        starboard_y = -math.cos(self.heading)

        self.x += (self.speed * forward_x + self.lateral_speed * starboard_x) * dt
        self.y += (self.speed * forward_y + self.lateral_speed * starboard_y) * dt

    def distance_to_point(self, point: np.ndarray) -> float:
        """
        Computes the distance to a point.

        Args:
            point: Point position.

        Returns:
            Euclidean distance.
        """

        point = np.asarray(point, dtype=np.float32)
        return float(np.linalg.norm(self.get_position() - point))

    def clearance_to_point(self, point: np.ndarray, own_radius: float) -> float:
        """
        Computes the clearance to a point.

        The clearance equals the center-to-center distance minus the combined
        collision radii.

        Args:
            point: Point position.
            own_radius: Radius of the other object.

        Returns:
            Signed clearance distance.
        """

        return self.distance_to_point(point) - (self.radius + float(own_radius))

    def collides_with_point(self, point: np.ndarray, own_radius: float) -> bool:
        """
        Checks whether the entity collides with a point.

        Args:
            point: Point position.
            own_radius: Radius of the other object.

        Returns:
            `True` if a collision exists.
        """

        return self.clearance_to_point(point, own_radius) <= 0.0

    def collision_check(self, point: np.ndarray, own_radius: float) -> bool:
        """
        Alias for :meth:`collides_with_point`.

        Args:
            point: Point position.
            own_radius: Radius of the other object.

        Returns:
            `True` if a collision exists.
        """

        return self.collides_with_point(point, own_radius)

    def is_fully_inside_polygon(self, polygon: Polygon, own_radius: float) -> bool:
        """
        Checks whether the entity fits inside a polygon.

        The entity is treated as a circle whose radius is enlarged by the own
        ship radius.

        Args:
            polygon: Navigable polygon.
            own_radius: Own-ship radius.

        Returns:
            `True` if the entity is fully contained.
        """

        inflated = float(self.radius) + float(own_radius)
        return bool(polygon.contains(Point(self.x, self.y).buffer(inflated)))

    def overlaps_with(self, other: "CollisionEntity", own_radius: float, margin: float = 0.0) -> bool:
        """
        Checks whether two entities overlap.

        Args:
            other: Other collision entity.
            own_radius: Own-ship radius.
            margin: Additional safety margin.

        Returns:
            `True` if the entities overlap.
        """

        distance = float(np.linalg.norm(self.get_position() - other.get_position()))
        min_distance = float(self.radius) + float(other.radius) + 2.0 * float(own_radius) + float(margin)
        return distance < min_distance

    def passed_by_os(self, os_pos, os_heading):
        """
        Checks whether the own ship has passed the entity.

        The entity is considered passed once it lies abaft the own ship's beam.

        Args:
            os_pos: Own-ship position.
            os_heading: Own-ship heading.

        Returns:
            `True` if the own ship has passed the entity.
        """

        # Check if TS is now abaft OS beam (OS has passed it)
        dx = self.x - float(os_pos[0])
        dy = self.y - float(os_pos[1])
        rel_bearing = math.atan2(dy, dx) - os_heading
        rel_bearing = (rel_bearing + math.pi) % (2.0 * math.pi) - math.pi
        return abs(rel_bearing) > math.pi / 2.0

    def passing_signal(self, os_pos, scenario: str) -> float:
        """
        Signed scalar in TS frame whose sign-flip marks an OS-passes-TS event.

        crossing-right: lateral offset (cross with TS heading); flips when OS
            crosses the TS heading axis (stern/bow crossing).
        head-on / overtaking: longitudinal offset (dot with TS heading); flips
            when OS passes TS abeam (relative bearing TS->OS = +/-90 deg).
        """

        dx = float(os_pos[0]) - self.x
        dy = float(os_pos[1]) - self.y
        if scenario == "crossing-right":
            return math.cos(self.heading) * dy - math.sin(self.heading) * dx
        return math.cos(self.heading) * dx + math.sin(self.heading) * dy

    def ts_side_from_os_perspective(self, os_pos: np.ndarray, os_heading: float) -> str:
        """
        Which side of the OS is this TS on?

        Uses cross product of OS forward vector with the vector from OS to TS.
        Returns "port", "starboard", or "ahead".
        """

        dx = self.x - float(os_pos[0])
        dy = self.y - float(os_pos[1])
        cross_z = math.cos(os_heading) * dy - math.sin(os_heading) * dx

        if cross_z > EPSILON:
            return "port"
        elif cross_z < -EPSILON:
            return "starboard"
        return "ahead"
