import math
import numpy as np

from shapely.geometry import Point, Polygon

EPSILON = 1e-9

class CollisionEntity:
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
        return np.array([self.x, self.y], dtype=np.float32)

    def set_position(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def update_position(self, x: float, y: float) -> None:
        self.set_position(x, y)

    def set_heading(self, heading: float) -> None:
        self.heading = float(heading)

    def get_heading(self) -> float:
        return float(self.heading)

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)

    def get_speed(self) -> float:
        return float(self.speed)

    def set_velocity(self, velocity: np.ndarray) -> None:
        vel = np.asarray(velocity, dtype=np.float32)
        vx = float(vel[0])
        vy = float(vel[1])
        self.speed = float(np.linalg.norm(vel))
        self.heading = float(math.atan2(vy, vx)) if self.speed > 1e-9 else self.heading

    def get_velocity(self) -> np.ndarray:
        return np.array(
            [
                self.speed * math.cos(self.heading),
                self.speed * math.sin(self.heading),
            ],
            dtype=np.float32,
        )

    def set_lateral_speed(self, lateral_speed: float) -> None:
        self.lateral_speed = float(lateral_speed)

    def get_lateral_speed(self) -> float:
        return float(self.lateral_speed)

    def step(self, dt: float) -> None:
        forward_x = math.cos(self.heading)
        forward_y = math.sin(self.heading)

        # Starboard = 90 deg clockwise from heading
        starboard_x = math.sin(self.heading)
        starboard_y = -math.cos(self.heading)

        self.x += (self.speed * forward_x + self.lateral_speed * starboard_x) * dt
        self.y += (self.speed * forward_y + self.lateral_speed * starboard_y) * dt

    def distance_to_point(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=np.float32)
        return float(np.linalg.norm(self.get_position() - point))

    def clearance_to_point(self, point: np.ndarray, own_radius: float) -> float:
        return self.distance_to_point(point) - (self.radius + float(own_radius))

    def collides_with_point(self, point: np.ndarray, own_radius: float) -> bool:
        return self.clearance_to_point(point, own_radius) <= 0.0

    def collision_check(self, point: np.ndarray, own_radius: float) -> bool:
        return self.collides_with_point(point, own_radius)

    def is_fully_inside_polygon(self, polygon: Polygon, own_radius: float) -> bool:
        inflated = float(self.radius) + float(own_radius)
        return bool(polygon.contains(Point(self.x, self.y).buffer(inflated)))

    def overlaps_with(self, other: "CollisionEntity", own_radius: float, margin: float = 0.0) -> bool:
        distance = float(np.linalg.norm(self.get_position() - other.get_position()))
        min_distance = float(self.radius) + float(other.radius) + 2.0 * float(own_radius) + float(margin)
        return distance < min_distance

    def passed_by_os(self, os_pos, os_heading):
        # Check if TS is now abaft OS beam (OS has passed it)
        dx = self.x - float(os_pos[0])
        dy = self.y - float(os_pos[1])
        rel_bearing = math.atan2(dy, dx) - os_heading
        rel_bearing = (rel_bearing + math.pi) % (2.0 * math.pi) - math.pi
        return abs(rel_bearing) > math.pi / 2.0

    def passing_signal(self, os_pos, scenario: str) -> float:
        """Signed scalar in TS frame whose sign-flip marks an OS-passes-TS event.

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

