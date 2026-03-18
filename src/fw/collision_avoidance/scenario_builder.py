import numpy as np
from fw.collision_avoidance.collision_entity import CollisionEntity

import math
from typing import Optional

EPSILON = 1e-9


class ScenarioBuilder:
    def __init__(self, env) -> None:
        self.env = env
        self.scenario_types = ["static", "overtaking", "head-on"]
        self.scenario_probs = [1, 0, 0]

    def cache_checkpoint_segments(self) -> None:
        """
        Cache checkpoint segment geometry for the current episode geometry.
        Rebuild whenever self.checkpoints changes.
        """
        env = self.env
        seg_starts = []
        seg_ends = []
        seg_lengths = []
        seg_tangents = []
        seg_normals = []
        seg_cp_end_indices = []

        for i in range(1, len(env.checkpoints)):
            pa = np.asarray(env.checkpoints[i - 1]["pos"], dtype=np.float32)
            pb = np.asarray(env.checkpoints[i]["pos"], dtype=np.float32)

            seg = pb - pa
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= EPSILON:
                continue

            tangent = (seg / seg_len).astype(np.float32)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

            seg_starts.append(pa)
            seg_ends.append(pb)
            seg_lengths.append(seg_len)
            seg_tangents.append(tangent)
            seg_normals.append(normal)
            seg_cp_end_indices.append(i)  # segment is checkpoints[i-1] -> checkpoints[i]

        env._seg_starts = np.asarray(seg_starts, dtype=np.float32)
        env._seg_ends = np.asarray(seg_ends, dtype=np.float32)
        env._seg_lengths = np.asarray(seg_lengths, dtype=np.float32)
        env._seg_tangents = np.asarray(seg_tangents, dtype=np.float32)
        env._seg_normals = np.asarray(seg_normals, dtype=np.float32)
        env._seg_cp_end_indices = np.asarray(seg_cp_end_indices, dtype=np.int32)

        # Useful reusable subsets
        n = len(env._seg_lengths)
        self._all_seg_indices = np.arange(n, dtype=np.int32)

        # Match your static/overtaking convention of skipping the first few and last few
        if n > 5:
            env._eligible_dynamic_seg_indices = np.arange(2, n - 2, dtype=np.int32)
        else:
            env._eligible_dynamic_seg_indices = env._all_seg_indices.copy()


    def remaining_distance_to_goal(self, pos: np.ndarray, next_checkpoint_idx: int) -> float:
        """
        Remaining path distance from a position to the goal, assuming the vessel
        is currently heading toward checkpoints[next_checkpoint_idx].
        """
        env = self.env
        if next_checkpoint_idx >= len(env.checkpoints):
            return 0.0

        pos = np.asarray(pos, dtype=np.float32)

        rem = float(np.linalg.norm(
            np.asarray(env.checkpoints[next_checkpoint_idx]["pos"], dtype=np.float32) - pos
        ))

        for i in range(next_checkpoint_idx + 1, len(env.checkpoints)):
            pa = np.asarray(env.checkpoints[i - 1]["pos"], dtype=np.float32)
            pb = np.asarray(env.checkpoints[i]["pos"], dtype=np.float32)
            rem += float(np.linalg.norm(pb - pa))

        return rem

    def sample_collision_entity_scenario(self) -> None:
        """Sample the current episode's collision entities."""
        env = self.env
        if not getattr(env, "enable_obstacles", False):
            env.collision_entities = []
            env._sync_collision_entity_views()
            return

        env.obstacle_count = env._rng.choice(env.obstacle_population, p=env.obstacle_probs)
        chosen_scenario = env._rng.choice(self.scenario_types, p=self.scenario_probs)

        env.collision_entities = self.compute_collision_entity_positions(chosen_scenario, env.obstacle_count) or []
        env._sync_collision_entity_views()

    def compute_collision_entity_positions(self, episode_type: Optional[str] = None, n: int = 1):
        entities = []
        match episode_type:
            case "static":
                entities.extend(self.compute_static_collision_entities())
            case "overtaking":
                for i in range(n):
                    entities.extend(self.compute_overtaking_collision_entities())
            case "head-on":
                for i in range(n):
                    entities.extend(self.compute_headon_collision_entities())
            case "crossing":
                raise NotImplementedError("Crossing scenario is not implemented yet.")
            case _:
                raise ValueError(f"Unknown episode_type '{episode_type}'")

        return entities

    def compute_static_collision_entities(self):
        env = self.env
        if not hasattr(env, "checkpoints") or len(env.checkpoints) < 2:
            return None

        collision_entities = []
        max_lat = float(getattr(env, "static_ship_lateral_max", 0.0))
        segment_margin_frac = float(getattr(env, "static_ship_segment_margin_frac", 0.0))

        for _ in range(env.obstacle_count):
            eligible_indices = list(range(3, len(env.checkpoints) - 2))
            if not eligible_indices:
                break

            seg_lengths = np.array([
                np.linalg.norm(
                    np.asarray(env.checkpoints[i]["pos"], dtype=np.float32)
                    - np.asarray(env.checkpoints[i - 1]["pos"], dtype=np.float32)
                )
                for i in eligible_indices
            ], dtype=np.float32)
            seg_lengths = np.maximum(seg_lengths, EPSILON)
            probs = seg_lengths / seg_lengths.sum()

            random_seg = int(eligible_indices[int(env._rng.choice(len(eligible_indices), p=probs))])
            pa = np.asarray(env.checkpoints[random_seg - 1]["pos"], dtype=np.float32)
            pb = np.asarray(env.checkpoints[random_seg]["pos"], dtype=np.float32)
            seg = (pb - pa).astype(np.float32)
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= EPSILON:
                continue

            tangent = seg / seg_len
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            ship_radius = float(env._rng.uniform(env.static_ship_radius_min, env.static_ship_radius_max))
            pos = (0.5 * (pa + pb)).astype(np.float32)
            candidate_entity = CollisionEntity(
                velocity=np.zeros(2, dtype=np.float32),
                radius=ship_radius,
                x=float(pos[0]),
                y=float(pos[1]),
                entity_type="static",
            )

            t_min = max(0.0, segment_margin_frac)
            t_max = min(1.0, 1.0 - segment_margin_frac)
            if t_min > t_max:
                t_min, t_max = 0.0, 1.0

            for _ in range(30):
                t = float(env._rng.uniform(t_min, t_max))
                base = pa + tangent * (t * seg_len)
                lat = float(np.clip(env._rng.normal(0.0, max_lat / 3.0), -max_lat, max_lat)) if max_lat > 0.0 else 0.0
                cand = base + normal * lat
                candidate_entity.set_position(float(cand[0]), float(cand[1]))
                if not candidate_entity.is_fully_inside_polygon(env.polygon_shape, own_radius=env.own_ship_radius):
                    continue
                too_close = any(
                    candidate_entity.overlaps_with(existing, own_radius=env.own_ship_radius)
                    for existing in collision_entities
                )
                if too_close:
                    continue
                break

            collision_entities.append(candidate_entity)

        return collision_entities

    def compute_overtaking_collision_entities(self):
        env = self.env
        seg_indices = getattr(env, "_eligible_dynamic_seg_indices", np.array([], dtype=np.int32))
        if len(seg_indices) == 0:
            return []

        seg_lengths_arr = env._seg_lengths[seg_indices]
        total_route_length = float(np.sum(seg_lengths_arr))
        os_time_to_goal = total_route_length / max(env.os_nominal_speed, EPSILON)
        ts_speed = float(env._rng.uniform(env.overtaking_ts_v_min, env.overtaking_ts_v_max))
        ship_radius = float(env._rng.uniform(env.static_ship_radius_min, env.static_ship_radius_max))

        segment_margin_frac = float(getattr(env, "static_ship_segment_margin_frac", 0.0))
        t_min_global = max(0.0, segment_margin_frac)
        t_max_global = min(1.0, 1.0 - segment_margin_frac)
        if t_min_global > t_max_global:
            t_min_global, t_max_global = 0.0, 1.0

        candidate_segments = []
        suffix_after = 0.0
        for local_rev_idx in range(len(seg_indices) - 1, -1, -1):
            seg_len = float(seg_lengths_arr[local_rev_idx])
            required_remaining_len = os_time_to_goal * ts_speed
            t_upper_time = 1.0 - ((required_remaining_len - suffix_after) / max(seg_len, EPSILON))
            t_lo = t_min_global
            t_hi = min(t_max_global, t_upper_time)
            if t_hi > t_lo:
                feasible_len = (t_hi - t_lo) * seg_len
                candidate_segments.append({
                    "seg_idx": int(seg_indices[local_rev_idx]),
                    "t_lo": float(t_lo),
                    "t_hi": float(t_hi),
                    "weight": float(max(feasible_len, EPSILON)),
                })
            suffix_after += seg_len

        if not candidate_segments:
            return []

        weights = np.asarray([c["weight"] for c in candidate_segments], dtype=np.float32)
        weights = weights / weights.sum()
        choice = candidate_segments[int(env._rng.choice(len(candidate_segments), p=weights))]

        seg_idx = int(choice["seg_idx"])
        t_spawn = float(env._rng.uniform(choice["t_lo"], choice["t_hi"]))
        pa = env._seg_starts[seg_idx]
        tangent = env._seg_tangents[seg_idx]
        normal = env._seg_normals[seg_idx]
        seg_len = float(env._seg_lengths[seg_idx])
        base = pa + tangent * (t_spawn * seg_len)
        spawn_pos = base + normal * 0.0
        ts_heading = float(math.atan2(tangent[1], tangent[0]))

        entity = CollisionEntity(
            radius=ship_radius,
            x=float(spawn_pos[0]),
            y=float(spawn_pos[1]),
            entity_type="overtaking",
            heading=ts_heading,
            speed=float(ts_speed),
            is_dynamic=True,
        )
        entity.path_speed = float(ts_speed)
        entity.path_segment_index = int(seg_idx)
        entity.next_checkpoint_index = int(env._seg_cp_end_indices[seg_idx])
        entity.path_direction = 1
        entity.path_offset = 0.0
        entity.route_mode = "follow_checkpoints"
        entity.route_finished = False

        if not entity.is_fully_inside_polygon(env.polygon_shape, own_radius=env.own_ship_radius):
            return []
        if any(entity.overlaps_with(existing, own_radius=env.own_ship_radius) for existing in
               (env.collision_entities or [])):
            return []
        return [entity]

    def compute_headon_collision_entities(self):
        env = self.env
        seg_indices = getattr(env, "_eligible_dynamic_seg_indices", np.array([], dtype=np.int32))
        if len(seg_indices) == 0:
            return []

        seg_lengths = np.maximum(env._seg_lengths[seg_indices], EPSILON)
        probs = seg_lengths / seg_lengths.sum()
        seg_idx = int(seg_indices[int(env._rng.choice(len(seg_indices), p=probs))])

        segment_margin_frac = float(getattr(env, "static_ship_segment_margin_frac", 0.0))
        t_min = max(0.0, segment_margin_frac)
        t_max = min(1.0, 1.0 - segment_margin_frac)
        if t_min > t_max:
            t_min, t_max = 0.0, 1.0
        t_spawn = float(env._rng.uniform(t_min, t_max))

        ts_speed = float(env._rng.uniform(env.headon_ts_v_min, env.headon_ts_v_max))
        ship_radius = float(env._rng.uniform(env.static_ship_radius_min, env.static_ship_radius_max))
        tangent = env._seg_tangents[seg_idx]
        reverse_tangent = -tangent
        pa = env._seg_starts[seg_idx]
        seg_len = float(env._seg_lengths[seg_idx])
        spawn_pos = pa + tangent * (t_spawn * seg_len)
        ts_heading = float(math.atan2(reverse_tangent[1], reverse_tangent[0]))

        entity = CollisionEntity(
            radius=ship_radius,
            x=float(spawn_pos[0]),
            y=float(spawn_pos[1]),
            entity_type="head-on",
            heading=ts_heading,
            speed=float(ts_speed),
            is_dynamic=True,
        )
        entity.path_speed = float(ts_speed)
        entity.path_segment_index = int(seg_idx)
        entity.next_checkpoint_index = int(env._seg_cp_end_indices[seg_idx] - 1)
        entity.path_direction = -1
        entity.path_offset = 0.0
        entity.route_mode = "follow_checkpoints"
        entity.route_finished = False
        entity.dodge_trigger_range = float(env.polar_max_range)
        entity.dodge_lateral_speed = 1.0
        entity.lateral_speed = 0.0

        if not entity.is_fully_inside_polygon(env.polygon_shape, own_radius=env.own_ship_radius):
            return []
        if any(entity.overlaps_with(existing, own_radius=env.own_ship_radius) for existing in
               (env.collision_entities or [])):
            return []
        return [entity]

    def check_collision_entity_collision(self, ship_index: int = 0) -> bool:
        env = self.env
        entities = getattr(env, "collision_entities", []) or []
        return any(
            entity.collides_with_point(env.ship_pos[ship_index], own_radius=env.own_ship_radius)
            for entity in entities
        )