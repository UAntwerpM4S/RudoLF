import random

import numpy as np
from fw.collision_avoidance.collision_entity import CollisionEntity

import math
from typing import Optional

EPSILON = 1e-9


class ScenarioBuilder:
    """Episode-level scenario builder.

    Reads scenario_types, scenario_probs, and encounter_range from the
    environment so that subclass environments can configure them.

    Subclass this builder (e.g. AdaptiveScenarioBuilder) to implement
    step-level encounter spawning via on_step().
    """

    def __init__(self, env) -> None:
        self.env = env

    # ------------------------------------------------------------------
    # Lifecycle hooks (called by PySimColavBaseEnv)
    # ------------------------------------------------------------------
    def on_reset(self) -> None:
        """Called at episode reset. Builds the full episode scenario."""
        env = self.env
        if not getattr(env, "enable_obstacles", False):
            env.collision_entities = []
            env._sync_collision_entity_views()
            return
        chosen_scenario = self.sample_collision_entity_scenario()
        env.collision_entities = self.compute_collision_entity_positions(chosen_scenario, env.obstacle_count) or []
        env._sync_collision_entity_views()

    def on_step(self, dt: float) -> None:
        """Called each environment step for step-level encounter spawning.

        With probability ``encounter_prob`` and when no encounter is already
        active, a single TS is spawned on the segment just outside the OS
        encounter range.  Once the TS leaves forget range it is removed so
        the next encounter can trigger.
        """
        env = self.env
        if not getattr(env, "enable_obstacles", False):
            env.collision_entities = []
            env._sync_collision_entity_views()
            return

        encounter_prob = 0.4
        encounter = env._rng.random() < encounter_prob

        # Spawn a new encounter if none is active
        if encounter and not env.collision_entities:
            chosen_scenario = self.sample_collision_entity_scenario()
            if chosen_scenario != "path-following":
                seg_idx = self._find_segment_at_encounter_range()
                if seg_idx is not None:
                    entities = self.compute_collision_entity_positions(
                        chosen_scenario, n=1, target_seg_idx=seg_idx,
                    )
                    env.collision_entities = entities or []
                    env._sync_collision_entity_views()

        # Clean up TS that has left forget range
        if env.collision_entities and len(env.collision_entities) == 1:
            if getattr(env, "ts_idx", None) is not None:
                ts = env.collision_entities[env.ts_idx]
                dist = env.collision_entities[0].distance_to_point(env.ship_pos)
                if dist > env.forget_range:
                    env.collision_entities = []
                    env.ts_idx = None
                    env._sync_collision_entity_views()
                    # Reset the classifier so it doesn't hold a stale tracked_ts
                    # index into the now-empty collision_entities list. Without
                    # this, _calculate_colav_reward() reads tracked_ts=0 and
                    # tries collision_entities[0] → IndexError.
                    if hasattr(env, "scenario_classifier"):
                        env.scenario_classifier.reset()

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
        env.obstacle_count = env._rng.choice(env.obstacle_population, p=env.obstacle_probs)
        chosen_scenario = env._rng.choice(env.scenario_types, p=env.scenario_probs)
        #print(f"Chosen scenario: {chosen_scenario} with {env.obstacle_count} obstacle(s)")
        return chosen_scenario

    def _find_segment_at_encounter_range(self) -> Optional[int]:
        """Find the cached-segment index whose start lies just beyond
        ``encounter_range`` ahead of the OS along the path.

        Walks forward from the OS's current checkpoint, accumulating
        path length.  Returns the first segment index (into the cached
        ``_seg_*`` arrays) where the accumulated distance exceeds
        ``encounter_range``, or ``None`` if no such segment exists
        (OS is too close to the end of the route).
        """
        env = self.env
        accumulated = 0.0
        encounter_range = float(env.encounter_range)

        # Walk from the OS's current target checkpoint onward
        start_cp = max(1, env.checkpoint_index)
        for cp_idx in range(start_cp, len(env.checkpoints)):
            # Map checkpoint-pair index to cached segment index
            # _seg_cp_end_indices[seg_i] == cp_idx means segment seg_i
            # connects checkpoints[cp_idx-1] -> checkpoints[cp_idx].
            matches = np.where(env._seg_cp_end_indices == cp_idx)[0]
            if len(matches) == 0:
                continue
            seg_i = int(matches[0])
            seg_len = float(env._seg_lengths[seg_i])
            accumulated += seg_len
            if accumulated >= encounter_range:
                return seg_i

        return None

    def compute_collision_entity_positions(
        self,
        scenario_type: Optional[str] = None,
        n: int = 1,
        target_seg_idx: Optional[int] = None,
    ):
        """Build collision entities for *episode_type*.

        Args:
            scenario_type: Scenario name (e.g. "head-on").
            n: How many entities to create (episode-level).  For
               step-level spawning this is always 1.
            target_seg_idx: If given, force placement on this cached
               segment index (used by step-level spawning).
        """
        entities = []
        match scenario_type:
            case "path-following":
                pass
            case "static":
                entities.extend(self.compute_static_collision_entities(target_seg_idx=target_seg_idx))
            case "overtaking":
                for i in range(n):
                    entities.extend(self.compute_overtaking_collision_entities(target_seg_idx=target_seg_idx))
            case "head-on":
                for i in range(n):
                    entities.extend(self.compute_headon_collision_entities(target_seg_idx=target_seg_idx))
            case "crossing-right":
                for i in range(n):
                    entities.extend(self.compute_crossing_right_collision_entities(target_seg_idx=target_seg_idx))
            case "crossing-left":
                for i in range(n):
                    entities.extend(self.compute_crossing_left_collision_entities(target_seg_idx=target_seg_idx))
            case _:
                raise ValueError(f"Unknown episode_type '{scenario_type}'")

        return entities

    def compute_static_collision_entities(self, target_seg_idx: Optional[int] = None):
        """
        Randomly placed obstacle_count static obstacles along the path, laterally randomized (uniform).
        When target_seg_idx is given (step-level), places exactly one obstacle on that segment.
        """
        env = self.env
        if not hasattr(env, "checkpoints") or len(env.checkpoints) < 2:
            return None

        collision_entities = []
        max_lat = float(getattr(env, "static_ship_lateral_max", 0.0))
        segment_margin_frac = float(getattr(env, "static_ship_segment_margin_frac", 0.0))

        n_to_place = 1 if target_seg_idx is not None else env.obstacle_count
        for _ in range(n_to_place):
            if target_seg_idx is not None:
                # Step-level: use the checkpoint index that corresponds to the
                # cached segment's end checkpoint.
                random_seg = int(env._seg_cp_end_indices[target_seg_idx])
            else:
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

            # Compute signed CTE of this obstacle relative to its path segment
            obs_pos = candidate_entity.get_position()
            to_obs = obs_pos - pa
            signed_cte = float(np.dot(to_obs, normal))
            candidate_entity.path_cte = signed_cte
            if signed_cte > EPSILON:
                candidate_entity.path_side = "port"
            elif signed_cte < -EPSILON:
                candidate_entity.path_side = "starboard"
            else:
                candidate_entity.path_side = "on_path"

            collision_entities.append(candidate_entity)

        return collision_entities

    def compute_overtaking_collision_entities(self, target_seg_idx: Optional[int] = None):
        """
        Places an overtaking TS along the path.

        Episode-level (target_seg_idx is None): selects a segment that
        guarantees the OS will catch the TS before it reaches the goal.

        Step-level (target_seg_idx given): places the TS on the specified
        segment just outside encounter range.
        """
        env = self.env

        segment_margin_frac = float(getattr(env, "static_ship_segment_margin_frac", 0.0))
        t_min_global = max(0.0, segment_margin_frac)
        t_max_global = min(1.0, 1.0 - segment_margin_frac)
        if t_min_global > t_max_global:
            t_min_global, t_max_global = 0.0, 1.0

        ts_speed = float(env._rng.uniform(env.overtaking_ts_v_min, env.overtaking_ts_v_max))
        ship_radius = float(env._rng.uniform(env.static_ship_radius_min, env.static_ship_radius_max))

        if target_seg_idx is not None:
            # Step-level: place on the given segment
            seg_idx = target_seg_idx
            t_spawn = float(env._rng.uniform(t_min_global, t_max_global))
        else:
            # Episode-level: original timing-constrained placement
            seg_indices = getattr(env, "_eligible_dynamic_seg_indices", np.array([], dtype=np.int32))
            if len(seg_indices) == 0:
                return []

            seg_lengths_arr = env._seg_lengths[seg_indices]
            total_route_length = float(np.sum(seg_lengths_arr))
            os_time_to_goal = total_route_length / max(env.os_nominal_speed, EPSILON)

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

    def compute_headon_collision_entities(self, target_seg_idx: Optional[int] = None):
        """
        Places a head-on TS that travels backward along the path.

        Episode-level (target_seg_idx is None): picks a random eligible segment.
        Step-level (target_seg_idx given): places on the specified segment.
        """
        env = self.env

        segment_margin_frac = float(getattr(env, "static_ship_segment_margin_frac", 0.0))
        t_min = max(0.0, segment_margin_frac)
        t_max = min(1.0, 1.0 - segment_margin_frac)
        if t_min > t_max:
            t_min, t_max = 0.0, 1.0

        if target_seg_idx is not None:
            seg_idx = target_seg_idx
        else:
            seg_indices = getattr(env, "_eligible_dynamic_seg_indices", np.array([], dtype=np.int32))
            if len(seg_indices) == 0:
                return []
            seg_lengths = np.maximum(env._seg_lengths[seg_indices], EPSILON)
            probs = seg_lengths / seg_lengths.sum()
            seg_idx = int(seg_indices[int(env._rng.choice(len(seg_indices), p=probs))])

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
        entity.dodge_trigger_range = float(env.encounter_range)
        entity.dodge_lateral_speed = 1.0
        entity.lateral_speed = 0.0
        entity.dodged = False

        if not entity.is_fully_inside_polygon(env.polygon_shape, own_radius=env.own_ship_radius):
            return []
        if any(entity.overlaps_with(existing, own_radius=env.own_ship_radius) for existing in
               (env.collision_entities or [])):
            return []
        return [entity]

    def _compute_crossing_collision_entities(self, side: str, target_seg_idx: Optional[int] = None):
        """
        Shared logic for crossing-right and crossing-left scenarios.

        Places a dormant TS on one side of the OS path, heading perpendicular
        across it. The TS is activated (starts moving) once the OS enters
        activation range during stepping.

        Episode-level (target_seg_idx is None): tries all eligible segments.
        Step-level (target_seg_idx given): only tries the specified segment.

        side: "starboard" for crossing-right (TS on OS starboard, OS give-way)
              "port"      for crossing-left  (TS on OS port, OS stand-on)
        """
        env = self.env

        if target_seg_idx is not None:
            seg_indices = np.array([target_seg_idx], dtype=np.int32)
        else:
            seg_indices = getattr(env, "_eligible_dynamic_seg_indices", np.array([], dtype=np.int32))
            if len(seg_indices) == 0:
                return []

        segment_margin_frac = float(getattr(env, "static_ship_segment_margin_frac", 0.0))
        t_min = max(0.0, segment_margin_frac)
        t_max = min(1.0, 1.0 - segment_margin_frac)
        if t_min > t_max:
            t_min, t_max = 0.0, 1.0

        ship_radius = float(env._rng.uniform(env.static_ship_radius_min, env.static_ship_radius_max))
        activation_range = float(env.encounter_range) + float(getattr(env, "crossing_activation_margin", 50.0))

        # Target bearing range at classification (radians).
        # d_ts = activation_range * sin(bearing) gives the lateral offset
        # that produces the desired relative bearing when OS-to-TS ≈ activation_range.
        bearing_min = math.radians(float(getattr(env, "crossing_bearing_min", 25.0)))
        bearing_max = math.radians(float(getattr(env, "crossing_bearing_max", 50.0)))
        d_ts_min_abs = float(getattr(env, "crossing_d_ts_min", 50.0))

        target_bearing = float(env._rng.uniform(bearing_min, bearing_max))
        d_ts_ideal = activation_range * math.sin(target_bearing)

        entity_type = "crossing-right" if side == "starboard" else "crossing-left"

        # Uniform shuffle of eligible segments (avoids bias toward longer/wider
        # segments that would make narrow-canal placements rare).
        seg_order = env._rng.choice(len(seg_indices), size=len(seg_indices), replace=False)

        for seg_local_idx in seg_order:
            seg_idx = int(seg_indices[int(seg_local_idx)])
            tangent = env._seg_tangents[seg_idx]
            normal = env._seg_normals[seg_idx]  # points to port side of the path
            pa = env._seg_starts[seg_idx]
            seg_len = float(env._seg_lengths[seg_idx])

            # Determine spawn offset direction and TS heading
            # normal points to port; -normal points to starboard
            if side == "starboard":
                offset_dir = -normal       # spawn on starboard side
                ts_heading_vec = normal    # head towards port
            else:
                offset_dir = normal        # spawn on port side
                ts_heading_vec = -normal   # head towards starboard

            ts_heading = float(math.atan2(ts_heading_vec[1], ts_heading_vec[0]))

            # Try a few random crossing points along this segment.
            # For each crossing point, try the ideal d_ts first, then shrink
            # until the TS fits inside the canal polygon.
            for _ in range(10):
                t_spawn = float(env._rng.uniform(t_min, t_max))
                crossing_point = pa + tangent * (t_spawn * seg_len)

                placed = False
                d_ts = d_ts_ideal
                while d_ts >= d_ts_min_abs:
                    candidate_pos = crossing_point + offset_dir * d_ts

                    candidate = CollisionEntity(
                        radius=ship_radius,
                        x=float(candidate_pos[0]),
                        y=float(candidate_pos[1]),
                        entity_type=entity_type,
                        heading=ts_heading,
                        speed=0.0,
                        is_dynamic=True,
                    )
                    if (candidate.is_fully_inside_polygon(env.polygon_shape, own_radius=env.own_ship_radius)
                            and not any(candidate.overlaps_with(existing, own_radius=env.own_ship_radius)
                                        for existing in (env.collision_entities or []))):
                        placed = True
                        break
                    d_ts *= 0.8  # shrink by 20% and retry

                if not placed:
                    continue

                entity = CollisionEntity(
                    radius=ship_radius,
                    x=float(candidate_pos[0]),
                    y=float(candidate_pos[1]),
                    entity_type=entity_type,
                    heading=ts_heading,
                    speed=0.0,  # dormant until activated
                    is_dynamic=True,
                )
                entity.crossing_point = crossing_point.copy()
                entity.crossing_d_ts = float(d_ts)
                entity.crossing_path_tangent = tangent.copy()
                entity.route_mode = "straight_line"
                entity.route_finished = False
                entity.crossing_activated = False
                entity.crossing_activation_range = activation_range

                return [entity]

        # No valid placement found across all segments
        return []

    def compute_crossing_right_collision_entities(self, target_seg_idx: Optional[int] = None):
        """
        Crossing-right: TS on OS starboard side, heading across the path
        towards port. OS is the give-way vessel (must pass behind TS stern).
        """
        return self._compute_crossing_collision_entities(side="starboard", target_seg_idx=target_seg_idx)

    def compute_crossing_left_collision_entities(self, target_seg_idx: Optional[int] = None):
        """
        Crossing-left: TS on OS port side, heading across the path
        towards starboard. OS is the stand-on vessel (maintain course/speed).
        """
        return self._compute_crossing_collision_entities(side="port", target_seg_idx=target_seg_idx)

    def check_collision_entity_collision(self) -> bool:
        """Checks collision between OS and TS"""
        env = self.env
        entities = getattr(env, "collision_entities", []) or []
        return any(
            entity.collides_with_point(env.ship_pos, own_radius=env.own_ship_radius)
            for entity in entities
        )