import math
import numpy as np

from typing import Optional
from fw.collision_avoidance.collision_entity import CollisionEntity

EPSILON = 1e-9


class ScenarioBuilder:
    """
    Places obstacles and target ships (TS) for COLAV episodes.

    Scenario settings (scenario_types, scenario_probs, encounter_range, ...)
    are read from the owning env. Spawning happens either once per episode
    (on_reset) or during the episode (on_step, scenario_obstacle_level="step").
    """

    def __init__(self, env) -> None:
        """
        Args:
            env: Environment that owns the builder.
        """

        self.step_counter: int = 0
        self.env = env
        self.chosen_scenario: Optional[str] = None

        # Focused-encounter training state (per episode): PF, encounter, PF tail
        self._encounters_this_episode: int = 0  # encounters spawned so far
        self._pathfollowing_encounter_active: bool = False  # PF-only "encounter" running
        self._pathfollowing_encounter_counter: int = 0
        self._post_encounter_active: bool = False  # post-encounter PF tail running
        self._post_encounter_counter: int = 0
        self._truncate_requested: bool = False  # tail done; env should truncate

        # Single-encounter eval mode state (per episode)
        self._single_encounter_spawned: bool = False  # dormant TS has been placed
        self._single_encounter_done: bool = False     # TS has despawned; no more encounters
        self._single_encounter_ts_seen: bool = False  # OS came within encounter_range at least once

    # ----------------
    # Lifecycle hooks
    # ----------------
    def on_reset(self) -> None:
        """
        Reset per-episode state (in every mode, so counters don't leak after
        early terminations). At episode level also build the episode's
        entities; at step level they are spawned later by on_step().
        """

        env = self.env

        self.step_counter = 0
        self.chosen_scenario = None

        self._encounters_this_episode = 0
        self._pathfollowing_encounter_active = False
        self._pathfollowing_encounter_counter = 0
        self._post_encounter_active = False
        self._post_encounter_counter = 0
        self._truncate_requested = False

        self._single_encounter_spawned = False
        self._single_encounter_done = False
        self._single_encounter_ts_seen = False

        # Step-level mode: encounters are spawned by on_step()
        if getattr(env, "scenario_obstacle_level", None) == "step":
            env.collision_entities = []
            env._sync_collision_entity_views()
            # Single-encounter mode places its dormant TS now, so the OS
            # approaches it from path-following distance.
            if getattr(env, "single_encounter_per_episode", False):
                self._spawn_single_dormant_encounter()
            return

        if not getattr(env, "enable_obstacles", False):
            env.collision_entities = []
            env._sync_collision_entity_views()
            self.chosen_scenario = "path-following"
            self._push_scenario_to_classifier("path-following")
            return

        chosen_scenario = self.sample_collision_entity_scenario()
        self.chosen_scenario = chosen_scenario
        env.collision_entities = self.compute_collision_entity_positions(chosen_scenario, env.obstacle_count) or []
        # Ground-truth scenario tag, used by the classifier and by metrics.
        for entity in env.collision_entities:
            if getattr(entity, "spawn_scenario_label", None) is None:
                entity.spawn_scenario_label = chosen_scenario
        env._sync_collision_entity_views()
        # Lock the classifier to this choice so geometric misclassification
        # (e.g. head-on seen as crossing-right) cannot corrupt reward shaping.
        self._push_scenario_to_classifier(chosen_scenario)

    def _push_scenario_to_classifier(self, scenario_label: Optional[str]) -> None:
        """
        Lock the classifier to the builder's scenario, if the env has
        ground_truth_classification=True. Otherwise, the classifier stays purely
        geometric, so its error rate can be measured against the ground truth.
        """

        env = self.env
        if not getattr(env, "ground_truth_classification", True):
            return
        clf = getattr(env, "scenario_classifier", None)
        if clf is None:
            return
        clf.set_episode_scenario(scenario_label)

    def _is_training_focused(self) -> bool:
        """
        True when focused-encounter training is active (training only).
        """

        env = self.env
        return (
            not getattr(env, "eval", False)
            and bool(getattr(env, "focused_encounter_mode", False))
        )

    def _active_cooldown(self) -> int:
        """
        Pre-encounter cooldown (steps) for the current train/eval mode.
        """

        env = self.env
        if getattr(env, "eval", False):
            return int(getattr(env, "eval_encounter_cooldown", 500))
        return int(getattr(env, "train_encounter_cooldown", 150))

    def _active_encounter_prob(self) -> float:
        """
        Per-step encounter spawn probability for the current train/eval mode.
        """

        env = self.env
        if getattr(env, "eval", False):
            return float(getattr(env, "eval_encounter_prob", 0.004))
        return float(getattr(env, "train_encounter_prob", 1.0))

    def on_step(self, dt: float) -> None:
        """
        Step-level encounter spawning, called every env step.

        - Eval mode: after the cooldown a TS may spawn each step with
          eval_encounter_prob; several encounters per episode, runs to max_steps.
        - Focused-encounter training: one encounter per episode. After
          train_encounter_cooldown steps a TS spawns with train_encounter_prob.
          Once it despawns, a post_encounter_steps countdown runs and then
          _truncate_requested is set so the env can truncate cleanly.
        - Single-encounter eval mode: handled by _single_encounter_on_step().
        """

        env = self.env
        if not getattr(env, "enable_obstacles", False):
            env.collision_entities = []
            env._sync_collision_entity_views()
            return

        # Dormant TS was placed in on_reset(); only despawn is handled here.
        if getattr(env, "single_encounter_per_episode", False):
            self._single_encounter_on_step()
            return

        is_focused = self._is_training_focused()

        # 1. Post-encounter countdown (focused training only).
        if self._post_encounter_active:
            self._post_encounter_counter += 1
            if self._post_encounter_counter >= int(getattr(env, "post_encounter_steps", 150)):
                self._truncate_requested = True
            return

        # 1b. Sampled "path-following" encounter (focused training only): no TS,
        #     run PF for pathfollowing_encounter_steps, then the same
        #     post-encounter tail as a real encounter.
        if self._pathfollowing_encounter_active:
            self._pathfollowing_encounter_counter += 1
            if self._pathfollowing_encounter_counter >= int(
                getattr(env, "pathfollowing_encounter_steps", 150)
            ):
                self._pathfollowing_encounter_active = False
                self._post_encounter_active = True
                self._post_encounter_counter = 0
            return

        # 2. Pre-encounter cooldown / recovery.
        cooldown = self._active_cooldown()
        if self.step_counter != cooldown:
            self.step_counter += 1
            return

        # 3. Encounter spawn (at most one in focused training).
        spawn_blocked = is_focused and self._encounters_this_episode >= 1
        if not spawn_blocked:
            encounter_prob = self._active_encounter_prob()
            encounter = env._rng.random() < encounter_prob

            if encounter and not env.collision_entities:
                chosen_scenario = self.sample_collision_entity_scenario()
                if chosen_scenario == "path-following":
                    # Focused training: count PF as the encounter and lock
                    # the classifier to PF for its duration.
                    if is_focused:
                        self._pathfollowing_encounter_active = True
                        self._pathfollowing_encounter_counter = 0
                        self._push_scenario_to_classifier("path-following")
                        self._encounters_this_episode += 1
                else:
                    seg_idx = self._find_segment_at_encounter_range()
                    if seg_idx is not None:
                        entities = self.compute_collision_entity_positions(
                            chosen_scenario, n=1, target_seg_idx=seg_idx,
                        )
                        if entities:
                            for entity in entities:
                                if getattr(entity, "spawn_scenario_label", None) is None:
                                    entity.spawn_scenario_label = chosen_scenario
                        env.collision_entities = entities or []
                        env._sync_collision_entity_views()
                        if entities:
                            self._push_scenario_to_classifier(chosen_scenario)
                            self._encounters_this_episode += 1

        # 4. Despawn the TS once its route is finished (e.g. an overtaking TS
        #    the OS never catches) or once it leaves forget_range after having
        #    been within encounter_range (_despawn_engaged, the range at which
        #    the classifier starts tracking it). A TS whose closest approach
        #    stays between the two ranges is not removed for distance.
        #    Only dynamic TS, and only in envs with TRACKS_TARGET_SHIP.
        if (
            env.collision_entities
            and len(env.collision_entities) == 1
            and getattr(env, "TRACKS_TARGET_SHIP", False)
            and bool(getattr(env.collision_entities[0], "is_dynamic", False))
        ):
            ts = env.collision_entities[0]
            dist = ts.distance_to_point(env.ship_pos)
            if dist <= env.encounter_range:
                ts._despawn_engaged = True
            route_finished = bool(getattr(ts, "route_finished", False))
            left_range = bool(getattr(ts, "_despawn_engaged", False)) and dist > env.forget_range
            if left_range or route_finished:
                env.collision_entities = []
                env.ts_idx = None
                env._sync_collision_entity_views()
                self.step_counter = 0
                # Drop the classifier's now-stale tracked TS index.
                if hasattr(env, "scenario_classifier"):
                    env.scenario_classifier.reset()

                if is_focused:
                    self._post_encounter_active = True
                    self._post_encounter_counter = 0

    def cache_checkpoint_segments(self) -> None:
        """
        Cache per-segment geometry (env._seg_*) of env.checkpoints. Rebuild
        whenever the checkpoints change.
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

        # Dynamic TS never spawn on the first 2 or last 5 segments.
        n = len(env._seg_lengths)
        if n > 5:
            env._eligible_dynamic_seg_indices = np.arange(2, n - 5, dtype=np.int32)
        else:
            env._eligible_dynamic_seg_indices = np.arange(n, dtype=np.int32)


    def sample_collision_entity_scenario(self):
        """
        Sample env.obstacle_count and return a sampled scenario type.
        """

        env = self.env
        env.obstacle_count = env._rng.choice(env.obstacle_population, p=env.obstacle_probs)
        chosen_scenario = env._rng.choice(env.scenario_types, p=env.scenario_probs)
        return chosen_scenario

    def _find_segment_at_encounter_range(self) -> Optional[int]:
        """
        Return the cached segment index where the path distance ahead of the OS
        first reaches encounter_range, or None if that segment is not an
        eligible dynamic segment (or the route ends first).

        The distance starts with the remaining length of the OS's current
        segment (OS projected onto it), not its full length; otherwise the TS
        could spawn well inside encounter_range.
        """

        env = self.env
        accumulated = 0.0
        encounter_range = float(env.encounter_range)

        eligible = getattr(env, "_eligible_dynamic_seg_indices", None)
        if eligible is None or len(eligible) == 0:
            return None
        eligible_set = set(int(i) for i in eligible.tolist())

        start_cp = max(1, env.checkpoint_index)
        first = True
        ship_pos = np.asarray(env.ship_pos, dtype=np.float32)
        for cp_idx in range(start_cp, len(env.checkpoints)):
            # Segment seg_i connects checkpoints[cp_idx - 1] and checkpoints[cp_idx].
            matches = np.where(env._seg_cp_end_indices == cp_idx)[0]
            if len(matches) == 0:
                continue
            seg_i = int(matches[0])
            seg_len = float(env._seg_lengths[seg_i])

            if first:
                pa = env._seg_starts[seg_i]
                tangent = env._seg_tangents[seg_i]
                travelled = float(np.dot(ship_pos - pa, tangent))
                travelled = float(np.clip(travelled, 0.0, seg_len))
                accumulated += seg_len - travelled
                first = False
            else:
                accumulated += seg_len

            if accumulated >= encounter_range:
                # Skip the encounter rather than spawn on an excluded segment.
                if seg_i in eligible_set:
                    return seg_i
                return None

        return None

    def compute_collision_entity_positions(
        self,
        scenario_type: Optional[str] = None,
        n: int = 1,
        target_seg_idx: Optional[int] = None,
    ):
        """
        Build the collision entities for a scenario.

        Args:
            scenario_type: Scenario name (e.g. "head-on").
            n: Number of dynamic TS to create (1 for step-level spawning).
                Static scenarios use env.obstacle_count instead.
            target_seg_idx: If given, place on this cached segment index
                (step-level spawning).

        Raises:
            ValueError: For an unknown scenario type.
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
        Place env.obstacle_count static obstacles on length-weighted random path
        segments, with a clipped Gaussian lateral offset (sigma = max_lat / 3).
        With target_seg_idx (step level), place exactly one on that segment.
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
                random_seg = int(env._seg_cp_end_indices[target_seg_idx])
            else:
                eligible_indices = list(range(5, len(env.checkpoints) - 3))
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

    def compute_overtaking_collision_entities(self, target_seg_idx: Optional[int] = None):
        """
        Place an overtaking TS moving along the path.

        Episode level (target_seg_idx None): timing-constrained spawn point so
        the OS catches the TS before it reaches the goal.
        Step level: place it on the given segment.
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
            seg_idx = target_seg_idx
            t_spawn = float(env._rng.uniform(t_min_global, t_max_global))
        else:
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
        # Lateral jitter only (overtaker shares the OS lane). The TS travels
        # along the path, so positive path_offset is to its (and the OS's)
        # starboard, i.e. -normal.
        jitter = float(getattr(env, "overtaking_lateral_jitter", 0.0))
        lat = float(env._rng.uniform(-jitter, jitter)) if jitter > 0.0 else 0.0
        spawn_pos = base + (-normal) * lat
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
        entity.path_offset = lat
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
        Place a head-on TS traveling backward along the path, on a
        length-weighted random eligible segment or on target_seg_idx.
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
        normal = env._seg_normals[seg_idx]
        reverse_tangent = -tangent
        pa = env._seg_starts[seg_idx]
        seg_len = float(env._seg_lengths[seg_idx])
        # Lane offset: headon_lateral_mean plus uniform jitter. The TS travels
        # against the path, so positive path_offset (+normal) is the TS's
        # starboard and the OS's port side.
        mean_shift = float(getattr(env, "headon_lateral_mean", 0.0))
        jitter = float(getattr(env, "headon_lateral_jitter", 0.0))
        lat = mean_shift + (float(env._rng.uniform(-jitter, jitter)) if jitter > 0.0 else 0.0)
        spawn_pos = pa + tangent * (t_spawn * seg_len) + normal * lat
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
        entity.path_offset = lat
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
        Place a dormant crossing TS beside the path, heading perpendicular to it.
        It starts moving once the OS is within crossing_activation_range.

        Tries all eligible segments in random order (episode level) or only
        target_seg_idx (step level). Returns [] if nothing fits.

        Args:
            side: "starboard" for crossing-right (OS give-way) or "port" for
                crossing-left (OS stand-on).
            target_seg_idx: Optional cached segment index.
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

        # Lateral offset d_ts = activation_range * sin(bearing) gives the target
        # relative bearing when the OS-TS distance is about activation_range.
        bearing_min = math.radians(float(getattr(env, "crossing_bearing_min", 25.0)))
        bearing_max = math.radians(float(getattr(env, "crossing_bearing_max", 50.0)))
        d_ts_min_abs = float(getattr(env, "crossing_d_ts_min", 50.0))

        target_bearing = float(env._rng.uniform(bearing_min, bearing_max))
        d_ts_ideal = activation_range * math.sin(target_bearing)

        entity_type = "crossing-right" if side == "starboard" else "crossing-left"

        # Unweighted shuffle, so narrow-canal segments are not underrepresented.
        seg_order = env._rng.choice(len(seg_indices), size=len(seg_indices), replace=False)

        for seg_local_idx in seg_order:
            seg_idx = int(seg_indices[int(seg_local_idx)])
            tangent = env._seg_tangents[seg_idx]
            normal = env._seg_normals[seg_idx]  # points to port side of the path
            pa = env._seg_starts[seg_idx]
            seg_len = float(env._seg_lengths[seg_idx])

            if side == "starboard":
                offset_dir = -normal       # spawn on starboard side
                ts_heading_vec = normal    # head towards port
            else:
                offset_dir = normal        # spawn on port side
                ts_heading_vec = -normal   # head towards starboard

            ts_heading = float(math.atan2(ts_heading_vec[1], ts_heading_vec[0]))

            # Up to 10 crossing points; per point shrink d_ts from the ideal
            # value until the TS fits in the polygon.
            for _ in range(10):
                t_spawn = float(env._rng.uniform(t_min, t_max))
                crossing_point = pa + tangent * (t_spawn * seg_len)

                placed = False
                d_ts = d_ts_ideal
                candidate_pos = None

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

        return []

    def compute_crossing_right_collision_entities(self, target_seg_idx: Optional[int] = None):
        """
        TS on the OS starboard side heading to port; OS gives way (passes astern).
        """

        return self._compute_crossing_collision_entities(side="starboard", target_seg_idx=target_seg_idx)

    def compute_crossing_left_collision_entities(self, target_seg_idx: Optional[int] = None):
        """
        TS on the OS port side heading to starboard; OS stands on.
        """

        return self._compute_crossing_collision_entities(side="port", target_seg_idx=target_seg_idx)

    def check_collision_entity_collision(self) -> bool:
        """
        True if the OS collides with any collision entity.
        """

        env = self.env
        entities = getattr(env, "collision_entities", []) or []
        return any(
            entity.collides_with_point(env.ship_pos, own_radius=env.own_ship_radius)
            for entity in entities
        )

    # ---------------------------------------------------------------
    # Single-encounter eval mode
    # ---------------------------------------------------------------
    def _resolve_single_encounter_seg_idx(self) -> Optional[int]:
        """
        Return single_encounter_target_segment_index if valid for _seg_*, else None.
        """

        env = self.env
        target = getattr(env, "single_encounter_target_segment_index", None)
        if target is None:
            return None
        target = int(target)
        n_segments = int(len(getattr(env, "_seg_lengths", [])))
        if n_segments <= 0 or not (0 <= target < n_segments):
            return None
        return target

    def _spawn_single_dormant_encounter(self) -> None:
        """
        Place one dormant TS on the configured segment using the regular
        scenario builders. Overtaking/head-on TS get speed 0 plus a dormant
        marker, so the env step loop starts them within encounter_range;
        crossing TS are already dormant.
        """

        env = self.env
        seg_idx = self._resolve_single_encounter_seg_idx()
        if seg_idx is None:
            return

        scenario = getattr(env, "single_encounter_scenario", None) or "overtaking"

        entities = self.compute_collision_entity_positions(
            scenario, n=1, target_seg_idx=seg_idx,
        ) or []
        if not entities:
            return

        for entity in entities:
            if getattr(entity, "spawn_scenario_label", None) is None:
                entity.spawn_scenario_label = scenario
            if scenario in ("crossing-right", "crossing-left"):
                continue
            orig_speed = float(getattr(entity, "path_speed", entity.get_speed()))
            entity._dormant_original_speed = orig_speed
            entity.dormant_activation_range = float(env.encounter_range)
            entity.dormant_pending_activation = True
            entity.set_speed(0.0)
            if hasattr(entity, "path_speed"):
                entity.path_speed = 0.0

        env.collision_entities = entities
        env._sync_collision_entity_views()
        # Lets the live renderer center the camera here (same view across runs).
        env._single_encounter_ts_spawn_pos = np.asarray(
            entities[0].get_position(), dtype=np.float32
        ).copy()
        self._push_scenario_to_classifier(scenario)
        self._single_encounter_spawned = True
        self._encounters_this_episode += 1

    def _single_encounter_on_step(self) -> None:
        """
        Single-encounter eval mode: despawn the TS once its route is finished
        or the OS leaves forget_range (after having been within
        encounter_range). Nothing respawns afterward.
        """

        env = self.env

        if self._single_encounter_done:
            return

        # Spawn failed at reset (e.g. unresolved segment): never retry.
        if not self._single_encounter_spawned:
            return

        entities = getattr(env, "collision_entities", []) or []
        if not entities:
            self._single_encounter_done = True
            return

        ts = entities[0]
        dist = ts.distance_to_point(env.ship_pos)

        # Distance despawn only after the OS has been within encounter_range;
        # the dormant TS starts beyond forget_range.

        if dist <= float(env.encounter_range):
            self._single_encounter_ts_seen = True

        already_seen = getattr(self, "_single_encounter_ts_seen", False)
        route_finished = bool(getattr(ts, "route_finished", False))
        if route_finished or (already_seen and dist > env.forget_range):
            env.collision_entities = []
            env.ts_idx = None
            env._sync_collision_entity_views()
            if hasattr(env, "scenario_classifier"):
                env.scenario_classifier.reset()
            self._single_encounter_done = True
