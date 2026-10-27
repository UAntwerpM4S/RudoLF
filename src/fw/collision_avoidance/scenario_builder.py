import math
import numpy as np

from typing import Optional
from fw.collision_avoidance.collision_entity import CollisionEntity

EPSILON = 1e-9


class ScenarioBuilder:
    """
    Episode-level scenario builder.

    Reads scenario_types, scenario_probs, and encounter_range from the
    environment so that subclass environments can configure them.

    Subclass this builder (e.g. AdaptiveScenarioBuilder) to implement
    step-level encounter spawning via on_step().
    """

    def __init__(self, env) -> None:
        """
        Initializes the scenario builder.

        Args:
            env: Environment that owns the builder.
        """

        self.step_counter: int = 0
        self.env = env
        self.chosen_scenario: Optional[str] = None

        # Focused-encounter training state (per episode)
        # Follows a sequence of PF steps -> Encounter steps -> PF step
        self._encounters_this_episode: int = 0 # Number of encounters that have already spawned in the current episode.
        self._pathfollowing_encounter_active: bool = False # dedicated PF steps are active instead of encounter
        self._pathfollowing_encounter_counter: int = 0 # counts steps during active PF steps
        self._post_encounter_active: bool = False # True if post-encounter cooldown is active
        self._post_encounter_counter: int = 0 # counts number of steps in the post encounter
        self._truncate_requested: bool = False # True when the post-encounter is done, and the episode is truncated

        # Single-encounter eval mode state (per episode)
        self._single_encounter_spawned: bool = False  # dormant TS has been placed
        self._single_encounter_done: bool = False     # TS has despawned; no more encounters
        self._single_encounter_ts_seen: bool = False  # OS came within encounter_range at least once

    # ----------------
    # Lifecycle hooks
    # ----------------
    def on_reset(self) -> None:
        """
        Called at episode reset.

        Resets per-episode builder state regardless of mode (so the
        recovery counter doesn't leak across episodes after early
        terminations such as collisions). At episode level, also builds
        the full episode scenario; at step level, entities are spawned
        later by `on_step()`.
        """

        env = self.env

        self.step_counter = 0
        self.chosen_scenario = None

        # Reset focused-encounter training state
        self._encounters_this_episode = 0
        self._pathfollowing_encounter_active = False
        self._pathfollowing_encounter_counter = 0
        self._post_encounter_active = False
        self._post_encounter_counter = 0
        self._truncate_requested = False

        # Reset single-encounter eval-mode state
        self._single_encounter_spawned = False
        self._single_encounter_done = False
        self._single_encounter_ts_seen = False

        # Step-level mode: encounters are spawned by on_step()
        if getattr(env, "scenario_obstacle_level", None) == "step":
            env.collision_entities = []
            env._sync_collision_entity_views()
            # Single-encounter mode places the dormant TS at episode reset
            # so the OS can approach it from path-following distance.
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
        # Tag every spawned entity with its ground-truth scenario (used by
        # the classifier's ground_truth_extended_label property and by metrics).
        for entity in env.collision_entities:
            if getattr(entity, "spawn_scenario_label", None) is None:
                entity.spawn_scenario_label = chosen_scenario
        env._sync_collision_entity_views()
        # Lock the classifier to the builder's choice so misclassification
        # (e.g. a head-on being geometrically labeled as crossing-right)
        # cannot poison reward shaping during training.
        self._push_scenario_to_classifier(chosen_scenario)

    def _push_scenario_to_classifier(self, scenario_label: Optional[str]) -> None:
        """
        Propagate the builder's chosen scenario to the classifier.

        Only applies when the env has ground_truth_classification=True.
        When False, the classifier runs in pure geometric mode at both
        episode and step levels (so its misclassification rate can be
        measured against the builder's ground truth).
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
        True when focused-encounter training is active for this env.
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
        Encounter spawn probability per step for the current mode.
        """

        env = self.env
        if getattr(env, "eval", False):
            return float(getattr(env, "eval_encounter_prob", 0.004))
        return float(getattr(env, "train_encounter_prob", 1.0))

    def on_step(self, dt: float) -> None:
        """
        Called each environment step for step-level encounter spawning.

        Behavior depends on training vs eval mode:

        * **Eval mode**: original behavior; after a cooldown, a TS may
          spawn each step with `eval_encounter_prob`.  Multiple
          encounters per episode are allowed; episode runs to `max_steps`.
        * **Focused-encounter training mode**: a single encounter per
          episode.  After `train_encounter_cooldown` steps of recovery,
          a TS spawns with probability `train_encounter_prob` (typically
          1.0).  Once the TS leaves `forget_range`, a post-encounter
          countdown of `post_encounter_steps` runs and then `_truncate_requested`
          is set so the env step loop can clean-truncate the episode.
        """

        env = self.env
        if not getattr(env, "enable_obstacles", False):
            env.collision_entities = []
            env._sync_collision_entity_views()
            return

        # Single-encounter eval mode: bypasses the normal cooldown / spawn /
        # focused-training plumbing. The dormant TS was placed in on_reset();
        # here we only handle despawn (OS leaves forget_range) and ensure no
        # further encounters spawn for the rest of the episode.
        if getattr(env, "single_encounter_per_episode", False):
            self._single_encounter_on_step()
            return

        is_focused = self._is_training_focused()

        # ----------------------------------------------------------
        # 1. Post-encounter countdown (focused training only)
        #    Runs after the (only) encounter's TS has left forget range.
        # ----------------------------------------------------------
        if self._post_encounter_active:
            self._post_encounter_counter += 1
            if self._post_encounter_counter >= int(getattr(env, "post_encounter_steps", 150)):
                self._truncate_requested = True
            return

        # ----------------------------------------------------------
        # 1b. Path-following 'encounter' active (focused training only)
        #     The episode sampled "path-following" as its encounter type,
        #     so no TS is spawned; we just run PF for
        #     `pathfollowing_encounter_steps` and then transition into
        #     the post-encounter countdown so the episode terminates with
        #     the same 150-step PF tail as a normal TS encounter.
        # ----------------------------------------------------------
        if self._pathfollowing_encounter_active:
            self._pathfollowing_encounter_counter += 1
            if self._pathfollowing_encounter_counter >= int(
                getattr(env, "pathfollowing_encounter_steps", 150)
            ):
                self._pathfollowing_encounter_active = False
                self._post_encounter_active = True
                self._post_encounter_counter = 0
            return

        # ----------------------------------------------------------
        # 2. Pre-encounter cooldown / recovery
        # ----------------------------------------------------------
        cooldown = self._active_cooldown()
        if self.step_counter != cooldown:
            self.step_counter += 1
            return

        # ----------------------------------------------------------
        # 3. Encounter spawn (skip if focused training already had one)
        # ----------------------------------------------------------
        spawn_blocked = is_focused and self._encounters_this_episode >= 1
        if not spawn_blocked:
            encounter_prob = self._active_encounter_prob()
            encounter = env._rng.random() < encounter_prob

            if encounter and not env.collision_entities:
                chosen_scenario = self.sample_collision_entity_scenario()
                if chosen_scenario == "path-following":
                    # Focused training: treat the sampled "path-following"
                    # as the encounter: start the PF-only encounter timer
                    # and lock the classifier to PF for the duration so
                    # reward shaping / observations stay in pure-PF mode.
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
                            # Honor ground_truth_classification for step-level too.
                            self._push_scenario_to_classifier(chosen_scenario)
                            self._encounters_this_episode += 1

        # ----------------------------------------------------------
        # 4. Clean up TS that has left forget range OR has finished its
        #    route. The route_finished check covers the case where a
        #    path-following TS (e.g. an overtaking TS that the OS never
        #    catches up to) reaches the end of the trajectory
        # ----------------------------------------------------------
        if env.collision_entities and len(env.collision_entities) == 1:
            if getattr(env, "ts_idx", None) is not None and env.ts_idx < len(env.collision_entities):
                ts = env.collision_entities[env.ts_idx]
                dist = env.collision_entities[0].distance_to_point(env.ship_pos)
                route_finished = bool(getattr(ts, "route_finished", False))
                if dist > env.forget_range or route_finished:
                    env.collision_entities = []
                    env.ts_idx = None
                    env._sync_collision_entity_views()
                    self.step_counter = 0
                    # Reset the classifier so it doesn't hold a stale tracked_ts
                    # index into the now-empty collision_entities list.
                    if hasattr(env, "scenario_classifier"):
                        env.scenario_classifier.reset()

                    # Focused training: encounter is over, start the
                    # post-encounter countdown to clean truncation.
                    if is_focused:
                        self._post_encounter_active = True
                        self._post_encounter_counter = 0

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

        # Match static/overtaking convention of skipping the first few and last few
        n = len(env._seg_lengths)
        if n > 5:
            env._eligible_dynamic_seg_indices = np.arange(2, n - 5, dtype=np.int32)
        else:
            env._eligible_dynamic_seg_indices = np.arange(n, dtype=np.int32)


    def sample_collision_entity_scenario(self):
        """
        Sample the current episode's collision entities.
        """

        env = self.env
        env.obstacle_count = env._rng.choice(env.obstacle_population, p=env.obstacle_probs)
        chosen_scenario = env._rng.choice(env.scenario_types, p=env.scenario_probs)
        return chosen_scenario

    def _find_segment_at_encounter_range(self) -> Optional[int]:
        """
        Find the cached-segment index whose start lies just beyond
        `encounter_range` ahead of the OS along the path.

        Walks forward from the OS's current position (projected onto its
        current segment), accumulating remaining path length.  Returns the
        first segment index (into the cached `_seg_*` arrays) where the
        accumulated distance exceeds `encounter_range` AND the segment
        is one of the eligible dynamic segments, or `None` if no such
        segment exists (OS is too close to the start/end of the route).

        Note on the "remaining" accounting: the first iteration must use
        the *remaining* distance on the OS's current segment, not the
        full segment length. Without this correction, when the OS is
        deep inside its current segment (which is the common case after
        the recovery wait), the function over-counts by up to one full
        `CHECKPOINTS_DISTANCE` and returns a segment that is well
        inside `encounter_range`, leaving the agent no time to react.
        """

        env = self.env
        accumulated = 0.0
        encounter_range = float(env.encounter_range)

        eligible = getattr(env, "_eligible_dynamic_seg_indices", None)
        if eligible is None or len(eligible) == 0:
            return None
        eligible_set = set(int(i) for i in eligible.tolist())

        # Walk from the OS's current target checkpoint onward
        start_cp = max(1, env.checkpoint_index)
        first = True
        ship_pos = np.asarray(env.ship_pos, dtype=np.float32)
        for cp_idx in range(start_cp, len(env.checkpoints)):
            # Map checkpoint-pair index to cached segment index
            # _seg_cp_end_indices[seg_i] == cp_idx means segment seg_i
            # connects checkpoints[cp_idx-1] -> checkpoints[cp_idx].
            matches = np.where(env._seg_cp_end_indices == cp_idx)[0]
            if len(matches) == 0:
                continue
            seg_i = int(matches[0])
            seg_len = float(env._seg_lengths[seg_i])

            if first:
                # Project the OS onto the current segment and use the
                # remaining distance to its end checkpoint, not the
                # full segment length.
                pa = env._seg_starts[seg_i]
                tangent = env._seg_tangents[seg_i]
                travelled = float(np.dot(ship_pos - pa, tangent))
                travelled = float(np.clip(travelled, 0.0, seg_len))
                accumulated += seg_len - travelled
                first = False
            else:
                accumulated += seg_len

            if accumulated >= encounter_range:
                # Honor the start/end skip applied at episode level: if
                # the segment that lands "just past encounter_range"
                # falls in the excluded first-few or last-few region,
                # skip this encounter entirely rather than spawning on
                # an off-limits segment.
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
        Build collision entities for *episode_type*.

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
        # Lateral jitter (eval-mode realism): no mean shift since overtaker shares OS lane.
        # Sign convention: entity travels along path, so entity-starboard = -normal.
        # Positive path_offset => entity's starboard => OS's starboard.
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
        normal = env._seg_normals[seg_idx]
        reverse_tangent = -tangent
        pa = env._seg_starts[seg_idx]
        seg_len = float(env._seg_lengths[seg_idx])
        # Lateral lane (eval-mode realism): mean shift to OS-port ("left lane")
        # plus uniform jitter. Sign: head-on travels against path, so entity-starboard
        # equals +normal (= path-port = OS-port). Positive path_offset => OS-port lane.
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
        """
        Checks collision between OS and TS.
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
        Resolve the target segment index into the cached _seg_* arrays.
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
        Place one dormant TS on the user-specified segment.

        Reuses the existing per-scenario builders to get a geometry-correct
        placement (segment, lateral offset, heading) and then zeros out the
        TS speed so it sits still until the env step loop activates it.
        Crossings already spawn dormant via the bearing-aware machinery, so
        we only attach the generic dormant marker for overtaking / head-on.
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
            # Crossings have their own activation flow; leave their speed
            # (which the crossing builder already set to 0.0) alone.
            if scenario in ("crossing-right", "crossing-left"):
                continue
            # Generic dormant marker: store the would-be motion speed, then
            # park the entity. Step-loop activation flips speed back on once
            # the OS is within encounter_range.
            orig_speed = float(getattr(entity, "path_speed", entity.get_speed()))
            entity._dormant_original_speed = orig_speed
            entity.dormant_activation_range = float(env.encounter_range)
            entity.dormant_pending_activation = True
            entity.set_speed(0.0)
            if hasattr(entity, "path_speed"):
                entity.path_speed = 0.0

        env.collision_entities = entities
        env._sync_collision_entity_views()
        # Cache spawn position so the live renderer can center the camera
        # on it (paper-comparison mode: identical view across runs).
        env._single_encounter_ts_spawn_pos = np.asarray(
            entities[0].get_position(), dtype=np.float32
        ).copy()
        self._push_scenario_to_classifier(scenario)
        self._single_encounter_spawned = True
        self._encounters_this_episode += 1

    def _single_encounter_on_step(self) -> None:
        """
        Per-step bookkeeping for single-encounter eval mode.

        * Despawn the dormant/active TS once the OS leaves forget_range.
        * After despawn, no further entities are spawned for the rest of
          the episode (`_single_encounter_done` blocks any respawn path).
        """

        env = self.env

        # If despawn already happened, hold the empty-entity state.
        if self._single_encounter_done:
            return

        # If reset/spawn somehow missed (e.g. target_seg_idx unresolved),
        # don't keep retrying: block further spawns to honor the spec.
        if not self._single_encounter_spawned:
            return

        entities = getattr(env, "collision_entities", []) or []
        if not entities:
            self._single_encounter_done = True
            return

        ts = entities[0]
        dist = ts.distance_to_point(env.ship_pos)

        # Despawn is only valid AFTER the OS has approached the TS at least
        # once; otherwise the dormant TS placed far ahead on the route
        # would despawn on step 0 (dist > forget_range from the start).
        # Use "ever been within encounter_range" as the activation gate,
        # which is the same trigger that wakes the TS up.
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
