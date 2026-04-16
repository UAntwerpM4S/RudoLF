"""Base environment for collision avoidance scenarios.

Provides scenario construction, entity advancement along checkpoints,
and collision detection infrastructure. Subclasses add observation-specific
logic (e.g. polar rangefinder, replanner) and reward shaping.

Class hierarchy:
    PySimEnv (path following)
      -> PySimColavBaseEnv (scenario + entity management)
          -> PySimColavPolarEnv (rangefinder approach)
          -> PySimColavPlannerEnv (replanner approach)
"""
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gymnasium as gym
from fw.collision_avoidance.scenario_builder import ScenarioBuilder
from fw.collision_avoidance.scenario_classifier import ScenarioClassifier

import math

from fw.environments.py_sim_env import PySimEnv

try:
    matplotlib.use('TkAgg')
except ImportError:
    print("Warning: 'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use('Agg')

from typing import Optional, Tuple, Dict
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
import numpy as np


# -----------------------
# Module-level constants
# -----------------------
CHECKPOINTS_DISTANCE = 350
MIN_GRID_POS = -11700
MAX_GRID_POS = 14500

SUCCESS_REWARD = 50.0
CHECKPOINT_AREA_SIZE = 5.0
TARGET_AREA_SIZE = 7.0

MAX_FIG_WIDTH = 1200
MAX_FIG_HEIGHT = 900
DPI = 100

PERPENDICULAR_LINE_LENGTH = 50.0
PERPENDICULAR_LINE_PROXIMITY = 2.0
HEADING_CHANGE_THRESHOLD = np.pi / 2.0
CROSS_TRACK_TERMINATION_MULTIPLIER = 2.0
EPSILON = 1e-9


class PySimColavBaseEnv(PySimEnv):
    """Base collision avoidance environment with scenario construction
    and entity management.

    This environment extends PySimEnv with:
    - Configurable scenario generation (obstacle placement, TS encounters)
    - Entity advancement along checkpoints (overtaking, head-on, crossing)
    - Collision detection (bank + entity)
    - Scenario classification and COLREG tracking
    - CRI (Collision Risk Index) computation

    Subclasses must override:
        _calculate_colav_reward() -> (float, bool)

    Subclasses may override:
        _on_entities_stepped()  — post-entity-step hook
        _initialize_observation_space()  — extend observation space
        _get_obs()  — extend observation vector
        _on_episode_reset()  — additional per-episode reset logic
    """

    def __init__(
            self,
            render_mode: Optional[str] = None,
            time_step: float = 1,
            max_steps: int = 1500,
            verbose: Optional[bool] = None,
            ship_pos: Optional[np.ndarray] = None,
            target_pos: Optional[np.ndarray] = None,
            dynamics_model: Optional[Model] = Model.FOSSEN,
            ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
            wind: bool = False,
            current: bool = False,
            numerical_damping: bool = False,
            # --- COLAV configuration ---
            enable_obstacles: bool = True,
            enable_variable_ship_size: bool = False,
            encounter_range: float = 500.0,
            eval: bool = True,
            # Scenario type configuration
            scenario_types: Optional[list] = None,
            scenario_probs: Optional[list] = None,
            scenario_obstacle_level: Optional[str] = None,
            # Obstacle population configuration
            obstacle_population: Optional[list] = None,
            obstacle_weights: Optional[list] = None,
    ) -> None:

        # Generic detection range that drives scenario activation, dodge triggers,
        # and classification. Subclasses set this to their sensor range
        # (e.g. PySimColavPolarEnv sets encounter_range = polar_max_range).
        self.encounter_range = float(encounter_range)

        # Scenario measurement parameters
        self.enable_variable_ship_size = bool(enable_variable_ship_size)
        self.static_ship_radius = 20
        self.static_ship_radius_min = 20.0
        self.static_ship_radius_max = 40.0 if self.enable_variable_ship_size else self.static_ship_radius_min
        self.static_ship_radii = []
        self.static_ship_segment_margin_frac = 0.1
        self.static_cte_half_value = 15.0
        self.enable_obstacles = bool(enable_obstacles)
        self.collision_entities = []
        self.obstacle_count = 0
        self.obstacle_pos = None
        self.static_ship_lateral_max = 100.0

        # Obstacle population (how many TS per episode)
        self.obstacle_population = obstacle_population if obstacle_population is not None else [0, 1, 2, 3]
        self.obstacle_weights = obstacle_weights if obstacle_weights is not None else [0, 100, 0, 0]
        self.obstacle_probs = np.asarray(self.obstacle_weights, dtype=np.float32)
        self.obstacle_probs /= self.obstacle_probs.sum()

        # Scenario type configuration
        self.scenario_types = scenario_types if scenario_types is not None else [
            "path-following", "static", "overtaking", "head-on", "crossing-right", "crossing-left"
        ]
        self.scenario_probs = scenario_probs if scenario_probs is not None else [
            0.0, 0.0, 0.33, 0.33, 0.34, 0.0
        ]
        self.scenario_obstacle_level = str(scenario_obstacle_level) if scenario_obstacle_level is not None else "episode"

        # OS and TS speed configuration
        self.os_nominal_speed = 2.0
        self.overtaking_ts_v_min = 1.0
        self.overtaking_ts_v_max = self.os_nominal_speed - 0.3
        self.headon_ts_v_min = self.os_nominal_speed + 1.0
        self.headon_ts_v_max = self.os_nominal_speed + 3.0
        self.crossing_ts_v_min = 1.0
        self.crossing_ts_v_max = 5.0
        self.crossing_activation_margin = 0.0
        self.crossing_speed_factor = 1.0
        self.own_ship_radius = 0.0

        # Evaluation / tracking
        self.eval = bool(eval)
        self.min_distance = float("inf")

        # Per-episode minimum distance tracking
        self._episode_min_distances: list[float] = []
        self._current_episode_min_dist: float = float("inf")
        self._current_episode_collided: bool = False

        self._last_collision_info = {}
        self._colreg_violations = 0
        self._colreg_correct = 0
        self.ts_passed = None

        # Passing snapshot: list of dicts with OS/TS positions at the moment of passing
        self._passing_snapshots: list[dict] = []

        # CRI parameters (Zhen et al. 2022 / Zhao et al. 2026)
        self.cri_k1 = 1.0
        self.cri_k2 = 1.0
        self.cri_alpha1 = -1.0
        self.cri_alpha2 = -1.0
        self.cri_D = 50.0
        self.cri_T = 120.0
        self.cri_omega1 = 0.5
        self.cri_omega2 = 0.5

        # Call parent (PySimEnv) init
        super().__init__(
            render_mode=render_mode,
            time_step=time_step,
            max_steps=max_steps,
            verbose=verbose,
            start_pos=ship_pos,
            target_pos=target_pos,
            dynamics_model=dynamics_model,
            ship_model=ship_model,
            wind=wind,
            current=current,
            numerical_damping=numerical_damping,
        )

        # Scenario builder
        self.scenario_builder = ScenarioBuilder(self)
        self.scenario_builder.cache_checkpoint_segments()
        self.forget_range = self.encounter_range + 20

        # Scenario classifier
        self.scenario_classifier = ScenarioClassifier(
            classify_range=self.encounter_range,
            forget_range=self.forget_range,
        )
        self.ts_passed = None

        # Cache base geometry for episode resets
        self._base_initial_ship_pos = self.initial_ship_pos.copy()
        self._base_target_pos = self.target_pos.copy()
        self._base_overall = np.asarray(self.overall, dtype=np.float32).copy()
        self._base_obstacles = np.asarray(self.obstacles, dtype=np.float32).copy()
        self._base_checkpoints = [
            {
                "pos": cp["pos"].copy(),
                "radius": float(cp["radius"]),
                "reward": float(cp["reward"]),
                "perpendicular_line": (
                    cp["perpendicular_line"][0].copy(),
                    cp["perpendicular_line"][1].copy(),
                ),
            }
            for cp in self.checkpoints
        ]

        # Initial scenario sampling
        if self.scenario_obstacle_level == "episode":
            self.scenario_builder.on_reset()
        self._on_entities_stepped()

        # Re-initialize after scenario sampling
        self._initialize_simulation()

        # Render helpers
        self._collision_entity_patches = []
        self._collision_entity_render_key = None

    # -------------------------
    # Utility & Math
    # -------------------------
    def _compute_cri(self, os_pos, os_heading, os_u, os_v, ts) -> float:
        """Compute Collision Risk Index (CRI) for a target ship."""
        ts_pos = ts.get_position()
        ts_vel = ts.get_velocity()

        cos_h = math.cos(os_heading)
        sin_h = math.sin(os_heading)
        os_vx = os_u * cos_h - os_v * sin_h
        os_vy = os_u * sin_h + os_v * cos_h

        dx = float(ts_pos[0] - os_pos[0])
        dy = float(ts_pos[1] - os_pos[1])
        dvx = float(ts_vel[0]) - os_vx
        dvy = float(ts_vel[1]) - os_vy

        vr_sq = dvx * dvx + dvy * dvy
        if vr_sq < 1e-12:
            return 0.0

        tcpa = -(dx * dvx + dy * dvy) / vr_sq
        dcpa_x = dx + dvx * tcpa
        dcpa_y = dy + dvy * tcpa
        dcpa = math.sqrt(dcpa_x * dcpa_x + dcpa_y * dcpa_y)

        cri_dcpa = self.cri_k1 * math.exp(self.cri_alpha1 * dcpa / self.cri_D)

        if tcpa <= 0.0:
            cri_tcpa = 0.0
        else:
            cri_tcpa = self.cri_k2 * math.exp(self.cri_alpha2 * tcpa / self.cri_T)

        return self.cri_omega1 * cri_dcpa + self.cri_omega2 * cri_tcpa

    # -----------------------
    # Dynamics update & step
    # -----------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment timestep with collision entity management."""
        action = np.asarray(action, dtype=np.float32)

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            self.previous_action = self.performed_action.copy()
            self.previous_ship_pos = self.ship_pos.copy()
            self.previous_heading = self.phys_sim.state.heading

            self.phys_sim.step(action, self.enable_smoothing)
            self.ship_pos = np.array(
                [self.phys_sim.state.x, self.phys_sim.state.y], dtype=np.float32
            )
            self.performed_action = self.phys_sim.actuators.current_actuator_state.copy()
            self.step_count += 1

        # Capture collision segment before reward may advance checkpoint
        prev_idx = max(0, self.checkpoint_index - 1)
        self._collision_segment_start = np.asarray(self.checkpoints[prev_idx]['pos'], dtype=np.float32)
        curr_idx = min(self.checkpoint_index, len(self.checkpoints) - 1)
        self._collision_segment_end = np.asarray(self.checkpoints[curr_idx]['pos'], dtype=np.float32)

        # Step collision entities
        self._step_collision_entities()

        # Adaptive scenario builder hook
        if self.scenario_obstacle_level == "step":
            self.scenario_builder.on_step(self.time_step)

        # Track minimum distance to collision entities
        if getattr(self, "collision_entities", None):
            for entity in self.collision_entities:
                dist = entity.distance_to_point(self.ship_pos)
                if dist < self._current_episode_min_dist:
                    self._current_episode_min_dist = dist

        # Post-entity-step hook (subclass: e.g. rebuild polar geometries)
        self._on_entities_stepped()

        # Calculate reward and termination
        reward = 0.0
        terminated = False
        truncated = False

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            reward, terminated = self._calculate_colav_reward()

            # Collision detection (shared across all subclasses)
            collision_entity_collision = False
            if getattr(self, "collision_entities", None):
                collision_entity_collision = self.scenario_builder.check_collision_entity_collision()
                if collision_entity_collision:
                    terminated = True

            bank_collision = not self._ship_in_open_water(self.ship_pos)

            # Build collision info
            self._last_collision_info = {}
            if bank_collision:
                self._last_collision_info = {"collision": True, "collision_type": "bank"}
            if collision_entity_collision:
                segment_vec = self._collision_segment_end - self._collision_segment_start
                ship_vec = self.ship_pos - self._collision_segment_start
                cross_z = float(segment_vec[0] * ship_vec[1] - segment_vec[1] * ship_vec[0])
                collision_side = "port" if cross_z > 0.0 else "starboard"
                cross_track_error = self._distance_from_point_to_line(
                    self.ship_pos, self._collision_segment_start, self._collision_segment_end
                )
                self._last_collision_info = {
                    "collision": True,
                    "collision_type": "ship",
                    "collision_cte": abs(cross_track_error),
                    "collision_side": collision_side,
                }

            # Failure penalty
            success = self.checkpoint_index >= len(self.checkpoints)
            if terminated and not success:
                reward = -250.0

            if self.step_count >= self.max_steps and not terminated:
                reward = -250
                truncated = True

            self.phys_sim.state.terminated = terminated
            self.phys_sim.state.truncated = truncated

        # Track collision for min-distance exclusion
        if self._last_collision_info.get("collision"):
            self._current_episode_collided = True

        info = {
            "rewards": reward,
            "terminations": terminated,
            "truncations": truncated,
            "collision_info": self._last_collision_info,
            "colreg_violations": self._colreg_violations,
            "colreg_correct": self._colreg_correct,
            "min_distance": self._current_episode_min_dist,
        }

        return self._get_obs(), reward, terminated or truncated, False, info

    # --------------------------------
    # Reward (override in subclass)
    # --------------------------------
    def _calculate_colav_reward(self) -> Tuple[float, bool]:
        """Calculate COLAV reward. Subclasses override this with domain-specific shaping.

        Returns:
            (reward, terminated): The shaped reward and whether the episode ends.
            Note: collision detection and failure penalty are handled by step().
        """
        return super()._calculate_reward()

    # -----------------------
    # Hooks for subclasses
    # -----------------------
    def _on_entities_stepped(self) -> None:
        """Called after collision entities have been stepped.
        Override in subclass for sensor geometry updates (e.g. polar scan rebuild)."""
        pass

    def _on_episode_reset(self) -> None:
        """Called at the end of env_specific_reset(), after scenario sampling.
        Override in subclass for additional per-episode setup."""
        pass

    # ---------------------------
    # Collision-entity helpers
    # ---------------------------
    def _sync_collision_entity_views(self) -> None:
        if not getattr(self, "collision_entities", None):
            self.obstacle_pos = np.zeros((0, 2), dtype=np.float32)
            self.static_ship_radii = np.zeros((0,), dtype=np.float32)
            self.obstacle_count = 0
            return

        self.obstacle_count = len(self.collision_entities)
        self.obstacle_pos = np.array(
            [entity.get_position() for entity in self.collision_entities],
            dtype=np.float32,
        )
        self.static_ship_radii = np.array(
            [float(entity.radius) for entity in self.collision_entities],
            dtype=np.float32,
        )

    def _step_collision_entities(self) -> None:
        if not getattr(self, "collision_entities", None):
            return

        moved = False

        active_ship_positions = []
        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            active_ship_positions = [self.ship_pos]

        for entity in self.collision_entities:
            if not getattr(entity, "is_dynamic", False):
                continue

            old_pos = entity.get_position().copy()

            if getattr(entity, "entity_type", None) == "head-on":
                 if active_ship_positions and not getattr(entity, "dodged", False):
                     entity_pos = entity.get_position()
                     min_dist = min(
                         float(np.linalg.norm(entity_pos - os_pos))
                         for os_pos in active_ship_positions
                     )
                     if min_dist <= self.encounter_range / 2:
                         entity.dodged = True
                         dodge_angle = math.radians(20)
                         entity.set_heading(entity.get_heading() - dodge_angle)
                         entity.set_lateral_speed(0.0)
                         entity.route_mode = None

            if getattr(entity, "entity_type", "") in ("crossing-right", "crossing-left"):
                if not getattr(entity, "crossing_activated", False) and active_ship_positions:
                    entity_pos = entity.get_position()
                    crossing_point = getattr(entity, "crossing_point", None)

                    best_os_pos = None
                    best_dist = float("inf")
                    for os_pos in active_ship_positions:
                        d = float(np.linalg.norm(entity_pos - os_pos))
                        if d < best_dist:
                            best_dist = d
                            best_os_pos = os_pos

                    activation_range = getattr(entity, "crossing_activation_range", self.encounter_range + 50.0)

                    d_ts = float(getattr(entity, "crossing_d_ts", 0))
                    if d_ts > 0 and self.crossing_ts_v_min > 0:
                        sf = float(getattr(self, "crossing_speed_factor", 1.0))
                        v_os_est = float(max(abs(self.phys_sim.state.u), self.os_nominal_speed * 0.5))
                        timing_cap = sf * d_ts * v_os_est / self.crossing_ts_v_min
                        activation_range = min(activation_range, timing_cap)

                    if best_os_pos is not None and best_dist <= activation_range and crossing_point is not None:
                        to_cp = crossing_point - best_os_pos
                        path_tangent = getattr(entity, "crossing_path_tangent", None)
                        if path_tangent is not None:
                            forward = np.asarray(path_tangent, dtype=np.float32)
                        else:
                            os_heading = float(self.phys_sim.state.heading)
                            forward = np.array([math.cos(os_heading), math.sin(os_heading)], dtype=np.float32)
                        ahead_dot = float(np.dot(to_cp, forward))

                        if ahead_dot > 0.0:
                            d_os = float(np.linalg.norm(to_cp))
                            v_os_actual = float(max(abs(self.phys_sim.state.u), self.os_nominal_speed * 0.5))
                            speed_factor = float(getattr(self, "crossing_speed_factor", 1.0))
                            if d_os > EPSILON:
                                v_ts = speed_factor * d_ts * v_os_actual / d_os
                            else:
                                v_ts = float(self.crossing_ts_v_max)

                            v_ts = float(np.clip(v_ts, self.crossing_ts_v_min, self.crossing_ts_v_max))

                            entity.crossing_activated = True
                            entity.set_speed(v_ts)

            route_mode = getattr(entity, "route_mode", None)

            if route_mode == "follow_checkpoints":
                self._advance_entity_along_checkpoints(entity, self.time_step)
            else:
                entity.step(self.time_step)

            if not np.allclose(old_pos, entity.get_position()):
                moved = True

        if moved:
            self._sync_collision_entity_views()

    def _advance_entity_along_checkpoints(self, entity, dt: float) -> None:
        remaining_dist = float(entity.path_speed) * float(dt)
        path_direction = int(getattr(entity, "path_direction", 1))
        path_offset = float(getattr(entity, "path_offset", 0.0))

        while remaining_dist > EPSILON and not getattr(entity, "route_finished", False):
            next_idx = int(getattr(entity, "next_checkpoint_index", -1))

            if next_idx < 0 or next_idx >= len(self.checkpoints):
                entity.route_finished = True
                entity.set_speed(0.0)
                return

            if path_direction >= 0:
                if next_idx <= 0:
                    entity.route_finished = True
                    entity.set_speed(0.0)
                    return
                cp_a = np.asarray(self.checkpoints[next_idx - 1]["pos"], dtype=np.float32)
                cp_b = np.asarray(self.checkpoints[next_idx]["pos"], dtype=np.float32)
                travel_dir = cp_b - cp_a
            else:
                if next_idx >= len(self.checkpoints) - 1:
                    entity.route_finished = True
                    entity.set_speed(0.0)
                    return
                cp_a = np.asarray(self.checkpoints[next_idx + 1]["pos"], dtype=np.float32)
                cp_b = np.asarray(self.checkpoints[next_idx]["pos"], dtype=np.float32)
                travel_dir = cp_b - cp_a

            travel_norm = float(np.linalg.norm(travel_dir))
            if travel_norm <= EPSILON:
                entity.path_segment_index += path_direction
                entity.next_checkpoint_index += path_direction
                continue

            travel_dir = travel_dir / travel_norm
            starboard = np.array([travel_dir[1], -travel_dir[0]], dtype=np.float32)
            target_center = np.asarray(self.checkpoints[next_idx]["pos"], dtype=np.float32)
            target = target_center + path_offset * starboard

            pos = entity.get_position()
            delta = target - pos
            dist = float(np.linalg.norm(delta))

            if dist <= EPSILON:
                entity.set_position(float(target[0]), float(target[1]))
                entity.path_segment_index += path_direction
                entity.next_checkpoint_index += path_direction

                if entity.next_checkpoint_index < 0 or entity.next_checkpoint_index >= len(self.checkpoints):
                    entity.route_finished = True
                    entity.set_speed(0.0)
                    return
                continue

            direction = delta / dist
            entity.set_heading(float(math.atan2(direction[1], direction[0])))
            entity.set_speed(float(entity.path_speed))

            move_dist = min(remaining_dist, dist)
            entity.step(move_dist / max(entity.get_speed(), EPSILON))
            remaining_dist -= move_dist

            if move_dist >= dist - EPSILON:
                entity.set_position(float(target[0]), float(target[1]))
                entity.path_segment_index += path_direction
                entity.next_checkpoint_index += path_direction

                if entity.next_checkpoint_index < 0 or entity.next_checkpoint_index >= len(self.checkpoints):
                    entity.route_finished = True
                    entity.set_speed(0.0)
                    return

    # -----------------------
    # Reset
    # -----------------------
    def env_specific_reset(self) -> None:
        """Reset episode-specific geometry and build collision entities."""
        # Finalize previous episode's min distance
        if hasattr(self, "_current_episode_min_dist"):
            if not self._current_episode_collided and self._current_episode_min_dist < float("inf"):
                self._episode_min_distances.append(self._current_episode_min_dist)
        self._current_episode_min_dist = float("inf")
        self._current_episode_collided = False

        self.collision_entities = []
        self.obstacle_count = 0
        self.obstacle_pos = None
        self.static_ship_radii = []

        self._last_collision_info = {}

        # Reset scenario classifier and COLREG tracking
        self.scenario_classifier.reset()
        self.ts_passed = None
        self._colreg_violations = 0
        self._colreg_correct = 0

        # Remove passing snapshot artists from the plot and reset
        for artists in getattr(self, "_passing_snapshot_artists", []):
            for artist in artists:
                artist.remove()
        self._passing_snapshot_artists = []
        self._passing_snapshots = []

        # Build scenario for this episode (or initialize adaptive state)
        if self.scenario_obstacle_level == "episode":
            self.scenario_builder.on_reset()

        # Clear trail histories
        self.os_trail_history = []
        for trail_line in getattr(self, "ts_trails", []):
            trail_line.remove()
        self.ts_trail_history = []
        self.ts_trails = []

        # Subclass hook
        self._on_episode_reset()

    # -----------------------
    # Rendering
    # -----------------------
    def _initialize_rendering(self) -> None:
        super()._initialize_rendering()

        self.ship_plot_colav, = self.ax.plot([], [], 'bo', markersize=10, label='OS')
        self.heading_line_colav, = self.ax.plot([], [], color='black', linewidth=2)

        # OS trail line
        self.os_trail_history = []
        self.os_trail_line, = self.ax.plot(
            [], [], color='blue', alpha=0.5, linewidth=1.5, linestyle='-', label='OS trail',
        )

        # TS trail lines (created dynamically)
        self.ts_trails = []
        self.ts_trail_history = []

        # Passing snapshot artists
        self._passing_snapshot_artists = []

    def _get_collision_entity_render_key(self):
        entities = getattr(self, "collision_entities", []) or []
        if not entities:
            return ("none", 0)
        rounded = tuple(
            (float(round(entity.x, 2)), float(round(entity.y, 2)), float(round(entity.radius, 2)), entity.entity_type)
            for entity in entities
        )
        return (len(entities), rounded)

    def _clear_collision_entity_patches(self) -> None:
        if not hasattr(self, "_collision_entity_patches"):
            self._collision_entity_patches = []
            return
        for patch in self._collision_entity_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self._collision_entity_patches = []

    def _draw_collision_entities(self) -> None:
        if not hasattr(self, "ax"):
            return
        self._clear_collision_entity_patches()
        entities = getattr(self, "collision_entities", []) or []
        for i, entity in enumerate(entities):
            sx, sy = float(entity.x), float(entity.y)
            radius = float(entity.radius)
            safety = patches.Circle(
                (sx, sy), radius, fill=False, edgecolor="gray",
                linewidth=2, alpha=0.8,
                label="Collision entity" if i == 0 else ""
            )
            self.ax.add_patch(safety)
            self._collision_entity_patches.append(safety)

    def _draw_fixed_landmarks(self) -> None:
        self.start_pos = np.asarray(self.initial_ship_pos, dtype=np.float32).copy()
        super()._draw_fixed_landmarks()

    def render(self) -> None:
        """Render the environment with collision entities and trails."""
        if self.render_mode != 'human':
            return

        if self.initialize_plots and not hasattr(self, 'ship_plot_colav'):
            self._initialize_rendering()
            self._draw_fixed_landmarks()
            self._draw_collision_entities()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig.canvas.mpl_connect('draw_event', self._on_draw)
            plt.show(block=False)

        canvas = self.fig.canvas

        def _safe(callable_obj):
            try:
                callable_obj()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass

        new_key = self._get_collision_entity_render_key()
        if new_key != getattr(self, "_collision_entity_render_key", None):
            self._collision_entity_render_key = new_key
            self._draw_collision_entities()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        if hasattr(self, 'background'):
            _safe(lambda: canvas.restore_region(self.background))

        heading_line_length = 30.0

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            self.os_trail_history.append(self.ship_pos.copy())

        entities = getattr(self, "collision_entities", []) or []
        while len(self.ts_trail_history) < len(entities):
            self.ts_trail_history.append([])
            trail_line, = self.ax.plot(
                [], [], color='red', alpha=0.5, linewidth=1.5, linestyle='--',
                label='TS trail' if len(self.ts_trails) == 0 else '',
            )
            self.ts_trails.append(trail_line)

        for idx, entity in enumerate(entities):
            self.ts_trail_history[idx].append(entity.get_position().copy())

        if self.os_trail_history:
            pts = np.array(self.os_trail_history)
            self.os_trail_line.set_data(pts[:, 0], pts[:, 1])

        for idx in range(len(entities)):
            if self.ts_trail_history[idx]:
                pts = np.array(self.ts_trail_history[idx])
                self.ts_trails[idx].set_data(pts[:, 0], pts[:, 1])

        self.ship_plot_colav.set_data(self.ship_pos[0:1], self.ship_pos[1:2])

        heading = self.phys_sim.state.heading
        hx = self.ship_pos[0] + np.cos(heading) * heading_line_length
        hy = self.ship_pos[1] + np.sin(heading) * heading_line_length
        self.heading_line_colav.set_data(
            [self.ship_pos[0], hx], [self.ship_pos[1], hy]
        )

        # Subclass rendering hook
        self._render_subclass(heading)

        # Passing snapshot artists
        while len(self._passing_snapshot_artists) < len(self._passing_snapshots):
            idx = len(self._passing_snapshot_artists)
            snap = self._passing_snapshots[idx]
            os_p = snap["os_pos"]
            ts_p = snap["ts_pos"]
            dist = snap["dist"]

            os_marker, = self.ax.plot(
                [os_p[0]], [os_p[1]], 'o', markersize=12,
                markerfacecolor='none', markeredgecolor='blue',
                markeredgewidth=2, zorder=5,
            )
            ts_marker, = self.ax.plot(
                [ts_p[0]], [ts_p[1]], 'o', markersize=12,
                markerfacecolor='none', markeredgecolor='red',
                markeredgewidth=2, zorder=5,
            )
            dist_line, = self.ax.plot(
                [os_p[0], ts_p[0]], [os_p[1], ts_p[1]],
                color='gray', linestyle='--', linewidth=1.2, zorder=4,
            )
            mid_x = (os_p[0] + ts_p[0]) / 2.0
            mid_y = (os_p[1] + ts_p[1]) / 2.0
            dist_text = self.ax.text(
                mid_x, mid_y, f"{dist:.0f}m", fontsize=7, color='gray',
                fontweight='bold', ha='center', va='bottom', zorder=6,
            )
            os_label = self.ax.text(
                os_p[0] + 2, os_p[1] + 2, "OS", fontsize=7, color='blue',
                fontweight='bold', ha='center', va='top', zorder=6,
            )
            ts_label = self.ax.text(
                ts_p[0] + 2, ts_p[1] + 2, "TS", fontsize=7, color='red',
                fontweight='bold', ha='center', va='top', zorder=6,
            )
            self._passing_snapshot_artists.append(
                (os_marker, ts_marker, dist_line, dist_text, os_label, ts_label)
            )

        # Blit
        try:
            if hasattr(self.fig.canvas, 'blit'):
                self.ax.draw_artist(self.os_trail_line)
                for tl in self.ts_trails:
                    self.ax.draw_artist(tl)
                for artists in self._passing_snapshot_artists:
                    for artist in artists:
                        self.ax.draw_artist(artist)
                self.ax.draw_artist(self.ship_plot_colav)
                self.ax.draw_artist(self.heading_line_colav)
                self._blit_subclass()
                self.fig.canvas.blit(self.ax.bbox)
            else:
                self.fig.canvas.draw()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            _safe(self.fig.canvas.draw)
        _safe(canvas.flush_events)

    def _render_subclass(self, heading: float) -> None:
        """Override in subclass to add rendering (e.g. polar beams)."""
        pass

    def _blit_subclass(self) -> None:
        """Override in subclass to blit additional artists (e.g. polar beams)."""
        pass
